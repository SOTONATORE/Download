"""
Адаптер выдачи через GitHub Releases (Generate/core/release_adapter.py).

SPEC 9 и 0.2: ядро работает с хранилищем через абстрактный интерфейс
ReleaseStore; детали `gh` изолированы в GhReleaseStore / GhCommandRunner.
SPEC 0.1: токен берётся только из окружения (GH_TOKEN / GITHUB_TOKEN),
в аргументы командной строки, логи и тексты ошибок не попадает.
SPEC 0.5: все сообщения на русском языке.

Стадия 6: выдача генерации идёт в два релиза на хэш final.srt:
ФИНАЛ · ГЕНЕРАЦИЯ (final-gen-<hash16>: служебные файлы, превью, ссылки) и части
ГЕНЕРАЦИЯ · ЧАСТЬ N (gen-part<N>-<hash16>: только клипы и фото, до 900 файлов).
Имена, теги и описания - в release_names.py (общие с Download/publish_stock.py).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional, Sequence

try:
    from .gemini_prompts import GeminiInputError, GeminiRuntimeError
    from .naming import clip_filename, parse_clip_num
    from .release_names import (final_tag, final_title, part_tag, part_title, release_body, hash16)
    from .srt_parser import srt_hash
except ImportError:
    from gemini_prompts import GeminiInputError, GeminiRuntimeError
    from naming import clip_filename, parse_clip_num
    from release_names import (final_tag, final_title, part_tag, part_title, release_body, hash16)
    from srt_parser import srt_hash

PROMPTS_FILENAME = "prompts.json"
LINKS_FILENAME = "generated_links.txt"
SERVICE_FILES = frozenset({PROMPTS_FILENAME, LINKS_FILENAME})
# Диагностика карт (SPEC 0.1, правило портов): ssh_diag.log, vast_logs.txt, worker_log.txt, instance.json.
# В GitHub Release не выкладываются; остаются только в артефакте запуска.
CARD_DIAGNOSTIC_FILES = frozenset({"ssh_diag.log", "vast_logs.txt", "worker_log.txt", "instance.json"})

# Не больше 900 файлов в части: лимит GitHub на ассеты релиза 1000, запас на служебные файлы.
DEFAULT_ASSET_LIMIT = 900
STAGE_GEN = "gen"

# Шаблоны токенов GitHub: вырезаем даже если токен не был передан явно.
_TOKEN_RE = re.compile(r"(gh[pousr]_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,})")


# фото (STEP 7, блок B2): файл фото «{num}.png»; распознаётся наряду с шаблоном клипа
PHOTO_FILENAME_TEMPLATE = "{num}.png"


def _clip_num_any(name: str, template: str):
    """Номер клипа по имени файла: по шаблону видео или по шаблону фото (naming.py не меняется)."""
    num = parse_clip_num(name, template)
    if num is None:
        num = parse_clip_num(name, PHOTO_FILENAME_TEMPLATE)
    return num


@dataclass
class ReleaseInfo:
    tag: str
    title: str = ""
    body: str = ""
    is_draft: bool = False
    assets: list = field(default_factory=list)  # имена файлов



# ---------------------------------------------------------------------------
# Абстрактное хранилище (SPEC 0.2)
# ---------------------------------------------------------------------------


class ReleaseStore(ABC):
    """Интерфейс хранилища релизов; GitHub-адаптер — одна из реализаций."""

    @abstractmethod
    def find_releases(self, tag: str) -> list:
        """Все релизы (включая черновики) с точно таким тегом."""

    @abstractmethod
    def create_release(self, tag: str, title: str, body: str) -> None:
        """Создаёт релиз."""

    @abstractmethod
    def upload_asset(self, tag: str, path: str) -> None:
        """Загружает файл в релиз."""


# ---------------------------------------------------------------------------
# Выполнение `gh`
# ---------------------------------------------------------------------------

def scrub_secrets(text: str, tokens: Sequence[str] = ()) -> str:
    """Убирает токены (известные и похожие по шаблону) из текста."""
    for t in tokens:
        if t:
            text = text.replace(t, "***")
    return _TOKEN_RE.sub("***", text)


# Имена переменных окружения, которые получает процесс gh (SPEC 0.1; этап 9а, G09, F25).
# PATH — поиск исполняемого файла gh; HOME — каталог настроек gh (~/.config/gh); GH_TOKEN — токен для gh.
# Всё остальное не передаётся: в частности GEN_VAST_API_KEY, GEMINI1…GEMINI10, HF_TOKEN, WORKER_TOKEN, GITHUB_TOKEN.
# Для запуска gh на Windows (локальный запуск) могут понадобиться USERPROFILE, APPDATA, SYSTEMROOT: НЕ ПРОВЕРЕНО, в константу не добавлены.
GH_ENV_NAMES = ("PATH", "HOME", "GH_TOKEN")


class GhCommandRunner:
    """Запускает `gh <args>` через subprocess; токен только в env процесса."""

    def __init__(self, environ: Optional[Mapping[str, str]] = None,
                 run: Callable = subprocess.run, gh_bin: str = "gh",
                 timeout: int = 600):
        self._environ = environ if environ is not None else os.environ
        self._run = run
        self._gh = gh_bin
        self._timeout = timeout

    def _token(self) -> str:
        token = self._environ.get("GH_TOKEN") or self._environ.get("GITHUB_TOKEN")
        if not token:
            raise GeminiInputError(
                "Не задан токен GitHub: ожидается переменная окружения GH_TOKEN или GITHUB_TOKEN."
            )
        return token

    def __call__(self, args: Sequence[str]) -> str:
        token = self._token()
        # окружение процесса gh: только GH_ENV_NAMES; GH_TOKEN берётся из _token(); остальной os.environ не передаётся (этап 9а)
        env = {name: self._environ[name] for name in GH_ENV_NAMES if name in self._environ}
        env["GH_TOKEN"] = token
        try:
            proc = self._run([self._gh, *args], env=env, capture_output=True,
                             text=True, timeout=self._timeout)
        except FileNotFoundError:
            raise GeminiRuntimeError("Не найден исполняемый файл gh (GitHub CLI).") from None
        except subprocess.TimeoutExpired:
            raise GeminiRuntimeError(
                f"Команда gh не уложилась в {self._timeout} с и была прервана."
            ) from None
        except Exception as exc:  # безопасно: только тип исключения
            raise GeminiRuntimeError(
                f"Не удалось запустить gh ({type(exc).__name__})."
            ) from None
        if proc.returncode != 0:
            detail = scrub_secrets((proc.stderr or "").strip(), [token])
            raise GeminiRuntimeError(
                f"Команда gh завершилась с кодом {proc.returncode}: {detail}"
            )
        return scrub_secrets(proc.stdout or "", [token])


_JQ = ('.[] | {tag_name, name, body, draft, assets: [.assets[].name]} | tojson')


class GhReleaseStore(ReleaseStore):
    """Реализация ReleaseStore поверх GitHub CLI."""

    def __init__(self, repo: str, runner: Optional[Callable] = None):
        self.repo = repo
        self._runner = runner or GhCommandRunner()

    def find_releases(self, tag: str) -> list:
        out = self._runner(["api", "--paginate",
                            f"repos/{self.repo}/releases?per_page=100", "--jq", _JQ])
        found = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                raise GeminiRuntimeError("Не удалось разобрать список релизов от GitHub.") from None
            if d.get("tag_name") == tag:
                found.append(ReleaseInfo(tag=tag, title=d.get("name") or "",
                                         body=d.get("body") or "",
                                         is_draft=bool(d.get("draft")),
                                         assets=list(d.get("assets") or [])))
        return found

    def create_release(self, tag: str, title: str, body: str) -> None:
        self._runner(["release", "create", tag, "--repo", self.repo,
                      "--title", title, "--notes", body])

    def upload_asset(self, tag: str, path: str) -> None:
        self._runner(["release", "upload", tag, path, "--repo", self.repo, "--clobber"])


# ---------------------------------------------------------------------------
# Ссылки (SPEC 9.4)
# ---------------------------------------------------------------------------

def _is_already_exists(exc: Exception) -> bool:
    """Ошибка GitHub «релиз или ассет уже существует» (HTTP 422, already_exists)."""
    text = str(exc).lower()
    return "already_exists" in text or ("422" in text and "exist" in text)


def create_generated_links(repo: str, tag: str, asset_filenames: Sequence[str],
                           tag_of: Optional[Mapping[str, str]] = None) -> str:
    """
    Прямые ссылки, по одной на строку. Служебные файлы исключаются.
    tag_of: имя файла -> тег релиза (для клипов из частей -partN);
    без него все файлы считаются лежащими в релизе `tag`.
    """
    lines = []
    for name in asset_filenames:
        if name in SERVICE_FILES:
            continue
        t = (tag_of or {}).get(name, tag)
        lines.append(f"https://github.com/{repo}/releases/download/{t}/{name}")
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Менеджер выдачи генерации (стадия 6)
# ---------------------------------------------------------------------------

class ReleaseUploader:
    """Выдача генерации: ФИНАЛ · ГЕНЕРАЦИЯ и части ГЕНЕРАЦИЯ · ЧАСТЬ N.

    Финал (final-gen-<hash16>) открывается в prepare(), то есть в начале запуска, и
    наполняется по ходу: prompts.json и missing.txt, превью, отчёты, логи и
    generated_links.txt (ссылка на каждый клип, после каждой загрузки клипа).
    Клипы и фото идут только в части gen-part<N>-<hash16> (до 900 файлов).
    Проверка тела (полный хэш в описании) действует и для финала, и для частей.
    """

    def __init__(self, store: ReleaseStore, segments, model: str,
                 release_tag: Optional[str] = None,
                 release_asset_limit: int = DEFAULT_ASSET_LIMIT,
                 filename_template: str = "{num}.mp4",
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 log: Callable[[str], None] = print,
                 repo: str = ""):
        if release_asset_limit <= 0:
            raise GeminiInputError(
                f"release_asset_limit должен быть положительным: {release_asset_limit}"
            )
        self.store = store
        self.model = model
        self.full_hash = srt_hash(segments)
        self.h16 = hash16(self.full_hash)
        self.final_tag = final_tag(STAGE_GEN, self.full_hash)
        self.repo = repo
        self.limit = release_asset_limit
        self.template = filename_template
        self._now = now
        self._log = log
        if release_tag:
            # своё имя релиза больше не поддерживается: раскладка фиксирована (стадия 6)
            self._log(f"Свой тег релиза {release_tag!r} не используется: раскладка релизов фиксирована.")
        self._counts: dict = {}      # тег части -> число файлов в ней
        self._parts: list = []       # теги частей ГЕНЕРАЦИЯ по порядку
        self.asset_tags: dict = {}   # имя файла клипа -> тег части
        self.done: set = set()
        self._prepared = False

    # --- предполётные проверки ------------------------------------------------
    def _check_release(self, tag: str) -> Optional[ReleaseInfo]:
        found = self.store.find_releases(tag)
        if len(found) > 1:
            raise GeminiInputError(
                f"Найдено несколько релизов с тегом {tag}, удалите лишние вручную."
            )
        if not found:
            return None
        rel = found[0]
        if self.full_hash not in (rel.body or ""):
            raise GeminiInputError("Тег совпал, содержимое нет: возможен другой final.srt.")
        return rel

    def prepare(self) -> set:
        """Открывает ФИНАЛ · ГЕНЕРАЦИЯ (если его нет) и читает уже загруженные клипы частей.
        Возвращает номера уже загруженных клипов (их пропускаем)."""
        rel = self._check_release(self.final_tag)
        if rel is None:
            self._create_final()
        else:
            self._log(f"Продолжение: релиз {self.final_tag} уже есть, новый финал не создаётся.")
        done: set = set()
        n = 1
        while True:
            pt = part_tag(STAGE_GEN, n, self.full_hash)
            part = self._check_release(pt)
            if part is None:
                break
            self._parts.append(pt)
            self._absorb(part, done)
            n += 1
        self.done = done
        self._prepared = True
        self._log(f"В частях ГЕНЕРАЦИЯ уже {len(done)} клипов.")
        return set(done)

    def _create_final(self) -> None:
        self.store.create_release(self.final_tag,
                                  final_title(STAGE_GEN, self.full_hash, self._now()),
                                  release_body(STAGE_GEN, "final", self.full_hash))
        self._log(f"Создан релиз {self.final_tag} (ФИНАЛ · ГЕНЕРАЦИЯ).")

    def _absorb(self, rel: ReleaseInfo, done: set) -> None:
        self._counts[rel.tag] = len(rel.assets)
        for name in rel.assets:
            num = _clip_num_any(name, self.template)
            if num is not None:
                done.add(num)
                self.asset_tags[name] = rel.tag

    # --- загрузка -------------------------------------------------------------
    def _ensure_prepared(self) -> None:
        if not self._prepared:
            raise GeminiInputError("Сначала нужно вызвать prepare() (предполётные проверки).")

    def _active_tag(self) -> str:
        """Часть, куда идёт очередной клип; при достижении лимита открывается следующая."""
        if self._parts and self._counts.get(self._parts[-1], 0) < self.limit:
            return self._parts[-1]
        n = len(self._parts) + 1
        pt = part_tag(STAGE_GEN, n, self.full_hash)
        self.store.create_release(pt, part_title(STAGE_GEN, n, self.full_hash, self._now()),
                                  release_body(STAGE_GEN, "part", self.full_hash))
        self._parts.append(pt)
        self._counts[pt] = 0
        self._log(f"Создан релиз {pt} (ГЕНЕРАЦИЯ · ЧАСТЬ {n}).")
        return pt

    def upload_clip(self, path: str) -> bool:
        """Загружает клип или фото в активную часть ГЕНЕРАЦИЯ; False, если номер уже загружен.
        После загрузки обновляет generated_links.txt в финале."""
        self._ensure_prepared()
        name = os.path.basename(path)
        num = _clip_num_any(name, self.template)
        if num is None:
            raise GeminiInputError(f"Имя файла не соответствует шаблону клипа: {name}")
        if num in self.done:
            return False
        tag = self._active_tag()
        self.store.upload_asset(tag, path)
        self._counts[tag] += 1
        self.asset_tags[name] = tag
        self.done.add(num)
        self._refresh_links()
        return True

    def upload_preview(self, path: str) -> None:
        """Превью preview_<x>_<y>.png -> ФИНАЛ · ГЕНЕРАЦИЯ (исключение: превью - медиа, но живут в финале).

        В счётчики частей и в asset_tags (generated_links.txt) не попадает."""
        self._ensure_prepared()
        name = os.path.basename(path)
        if not re.fullmatch(r"preview_\d+_\d+\.png", name):
            raise GeminiRuntimeError(f"Не имя превью: {name}")
        self._upload_service_asset(self.final_tag, path)

    def upload_clip_num(self, num: int, directory: str, width: int = 0) -> bool:
        """Загружает клип по номеру из каталога (имя по clip_filename)."""
        name = clip_filename(num, width, self.template)
        return self.upload_clip(os.path.join(directory, name))

    def upload_run_file(self, path: str) -> None:
        """Текстовый файл запуска (missing.txt, отчёт, логи) -> ФИНАЛ · ГЕНЕРАЦИЯ (SPEC 9.1, 9.7).

        Клипы, превью через этот путь не принимаются; generated_links.txt ведётся отдельно."""
        self._ensure_prepared()
        name = os.path.basename(path)
        if _clip_num_any(name, self.template) is not None:
            raise GeminiInputError(f"Клип не относится к финалу: {name}")
        if name == LINKS_FILENAME:
            raise GeminiInputError(f"{LINKS_FILENAME} ведётся отдельно (upload_service_file).")
        if name in CARD_DIAGNOSTIC_FILES:
            raise GeminiInputError(f"Диагностика карты не выкладывается в релиз: {name}")
        self._upload_service_asset(self.final_tag, path)

    def _upload_service_asset(self, tag: str, path: str) -> None:
        """Загрузка файла в релиз. Ответ 422 «already exists» сразу после создания считается нормой:
        файл грузится повторно один раз. Если и повтор падает, ошибка идёт дальше."""
        try:
            self.store.upload_asset(tag, path)
        except GeminiRuntimeError as exc:
            if not _is_already_exists(exc):
                raise
            self._log(f"Релиз {tag}: загрузка {os.path.basename(path)} ответила «already exists», повтор.")
            self.store.upload_asset(tag, path)

    def upload_service_file(self, path: str) -> None:
        """prompts.json и generated_links.txt -> ФИНАЛ · ГЕНЕРАЦИЯ (SPEC 9.4, 9.7).

        generated_links.txt нулевой длины (клипов нет) не выкладывается: GitHub отвергает
        пустой ассет (HTTP 400, Bad Content-Length). prompts.json проверка не касается."""
        self._ensure_prepared()
        name = os.path.basename(path)
        if name not in SERVICE_FILES:
            raise GeminiInputError(f"Не служебный файл: {name}")
        if name == LINKS_FILENAME and os.path.isfile(path) and os.path.getsize(path) == 0:
            self._log("Нет готовых клипов: generated_links.txt не выкладывается.")
            return
        if name == LINKS_FILENAME:
            self._upload_service_asset(self.final_tag, path)
            return
        self.upload_run_file(path)

    def links_text(self, repo: str) -> str:
        """Содержимое generated_links.txt по всем загруженным клипам всех частей.

        Пустая строка, если клипов нет (SPEC 9.4)."""
        names = sorted(self.asset_tags,
                       key=lambda n: (_clip_num_any(n, self.template) or 0, n))
        return create_generated_links(repo, "", names, self.asset_tags)

    def _refresh_links(self) -> None:
        """Перезаписывает generated_links.txt в финале после каждой загрузки клипа.

        Без репозитория (repo пуст) ссылок не строим: в лог пишется одна строка."""
        if not self.repo:
            self._log("Репозиторий не задан: generated_links.txt не обновлён.")
            return
        text = self.links_text(self.repo)
        if not text:
            return
        tmpdir = tempfile.mkdtemp(prefix="links-")
        try:
            path = os.path.join(tmpdir, LINKS_FILENAME)
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            self._upload_service_asset(self.final_tag, path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
