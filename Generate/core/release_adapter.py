"""
Адаптер выдачи клипов через GitHub Releases (Generate/core/release_adapter.py).

SPEC 9 и 0.2: ядро работает с хранилищем через абстрактный интерфейс
ReleaseStore; детали `gh` изолированы в GhReleaseStore / GhCommandRunner.
SPEC 0.1: токен берётся только из окружения (GH_TOKEN / GITHUB_TOKEN),
в аргументы командной строки, логи и тексты ошибок не попадает.
SPEC 0.5: все сообщения на русском языке.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional, Sequence

try:
    from .gemini_prompts import GeminiInputError, GeminiRuntimeError
    from .naming import clip_filename, parse_clip_num
    from .srt_parser import srt_hash
except ImportError:
    from gemini_prompts import GeminiInputError, GeminiRuntimeError
    from naming import clip_filename, parse_clip_num
    from srt_parser import srt_hash

PROMPTS_FILENAME = "prompts.json"
LINKS_FILENAME = "generated_links.txt"
SERVICE_FILES = frozenset({PROMPTS_FILENAME, LINKS_FILENAME})

DEFAULT_ASSET_LIMIT = 900
HASH_PREFIX_LEN = 12
CLIP_TAG_PREFIX = "clips-"

# Шаблоны токенов GitHub: вырезаем даже если токен не был передан явно.
_TOKEN_RE = re.compile(r"(gh[pousr]_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,})")


# ---------------------------------------------------------------------------
# Теги, заголовки, описания (SPEC 9.1)
# ---------------------------------------------------------------------------

def make_tag(segments, release_tag: Optional[str] = None) -> str:
    """clips-<первые 12 символов srt_hash> (клип-релиз, SPEC 9.1), если не задан свой тег."""
    if release_tag:
        return release_tag
    return f"{CLIP_TAG_PREFIX}{srt_hash(segments)[:HASH_PREFIX_LEN]}"


def part_tag(tag: str, part: int) -> str:
    """Тег части клип-релиза: <tag>-part2, <tag>-part3, ... (SPEC 9.5)."""
    return f"{tag}-part{part}"


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _check_run_id(run_id: str) -> str:
    if not run_id or not _RUN_ID_RE.match(run_id):
        raise GeminiInputError(f"Недопустимый идентификатор запуска: {run_id!r}")
    return run_id


def service_tag(run_id: str) -> str:
    """Тег служебного релиза run-<run_id>-service (SPEC 9.1)."""
    return f"run-{_check_run_id(run_id)}-service"


def final_tag(run_id: str) -> str:
    """Тег итогового релиза run-<run_id>-final (SPEC 9.1)."""
    return f"run-{_check_run_id(run_id)}-final"


def format_title(full_hash: str, when: datetime, part: int = 1) -> str:
    """clips-<12> · часть N · YYYY-MM-DD HH:MM UTC (SPEC 9.1)."""
    when = when.astimezone(timezone.utc)
    return (f"{CLIP_TAG_PREFIX}{full_hash[:HASH_PREFIX_LEN]} · часть {part} · "
            f"{when:%Y-%m-%d %H:%M} UTC")


def format_body(full_hash: str, model: str) -> str:
    """Описание клип-релиза: полный хэш и профиль модели (SPEC 9.1)."""
    return f"hash: {full_hash}\nmodel: {model}"


def format_run_title(kind: str, run_id: str, when: datetime) -> str:
    """Заголовок служебного или итогового релиза (SPEC 9.1)."""
    when = when.astimezone(timezone.utc)
    return f"run-{run_id} · {kind} · {when:%Y-%m-%d %H:%M} UTC"


def format_run_body(kind: str, run_id: str) -> str:
    """Описание служебного или итогового релиза: без хэша, только идентификаторы."""
    return f"run: {run_id}\nkind: {kind}"


# ---------------------------------------------------------------------------
# Абстрактное хранилище (SPEC 0.2)
# ---------------------------------------------------------------------------

@dataclass
class ReleaseInfo:
    tag: str
    title: str = ""
    body: str = ""
    is_draft: bool = False
    assets: list = field(default_factory=list)  # имена файлов


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
# Всё остальное не передаётся: в частности GEN_VAST_API_KEY, GEN_GEMINI_API_KEY, HF_TOKEN, WORKER_TOKEN, GITHUB_TOKEN.
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
# Менеджер выдачи (SPEC 9.2, 9.5)
# ---------------------------------------------------------------------------

class ReleaseUploader:
    """Выдача клипов и служебных файлов (SPEC 9.1-9.7).

    Проверки 9.2 относятся только к клип-релизам (тег clips-<хэш> и его части).
    Служебный (run-<run_id>-service) и итоговый (run-<run_id>-final) релизы создаются
    на каждый запуск и хэш не проверяют. Им нужен run_id: без него служебные файлы
    (prompts.json, generated_links.txt, missing.txt) не загружаются, и это ошибка,
    а не тихий пропуск. Клипы без run_id загружаются как раньше.
    """

    _KIND_LABEL = {"service": "служебный", "final": "итоговый"}

    def __init__(self, store: ReleaseStore, segments, model: str,
                 release_tag: Optional[str] = None,
                 release_asset_limit: int = DEFAULT_ASSET_LIMIT,
                 filename_template: str = "{num}.mp4",
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 log: Callable[[str], None] = print,
                 run_id: Optional[str] = None):
        if release_asset_limit <= 0:
            raise GeminiInputError(
                f"release_asset_limit должен быть положительным: {release_asset_limit}"
            )
        self.store = store
        self.model = model
        self.full_hash = srt_hash(segments)
        self.tag = make_tag(segments, release_tag)
        self.limit = release_asset_limit
        self.template = filename_template
        self._now = now
        self._log = log
        self.run_id = _check_run_id(run_id) if run_id is not None else None
        self.service_tag = service_tag(self.run_id) if self.run_id is not None else None
        self.final_tag = final_tag(self.run_id) if self.run_id is not None else None
        self._counts: dict = {}      # тег клип-релиза -> число файлов в нём
        self._parts: list = [self.tag]
        self.asset_tags: dict = {}   # имя файла клипа -> тег клип-релиза
        self.done: set = set()
        self._prepared = False
        self._service_opened: Optional[str] = None  # служебный релиз, открытый в этом запуске

    # --- предполётные проверки (клип-релизы) ----------------------------------
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
        """Проверки 1-3 для клип-релизов. Возвращает номера уже загруженных клипов.

        При заданном run_id заодно открывает служебный релиз (начало запуска, 9.1)."""
        done: set = set()
        main = self._check_release(self.tag)
        if main is None:
            self.store.create_release(self.tag, format_title(self.full_hash, self._now()),
                                      format_body(self.full_hash, self.model))
            self._counts[self.tag] = 0
            self._log(f"Создан релиз {self.tag}.")
        else:
            self._absorb(main, done)
            n = 2
            while True:
                pt = part_tag(self.tag, n)
                rel = self._check_release(pt)
                if rel is None:
                    break
                self._parts.append(pt)
                self._absorb(rel, done)
                n += 1
            self._log(f"Продолжение: в релизах уже {len(done)} клипов.")
        self.done = done
        self._prepared = True
        if self.run_id is not None:
            self._open_run_release("service")
        return set(done)

    def _absorb(self, rel: ReleaseInfo, done: set) -> None:
        self._counts[rel.tag] = len(rel.assets)
        for name in rel.assets:
            num = parse_clip_num(name, self.template)
            if num is not None:
                done.add(num)
                self.asset_tags[name] = rel.tag

    # --- служебный и итоговый релизы (SPEC 9.1, 9.2) ---------------------------
    def _open_run_release(self, kind: str) -> str:
        """Создаёт служебный или итоговый релиз, если его нет. Хэш не проверяется.

        Больше одного релиза с тегом: та же остановка, что и для клип-релизов."""
        if self.run_id is None:
            raise GeminiInputError("Не задан идентификатор запуска (run_id): релиз запуска не открыт.")
        tag = self.service_tag if kind == "service" else self.final_tag
        # Служебный релиз уже открыт в этом запуске (prepare() и первая загрузка): повторно не создаём.
        # Список релизов GitHub может отставать сразу после создания, поэтому опираться только на него нельзя.
        if kind == "service" and self._service_opened == tag:
            return tag
        found = self.store.find_releases(tag)
        if len(found) > 1:
            raise GeminiInputError(
                f"Найдено несколько релизов с тегом {tag}, удалите лишние вручную."
            )
        if not found:
            try:
                self.store.create_release(tag,
                                          format_run_title(self._KIND_LABEL[kind], self.run_id, self._now()),
                                          format_run_body(kind, self.run_id))
                self._log(f"Создан релиз {tag}.")
            except GeminiRuntimeError as exc:
                # Повторная попытка: релиз уже появился (422 already_exists). Для служебного это не ошибка:
                # файлы всё равно грузятся через upload --clobber, поэтому продолжаем с существующим релизом.
                if kind != "service" or not _is_already_exists(exc):
                    raise
                self._log(f"Служебный релиз {tag} уже существует: продолжаем с ним.")
        if kind == "service":
            self._service_opened = tag
        return tag

    def open_service_release(self) -> str:
        """Открывает служебный релиз run-<run_id>-service (в начале запуска)."""
        self._ensure_prepared()
        return self._open_run_release("service")

    def open_final_release(self) -> str:
        """Открывает итоговый релиз run-<run_id>-final (один раз, в конце запуска)."""
        self._ensure_prepared()
        return self._open_run_release("final")

    # --- загрузка -------------------------------------------------------------
    def _ensure_prepared(self) -> None:
        if not self._prepared:
            raise GeminiInputError("Сначала нужно вызвать prepare() (предполётные проверки).")

    def _active_tag(self) -> str:
        last = self._parts[-1]
        if self._counts.get(last, 0) < self.limit:
            return last
        n = len(self._parts) + 1
        pt = part_tag(self.tag, n)
        self.store.create_release(pt, format_title(self.full_hash, self._now(), n),
                                  format_body(self.full_hash, self.model))
        self._parts.append(pt)
        self._counts[pt] = 0
        self._log(f"Лимит {self.limit} файлов достигнут, создан релиз {pt}.")
        return pt

    def upload_clip(self, path: str) -> bool:
        """Загружает клип в активную часть клип-релиза; False, если номер уже загружен."""
        self._ensure_prepared()
        name = os.path.basename(path)
        num = parse_clip_num(name, self.template)
        if num is None:
            raise GeminiInputError(f"Имя файла не соответствует шаблону клипа: {name}")
        if num in self.done:
            return False
        tag = self._active_tag()
        self.store.upload_asset(tag, path)
        self._counts[tag] += 1
        self.asset_tags[name] = tag
        self.done.add(num)
        return True

    def upload_clip_num(self, num: int, directory: str, width: int = 0) -> bool:
        """Загружает клип по номеру из каталога (имя по clip_filename)."""
        name = clip_filename(num, width, self.template)
        return self.upload_clip(os.path.join(directory, name))

    def upload_run_file(self, path: str) -> None:
        """Файл запуска (prompts.json, missing.txt, отчёт, логи) в служебный релиз (SPEC 9.1, 9.7).

        Клипы и generated_links.txt сюда не принимаются."""
        self._ensure_prepared()
        name = os.path.basename(path)
        if self.run_id is None:
            raise GeminiInputError(
                "Служебные файлы некуда загрузить: не задан идентификатор запуска (run_id)."
            )
        if parse_clip_num(name, self.template) is not None:
            raise GeminiInputError(f"Клип не относится к служебному релизу: {name}")
        if name == LINKS_FILENAME:
            raise GeminiInputError(f"{LINKS_FILENAME} загружается в итоговый релиз, не в служебный.")
        tag = self._open_run_release("service")
        self.store.upload_asset(tag, path)

    def upload_service_file(self, path: str) -> None:
        """prompts.json -> служебный релиз; generated_links.txt -> итоговый релиз (SPEC 9.4, 9.7).

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
            if self.run_id is None:
                raise GeminiInputError(
                    "generated_links.txt некуда загрузить: не задан идентификатор запуска (run_id)."
                )
            tag = self._open_run_release("final")
            self.store.upload_asset(tag, path)
            return
        self.upload_run_file(path)

    def links_text(self, repo: str) -> str:
        """Содержимое generated_links.txt по всем загруженным клипам всех клип-релизов.

        Пустая строка, если клипов нет (SPEC 9.4)."""
        names = sorted(self.asset_tags,
                       key=lambda n: (parse_clip_num(n, self.template) or 0, n))
        return create_generated_links(repo, self.tag, names, self.asset_tags)
