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

# Шаблоны токенов GitHub: вырезаем даже если токен не был передан явно.
_TOKEN_RE = re.compile(r"(gh[pousr]_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,})")


# ---------------------------------------------------------------------------
# Теги, заголовки, описания (SPEC 9.1)
# ---------------------------------------------------------------------------

def make_tag(segments, release_tag: Optional[str] = None) -> str:
    """gen-<первые 12 символов srt_hash>, если не задан свой тег."""
    if release_tag:
        return release_tag
    return f"gen-{srt_hash(segments)[:HASH_PREFIX_LEN]}"


def part_tag(tag: str, part: int) -> str:
    """Тег части: <tag>-part2, <tag>-part3, ..."""
    return f"{tag}-part{part}"


def format_title(full_hash: str, when: datetime, part: int = 1) -> str:
    """gen-<12> · YYYY-MM-DD HH:MM UTC [· часть N]."""
    when = when.astimezone(timezone.utc)
    title = f"gen-{full_hash[:HASH_PREFIX_LEN]} · {when:%Y-%m-%d %H:%M} UTC"
    if part > 1:
        title += f" · часть {part}"
    return title


def format_body(full_hash: str, model: str) -> str:
    """Описание релиза: полный хэш и профиль модели."""
    return f"hash: {full_hash}\nmodel: {model}"


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
        env = dict(self._environ)
        env["GH_TOKEN"] = token
        env.pop("GITHUB_TOKEN", None)
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
            detail = scrub_secrets((proc.stderr or "").strip(), [token])[:500]
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
    def __init__(self, store: ReleaseStore, segments, model: str,
                 release_tag: Optional[str] = None,
                 release_asset_limit: int = DEFAULT_ASSET_LIMIT,
                 filename_template: str = "{num}.mp4",
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 log: Callable[[str], None] = print):
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
        self._counts: dict = {}      # тег -> число файлов в релизе
        self._parts: list = [self.tag]
        self.asset_tags: dict = {}   # имя файла клипа -> тег релиза
        self.done: set = set()
        self._prepared = False

    # --- предполётные проверки -------------------------------------------
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
        """Проверки 1-3. Возвращает номера уже загруженных клипов."""
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
        return set(done)

    def _absorb(self, rel: ReleaseInfo, done: set) -> None:
        self._counts[rel.tag] = len(rel.assets)
        for name in rel.assets:
            num = parse_clip_num(name, self.template)
            if num is not None:
                done.add(num)
                self.asset_tags[name] = rel.tag

    # --- загрузка ----------------------------------------------------------
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
        """Загружает клип; False, если клип с таким номером уже загружен."""
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

    def upload_service_file(self, path: str) -> None:
        """prompts.json / generated_links.txt: только в основной релиз."""
        self._ensure_prepared()
        name = os.path.basename(path)
        if name not in SERVICE_FILES:
            raise GeminiInputError(f"Не служебный файл: {name}")
        self.store.upload_asset(self.tag, path)
        self._counts[self.tag] = self._counts.get(self.tag, 0) + 1

    def links_text(self, repo: str) -> str:
        """Содержимое generated_links.txt по всем загруженным клипам."""
        names = sorted(self.asset_tags,
                       key=lambda n: (parse_clip_num(n, self.template) or 0, n))
        return create_generated_links(repo, self.tag, names, self.asset_tags)
