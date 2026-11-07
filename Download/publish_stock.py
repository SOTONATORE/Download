"""publish_stock.py - выдача стоков в GitHub Releases (стадия 6, раскладка релизов).

Раскладка на один final.srt (хэш hash16 = первые 16 символов sha256 от srt_hash):
  ФИНАЛ · СТОКИ          тег final-stock-<hash16>    текстовые файлы стоков и stock_links.txt; медиа нет
  СТОКИ · ЧАСТЬ N        тег stock-part<N>-<hash16>  только стоковые медиа, не больше 900 файлов в части
Имена, теги и описания - Generate/core/release_names.py (общие с генерацией); хэш - srt_hash
из Generate/core/srt_parser.py: оба конвейера считают его одной функцией.

Команды:
  open    --srt final.srt                       открыть (или продолжить) ФИНАЛ · СТОКИ
  upload  --srt final.srt --media-dir DIR [--text FILE ...]
                                                текстовые файлы, медиа по частям, stock_links.txt

Токен берётся только из GH_TOKEN (SPEC 0.1), в командную строку и в логи не попадает.
Репозиторий: --repo или GITHUB_REPOSITORY. Все сообщения на русском языке.
Скрипт работает на раннере без зависимостей Generate (только stdlib и Generate/core/*.py).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

# репозиторий в sys.path: `python Download/publish_stock.py` кладёт в путь только каталог Download/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Generate.core.release_names import (  # noqa: E402
    final_tag, final_title, part_tag, part_title, release_body, hash16)
from Generate.core.srt_parser import SrtError, parse_srt, srt_hash  # noqa: E402

STAGE = "stock"
ASSET_PART_LIMIT = 900          # не больше 900 файлов в части (лимит GitHub 1000 на релиз, запас)
LINKS_FILENAME = "stock_links.txt"
_TOKEN_RE = re.compile(r"(gh[pousr]_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,})")


class PublishError(Exception):
    """Ошибка выдачи с текстом на русском языке (показывается в логе как есть)."""


@dataclasses.dataclass
class Rel:
    """Релиз, как его видит публикатор: тег, заголовок, описание, имена ассетов."""
    tag: str
    title: str = ""
    body: str = ""
    assets: list = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Хранилище: настоящее (gh CLI) и подмена для тестов
# ---------------------------------------------------------------------------

class GhStore:
    """Хранилище поверх gh CLI. Токен только в окружении процесса gh."""

    _JQ = '.[] | {tag_name, name, body, assets: [.assets[].name]} | tojson'

    def __init__(self, repo: str, token: str, gh_bin: str = "gh",
                 run: Callable = subprocess.run, timeout: int = 600):
        if not repo:
            raise PublishError("Не задан репозиторий (GITHUB_REPOSITORY или --repo).")
        if not token:
            raise PublishError("Не задан токен GitHub: ожидается переменная окружения GH_TOKEN.")
        self.repo = repo
        self._token = token
        self._gh = gh_bin
        self._run = run
        self._timeout = timeout

    def _call(self, args: Sequence[str]) -> str:
        env = dict(os.environ)
        env["GH_TOKEN"] = self._token
        try:
            proc = self._run([self._gh, *args], env=env, capture_output=True, text=True,
                             timeout=self._timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PublishError(f"Не удалось выполнить gh ({type(exc).__name__}).") from None
        if proc.returncode != 0:
            detail = _TOKEN_RE.sub("***", (proc.stderr or "").strip().replace(self._token, "***"))
            raise PublishError(f"Команда gh завершилась с кодом {proc.returncode}: {detail}")
        return (proc.stdout or "").replace(self._token, "***")

    def find_releases(self, tag: str) -> list:
        import json
        out = self._call(["api", "--paginate", f"repos/{self.repo}/releases?per_page=100",
                          "--jq", self._JQ])
        found = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                raise PublishError("Не удалось разобрать список релизов от GitHub.") from None
            if d.get("tag_name") == tag:
                found.append(Rel(tag=tag, title=d.get("name") or "", body=d.get("body") or "",
                                 assets=list(d.get("assets") or [])))
        return found

    def create_release(self, tag: str, title: str, body: str) -> None:
        self._call(["release", "create", tag, "--repo", self.repo, "--title", title, "--notes", body])

    def upload_asset(self, tag: str, path: str) -> None:
        self._call(["release", "upload", tag, path, "--repo", self.repo, "--clobber"])


# ---------------------------------------------------------------------------
# Выдача стоков
# ---------------------------------------------------------------------------

def check_release(store, tag: str, full_hash: str) -> Optional[Rel]:
    """Один релиз с тегом (или None). Два релиза с одним тегом - остановка; без полного хэша в описании - тоже."""
    found = store.find_releases(tag)
    if len(found) > 1:
        raise PublishError(f"Найдено несколько релизов с тегом {tag}, удалите лишние вручную.")
    if not found:
        return None
    if full_hash not in (found[0].body or ""):
        raise PublishError("Тег совпал, содержимое нет: возможен другой final.srt.")
    return found[0]


class StockPublisher:
    """Выдача стоков одного запуска. Финал ищется по хэшу; уже выложенное пропускается."""

    def __init__(self, store, full_hash: str, repo: str,
                 limit: int = ASSET_PART_LIMIT,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 log: Callable[[str], None] = print):
        if limit <= 0:
            raise PublishError(f"Лимит файлов в части должен быть положительным: {limit}")
        self.store = store
        self.full = full_hash
        self.h16 = hash16(full_hash)
        self.repo = repo
        self.limit = limit
        self._now = now
        self._log = log
        self.final_tag = final_tag(STAGE, full_hash)
        self.parts: list = []        # теги частей СТОКИ по порядку
        self.counts: dict = {}       # тег части -> число файлов в ней
        self.asset_tags: dict = {}   # имя медиа -> тег части
        self.done: set = set()       # имена медиа, уже лежащие в частях
        self._opened = False

    def open(self) -> str:
        """Открывает ФИНАЛ · СТОКИ, если его нет; если есть с тем же хэшем - продолжает. Читает части."""
        rel = check_release(self.store, self.final_tag, self.full)
        if rel is None:
            self.store.create_release(self.final_tag, final_title(STAGE, self.full, self._now()),
                                      release_body(STAGE, "final", self.full))
            self._log(f"Создан релиз {self.final_tag} (ФИНАЛ · СТОКИ).")
        else:
            self._log(f"Продолжение: релиз {self.final_tag} уже есть, новый финал не создаётся.")
        n = 1
        while True:
            pt = part_tag(STAGE, n, self.full)
            part = check_release(self.store, pt, self.full)
            if part is None:
                break
            self.parts.append(pt)
            self.counts[pt] = len(part.assets)
            for name in part.assets:
                self.done.add(name)
                self.asset_tags[name] = pt
            n += 1
        self._opened = True
        self._log(f"В частях СТОКИ уже {len(self.done)} медиа-файлов.")
        return self.final_tag

    def _ensure_open(self) -> None:
        if not self._opened:
            self.open()

    def upload_text(self, paths: Sequence[str]) -> None:
        """Текстовые файлы стоков -> ФИНАЛ · СТОКИ. Пустые файлы не выкладываются (GitHub отвергает 0 байт)."""
        self._ensure_open()
        for path in paths:
            if not os.path.isfile(path):
                self._log(f"Файл {os.path.basename(path)} не найден: в ФИНАЛ · СТОКИ не выложен.")
                continue
            if os.path.getsize(path) == 0:
                self._log(f"Файл {os.path.basename(path)} пустой: в ФИНАЛ · СТОКИ не выложен.")
                continue
            self.store.upload_asset(self.final_tag, path)

    def _active_tag(self) -> str:
        """Часть, куда идёт очередной файл; при достижении лимита открывается следующая."""
        if self.parts and self.counts.get(self.parts[-1], 0) < self.limit:
            return self.parts[-1]
        n = len(self.parts) + 1
        pt = part_tag(STAGE, n, self.full)
        self.store.create_release(pt, part_title(STAGE, n, self.full, self._now()),
                                  release_body(STAGE, "part", self.full))
        self.parts.append(pt)
        self.counts[pt] = 0
        self._log(f"Создан релиз {pt} (СТОКИ · ЧАСТЬ {n}).")
        return pt

    def upload_media(self, paths: Sequence[str]) -> int:
        """Стоковые медиа -> части СТОКИ по одному файлу; после каждого - обновление stock_links.txt.
        Файл с уже выложенным именем пропускается. Возвращает число новых загрузок."""
        self._ensure_open()
        uploaded = 0
        for path in paths:
            name = os.path.basename(path)
            if name in self.done:
                continue
            tag = self._active_tag()
            self.store.upload_asset(tag, path)
            self.counts[tag] += 1
            self.asset_tags[name] = tag
            self.done.add(name)
            uploaded += 1
            self.refresh_links()
        if uploaded == 0:
            self.refresh_links()
        return uploaded

    def links_text(self) -> str:
        """Одна прямая ссылка на каждое выложенное медиа; пусто, если медиа нет."""
        base = f"https://github.com/{self.repo}/releases/download"
        return "".join(f"{base}/{self.asset_tags[name]}/{name}\n" for name in sorted(self.asset_tags))

    def refresh_links(self) -> None:
        """Перезаписывает stock_links.txt в ФИНАЛ · СТОКИ (gh release upload --clobber)."""
        text = self.links_text()
        if not text:
            return
        tmpdir = tempfile.mkdtemp(prefix="stock-links-")
        try:
            path = os.path.join(tmpdir, LINKS_FILENAME)
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            self.store.upload_asset(self.final_tag, path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def load_hash(srt_path: str) -> tuple:
    """Читает final.srt текущего запуска и возвращает (полный хэш, сегменты). Ошибка SRT - PublishError."""
    if not os.path.isfile(srt_path):
        raise PublishError(f"Файл {srt_path} не найден: выдача стоков в ФИНАЛ · СТОКИ невозможна.")
    try:
        segments = parse_srt(srt_path)
    except SrtError as exc:
        raise PublishError(f"final.srt не разобран: {exc}") from None
    return srt_hash(segments), segments


def media_files(media_dir: str) -> list:
    """Файлы стоков в каталоге (без подкаталогов), по имени."""
    if not os.path.isdir(media_dir):
        return []
    return [os.path.join(media_dir, n) for n in sorted(os.listdir(media_dir))
            if os.path.isfile(os.path.join(media_dir, n))]


# ---------------------------------------------------------------------------
# Командная строка
# ---------------------------------------------------------------------------

def _build_publisher(args) -> StockPublisher:
    full, _segments = load_hash(args.srt)
    repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
    store = GhStore(repo, os.environ.get("GH_TOKEN", ""))
    return StockPublisher(store, full, repo, limit=args.limit)


def cmd_open(args) -> int:
    pub = _build_publisher(args)
    tag = pub.open()
    print(f"ФИНАЛ · СТОКИ: {tag}")
    return 0


def cmd_upload(args) -> int:
    pub = _build_publisher(args)
    pub.open()
    pub.upload_text(args.text or [])
    n = pub.upload_media(media_files(args.media_dir))
    print(f"Стоки: новых медиа {n}, всего в частях {len(pub.done)}, частей {len(pub.parts)}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Выдача стоков в ФИНАЛ · СТОКИ и СТОКИ · ЧАСТЬ N (стадия 6).")
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn, help_text in (("open", cmd_open, "открыть или продолжить ФИНАЛ · СТОКИ"),
                                ("upload", cmd_upload, "выложить тексты и медиа стоков")):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--srt", required=True, help="final.srt текущего запуска (его хэш - тег)")
        sp.add_argument("--repo", default="", help="owner/name (по умолчанию GITHUB_REPOSITORY)")
        sp.add_argument("--limit", type=int, default=ASSET_PART_LIMIT, help="файлов в части (по умолчанию 900)")
        if name == "upload":
            sp.add_argument("--media-dir", default="downloaded_media", help="каталог стоковых медиа")
            sp.add_argument("--text", nargs="*", default=[], help="текстовые файлы стоков")
        sp.set_defaults(func=fn)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PublishError as exc:
        print(f"Ошибка выдачи стоков: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
