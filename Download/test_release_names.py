"""Офлайн-тесты раскладки релизов стоков (стадия 6): имена, теги, части, повтор, дубликаты.

Без сети и без настоящего gh: хранилище подменено FakeStore в памяти.
Запуск (по решению заказчика не выполняется в этой стадии): python -m pytest Download/test_release_names.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import publish_stock as ps  # noqa: E402
from Generate.core import release_names as rn  # noqa: E402
from Generate.core.srt_parser import Segment, srt_hash  # noqa: E402

NOW = datetime(2026, 11, 7, 22, 35, tzinfo=timezone.utc)
SEGS = [Segment(1, 0, 1000, "a"), Segment(2, 1000, 2000, "b")]
FULL = srt_hash(SEGS)
H16 = FULL[:16]


class FakeStore:
    """Хранилище в памяти: тег -> список релизов; загрузки и создания записываются."""

    def __init__(self):
        self.releases = {}   # tag -> list[ps.Rel]
        self.created = []    # (tag, title, body)
        self.uploads = []    # (tag, basename)

    def add(self, tag, body, assets=(), title=""):
        self.releases.setdefault(tag, []).append(ps.Rel(tag, title, body, list(assets)))

    def find_releases(self, tag):
        return list(self.releases.get(tag, []))

    def create_release(self, tag, title, body):
        self.created.append((tag, title, body))
        self.add(tag, body, title=title)

    def upload_asset(self, tag, path):
        name = Path(path).name
        self.uploads.append((tag, name))
        rel = self.releases[tag][0]
        if name not in rel.assets:
            rel.assets.append(name)


def pub(store, limit=ps.ASSET_PART_LIMIT, full=FULL, repo="owner/repo"):
    return ps.StockPublisher(store, full, repo, limit=limit, now=lambda: NOW, log=lambda m: None)


def media(n, prefix="clip"):
    return [f"/tmp/stock/{prefix}_{i:04d}.mp4" for i in range(n)]


# --- имена, теги, заголовки: шесть видов -------------------------------------

def test_six_kinds_tags_titles_and_bodies():
    assert rn.final_tag("stock", FULL) == f"final-stock-{H16}"
    assert rn.final_tag("gen", FULL) == f"final-gen-{H16}"
    assert rn.part_tag("stock", 1, FULL) == f"stock-part1-{H16}"
    assert rn.part_tag("stock", 2, FULL) == f"stock-part2-{H16}"
    assert rn.part_tag("gen", 1, FULL) == f"gen-part1-{H16}"
    assert rn.part_tag("gen", 2, FULL) == f"gen-part2-{H16}"

    assert rn.final_title("stock", FULL, NOW) == f"ФИНАЛ · СТОКИ · 2026-11-07 22:35 UTC · {H16}"
    assert rn.final_title("gen", FULL, NOW) == f"ФИНАЛ · ГЕНЕРАЦИЯ · 2026-11-07 22:35 UTC · {H16}"
    assert rn.part_title("stock", 1, FULL, NOW) == f"СТОКИ · ЧАСТЬ 1 · 2026-11-07 22:35 UTC · {H16}"
    assert rn.part_title("gen", 2, FULL, NOW) == f"ГЕНЕРАЦИЯ · ЧАСТЬ 2 · 2026-11-07 22:35 UTC · {H16}"

    # номер части есть только у частей
    assert "ЧАСТЬ" not in rn.final_title("stock", FULL, NOW)
    assert FULL in rn.release_body("stock", "final", FULL)
    assert FULL in rn.release_body("stock", "part", FULL)


def test_hash16_is_first_16_hex_chars_and_same_function_both_sides():
    assert rn.hash16(FULL) == H16 and len(H16) == 16
    # Download берёт хэш той же функцией, что Generate (srt_hash из srt_parser)
    assert ps.srt_hash is srt_hash


def test_title_time_is_utc():
    from datetime import timedelta
    msk = timezone(timedelta(hours=3))
    when = datetime(2026, 11, 8, 1, 35, tzinfo=msk)   # то же, что 22:35 UTC
    assert rn.final_title("stock", FULL, when) == f"ФИНАЛ · СТОКИ · 2026-11-07 22:35 UTC · {H16}"


# --- части: 900 файлов в части --------------------------------------------------

def test_1000_files_go_to_two_parts_900_and_100():
    s = FakeStore()
    p = pub(s)
    p.open()
    p.upload_media(media(1000))
    sizes = [p.counts[t] for t in p.parts]
    assert sizes == [900, 100]
    assert p.parts == [f"stock-part1-{H16}", f"stock-part2-{H16}"]
    part_uploads = [t for t, _ in s.uploads if t.startswith("stock-part")]
    assert part_uploads.count(f"stock-part1-{H16}") == 900
    assert part_uploads.count(f"stock-part2-{H16}") == 100


def test_1500_files_parts_900_600_and_final_has_1500_links():
    s = FakeStore()
    p = pub(s)
    p.open()
    p.upload_media(media(1500))
    assert [p.counts[t] for t in p.parts] == [900, 600]
    links = p.links_text().splitlines()
    assert len(links) == 1500
    assert links[0].startswith(f"https://github.com/owner/repo/releases/download/stock-part1-{H16}/")
    assert links[-1].startswith(f"https://github.com/owner/repo/releases/download/stock-part2-{H16}/")


def test_part_title_numbering_only_for_parts_in_created_list():
    s = FakeStore()
    p = pub(s)
    p.open()
    p.upload_media(media(901))
    titles = [t for tag, t, _ in s.created]
    assert titles[0] == rn.final_title("stock", FULL, NOW)
    assert titles[1] == rn.part_title("stock", 1, FULL, NOW)
    assert titles[2] == rn.part_title("stock", 2, FULL, NOW)


# --- повтор: тот же хэш - продолжаем, новый финал не создаём ---------------------

def test_existing_final_with_same_hash_is_continued():
    s = FakeStore()
    s.add(f"final-stock-{H16}", rn.release_body("stock", "final", FULL), ["missing.txt"])
    s.add(f"stock-part1-{H16}", rn.release_body("stock", "part", FULL), ["clip_0000.mp4"])
    p = pub(s)
    p.open()
    assert s.created == []                                  # финал не создан заново
    assert p.done == {"clip_0000.mp4"}
    p.upload_media(media(2))
    assert [t for t, _ in s.uploads if t.startswith("stock-part")] == [f"stock-part1-{H16}"]


def test_resume_skips_media_already_in_parts():
    s = FakeStore()
    s.add(f"final-stock-{H16}", rn.release_body("stock", "final", FULL))
    s.add(f"stock-part1-{H16}", rn.release_body("stock", "part", FULL),
          [f"clip_{i:04d}.mp4" for i in range(3)])
    p = pub(s)
    p.open()
    n = p.upload_media(media(5))
    assert n == 2
    assert sorted(name for _, name in s.uploads if name.endswith(".mp4")) == [
        "clip_0003.mp4", "clip_0004.mp4"]


# --- дубликаты и чужое содержимое: остановка с ошибкой ----------------------------

def test_two_releases_with_same_tag_stop_with_error():
    s = FakeStore()
    body = rn.release_body("stock", "final", FULL)
    s.add(f"final-stock-{H16}", body)
    s.add(f"final-stock-{H16}", body)
    with pytest.raises(ps.PublishError) as e:
        pub(s).open()
    assert str(e.value) == f"Найдено несколько релизов с тегом final-stock-{H16}, удалите лишние вручную."


def test_same_tag_other_content_stops_with_error():
    s = FakeStore()
    s.add(f"final-stock-{H16}", "hash: " + "0" * 64)
    with pytest.raises(ps.PublishError) as e:
        pub(s).open()
    assert "Тег совпал, содержимое нет" in str(e.value)


def test_empty_text_files_are_not_uploaded(tmp_path):
    s = FakeStore()
    p = pub(s)
    p.open()
    empty = tmp_path / "download_failed.txt"
    empty.write_bytes(b"")
    full = tmp_path / "missing.txt"
    full.write_text("1\n", encoding="utf-8")
    p.upload_text([str(empty), str(full), str(tmp_path / "нет.txt")])
    assert s.uploads == [(f"final-stock-{H16}", "missing.txt")]


def test_links_file_replaced_each_time_and_lists_media_only():
    s = FakeStore()
    p = pub(s)
    p.open()
    p.upload_media(media(2))
    link_uploads = [(t, n) for t, n in s.uploads if n == ps.LINKS_FILENAME]
    assert link_uploads and all(t == f"final-stock-{H16}" for t, _ in link_uploads)
    assert p.links_text().count("\n") == 2
    assert all(ps.LINKS_FILENAME != n for t, n in s.uploads if t.startswith("stock-part"))
