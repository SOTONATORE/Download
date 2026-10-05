import os

import pytest

from Generate.core.srt_parser import (
    Segment, SrtError, check_srt, parse_srt, parse_srt_text, srt_hash,
)

NORMAL = """1
00:00:00,000 --> 00:00:03,843
Hello
world.

2
00:00:04,000 --> 00:00:06,500
Second line.

3
00:00:07.000 --> 00:00:09.250
Dot separator.
"""


def test_normal():
    segs = parse_srt_text(NORMAL)
    assert [s.num for s in segs] == [1, 2, 3]
    assert segs[0] == Segment(1, 0, 3843, "Hello world.")
    assert segs[2].start_ms == 7000 and segs[2].end_ms == 9250


def test_crlf():
    segs = parse_srt_text(NORMAL.replace("\n", "\r\n"))
    assert len(segs) == 3 and segs[0].text == "Hello world."


def test_bom_in_text_and_file(tmp_path):
    assert len(parse_srt_text("\ufeff" + NORMAL)) == 3
    p = tmp_path / "a.srt"
    p.write_bytes(b"\xef\xbb\xbf" + NORMAL.encode("utf-8"))
    assert len(parse_srt(str(p))) == 3


def test_tags_removed():
    raw = "1\n00:00:00,000 --> 00:00:01,000\n{\\an8}<i>Hi</i>  <b>there</b>\n"
    assert parse_srt_text(raw)[0].text == "Hi there"


def test_empty_reply_allowed():
    raw = "1\n00:00:00,000 --> 00:00:01,000\n\n\n2\n00:00:01,000 --> 00:00:02,000\nText\n"
    segs = parse_srt_text(raw)
    assert segs[0].text == "" and segs[1].text == "Text"
    assert any("пустой текст" in w and "1" in w for w in check_srt(segs))


def test_sorted_by_num():
    raw = ("2\n00:00:01,000 --> 00:00:02,000\nB\n\n"
           "1\n00:00:00,000 --> 00:00:01,000\nA\n")
    assert [s.num for s in parse_srt_text(raw)] == [1, 2]


def test_empty_file():
    for raw in ("", "   \n\n", "\ufeff"):
        with pytest.raises(SrtError, match="пуст"):
            parse_srt_text(raw)


def test_duplicate_numbers():
    raw = ("1\n00:00:00,000 --> 00:00:01,000\nA\n\n"
           "1\n00:00:01,000 --> 00:00:02,000\nB\n")
    with pytest.raises(SrtError, match=r"Дублирующиеся.*\[1\]"):
        parse_srt_text(raw)


def test_missing_number():
    raw = ("1\n00:00:00,000 --> 00:00:01,000\nA\n\n"
           "3\n00:00:01,000 --> 00:00:02,000\nB\n")
    with pytest.raises(SrtError, match=r"Пропущены.*\[2\]"):
        parse_srt_text(raw)


def test_broken_block_not_skipped():
    raw = ("1\n00:00:00,000 --> 00:00:01,000\nA\n\n"
           "2\n00:00:01,000 -> 00:00:02,000\nB\n\n"
           "3\n00:00:02,000 --> 00:00:03,000\nC\n")
    with pytest.raises(SrtError, match="разобрать"):
        parse_srt_text(raw)


def test_garbage_block():
    raw = "1\n00:00:00,000 --> 00:00:01,000\nA\n\nмусор без номера\n"
    with pytest.raises(SrtError, match="разобрать"):
        parse_srt_text(raw)


def test_end_before_start():
    raw = "1\n00:00:05,000 --> 00:00:01,000\nA\n"
    with pytest.raises(SrtError, match=r"end < start.*\[1\]"):
        parse_srt_text(raw)


def test_check_overlap_and_zero():
    segs = [Segment(1, 0, 2000, "a"), Segment(2, 1500, 1500, "b"),
            Segment(3, 3000, 4000, "c")]
    w = check_srt(segs)
    assert any("Сегмент 2" in x and "перекрыв" in x for x in w)
    assert any("Сегмент 2" in x and "нулев" in x for x in w)
    assert not any("Сегмент 3" in x for x in w)
    assert check_srt([Segment(1, 0, 1000, "a"), Segment(2, 1000, 2000, "b")]) == []


def test_hash_stable_and_sensitive():
    a = parse_srt_text(NORMAL)
    assert srt_hash(a) == srt_hash(parse_srt_text(NORMAL))
    b = list(a)
    b[1] = Segment(2, 4000, 6501, "Second line.")
    assert srt_hash(a) != srt_hash(b)


def _find_real_srt():
    here = os.path.dirname(__file__)
    for p in (os.environ.get("SRT_TEST_FILE", ""),
              os.path.join(here, "data", "final.srt"),
              os.path.join(here, "final.srt")):
        if p and os.path.exists(p):
            return p
    return None


def test_real_srt():
    path = _find_real_srt()
    if path is None:
        pytest.skip("тестовый SRT не найден")
    segs = parse_srt(path)
    assert len(segs) == segs[-1].num - segs[0].num + 1
    assert segs[0].num == 1
    assert all(s.end_ms >= s.start_ms for s in segs)
