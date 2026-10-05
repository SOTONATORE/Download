"""Офлайн-тесты core/missing.py."""
from __future__ import annotations

import logging

import pytest

try:
    from Generate.core.gemini_prompts import GeminiInputError
    from Generate.core.missing import STUB_LINE, parse_missing
    from Generate.core.srt_parser import Segment
except ImportError:  # запуск из каталога Generate
    from core.gemini_prompts import GeminiInputError  # type: ignore
    from core.missing import STUB_LINE, parse_missing  # type: ignore
    from core.srt_parser import Segment  # type: ignore


@pytest.fixture
def segments():
    return [Segment(n, n * 1000, n * 1000 + 500, f"текст {n}") for n in range(1, 11)]


def _write(tmp_path, data: bytes):
    p = tmp_path / "missing.txt"
    p.write_bytes(data)
    return p


def test_normal(tmp_path, segments):
    assert parse_missing(_write(tmp_path, b"1\n2\n3"), segments) == [1, 2, 3]


def test_accepts_str_path(tmp_path, segments):
    assert parse_missing(str(_write(tmp_path, b"4\n")), segments) == [4]


def test_sorted_ascending(tmp_path, segments):
    assert parse_missing(_write(tmp_path, b"7\n2\n5\n"), segments) == [2, 5, 7]


def test_bom(tmp_path, segments):
    assert parse_missing(_write(tmp_path, b"\xef\xbb\xbf1\n2\n"), segments) == [1, 2]


def test_crlf_and_cr(tmp_path, segments):
    assert parse_missing(_write(tmp_path, b"1\r\n2\r\n3\r\n"), segments) == [1, 2, 3]
    assert parse_missing(_write(tmp_path, b"1\r2\r3"), segments) == [1, 2, 3]


def test_whitespace_and_blank_lines(tmp_path, segments):
    data = b"  1  \n\n\t2\t \n   \n3\n\n"
    assert parse_missing(_write(tmp_path, data), segments) == [1, 2, 3]


def test_duplicates_merged_and_warned(tmp_path, segments, caplog):
    with caplog.at_level(logging.WARNING, logger="Generate.missing"):
        res = parse_missing(_write(tmp_path, b"3\n1\n3\n3\n1\n"), segments)
    assert res == [1, 3]
    assert any("повторяющихся" in r.getMessage() for r in caplog.records)


def test_leading_zeros(tmp_path, segments):
    assert parse_missing(_write(tmp_path, b"001\n02\n"), segments) == [1, 2]


def test_stub_returns_empty(tmp_path, segments):
    assert parse_missing(_write(tmp_path, (STUB_LINE + "\n").encode()), segments) == []


def test_stub_case_insensitive_trimmed_bom_crlf(tmp_path, segments):
    data = b"\xef\xbb\xbf\r\n   " + STUB_LINE.upper().encode() + b"   \r\n\r\n"
    assert parse_missing(_write(tmp_path, data), segments) == []


@pytest.mark.parametrize("data", [b"", b"   \n\r\n\t\n"])
def test_empty_file(tmp_path, segments, data):
    with pytest.raises(GeminiInputError) as ei:
        parse_missing(_write(tmp_path, data), segments)
    assert ei.value.exit_code == 2
    assert "пуст" in str(ei.value)


@pytest.mark.parametrize("bad", ["abc", "1.5", "-3", "1, 2", "12a", "²"])
def test_invalid_line(tmp_path, segments, bad):
    with pytest.raises(GeminiInputError) as ei:
        parse_missing(_write(tmp_path, f"1\n{bad}\n".encode()), segments)
    assert ei.value.exit_code == 2
    assert bad in str(ei.value)
    assert "недопустимая строка" in str(ei.value)


def test_stub_with_extra_text_is_invalid(tmp_path, segments):
    with pytest.raises(GeminiInputError):
        parse_missing(_write(tmp_path, (STUB_LINE + " Ещё\n").encode()), segments)


def test_numbers_mixed_with_stub(tmp_path, segments):
    data = ("1\n" + STUB_LINE + "\n2\n").encode()
    with pytest.raises(GeminiInputError) as ei:
        parse_missing(_write(tmp_path, data), segments)
    assert ei.value.exit_code == 2
    assert "одновременно" in str(ei.value)


def test_numbers_not_in_srt(tmp_path, segments):
    with pytest.raises(GeminiInputError) as ei:
        parse_missing(_write(tmp_path, b"1\n11\n99\n"), segments)
    assert ei.value.exit_code == 2
    msg = str(ei.value)
    assert "[11, 99]" in msg
    assert "отсутствуют в SRT" in msg


def test_file_not_found(tmp_path, segments):
    with pytest.raises(GeminiInputError) as ei:
        parse_missing(tmp_path / "nope.txt", segments)
    assert "не найден" in str(ei.value)


def test_not_utf8(tmp_path, segments):
    with pytest.raises(GeminiInputError) as ei:
        parse_missing(_write(tmp_path, b"\xff\xfe\x00\x80"), segments)
    assert "UTF-8" in str(ei.value)


def test_messages_are_russian_and_no_traceback(tmp_path, segments):
    with pytest.raises(GeminiInputError) as ei:
        parse_missing(_write(tmp_path, b"xyz"), segments)
    msg = str(ei.value)
    assert any("а" <= ch.lower() <= "я" for ch in msg)
    assert "Traceback" not in msg
    assert ei.value.__cause__ is None
