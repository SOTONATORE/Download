"""Офлайн-тесты Generate/core/naming.py (без сети, только pytest)."""

import pytest

try:
    from Generate.core.naming import clip_filename, parse_clip_num
except ImportError:
    from core.naming import clip_filename, parse_clip_num


# --- clip_filename ---

@pytest.mark.parametrize(
    "num, expected",
    [(0, "0.mp4"), (7, "7.mp4"), (42, "42.mp4"), (128, "128.mp4")],
)
def test_clip_filename_width_zero_default(num, expected):
    assert clip_filename(num) == expected


@pytest.mark.parametrize("width", [0, -1, -10])
def test_clip_filename_non_positive_width_no_padding(width):
    assert clip_filename(7, width=width) == "7.mp4"


@pytest.mark.parametrize(
    "num, expected",
    [(7, "007.mp4"), (42, "042.mp4"), (128, "128.mp4"), (0, "000.mp4")],
)
def test_clip_filename_width_3(num, expected):
    assert clip_filename(num, width=3) == expected


@pytest.mark.parametrize(
    "num, expected",
    [(7, "0007.mp4"), (42, "0042.mp4"), (1234, "1234.mp4")],
)
def test_clip_filename_width_4(num, expected):
    assert clip_filename(num, width=4) == expected


def test_clip_filename_number_longer_than_width_not_truncated():
    assert clip_filename(12345, width=3) == "12345.mp4"


def test_clip_filename_custom_template():
    assert clip_filename(5, width=3, template="clip_{num}.mov") == "clip_005.mov"
    assert clip_filename(5, template="seg-{num}.mp4") == "seg-5.mp4"


@pytest.mark.parametrize("num", [-1, -42])
def test_clip_filename_negative_raises_russian_message(num):
    with pytest.raises(ValueError) as exc:
        clip_filename(num)
    assert "Номер клипа" in str(exc.value)


# --- parse_clip_num ---

@pytest.mark.parametrize(
    "name, expected",
    [("7.mp4", 7), ("42.mp4", 42), ("0042.mp4", 42), ("0.mp4", 0), ("000.mp4", 0)],
)
def test_parse_clip_num_valid(name, expected):
    assert parse_clip_num(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "prompts.json",
        "generated_links.txt",
        "missing.txt",
        "7.mkv",
        "7.MP4",
        "7.mp4.part",
        "clip7.mp4",
        "a7.mp4",
        "-5.mp4",
        "1.5.mp4",
        ".mp4",
        "",
    ],
)
def test_parse_clip_num_non_clip_returns_none(name):
    assert parse_clip_num(name) is None


def test_parse_clip_num_with_directory_path():
    assert parse_clip_num("/path/to/12.mp4") == 12
    assert parse_clip_num("relative/dir/0042.mp4") == 42
    assert parse_clip_num("/path/to/prompts.json") is None


def test_parse_clip_num_digits_in_directory_ignored():
    assert parse_clip_num("/data/123/notes.txt") is None
    assert parse_clip_num("/data/123/9.mp4") == 9


def test_parse_clip_num_custom_template():
    assert parse_clip_num("clip_005.mov", template="clip_{num}.mov") == 5
    assert parse_clip_num("5.mp4", template="clip_{num}.mov") is None


def test_parse_clip_num_template_without_placeholder_raises():
    with pytest.raises(ValueError) as exc:
        parse_clip_num("7.mp4", template="clip.mp4")
    assert "плейсхолдер" in str(exc.value)


# --- круговая проверка ---

@pytest.mark.parametrize("width", [0, 3, 4])
@pytest.mark.parametrize("num", [0, 7, 42, 128, 9999])
def test_roundtrip(num, width):
    assert parse_clip_num(clip_filename(num, width=width)) == num
