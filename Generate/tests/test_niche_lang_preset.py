"""Ниша cartoon (N6): текст правила на кадре, allow_text, пресет ниши, язык SRT (lingua).

Тесты написаны, не запускались (прогоняет пользователь на GitHub). Тесты с lingua пропускаются,
если библиотеки нет (pytest.importorskip).
"""
from pathlib import Path

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import GeminiInputError

NICHE_DIR = Path(__file__).resolve().parents[1] / "Nishes" / "cartoon"
RUSSIAN_SRT = (
    "1\n00:00:00,000 --> 00:00:03,000\nВ центральном архиве есть картонная карточка.\n\n"
    "2\n00:00:03,100 --> 00:00:06,000\nНа этой карточке записано немного.\n"
)
ENGLISH_SRT = "1\n00:00:00,000 --> 00:00:02,000\nThe cat is on the table and it is black.\n"


# --- правило текста на кадре в style.md ----------------------------------------------------------

def test_style_text_rule_has_the_four_word_and_one_word_per_part_rules():
    text = (NICHE_DIR / "prompts" / "style.md").read_text(encoding="utf-8")
    # Точные фразы из style.md (абзац про текст на кадре, строка 29 и раздел Final Check).
    for phrase in ("1 to 4 words", "one word for each part", "at most 4 parts",
                   "exactly one text phrase per frame",  # режим текстовых кадров (a)
                   "one main title plus the short labels or dates that the layout requires"):  # режим сравнения и таймлайна (b)
        assert phrase in text, phrase


# --- allow_text -----------------------------------------------------------------------------------

def test_allow_text_true_adds_frame_text_to_relaxed_set():
    lines = ["type: image", "allow_text: true", "relaxed_rules: [art_medium]", 'text_language: "srt"']
    relaxed, text_language = gp.profile_relaxation(lines, "x")
    assert relaxed == frozenset({"art_medium", "frame_text"})
    assert text_language == "srt"


def test_allow_text_true_still_requires_text_language():
    with pytest.raises(GeminiInputError, match="text_language"):
        gp.profile_relaxation(["type: image", "allow_text: true"], "x")


def test_allow_text_absent_does_not_relax_frame_text():
    relaxed, text_language = gp.profile_relaxation(["type: image", "relaxed_rules: [art_medium]"], "x")
    assert relaxed == frozenset({"art_medium"})
    assert text_language is None


def test_allow_text_bad_value_is_error():
    with pytest.raises(GeminiInputError, match="allow_text"):
        gp.profile_relaxation(["allow_text: yes"], "x")


def test_real_cartoon_profile_relaxes_frame_text_through_allow_text():
    lines = (NICHE_DIR / "profile.yaml").read_text(encoding="utf-8").splitlines()
    relaxed, text_language = gp.profile_relaxation(lines, "cartoon")
    assert "frame_text" in relaxed
    assert text_language == "srt"


# --- пресет ниши ----------------------------------------------------------------------------------

def test_preset_none_reads_niche_preset_file():
    path = NICHE_DIR / "prompts" / "preset.md"
    assert gp.preset_text("none", str(path)) == path.read_text(encoding="utf-8").strip()


def test_preset_none_without_niche_file_is_empty():
    assert gp.preset_text("none", None) == ""


def test_named_preset_wins_over_niche_file(tmp_path, monkeypatch):
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "named.md").write_text("Named preset text.\n", encoding="utf-8")
    monkeypatch.setattr(gp, "PRESETS_DIR", presets)
    assert gp.preset_text("named", str(NICHE_DIR / "prompts" / "preset.md")) == "Named preset text."


def test_empty_niche_preset_file_is_error(tmp_path):
    empty = tmp_path / "preset.md"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(GeminiInputError, match="пуст"):
        gp.preset_text("none", str(empty))


def test_cli_finds_niche_preset_only_for_none():
    from Generate import cli
    found = cli._niche_preset_file("cartoon", "none")
    assert found is not None and found.endswith("prompts/preset.md")
    assert cli._niche_preset_file("cartoon", "named") is None


# --- язык текста по SRT (lingua) -----------------------------------------------------------------

def test_srt_russian_is_detected(tmp_path):
    pytest.importorskip("lingua")
    path = tmp_path / "ru.srt"
    path.write_text(RUSSIAN_SRT, encoding="utf-8")
    assert gp.detect_srt_language(path) == "Russian"


def test_srt_english_is_detected(tmp_path):
    pytest.importorskip("lingua")
    path = tmp_path / "en.srt"
    path.write_text(ENGLISH_SRT, encoding="utf-8")
    assert gp.detect_srt_language(path) == "English"


def test_srt_mixed_text_is_russian_error(tmp_path):
    pytest.importorskip("lingua")
    path = tmp_path / "mixed.srt"
    path.write_text("Привет друзья. hello friends. Привет друзья. hello friends.\n", encoding="utf-8")
    with pytest.raises(GeminiInputError, match="не определён"):
        gp.detect_srt_language(path)


def test_srt_too_short_is_russian_error(tmp_path):
    path = tmp_path / "short.srt"
    path.write_text("1\n00:00:00,000 --> 00:00:01,000\nДа.\n", encoding="utf-8")
    with pytest.raises(GeminiInputError, match="мало букв"):
        gp.detect_srt_language(path)
