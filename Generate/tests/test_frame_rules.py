"""Тесты запрещённых типов кадров и правила перефразирования (шаг 4 плана). Не запускаются в этой сессии."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Generate.core import frame_rules as fr  # noqa: E402
from Generate.core import prompt_check as pc  # noqa: E402


# Тексты с запрещённым типом кадра (английский текст сегмента): правило должно сработать.
POSITIVE = [
    "The passport lies on the table.",
    "Passports were stamped for exile, not return.",  # сегмент 132
    "A map of the empire hangs in the hall.",
    "A spreadsheet with budget figures on a desk.",
]

# Тексты без запрещённых типов: правило не должно сработать.
NEGATIVE = [
    "A formal dinner in a palace hall.",   # formal: исключение «formal»
    "The format of the old hall is grand.",  # format: исключение «format»
    "A bed sheet lies on the floor of the room.",  # sheet без spread: не запрещено
    "A sheet of ice covers the lake.",
]


@pytest.mark.parametrize("text", POSITIVE)
def test_needs_rephrase_true_for_forbidden_frame(text):
    assert fr.needs_rephrase(text) is True


@pytest.mark.parametrize("text", NEGATIVE)
def test_needs_rephrase_false_without_forbidden_frame(text):
    assert fr.needs_rephrase(text) is False


def test_rephrase_note_empty_for_clean_text():
    assert fr.rephrase_note("A formal dinner in a palace hall.") == ""


def test_rephrase_note_returns_rule_for_forbidden_text():
    assert fr.rephrase_note("A map of the empire hangs in the hall.") == fr.REPHRASE_RULE_EN


def test_regex_matches_word_forms_of_map():
    assert fr.FORBIDDEN_FRAME_RE.search("maps") is not None
    assert fr.FORBIDDEN_FRAME_RE.search("Maps") is not None


def test_regex_does_not_match_format_word():
    # «form» покрывает formats/forms, но не «format» (исключение в _STEM_EXCLUDE).
    assert fr.needs_rephrase("format of the frame") is False


def test_frame_rules_uses_the_list_from_prompt_check():
    # Список не копируется: это тот же объект, что и в prompt_check.py.
    assert fr.FORBIDDEN_IMAGE_WORDS_EN is pc.FORBIDDEN_IMAGE_WORDS_EN
    assert fr._STEM_EXCLUDE is pc._STEM_EXCLUDE


def test_rule_is_three_to_five_sentences():
    sentences = [s for s in fr.REPHRASE_RULE_EN.split(".") if s.strip()]
    assert 3 <= len(sentences) <= 5
