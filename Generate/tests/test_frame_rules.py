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


# Тексты с запрещённым типом кадра: правило должно сработать.
POSITIVE = [
    "На карте видно Османскую империю",
    "в газете напечатали",
    "покажем диаграмму",
    "паспорт солдата",
    "флаг над дворцом",
]

# Тексты без запрещённых типов: правило не должно сработать.
NEGATIVE = [
    "солдаты идут по улице",
    "мечеть на закате",
    "корабль в проливе",
]


@pytest.mark.parametrize("text", POSITIVE)
def test_needs_rephrase_true_for_forbidden_frame(text):
    assert fr.needs_rephrase(text) is True


@pytest.mark.parametrize("text", NEGATIVE)
def test_needs_rephrase_false_without_forbidden_frame(text):
    assert fr.needs_rephrase(text) is False


def test_rephrase_note_empty_for_clean_text():
    assert fr.rephrase_note("мечеть на закате") == ""


def test_rephrase_note_returns_rule_for_forbidden_text():
    assert fr.rephrase_note("На карте видно Османскую империю") == fr.REPHRASE_RULE_RU


def test_regex_matches_word_forms_of_map():
    assert fr.FORBIDDEN_FRAME_RE.search("карту") is not None
    assert fr.FORBIDDEN_FRAME_RE.search("карты") is not None


def test_regex_does_not_match_picture():
    # «картина» начинается с «карт», но это не запрещённый тип кадра (исключение в _STEM_EXCLUDE).
    assert fr.FORBIDDEN_FRAME_RE.search("картина") is None
    assert fr.needs_rephrase("картина на стене") is False


def test_regex_does_not_match_format_word():
    # «форм» покрывает формы, но не «формат» (исключение в _STEM_EXCLUDE).
    assert fr.needs_rephrase("формат кадра") is False


def test_rule_text_names_no_forbidden_object_as_instruction():
    # Правило не должно называть запрещённые предметы словами, которые модель могла бы
    # принять за задание их нарисовать.
    lowered = fr.REPHRASE_RULE_RU.lower()
    for word in ("карта", "календарь", "газета", "паспорт"):
        assert f" {word} " not in f" {lowered} ", word


def test_rule_is_three_to_five_sentences():
    sentences = [s for s in fr.REPHRASE_RULE_RU.split(".") if s.strip()]
    assert 3 <= len(sentences) <= 5
