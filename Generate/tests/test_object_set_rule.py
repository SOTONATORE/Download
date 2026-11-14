"""Правило наборов предметов (rules.yaml: object_sets): стопки, кучи, ряды и линии предметов.

Тесты написаны, не запускались (прогоняет пользователь на GitHub).
Имя ниши cartoon_photos названо только здесь, в тесте; в коде его нет.
"""
import hashlib
from pathlib import Path

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core import prompt_check as pc
from Generate.core.gemini_prompts import GeminiInputError, PromptConfig, RulesValues

GEN = Path(__file__).resolve().parents[1]
NICHE_DIR = GEN / "Nishes" / "cartoon_photos"
LTX_DIR = GEN / "model_profiles" / "models" / "ltx25"
ZIM_DIR = GEN / "model_profiles" / "models" / "zimage_turbo"

# английская строка, которую добавляет правило object_sets: forbidden (одна на системную инструкцию)
OBJECT_SET_LINE = ("- Show one object, or two or three objects of one kind. "
                   "Never show stacks, piles, rows or lines of objects.\n")

# sha256 и длина системной инструкции ltx25 и zimage_turbo: скопированы из test_rules_values.py (тот файл не меняется)
LTX_SHA256 = "50391333ebd2262acbd02c9a50cbb8d6f8e078d8371dc24eb9559c364edeaeb1"
LTX_BYTES = 19940
ZIM_SHA256 = "9a894f59b81b712a6f4aa87a8f4cf8f03ce95937525b4bd84527fc43ed2fe21e"
ZIM_BYTES = 6735


def _codes(text, rules, media="image", max_characters=None):
    return [code for code, _ in pc.content_hits(text, media, frozenset(), max_characters, rules)]


# (a) стопка: флаг при forbidden, без флага при allowed; small_objects разрешены, чтобы видеть только это правило
def test_stack_of_sheets_is_flagged_when_object_sets_forbidden():
    rules = RulesValues(small_objects_allowed=True, object_sets_forbidden=True)
    assert pc.PROBLEM_OBJECT_SET in _codes("a stack of blank sheets on a desk", rules)


def test_stack_of_sheets_passes_when_object_sets_allowed():
    rules = RulesValues(small_objects_allowed=True, object_sets_forbidden=False)
    assert pc.PROBLEM_OBJECT_SET not in _codes("a stack of blank sheets on a desk", rules)


def test_stack_of_sheets_is_reported_through_find_problems():
    rules = RulesValues(small_objects_allowed=True, object_sets_forbidden=True)
    problems = pc.find_problems("a stack of blank sheets on a desk", "image", frozenset(), None, rules)
    assert pc.PROBLEM_OBJECT_SET in [pc.problem_code(p) for p in problems]


# (b) «rows of»: правило людей срабатывает при любом значении; новое правило этот фрагмент не дублирует
@pytest.mark.parametrize("object_sets_forbidden", [False, True])
def test_rows_of_shelves_is_flagged_by_people_rule_with_either_value(object_sets_forbidden):
    rules = RulesValues(object_sets_forbidden=object_sets_forbidden)
    codes = _codes("rows of metal shelves", rules)
    assert pc.PROBLEM_PEOPLE_COUNT in codes
    assert pc.PROBLEM_OBJECT_SET not in codes


# (c) «two pens»: новое правило не срабатывает; обработка мелких предметов прежняя
def test_two_pens_are_not_flagged_by_object_set_rule():
    rules = RulesValues(object_sets_forbidden=True)
    codes = _codes("two pens on a desk", rules)
    assert pc.PROBLEM_OBJECT_SET not in codes
    assert pc.PROBLEM_SMALL_OBJECT in codes


# (d) системная инструкция: cartoon — строка ровно один раз; ltx25 и zimage_turbo — строки нет, байты как прежде
def test_cartoon_system_instruction_has_object_set_line_exactly_once():
    rules = gp.load_rules(NICHE_DIR / "prompts" / "rules.yaml")
    assert rules.object_sets_forbidden is True
    cfg = PromptConfig(style_file=str(NICHE_DIR / "prompts" / "style.md"), media="image", preset="none",
                       relaxed_rules=frozenset({"art_medium", "small_objects"}), rules=rules)
    assert gp.build_system_instruction(cfg).count(OBJECT_SET_LINE) == 1


@pytest.mark.parametrize("profile, media, sha, size", [
    (LTX_DIR, "video", LTX_SHA256, LTX_BYTES),
    (ZIM_DIR, "image", ZIM_SHA256, ZIM_BYTES),
])
def test_model_profiles_have_no_object_set_line_and_keep_stored_sha(profile, media, sha, size):
    style = profile / "prompts" / "style.md"
    rules = gp.load_rules(profile / "prompts" / "rules.yaml")
    for cfg in (PromptConfig(style_file=str(style), media=media, preset="none"),
                PromptConfig(style_file=str(style), media=media, preset="none", rules=rules)):
        text = gp.build_system_instruction(cfg)
        assert OBJECT_SET_LINE not in text
        data = text.encode("utf-8")
        assert len(data) == size
        assert hashlib.sha256(data).hexdigest() == sha


# (e) неизвестное значение падает при загрузке по-русски с именем ключа; allowed и forbidden разбираются верно
def test_unknown_object_sets_value_fails_at_load_naming_key(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("object_sets: sometimes\n", encoding="utf-8")
    with pytest.raises(GeminiInputError) as exc:
        gp.load_rules(rules_file)
    message = str(exc.value)
    assert "object_sets" in message
    assert "Ошибка входных данных" in message


@pytest.mark.parametrize("value, expected", [("allowed", False), ("forbidden", True)])
def test_object_sets_value_maps_to_object_sets_forbidden(tmp_path, value, expected):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(f"object_sets: {value}\n", encoding="utf-8")
    assert gp.load_rules(rules_file).object_sets_forbidden is expected
