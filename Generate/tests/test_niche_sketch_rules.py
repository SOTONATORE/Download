"""Блок 6B: правила эскиза для ниши cartoon_photos (rules.yaml).

Разрешённые типы кадра снимают только свои слова; запрещённые (флаг, паспорт, удостоверение) остаются;
фигурки, строки текста, несколько кружков и штрихи не считаются людьми при max_characters;
толстые контуры и векторный стиль ловятся при forbid_thick_vector; частичное разрешение бумаги
не снимает запрет на бумагу в системной инструкции.

Тесты написаны, не запускались (прогоняет пользователь на GitHub). Правила читаются из tmp_path/rules.yaml
через gp.load_rules; сеть не используется.
"""
import pytest

from Generate.core import frame_rules as fr
from Generate.core import gemini_prompts as gp
from Generate.core import prompt_check as pc

# набор разрешённых типов кадра из Nishes/cartoon_photos/prompts/rules.yaml, плюс lettering (тот же словарь)
CARTOON_ALLOWED = ("letter, map, diagram, chart, infographic, calendar, spreadsheet, newspaper, document, "
                   "paper, blank sheet, sheet of paper")
ALL_ALLOWED = CARTOON_ALLOWED + ", lettering"
PAPER_SEVEN = "paper, document, newspaper, letter, spreadsheet, blank sheet, sheet of paper"

ALLOWED_CASES = [
    ("lettering", "the word HELLO drawn as hand lettering on a wall"),
    ("map", "a simple map of the old town on the table"),
    ("diagram", "a diagram of the water cycle in thin lines"),
    ("chart", "a bar chart of yearly sales on a white background"),
    ("spreadsheet", "a spreadsheet on the desk"),
    ("newspaper", "an old newspaper lying on a bench"),
    ("document", "a document in a folder on a shelf"),
    ("blank sheet", "a blank sheet on a wooden desk"),
    ("sheet of paper", "a sheet of paper on the desk"),
]

FORBIDDEN_CASES = [
    ("flag", "a flag waving on a pole"),
    ("passport", "an open passport on the table"),
    ("id card", "an id card lying on the counter"),
]

FALSE_POSITIVE_CASES = [
    "Stick-figures sketched on a white card.",
    "lines of text across the page",
    "several circles drawn in ink",
    "many tick marks along a ruler",
]

THICK_CASES = [
    "a figure drawn with bold outlines",
    "a flat vector sketch of a house",
]


def _sketch_rules(tmp_path, *, allowed=ALL_ALLOWED, thick=True):
    # rules.yaml из строк вида key: value — тот же разбор, что у реальной ниши
    text = (f"max_characters: 4\n"
            f"animals_count: true\n"
            f"allowed_shot_items: {allowed}\n"
            f"forbid_thick_vector: {'true' if thick else 'false'}\n")
    path = tmp_path / "rules.yaml"
    path.write_text(text, encoding="utf-8")
    return gp.load_rules(path)


def _codes(prompt, rules):
    return [pc.problem_code(p) for p in pc.find_problems(prompt, media="image", max_characters=4, rules=rules)]


@pytest.mark.parametrize("item,prompt", ALLOWED_CASES)
def test_allowed_item_passes_find_problems_and_no_rephrase(tmp_path, item, prompt):
    rules = _sketch_rules(tmp_path)
    assert pc.find_problems(prompt, media="image", max_characters=4, rules=rules) == [], item
    assert fr.rephrase_note(prompt, rules.allowed_shot_items) == "", item


@pytest.mark.parametrize("item,prompt", FORBIDDEN_CASES)
def test_forbidden_item_is_still_flagged(tmp_path, item, prompt):
    rules = _sketch_rules(tmp_path)
    assert pc.PROBLEM_FORBIDDEN_SHOT_TYPE in _codes(prompt, rules), item
    assert fr.rephrase_note(prompt, rules.allowed_shot_items) != "", item


@pytest.mark.parametrize("text", FALSE_POSITIVE_CASES)
def test_figures_lines_circles_ticks_are_not_people_count(tmp_path, text):
    rules = _sketch_rules(tmp_path)
    assert pc.PROBLEM_PEOPLE_COUNT not in _codes(text, rules), text


@pytest.mark.parametrize("text", THICK_CASES)
def test_thick_vector_flagged_when_forbidden(tmp_path, text):
    rules = _sketch_rules(tmp_path, thick=True)
    assert pc.PROBLEM_THICK_VECTOR in _codes(text, rules), text


@pytest.mark.parametrize("text", THICK_CASES)
def test_same_inputs_without_forbid_thick_vector_have_no_thick_problem(tmp_path, text):
    rules = _sketch_rules(tmp_path, thick=False)
    assert pc.PROBLEM_THICK_VECTOR not in _codes(text, rules), text


def test_image_tag_is_not_a_thick_vector_problem_but_ref_tag_rule_matches(tmp_path):
    # <imageN> проверяет не find_problems, а правило меток без референсов (gemini_prompts._REF_TAG_RX)
    rules = _sketch_rules(tmp_path, thick=True)
    text = "the figure <image1> stands by the door"
    assert pc.PROBLEM_THICK_VECTOR not in _codes(text, rules)
    assert gp._REF_TAG_RX.search(text) is not None


def _system_text(tmp_path, rules):
    style = tmp_path / "style.md"
    style.write_text("# Style\nPlain scene.\n", encoding="utf-8")
    cfg = gp.PromptConfig(style_file=str(style), media="image", rules=rules)
    return gp.build_system_instruction(cfg)


def test_partial_paper_allowance_keeps_paper_ban_in_system_instruction(tmp_path):
    partial = _sketch_rules(tmp_path, allowed="letter, map, blank sheet")  # не все семь имён
    text = _system_text(tmp_path, partial)
    assert gp._PAPER_LINE in text
    assert gp._PAPER_LINE_PAPER_ALLOWED not in text


def test_full_paper_allowance_replaces_paper_ban_in_system_instruction(tmp_path):
    full = _sketch_rules(tmp_path, allowed=PAPER_SEVEN)
    text = _system_text(tmp_path, full)
    assert gp._PAPER_LINE_PAPER_ALLOWED in text
    assert gp._PAPER_LINE not in text
