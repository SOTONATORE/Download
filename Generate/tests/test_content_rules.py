"""Правила содержания кадра и русские запрещённые типы кадров (после разбора прогона).

Проверяют: русский текст сегмента ловит запрещённые типы; промпт с детьми, наготой, толпой,
мелкими предметами, речью, руками у огня, битвами, оружием, крупными планами или чужой внешностью
даёт нарушение; чистый промпт нарушений не даёт; системная инструкция содержит правила.
Не запускаются в сети и не требуют ключей.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Generate.core import frame_rules as fr  # noqa: E402
from Generate.core import prompt_check as pc  # noqa: E402
from Generate.core import gemini_prompts as gp  # noqa: E402

STYLE_DIR = _HERE.parents[1] / "model_profiles" / "models"


# ---------------------------------------------------------------------------
# Русские запрещённые типы кадров (текст сегмента)
# ---------------------------------------------------------------------------

RU_POSITIVE = [
    "На карте империи висит флаг.",
    "Письмо лежит на столе.",
    "Паспорт в руках старика.",
    "Чистый лист бумаги на столе.",
    "Документы сложены в ящик.",
]
RU_NEGATIVE = [
    "Документальный фильм о степи.",   # документальный: не документ
    "Письменный приказ висит на стене.",  # письменный: не письмо
    "Картина висит в зале.",            # картина: не карта
    "Русская форма на солдатах.",       # форма (мундир): не запрещена
    "Солдаты скачут по степи.",
]


@pytest.mark.parametrize("text", RU_POSITIVE)
def test_russian_forbidden_frame_detected(text):
    assert fr.needs_rephrase(text) is True
    assert fr.rephrase_note(text) == fr.REPHRASE_RULE_EN


@pytest.mark.parametrize("text", RU_NEGATIVE)
def test_russian_text_without_forbidden_frame_not_flagged(text):
    assert fr.needs_rephrase(text) is False


# ---------------------------------------------------------------------------
# Содержание кадра: нарушения
# ---------------------------------------------------------------------------

def codes(prompt, media="video"):
    return {pc.problem_code(p) for p in pc.find_problems(prompt, media)}


@pytest.mark.parametrize("prompt, code", [
    ("A young boy rides a brown horse across the steppe.", pc.PROBLEM_CHILD),
    ("A baby sleeps under a wool blanket.", pc.PROBLEM_CHILD),
    ("A naked man stands by the river.", pc.PROBLEM_NUDITY),
    ("A shirtless rider leads the horse.", pc.PROBLEM_NUDITY),
    ("A crowd of villagers gathers in the square.", pc.PROBLEM_PEOPLE_COUNT),
    ("Horsemen gather on a ridge at dusk.", pc.PROBLEM_PEOPLE_COUNT),
    ("Three horses graze near the well.", pc.PROBLEM_MANY_OF_ONE),
    ("A woman stitches thick felt with a large needle.", pc.PROBLEM_SMALL_OBJECT),
    ("A man slides the beads of an abacus with his fingers.", pc.PROBLEM_SMALL_OBJECT),
    ("She tucks a bundle beneath a pillow.", pc.PROBLEM_SMALL_OBJECT),
    ("The old man says something and laughs.", pc.PROBLEM_SPEECH),
    ("A woman answers while the child watches.", pc.PROBLEM_SPEECH),
    ("He holds his hands close to the flames.", pc.PROBLEM_FIRE_HANDS),
    ("Soldiers warm their hands over a campfire.", pc.PROBLEM_FIRE_HANDS),
    ("Horsemen charge across the steppe as gunfire rings out.", pc.PROBLEM_CHAOS),
    ("A cloud of dust and riders merge into the smoke.", pc.PROBLEM_CHAOS),
    ("A soldier holds a rifle at his side.", pc.PROBLEM_WEAPON),
    ("He draws a sabre from its sheath.", pc.PROBLEM_WEAPON),
])
def test_content_violation_detected(prompt, code):
    assert code in codes(prompt)


def test_close_up_only_for_images():
    prompt = "A close-up portrait of an old man's face in sharp focus."
    assert pc.PROBLEM_CLOSE_FACE in codes(prompt, "image")
    assert pc.PROBLEM_CLOSE_FACE not in codes(prompt, "video")


def test_people_limit_video_two_people():
    prompt = "A man and a woman stand by the cart while a dog waits."
    assert pc.PROBLEM_PEOPLE_COUNT not in codes(prompt, "video")
    three = "A man, a woman and a soldier stand by the cart."
    assert pc.PROBLEM_PEOPLE_COUNT in codes(three, "video")


def test_people_limit_image_one_person():
    prompt = "A man and a woman stand by the wall."
    assert pc.PROBLEM_PEOPLE_COUNT in codes(prompt, "image")


# ---------------------------------------------------------------------------
# Чистые промпты: нарушений нет
# ---------------------------------------------------------------------------

CLEAN = [
    "A wide shot of the steppe at dusk, dry grass bending under a pale sky, the camera drifts slowly "
    "to the right and ends on a distant hill.",
    "A medium shot of a woman in a grey wool coat standing by a yurt, warm lamplight on the felt walls.",
    "A static wide frame of a horse standing alone on a bare hill under a white sky.",
]


@pytest.mark.parametrize("prompt", CLEAN)
def test_clean_prompt_has_no_content_problems(prompt):
    assert pc.content_hits(prompt, "video") == []


# ---------------------------------------------------------------------------
# Внешность не из текста сегмента
# ---------------------------------------------------------------------------

def test_ethnicity_not_in_text_is_flagged():
    prompt = "A dark-skinned man walks across the steppe."
    seg = "Тажигали шёл по степи и думал о сыне."
    assert gp.ethnicity_reasons(prompt, seg) == [pc.CONTENT_REASON_RU[pc.PROBLEM_ETHNICITY]]


def test_ethnicity_allowed_when_text_names_it():
    prompt = "An Arab merchant walks across the market."
    seg = "Араб-торговец шёл по базару."
    assert gp.ethnicity_reasons(prompt, seg) == []


def test_no_ethnic_word_no_reason():
    prompt = "A man in a grey coat walks across the steppe."
    assert gp.ethnicity_reasons(prompt, "Тажигали шёл по степи.") == []


# ---------------------------------------------------------------------------
# Валидатор ответа Gemini: нарушение содержания даёт needs_review с причиной
# ---------------------------------------------------------------------------

def _cfg(media):
    style = STYLE_DIR / ("ltx25" if media == "video" else "zimage_turbo") / "prompts" / "style.md"
    return gp.PromptConfig(style_file=str(style), media=media)


def test_parse_and_validate_flags_child_and_keeps_clean():
    raw = json.dumps([
        {"segment_index": 1, "prompt": "A young boy rides a horse across the steppe."},
        {"segment_index": 2, "prompt": "A medium shot of a woman standing by a yurt at dusk."},
    ])
    results, problems = gp.parse_and_validate(raw, [1, 2], _cfg("video"), {1: "", 2: ""})
    assert results[1].needs_review is True
    assert pc.CONTENT_REASON_RU[pc.PROBLEM_CHILD] in results[1].review_reason
    assert results[2].needs_review is False
    assert any(p.num == 1 for p in problems)


def test_parse_and_validate_flags_ethnicity_only_when_texts_given():
    raw = json.dumps([{"segment_index": 7, "prompt": "A dark-skinned man walks across the steppe."}])
    no_texts, _ = gp.parse_and_validate(raw, [7], _cfg("video"))
    assert no_texts[7].needs_review is False
    with_texts, _ = gp.parse_and_validate(raw, [7], _cfg("video"), {7: "Тажигали шёл по степи."})
    assert with_texts[7].needs_review is True
    assert pc.CONTENT_REASON_RU[pc.PROBLEM_ETHNICITY] in with_texts[7].review_reason


def test_retry_hints_exist_for_every_content_reason():
    for reason in pc.CONTENT_REASON_RU.values():
        assert reason in gp._STYLE_HINTS


# ---------------------------------------------------------------------------
# Системная инструкция и стили: правила попали в текст для модели
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("media", ["video", "image"])
def test_system_instruction_contains_content_rules(media):
    system = gp.build_system_instruction(_cfg(media))
    assert "Never show children" in system
    assert "Never show anyone naked" in system
    assert "No speech of any kind" in system
    assert "No weapons in the frame" in system
    assert ("Never show close-ups" in system) == (media == "image")


def test_photo_style_has_no_fixed_ottoman_setting():
    text = (STYLE_DIR / "zimage_turbo" / "prompts" / "style.md").read_text(encoding="utf-8")
    assert "Ottoman" not in text
    assert "Kazakh steppe" in text


def test_repeat_mentions_of_one_person_are_not_counted():
    # один человек, упомянутый повторно: не нарушение даже для фото (один человек)
    prompt = ("One man in a military tunic stands on stone steps. The man holds a cap. "
              "The standing man is lit from above. Static figure.")
    assert pc.PROBLEM_PEOPLE_COUNT not in codes(prompt, "image")


def test_two_introduced_people_still_counted_for_image():
    prompt = "One man stands by the wall and a woman sits near the door."
    assert pc.PROBLEM_PEOPLE_COUNT in codes(prompt, "image")
