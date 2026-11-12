"""Тесты шага 6: проверка готовых промптов на запрещённые типы кадров.

Сеть не используется: Gemini заменяется заглушкой transport, которая отдаёт заранее заданные ответы.
"""
import json

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import PromptConfig
from Generate.core.prompt_check import (
    MAX_REWORK_ATTEMPTS,
    content_hits,
    find_camera_motion,
    PROBLEM_FORBIDDEN_FORMAT,
    PROBLEM_FORBIDDEN_SHOT_TYPE,
    PROBLEM_LEFTOVER_MARKER,
    find_problems,
    make_problem,
    rework_remark,
)
from Generate.core.srt_parser import Segment


def seg(num, text):
    return Segment(num=num, start_ms=0, end_ms=1000, text=text)


def cfg_for(tmp_path):
    style = tmp_path / "style.md"
    style.write_text("# Style guide\nWrite one paragraph.\n", encoding="utf-8")
    return PromptConfig(media="video", style_file=str(style), max_prompt_chars=300, batch_size=10)


class ScriptedTransport:
    """Заглушка Gemini: отдаёт ответы по очереди и запоминает вызовы."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_json(self, model, system, user, schema):
        self.calls.append(user)
        return json.dumps(self.replies.pop(0))


def answer(num, text):
    return [{"segment_index": num, "prompt": text}]


# ---------------------------------------------------------------- find_problems

CLEAN = "A calm street in Istanbul at dusk, warm light, cobblestones."


def test_clean_prompt_has_no_problems():
    assert find_problems(CLEAN) == []


def test_map_is_forbidden():
    assert find_problems("A detailed map of the Balkans on a wooden table.")


def test_newspaper_is_forbidden():
    assert find_problems("A newspaper lying on a chair.")


def test_paper_is_forbidden():
    problems = find_problems("Stacked papers lie on the desk of the room.")
    assert len(problems) == 1
    assert "paper" in problems[0].lower()


def test_letter_is_forbidden():
    problems = find_problems("A letter lies open on the table of the room.")
    assert len(problems) == 1
    assert "letter" in problems[0].lower()


def test_parchment_is_forbidden():
    problems = find_problems("Old parchment hangs on the wall of the hall.")
    assert len(problems) == 1
    assert "parchment" in problems[0].lower()


def test_bed_sheet_is_not_forbidden():
    assert find_problems("A bed sheet lies on the floor of the room.") == []


def test_bread_is_not_forbidden():
    assert find_problems("The bread is baked every morning.") == []


def test_flag_is_forbidden():
    assert find_problems("Two soldiers hold a flag above the gate.")


def test_leftover_marker_is_a_problem():
    assert find_problems("[CHAR_1] stands by the fountain.")


def test_formal_is_not_form():
    assert find_problems("A formal dinner in a palace hall.") == []


def test_format_is_not_form():
    assert find_problems("A format of the old photograph is visible on the wall.") == []


# ---------------------------------------------------------------- rework_remark

def test_rework_remark_contains_each_problem():
    problems = [make_problem(PROBLEM_FORBIDDEN_SHOT_TYPE, "map"), make_problem(PROBLEM_LEFTOVER_MARKER, "[CHAR_1]")]
    remark = rework_remark(problems)
    assert "forbidden shot type: «map»" in remark
    assert "unfilled marker: «[CHAR_1]»" in remark
    assert "запрещённый" not in remark  # модель получает только английский текст


def test_max_rework_attempts_is_two():
    assert MAX_REWORK_ATTEMPTS == 2


# ---------------------------------------------------------------- hook in gemini_prompts

def test_hook_keeps_clip_after_repair_rounds(tmp_path):
    # первый ответ и все REPAIR_ROUNDS исправлений содержат «map»: прогон не останавливается,
    # клип сохраняется с текстом последней версии и помечается на проверку
    t = ScriptedTransport([
        answer(1, "A detailed map of the Balkans on a wooden table."),
        answer(1, "A detailed map of the coast."),
        answer(1, "A map on a table."),
        answer(1, "A map of the old town."),
        answer(1, "A map of a harbour."),
        answer(1, "A map of the bazaar."),
    ])
    results = gp.generate_prompts([seg(1, "The map was drawn.")], [1], cfg_for(tmp_path), t, "m")
    res = next(r for r in results if r.num == 1)
    assert res.needs_review is True
    assert res.prompt.strip()
    # один первичный вызов и по одному вызову на каждый круг исправления
    assert len(t.calls) == 1 + gp.REPAIR_ROUNDS
    assert res.review_reason


def test_hook_succeeds_after_one_clean_rework(tmp_path):
    t = ScriptedTransport([
        answer(1, "A detailed map of the Balkans on a wooden table."),
        answer(1, CLEAN),
    ])
    res = gp.generate_prompts([seg(1, "The map was drawn.")], [1], cfg_for(tmp_path), t, "m")
    assert len(t.calls) == 2
    assert res[0].prompt == CLEAN


# ---------------------------------------------------------------- ordinary sheet is allowed

def test_ordinary_sheet_is_allowed():
    # обычное sheet (простыня, лист льда) не входит в запрещённые типы кадров; запрещено только spreadsheet
    assert find_problems("A bed sheet lies on the floor of the room.") == []
    assert find_problems("A sheet of ice covers the lake.") == []


# ---------------------------------------------------------------- find_camera_motion (только для фото)

@pytest.mark.parametrize("text", [
    "The camera slowly zooms toward the bridge.",
    "The frame tracks along the wall.",
])
def test_camera_motion_found(text):
    assert find_camera_motion(text)


@pytest.mark.parametrize("text", [
    "The path follows the river under a grey sky.",
    "A rotating wooden wheel stands by the mill.",
    "The frame of the window is painted white.",
    "A static camera holds on the empty street.",
])
def test_camera_motion_not_found(text):
    assert find_camera_motion(text) == []


# ---------------------------------------------------------------- соотношения сторон и бумажные листы

@pytest.mark.parametrize("text", [
    "The room is shown in 16:9.",
    "A widescreen view of the hall.",
])
def test_format_terms_are_forbidden(text):
    problems = find_problems(text)
    assert len(problems) == 1
    assert problems[0].startswith(PROBLEM_FORBIDDEN_FORMAT)


def test_blank_sheet_is_forbidden():
    problems = find_problems("A blank sheet lies on the table.")
    assert len(problems) == 1
    assert "blank sheet" in problems[0]


def test_sheet_of_paper_is_forbidden():
    # фраза даёт одно сообщение; существующая основа paper даёт второе (сочетание не заменяет основу)
    problems = find_problems("Writing on a sheet of paper in the study.")
    assert len(problems) == 2
    assert any("sheet of paper" in p for p in problems)


def test_sheet_of_parchment_gives_phrase_and_stem_problems():
    # фраза «sheet of parchment» и основа parchment: два разных сообщения, повтор основы схлопывается
    problems = find_problems("Old parchment on a sheet of parchment.")
    assert len(problems) == 2
    assert any("sheet of parchment" in p for p in problems)
    assert any("parchment" in p and "sheet" not in p for p in problems)


def test_sheet_of_ice_is_not_forbidden():
    assert find_problems("A sheet of ice covers the lake.") == []


# ---------------------------------------------------------------- флаг после исправления и остановка пакета

def test_fixed_clip_is_not_flagged(tmp_path):
    # первичный ответ с нарушением (флаг от шага 6), исправление прошло: флаг снимается, текст без стиля не нужен
    t = ScriptedTransport([
        answer(1, "A detailed map of the Balkans on a wooden table."),
        answer(1, CLEAN),
    ])
    res = gp.generate_prompts([seg(1, "The map was drawn.")], [1], cfg_for(tmp_path), t, "m")
    assert res[0].prompt == CLEAN
    assert res[0].needs_review is False
    assert res[0].review_reason == ""


def test_people_count_ignores_repeated_mentions():
    # два человека, названных несколько раз, — это не четыре человека
    from Generate.core.prompt_check import count_people
    text = ("An older man with grey hair stands by the wall. Beside him, a middle-aged man stands facing him. "
            "The older man turns his head. The middle-aged man lowers his gaze.")
    assert count_people(text) == 2
    assert content_hits(text, "video") == []


def test_people_count_still_flags_three_people_and_crowds():
    from Generate.core.prompt_check import count_people
    assert count_people("An old man stands. A woman sits. A girl plays.") == 3
    assert count_people("A crowd of men gathers. Two riders ride past.") >= 3


def test_rework_counter_counts_every_clip_in_batch():
    # пакет исправления: строки «Prompt for segment_index N:» идут все, считаются все номера
    from Generate.cli import _ReworkCounter

    class Inner:
        def generate_json(self, model, system, user, schema):
            return "[]"

    counter = _ReworkCounter(Inner())
    user = "Prompt for segment_index 3: a map. x\nPrompt for segment_index 7: a sheet. y\nPrompt for segment_index 9: z"
    counter.generate_json("m", "s", user, {})
    assert counter.rework_nums == {3, 7, 9}
