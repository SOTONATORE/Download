"""Тесты шага 6: проверка готовых промптов на запрещённые типы кадров.

Сеть не используется: Gemini заменяется заглушкой transport, которая отдаёт заранее заданные ответы.
"""
import json

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import GeminiRuntimeError, PromptConfig
from Generate.core.prompt_check import (
    MAX_REWORK_ATTEMPTS,
    find_problems,
    rework_remark,
)
from Generate.core.srt_parser import Segment


def seg(num, text):
    return Segment(num=num, start_ms=0, end_ms=1000, text=text)


def cfg_for(tmp_path):
    style = tmp_path / "style.md"
    style.write_text("# Style guide\nWrite one paragraph.\n", encoding="utf-8")
    return PromptConfig(style_file=str(style), max_prompt_chars=300, batch_size=10)


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
    problems = ["запрещённый тип кадра: «map»", "незаполненный маркер: «[CHAR_1]»"]
    remark = rework_remark(problems)
    for p in problems:
        assert p in remark


def test_max_rework_attempts_is_two():
    assert MAX_REWORK_ATTEMPTS == 2


# ---------------------------------------------------------------- hook in gemini_prompts

def test_hook_stops_after_two_rework_calls(tmp_path):
    # первый ответ и оба переписывания содержат «map»: после второго переписывания прогон останавливается
    t = ScriptedTransport([
        answer(1, "A detailed map of the Balkans on a wooden table."),
        answer(1, "A detailed map of the coast."),
        answer(1, "A map on a table."),
    ])
    with pytest.raises(GeminiRuntimeError):
        gp.generate_prompts([seg(1, "The map was drawn.")], [1], cfg_for(tmp_path), t, "m")
    # один первичный вызов и ровно два вызова на переписывание
    assert len(t.calls) == 1 + MAX_REWORK_ATTEMPTS


def test_hook_succeeds_after_one_clean_rework(tmp_path):
    t = ScriptedTransport([
        answer(1, "A detailed map of the Balkans on a wooden table."),
        answer(1, CLEAN),
    ])
    res = gp.generate_prompts([seg(1, "The map was drawn.")], [1], cfg_for(tmp_path), t, "m")
    assert len(t.calls) == 2
    assert res[0].prompt == CLEAN
