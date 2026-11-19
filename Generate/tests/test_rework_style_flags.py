"""Очередь исправлений шага 6 для любых флагов разбора при rework_style_flag_without_refs (без сети).

Проверяются: допуск клипа в очередь при флаге и без него; слияние находок find_problems с причинами разбора в
_run_batch (раунд 0) и в _WaveJob.process_fix; подсказки для повтора (по-английски); текст субтитра для повтора;
сохранение очереди при пустом ответе исправления; обновление setting. Модель подменяется фейковым транспортом.
"""
import json
import threading
from types import SimpleNamespace

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import BatchRequest, PromptConfig, PromptResult
from Generate.core.prompt_check import (CONTENT_HINT_BY_REASON, CONTENT_REASON_RU, PROBLEM_CHILD,
                                        PROBLEM_DANGLING_TEXT, PROBLEM_ETHNICITY,
                                        PROBLEM_LETTERING_NOT_IN_SEGMENT, PROBLEM_REF_TAG,
                                        make_problem, problem_code)

KEYWORD = "стиль: keyword spam"
QUOTES = "стиль: кавычки"
CINEMA = "стиль: кино и камера"
ETHNIC = CONTENT_REASON_RU[PROBLEM_ETHNICITY]
LENGTH = "длина 1600 больше лимита 1500"
GARBAGE = "служебный мусор: тестовая метка"
PARSE = gp._PARSE_PREFIX
RENDERED = "A stone bridge over a river. Rendered in clean 2D editorial comic style."
ALL_FLAGS = [KEYWORD, QUOTES, CINEMA, ETHNIC, LENGTH, GARBAGE, "; ".join([KEYWORD, QUOTES, ETHNIC])]


def _cfg(**kw):
    return PromptConfig(style_file="style.md", media="image", **kw)


def _res(reason, needs_review=True):
    return PromptResult(1, "a scene", needs_review, reason)


class _FakeTransport:
    """Фейковый транспорт: всегда отдаёт один и тот же сырой ответ и считает запросы."""

    def __init__(self, raw):
        self.raw = raw
        self.calls = 0
        self.users = []

    def generate_json(self, model, system, user, schema):
        self.calls += 1
        self.users.append(user)
        return self.raw


def _run_batch(raw, cfg, texts=None, nums=(1,)):
    builder = SimpleNamespace(cfg=cfg, texts=texts or {})
    batch = BatchRequest(nums=tuple(nums), system="s", user="u", schema={})
    transport = _FakeTransport(raw)
    return gp._run_batch(transport, "m", builder, batch), transport


def _job(cfg, texts, enqueued):
    """Волна без пула: только поля, которые читает process_fix; очередь возврата собирается в enqueued."""
    job = gp._WaveJob.__new__(gp._WaveJob)
    job.builder = SimpleNamespace(cfg=cfg, system="s", schema={}, texts=texts)
    job.model = "m"
    job.style_text = "STYLE"
    job.tag = ""
    job.wave = "w"
    job.pkg_round = {}
    job.round_no = 1
    job.round_stats = {}
    job.round_left = {}
    job.stat_lock = threading.Lock()
    job.out_lock = threading.Lock()
    job.out = {}
    job.on_batch = None
    job.enqueue = lambda clips: enqueued.extend(clips)
    return job


class _Logged:
    def __init__(self, raw):
        self.raw = raw
        self.users = []

    def start_batch(self, n, unit="Батч"):
        pass

    def generate_json(self, model, system, user, schema):
        self.users.append(user)
        return self.raw


def _clip(num, prompt_raw, pending, extra=None):
    return PromptResult(num, "", True, "; ".join(pending), extra or {}, rework_pending=list(pending),
                        rework_round=0, prompt_raw=prompt_raw)


# --- допуск в очередь ---------------------------------------------------------

@pytest.mark.parametrize("reason", ALL_FLAGS)
def test_any_flag_queued_without_refs_when_flag_on(reason):
    assert gp._rework_gate_open(_cfg(rework_style_flag_without_refs=True), _res(reason)) is True


@pytest.mark.parametrize("reason", ALL_FLAGS)
def test_any_flag_not_queued_without_refs_when_flag_off(reason):
    assert gp._rework_gate_open(_cfg(rework_style_flag_without_refs=False), _res(reason)) is False
    assert gp._rework_gate_open(_cfg(), _res(reason)) is False


def test_refs_enabled_still_queues_without_flag():
    assert gp._rework_gate_open(_cfg(refs_enabled=True), _res(LENGTH)) is True


def test_gate_closed_for_fail_fast_no_review_and_empty_reason():
    flag = _cfg(rework_style_flag_without_refs=True)
    assert gp._rework_gate_open(_cfg(rework_style_flag_without_refs=True, fail_fast=True), _res(KEYWORD)) is False
    assert gp._rework_gate_open(flag, _res(KEYWORD, needs_review=False)) is False
    assert gp._rework_gate_open(flag, _res("")) is False


def test_round0_keyword_spam_queued_with_flag_on_refs_off():
    raw = json.dumps([{"segment_index": 1, "prompt": RENDERED}])
    final, transport = _run_batch(raw, _cfg(rework_style_flag_without_refs=True))
    assert final[1].needs_review is True
    assert PARSE + KEYWORD in final[1].rework_pending
    assert transport.calls == 1   # нарушение содержания или стиля не даёт повторного запроса


def test_round0_keyword_spam_not_queued_with_flag_off():
    raw = json.dumps([{"segment_index": 1, "prompt": RENDERED}])
    final, _ = _run_batch(raw, _cfg())
    assert final[1].needs_review is True
    assert final[1].rework_pending == []


# --- слияние находок и причин разбора -----------------------------------------

def test_merge_keeps_parse_reason_next_to_find_problems_hit_round0():
    # find_problems даёт тег <imageN> без референсов; стиль «keyword spam» не покрыт и должен попасть в очередь
    raw = json.dumps([{"segment_index": 1, "prompt": "A stone bridge <image1> over a river. Rendered in 4k."}])
    final, _ = _run_batch(raw, _cfg(rework_style_flag_without_refs=True))
    pending = final[1].rework_pending
    assert PROBLEM_REF_TAG in {problem_code(p) for p in pending}
    assert PARSE + KEYWORD in pending
    assert PARSE + gp.REF_TAG_REASON not in pending   # уже покрыто находкой тега


def test_merge_skips_content_reason_covered_by_same_code():
    found = [make_problem(PROBLEM_CHILD, "a child")]
    out = gp._merge_parse_reasons(found, [CONTENT_REASON_RU[PROBLEM_CHILD], KEYWORD])
    assert out == found + [PARSE + KEYWORD]


def test_merge_keeps_ethnicity_even_with_findings():
    found = [make_problem(PROBLEM_REF_TAG, "<image1>")]
    out = gp._merge_parse_reasons(found, [ETHNIC])
    assert out == found + [PARSE + ETHNIC]


def test_merge_skips_refs_reasons_when_refs_codes_found():
    found = [f"{gp.REF_CODE_INVALID}: «x»"]
    out = gp._merge_parse_reasons(found, [gp.REF_REASON_NOT_LIST, gp.LABEL_REASON, gp.REF_REASON_COMBO, KEYWORD])
    assert out == found + [PARSE + KEYWORD]


def test_merge_without_findings_dedupes_and_drops_empty():
    assert gp._merge_parse_reasons([], [KEYWORD, KEYWORD, "", QUOTES]) == [PARSE + KEYWORD, PARSE + QUOTES]


def test_process_fix_merges_find_problems_and_parse_reason():
    enqueued = []
    job = _job(_cfg(rework_style_flag_without_refs=True), {}, enqueued)
    clip = _clip(6, "old text", [PARSE + KEYWORD])
    logged = _Logged(json.dumps([{"segment_index": 6, "prompt": "A stone bridge <image1> over a river. Rendered in 4k."}]))
    job.process_fix(logged, [clip], 1)
    codes = {problem_code(p) for p in clip.rework_pending}
    assert PROBLEM_REF_TAG in codes
    assert PARSE + KEYWORD in clip.rework_pending
    assert clip in enqueued


# --- подсказки и текст субтитра -----------------------------------------------

def test_keyword_spam_hint_names_rendered_and_rendering():
    hint = gp._parse_hint_en(PARSE + KEYWORD)
    assert "rendered" in hint and "rendering" in hint


def test_parse_hints_for_length_garbage_and_ethnicity_are_english():
    length = gp._parse_hint_en(PARSE + LENGTH)
    assert "1500" in length and length.isascii()
    garbage = gp._parse_hint_en(PARSE + GARBAGE)
    assert "service text" in garbage and garbage.isascii()
    assert gp._parse_hint_en(PARSE + ETHNIC) == CONTENT_HINT_BY_REASON[ETHNIC]


def test_subtitle_note_for_lettering_mismatch_in_process_fix():
    enqueued = []
    texts = {5: "Hello   there"}
    job = _job(_cfg(rework_style_flag_without_refs=True), texts, enqueued)
    clip = _clip(5, "old", [make_problem(PROBLEM_LETTERING_NOT_IN_SEGMENT, "Hi")])
    logged = _Logged(json.dumps([{"segment_index": 5, "prompt": "A stone bridge over a river."}]))
    job.process_fix(logged, [clip], 1)
    assert "Subtitle text for segment_index 5: Hello there." in logged.users[0]


def test_subtitle_note_included_for_ethnicity_and_dangling_only():
    texts = {7: "Some line"}
    eth = _clip(7, "x", [PARSE + ETHNIC])
    dang = _clip(7, "x", [make_problem(PROBLEM_DANGLING_TEXT, "the")])
    style = _clip(7, "x", [PARSE + KEYWORD])
    assert "Subtitle text for segment_index 7: Some line." in gp._rework_subtitle_note(eth, texts)
    assert "Subtitle text for segment_index 7: Some line." in gp._rework_subtitle_note(dang, texts)
    assert gp._rework_subtitle_note(style, texts) == ""


def test_subtitle_note_plain_style_clip_has_no_subtitle_in_process_fix():
    job = _job(_cfg(rework_style_flag_without_refs=True), {9: "Line nine"}, [])
    clip = _clip(9, "old", [PARSE + KEYWORD])
    logged = _Logged(json.dumps([{"segment_index": 9, "prompt": "A stone bridge over a river."}]))
    job.process_fix(logged, [clip], 1)
    assert "Subtitle text" not in logged.users[0]


# --- пустой ответ исправления и setting ---------------------------------------

def test_empty_fix_response_preserves_previous_rework_pending():
    job = _job(_cfg(rework_style_flag_without_refs=True), {}, [])
    old = [PARSE + KEYWORD, make_problem(PROBLEM_REF_TAG, "<image1>")]
    clip = _clip(3, "previous raw text", old)
    job.process_fix(_Logged("[]"), [clip], 1)
    assert clip.rework_pending == old
    assert clip.prompt_raw == "previous raw text"
    assert not any("номер отсутствует" in p for p in clip.rework_pending)


def test_fix_response_updates_extra_setting():
    job = _job(_cfg(rework_style_flag_without_refs=True), {}, [])
    clip = _clip(8, "old", [PARSE + KEYWORD], extra={"setting": "old setting"})
    raw = json.dumps([{"segment_index": 8, "prompt": "A stone bridge over a river.", "setting": "river valley"}])
    job.process_fix(_Logged(raw), [clip], 1)
    assert clip.extra["setting"] == "river valley"
