"""Тесты четырёх исправлений валидации cartoon_photos по прогону.

Покрывает: ложный конфликт рта на многоперсонажной сцене (клип 641), окно пустой таблички на шаблоне
«header band» из style.md (клип 26), хвост «in <шрифт> lettering» в ветке 3 надписи (клипы 128, 234, 795),
исключение глагола «forms» (клип 401), пересмотр клипа 641 при возобновлении без вызова модели.
Сеть не используется. Промпт клипа 641 собран по фразам из отчёта прогона (полный текст в архив не входит).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

try:
    from Generate.core import gemini_orchestrator as go
    from Generate.core import gemini_prompts as gp
    from Generate.core import prompt_check as pc
    from Generate.core import prompt_pipeline as pp
    from Generate.core.srt_parser import Segment as SrtSegment
except ImportError:  # плоский запуск из папки Generate (как в остальных тестах)
    from core import gemini_orchestrator as go  # type: ignore
    from core import gemini_prompts as gp  # type: ignore
    from core import prompt_check as pc  # type: ignore
    from core import prompt_pipeline as pp  # type: ignore
    from core.srt_parser import Segment as SrtSegment  # type: ignore


# Клип 641: два персонажа; рот одного — closed-line smile, рот другого — worried wavy mouth
PROMPT_641 = (
    "Medium shot: the main character, a stick-thin figure, stands at the left edge of a wheat field and turns toward an adult figure "
    "in a dark coat. The main character has a pure-white round head and a closed-line smile. "
    "The adult has a worried wavy mouth and looks down at the soil."
)
SEG_641 = "Герой говорит с незнакомцем у поля."

# style.md, Carrier 2 (full-width header band), с реальной надписью
CARRIER_2 = ("Full-width pale cream top header band separated by a thin horizontal black line reading "
             "ЗЕМЛЯНАЯ ГРУША in bold black uppercase lettering.")

# Надписи в кадре: Carrier 1 / Carrier 3 с хвостом «in bold black uppercase lettering»
CARRIER_1 = ("Bold black uppercase lettering centered at the top of a warm pale cream backdrop reads МИР in bold "
             "black uppercase lettering.")
CARRIER_3 = "A clean black uppercase lettering reading КАРТОФЕЛЬ in bold black uppercase lettering."
CARRIER_3_GAS = "A clean black uppercase lettering reading ГАЗООБРАЗОВАНИЕ in bold black uppercase lettering."
CARRIER_3_TOPI = "A clean black uppercase lettering reading ТОПИНАМБУР in bold black uppercase lettering."


def _labels(text):
    return [pc._lettering_phrase(m) for m in pc._LETTERING_RX.finditer(text)]


def _codes(prompt):
    return [pc.problem_code(p) for p in pc.find_problems(prompt, "image", max_characters=4)]


# ---------------------------------------------------------------------------
# 1. _mouth_conflict
# ---------------------------------------------------------------------------

def test_mouth_conflict_clip_641_two_characters_is_not_conflict():
    assert gp._mouth_conflict(PROMPT_641) is False


def test_mouth_conflict_single_character_with_both_mouths_is_conflict():
    text = ("The main character has a round white head, a closed-line smile and a worried wavy mouth.")
    assert gp._mouth_conflict(text) is True


def test_mouth_conflict_single_character_one_clause_both_mouths():
    text = "A single character with a closed-line smile and an open O-shaped mouth stands in the field."
    assert gp._mouth_conflict(text) is True


def test_mouth_above_head_still_conflict():
    assert gp._mouth_conflict("The character's mouth sits above the head.") is True


def test_mouth_conflict_two_characters_same_clause_is_conflict():
    text = "The main character and the adult figure stand side by side; the adult has a closed-line smile and a worried wavy mouth."
    assert gp._mouth_conflict(text) is True


# ---------------------------------------------------------------------------
# 2. _blank_card
# ---------------------------------------------------------------------------

def test_blank_card_header_band_carrier_2_is_not_blank():
    assert gp._blank_card(CARRIER_2) is False


def test_blank_card_real_blank_label_card_is_blank():
    assert gp._blank_card("A blank white label card on the left.") is True


def test_blank_card_real_empty_plaque_is_blank():
    assert gp._blank_card("A muted red title plaque with nothing written on it sits by the tuber.") is True


def test_blank_card_reading_in_next_sentence_does_not_count():
    # «reading» за границей предложения не относится к табличке
    assert gp._blank_card("A blank white label card on the left. The text reads nothing here.") is True


# ---------------------------------------------------------------------------
# 3. _LETTERING_RX: хвост «in <шрифт> lettering» не попадает в надпись
# ---------------------------------------------------------------------------

def test_lettering_carrier_1_extracts_only_label():
    assert _labels(CARRIER_1) == ["МИР"]


def test_lettering_carrier_3_extracts_only_label():
    assert _labels(CARRIER_3) == ["КАРТОФЕЛЬ"]
    assert _labels(CARRIER_3_GAS) == ["ГАЗООБРАЗОВАНИЕ"]
    assert _labels(CARRIER_3_TOPI) == ["ТОПИНАМБУР"]


def test_lettering_carrier_2_extracts_only_label():
    assert _labels(CARRIER_2) == ["ЗЕМЛЯНАЯ ГРУША"]


def test_segment_lettering_mismatch_clean_for_matching_segments():
    assert pc.segment_lettering_mismatch(CARRIER_3, "На столе лежит картофель.") == []
    assert pc.segment_lettering_mismatch(CARRIER_3_GAS, "Это газообразование в почве.") == []
    assert pc.segment_lettering_mismatch(CARRIER_3_TOPI, "Топинамбур лежит рядом.") == []
    assert pc.segment_lettering_mismatch(CARRIER_1, "Слово мир написано наверху.") == []


# ---------------------------------------------------------------------------
# 4. «forms» на месте глагола
# ---------------------------------------------------------------------------

def test_verb_forms_with_preposition_is_allowed():
    assert _codes("a small crack forms on the surface") == []


def test_noun_forms_still_flagged():
    assert "forbidden_shot_type" in _codes("a stack of forms on the desk")
    assert "forbidden_shot_type" in _codes("stack of tax forms")


def test_verb_form_with_article_still_handled_as_before():
    # одиночное «form» с «a» не исключалось и раньше: поведение не меняется
    assert "forbidden_shot_type" in _codes("tubers form a heap on the table")


# ---------------------------------------------------------------------------
# 5. пересмотр клипа 641 при возобновлении без вызова модели
# ---------------------------------------------------------------------------

class _NoCallTransport:
    def __init__(self):
        self.calls = 0

    def generate_json(self, *a, **kw):
        self.calls += 1
        raise AssertionError("Gemini не должен вызываться при пересмотре клипов «на проверку»")


def _timings(monkeypatch):
    def fake(segments, fps, frame_rule, clip_min, clip_max, gpu_max, tail):
        return [SimpleNamespace(num=s.num, start_ms=s.start_ms, end_ms=s.end_ms,
                                clip_ms=s.end_ms - s.start_ms, num_frames=49) for s in segments]
    monkeypatch.setattr(go, "compute_timings", fake)


def _timing():
    return go.TimingParams(fps=25, frame_rule={}, clip_min_sec=1.0, clip_max_sec=10.0,
                           gpu_max_sec=10.0, tail_pad_sec=0.0)


def _photo_cfg():
    # batch_passage: как у профиля cartoon_photos — пакетные правила стиля (рот, табличка, надпись) включены
    return gp.PromptConfig(style_file="unused.md", media="image", max_characters=4, batch_passage=True)


def _entry_641():
    return {
        "num": 641, "prompt": PROMPT_641 + " Soft light is stable.", "status": "ready", "attempts": 6,
        "error": "персонаж: рот противоречит себе или вынесен над головой", "needs_review": True, "extra": {},
        "rework_pending": [gp._PARSE_PREFIX + "персонаж: рот противоречит себе или вынесен над головой"],
        "rework_round": 5, "prompt_raw": PROMPT_641,
    }


def test_resume_clip_641_clears_review_without_llm(monkeypatch, tmp_path):
    _timings(monkeypatch)
    segs = [SrtSegment(num=641, start_ms=0, end_ms=10_000, text=SEG_641)]
    path = tmp_path / "prompts.json"
    state = {"srt_hash": go.srt_hash(segs), "model": "", "profile": "cartoon_photos",
             "prompts": {"641": _entry_641()}}
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    transport = _NoCallTransport()

    changed = go.recover_resolved_file(path, segs, _photo_cfg(), _photo_cfg())

    assert changed == [641]
    assert transport.calls == 0
    saved = json.loads(path.read_text(encoding="utf-8"))["prompts"]["641"]
    assert saved["needs_review"] is False
    assert saved["error"] == ""
    assert saved["rework_pending"] == []
    assert "prompt_raw" not in saved
