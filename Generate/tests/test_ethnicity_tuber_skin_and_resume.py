"""Офлайн-тесты: «тёмная кожа» у растения не признак внешности; фрагмент в подсказке повтора; пересмотр
клипов «на проверку» при возобновлении из prompts.json.

Без сети и без реального Gemini: транспорт считает вызовы и падает, если его вызвали. Файлы пишутся только
в tmp_path. Тайминг подменяется (в тестах нет настоящего видео).
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
except ImportError:  # запуск из каталога Generate
    from core import gemini_orchestrator as go  # type: ignore
    from core import gemini_prompts as gp  # type: ignore
    from core import prompt_check as pc  # type: ignore
    from core import prompt_pipeline as pp  # type: ignore
    from core.srt_parser import Segment as SrtSegment  # type: ignore


ETH_RU = pc.CONTENT_REASON_RU[pc.PROBLEM_ETHNICITY]

# Текст сегмента 587 без корней этнических признаков (ни «тёмн», ни «dark» и т. п.)
SEG_TUBER = "Топинамбур лежит на разделочной доске, рядом половинка с белой мякотью."
# Промпт клипа 587 по фразе из отчёта прогона (полный текст промпта в архив не входит): клубень, а не человек
PROMPT_587 = ("A wooden board with one Jerusalem artichoke tuber on it, one halved tuber beside it "
              "showing crisp white flesh and thin dark skin.")
TUBER_CASES = [
    PROMPT_587,
    "A potato with dark skin lies on a sack.",
    "A sunchoke root with dark skin sits in the soil.",
    "Topinambur fruit with dark skin is cut in half.",
    "A vegetable with thin dark skin rests on a plate.",
]
PERSON_CASES = [
    "A man with dark skin walks along the river.",
    "A woman with dark skin smiles at the camera.",
    "A dark-skinned woman walks along the road.",
    "A person with dark skin tone stands by the gate.",
]


# ---------------------------------------------------------------------------
# 1–2. prompt_check: ложное срабатывание и человеческие формы
# ---------------------------------------------------------------------------

def test_tuber_clip_587_prompt_is_not_ethnicity():
    assert pc.ethnicity_hits(PROMPT_587, SEG_TUBER) == []
    assert pc.ethnicity_reasons(PROMPT_587, SEG_TUBER) == []


def test_botanical_dark_skin_is_not_ethnicity_for_any_plant_word():
    for prompt in TUBER_CASES:
        assert pc.ethnicity_hits(prompt, SEG_TUBER) == [], prompt
        assert pc.ethnicity_reasons(prompt, SEG_TUBER) == [], prompt


def test_human_dark_skin_still_triggers_without_root_in_segment():
    for prompt in PERSON_CASES:
        hits = pc.ethnicity_hits(prompt, SEG_TUBER)
        assert hits, prompt
        assert pc.ethnicity_reasons(prompt, SEG_TUBER) == [ETH_RU], prompt


def test_human_dark_skin_passes_when_root_in_segment():
    seg = "Тёмный мужчина идёт по берегу реки."   # корень «темн» есть в тексте сегмента
    for prompt in PERSON_CASES:
        assert pc.ethnicity_hits(prompt, seg) == [], prompt
        assert pc.ethnicity_reasons(prompt, seg) == [], prompt


def test_hits_return_exact_matched_fragment():
    assert pc.ethnicity_hits("A man with dark skin walks.", "Человек идёт.") == ["dark skin"]
    assert pc.ethnicity_hits("A dark-skinned woman walks.", "Женщина идёт.") == ["dark-skinned"]


def test_other_ethnic_patterns_unchanged():
    assert pc.ethnicity_hits("An African man walks across the savanna.", "Он идёт.") == ["African man"]
    assert pc.ethnicity_hits("An East African grassland with acacia trees.", "Саванна.") == []


def test_ethnicity_reasons_keeps_old_shape():
    assert pc.ethnicity_reasons("An African man walks.", "Он идёт.") == [ETH_RU]
    assert pc.ethnicity_reasons("A plain street.", "Улица.") == []


# ---------------------------------------------------------------------------
# 3. подсказка повтора содержит фрагмент и указание на него
# ---------------------------------------------------------------------------

def test_rework_text_names_matched_fragment_and_hint():
    problem = pc.make_problem(pc.PROBLEM_ETHNICITY, "man with dark skin")
    text = gp._rework_text([problem])
    assert "man with dark skin" in text
    assert pc.CONTENT_HINT_EN[pc.PROBLEM_ETHNICITY] in text
    assert "exact phrase" in text


def test_parse_and_validate_ethnicity_goes_to_rework_with_fragment():
    """Разбор ответа видит «dark skin» у человека; очередь получает нарушение с фрагментом и текст повтора его называет."""
    cfg = gp.PromptConfig(style_file="unused.md", media="video")
    raw = json.dumps([{"segment_index": 587, "prompt": "A man with dark skin walks along the river."}])
    texts = {586: "Ферма на закате.", 587: SEG_TUBER, 588: "Дети бегут к реке."}
    results, _ = gp.parse_and_validate(raw, [587], cfg, texts)
    assert results[587].needs_review is True
    eth = gp.ethnicity_problems(results[587].prompt, 587, texts)
    assert eth == [pc.make_problem(pc.PROBLEM_ETHNICITY, "dark skin")]
    text = gp._rework_text(eth)
    assert "«dark skin»" in text or "dark skin" in text


def test_tuber_prompt_gives_no_ethnicity_problem_in_parse():
    cfg = gp.PromptConfig(style_file="unused.md", media="video")
    raw = json.dumps([{"segment_index": 587, "prompt": PROMPT_587}])
    texts = {586: "Ферма на закате.", 587: SEG_TUBER, 588: "Дети бегут к реке."}
    results, _ = gp.parse_and_validate(raw, [587], cfg, texts)
    assert ETH_RU not in (results[587].review_reason or "")
    assert gp.ethnicity_problems(PROMPT_587, 587, texts) == []


# ---------------------------------------------------------------------------
# 4–5. возобновление из prompts.json: пересмотр без вызова модели
# ---------------------------------------------------------------------------

class _NoCallTransport:
    """Транспорт, который не должен вызываться: любой запрос — ошибка теста."""

    def __init__(self):
        self.calls = 0

    def generate_json(self, *a, **kw):
        self.calls += 1
        raise AssertionError("Gemini не должен вызываться при пересмотре клипов «на проверку»")


def _segments():
    return [
        SrtSegment(num=586, start_ms=0, end_ms=10_000, text="Ферма на закате."),
        SrtSegment(num=587, start_ms=10_000, end_ms=20_000, text=SEG_TUBER),
        SrtSegment(num=588, start_ms=20_000, end_ms=30_000, text="Дети бегут к реке."),
    ]


def _entry_587(prompt_raw=PROMPT_587, round_no=5, error="внешность или происхождение не из текста сегмента"):
    return {
        "num": 587, "start_ms": 10_000, "end_ms": 20_000, "clip_ms": 10_000, "num_frames": 49,
        "prompt": prompt_raw + " Cinematic lighting is stable.", "status": "ready", "attempts": 6,
        "error": error, "needs_review": True, "extra": {},
        "rework_pending": [gp._PARSE_PREFIX + ETH_RU], "rework_round": round_no, "prompt_raw": prompt_raw,
    }


def _state_file(path, segments, prompts):
    state = {"srt_hash": go.srt_hash(segments), "model": "", "profile": "ltx25",
             "prompts": {str(k): v for k, v in prompts.items()}}
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return state


def _cfg():
    return gp.PromptConfig(style_file="unused.md", media="video")


def _timings(monkeypatch):
    def fake(segments, fps, frame_rule, clip_min, clip_max, gpu_max, tail):
        return [SimpleNamespace(num=s.num, start_ms=s.start_ms, end_ms=s.end_ms,
                                clip_ms=s.end_ms - s.start_ms, num_frames=49) for s in segments]
    monkeypatch.setattr(go, "compute_timings", fake)


def _timing():
    return go.TimingParams(fps=25, frame_rule={}, clip_min_sec=1.0, clip_max_sec=10.0,
                           gpu_max_sec=10.0, tail_pad_sec=0.0)


def test_resume_build_prompts_clears_resolved_tuber_clip(monkeypatch, tmp_path):
    _timings(monkeypatch)
    segs = _segments()
    path = tmp_path / "prompts.json"
    _state_file(path, segs, {587: _entry_587()})
    transport = _NoCallTransport()

    run = pp.build_prompts(
        [587], segments=segs, photo_nums=set(), cfg=_cfg(), profile="ltx25", timing=_timing(),
        photo_profile=None, photo_cfg=None, transport=transport, gemini_options={"models": ("m",)},
        prompts_path=path)

    assert transport.calls == 0
    saved = json.loads(path.read_text(encoding="utf-8"))["prompts"]["587"]
    assert saved["needs_review"] is False
    assert saved["error"] == ""
    assert saved["rework_pending"] == []
    assert "prompt_raw" not in saved
    assert saved["status"] == "ready"
    assert run.ready == [587]


def test_resume_generate_and_store_clears_resolved_tuber_clip(monkeypatch, tmp_path):
    _timings(monkeypatch)
    segs = _segments()
    path = tmp_path / "prompts.json"
    _state_file(path, segs, {587: _entry_587()})
    transport = _NoCallTransport()

    state = go.generate_and_store(
        segs, [587], _cfg(), transport, path, profile="ltx25", timing=_timing(),
        models=("m",))

    assert transport.calls == 0
    entry = state["prompts"]["587"]
    assert entry["needs_review"] is False and entry["error"] == "" and entry["rework_pending"] == []
    assert "prompt_raw" not in entry
    on_disk = json.loads(path.read_text(encoding="utf-8"))["prompts"]["587"]
    assert on_disk["needs_review"] is False


def test_resume_keeps_genuinely_invalid_clip_on_review(monkeypatch, tmp_path):
    _timings(monkeypatch)
    segs = _segments()
    path = tmp_path / "prompts.json"
    bad = _entry_587(prompt_raw="A man with dark skin walks along the riverbank.",
                     error=ETH_RU)
    bad["prompt"] = "A man with dark skin walks along the riverbank. Cinematic lighting is stable."
    bad["rework_pending"] = [gp._PARSE_PREFIX + ETH_RU]
    _state_file(path, segs, {587: bad})
    transport = _NoCallTransport()

    pp.build_prompts(
        [587], segments=segs, photo_nums=set(), cfg=_cfg(), profile="ltx25", timing=_timing(),
        photo_profile=None, photo_cfg=None, transport=transport, gemini_options={"models": ("m",)},
        prompts_path=path)

    assert transport.calls == 0
    saved = json.loads(path.read_text(encoding="utf-8"))["prompts"]["587"]
    assert saved["needs_review"] is True
    assert saved["rework_round"] == 5
    assert saved["error"] == ETH_RU


def test_recheck_returns_queue_for_invalid_clip_with_rounds_left():
    """Нарушение по текущим правилам и круги не кончились — клип уходит в очередь с фрагментом, без модели."""
    segs = _segments()
    texts = {s.num: s.text for s in segs}
    prompts = {"587": _entry_587(prompt_raw="A man with dark skin walks along the riverbank.",
                                 round_no=2, error=ETH_RU)}
    prompts["587"]["rework_pending"] = []
    changed = go.recheck_resolved(prompts, texts, _cfg())
    assert changed == [587]
    e = prompts["587"]
    assert e["needs_review"] is False
    assert pc.make_problem(pc.PROBLEM_ETHNICITY, "dark skin") in e["rework_pending"]
    assert e["rework_round"] == 2


def test_recheck_skips_duplicate_queue_entries():
    segs = _segments()
    texts = {s.num: s.text for s in segs}
    e = _entry_587()
    e["rework_pending"] = [pc.make_problem(pc.PROBLEM_DUPLICATE_PROMPT, "x")]
    prompts = {"587": e}
    assert go.recheck_resolved(prompts, texts, _cfg()) == []
    assert prompts["587"]["needs_review"] is True
