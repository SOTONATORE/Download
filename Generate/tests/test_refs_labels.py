"""Метки <imageN> и refs (профиль с refs_enabled), режим on_unfixed. Тесты написаны, не запускались."""
import json

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import GeminiError, PromptConfig


def _cfg(**kw):
    kw.setdefault("style_file", "x")
    kw.setdefault("media", "video")
    kw.setdefault("refs_enabled", True)
    kw.setdefault("response_format", [{"segment_index": int, "prompt": str, "refs": [int]}])
    return PromptConfig(**kw)


def _raw(items):
    return json.dumps(items)


def test_label_mismatch_is_problem():
    cfg = _cfg()
    ok = [{"segment_index": 1, "prompt": "A rider on a horse <image1> and a wagon <image2>.", "refs": [1, 3]}]
    _, problems = gp.parse_and_validate(_raw(ok), [1], cfg)
    assert problems == []
    bad = [{"segment_index": 1, "prompt": "A rider on a horse <image3>.", "refs": [1, 3]}]
    _, problems = gp.parse_and_validate(_raw(bad), [1], cfg)
    assert any(p.reason == gp.LABEL_REASON for p in problems)
    no_refs_tag = [{"segment_index": 2, "prompt": "A wagon <image1>.", "refs": []}]
    _, problems = gp.parse_and_validate(_raw(no_refs_tag), [2], cfg)
    assert any(p.reason == gp.LABEL_REASON for p in problems)


def test_refs_missing_is_problem():
    cfg = _cfg()
    missing = [{"segment_index": 1, "prompt": "A wagon on a road."}]
    results, problems = gp.parse_and_validate(_raw(missing), [1], cfg)
    assert any(p.reason == gp.REF_REASON_MISSING for p in problems)
    assert "refs" not in results[1].extra
    dup = [{"segment_index": 1, "prompt": "A wagon.", "refs": [2, 2]}]
    _, problems = gp.parse_and_validate(_raw(dup), [1], cfg)
    assert any(p.reason == gp.REF_REASON_DUP for p in problems)
    unsorted = [{"segment_index": 1, "prompt": "A wagon.", "refs": [5, 2]}]
    results, problems = gp.parse_and_validate(_raw(unsorted), [1], cfg)
    assert problems == [] and results[1].extra["refs"] == [2, 5]


def test_on_unfixed_fail_stops_run():
    cfg_fail = _cfg(on_unfixed="fail")
    res = gp.PromptResult(4, "A wagon.", True, "метки <imageN>: не совпадают с refs сегмента")
    with pytest.raises(GeminiError) as info:
        gp._fail_unfixed([res], cfg_fail)
    assert "сегмент 4" in str(info.value)
    gp._fail_unfixed([gp.PromptResult(5, "ok", False)], cfg_fail)  # чистые результаты не останавливают


def test_on_unfixed_keep_default_unchanged():
    cfg = PromptConfig(style_file="x", media="video")
    assert cfg.on_unfixed == "keep" and cfg.refs_enabled is False
    # в keep вызов _fail_unfixed не делается: проверка стоит в generate_prompts под условием on_unfixed == "fail"
    res = gp.PromptResult(4, "A wagon.", True, "стиль: отрицание")
    assert res.needs_review is True
    assert gp.response_schema([{"segment_index": int, "prompt": str}])["items"]["required"] == ["segment_index", "prompt"]
