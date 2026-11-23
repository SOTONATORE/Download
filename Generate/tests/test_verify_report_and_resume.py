"""Отчёт проверки: причины needs_review из prompts.json; восстановление клипов «на проверку» при возобновлении.

Без сети и без установки пакетов: транспорт и генерация подменяются заглушками (monkeypatch).
Проверяются: отчёт cli._verify_report с разбивкой причин; _rework_seeds с флагом и без него;
проводка флага в _photo_step и build_prompts (rework_style_flag_without_refs=True).
"""
from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Generate import cli  # noqa: E402
from Generate.core import gemini_orchestrator as go  # noqa: E402
from Generate.core import gemini_prompts as gp  # noqa: E402
from Generate.core import prompt_pipeline as pp  # noqa: E402
from Generate.core.gemini_prompts import PromptConfig  # noqa: E402
from Generate.core.srt_parser import Segment as SrtSegment  # noqa: E402

SPAM = "стиль: keyword spam"
STYLE = "Soft pastel cartoon style."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


def _photo_entry(num, prompt, *, status="ready", needs_review=False, error="", **extra):
    """Запись фото без start_ms (как у волны фото)."""
    entry = {"num": num, "num_frames": 1, "prompt": prompt, "status": status,
             "attempts": 1, "error": error, "needs_review": needs_review, "extra": {}}
    entry.update(extra)
    return entry


def _video_entry(num, prompt, **kw):
    entry = _photo_entry(num, prompt, **kw)
    entry.update({"start_ms": (num - 1) * 10_000, "end_ms": num * 10_000, "clip_ms": 10_000})
    entry.pop("num_frames")
    entry["num_frames"] = 49
    return entry


# --- отчёт проверки -----------------------------------------------------------

def _counts(flag_photo, reasons):
    zero = {"ready": 0, "forbidden": 0, "leftover": 0, "content": 0, "frames_bad": 0, "rework": 0}
    return {
        "video_nums": [], "photo_nums": list(range(1, 60)),
        "video": {**zero, "flagged": []},
        "photo": {**zero, "frames_rule": 1, "flagged": flag_photo},
        "review_reasons": reasons,
    }


PREP = {"video_model": "none", "video_profile": "ltx25", "photo_profile": "cartoon", "missing": [1]}


def test_review_reasons_breakdown_lists_reason_and_clips():
    prompts = {}
    for n in range(26, 51):
        prompts[str(n)] = _photo_entry(n, "x", needs_review=True, error=SPAM)
    for n in range(251, 276):
        prompts[str(n)] = _photo_entry(n, "x", needs_review=True, error=SPAM)
    nums = sorted(int(k) for k, v in prompts.items() if v["needs_review"])
    reasons = cli._review_reasons(prompts, nums)
    assert list(reasons) == [SPAM]
    assert len(reasons[SPAM]) == 50
    assert reasons[SPAM][0] == 26 and reasons[SPAM][-1] == 275


def test_review_reasons_split_dedupe_and_missing_reason():
    prompts = {
        "1": _photo_entry(1, "x", needs_review=True, error="стиль: кавычки; стиль: кавычки; " + SPAM),
        "2": _photo_entry(2, "x", needs_review=True, error=""),
        "3": _photo_entry(3, "x", needs_review=True, error=SPAM),
    }
    reasons = cli._review_reasons(prompts, [1, 2, 3])
    assert reasons[SPAM] == [1, 3]
    assert reasons["стиль: кавычки"] == [1]
    assert reasons["причина не указана"] == [2]


def test_verify_report_contains_breakdown_and_keeps_existing_sections():
    reasons = {SPAM: [26, 27, 251]}
    report = cli._verify_report(PREP, _counts(list(range(26, 28)) + [251], reasons), [])
    assert "### Причины отправки на проверку (needs_review)" in report
    assert SPAM in report
    assert "клипов 3" in report
    assert "26, 27, 251" in report
    for heading in ("### Проверка запрещённых слов и маркеров",
                    "### Проверка содержания кадра",
                    "### Проверка num_frames",
                    "## Проверка Gemini: отчёт"):
        assert heading in report
    assert "На проверку (флаг needs_review" in report


def test_verify_report_without_reasons_key_is_unchanged():
    counts = _counts([], {})
    counts.pop("review_reasons")
    report = cli._verify_report(PREP, counts, [])
    assert "Причины отправки на проверку" not in report


def test_verify_report_empty_reasons_says_no_clips():
    report = cli._verify_report(PREP, _counts([], {}), [])
    assert "### Причины отправки на проверку (needs_review)" in report
    assert "клипов на проверке нет" in report


# --- _rework_seeds --------------------------------------------------------------

def test_rework_seeds_default_skips_entry_without_rework_pending():
    prompts = {"5": _photo_entry(5, "Scene " + STYLE, needs_review=True, error=SPAM)}
    assert go._rework_seeds(prompts, photo=True) == []


def test_rework_seeds_flag_on_recovers_needs_review_entry():
    prompts = {"5": _photo_entry(5, "A man stands in a room. " + STYLE, needs_review=True, error=SPAM)}
    seeds = go._rework_seeds(prompts, photo=True, rework_needs_review=True, style_text=STYLE)
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.num == 5
    assert seed.prompt_raw == "A man stands in a room"
    assert seed.rework_pending == [gp._PARSE_PREFIX + SPAM]
    assert seed.rework_round == 0
    assert seed.needs_review is True


def test_rework_seeds_flag_on_uses_prompt_raw_when_present():
    entry = _photo_entry(7, "A cat. " + STYLE, needs_review=True, error="")
    entry["prompt_raw"] = "A cat"
    seeds = go._rework_seeds({"7": entry}, photo=True, rework_needs_review=True, style_text=STYLE)
    assert seeds[0].prompt_raw == "A cat"
    assert seeds[0].rework_pending == [gp._PARSE_PREFIX + "ответ помечен на проверку"]


def test_rework_seeds_flag_on_ignores_round_at_limit():
    prompts = {"5": _photo_entry(5, "A man. " + STYLE, needs_review=True, error=SPAM,
                                 rework_round=gp.REPAIR_ROUNDS)}
    assert go._rework_seeds(prompts, photo=True, rework_needs_review=True, style_text=STYLE) == []


def test_rework_seeds_flag_on_respects_only_and_wave():
    prompts = {
        "5": _photo_entry(5, "A man. " + STYLE, needs_review=True, error=SPAM),
        "6": _video_entry(6, "Video scene. " + STYLE, needs_review=True, error=SPAM),
        "7": _photo_entry(7, "Other. " + STYLE, needs_review=False, error=""),
    }
    photo = go._rework_seeds(prompts, photo=True, rework_needs_review=True, style_text=STYLE)
    assert [s.num for s in photo] == [5]
    video = go._rework_seeds(prompts, photo=False, rework_needs_review=True, style_text=STYLE)
    assert [s.num for s in video] == [6]
    only = go._rework_seeds(prompts, photo=True, only={9}, rework_needs_review=True, style_text=STYLE)
    assert only == []


def test_rework_seeds_flag_on_does_not_touch_needs_review_false():
    prompts = {"5": _photo_entry(5, "A man. " + STYLE, needs_review=False, error=SPAM)}
    assert go._rework_seeds(prompts, photo=True, rework_needs_review=True, style_text=STYLE) == []


def test_rework_kwargs_off_for_plain_config_and_mock_like_objects():
    plain = PromptConfig(style_file="s.md", media="image")
    assert go.rework_kwargs(plain) == {"rework_needs_review": False, "style_text": ""}
    mock_like = SimpleNamespace(rework_style_flag_without_refs=object(), fail_fast=False)
    assert go.rework_needs_review_for(mock_like) is False


def test_rework_needs_review_respects_fail_fast():
    cfg = PromptConfig(style_file="s.md", media="image", rework_style_flag_without_refs=True, fail_fast=True)
    assert go.rework_needs_review_for(cfg) is False
    cfg2 = PromptConfig(style_file="s.md", media="image", rework_style_flag_without_refs=True)
    assert go.rework_needs_review_for(cfg2) is True


# --- проводка: _photo_step и build_prompts -------------------------------------

def _style_file(tmp_path):
    p = tmp_path / "preset.md"
    p.write_text(STYLE, encoding="utf-8")
    return str(p)


def _photo_cfg(tmp_path):
    return PromptConfig(style_file="style.md", media="image", preset=gp.PRESET_NONE,
                        niche_preset_file=_style_file(tmp_path), rework_style_flag_without_refs=True)


def _write_state(path: Path, prompts: dict) -> dict:
    state = {"srt_hash": "x", "model": "fake", "profile": "cartoon", "prompts": prompts}
    go.atomic_write_json(path, state)
    return state


def test_photo_step_recovers_needs_review_entry_when_flag_on(monkeypatch, tmp_path):
    calls = []

    def fake_generate_prompts(segments, wanted, cfg, managed, model, on_batch=None, **kw):
        calls.append({"pending": list(wanted), "rework": kw.get("rework")})

    monkeypatch.setattr(pp.gp, "generate_prompts", fake_generate_prompts)
    prompts = {"26": _photo_entry(26, "A man in a room. " + STYLE, needs_review=True, error=SPAM)}
    state = {"prompts": prompts}
    watch = SimpleNamespace(calls=0)
    managed = SimpleNamespace(active_model="m")
    pp._photo_step([26], segments=[], photo_cfg=_photo_cfg(tmp_path), watch=watch,
                   prompts_path=tmp_path / "prompts.json", gemini_options={}, errors=[], aborts=[],
                   managed=managed, state=state, lock=threading.Lock())
    assert len(calls) == 1
    assert calls[0]["pending"] == []
    seeds = calls[0]["rework"]
    assert [s.num for s in seeds] == [26]
    assert seeds[0].prompt_raw == "A man in a room"
    assert seeds[0].rework_pending == [gp._PARSE_PREFIX + SPAM]


def test_photo_step_flag_off_does_not_recover(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pp.gp, "generate_prompts",
                        lambda segments, wanted, cfg, managed, model, on_batch=None, **kw: calls.append(kw))
    cfg = _photo_cfg(tmp_path)
    cfg.rework_style_flag_without_refs = False
    prompts = {"26": _photo_entry(26, "A man. " + STYLE, needs_review=True, error=SPAM)}
    pp._photo_step([26], segments=[], photo_cfg=cfg, watch=SimpleNamespace(calls=0),
                   prompts_path=tmp_path / "prompts.json", gemini_options={}, errors=[], aborts=[],
                   managed=SimpleNamespace(active_model="m"), state={"prompts": prompts},
                   lock=threading.Lock())
    assert calls == []   # не готово, нет очереди, нет флага: вызова модели нет


def test_build_prompts_photo_only_resumes_needs_review_from_file(monkeypatch, tmp_path):
    prompts_path = tmp_path / "prompts.json"
    prompts = {"26": _photo_entry(26, "A man in a room. " + STYLE, needs_review=True, error=SPAM)}
    state = _write_state(prompts_path, prompts)
    seen = []

    def fake_generate_prompts(segments, wanted, cfg, managed, model, on_batch=None, **kw):
        seen.append({"pending": list(wanted), "rework": [s.num for s in kw.get("rework", [])]})

    monkeypatch.setattr(pp.gp, "generate_prompts", fake_generate_prompts)
    monkeypatch.setattr(pp.go, "ManagedTransport", lambda *a, **k: SimpleNamespace(active_model="m"))
    monkeypatch.setattr(pp, "_load_shared", lambda st, path, segments, profile: state)
    cfg = PromptConfig(style_file="style.md", media="image", rework_style_flag_without_refs=True)
    run = pp.build_prompts(
        [26], segments=[SrtSegment(num=26, start_ms=0, end_ms=10_000, text="t")], photo_nums={26},
        cfg=cfg, profile="cartoon", timing=None, photo_profile="cartoon",
        photo_cfg=_photo_cfg(tmp_path), transport=SimpleNamespace(), gemini_options={},
        prompts_path=prompts_path)
    assert seen and seen[0]["rework"] == [26]
    assert run is not None
