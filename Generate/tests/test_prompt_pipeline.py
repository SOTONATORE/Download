"""Офлайн-тесты общего конвейера промптов и распределения без фото (V1, V2).

Без сети и без реального Gemini: вызовы go.generate_and_store, gp.generate_prompts и транспорта подменяются.
Файлы пишутся только в tmp_path. Тесты не запускались при подготовке изменения (владелец запускает их на GitHub).
"""
from __future__ import annotations

import json

try:
    from Generate.core import gemini_prompts as gp
    from Generate.core import orchestrator as orch
    from Generate.core import prompt_pipeline as pp
    from Generate.core.srt_parser import Segment as SrtSegment
except ImportError:  # запуск из каталога Generate
    from core import gemini_prompts as gp  # type: ignore
    from core import orchestrator as orch  # type: ignore
    from core import prompt_pipeline as pp  # type: ignore
    from core.srt_parser import Segment as SrtSegment  # type: ignore


def srt_segments():
    """Пять сегментов SRT по 10 с каждый: номера 1..5."""
    return [
        SrtSegment(num=n, start_ms=(n - 1) * 10_000, end_ms=n * 10_000, text=f"текст {n}")
        for n in range(1, 6)
    ]


MISSING = [2, 4]


def _fake_video_store(calls: list):
    """Подмена go.generate_and_store: все запрошенные номера получают готовый промпт, пишет prompts.json."""
    def fake(segments, needed, cfg, transport, prompts_path, **kw):
        calls.append({"needed": list(needed), "profile": kw.get("profile")})
        state = {"srt_hash": "x", "model": "fake", "profile": kw.get("profile"),
                 "prompts": {str(n): {"num": n, "status": "ready", "prompt": f"видео {n}", "num_frames": 49}
                             for n in sorted(needed)}}
        pp.go.atomic_write_json(prompts_path, state)
        return state
    return fake


def test_video_only_build_prompts_ready_is_all_needed(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(pp.go, "generate_and_store", _fake_video_store(calls))
    run = pp.build_prompts(
        [2, 4], segments=srt_segments(), photo_nums=set(), cfg="cfg", profile="ltx25", timing="timing",
        photo_profile=None, photo_cfg=None, transport="transport", gemini_options={"models": ("m",)},
        prompts_path=tmp_path / "prompts.json")
    assert run.ready == [2, 4]                      # те же номера, в порядке needed
    assert calls == [{"needed": [2, 4], "profile": "ltx25"}]  # один вызов видео-пути, без лишних аргументов
    assert run.errors == []


def test_video_only_never_enters_photo_path(monkeypatch, tmp_path):
    monkeypatch.setattr(pp.go, "generate_and_store", _fake_video_store([]))

    def boom(*a, **kw):
        raise AssertionError("фото-путь не должен вызываться без photo_profile")

    monkeypatch.setattr(pp.go, "load_state", boom)
    monkeypatch.setattr(pp.gp, "generate_prompts", boom)
    # даже если номера отмечены как фото, без photo_profile все идут в видео (как раньше)
    run = pp.build_prompts(
        [2, 4], segments=srt_segments(), photo_nums={2}, cfg="cfg", profile="ltx25", timing="timing",
        photo_profile=None, photo_cfg=None, transport="transport", gemini_options={},
        prompts_path=tmp_path / "prompts.json")
    assert run.ready == [2, 4]


def test_photo_entry_has_num_frames_one_and_no_timing_fields(monkeypatch, tmp_path):
    video_calls: list = []
    monkeypatch.setattr(pp.go, "generate_and_store", _fake_video_store(video_calls))

    def fake_generate_prompts(segments, wanted, cfg, transport, model, on_batch=None):
        results = [gp.PromptResult(num=n, prompt=f"фото {n}") for n in wanted]
        on_batch(results)
        return results

    monkeypatch.setattr(pp.gp, "generate_prompts", fake_generate_prompts)
    path = tmp_path / "prompts.json"
    run = pp.build_prompts(
        [2, 4], segments=srt_segments(), photo_nums={2, 4}, cfg="cfg", profile="ltx25", timing="timing",
        photo_profile="zimage_turbo", photo_cfg="photo_cfg", transport=None, gemini_options={},
        prompts_path=path)
    assert video_calls == []                        # все номера фото: видео-путь не вызывался
    assert run.ready == [2, 4]
    entries = json.loads(path.read_text(encoding="utf-8"))["prompts"]
    for n in ("2", "4"):
        entry = entries[n]
        assert entry["num_frames"] == 1             # фото: один кадр (PHOTO_NUM_FRAMES)
        assert entry["status"] == "ready"
        # фото-записи не содержат тайминга видео: start_ms, end_ms, clip_ms
        assert not {"start_ms", "end_ms", "clip_ms"} & set(entry)


def test_split_video_only_makes_no_gemini_call_and_keeps_keys(tmp_path):
    def send_must_not_run(prompt):
        raise AssertionError("в видео-режиме без фото Gemini для оценки не вызывается")

    dist = tmp_path / "distribution.json"
    result = orch.split_for_run(
        srt_segments(), MISSING, video_model="ltx25", video_percent=50, intro_seconds=0,
        send=send_must_not_run, distribution_path=dist, photo_on=False)
    assert set(result) == {"video", "photo", "budget_seconds", "video_seconds", "intro_seconds_used"}
    assert result["video"] == [2, 4]
    assert result["photo"] == []
    assert result["budget_seconds"] == 20.0         # сумма длительностей сегментов 2 и 4
    assert result["video_seconds"] == 20.0
    assert result["intro_seconds_used"] == 0
    # распределение всё равно записано (журнал и артефакт ожидают файл)
    data = json.loads(dist.read_text(encoding="utf-8"))
    assert [r["mode"] for r in data["segments"]] == ["video", "video"]


def test_plan_split_both_active_still_scores_once():
    calls: list = []

    class FakeInner:
        def generate_json(self, model, system, user, schema):
            calls.append(user)
            import re
            ids = [int(m) for m in re.findall(r"^id (\d+) \(", user, flags=re.MULTILINE)]
            return json.dumps([{"id": i, "video_value": 90 if i == 2 else 10} for i in ids])

    result = orch.plan_split(
        srt_segments(), MISSING, video_model="ltx25", video_percent=50, intro_seconds=0,
        transport=FakeInner(), gemini_options={}, photo_on=True, distribution_path=None)
    assert len(calls) == 1                          # оценка видео/фото выполняется, когда включены оба вида
    assert result["video"] == [2] and result["photo"] == [4]


def test_plan_split_photo_only_never_scores():
    class MustNotCall:
        def generate_json(self, *a, **kw):
            raise AssertionError("video_model none: Gemini не вызывается")

    result = orch.plan_split(
        srt_segments(), MISSING, video_model="none", video_percent=100, intro_seconds=10,
        transport=MustNotCall(), gemini_options={}, photo_on=True, distribution_path=None)
    assert result["video"] == [] and result["photo"] == [2, 4]
