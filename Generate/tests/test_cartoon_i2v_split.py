"""Офлайн-тесты равномерного отбора сегментов для I2V (select_i2v_segments)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from Generate.core.segment_split import Segment, select_i2v_segments
from Generate.core.srt_parser import parse_srt

ROOT = Path(__file__).resolve().parents[2]
FINAL_SRT = ROOT / "final.srt"


def _make_uniform_segments(count: int, dur: float = 5.0, start_at: float = 0.0) -> list[Segment]:
    return [
        Segment(
            id=i + 1,
            start=start_at + i * dur,
            end=start_at + (i + 1) * dur,
            text=f"Segment {i + 1}",
        )
        for i in range(count)
    ]


def test_segments_shorter_than_min_anim_sec_are_never_selected():
    segs = [
        Segment(id=1, start=0.0, end=1.0, text="short 1s"),
        Segment(id=2, start=1.0, end=2.99, text="short 1.99s"),
        Segment(id=3, start=3.0, end=5.0, text="exact 2.0s"),
        Segment(id=4, start=5.0, end=9.0, text="long 4.0s"),
    ]
    # Даже при 100%/100% и максимальной оценке 100 короткие сегменты не проходят фильтр min_anim_sec=2.0
    values = {s.id: 100 for s in segs}
    res = select_i2v_segments(
        segs,
        values,
        intro_seconds=180.0,
        intro_video_percent=100.0,
        after_intro_video_percent=100.0,
        min_anim_sec=2.0,
    )
    assert res["eligible"] == [3, 4]
    assert res["video"] == [3, 4]
    assert res["photo"] == [1, 2]


def test_intro_and_after_intro_zones_keep_target_ratios_and_window_uniformity():
    # 12 минут = 720 с: 144 сегмента по 5 с. Первые 180 с (36 сегментов) — вступление (70%), далее — 10%.
    segs = _make_uniform_segments(144, dur=5.0)
    values = {s.id: ((s.id * 37) % 101) for s in segs}

    res = select_i2v_segments(
        segs,
        values,
        intro_seconds=180.0,
        intro_video_percent=70.0,
        after_intro_video_percent=10.0,
        min_anim_sec=2.0,
        window_seconds=45.0,
    )

    video_set = set(res["video"])
    intro_segs = [s for s in segs if s.start < 180.0]
    after_segs = [s for s in segs if s.start >= 180.0]

    intro_dur = sum(s.duration for s in intro_segs)
    intro_vid = sum(s.duration for s in intro_segs if s.id in video_set)
    after_dur = sum(s.duration for s in after_segs)
    after_vid = sum(s.duration for s in after_segs if s.id in video_set)

    # Фактическая доля близка к 70% во вступлении и к 10% после вступления
    assert 0.55 <= intro_vid / intro_dur <= 0.78
    assert 0.05 <= after_vid / after_dur <= 0.16

    # Равномерность в окнах по 45 с: ни одно окно основной части не даёт всплеска (> 10 с из 45 с)
    windows: dict[int, list[Segment]] = {}
    for s in segs:
        windows.setdefault(math.floor(s.start / 45.0), []).append(s)

    for w_idx, members in sorted(windows.items()):
        w_vid = sum(s.duration for s in members if s.id in video_set)
        if members[0].start >= 180.0:
            assert w_vid <= 10.0, f"окно {w_idx} основной части получило всплеск {w_vid} с"
        else:
            assert w_vid >= 20.0, f"окно {w_idx} вступления недополучило видео: {w_vid} с"


def test_hundred_percent_mode_selects_all_eligible_and_rejects_short():
    segs = [
        Segment(id=1, start=0.0, end=1.5, text="short intro"),
        Segment(id=2, start=1.5, end=4.0, text="ok intro"),
        Segment(id=3, start=200.0, end=201.9, text="short after"),
        Segment(id=4, start=201.9, end=206.0, text="ok after"),
    ]
    values = {2: 10, 4: 10}  # не пригодные по длительности не обязаны присутствовать в values
    res = select_i2v_segments(
        segs,
        values,
        intro_seconds=180.0,
        intro_video_percent=100.0,
        after_intro_video_percent=100.0,
        min_anim_sec=2.0,
    )
    assert res["video"] == [2, 4]
    assert res["photo"] == [1, 3]
    assert res["eligible"] == [2, 4]


def test_zero_percent_mode_selects_no_video():
    segs = _make_uniform_segments(10, dur=5.0)
    values = {s.id: 100 for s in segs}
    res = select_i2v_segments(
        segs,
        values,
        intro_seconds=20.0,
        intro_video_percent=0.0,
        after_intro_video_percent=0.0,
        min_anim_sec=2.0,
    )
    assert res["video"] == []
    assert res["photo"] == [s.id for s in segs]
    assert res["video_seconds"] == 0.0


def test_validation_errors():
    segs = _make_uniform_segments(2, dur=3.0)
    with pytest.raises(ValueError):
        select_i2v_segments(segs, {1: 50, 2: 50}, -1.0, 70.0, 10.0)
    with pytest.raises(ValueError):
        select_i2v_segments(segs, {1: 50, 2: 50}, 10.0, 120.0, 10.0)
    with pytest.raises(ValueError):
        select_i2v_segments(segs, {1: 50, 2: 50}, 10.0, 70.0, -5.0)
    with pytest.raises(ValueError):
        select_i2v_segments(segs, {1: 50, 2: 50}, 10.0, 70.0, 10.0, min_anim_sec=-0.1)
    with pytest.raises(ValueError):
        select_i2v_segments(segs, {1: 50}, 10.0, 70.0, 10.0)


def test_realistic_durations_from_final_srt_or_synthetic():
    if FINAL_SRT.is_file():
        srt_segs = parse_srt(FINAL_SRT)
        segs = [
            Segment(id=s.num, start=s.start_ms / 1000.0, end=s.end_ms / 1000.0, text=s.text)
            for s in srt_segs
        ]
    else:
        # Синтетический фолбэк с чередованием коротких (1.2 с) и обычных (3.5–6.0 с) сегментов
        segs = []
        t = 0.0
        for i in range(1, 151):
            dur = 1.2 if i % 5 == 0 else (3.5 + (i % 4) * 0.8)
            segs.append(Segment(id=i, start=t, end=t + dur, text=f"Line {i}"))
            t += dur

    values = {
        s.id: (85 if s.id % 3 == 0 else 45)
        for s in segs
        if s.duration >= 2.0 - 1e-9
    }
    res = select_i2v_segments(
        segs,
        values,
        intro_seconds=180.0,
        intro_video_percent=70.0,
        after_intro_video_percent=10.0,
        min_anim_sec=2.0,
        window_seconds=45.0,
    )
    by_id = {s.id: s for s in segs}
    assert res["video"], "при реалистичных длительностях должны выбираться видео-сегменты"
    for vid in res["video"]:
        assert by_id[vid].duration >= 2.0 - 1e-9

    # Проверяем отсутствие всплесков в 45-секундных окнах основной части
    video_set = set(res["video"])
    after_windows: dict[int, list[Segment]] = {}
    for s in segs:
        if s.start >= 180.0:
            after_windows.setdefault(math.floor(s.start / 45.0), []).append(s)
    for w_idx, members in after_windows.items():
        win_total = sum(s.duration for s in members)
        win_vid = sum(s.duration for s in members if s.id in video_set)
        if win_total >= 20.0:
            assert win_vid <= win_total * 0.35 + 3.0, f"окно {w_idx}: слишком высокая доля видео {win_vid}/{win_total}"
