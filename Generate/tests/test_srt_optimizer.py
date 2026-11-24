"""Офлайн-тесты оптимизатора SRT для ниши картунов (Generate.core.srt_optimizer)."""
from __future__ import annotations

from pathlib import Path

import pytest

from Generate.core.srt_optimizer import (
    DEFAULT_MAX_SEC,
    DEFAULT_MIN_SEC,
    optimize_segments,
    optimize_srt_file,
    render_srt,
)
from Generate.core.srt_parser import Segment, parse_srt


def _make_cues(durations_ms: list[int], texts: list[str] | None = None, start_ms: int = 0) -> list[Segment]:
    cues: list[Segment] = []
    t = start_ms
    for i, dur in enumerate(durations_ms, start=1):
        text = texts[i - 1] if texts is not None else f"Cue {i}"
        cues.append(Segment(num=i, start_ms=t, end_ms=t + dur, text=text))
        t += dur
    return cues


def test_ten_one_second_cues_merge_within_default_bounds():
    cues = _make_cues([1000] * 10)
    out = optimize_segments(cues)

    assert len(out) < len(cues)
    assert [s.num for s in out] == list(range(1, len(out) + 1))
    min_ms = round(DEFAULT_MIN_SEC * 1000)
    max_ms = round(DEFAULT_MAX_SEC * 1000)
    for seg in out:
        dur = seg.end_ms - seg.start_ms
        assert min_ms <= dur <= max_ms, f"сегмент №{seg.num} вне [{min_ms}, {max_ms}] мс: {dur} мс"
    assert " ".join(s.text for s in out) == " ".join(c.text for c in cues)


def test_sentence_boundaries_are_respected_for_comparable_durations():
    # 7 реплик по 1 с допускают только два строгих разбиения [2.5, 5.0]: 3+4 и 4+3
    # с одинаковой стоимостью по длительности — выбор определяется концом предложения.
    texts_break_at_3 = ["one", "two", "three.", "four", "five", "six", "seven."]
    out_3 = optimize_segments(_make_cues([1000] * 7, texts=texts_break_at_3))
    assert [(s.end_ms - s.start_ms) for s in out_3] == [3000, 4000]
    assert out_3[0].text.endswith(".")

    texts_break_at_4 = ["one", "two", "three", "four.", "five", "six", "seven."]
    out_4 = optimize_segments(_make_cues([1000] * 7, texts=texts_break_at_4))
    assert [(s.end_ms - s.start_ms) for s in out_4] == [4000, 3000]
    assert out_4[0].text.endswith(".")


def test_empty_cues_separate_blocks_and_remain_in_place():
    cues = [
        Segment(num=1, start_ms=0, end_ms=1000, text="a1"),
        Segment(num=2, start_ms=1000, end_ms=2000, text="a2"),
        Segment(num=3, start_ms=2000, end_ms=3000, text="a3"),
        Segment(num=4, start_ms=3000, end_ms=4000, text="a4."),
        Segment(num=5, start_ms=4000, end_ms=4500, text=""),
        Segment(num=6, start_ms=4500, end_ms=5500, text="b1"),
        Segment(num=7, start_ms=5500, end_ms=6500, text="b2"),
        Segment(num=8, start_ms=6500, end_ms=7500, text="b3"),
        Segment(num=9, start_ms=7500, end_ms=8500, text="b4."),
    ]
    out = optimize_segments(cues)

    assert len(out) == 3
    assert [s.num for s in out] == [1, 2, 3]
    assert out[0] == Segment(num=1, start_ms=0, end_ms=4000, text="a1 a2 a3 a4.")
    assert out[1] == Segment(num=2, start_ms=4000, end_ms=4500, text="")
    assert out[2] == Segment(num=3, start_ms=4500, end_ms=8500, text="b1 b2 b3 b4.")


def test_unavoidable_short_cue_does_not_trigger_global_greedy_fallback():
    # Реплика 1 (2.32 с) + пауза 0.12 с + реплика 2 (2.60 с) = 5.04 с > 5.0 с.
    # Реплику 1 нельзя склеить без превышения потолка, но глобальное ДП должно
    # сохранить оптимальное разбиение для остальной части файла без жадного фолбэка.
    cues = [
        Segment(num=1, start_ms=0, end_ms=2320, text="Short unavoidable."),
        Segment(num=2, start_ms=2440, end_ms=5040, text="Next cue."),
    ]
    t = 5040
    for i in range(3, 13):
        cues.append(Segment(num=i, start_ms=t, end_ms=t + 1000, text=f"Tail {i}."))
        t += 1000

    warns: list[str] = []
    out = optimize_segments(cues, warn=warns.append)

    assert warns == [], f"неожиданный жадный фолбэк: {warns}"
    durs = [s.end_ms - s.start_ms for s in out]
    assert all(d <= 5000 for d in durs), f"обнаружен сегмент длиннее 5.0 с: {durs}"
    under_min = [s for s in out if (s.end_ms - s.start_ms) < 2500]
    assert len(under_min) == 1
    assert under_min[0].start_ms == 0 and under_min[0].end_ms == 2320
    rest_durs = [s.end_ms - s.start_ms for s in out[1:]]
    assert all(2500 <= d <= 5000 for d in rest_durs), f"остальные сегменты вне [2.5, 5.0] с: {rest_durs}"


def test_optimize_srt_file_round_trips_through_parse_srt(tmp_path: Path):
    cues = _make_cues(
        [1000] * 8,
        texts=["One", "two", "three", "four.", "Five", "six", "seven", "eight."],
    )
    in_path = tmp_path / "input.srt"
    out_path = tmp_path / "final_optimized.srt"
    in_path.write_text(render_srt(cues), encoding="utf-8")

    stats = optimize_srt_file(str(in_path), str(out_path))

    assert out_path.is_file()
    parsed = parse_srt(str(out_path))
    assert stats.input_count == 8
    assert stats.output_count == len(parsed)
    assert stats.out_of_range == 0
    assert stats.warnings == ()
    assert [s.num for s in parsed] == list(range(1, len(parsed) + 1))
    for seg in parsed:
        assert 2500 <= (seg.end_ms - seg.start_ms) <= 5000
    assert " ".join(s.text for s in parsed) == "One two three four. Five six seven eight."


@pytest.mark.parametrize(
    ("min_sec", "target_sec", "max_sec"),
    [
        (0.0, 4.5, 5.0),
        (-1.0, 4.5, 5.0),
        (4.6, 4.5, 5.0),
        (2.5, 5.5, 5.0),
        (float("nan"), 4.5, 5.0),
    ],
)
def test_invalid_params_raise_value_error(min_sec: float, target_sec: float, max_sec: float):
    cues = _make_cues([1000] * 4)
    with pytest.raises(ValueError):
        optimize_segments(cues, min_sec=min_sec, target_sec=target_sec, max_sec=max_sec)
