"""Офлайн-тесты core/segment_split.py (STEP 3). Без сети и без файлов."""
from __future__ import annotations

import pytest

try:
    from Generate.core.segment_split import Segment, all_photo, select_video_segments
except ImportError:  # запуск из каталога Generate
    from core.segment_split import Segment, all_photo, select_video_segments


def seg(i, start, dur, text=""):
    return Segment(id=i, start=float(start), end=float(start + dur), text=text)


def test_intro_is_video_even_over_budget():
    # Вступление 10 с, бюджет 2 с (10% от 20 с): вступление всё равно видео.
    segs = [seg(1, 0, 5), seg(2, 5, 5), seg(3, 60, 10)]
    values = {1: 0, 2: 0, 3: 100}
    res = select_video_segments(segs, values, video_percent=10, intro_seconds=10)
    assert res["video"] == [1, 2]
    assert res["video_seconds"] == 10
    assert res["video_seconds"] > res["budget_seconds"]
    assert res["photo"] == [3]


def test_video_seconds_within_budget_unless_only_intro_exceeds():
    segs = [seg(1, 0, 4), seg(2, 50, 6), seg(3, 100, 6), seg(4, 150, 6)]
    values = {1: 50, 2: 90, 3: 80, 4: 10}
    res = select_video_segments(segs, values, video_percent=50, intro_seconds=0)
    assert res["budget_seconds"] == pytest.approx(11)
    assert res["video_seconds"] <= res["budget_seconds"]
    assert res["video"] == [1, 2]  # окно 0: 1 (50); окно 1: 2 (90), 4+6=10 <= 11; окна 2, 3 не влезают


def test_sort_inside_window_by_value_then_start():
    # Все три в окне 0 (0–45 с). Бюджет 10 с: берём по убыванию value.
    segs = [seg(1, 5, 4), seg(2, 10, 4), seg(3, 20, 4)]
    values = {1: 70, 2: 90, 3: 70}
    # total 12, бюджет 80% = 9.6: сначала 2 (90, 4 с), затем 1 (70, раньше по start, 8 с), 3 не влезает
    res = select_video_segments(segs, values, video_percent=80, intro_seconds=0)
    assert res["video"] == [1, 2]  # порядок вывода по start
    assert res["photo"] == [3]


def test_tie_in_value_earlier_start_first():
    segs = [seg(1, 20, 4), seg(2, 5, 4)]
    values = {1: 50, 2: 50}
    res = select_video_segments(segs, values, video_percent=50, intro_seconds=0)
    # total 8, budget 4: одна помещается, при равенстве выигрывает более ранний start
    assert res["video"] == [2]


def test_window_order_is_time_order():
    # Окно 0: сегмент 1 (value 10). Окно 1 (45–90): сегмент 2 (value 100).
    # Бюджет 4 с: окно 0 идёт первым и занимает бюджет, даже с низким value.
    segs = [seg(1, 0, 4), seg(2, 50, 4)]
    values = {1: 10, 2: 100}
    res = select_video_segments(segs, values, video_percent=50, intro_seconds=0)
    assert res["video"] == [1]
    assert res["photo"] == [2]


def test_skip_segment_that_does_not_fit_and_try_next():
    # Окно 0: сегмент 1 (10 с, value 90) не влезает в бюджет 6 с; сегмент 2 (4 с, value 40) влезает.
    segs = [seg(1, 0, 10), seg(2, 12, 4), seg(3, 30, 2)]
    values = {1: 90, 2: 40, 3: 10}
    res = select_video_segments(segs, values, video_percent=50, intro_seconds=0)
    # total 16, budget 8: 1 пропущен (10 > 8), 2 берётся (4), 3 влезает (4+2=6)
    assert res["video"] == [2, 3]
    assert res["photo"] == [1]


def test_each_segment_in_exactly_one_list():
    segs = [seg(i, i * 7, 6) for i in range(1, 9)]
    values = {i: (i * 37) % 101 for i in range(1, 9)}
    res = select_video_segments(segs, values, video_percent=45, intro_seconds=10)
    assert sorted(res["video"] + res["photo"]) == sorted(s.id for s in segs)
    assert set(res["video"]).isdisjoint(res["photo"])
    assert res["video"] == sorted(res["video"], key=lambda i: segs[i - 1].start)
    assert res["photo"] == sorted(res["photo"], key=lambda i: segs[i - 1].start)


def test_video_percent_zero_gives_only_intro():
    segs = [seg(1, 0, 4), seg(2, 4, 4), seg(3, 30, 4)]
    values = {1: 50, 2: 50, 3: 100}
    res = select_video_segments(segs, values, video_percent=0, intro_seconds=5)
    assert res["video"] == [1, 2]
    assert res["photo"] == [3]
    assert res["budget_seconds"] == 0


def test_video_percent_zero_no_intro_gives_no_video():
    segs = [seg(1, 0, 4), seg(2, 30, 4)]
    values = {1: 50, 2: 50}
    res = select_video_segments(segs, values, video_percent=0, intro_seconds=0)
    assert res["video"] == []
    assert res["photo"] == [1, 2]
    assert res["video_seconds"] == 0


@pytest.mark.parametrize("percent", [-1, 100.5, 101, True, "50"])
def test_video_percent_out_of_range_raises(percent):
    with pytest.raises(ValueError):
        select_video_segments([seg(1, 0, 4)], {1: 50}, video_percent=percent, intro_seconds=0)


@pytest.mark.parametrize("bad", [-1, 101, float("nan"), None])
def test_value_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        select_video_segments([seg(1, 0, 4)], {1: bad}, video_percent=50, intro_seconds=0)


def test_missing_value_raises():
    with pytest.raises(ValueError):
        select_video_segments([seg(1, 0, 4), seg(2, 10, 4)], {1: 50}, video_percent=50, intro_seconds=0)


def test_negative_intro_raises():
    with pytest.raises(ValueError):
        select_video_segments([seg(1, 0, 4)], {1: 50}, video_percent=50, intro_seconds=-1)


def test_all_photo_structure():
    segs = [seg(2, 10, 4), seg(1, 0, 4)]
    res = all_photo(segs)
    assert res["photo"] == [1, 2]
    assert res["video"] == []
    assert res["video_seconds"] == 0
    assert set(res) == {"video", "photo", "budget_seconds", "video_seconds", "intro_seconds_used"}
