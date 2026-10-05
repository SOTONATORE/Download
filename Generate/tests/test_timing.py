import random
from fractions import Fraction
from math import ceil

import pytest

from Generate.core.srt_parser import Segment
from Generate.core.timing import compute_timings, frames_for_rule

R81 = {"kind": "8k+1", "min_frames": 9}


def seg(num, start, end):
    return Segment(num, start, end, "t")


def run(segs, fps=24, rule=R81, mn=2.0, mx=8.0, gpu=12.0, pad=1.0):
    return compute_timings(segs, fps, rule, mn, mx, gpu, pad)


def valid_8k1(n):
    return n >= 9 and (n - 1) % 8 == 0


# ---------------- frames_for_rule ----------------

def test_4s_24fps_is_97():
    assert frames_for_rule(4000, 24, R81) == 97


def test_boundary_exact_and_plus_one():
    # 97 кадров = 4041,67 мс -> ровно на границе нужно 98 кадров -> 105
    assert frames_for_rule(4000, 24, R81) == 97
    assert frames_for_rule(4042, 24, R81) == 105   # ceil(97.008)=98 -> 105
    assert frames_for_rule(4041, 24, R81) == 97    # ceil(96.984)=97
    assert frames_for_rule(3999, 24, R81) == 97    # ceil(95.976)=96 -> 97
    assert frames_for_rule(4000, 24, R81) == 97    # ceil(96.0)=96 -> 97 (без float-хвоста)


@pytest.mark.parametrize("raw", [0, 1, 100, 300, 375])
def test_small_values_min_frames(raw):
    assert frames_for_rule(raw, 24, R81) == 9


def test_negative_raw_min_frames():
    assert frames_for_rule(-500, 24, R81) == 9


def test_rule_any():
    rule = {"kind": "any", "min_frames": 5}
    assert frames_for_rule(4000, 24, rule) == 96
    assert frames_for_rule(10, 24, rule) == 5


def test_rule_nk1_param():
    rule = {"kind": "nk+1", "n": 4, "min_frames": 5}
    assert frames_for_rule(4000, 24, rule) == 97   # 96 -> 4*24+1
    assert frames_for_rule(4100, 24, rule) == 101  # ceil(98.4)=99 -> 101
    assert frames_for_rule(10, 24, rule) == 5


def test_rule_8k1_step_from_rule():
    assert frames_for_rule(4100, 24, {"kind": "8k+1", "n": 4}) == 101


def test_unknown_rule():
    with pytest.raises(ValueError, match="Неизвестное правило"):
        frames_for_rule(1000, 24, {"kind": "zzz"})


def test_nk1_requires_n():
    with pytest.raises(ValueError, match="n"):
        frames_for_rule(1000, 24, {"kind": "nk+1"})


def test_exact_integer_math_no_float_artifacts():
    # 4.0 c * 25 fps = 100 кадров -> 105; 3.84 c * 25 = 96 -> 97
    assert frames_for_rule(4000, 25, R81) == 105
    assert frames_for_rule(3840, 25, R81) == 97
    assert frames_for_rule(1, 30, R81) == 9


# ---------------- compute_timings ----------------

def test_normal_segment():
    t = run([seg(1, 0, 3000), seg(2, 4000, 6000), seg(3, 8000, 9000)])
    assert t[0].raw_ms == 4000 and t[0].num_frames == 97
    assert t[0].clip_ms == 4042 and t[0].warnings == []
    assert t[0].num == 1 and t[0].start_ms == 0 and t[0].end_ms == 3000


def test_big_pause_included_once():
    t = run([seg(1, 0, 1000), seg(2, 6000, 7000)])
    assert t[0].raw_ms == 6000          # пауза уже внутри
    assert t[0].num_frames == 145       # ceil(144)=144 -> 145


def test_zero_pause():
    t = run([seg(1, 0, 4000), seg(2, 4000, 8000)])
    assert t[0].raw_ms == 4000


def test_last_segment_tail_pad():
    t = run([seg(1, 0, 1000), seg(2, 2000, 5000)], pad=1.5)
    assert t[1].raw_ms == 3000 + 1500
    assert t[1].num_frames == 113       # ceil(108)=108 -> 113


def test_zero_raw_fallback_to_own_length():
    t = run([seg(1, 1000, 4000), seg(2, 1000, 5000)])
    assert t[0].raw_ms == 3000
    assert any("некорректная длина" in w for w in t[0].warnings)


def test_negative_raw_fallback_to_own_length():
    t = run([seg(1, 5000, 8000), seg(2, 4000, 6000)])
    assert t[0].raw_ms == 3000
    assert any("некорректная длина" in w for w in t[0].warnings)


def test_zero_raw_and_zero_own_length_uses_clip_min():
    t = run([seg(1, 1000, 1000), seg(2, 1000, 2000)], mn=2.0)
    assert t[0].raw_ms == 2000
    assert any("некорректная длина" in w for w in t[0].warnings)
    assert not any("короче" in w for w in t[0].warnings)


def test_warning_short():
    t = run([seg(1, 0, 500), seg(2, 1000, 2000)], mn=2.0)
    assert any("короче минимума" in w for w in t[0].warnings)
    assert t[0].raw_ms == 1000 and t[0].num_frames == 25  # ceil(24)=24 -> 25


def test_warning_long_not_gpu():
    t = run([seg(1, 0, 1000), seg(2, 9000, 10000)], mx=8.0, gpu=12.0)
    w = t[0].warnings
    assert any("длиннее рекомендуемого" in x for x in w)
    assert not any("карта" in x for x in w)


def test_warning_gpu():
    t = run([seg(1, 0, 1000), seg(2, 13000, 14000)], mx=8.0, gpu=12.0)
    w = t[0].warnings
    assert any("длиннее рекомендуемого" in x for x in w)
    assert any("выше максимума, который тянет карта" in x for x in w)
    assert len(t) == 2                   # клип остаётся в результате


def test_no_warnings_on_threshold_equal():
    t = run([seg(1, 0, 1000), seg(2, 2000, 3000), seg(3, 10000, 11000)],
            mn=2.0, mx=8.0)
    assert t[0].raw_ms == 2000 and t[0].warnings == []
    assert t[1].raw_ms == 8000 and t[1].warnings == []


def test_frame_rule_switch():
    segs = [seg(1, 0, 3000), seg(2, 4000, 6000)]
    assert run(segs, rule=R81)[0].num_frames == 97
    assert run(segs, rule={"kind": "any", "min_frames": 1})[0].num_frames == 96
    assert run(segs, rule={"kind": "nk+1", "n": 4, "min_frames": 5})[0].num_frames == 97


@pytest.mark.parametrize("fps,frames", [(25, 105), (30, 121)])
def test_other_fps(fps, frames):
    t = run([seg(1, 0, 3000), seg(2, 4000, 6000)], fps=fps)
    assert t[0].num_frames == frames     # 100->105; 120->121


def test_empty_and_single():
    assert run([]) == []
    t = run([seg(7, 0, 3000)], pad=1.0)
    assert t[0].num == 7 and t[0].raw_ms == 4000


def test_bad_fps():
    with pytest.raises(ValueError):
        run([seg(1, 0, 1000)], fps=0)


def test_no_rounding_error_accumulation():
    # 10000 сегментов строго по 4,0 с: все по 97 кадров, сумма raw точна
    segs = [seg(i + 1, i * 4000, i * 4000 + 3000) for i in range(10000)]
    t = run(segs)
    assert all(c.num_frames == 97 and c.raw_ms == 4000 for c in t[:-1])
    assert sum(c.raw_ms for c in t[:-1]) == 4000 * 9999
    assert t[-1].raw_ms == 3000 + 1000


def test_matches_exact_fraction_reference():
    rnd = random.Random(1)
    starts = [0]
    for _ in range(3000):
        starts.append(starts[-1] + rnd.randint(1, 20000))
    segs = [seg(i + 1, s, s + 500) for i, s in enumerate(starts)]
    for fps in (24, 25, 30):
        t = run(segs, fps=fps)
        for i, c in enumerate(t[:-1]):
            raw = starts[i + 1] - starts[i]
            need = max(9, ceil(Fraction(raw * fps, 1000)))
            exp = 8 * ceil(Fraction(need - 1, 8)) + 1
            assert c.num_frames == exp


# ---------------- свойства ----------------

@pytest.mark.parametrize("fps", [8, 16, 24, 25, 30, 60])
@pytest.mark.parametrize("rule", [R81, {"kind": "any", "min_frames": 1},
                                  {"kind": "nk+1", "n": 4, "min_frames": 5}])
def test_property_clip_not_shorter_and_valid(fps, rule):
    rnd = random.Random(fps)
    for raw in list(range(0, 400)) + [rnd.randint(400, 60000) for _ in range(2000)]:
        f = frames_for_rule(raw, fps, rule)
        clip_ms = (f * 2000 + fps) // (2 * fps)
        assert clip_ms >= raw
        assert f >= rule.get("min_frames", 1)
        # допустимость: повторное округление ничего не меняет
        assert frames_for_rule((f * 1000) // fps, fps, rule) <= f
        if rule["kind"] == "8k+1":
            assert valid_8k1(f)
        if rule["kind"] == "nk+1":
            assert (f - 1) % 4 == 0
        # минимальность: на одно допустимое значение меньше уже не хватает
        if f > rule.get("min_frames", 1) and rule["kind"] != "any":
            step = 8 if rule["kind"] == "8k+1" else 4
            assert (f - step) * 1000 < raw * fps or f - step < rule["min_frames"]


def test_property_on_compute_timings():
    rnd = random.Random(5)
    starts = [0]
    for _ in range(500):
        starts.append(starts[-1] + rnd.randint(1, 15000))
    segs = [seg(i + 1, s, s + rnd.randint(0, 3000)) for i, s in enumerate(starts)]
    for c in run(segs):
        assert c.clip_ms >= c.raw_ms
        assert valid_8k1(c.num_frames)
