"""Offline unit tests for optional montage_cues.json integration."""
from __future__ import annotations

import hashlib
import json
import re
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from Generate.core import cues_loader as cl_cues
from Generate.core import gemini_prompts as gp
from Generate.core import prompt_pipeline as pp
from Generate.core import segment_split as ss
from Generate.core.cues_loader import (
    ALLOWED_ACCENT_STYLES,
    ALLOWED_MOODS,
    DEFAULT_CUES_FILENAME,
    SegmentCue,
    apply_cue_video_boost,
    build_batch_cue_guide,
    format_segment_cue,
    load_montage_cues,
    map_cues_to_segments,
)
from Generate.core.segment_split import Segment as SplitSegment
from Generate.core.srt_parser import Segment as SrtSegment


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access is forbidden in unit tests")

    monkeypatch.setattr(socket.socket, "connect", boom)


# ---------------------------------------------------------------------------
# load_montage_cues
# ---------------------------------------------------------------------------

def test_load_montage_cues_valid_and_normalizes_missing_lists(tmp_path: Path):
    p = tmp_path / DEFAULT_CUES_FILENAME
    payload = {
        "schema_version": 1,
        "mood_segments": [{"start": 0.0, "end": 10.0, "mood": "tension"}],
        "hard_music_cuts": [{"start": 5.0, "duration": 1.2}],
        # ambience_segments and accent_subtitles omitted intentionally
    }
    p.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_montage_cues(p)
    assert loaded["schema_version"] == 1
    assert loaded["mood_segments"] == [{"start": 0.0, "end": 10.0, "mood": "tension"}]
    assert loaded["hard_music_cuts"] == [{"start": 5.0, "duration": 1.2}]
    assert loaded["ambience_segments"] == []
    assert loaded["accent_subtitles"] == []


@pytest.mark.parametrize(
    "raw",
    [
        "{not valid json",
        "[]",
        json.dumps({"schema_version": 2}),
        json.dumps({"schema_version": True}),
        json.dumps({"schema_version": "1"}),
        json.dumps({"schema_version": 1, "mood_segments": {"start": 0}}),
        json.dumps({"schema_version": 1, "accent_subtitles": ["bad-item"]}),
    ],
)
def test_load_montage_cues_invalid_raises_value_error(tmp_path: Path, raw: str):
    p = tmp_path / "bad_cues.json"
    p.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        load_montage_cues(p)


# ---------------------------------------------------------------------------
# map_cues_to_segments
# ---------------------------------------------------------------------------

def test_map_cues_to_srt_segments_max_overlap_and_hard_cut():
    srt_segs = [
        SrtSegment(num=1, start_ms=0, end_ms=4000, text="Opening harbor view."),
        SrtSegment(num=2, start_ms=4000, end_ms=9000, text="Sudden danger warning."),
        SrtSegment(num=3, start_ms=15000, end_ms=19000, text="Quiet epilogue with no cues."),
    ]
    cues = {
        "schema_version": 1,
        "mood_segments": [
            {"start": 0.0, "end": 5.0, "mood": "curiosity"},   # seg 1: 4s; seg 2: 1s
            {"start": 5.0, "end": 10.0, "mood": "tension"},    # seg 2: 4s (wins max overlap)
        ],
        "ambience_segments": [
            {"start": 0.0, "end": 10.0, "ambience": "ocean_port"},
        ],
        "hard_music_cuts": [
            {"start": 6.0, "duration": 0.8},                   # overlaps seg 2 only
        ],
        "accent_subtitles": [
            {"start": 4.0, "end": 5.0, "style": "neutral"},    # seg 2: 1s
            {"start": 5.0, "end": 8.5, "style": "danger"},     # seg 2: 3.5s (wins max overlap)
        ],
    }

    mapped = map_cues_to_segments(srt_segs, cues)
    assert set(mapped.keys()) == {1, 2}
    assert mapped[1] == SegmentCue(mood="curiosity", ambience="ocean_port", hard_cut=False, accent_style=None)
    assert mapped[2] == SegmentCue(mood="tension", ambience="ocean_port", hard_cut=True, accent_style="danger")
    assert 3 not in mapped


def test_map_cues_to_split_segments_and_empty_inputs():
    split_segs = [
        SplitSegment(id=10, start=0.0, end=5.0, text="Action burst."),
        SplitSegment(id=11, start=5.0, end=10.0, text="Boundary touch only."),
    ]
    cues = {
        "schema_version": 1,
        "mood_segments": [{"start": 0.0, "end": 5.0, "mood": "action"}],
        "hard_music_cuts": [{"start": 1.0, "duration": 1.0}],
        "ambience_segments": [],
        "accent_subtitles": [{"start": 0.0, "end": 5.0, "accent_style": "stat"}],
    }

    mapped = map_cues_to_segments(split_segs, cues)
    assert set(mapped.keys()) == {10}
    assert mapped[10] == SegmentCue(mood="action", ambience=None, hard_cut=True, accent_style="stat")
    assert map_cues_to_segments(split_segs, None) == {}
    assert map_cues_to_segments(split_segs, {}) == {}


# ---------------------------------------------------------------------------
# format_segment_cue & build_batch_cue_guide
# ---------------------------------------------------------------------------

def test_format_segment_cue_and_dynamic_guide_only_includes_active():
    cue = SegmentCue(mood="tension", ambience="ocean_port", hard_cut=True, accent_style="danger")
    assert format_segment_cue(cue) == "mood=tension, ambience=ocean_port, hard_cut=true, accent=danger"
    assert format_segment_cue(SegmentCue()) == ""
    assert format_segment_cue(None) == ""

    assert build_batch_cue_guide([None, SegmentCue()]) == ""

    guide = build_batch_cue_guide([cue, None])
    assert guide.startswith(
        "Cue guide (apply only to segments marked with Cue below; "
        "never write cue labels, mood names or overlay text into the image):"
    )
    assert "- mood=tension:" in guide
    assert "- accent=danger:" in guide
    assert "- hard_cut=true:" in guide
    assert "- ambience:" in guide

    # Unused moods and accent styles MUST NOT appear in the dynamic batch guide
    for unused_mood in ("curiosity", "reveal", "calm", "action"):
        assert f"mood={unused_mood}" not in guide
    for unused_accent in ("stat", "positive", "neutral"):
        assert f"accent={unused_accent}" not in guide


# ---------------------------------------------------------------------------
# gemini_prompts backward compatibility & dynamic prompt injection
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path: Path, *, batch_passage: bool, segment_cues=None, batch_size: int = 2) -> gp.PromptConfig:
    style = tmp_path / "style.md"
    style.write_text("# Style\nFlat 2D cartoon illustration.", encoding="utf-8")
    return gp.PromptConfig(
        style_file=str(style),
        media="image",
        batch_size=batch_size,
        batch_passage=batch_passage,
        segment_cues=segment_cues,
    )


@pytest.mark.parametrize("batch_passage", [False, True])
def test_gemini_prompts_backward_compat_when_cues_none_or_empty(tmp_path: Path, batch_passage: bool):
    segs = [
        SrtSegment(num=1, start_ms=0, end_ms=3000, text="A boat sails into the harbor."),
        SrtSegment(num=2, start_ms=3000, end_ms=6000, text="A crane lifts a heavy crate."),
    ]
    cfg_none = _make_cfg(tmp_path, batch_passage=batch_passage, segment_cues=None)
    cfg_empty = _make_cfg(tmp_path, batch_passage=batch_passage, segment_cues={})

    sys_none = gp.build_system_instruction(cfg_none)
    sys_empty = gp.build_system_instruction(cfg_empty)
    assert sys_none == sys_empty

    reqs_none = gp.build_requests(segs, [1, 2], cfg_none)
    reqs_empty = gp.build_requests(segs, [1, 2], cfg_empty)
    assert len(reqs_none) == len(reqs_empty) == 1
    assert reqs_none[0].system == reqs_empty[0].system
    assert reqs_none[0].user == reqs_empty[0].user
    assert "| Cue:" not in reqs_none[0].user
    assert "Cue guide" not in reqs_none[0].user


@pytest.mark.parametrize("batch_passage", [False, True])
def test_gemini_prompts_injects_per_segment_cue_and_per_batch_guide_without_touching_system(
    tmp_path: Path, batch_passage: bool
):
    segs = [
        SrtSegment(num=1, start_ms=0, end_ms=3000, text="Harbor fog rolls in."),
        SrtSegment(num=2, start_ms=3000, end_ms=6000, text="A warning siren sounds."),
        SrtSegment(num=3, start_ms=6000, end_ms=9000, text="No cue on this batch."),
    ]
    seg_cues = {
        2: SegmentCue(mood="tension", ambience="ocean_port", hard_cut=True, accent_style="danger"),
    }
    cfg_base = _make_cfg(tmp_path, batch_passage=batch_passage, segment_cues=None, batch_size=2)
    cfg_cued = _make_cfg(tmp_path, batch_passage=batch_passage, segment_cues=seg_cues, batch_size=2)

    # System instruction SHA256 MUST stay 100% identical
    h_base = hashlib.sha256(gp.build_system_instruction(cfg_base).encode("utf-8")).hexdigest()
    h_cued = hashlib.sha256(gp.build_system_instruction(cfg_cued).encode("utf-8")).hexdigest()
    assert h_base == h_cued

    reqs_base = gp.build_requests(segs, [1, 2, 3], cfg_base)
    reqs_cued = gp.build_requests(segs, [1, 2, 3], cfg_cued)
    assert len(reqs_cued) == 2

    # Batch 1 (segments 1, 2): segment 2 has Cue tag, segment 1 does not, guide lists ONLY active cues
    batch1_user = reqs_cued[0].user
    assert " | Cue: mood=tension, ambience=ocean_port, hard_cut=true, accent=danger" in batch1_user
    assert batch1_user.count(" | Cue:") == 1
    assert "- mood=tension:" in batch1_user
    assert "- accent=danger:" in batch1_user
    assert "- hard_cut=true:" in batch1_user
    assert "- ambience:" in batch1_user
    for unused in ("mood=calm", "mood=action", "mood=reveal", "mood=curiosity", "accent=stat"):
        assert unused not in batch1_user

    # Batch 2 (segment 3 only): no cues in this batch -> byte-for-byte identical to baseline
    assert reqs_cued[1].user == reqs_base[1].user
    assert "Cue guide" not in reqs_cued[1].user


# ---------------------------------------------------------------------------
# apply_cue_video_boost + select_i2v_segments + build_i2v_prompts
# ---------------------------------------------------------------------------

def test_apply_cue_video_boost_deltas_and_clamping():
    base = {1: 50, 2: 50, 3: 50, 4: 50, 5: 50, 6: 90, 7: 10, 8: 50}
    cues = {
        1: SegmentCue(hard_cut=True),                      # +25 -> 75
        2: SegmentCue(mood="action"),                      # +15 -> 65
        3: SegmentCue(mood="tension"),                     # +10 -> 60
        4: SegmentCue(mood="reveal"),                      # +5  -> 55
        5: SegmentCue(mood="calm"),                        # -15 -> 35
        6: SegmentCue(hard_cut=True, mood="action"),       # +40 -> clamped 100
        7: SegmentCue(mood="calm"),                        # -15 -> clamped 0
        8: SegmentCue(mood="curiosity"),                   # +0  -> 50
    }
    boosted = apply_cue_video_boost(base, cues)
    assert boosted == {1: 75, 2: 65, 3: 60, 4: 55, 5: 35, 6: 100, 7: 0, 8: 50}
    # Does not mutate input
    assert base[1] == 50
    assert apply_cue_video_boost(base, None) == base
    assert apply_cue_video_boost(base, {}) == base


def test_apply_cue_video_boost_changes_select_i2v_segments_winner():
    segs = [
        SplitSegment(id=1, start=0.0, end=5.0, text="Calm landscape."),
        SplitSegment(id=2, start=5.0, end=10.0, text="Pivotal turn."),
        SplitSegment(id=3, start=10.0, end=15.0, text="Neutral scene."),
    ]
    base_values = {1: 65, 2: 55, 3: 40}
    # Without cues, seg 1 (65) wins the ~1-segment budget (35% of 15s = 5.25s)
    dist_base = ss.select_i2v_segments(
        segs, base_values, intro_seconds=0.0, intro_video_percent=0.0,
        after_intro_video_percent=35.0, min_anim_sec=2.0,
    )
    assert dist_base["video"] == [1]

    # With cues, seg 1 is calm (65-15=50) and seg 2 has hard_cut+tension (55+35=90) -> seg 2 wins
    cues = {
        1: SegmentCue(mood="calm"),
        2: SegmentCue(mood="tension", hard_cut=True),
    }
    boosted = apply_cue_video_boost(base_values, cues)
    dist_cued = ss.select_i2v_segments(
        segs, boosted, intro_seconds=0.0, intro_video_percent=0.0,
        after_intro_video_percent=35.0, min_anim_sec=2.0,
    )
    assert dist_cued["video"] == [2]


def test_build_i2v_prompts_applies_segment_cues_boost(monkeypatch, tmp_path: Path):
    photo_style = tmp_path / "photo_style.md"
    photo_style.write_text("# Style\nFlat 2D cartoon.", encoding="utf-8")
    video_style = tmp_path / "video_style.md"
    video_style.write_text("# I2V Style\nAnimate gently.", encoding="utf-8")

    photo_cfg = gp.PromptConfig(style_file=str(photo_style), media="image", batch_size=10)
    video_cfg = gp.PromptConfig(style_file=str(video_style), media="video", batch_size=10)

    segments = [
        SrtSegment(num=1, start_ms=0, end_ms=5000, text="Calm landscape."),
        SrtSegment(num=2, start_ms=5000, end_ms=10000, text="Sudden hard cut turn."),
        SrtSegment(num=3, start_ms=10000, end_ms=15000, text="Neutral beat."),
    ]
    prompts_path = tmp_path / "prompts.json"
    timing = SimpleNamespace(
        fps=24, frame_rule={"kind": "8k+1"},
        clip_min_sec=2.0, clip_max_sec=8.0, gpu_max_sec=10.0, tail_pad_sec=0.5,
    )

    def fake_build_prompts(needed, *, segments, photo_nums, cfg, profile, timing, photo_profile, photo_cfg, transport,
                           gemini_options, prompts_path):
        state = {
            "srt_hash": "h1",
            "model": "m",
            "profile": profile,
            "prompts": {
                str(n): {
                    "num": n, "num_frames": 1, "prompt": f"Frame {n}.",
                    "status": "ready", "attempts": 1, "error": "", "needs_review": False, "extra": {},
                }
                for n in needed
            },
        }
        Path(prompts_path).write_text(json.dumps(state), encoding="utf-8")
        return pp.PromptRun(ready=list(needed), state=state, errors=[], abort=None)

    selected_for_i2v: list[int] = []

    def fake_generate_prompts(segs, wanted, cfg, managed, model, on_batch=None, **kw):
        selected_for_i2v.extend(wanted)
        results = [gp.PromptResult(num=n, prompt=f"Animate {n}.") for n in wanted]
        if on_batch:
            on_batch(results)
        return results

    monkeypatch.setattr(pp, "build_prompts", fake_build_prompts)
    monkeypatch.setattr(pp.gp, "generate_prompts", fake_generate_prompts)
    monkeypatch.setattr(pp.go, "ManagedTransport", lambda *a, **k: SimpleNamespace(active_model="m"))

    def fake_score_send(prompt: str) -> str:
        ids = [int(m) for m in re.findall(r"^id (\d+) \(", prompt, flags=re.M)]
        # Base scores favor seg 1 (65) over seg 2 (55)
        scores = {1: 65, 2: 55, 3: 40}
        return json.dumps([{"id": i, "video_value": scores[i]} for i in ids])

    seg_cues = {
        1: SegmentCue(mood="calm"),
        2: SegmentCue(mood="tension", hard_cut=True),
    }

    pp.build_i2v_prompts(
        [1, 2, 3],
        segments=segments,
        photo_cfg=photo_cfg,
        photo_profile="cartoon",
        video_cfg=video_cfg,
        video_profile="cartoon_i2v",
        timing=timing,
        transport=SimpleNamespace(),
        gemini_options={},
        prompts_path=prompts_path,
        intro_seconds=0.0,
        intro_video_percent=0.0,
        after_intro_video_percent=35.0,
        min_anim_sec=2.0,
        score_send=fake_score_send,
        segment_cues=seg_cues,
    )

    assert selected_for_i2v == [2]
    saved = json.loads(prompts_path.read_text(encoding="utf-8"))
    assert saved["prompts"]["1"]["i2v"] is False
    assert saved["prompts"]["2"]["i2v"] is True
    assert saved["prompts"]["3"]["i2v"] is False
