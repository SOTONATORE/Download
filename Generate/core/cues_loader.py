"""Optional montage_cues.json loader, segment mapper, and prompt/I2V cue helpers.

Stdlib-only module: maps timeline cues (mood, ambience, hard music cuts,
accent subtitles) onto SRT or segment_split segments, formats per-segment cue
tags and minimal per-batch cue guides, and adjusts I2V video_value priorities.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

DEFAULT_CUES_FILENAME = "montage_cues.json"

ALLOWED_MOODS = frozenset({"curiosity", "tension", "reveal", "calm", "action"})
ALLOWED_ACCENT_STYLES = frozenset({"neutral", "danger", "positive", "stat"})

_MOOD_ORDER = ("curiosity", "tension", "reveal", "calm", "action")
_ACCENT_ORDER = ("danger", "stat", "positive", "neutral")

_MOOD_GUIDE: dict[str, str] = {
    "curiosity": "warm pale cream or soft dusty-blue light, intriguing focal detail or visual mystery.",
    "tension": "cooler pale dusty slate or shadowed backdrop, sharp muted terracotta or deep red focal contrast.",
    "reveal": "bright clean warm ochre or spotlight cone contrast emphasizing the main discovery.",
    "calm": "soft pale dusty-blue or warm cream backdrop, balanced muted olive and sage tones.",
    "action": "dynamic diagonal composition, forward posture or crisp physical motion cue with high contrast.",
}

_ACCENT_GUIDE: dict[str, str] = {
    "danger": (
        "keep upper or side backdrop extra uncluttered for an external editing overlay; "
        "convey risk visually (e.g., warning triangle or stark contrast)."
    ),
    "stat": (
        "keep upper or side backdrop extra uncluttered for an external editing number overlay; "
        "show concrete physical scale visually."
    ),
    "positive": (
        "keep upper or side backdrop extra uncluttered for an external editing overlay; "
        "convey working solution visually."
    ),
    "neutral": "keep upper or side backdrop extra uncluttered for an external editing overlay.",
}

_HARD_CUT_GUIDE = "high-impact pivotal visual turn with unmistakable foreground focal contrast."
_AMBIENCE_GUIDE = "reflect the named ambience in the physical setting and background elements naturally."

_CUE_GUIDE_HEADER = (
    "Cue guide (apply only to segments marked with Cue below; "
    "never write cue labels, mood names or overlay text into the image):"
)

_MOOD_VIDEO_DELTA: dict[str, int] = {
    "action": 15,
    "tension": 10,
    "reveal": 5,
    "calm": -15,
}
_HARD_CUT_VIDEO_DELTA = 25

_LIST_KEYS = ("mood_segments", "hard_music_cuts", "ambience_segments", "accent_subtitles")


@dataclass(frozen=True)
class SegmentCue:
    mood: Optional[str] = None
    ambience: Optional[str] = None
    hard_cut: bool = False
    accent_style: Optional[str] = None

    def is_empty(self) -> bool:
        return not (self.mood or self.ambience or self.hard_cut or self.accent_style)


def _to_float(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _segment_bounds(seg) -> tuple[int, float, float]:
    if hasattr(seg, "num") and hasattr(seg, "start_ms") and hasattr(seg, "end_ms"):
        return int(seg.num), float(seg.start_ms) / 1000.0, float(seg.end_ms) / 1000.0
    if hasattr(seg, "id") and hasattr(seg, "start") and hasattr(seg, "end"):
        return int(seg.id), float(seg.start), float(seg.end)
    raise TypeError(f"Unsupported segment object: {seg!r}")


def load_montage_cues(path: Union[str, Path]) -> dict:
    """Load and validate a montage_cues.json file (schema_version == 1)."""
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid montage cues JSON in {file_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Montage cues root must be a JSON object, got {type(data).__name__}.")

    version = data.get("schema_version", 1)
    if isinstance(version, bool) or version != 1:
        raise ValueError(f"Unsupported montage cues schema_version: {version!r} (expected 1).")

    normalized: dict = {"schema_version": 1}
    for key in _LIST_KEYS:
        raw_list = data.get(key)
        if raw_list is None:
            normalized[key] = []
        elif not isinstance(raw_list, list):
            raise ValueError(f"Montage cues field '{key}' must be a list, got {type(raw_list).__name__}.")
        else:
            items: list[dict] = []
            for idx, item in enumerate(raw_list):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Montage cues field '{key}[{idx}]' must be an object, got {type(item).__name__}."
                    )
                items.append(dict(item))
            normalized[key] = items
    return normalized


def map_cues_to_segments(segments: Sequence, cues: Optional[dict]) -> dict[int, SegmentCue]:
    """Map timeline cues onto srt_parser.Segment or segment_split.Segment instances."""
    if not cues or not isinstance(cues, Mapping):
        return {}

    mood_entries = cues.get("mood_segments") or []
    ambience_entries = cues.get("ambience_segments") or []
    hard_cut_entries = cues.get("hard_music_cuts") or []
    accent_entries = cues.get("accent_subtitles") or []

    mapped: dict[int, SegmentCue] = {}
    for seg in segments:
        seg_num, seg_start, seg_end = _segment_bounds(seg)

        best_mood: Optional[str] = None
        best_mood_ov = 0.0
        for item in mood_entries:
            if not isinstance(item, Mapping):
                continue
            raw_mood = item.get("mood")
            if not isinstance(raw_mood, str):
                continue
            mood = raw_mood.strip().lower()
            if mood not in ALLOWED_MOODS:
                continue
            c_start = _to_float(item.get("start"))
            c_end = _to_float(item.get("end"))
            if c_start is None or c_end is None or c_end <= c_start:
                continue
            ov = _overlap(seg_start, seg_end, c_start, c_end)
            if ov > best_mood_ov:
                best_mood_ov = ov
                best_mood = mood

        best_ambience: Optional[str] = None
        best_ambience_ov = 0.0
        for item in ambience_entries:
            if not isinstance(item, Mapping):
                continue
            raw_amb = item.get("ambience")
            if raw_amb is None:
                raw_amb = item.get("name") or item.get("label")
            if not isinstance(raw_amb, str):
                continue
            amb = raw_amb.strip()
            if not amb:
                continue
            c_start = _to_float(item.get("start"))
            c_end = _to_float(item.get("end"))
            if c_start is None or c_end is None or c_end <= c_start:
                continue
            ov = _overlap(seg_start, seg_end, c_start, c_end)
            if ov > best_ambience_ov:
                best_ambience_ov = ov
                best_ambience = amb

        hard_cut = False
        for item in hard_cut_entries:
            if not isinstance(item, Mapping):
                continue
            c_start = _to_float(item.get("start"))
            if c_start is None:
                continue
            dur = _to_float(item.get("duration"))
            if dur is not None:
                c_end = c_start + max(0.0, dur)
            else:
                c_end = _to_float(item.get("end"))
                if c_end is None:
                    c_end = c_start
            if c_end > c_start:
                if _overlap(seg_start, seg_end, c_start, c_end) > 0.0:
                    hard_cut = True
                    break
            elif seg_start <= c_start < seg_end:
                hard_cut = True
                break

        best_accent: Optional[str] = None
        best_accent_ov = 0.0
        for item in accent_entries:
            if not isinstance(item, Mapping):
                continue
            if "style" in item:
                raw_style = item.get("style")
            elif "accent_style" in item:
                raw_style = item.get("accent_style")
            else:
                raw_style = "neutral"
            if not isinstance(raw_style, str):
                continue
            style = raw_style.strip().lower()
            if style not in ALLOWED_ACCENT_STYLES:
                continue
            c_start = _to_float(item.get("start"))
            c_end = _to_float(item.get("end"))
            if c_start is None or c_end is None or c_end <= c_start:
                continue
            ov = _overlap(seg_start, seg_end, c_start, c_end)
            if ov > best_accent_ov:
                best_accent_ov = ov
                best_accent = style

        cue = SegmentCue(
            mood=best_mood,
            ambience=best_ambience,
            hard_cut=hard_cut,
            accent_style=best_accent,
        )
        if not cue.is_empty():
            mapped[seg_num] = cue

    return mapped


def format_segment_cue(cue: Optional[SegmentCue]) -> str:
    """Format a SegmentCue into a compact inline tag, or '' if empty."""
    if cue is None or cue.is_empty():
        return ""
    parts: list[str] = []
    if cue.mood:
        parts.append(f"mood={cue.mood}")
    if cue.ambience:
        parts.append(f"ambience={cue.ambience}")
    if cue.hard_cut:
        parts.append("hard_cut=true")
    if cue.accent_style:
        parts.append(f"accent={cue.accent_style}")
    return ", ".join(parts)


def build_batch_cue_guide(cues_in_batch: Iterable[Optional[SegmentCue]]) -> str:
    """Build a minimal cue guide covering ONLY cue attributes active in the batch."""
    active_moods: set[str] = set()
    active_accents: set[str] = set()
    has_hard_cut = False
    has_ambience = False

    for cue in cues_in_batch:
        if cue is None or cue.is_empty():
            continue
        if cue.mood and cue.mood in _MOOD_GUIDE:
            active_moods.add(cue.mood)
        if cue.accent_style and cue.accent_style in _ACCENT_GUIDE:
            active_accents.add(cue.accent_style)
        if cue.hard_cut:
            has_hard_cut = True
        if cue.ambience:
            has_ambience = True

    if not (active_moods or active_accents or has_hard_cut or has_ambience):
        return ""

    lines = [_CUE_GUIDE_HEADER]
    for mood in _MOOD_ORDER:
        if mood in active_moods:
            lines.append(f"- mood={mood}: {_MOOD_GUIDE[mood]}")
    for accent in _ACCENT_ORDER:
        if accent in active_accents:
            lines.append(f"- accent={accent}: {_ACCENT_GUIDE[accent]}")
    if has_hard_cut:
        lines.append(f"- hard_cut=true: {_HARD_CUT_GUIDE}")
    if has_ambience:
        lines.append(f"- ambience: {_AMBIENCE_GUIDE}")
    return "\n".join(lines)


def apply_cue_video_boost(
    video_values: Mapping[int, int],
    seg_cues: Optional[Mapping[int, SegmentCue]],
) -> dict[int, int]:
    """Adjust video_value priorities using montage cues, clamped to [0, 100]."""
    if not seg_cues:
        return dict(video_values)

    boosted: dict[int, int] = {}
    for seg_id, base_val in video_values.items():
        cue = seg_cues.get(seg_id)
        if cue is None or cue.is_empty():
            boosted[seg_id] = base_val
            continue
        delta = 0
        if cue.hard_cut:
            delta += _HARD_CUT_VIDEO_DELTA
        if cue.mood:
            delta += _MOOD_VIDEO_DELTA.get(cue.mood, 0)
        boosted[seg_id] = max(0, min(100, int(base_val) + delta))
    return boosted
