"""Оптимизация SRT для ниши картунов: склейка коротких реплик через ДП (C5 + C4).

Чистый модуль: только стандартная библиотека, без сети и БД.
Алгоритм повторяет подход Download/normalize_srt.py с настраиваемыми границами
(по умолчанию [2.5, 5.0] с, цель 4.5 с):
  1. Пустые реплики разделяют текстовые блоки и не склеиваются.
  2. В каждом блоке глобальное ДП (C5) минимизирует сумму квадратов отклонения
     от target_sec + штраф за разрыв не в конце предложения.
  3. Локальное уточнение (C4) сдвигает внутренние границы на ±1 реплику при улучшении.
  4. При невозможности строгого разбиения используется жадный фолбэк.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

try:
    from .srt_parser import Segment, parse_srt, parse_srt_text
except ImportError:
    from srt_parser import Segment, parse_srt, parse_srt_text  # type: ignore

OPTIMIZED_SRT_FILENAME = "final_optimized.srt"

DEFAULT_MIN_SEC = 2.5
DEFAULT_MAX_SEC = 5.0
DEFAULT_TARGET_SEC = 4.5
DEFAULT_SENTENCE_PENALTY = 1.5
UNDER_MIN_PENALTY = 1000.0
REFINE_PASSES = 20
COST_EPS = 1e-9

SENTENCE_CLOSERS = "\"'\u00bb\u201d\u2019)]"
SENTENCE_END_CHARS = ".!?\u2026"


@dataclass(frozen=True)
class OptimizeStats:
    input_count: int
    output_count: int
    min_sec: float
    median_sec: float
    max_sec: float
    out_of_range: int
    warnings: tuple[str, ...]


def _strip_closers(text: str) -> str:
    return text.rstrip(SENTENCE_CLOSERS + " ")


def ends_sentence(text: str) -> bool:
    core = _strip_closers(text)
    return bool(core) and core[-1] in SENTENCE_END_CHARS


def ms_to_ts(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms_rem = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms_rem:03d}"


def validate_params(min_sec: float, target_sec: float, max_sec: float) -> None:
    if not (math.isfinite(min_sec) and math.isfinite(target_sec) and math.isfinite(max_sec)):
        raise ValueError("Параметры оптимизации SRT должны быть конечными числами.")
    if not (min_sec > 0 and target_sec > 0 and max_sec > 0):
        raise ValueError("Все параметры оптимизации SRT должны быть > 0.")
    if not (min_sec <= target_sec <= max_sec):
        raise ValueError(
            f"Нужно min_sec <= target_sec <= max_sec, получено {min_sec} / {target_sec} / {max_sec}."
        )


def _seg_cost(
    block: Sequence[Segment],
    s: int,
    k: int,
    min_ms: int,
    max_ms: int,
    target_sec: float,
    sentence_penalty: float,
) -> float:
    dur_ms = block[k - 1].end_ms - block[s].start_ms
    if dur_ms > max_ms:
        # Одиночная реплика длиннее потолка допускается как крайний случай (не склеивается с другими).
        if k == s + 1:
            over = (dur_ms - max_ms) / 1000.0
            return UNDER_MIN_PENALTY * 10.0 + over * over
        return math.inf
    dur = dur_ms / 1000.0
    cost = (target_sec - dur) ** 2
    if dur_ms < min_ms:
        # Мягкий высокий штраф: используется только если строгое разбиение [min, max] невозможно,
        # чтобы одна «неудобная» реплика не ломала глобальное ДП для всего файла.
        deficit = (min_ms - dur_ms) / 1000.0
        cost += UNDER_MIN_PENALTY + 100.0 * (deficit ** 2)
    if not ends_sentence(block[k - 1].text):
        cost += sentence_penalty
    return cost


def _dp_partition(
    block: Sequence[Segment],
    min_ms: int,
    max_ms: int,
    target_sec: float,
    sentence_penalty: float,
) -> tuple[Optional[list[tuple[int, int]]], int]:
    n = len(block)
    best = [math.inf] * (n + 1)
    back = [-1] * (n + 1)
    best[0] = 0.0
    for s in range(n):
        if best[s] == math.inf:
            continue
        k = s + 1
        while k <= n:
            if k > s + 1 and block[k - 1].end_ms - block[s].start_ms > max_ms:
                break
            c = best[s] + _seg_cost(block, s, k, min_ms, max_ms, target_sec, sentence_penalty)
            if c < best[k]:
                best[k] = c
                back[k] = s
            k += 1
    reach = max(k for k in range(n + 1) if best[k] < math.inf)
    if best[n] == math.inf:
        return None, reach
    bounds: list[tuple[int, int]] = []
    k = n
    while k > 0:
        s = back[k]
        bounds.append((s, k))
        k = s
    bounds.reverse()
    return bounds, n


def _refine(
    block: Sequence[Segment],
    bounds: list[tuple[int, int]],
    min_ms: int,
    max_ms: int,
    target_sec: float,
    sentence_penalty: float,
) -> list[tuple[int, int]]:
    segs = list(bounds)
    for _ in range(REFINE_PASSES):
        changed = False
        for t in range(len(segs) - 1):
            a0, b = segs[t]
            _, c = segs[t + 1]
            base = (
                _seg_cost(block, a0, b, min_ms, max_ms, target_sec, sentence_penalty)
                + _seg_cost(block, b, c, min_ms, max_ms, target_sec, sentence_penalty)
            )
            best_val, best_b = base - COST_EPS, None
            for nb in (b - 1, b + 1):
                if nb <= a0 or nb >= c:
                    continue
                val = (
                    _seg_cost(block, a0, nb, min_ms, max_ms, target_sec, sentence_penalty)
                    + _seg_cost(block, nb, c, min_ms, max_ms, target_sec, sentence_penalty)
                )
                if val < best_val:
                    best_val, best_b = val, nb
            if best_b is not None:
                segs[t] = (a0, best_b)
                segs[t + 1] = (best_b, c)
                changed = True
        if not changed:
            break
    return segs


def _greedy_bounds(block: Sequence[Segment], min_ms: int, max_ms: int) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    s = 0
    n = len(block)
    for k in range(n):
        if k > s and block[k].end_ms - block[s].start_ms > max_ms:
            bounds.append((s, k))
            s = k
        if block[k].end_ms - block[s].start_ms >= min_ms:
            bounds.append((s, k + 1))
            s = k + 1
    if s < n:
        if bounds and block[n - 1].end_ms - block[bounds[-1][0]].start_ms <= max_ms:
            bounds[-1] = (bounds[-1][0], n)
        else:
            bounds.append((s, n))
    return bounds


def _build_segment(block: Sequence[Segment], s: int, k: int) -> Segment:
    return Segment(
        num=block[s].num,
        start_ms=block[s].start_ms,
        end_ms=block[k - 1].end_ms,
        text=" ".join(c.text for c in block[s:k]),
    )


def _partition_block(
    block: Sequence[Segment],
    min_sec: float,
    max_sec: float,
    target_sec: float,
    sentence_penalty: float,
    warn: Optional[Callable[[str], None]],
) -> list[Segment]:
    min_ms = round(min_sec * 1000)
    max_ms = round(max_sec * 1000)
    bounds, reach = _dp_partition(block, min_ms, max_ms, target_sec, sentence_penalty)
    if bounds is None:
        bad = block[min(reach, len(block) - 1)]
        if warn:
            warn(
                f"первая проблемная реплика №{bad.num}: от неё нельзя собрать кусок "
                f"{min_sec:g}-{max_sec:g} с; блок склеен жадно"
            )
        return [_build_segment(block, s, k) for s, k in _greedy_bounds(block, min_ms, max_ms)]
    bounds = _refine(block, bounds, min_ms, max_ms, target_sec, sentence_penalty)
    return [_build_segment(block, s, k) for s, k in bounds]


def optimize_segments(
    segments: Sequence[Segment],
    *,
    min_sec: float = DEFAULT_MIN_SEC,
    max_sec: float = DEFAULT_MAX_SEC,
    target_sec: float = DEFAULT_TARGET_SEC,
    sentence_penalty: float = DEFAULT_SENTENCE_PENALTY,
    warn: Optional[Callable[[str], None]] = None,
) -> list[Segment]:
    """Склеивает реплики SRT в сегменты [min_sec, max_sec] с перенумерацией 1..N."""
    validate_params(min_sec, target_sec, max_sec)
    raw_out: list[Segment] = []
    block: list[Segment] = []
    for seg in segments:
        if not seg.text:
            if block:
                raw_out.extend(
                    _partition_block(block, min_sec, max_sec, target_sec, sentence_penalty, warn)
                )
                block = []
            raw_out.append(Segment(seg.num, seg.start_ms, seg.end_ms, ""))
        else:
            block.append(seg)
    if block:
        raw_out.extend(
            _partition_block(block, min_sec, max_sec, target_sec, sentence_penalty, warn)
        )
    return [
        Segment(num=i, start_ms=s.start_ms, end_ms=s.end_ms, text=s.text)
        for i, s in enumerate(raw_out, start=1)
    ]


def render_srt(segments: Sequence[Segment]) -> str:
    return "".join(
        f"{i}\n{ms_to_ts(s.start_ms)} --> {ms_to_ts(s.end_ms)}\n{s.text}\n\n"
        for i, s in enumerate(segments, start=1)
    )


def compute_stats(
    input_count: int,
    segments: Sequence[Segment],
    *,
    min_sec: float = DEFAULT_MIN_SEC,
    max_sec: float = DEFAULT_MAX_SEC,
    warnings: Sequence[str] = (),
) -> OptimizeStats:
    min_ms = round(min_sec * 1000)
    max_ms = round(max_sec * 1000)
    durs = [s.end_ms - s.start_ms for s in segments if s.text]
    if not durs:
        return OptimizeStats(input_count, len(segments), 0.0, 0.0, 0.0, 0, tuple(warnings))
    return OptimizeStats(
        input_count=input_count,
        output_count=len(segments),
        min_sec=min(durs) / 1000.0,
        median_sec=statistics.median(durs) / 1000.0,
        max_sec=max(durs) / 1000.0,
        out_of_range=sum(d < min_ms or d > max_ms for d in durs),
        warnings=tuple(warnings),
    )


def optimize_srt_file(
    input_path: str,
    output_path: str,
    *,
    min_sec: float = DEFAULT_MIN_SEC,
    max_sec: float = DEFAULT_MAX_SEC,
    target_sec: float = DEFAULT_TARGET_SEC,
    sentence_penalty: float = DEFAULT_SENTENCE_PENALTY,
) -> OptimizeStats:
    """Читает input_path, оптимизирует весь SRT и атомарно записывает output_path."""
    cues = parse_srt(input_path)
    warns: list[str] = []
    segs = optimize_segments(
        cues,
        min_sec=min_sec,
        max_sec=max_sec,
        target_sec=target_sec,
        sentence_penalty=sentence_penalty,
        warn=warns.append,
    )
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(render_srt(segs))
        os.replace(tmp, output_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return compute_stats(len(cues), segs, min_sec=min_sec, max_sec=max_sec, warnings=warns)


def format_stats(stats: OptimizeStats, min_sec: float, target_sec: float, max_sec: float) -> str:
    return (
        f"Оптимизация SRT [{min_sec:g}..{max_sec:g} с, цель {target_sec:g} с]: "
        f"{stats.input_count} -> {stats.output_count} сегментов "
        f"(мин {stats.min_sec:.3f} с / медиана {stats.median_sec:.3f} с / макс {stats.max_sec:.3f} с, "
        f"вне диапазона {stats.out_of_range}, предупреждений {len(stats.warnings)})"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Оптимизация SRT для картунов (ДП C5 + C4)")
    ap.add_argument("input", help="Путь к исходному .srt")
    ap.add_argument("output", help="Путь к выходному .srt (например final_optimized.srt)")
    ap.add_argument("--min-sec", type=float, default=DEFAULT_MIN_SEC)
    ap.add_argument("--max-sec", type=float, default=DEFAULT_MAX_SEC)
    ap.add_argument("--target-sec", type=float, default=DEFAULT_TARGET_SEC)
    ap.add_argument("--sentence-penalty", type=float, default=DEFAULT_SENTENCE_PENALTY)
    args = ap.parse_args(argv)
    try:
        stats = optimize_srt_file(
            args.input,
            args.output,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            target_sec=args.target_sec,
            sentence_penalty=args.sentence_penalty,
        )
    except (ValueError, OSError) as exc:
        print(f"Ошибка оптимизации SRT: {exc}", file=sys.stderr)
        return 1
    for w in stats.warnings:
        print(f"ВНИМАНИЕ: {w}", file=sys.stderr)
    print(format_stats(stats, args.min_sec, args.target_sec, args.max_sec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
