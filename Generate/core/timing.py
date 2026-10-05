"""Расчёт длины клипа и числа кадров для сегментов SRT.

Чистый модуль: стандартная библиотека, без сети/БД/файлов.
Время только в целых миллисекундах, кадры считаются целочисленно.
Параметры (fps, правило кадров, пороги) приходят аргументами.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from Generate.core.srt_parser import Segment


@dataclass(frozen=True)
class ClipTiming:
    num: int
    start_ms: int
    end_ms: int
    raw_ms: int            # «нужная» длина до округления кадров (фактически использованная)
    num_frames: int        # итоговое число кадров по правилу модели
    clip_ms: int           # num_frames / fps в мс (округлено до целого)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Реестр правил допустимого числа кадров
# ---------------------------------------------------------------------------
# Правило: функция (need, rule) -> наименьшее допустимое число кадров >= need,
# где need уже не меньше rule["min_frames"]. Новое правило = одна функция
# с декоратором @frame_rule("имя").

_RULES: dict[str, Callable[[int, dict], int]] = {}


def frame_rule(kind: str):
    def deco(fn: Callable[[int, dict], int]):
        _RULES[kind] = fn
        return fn
    return deco


def _round_up_step_plus_one(need: int, step: int) -> int:
    """Наименьшее число вида step*k + 1 (k >= 0), не меньшее need."""
    if step <= 0:
        raise ValueError(f"Шаг правила кадров должен быть > 0, получено {step}.")
    k = max(0, -(-(need - 1) // step))  # ceil((need-1)/step) целочисленно
    return step * k + 1


@frame_rule("8k+1")
def _rule_8k1(need: int, rule: dict) -> int:
    # шаг 8 по умолчанию, но можно переопределить полем "n"
    return _round_up_step_plus_one(need, int(rule.get("n", 8)))


@frame_rule("nk+1")
def _rule_nk1(need: int, rule: dict) -> int:
    if "n" not in rule:
        raise ValueError('Для правила "nk+1" обязателен параметр "n".')
    return _round_up_step_plus_one(need, int(rule["n"]))


@frame_rule("any")
def _rule_any(need: int, rule: dict) -> int:
    return need


def frames_for_rule(raw_ms: int, fps: int, frame_rule: dict) -> int:
    """ceil(raw_ms*fps/1000) кадров, затем округление вверх по правилу модели."""
    if fps <= 0:
        raise ValueError(f"fps должен быть > 0, получено {fps}.")
    kind = frame_rule.get("kind")
    fn = _RULES.get(kind)
    if fn is None:
        raise ValueError(
            f"Неизвестное правило кадров: {kind!r}. Доступные: {sorted(_RULES)}."
        )
    raw = max(0, int(raw_ms))
    need = (raw * fps + 999) // 1000            # ceil без float
    need = max(need, int(frame_rule.get("min_frames", 1)), 1)
    return fn(need, frame_rule)


def _frames_to_ms(frames: int, fps: int) -> int:
    """frames*1000/fps, округление к ближайшему целому (половина вверх), без float."""
    return (frames * 2000 + fps) // (2 * fps)


def _sec_to_ms(sec: float) -> int:
    return int(round(sec * 1000))


def _fmt(ms: int) -> str:
    return f"{ms / 1000:.2f}".replace(".", ",") + " с"


def compute_timings(
    segments: list[Segment], fps: int, frame_rule: dict,
    clip_min_sec: float, clip_max_sec: float, gpu_max_sec: float,
    tail_pad_sec: float,
) -> list[ClipTiming]:
    """Длина клипа и число кадров для каждого сегмента (порядок по num)."""
    if fps <= 0:
        raise ValueError(f"fps должен быть > 0, получено {fps}.")
    segs = sorted(segments, key=lambda s: s.num)
    min_ms, max_ms, gpu_ms = map(_sec_to_ms, (clip_min_sec, clip_max_sec, gpu_max_sec))
    pad_ms = _sec_to_ms(tail_pad_sec)

    result: list[ClipTiming] = []
    for i, s in enumerate(segs):
        warns: list[str] = []
        if i + 1 < len(segs):
            raw = segs[i + 1].start_ms - s.start_ms
        else:
            raw = (s.end_ms - s.start_ms) + pad_ms

        if raw <= 0:
            own = s.end_ms - s.start_ms
            fallback = own if own > 0 else min_ms
            warns.append(
                f"некорректная длина (начало следующего не позже начала текущего, "
                f"{_fmt(raw)}); взято {_fmt(fallback)}"
            )
            raw = fallback

        if raw < min_ms:
            warns.append(f"короче минимума ({_fmt(raw)} < {_fmt(min_ms)})")
        if raw > max_ms:
            warns.append(
                f"длиннее рекомендуемого максимума ({_fmt(raw)} > {_fmt(max_ms)})"
            )
        if raw > gpu_ms:
            warns.append(
                f"выше максимума, который тянет карта ({_fmt(raw)} > {_fmt(gpu_ms)})"
            )

        frames = frames_for_rule(raw, fps, frame_rule)
        result.append(ClipTiming(
            num=s.num, start_ms=s.start_ms, end_ms=s.end_ms, raw_ms=raw,
            num_frames=frames, clip_ms=_frames_to_ms(frames, fps), warnings=warns,
        ))
    return result
