"""Распределение сегментов между видео и фото (STEP 3, PLAN_generate_photo_video).

Чистый модуль: только стандартная библиотека, без сети, файлов и БД.

Правила:
- бюджет видео = суммарная длительность всех сегментов * video_percent / 100;
- сегменты со start < intro_seconds идут в видео всегда, даже сверх бюджета;
- остальные делятся на окна по 45 с: окно = floor(start / 45);
- внутри окна сортировка по video_value по убыванию, затем по start по возрастанию;
- окна обходятся по времени; сегмент берётся, если укладывается в бюджет,
  иначе пропускается и проверяется следующий;
- всё невыбранное уходит в фото.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

WINDOW_SECONDS = 45
# Допуск на погрешность плавающей точки при сравнении с бюджетом.
_EPS = 1e-9


@dataclass(frozen=True)
class Segment:
    """Сегмент хронометража. start и end в секундах.

    В существующем srt_parser.Segment поля называются num, start_ms, end_ms;
    здесь они приведены к id, start, end (секунды) по STEP 3.
    """

    id: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_video_percent(video_percent) -> None:
    if not _is_number(video_percent) or not 0 <= video_percent <= 100:
        raise ValueError(f"video_percent должен быть числом от 0 до 100, получено: {video_percent!r}")


def _check_intro_seconds(intro_seconds) -> None:
    if not _is_number(intro_seconds) or intro_seconds < 0:
        raise ValueError(f"intro_seconds должен быть неотрицательным числом, получено: {intro_seconds!r}")


def _check_value(seg_id, value) -> None:
    if not _is_number(value) or not 0 <= value <= 100:
        raise ValueError(f"video_value сегмента {seg_id} должен быть числом от 0 до 100, получено: {value!r}")


def select_video_segments(
    segments: Sequence[Segment],
    video_values: Mapping[int, int],
    video_percent: float,
    intro_seconds: float,
) -> dict:
    """Выбирает, какие сегменты идут в видео, а какие в фото. Чистая функция.

    Возвращает словарь с ключами:
      video: id видео-сегментов в порядке start;
      photo: id фото-сегментов в порядке start;
      budget_seconds, video_seconds, intro_seconds_used.
    Бросает ValueError при неверном video_percent, неверном значении или отсутствующем значении.
    """
    _check_video_percent(video_percent)
    _check_intro_seconds(intro_seconds)

    ids = [s.id for s in segments]
    if len(set(ids)) != len(ids):
        raise ValueError("Номера сегментов должны быть уникальными.")
    for seg in segments:
        if seg.id not in video_values:
            raise ValueError(f"Нет video_value для сегмента {seg.id}.")
        _check_value(seg.id, video_values[seg.id])

    ordered = sorted(segments, key=lambda s: (s.start, s.id))
    total = sum(s.duration for s in ordered)
    budget = total * video_percent / 100

    selected: set[int] = set()
    video_seconds = 0.0

    # Вступление: всегда видео, даже сверх бюджета.
    intro = [s for s in ordered if s.start < intro_seconds]
    for seg in intro:
        selected.add(seg.id)
        video_seconds += seg.duration

    # Остальные сегменты по окнам 45 с.
    windows: dict[int, list[Segment]] = {}
    for seg in ordered:
        if seg.id in selected:
            continue
        windows.setdefault(math.floor(seg.start / WINDOW_SECONDS), []).append(seg)

    for key in sorted(windows):
        # По убыванию video_value; при равенстве раньше по времени (start).
        candidates = sorted(windows[key], key=lambda s: (-video_values[s.id], s.start, s.id))
        for seg in candidates:
            if video_seconds + seg.duration <= budget + _EPS:
                selected.add(seg.id)
                video_seconds += seg.duration

    return {
        "video": [s.id for s in ordered if s.id in selected],
        "photo": [s.id for s in ordered if s.id not in selected],
        "budget_seconds": budget,
        "video_seconds": video_seconds,
        "intro_seconds_used": intro_seconds,
    }


def all_photo(segments: Sequence[Segment]) -> dict:
    """Результат для video_model=none: все сегменты в фото, видео 0 с.

    Соответствует video_percent=0 и intro_seconds=0: бюджет 0, видео 0.
    """
    ordered = sorted(segments, key=lambda s: (s.start, s.id))
    return {
        "video": [],
        "photo": [s.id for s in ordered],
        "budget_seconds": 0.0,
        "video_seconds": 0.0,
        "intro_seconds_used": 0,
    }


def to_segments(srt_segments, missing_nums) -> list[Segment]:
    """Переводит srt_parser.Segment (num, start_ms, end_ms, text) в Segment этого модуля.

    Оставляет только номера из missing_nums; id = num, start и end переводятся из миллисекунд в секунды.
    Бросает ValueError, если пропущенного номера нет в SRT.
    """
    by_num = {s.num: s for s in srt_segments}
    wanted = sorted(set(missing_nums))
    absent = [n for n in wanted if n not in by_num]
    if absent:
        raise ValueError("Номера из missing.txt отсутствуют в SRT: " + ", ".join(map(str, absent)) + ".")
    return [
        Segment(id=s.num, start=s.start_ms / 1000, end=s.end_ms / 1000, text=s.text)
        for s in (by_num[n] for n in wanted)
    ]
