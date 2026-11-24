"""Распределение сегментов между видео и фото (STEP 3, PLAN_generate_photo_video).

Чистый модуль: только стандартная библиотека, без сети, файлов и БД.

Правила:
- бюджет видео = суммарная длительность всех сегментов * video_percent / 100;
- сегменты со start < intro_seconds идут в видео всегда, даже сверх бюджета;
- остальные делятся на окна по 45 с: окно = floor(start / 45);
- внутри окна сортировка по video_value по убыванию, затем по start по возрастанию;
- квота задаётся на каждое окно 45 с: квота окна = суммарная длительность всех сегментов окна
  * video_percent / 100; вступление, попавшее в окно, занимает часть этой квоты;
  сегмент берётся, если укладывается в квоту своего окна, иначе пропускается и проверяется следующий;
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

    # Окна 45 с: все сегменты окна (включая вступление) задают квоту окна.
    windows: dict[int, list[Segment]] = {}
    for seg in ordered:
        windows.setdefault(math.floor(seg.start / WINDOW_SECONDS), []).append(seg)

    for key in sorted(windows):
        members = windows[key]
        quota = sum(s.duration for s in members) * video_percent / 100
        # Вступление в окне уже занимает часть квоты.
        window_video = sum(s.duration for s in members if s.id in selected)
        # По убыванию video_value; при равенстве раньше по времени (start), затем по id.
        candidates = sorted((s for s in members if s.id not in selected),
                            key=lambda s: (-video_values[s.id], s.start, s.id))
        for seg in candidates:
            if window_video + seg.duration <= quota + _EPS:
                selected.add(seg.id)
                window_video += seg.duration
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


# Максимальный перенос остатка квоты между окнами 45 с (сек): предотвращает всплески в соседних окнах.
_I2V_MAX_CARRY_SEC = 3.0


def select_i2v_segments(
    segments: Sequence[Segment],
    video_values: Mapping[int, int],
    intro_seconds: float,
    intro_video_percent: float,
    after_intro_video_percent: float = 10.0,
    min_anim_sec: float = 2.0,
    window_seconds: float = WINDOW_SECONDS,
    *,
    total_video_percent: float | None = None,
    after_half_video_percent: float | None = None,
) -> dict:
    """Равномерный отбор сегментов для I2V-анимации в нише картунов. Чистая функция.

    Правила:
    - сегменты с duration < min_anim_sec всегда остаются фото (не допускаются к анимации);
    - в 3-зонном режиме (заданы total_video_percent или after_half_video_percent):
      общий бюджет = total_dur * total_video_percent / 100;
      зона 1 (start < intro_seconds) -> intro_video_percent;
      зона 3 (start >= 50% ролика) -> after_half_video_percent;
      зона 2 (между вступлением и 50% ролика) -> весь остаток общего бюджета;
    - в 2-зонном режиме (legacy, оба новых параметра None):
      вступление -> intro_video_percent, далее -> after_intro_video_percent;
    - внутри каждого окна window_seconds (45 с) квота считается локально по долям зоны,
      с минимальным ограниченным переносом остатка (не более _I2V_MAX_CARRY_SEC), без всплесков;
    - при низкой доле (< 50%) применяется мягкая защита от выбора подряд идущих сегментов.
    """
    _check_intro_seconds(intro_seconds)
    _check_video_percent(intro_video_percent)
    _check_video_percent(after_intro_video_percent)
    three_zone = total_video_percent is not None or after_half_video_percent is not None
    if three_zone:
        tot_pct = 20.0 if total_video_percent is None else float(total_video_percent)
        tail_pct = 5.0 if after_half_video_percent is None else float(after_half_video_percent)
        _check_video_percent(tot_pct)
        _check_video_percent(tail_pct)
    else:
        tot_pct = None
        tail_pct = None
    if not _is_number(min_anim_sec) or min_anim_sec < 0:
        raise ValueError(f"min_anim_sec должен быть неотрицательным числом, получено: {min_anim_sec!r}")
    if not _is_number(window_seconds) or window_seconds <= 0:
        raise ValueError(f"window_seconds должен быть положительным числом, получено: {window_seconds!r}")

    ids = [s.id for s in segments]
    if len(set(ids)) != len(ids):
        raise ValueError("Номера сегментов должны быть уникальными.")

    ordered = sorted(segments, key=lambda s: (s.start, s.id))
    eligible_ids = {s.id for s in ordered if s.duration >= min_anim_sec - _EPS}
    for seg in ordered:
        if seg.id in eligible_ids:
            if seg.id not in video_values:
                raise ValueError(f"Нет video_value для пригодного сегмента {seg.id}.")
            _check_value(seg.id, video_values[seg.id])

    intro_rate = intro_video_percent / 100.0
    half_seconds = ((ordered[0].start + ordered[-1].end) / 2.0) if ordered else 0.0

    if three_zone:
        assert tot_pct is not None and tail_pct is not None
        tail_rate = tail_pct / 100.0

        def _seg_zone(seg: Segment) -> int:
            if seg.start < intro_seconds:
                return 0
            if seg.start >= half_seconds:
                return 2
            return 1

        total_dur = sum(s.duration for s in ordered)
        intro_dur = sum(s.duration for s in ordered if _seg_zone(s) == 0)
        middle_dur = sum(s.duration for s in ordered if _seg_zone(s) == 1)
        tail_dur = sum(s.duration for s in ordered if _seg_zone(s) == 2)
        total_budget = total_dur * (tot_pct / 100.0)
        intro_budget = intro_dur * intro_rate
        tail_budget = tail_dur * tail_rate
        middle_budget = max(0.0, total_budget - intro_budget - tail_budget)
        middle_rate = min(1.0, middle_budget / middle_dur) if middle_dur > _EPS else 0.0
        middle_video_percent = middle_rate * 100.0

        def _seg_rate(seg: Segment) -> float:
            z = _seg_zone(seg)
            if z == 0:
                return intro_rate
            if z == 1:
                return middle_rate
            return tail_rate
    else:
        after_rate = after_intro_video_percent / 100.0
        middle_video_percent = after_intro_video_percent

        def _seg_zone(seg: Segment) -> int:
            return 0 if seg.start < intro_seconds else 1

        def _seg_rate(seg: Segment) -> float:
            return intro_rate if seg.start < intro_seconds else after_rate

    budget = sum(s.duration * _seg_rate(s) for s in ordered)
    pos_by_id = {s.id: idx for idx, s in enumerate(ordered)}
    selected: set[int] = set()
    video_seconds = 0.0

    windows: dict[int, list[Segment]] = {}
    for seg in ordered:
        windows.setdefault(math.floor(seg.start / window_seconds), []).append(seg)

    carry = 0.0
    prev_zone: int | None = None

    for key in sorted(windows):
        members = windows[key]
        win_dur = sum(s.duration for s in members)
        window_target = sum(s.duration * _seg_rate(s) for s in members)
        avg_rate = (window_target / win_dur) if win_dur > 0 else 0.0

        # Сброс остатка при смене зоны, чтобы квота одной зоны не утекала в соседнюю.
        cur_zone = _seg_zone(members[0])
        if prev_zone is not None and cur_zone != prev_zone:
            carry = 0.0
        prev_zone = cur_zone

        if window_target <= _EPS:
            carry = 0.0
            continue

        eligible_members = [s for s in members if s.id in eligible_ids]
        if not eligible_members:
            continue

        # 100% зона: выбираем все пригодные сегменты окна без ограничений.
        if avg_rate >= 1.0 - _EPS:
            for seg in eligible_members:
                selected.add(seg.id)
                video_seconds += seg.duration
            carry = 0.0
            continue

        # Ограниченный перенос остатка (не больше _I2V_MAX_CARRY_SEC и не больше половины квоты окна).
        carry_cap = min(_I2V_MAX_CARRY_SEC, max(window_target * 0.75, min_anim_sec))
        effective_quota = max(0.0, window_target + carry)
        candidates = sorted(eligible_members, key=lambda s: (-video_values[s.id], s.start, s.id))

        window_video = 0.0
        avoid_adjacent = avg_rate < 0.5

        def _is_adjacent(seg: Segment) -> bool:
            idx = pos_by_id[seg.id]
            if idx > 0 and ordered[idx - 1].id in selected:
                return True
            if idx + 1 < len(ordered) and ordered[idx + 1].id in selected:
                return True
            return False

        # Первый проход: при низкой доле избегаем соседних сегментов.
        passes = (True, False) if avoid_adjacent else (False,)
        for skip_adj in passes:
            for seg in candidates:
                if seg.id in selected:
                    continue
                if skip_adj and _is_adjacent(seg):
                    continue
                if window_video + seg.duration <= effective_quota + _EPS:
                    selected.add(seg.id)
                    window_video += seg.duration
                    video_seconds += seg.duration

        carry = max(-carry_cap, min(carry_cap, effective_quota - window_video))

    res = {
        "video": [s.id for s in ordered if s.id in selected],
        "photo": [s.id for s in ordered if s.id not in selected],
        "eligible": [s.id for s in ordered if s.id in eligible_ids],
        "budget_seconds": budget,
        "video_seconds": video_seconds,
        "intro_seconds_used": intro_seconds,
        "intro_video_percent": intro_video_percent,
        "after_intro_video_percent": after_intro_video_percent,
        "min_anim_sec": min_anim_sec,
    }
    if three_zone:
        res["total_video_percent"] = tot_pct
        res["after_half_video_percent"] = tail_pct
        res["middle_video_percent"] = round(middle_video_percent, 2)
        res["half_seconds"] = half_seconds
    return res
