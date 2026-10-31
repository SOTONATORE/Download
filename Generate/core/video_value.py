"""Оценка video_value (0–100) для сегментов через Gemini (STEP 3).

Модуль не импортирует клиент Gemini: отправитель передаётся параметром send(prompt) -> str.
Чистая логика: промпт, разбор ответа, запись результата в JSON.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

try:
    from .segment_split import Segment
except ImportError:  # запуск без пакета (плоская раскладка)
    from segment_split import Segment  # type: ignore

MAX_ATTEMPTS = 2

_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)


def build_value_prompt(segments: Sequence[Segment]) -> str:
    """Строит русскоязычную инструкцию для Gemini: оценить video_value каждого сегмента."""
    lines = [
        "Ты оцениваешь, насколько каждый сегмент видеоролика подходит для показа в виде видео,",
        "а не статичного фото.",
        "",
        "Для каждого сегмента поставь целое число video_value от 0 до 100:",
        "- движение или действие: 70–100;",
        "- статичный кадр: 0–30;",
        "- смешанный случай: 31–69.",
        "",
        "Верни ТОЛЬКО JSON-список объектов вида {\"id\": <номер>, \"video_value\": <целое число>},",
        "по одному объекту на каждый номер сегмента. Никакого текста до или после JSON.",
        "",
        "Сегменты:",
    ]
    for seg in sorted(segments, key=lambda s: (s.start, s.id)):
        lines.append(f"id {seg.id} ({seg.start:.2f}–{seg.end:.2f} с): {seg.text}")
    return "\n".join(lines)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_value_response(text: str, expected_ids: Sequence[int]) -> dict:
    """Разбирает ответ модели в словарь {id: video_value}.

    Снимает ограждение ```json, если оно есть. Требует ровно ожидаемые id,
    без дублей, и целые значения 0–100. Любая ошибка даёт ValueError на русском.
    """
    body = (text or "").strip()
    fenced = _FENCE_RE.match(body)
    if fenced:
        body = fenced.group(1).strip()

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise ValueError("Ответ модели не является корректным JSON.") from None

    if not isinstance(data, list):
        raise ValueError("Ответ модели должен быть JSON-списком объектов.")

    result: dict[int, int] = {}
    for item in data:
        if not isinstance(item, dict) or "id" not in item or "video_value" not in item:
            raise ValueError("Каждый элемент ответа должен содержать поля id и video_value.")
        seg_id, value = item["id"], item["video_value"]
        if not _is_int(seg_id):
            raise ValueError(f"Номер сегмента должен быть целым числом, получено: {seg_id!r}.")
        if not _is_int(value):
            raise ValueError(f"video_value сегмента {seg_id} должен быть целым числом, получено: {value!r}.")
        if not 0 <= value <= 100:
            raise ValueError(f"video_value сегмента {seg_id} вне диапазона 0–100: {value}.")
        if seg_id in result:
            raise ValueError(f"Номер сегмента {seg_id} повторяется в ответе модели.")
        result[seg_id] = value

    expected = set(expected_ids)
    missing = sorted(expected - set(result))
    extra = sorted(set(result) - expected)
    if missing:
        raise ValueError(f"В ответе модели нет сегментов: {missing}.")
    if extra:
        raise ValueError(f"В ответе модели лишние сегменты: {extra}.")
    return result


def score_segments(segments: Sequence[Segment], send: Callable[[str], str]) -> dict:
    """Получает video_value для всех сегментов. Не больше MAX_ATTEMPTS попыток.

    send — отправитель запроса: принимает строку промпта, возвращает текст ответа.
    Если обе попытки неудачны, бросает ValueError с сообщением о сегментах.
    """
    if not segments:
        return {}
    prompt = build_value_prompt(segments)
    expected = [s.id for s in segments]
    for _ in range(MAX_ATTEMPTS):
        try:
            text = send(prompt)
            return parse_value_response(text, expected)
        except Exception:  # любая неудача отправки или разбора: повтор, если попытки остались
            continue
    ids = ", ".join(str(i) for i in expected)
    raise ValueError(f"Не удалось получить video_value для сегментов: {ids}")


def write_distribution(
    path: str | Path,
    result: Mapping,
    segments: Sequence[Segment],
    values: Mapping[int, int],
) -> None:
    """Записывает распределение в JSON.

    Записи в порядке start: id, start, end, duration, video_value, mode ("video" или "photo").
    video_value берётся из values; если значения нет, записывается null.
    """
    video_ids = set(result["video"])
    records = []
    for seg in sorted(segments, key=lambda s: (s.start, s.id)):
        records.append({
            "id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "duration": seg.duration,
            "video_value": values.get(seg.id),
            "mode": "video" if seg.id in video_ids else "photo",
        })
    payload = {
        "budget_seconds": result["budget_seconds"],
        "video_seconds": result["video_seconds"],
        "intro_seconds_used": result["intro_seconds_used"],
        "segments": records,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
