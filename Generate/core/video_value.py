"""Оценка video_value (0–100) для сегментов через Gemini (STEP 3).

Модуль не импортирует клиент Gemini: отправитель передаётся параметром send(prompt) -> str.
Чистая логика: промпт, разбор ответа, запись результата в JSON.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Mapping, Sequence

try:
    from .segment_split import Segment
except ImportError:  # запуск без пакета (плоская раскладка)
    from segment_split import Segment  # type: ignore

log = logging.getLogger("Generate.video_value")

MAX_ATTEMPTS = 2
# сегментов в одном запросе оценки; большой запрос упирается в лимит вывода рассуждениями
SCORE_BATCH_SIZE = 25

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


def _score_chunk(no: int, total: int, chunk: Sequence[Segment], send: Callable[[str], str]) -> dict:
    """Оценка одного батча с повторами (MAX_ATTEMPTS). Бросает ValueError со списком номеров батча.

    no — номер батча по порядку (с 1), total — всего батчей; send может вызываться из нескольких потоков.
    """
    prompt = build_value_prompt(chunk)
    expected = [s.id for s in chunk]
    parsed = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log.info("Оценка video_value: батч %d из %d, сегменты %s–%s, попытка %d",
                 no, total, expected[0], expected[-1], attempt)
        try:
            text = send(prompt)
            parsed = parse_value_response(text, expected)
            break
        except Exception:  # любая неудача отправки или разбора: повтор, если попытки остались
            continue
    if parsed is None:
        ids = ", ".join(str(i) for i in expected)
        raise ValueError(f"Не удалось получить video_value для сегментов: {ids}")
    log.info("Оценка video_value: батч %d готов, результатов %d", no, len(parsed))
    return parsed


def score_segments(segments: Sequence[Segment], send: Callable[[str], str], workers: int = 1) -> dict:
    """Получает video_value для всех сегментов, батчами по SCORE_BATCH_SIZE.

    Не больше MAX_ATTEMPTS попыток на каждый батч.
    send — отправитель запроса: принимает строку промпта, возвращает текст ответа.
    workers — число параллельных батчей (по умолчанию 1: батчи идут в вызывающем потоке, как раньше).
    Значение больше 1 включает пул потоков; send тогда вызывается из нескольких потоков одновременно,
    поэтому отправитель должен быть потокобезопасным (ManagedTransport и KeyPool такие).
    Если обе попытки какого-либо батча неудачны, бросает ValueError с номерами сегментов этого батча;
    частичный результат не возвращается. При параллельном режиме батчи, которые ещё не начались, отменяются.
    """
    if not segments:
        return {}
    # батчи идут по времени начала, затем по номеру: порядок стабилен между запусками
    ordered = sorted(segments, key=lambda s: (s.start, s.id))
    chunks = [ordered[i:i + SCORE_BATCH_SIZE] for i in range(0, len(ordered), SCORE_BATCH_SIZE)]
    total = len(chunks)
    n_workers = max(1, min(workers, total))
    result: dict = {}
    if n_workers == 1:
        for no, chunk in enumerate(chunks, start=1):
            result.update(_score_chunk(no, total, chunk, send))
        return result
    # результаты собираются в порядке номеров батчей, а не в порядке завершения
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_score_chunk, no, total, chunk, send)
                   for no, chunk in enumerate(chunks, start=1)]
        try:
            for fut in futures:
                result.update(fut.result())
        except ValueError:
            # батч неудачен: остальные, которые ещё не начались, отменяем; частичный результат не отдаём
            for fut in futures:
                fut.cancel()
            raise
    return result


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
