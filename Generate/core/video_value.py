"""Оценка video_value (0–100) для сегментов через Gemini (STEP 3).

Модуль не импортирует клиент Gemini: отправитель передаётся параметром send(prompt) -> str.
Чистая логика: промпт, разбор ответа, запись результата в JSON.
"""
from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Mapping, Sequence

try:
    from .log_setup import KEY_EVENT_EXTRA
except ImportError:  # запуск без пакета (плоская раскладка)
    try:
        from log_setup import KEY_EVENT_EXTRA  # type: ignore
    except ImportError:
        KEY_EVENT_EXTRA = {"console_key": True}

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
    """Builds the English instruction for Gemini: rate the video_value (0-100) of each segment.

    The segment text is passed unchanged, whatever language the subtitles are in.
    """
    # инструкция на английском; текст сегментов добавляется как есть, на любом языке
    lines = [
        "You rate how well each segment of a video suits being shown as a moving video",
        "rather than as a static photo.",
        "",
        "For every segment give an integer video_value from 0 to 100:",
        "- movement or action: 70–100;",
        "- static frame: 0–30;",
        "- mixed case: 31–69.",
        "",
        "Return ONLY a JSON list of objects of the form {\"id\": <number>, \"video_value\": <integer>},",
        "one object for each segment number. No text before or after the JSON.",
        "",
        "Segments:",
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


def build_i2v_value_prompt(segments: Sequence[Segment], photo_prompts: Mapping[int, str]) -> str:
    """Builds the English instruction for LLM: rate I2V animation suitability (0-100) from SRT + photo prompt."""
    lines = [
        "You rate how well each 2D cartoon still frame suits being animated as a short image-to-video (I2V) clip",
        "rather than staying a static photo, based on BOTH the subtitle text and the already-written photo prompt.",
        "",
        "For every segment give an integer video_value from 0 to 100:",
        "- high value (70-100): character action (walking, running, climbing, cooking, working, pointing, reacting), "
        "personified microbes or crops interacting, macro hands peeling/slicing/handling objects, moving animals;",
        "- low value (0-30): static text or number cards, checklists, flat maps, timelines, charts, split A|B diagrams, "
        "plain inanimate object arrangements with no natural motion;",
        "- medium value (31-69): calm character pose or mild environmental motion (leaves swaying, balance scale tipping).",
        "",
        "Return ONLY a JSON list of objects of the form {\"id\": <number>, \"video_value\": <integer>},",
        "one object for each segment number. No text before or after the JSON.",
        "",
        "Segments:",
    ]
    for seg in sorted(segments, key=lambda s: (s.start, s.id)):
        photo = " ".join(str(photo_prompts.get(seg.id, "")).split())
        lines.append(
            f"id {seg.id} ({seg.start:.2f}–{seg.end:.2f} s, {seg.duration:.2f} s) | SRT: {seg.text} | Photo: {photo}"
        )
    return "\n".join(lines)


def _score_i2v_chunk(
    no: int,
    total: int,
    chunk: Sequence[Segment],
    photo_prompts: Mapping[int, str],
    send: Callable[[str], str],
) -> dict:
    prompt = build_i2v_value_prompt(chunk, photo_prompts)
    expected = [s.id for s in chunk]
    parsed = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log.info("Оценка I2V video_value: батч %d из %d, сегменты %s–%s, попытка %d",
                 no, total, expected[0], expected[-1], attempt)
        try:
            text = send(prompt)
            parsed = parse_value_response(text, expected)
            break
        except Exception:
            continue
    if parsed is None:
        ids = ", ".join(str(i) for i in expected)
        raise ValueError(f"Не удалось получить I2V video_value для сегментов: {ids}")
    log.info("Оценка I2V video_value: батч %d/%d готов (сегменты %s–%s, результатов %d)",
             no, total, expected[0], expected[-1], len(parsed), extra=KEY_EVENT_EXTRA)
    return parsed


def score_i2v_segments(
    segments: Sequence[Segment],
    photo_prompts: Mapping[int, str],
    send: Callable[[str], str],
    workers: int = 1,
    min_anim_sec: float = 0.0,
) -> dict:
    """Оценивает I2V video_value только для пригодных сегментов (duration >= min_anim_sec), по фото-промпту + SRT."""
    eligible = [s for s in segments if s.duration >= min_anim_sec - 1e-9]
    if not eligible:
        return {}
    ordered = sorted(eligible, key=lambda s: (s.start, s.id))
    chunks = [ordered[i:i + SCORE_BATCH_SIZE] for i in range(0, len(ordered), SCORE_BATCH_SIZE)]
    total = len(chunks)
    n_workers = max(1, min(workers, total))
    t0 = time.monotonic()
    log.info("=== Оценка I2V video_value: сегментов %d, батчей %d (по %d), потоков %d ===",
             len(ordered), total, SCORE_BATCH_SIZE, n_workers, extra=KEY_EVENT_EXTRA)
    result: dict = {}
    if n_workers == 1:
        for no, chunk in enumerate(chunks, start=1):
            result.update(_score_i2v_chunk(no, total, chunk, photo_prompts, send))
        log.info("=== Оценка I2V video_value готова: %d сегментов за %.1f с ===",
                 len(result), time.monotonic() - t0, extra=KEY_EVENT_EXTRA)
        return result
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_score_i2v_chunk, no, total, chunk, photo_prompts, send)
                   for no, chunk in enumerate(chunks, start=1)]
        try:
            for fut in futures:
                result.update(fut.result())
        except ValueError:
            for fut in futures:
                fut.cancel()
            raise
    log.info("=== Оценка I2V video_value готова: %d сегментов за %.1f с ===",
             len(result), time.monotonic() - t0, extra=KEY_EVENT_EXTRA)
    return result


def write_distribution(
    path: str | Path,
    result: Mapping,
    segments: Sequence[Segment],
    values: Mapping[int, int],
) -> None:
    """Записывает распределение в JSON.

    Записи в порядке start: id, start, end, duration, video_value, mode ("video" или "photo").
    При наличии I2V-полей в result также записывает intro_video_percent, after_intro_video_percent,
    min_anim_sec и eligible (в корне и для каждого сегмента).
    video_value берётся из values; если значения нет, записывается null.
    """
    video_ids = set(result["video"])
    has_eligible = "eligible" in result
    eligible_ids = set(result["eligible"]) if has_eligible else set()
    records = []
    for seg in sorted(segments, key=lambda s: (s.start, s.id)):
        rec = {
            "id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "duration": seg.duration,
            "video_value": values.get(seg.id),
            "mode": "video" if seg.id in video_ids else "photo",
        }
        if has_eligible:
            rec["eligible"] = seg.id in eligible_ids
        records.append(rec)
    payload = {
        "budget_seconds": result["budget_seconds"],
        "video_seconds": result["video_seconds"],
        "intro_seconds_used": result["intro_seconds_used"],
    }
    for key in (
        "total_video_percent",
        "intro_video_percent",
        "middle_video_percent",
        "after_half_video_percent",
        "after_intro_video_percent",
        "half_seconds",
        "min_anim_sec",
        "eligible",
    ):
        if key in result:
            payload[key] = result[key]
    payload["segments"] = records
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
