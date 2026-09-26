#!/usr/bin/env python3
"""
generate_queries.py

Генерирует requests.json с поисковыми запросами (стоковые/архивные видео и фото)
для каждого сегмента SRT-файла, используя Gemini API (structured output через
response_schema).

Использование:
    python generate_queries.py --input input.srt --output requests.json

Переменные окружения:
    GEMINI_API_KEY   - обязателен, ключ Gemini API
    GENQ_MODEL       - опционально, имя модели (по умолчанию gemini-3.8-flash)
    GENQ_BATCH_SIZE  - опционально, число сегментов в одном вызове (по умолчанию 100)

Зависимости:
    pip install google-genai
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3.8-flash"
# 65 536 выходных токенов - жёсткий лимит модели. При ~150-250 токенах на один
# JSON-объект сегмента (sites/query/fallback_query/type/is_entity/entity_keywords)
# 100 сегментов на батч даёт разумный запас прочности.
DEFAULT_BATCH_SIZE = 100
# Сколько соседних сегментов ДО и ПОСЛЕ батча передавать модели только как контекст
# (без включения их номеров в схему ответа) - чтобы не терять связность сюжета
# на границе двух батчей (например, сегмент 100/101 при батче 100).
CONTEXT_WINDOW = 3
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 4
SITES = ["pexels", "pixabay", "wikimedia", "nasa", "loc"]

SYSTEM_INSTRUCTION = """\
Ты помогаешь подбирать поисковые запросы для стоковых/архивных видео и фото под сцены видеоролика.
На вход ты получаешь пронумерованные сегменты сцен (номер = порядковый номер в исходном SRT, тайминг и текст).
Для КАЖДОГО сегмента без исключения (включая пустые/немые) верни объект по заданной схеме.

Правила:
- Если сегмент про конкретного named человека, историческое событие, историческое место или документ:
  sites = ["wikimedia", "loc"], порядок = приоритет. Для космоса/астрономии добавляй "nasa" первым.
- Если сегмент - абстрактная современная сцена без привязки к конкретной реальной сущности:
  sites = ["pexels", "pixabay"].
- query всегда на английском, 5-10 слов, специфичный под ПЕРВЫЙ сайт в списке sites:
  для wikimedia/loc/nasa - точные термины, даты, имена собственные;
  для pexels/pixabay - обычные стоковые формулировки.
- fallback_query - более общая версия на английском для pexels/pixabay на случай провала
  основного поиска, или null если запасной вариант не нужен.
- is_entity = true, если сегмент содержит конкретного named человека/событие/место.
- entity_keywords - варианты написания (английский + русский) ВСЕХ сущностей сегмента,
  если их несколько - объединяй варианты всех, а не только главной. Пустой список, если is_entity=false.
- Никогда не используй в query слова "creative commons", "free", "no copyright" - это не работает
  как поисковый термин.
- Пустые/немые сегменты тоже включай в ответ с нейтральным запросом, подобранным по контексту
  соседних сегментов; номер сегмента пропускать нельзя.
- Ключи итогового JSON - номера сегментов строками ("1", "2", ...), СТРОГО равные номеру блока
  в исходном SRT, без пропусков и сдвигов.

Тебе может быть передан дополнительный КОНТЕКСТ - соседние сегменты до и/или после основного
списка, помеченные отдельным блоком "Контекст ДО" / "Контекст ПОСЛЕ". Используй его только для
понимания сюжета (например, чтобы понять, к кому относится местоимение или продолжение мысли в
текущем сегменте) - но НЕ включай номера контекстных сегментов в ответ. Отвечай строго по
сегментам, помеченным как "Сегмент N" в блоке "Сегменты, для которых нужен ответ".
"""


@dataclass
class Segment:
    index: int
    start: str
    end: str
    text: str


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

SRT_BLOCK_RE = re.compile(
    r"(?P<index>\d+)\s*\n"
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})[^\n]*\n"
    r"(?P<text>.*?)(?=\n\s*\n\d+\s*\n|\Z)",
    re.DOTALL,
)


def parse_srt(path: str) -> list[Segment]:
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    raw = raw.replace("\r\n", "\n").strip() + "\n\n"

    segments: list[Segment] = []
    for m in SRT_BLOCK_RE.finditer(raw):
        idx = int(m.group("index"))
        text = m.group("text").strip()
        text = re.sub(r"<[^>]+>", "", text)  # убрать теги форматирования типа <i>
        text = re.sub(r"\{[^}]*\}", "", text)  # убрать ASS-теги типа {\an8}
        text = re.sub(r"\s+", " ", text).strip()
        segments.append(
            Segment(index=idx, start=m.group("start"), end=m.group("end"), text=text)
        )

    if not segments:
        raise ValueError(f"Не удалось распарсить ни одного сегмента из {path}")

    segments.sort(key=lambda s: s.index)

    seen: set[int] = set()
    duplicates: set[int] = set()
    for s in segments:
        if s.index in seen:
            duplicates.add(s.index)
        seen.add(s.index)

    if duplicates:
        raise ValueError(
            f"В SRT обнаружены дублирующиеся номера сегментов: {sorted(duplicates)}. "
            f"Похоже на повреждённый файл - генерация запросов остановлена."
        )

    expected = set(range(segments[0].index, segments[-1].index + 1))
    missing = expected - seen
    if missing:
        raise ValueError(
            f"В SRT отсутствуют номера сегментов: {sorted(missing)} (диапазон "
            f"{segments[0].index}..{segments[-1].index}, распарсено {len(segments)} из "
            f"{len(expected)}). Похоже на повреждённый/не до конца распарсенный файл - "
            f"генерация запросов остановлена, requests.json не создаётся."
        )

    return segments


# ---------------------------------------------------------------------------
# Gemini schema / prompt construction
# ---------------------------------------------------------------------------

def build_schema_for_batch(seg_indices: list[int]) -> types.Schema:
    entry_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "sites": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING, enum=SITES),
            ),
            "query": types.Schema(type=types.Type.STRING),
            "fallback_query": types.Schema(type=types.Type.STRING, nullable=True),
            "type": types.Schema(type=types.Type.STRING, enum=["image", "video"]),
            "is_entity": types.Schema(type=types.Type.BOOLEAN),
            "entity_keywords": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)
            ),
        },
        required=["sites", "query", "fallback_query", "type", "is_entity", "entity_keywords"],
    )
    return types.Schema(
        type=types.Type.OBJECT,
        properties={str(i): entry_schema for i in seg_indices},
        required=[str(i) for i in seg_indices],
    )


def _format_context_line(s: Segment) -> str:
    text = s.text if s.text else "(тишина / нет текста)"
    return f"[{s.index}] {text}"


def build_prompt(
    batch: list[Segment],
    context_before: list[Segment] | None = None,
    context_after: list[Segment] | None = None,
) -> str:
    lines: list[str] = []

    if context_before:
        lines.append("### Контекст ДО (не включай эти номера в ответ, только для связности сюжета)")
        lines.extend(_format_context_line(s) for s in context_before)
        lines.append("")

    lines.append("### Сегменты, для которых нужен ответ")
    for s in batch:
        text = s.text if s.text else "(тишина / нет текста)"
        lines.append(f"### Сегмент {s.index}\nТайминг: {s.start} --> {s.end}\nТекст: {text}\n")

    if context_after:
        lines.append("### Контекст ПОСЛЕ (не включай эти номера в ответ, только для связности сюжета)")
        lines.extend(_format_context_line(s) for s in context_after)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Вызов Gemini с ретраями
# ---------------------------------------------------------------------------

def call_gemini_batch(
    client: "genai.Client",
    model: str,
    batch: list[Segment],
    context_before: list[Segment] | None = None,
    context_after: list[Segment] | None = None,
) -> dict:
    schema = build_schema_for_batch([s.index for s in batch])
    prompt = build_prompt(batch, context_before, context_after)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=schema,
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            if not response.text:
                raise ValueError("Пустой ответ от Gemini API")
            return json.loads(response.text)

        except genai_errors.ClientError as e:
            code = getattr(e, "code", None)
            if code in (400, 401, 403):
                # Невалидный ключ или некорректный запрос - ретраить бессмысленно.
                logging.error("Невалидный ключ или запрос (код %s): %s", code, e)
                raise
            last_error = e  # например 429 - имеет смысл повторить
        except genai_errors.ServerError as e:
            last_error = e
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
        except Exception as e:  # сетевые сбои и т.п.
            last_error = e

        wait = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
        logging.warning(
            "Батч [%s..%s]: попытка %s/%s не удалась (%s). Повтор через %ss.",
            batch[0].index, batch[-1].index, attempt, MAX_RETRIES, last_error, wait,
        )
        time.sleep(wait)

    raise RuntimeError(
        f"Не удалось получить ответ для батча [{batch[0].index}..{batch[-1].index}] "
        f"после {MAX_RETRIES} попыток: {last_error}"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def make_batches(
    segments: list[Segment], batch_size: int, context_window: int
) -> list[tuple[list[Segment], list[Segment], list[Segment]]]:
    """Разбивает сегменты на батчи и добавляет к каждому контекст (соседние
    сегменты до/после), которые передаются модели, но не входят в схему ответа."""
    n = len(segments)
    batches = []
    for i in range(0, n, batch_size):
        batch = segments[i : i + batch_size]
        context_before = segments[max(0, i - context_window) : i]
        context_after = segments[i + batch_size : i + batch_size + context_window]
        batches.append((batch, context_before, context_after))
    return batches


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Генерация поисковых запросов из SRT через Gemini")
    parser.add_argument("--input", default=os.environ.get("GENQ_INPUT", "input.srt"))
    parser.add_argument("--output", default=os.environ.get("GENQ_OUTPUT", "requests.json"))
    parser.add_argument(
        "--batch-size", type=int, default=int(os.environ.get("GENQ_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    )
    parser.add_argument("--model", default=os.environ.get("GENQ_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.error("Переменная окружения GEMINI_API_KEY не задана.")
        return 1

    if not os.path.isfile(args.input):
        logging.error("Входной файл не найден: %s", args.input)
        return 1

    try:
        segments = parse_srt(args.input)
    except ValueError as e:
        logging.error("Ошибка парсинга SRT: %s", e)
        return 1
    logging.info("Распарсено сегментов: %s", len(segments))

    client = genai.Client(api_key=api_key)

    results: dict[str, dict] = {}

    batches = make_batches(segments, args.batch_size, CONTEXT_WINDOW)
    for batch_num, (batch, context_before, context_after) in enumerate(batches, start=1):
        logging.info(
            "Батч %s/%s: сегменты %s..%s", batch_num, len(batches), batch[0].index, batch[-1].index
        )
        try:
            batch_result = call_gemini_batch(client, args.model, batch, context_before, context_after)
        except Exception as e:
            # Жёсткое падение: тихая деградация (заглушки на часть сегментов) хуже явной
            # ошибки, которую можно сразу увидеть в логах и перезапустить job. Частичный
            # requests.json в этом случае НЕ создаётся.
            logging.error(
                "Батч %s..%s не обработан после %s попыток: %s. Прерываю выполнение, "
                "requests.json НЕ будет записан.",
                batch[0].index, batch[-1].index, MAX_RETRIES, e,
            )
            return 1

        for seg in batch:
            key = str(seg.index)
            if key not in batch_result:
                logging.error(
                    "Модель не вернула сегмент %s в ответе батча %s..%s. Прерываю выполнение, "
                    "requests.json НЕ будет записан.",
                    key, batch[0].index, batch[-1].index,
                )
                return 1
            results[key] = batch_result[key]

        progress_pct = batch_num / len(batches) * 100
        logging.info(
            "Батч %s/%s готов (%.0f%%): %s сегментов обработано из %s",
            batch_num, len(batches), progress_pct, len(results), len(segments),
        )

    # Финальная сортировка по числовому значению ключа (на случай, если батчи
    # обрабатывались не строго по порядку) - защита от рассинхрона нумерации.
    ordered = {str(k): results[str(k)] for k in sorted(int(k) for k in results.keys())}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)

    logging.info("Готово: %s сегментов записано в %s", len(ordered), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
