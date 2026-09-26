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
# JSON-объект сегмента (segment_index/sites/query/fallback_query/type/is_entity/
# entity_keywords) 100 сегментов на батч даёт разумный запас прочности.
DEFAULT_BATCH_SIZE = 100
# Сколько соседних сегментов ДО и ПОСЛЕ батча передавать модели только как контекст
# (без создания для них отдельных объектов в ответе) - чтобы не терять связность
# сюжета на границе двух батчей (например, сегмент 100/101 при батче 100).
CONTEXT_WINDOW = 3
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 4
SITES = ["pexels", "pixabay", "wikimedia", "nasa", "loc"]

SYSTEM_INSTRUCTION = """\
Ты помогаешь подбирать поисковые запросы для стоковых/архивных видео и фото под сцены видеоролика.
На вход ты получаешь пронумерованные сегменты сцен (номер = порядковый номер в исходном SRT, тайминг и текст).

Верни JSON-МАССИВ объектов - РОВНО по одному объекту на каждый сегмент из блока "Сегменты, для
которых нужен ответ" (включая пустые/немые сегменты), без пропусков и без дублей. Каждый объект
обязан содержать поле segment_index (целое число, точно равное номеру из "### Сегмент N") и
остальные поля по схеме. Порядок объектов в массиве не важен.

Правила заполнения остальных полей:
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

Тебе может быть передан дополнительный КОНТЕКСТ - соседние сегменты до и/или после основного
списка, помеченные отдельным блоком "Контекст ДО" / "Контекст ПОСЛЕ". Используй его только для
понимания сюжета (например, чтобы понять, к кому относится местоимение или продолжение мысли в
текущем сегменте) - но НЕ создавай для этих контекстных сегментов отдельные объекты в ответе.
Отвечай строго по сегментам, помеченным как "Сегмент N" в блоке "Сегменты, для которых нужен ответ".
"""

# Схема одного элемента массива (статическая - не зависит от размера батча).
SEGMENT_ENTRY_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "segment_index": types.Schema(type=types.Type.INTEGER),
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
    required=[
        "segment_index",
        "sites",
        "query",
        "fallback_query",
        "type",
        "is_entity",
        "entity_keywords",
    ],
)

# Ответ - JSON-массив таких объектов (официально задокументированный Gemini-паттерн
# для структурированного вывода переменной длины, в отличие от объекта с динамическими
# ключами-номерами, который, судя по всему, и приводил к 400 INVALID_ARGUMENT).
RESPONSE_SCHEMA = types.Schema(type=types.Type.ARRAY, items=SEGMENT_ENTRY_SCHEMA)


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
# Prompt construction
# ---------------------------------------------------------------------------

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
        lines.append("### Контекст ДО (не создавай для этих номеров объекты в ответе, только для связности сюжета)")
        lines.extend(_format_context_line(s) for s in context_before)
        lines.append("")

    lines.append("### Сегменты, для которых нужен ответ")
    for s in batch:
        text = s.text if s.text else "(тишина / нет текста)"
        lines.append(f"### Сегмент {s.index}\nТайминг: {s.start} --> {s.end}\nТекст: {text}\n")

    if context_after:
        lines.append("### Контекст ПОСЛЕ (не создавай для этих номеров объекты в ответе, только для связности сюжета)")
        lines.extend(_format_context_line(s) for s in context_after)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Вызов Gemini с ретраями
# ---------------------------------------------------------------------------

_STATUS_CODE_RE = re.compile(r"\b([1-5]\d{2})\b")


def _extract_status_code(e: Exception) -> Optional[int]:
    """Пытается достать HTTP-код из исключения даже если атрибут .code недоступен
    в текущей версии SDK - парсим из текстового представления ошибки."""
    code = getattr(e, "code", None)
    if isinstance(code, int):
        return code
    m = _STATUS_CODE_RE.search(str(e))
    return int(m.group(1)) if m else None


def call_gemini_batch(
    client: "genai.Client",
    model: str,
    batch: list[Segment],
    context_before: list[Segment] | None = None,
    context_after: list[Segment] | None = None,
) -> dict:
    prompt = build_prompt(batch, context_before, context_after)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    expected_indices = {s.index for s in batch}
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            if not response.text:
                raise ValueError("Пустой ответ от Gemini API")

            parsed = json.loads(response.text)
            if not isinstance(parsed, list):
                raise ValueError(
                    f"Ожидался JSON-массив, получен {type(parsed).__name__}: {str(parsed)[:300]}"
                )

            result: dict[str, dict] = {}
            for item in parsed:
                if "segment_index" not in item:
                    raise ValueError(f"В элементе ответа нет segment_index: {item}")
                idx = item.pop("segment_index")
                result[str(idx)] = item

            got_indices = {int(k) for k in result.keys()}
            missing = expected_indices - got_indices
            extra = got_indices - expected_indices
            if missing or extra:
                raise ValueError(
                    f"Несовпадение номеров сегментов в ответе батча [{batch[0].index}.."
                    f"{batch[-1].index}]: не хватает {sorted(missing)}, лишние {sorted(extra)}"
                )

            return result

        except genai_errors.ClientError as e:
            code = _extract_status_code(e)
            if code == 429:
                last_error = e  # лимит запросов - имеет смысл повторить
            else:
                # Любой другой 4xx (400/401/403/404/...) - структурная ошибка запроса
                # или ключа, ретраить бессмысленно. Печатаем полное тело ответа API.
                logging.error(
                    "Клиентская ошибка Gemini API (код %s), запрос некорректен либо ключ "
                    "невалиден - НЕ ретраю. Полный ответ API: %s",
                    code, e,
                )
                raise
        except genai_errors.ServerError as e:
            last_error = e  # 5xx - транзиентная ошибка сервера, есть смысл повторить
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
        except Exception as e:  # сетевые сбои и т.п.
            last_error = e

        if attempt < MAX_RETRIES:
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


def preflight_check(client: "genai.Client", model: str) -> None:
    """Быстрая проверка перед основным циклом, чтобы отличить проблему с ключом/моделью
    от проблемы конкретно в конструкции response_schema - и не сжигать батчи впустую."""
    logging.info("Preflight 1/2: проверка ключа и модели (без response_schema)...")
    try:
        response = client.models.generate_content(model=model, contents="Ответь одним словом: OK")
        if not response.text:
            raise ValueError("Пустой ответ на проверочный запрос без schema")
        logging.info("Preflight 1/2 пройден. Ответ модели: %r", response.text.strip()[:50])
    except genai_errors.ClientError as e:
        code = _extract_status_code(e)
        logging.error(
            "Preflight 1/2 НЕ пройден (код %s) - проблема в ключе/модели, а не в schema. "
            "Полный ответ API: %s",
            code, e,
        )
        raise

    logging.info("Preflight 2/2: проверка response_schema на 2 синтетических сегментах...")
    tiny_batch = [
        Segment(index=999901, start="00:00:00,000", end="00:00:01,000", text="A man walks through a forest."),
        Segment(index=999902, start="00:00:01,000", end="00:00:02,000", text=""),
    ]
    try:
        call_gemini_batch(client, model, tiny_batch)
        logging.info("Preflight 2/2 пройден - schema/config в порядке.")
    except Exception as e:
        logging.error(
            "Preflight 2/2 НЕ пройден - проблема именно в response_schema/config, "
            "не в размере батча. Ошибка: %s",
            e,
        )
        raise


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def make_batches(
    segments: list[Segment],
    batch_size: int,
    context_window: int,
    min_last_batch_ratio: float = 0.25,
) -> list[tuple[list[Segment], list[Segment], list[Segment]]]:
    """Разбивает сегменты на батчи и добавляет к каждому контекст (соседние
    сегменты до/после), которые передаются модели, но не входят в схему ответа.

    Если последний батч получается совсем маленьким (например 1 сегмент из 1001
    при batch_size=100), сливает его с предыдущим - иначе это отдельный вызов
    Gemini с полным системным промптом ради 1-2 строк, что просто тратит бюджет."""
    n = len(segments)
    boundaries = list(range(0, n, batch_size))

    if len(boundaries) >= 2:
        last_size = n - boundaries[-1]
        min_size = max(context_window, int(batch_size * min_last_batch_ratio))
        if last_size < min_size:
            boundaries.pop()

    batches = []
    for pos, i in enumerate(boundaries):
        end = boundaries[pos + 1] if pos + 1 < len(boundaries) else n
        batch = segments[i:end]
        context_before = segments[max(0, i - context_window) : i]
        context_after = segments[end : end + context_window]
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
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Пропустить preflight-проверки перед основным циклом (не рекомендуется).",
    )
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

    if not args.skip_preflight:
        try:
            preflight_check(client, args.model)
        except Exception:
            logging.error("Preflight-проверка не пройдена, основной цикл не запускается.")
            return 1

    results: dict[str, dict] = {}

    batches = make_batches(segments, args.batch_size, CONTEXT_WINDOW)
    for batch_num, (batch, context_before, context_after) in enumerate(batches, start=1):
        logging.info(
            "Батч %s/%s: сегменты %s..%s", batch_num, len(batches), batch[0].index, batch[-1].index
        )
        try:
            batch_result = call_gemini_batch(client, args.model, batch, context_before, context_after)
        except Exception as e:
            # Жёсткое падение: тихая деградация хуже явной ошибки, которую можно сразу
            # увидеть в логах и перезапустить job. Частичный requests.json НЕ создаётся.
            logging.error(
                "Батч %s..%s не обработан после %s попыток: %s. Прерываю выполнение, "
                "requests.json НЕ будет записан.",
                batch[0].index, batch[-1].index, MAX_RETRIES, e,
            )
            return 1

        results.update(batch_result)

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
