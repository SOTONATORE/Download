#!/usr/bin/env python3
"""
generate_queries.py

Генерирует requests.json с поисковыми запросами (стоковые/архивные видео и фото)
для каждого сегмента SRT-файла, используя Gemini API (structured output через
response_schema).

Использование:
    python generate_queries.py --input input.srt --output requests.json

Переменные окружения:
    GEMINI_API_KEY       - обязателен, ключ Gemini API
    GENQ_MODEL           - опционально, основная модель (по умолчанию gemini-3.5-flash-lite)
    GENQ_FALLBACK_MODELS - опционально, через запятую - модели для переключения при
                           исчерпании дневного лимита основной (по умолчанию
                           "gemini-3.1-flash-lite" - RPD 500, подтверждено на реальном
                           аккаунте; gemini-3.8-flash из цепочки исключена намеренно -
                           у неё RPD всего 20, что слишком мало для батчевой генерации)
    GENQ_BATCH_SIZE       - опционально, число сегментов в одном вызове (по умолчанию 100)

Возвращаемые коды:
    0 - requests.json успешно записан целиком
    1 - структурная ошибка (битый SRT, невалидный запрос/schema, ключ) - НЕ связана
        с дневной квотой, требует разбора кода/данных
    3 - дневной лимит исчерпан у всех моделей из списка (основной + fallback) - НЕ баг,
        нужно либо подождать сброса квоты (полночь по тихоокеанскому времени), либо
        включить billing, либо добавить ещё моделей в --fallback-models. Прогресс
        сохранён в чекпоинте, повторный запуск продолжит с прерванного места.

Зависимости:
    pip install google-genai
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

# gemini-3.8-flash исключена из значений по умолчанию: RPD 20/день (подтверждено на
# реальном аккаунте - счётчик показал 21/20, т.е. даже отклонённый запрос считается
# в счёт квоты). gemini-3.5-flash-lite даёт RPD 500 при сопоставимом качестве для
# этой чисто структурной задачи (генерация JSON по схеме, без творческой составляющей).
DEFAULT_MODEL = "gemini-3.5-flash-lite"
# RPD 500 у обеих Flash-Lite моделей против RPD 20 у gemini-3.8-flash - проверено
# напрямую в панели лимитов аккаунта (не только по статьям в вебе).
DEFAULT_FALLBACK_MODELS = ["gemini-3.1-flash-lite"]
DEFAULT_BATCH_SIZE = 100
CONTEXT_WINDOW = 3
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 4
# Практический потолок ожидания для НЕ-дневных 429 (RPM/TPM) внутри одного запуска -
# дневную квоту (часы ожидания) всё равно нет смысла ждать в рамках одной CI-джобы.
MAX_RATE_LIMIT_SLEEP_SECONDS = 90
# Сколько часов считать модель "всё ещё исчерпанной сегодня" без повторной проверки -
# грубая эвристика (RPD сбрасывается в полночь по тихоокеанскому времени, точный запас
# в UTC зависит от сезона/DST, поэтому берём консервативные 20 часов).
EXHAUSTED_MODEL_TTL_HOURS = 20
CHECKPOINT_SUFFIX = ".checkpoint.json"
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

RESPONSE_SCHEMA = types.Schema(type=types.Type.ARRAY, items=SEGMENT_ENTRY_SCHEMA)


@dataclass
class Segment:
    index: int
    start: str
    end: str
    text: str


class DailyQuotaExceededError(RuntimeError):
    """Дневной лимит запросов (RPD) для конкретной модели исчерпан - ретраить эту же
    модель бессмысленно до сброса квоты."""

    def __init__(self, model: str, message: str):
        super().__init__(message)
        self.model = model


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


def source_hash(segments: list[Segment]) -> str:
    """Хэш содержимого сегментов - чтобы не применить чекпоинт от другого SRT-файла."""
    h = hashlib.sha256()
    for s in segments:
        h.update(f"{s.index}|{s.start}|{s.end}|{s.text}".encode("utf-8"))
    return h.hexdigest()


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
# Разбор ошибок Gemini API (коды, quotaId, retryDelay)
# ---------------------------------------------------------------------------

_STATUS_CODE_RE = re.compile(r"\b([1-5]\d{2})\b")
_QUOTA_ID_RE = re.compile(r"quotaId['\"]?\s*:\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
_RETRY_HOURS_RE = re.compile(r"retry in\s+([\d.]+)\s*hours?\b", re.IGNORECASE)
_RETRY_SECONDS_RE = re.compile(r"retry in\s+([\d.]+)\s*s(?:econds)?\b", re.IGNORECASE)
_RETRY_DELAY_FIELD_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s['\"]", re.IGNORECASE)


def _extract_status_code(e: Exception) -> Optional[int]:
    """Достаёт HTTP-код из исключения даже если атрибут .code недоступен в текущей
    версии SDK - парсим из текстового представления ошибки как запасной вариант."""
    code = getattr(e, "code", None)
    if isinstance(code, int):
        return code
    m = _STATUS_CODE_RE.search(str(e))
    return int(m.group(1)) if m else None


def _is_daily_quota_error(e: Exception) -> bool:
    m = _QUOTA_ID_RE.search(str(e))
    return bool(m and "perday" in m.group(1).lower())


def _extract_retry_delay_seconds(e: Exception) -> Optional[float]:
    s = str(e)
    m = _RETRY_HOURS_RE.search(s)
    if m:
        return float(m.group(1)) * 3600
    m = _RETRY_SECONDS_RE.search(s)
    if m:
        return float(m.group(1))
    m = _RETRY_DELAY_FIELD_RE.search(s)
    if m:
        return float(m.group(1))
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hours_since(iso_ts: str) -> float:
    try:
        then = datetime.fromisoformat(iso_ts)
    except ValueError:
        return 9999.0
    return (datetime.now(timezone.utc) - then).total_seconds() / 3600


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

            if code == 429 and _is_daily_quota_error(e):
                # Дневная квота - ретраить эту модель бессмысленно в принципе.
                raise DailyQuotaExceededError(model, str(e)) from e

            if code == 429:
                # RPM/TPM - временное ограничение, ждём подсказанное API время (с потолком).
                delay = _extract_retry_delay_seconds(e)
                if delay is None:
                    delay = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
                capped_delay = min(delay, MAX_RATE_LIMIT_SLEEP_SECONDS)
                logging.warning(
                    "Батч [%s..%s]: 429 (не дневная квота) на модели %s, попытка %s/%s. "
                    "API просит подождать %.1fs (жду %.1fs). %s",
                    batch[0].index, batch[-1].index, model, attempt, MAX_RETRIES,
                    delay, capped_delay, e,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(capped_delay)
                last_error = e
                continue

            # Любой другой 4xx (400/401/403/404/...) - структурная ошибка запроса или
            # ключа, ретраить бессмысленно. Печатаем полное тело ответа API.
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


def pick_working_model(
    client: "genai.Client", candidates: list[str], exhausted_models: dict[str, str]
) -> tuple[str, list[str]]:
    """Пробует модели по очереди (основная + fallback), пропуская без лишнего запроса
    те, что уже отмечены исчерпанными сегодня. Возвращает (рабочая_модель, остаток_очереди).
    Мутирует exhausted_models при обнаружении новой исчерпанной модели."""
    remaining = list(candidates)
    while remaining:
        model = remaining[0]
        known_exhausted_at = exhausted_models.get(model)
        if known_exhausted_at and _hours_since(known_exhausted_at) < EXHAUSTED_MODEL_TTL_HOURS:
            logging.info(
                "Модель %s уже отмечена исчерпанной %.1fч назад - пропускаю без запроса.",
                model, _hours_since(known_exhausted_at),
            )
            remaining.pop(0)
            continue

        logging.info("Preflight: проверяю модель %s (без response_schema)...", model)
        try:
            response = client.models.generate_content(model=model, contents="Ответь одним словом: OK")
            if not response.text:
                raise ValueError("Пустой ответ на проверочный запрос без schema")
            logging.info("Модель %s доступна. Ответ: %r", model, response.text.strip()[:50])
            return model, remaining[1:]
        except genai_errors.ClientError as e:
            if _extract_status_code(e) == 429 and _is_daily_quota_error(e):
                logging.warning("У модели %s уже исчерпан дневной лимит: %s", model, e)
                exhausted_models[model] = _now_iso()
                remaining.pop(0)
                continue
            logging.error(
                "Preflight не пройден для модели %s (код %s, НЕ дневная квота) - похоже "
                "проблема в ключе/доступе, а не в лимитах. Полный ответ: %s",
                model, _extract_status_code(e), e,
            )
            raise

    raise RuntimeError(
        f"У всех моделей из списка ({', '.join(candidates)}) на сегодня исчерпан дневной лимит."
    )


def schema_preflight_check(client: "genai.Client", model: str) -> None:
    """Проверка response_schema на 2 синтетических сегментах - изолирует проблемы в
    самой схеме/конфиге от проблем с ключом/моделью (которые уже проверил pick_working_model)."""
    logging.info("Preflight: проверяю response_schema на 2 синтетических сегментах...")
    tiny_batch = [
        Segment(index=999901, start="00:00:00,000", end="00:00:01,000", text="A man walks through a forest."),
        Segment(index=999902, start="00:00:01,000", end="00:00:02,000", text=""),
    ]
    call_gemini_batch(client, model, tiny_batch)
    logging.info("Preflight по response_schema пройден.")


# ---------------------------------------------------------------------------
# Чекпоинт (для возобновления после исчерпания дневной квоты или любого прерывания)
# ---------------------------------------------------------------------------

def checkpoint_path_for(output_path: str) -> str:
    return output_path + CHECKPOINT_SUFFIX


def load_checkpoint(path: str, expected_hash: str) -> tuple[dict[str, dict], dict[str, str]]:
    if not os.path.isfile(path):
        return {}, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.warning("Не удалось прочитать чекпоинт %s (%s) - начинаю с нуля.", path, e)
        return {}, {}

    if data.get("source_hash") != expected_hash:
        raise ValueError(
            f"Чекпоинт {path} относится к ДРУГОМУ SRT-файлу (хэш содержимого не совпадает). "
            f"Если это ожидаемо (SRT намеренно изменился) - удалите файл чекпоинта вручную "
            f"и запустите заново. Продолжать с несовпадающим чекпоинтом небезопасно."
        )
    results = data.get("results", {})
    exhausted = data.get("exhausted_models", {})
    if results:
        logging.info("Найден чекпоинт: %s сегментов уже обработано ранее.", len(results))
    return results, exhausted


def save_checkpoint(path: str, src_hash: str, results: dict, exhausted_models: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            {"source_hash": src_hash, "results": results, "exhausted_models": exhausted_models},
            f, ensure_ascii=False,
        )
    os.replace(tmp_path, path)  # атомарная замена, не оставляет битый файл при сбое на записи


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
        "--fallback-models",
        default=os.environ.get("GENQ_FALLBACK_MODELS", ",".join(DEFAULT_FALLBACK_MODELS)),
        help="Через запятую - модели для переключения при исчерпании дневного лимита основной.",
    )
    parser.add_argument(
        "--skip-schema-preflight", action="store_true",
        help="Пропустить проверку response_schema на 2 синтетических сегментах (не рекомендуется).",
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

    src_hash = source_hash(segments)
    checkpoint_path = checkpoint_path_for(args.output)
    try:
        results, exhausted_models = load_checkpoint(checkpoint_path, src_hash)
    except ValueError as e:
        logging.error("%s", e)
        return 1

    client = genai.Client(api_key=api_key)

    fallback_models = [m.strip() for m in args.fallback_models.split(",") if m.strip()]
    candidates = [args.model] + [m for m in fallback_models if m != args.model]

    try:
        current_model, fallback_queue = pick_working_model(client, candidates, exhausted_models)
    except Exception:
        save_checkpoint(checkpoint_path, src_hash, results, exhausted_models)
        logging.error(
            "Не удалось найти рабочую модель среди %s - дневной лимит исчерпан у всех. "
            "Прогресс (%s из %s сегментов) сохранён в чекпоинте %s. Запустите скрипт "
            "повторно позже (лимит сбрасывается в полночь по тихоокеанскому времени) или "
            "добавьте больше моделей в --fallback-models / GENQ_FALLBACK_MODELS.",
            candidates, len(results), len(segments), checkpoint_path,
        )
        return 3

    if not args.skip_schema_preflight:
        try:
            schema_preflight_check(client, current_model)
        except Exception as e:
            logging.error("Preflight по response_schema не пройден: %s. Основной цикл не запускается.", e)
            return 1

    batches = make_batches(segments, args.batch_size, CONTEXT_WINDOW)

    for batch_num, (batch, context_before, context_after) in enumerate(batches, start=1):
        if all(str(s.index) in results for s in batch):
            logging.info(
                "Батч %s/%s (сегменты %s..%s) уже есть в чекпоинте, пропускаю.",
                batch_num, len(batches), batch[0].index, batch[-1].index,
            )
            continue

        logging.info(
            "Батч %s/%s: сегменты %s..%s (модель: %s)",
            batch_num, len(batches), batch[0].index, batch[-1].index, current_model,
        )

        while True:
            try:
                batch_result = call_gemini_batch(
                    client, current_model, batch, context_before, context_after
                )
                break
            except DailyQuotaExceededError as e:
                logging.warning(
                    "Дневной лимит исчерпан для модели %s: %s", e.model, e
                )
                exhausted_models[e.model] = _now_iso()
                save_checkpoint(checkpoint_path, src_hash, results, exhausted_models)
                if not fallback_queue:
                    logging.error(
                        "Дневной лимит исчерпан, а запасных моделей больше нет. Прогресс "
                        "(%s из %s сегментов) сохранён в чекпоинте %s. Запустите скрипт "
                        "повторно позже или добавьте больше моделей в --fallback-models.",
                        len(results), len(segments), checkpoint_path,
                    )
                    return 3
                current_model = fallback_queue.pop(0)
                logging.warning("Переключаюсь на запасную модель: %s", current_model)
                continue
            except Exception as e:
                # Структурная ошибка (не квота): жёсткое падение, но чекпоинт с уже
                # готовыми батчами остаётся на диске - requests.json не пишется.
                logging.error(
                    "Батч %s..%s не обработан после %s попыток: %s. Прерываю выполнение, "
                    "requests.json НЕ будет записан (чекпоинт с %s готовыми сегментами "
                    "сохранён в %s).",
                    batch[0].index, batch[-1].index, MAX_RETRIES, e, len(results), checkpoint_path,
                )
                save_checkpoint(checkpoint_path, src_hash, results, exhausted_models)
                return 1

        results.update(batch_result)
        save_checkpoint(checkpoint_path, src_hash, results, exhausted_models)

        progress_pct = len(results) / len(segments) * 100
        logging.info(
            "Батч %s/%s готов (%.0f%% сегментов): %s из %s",
            batch_num, len(batches), progress_pct, len(results), len(segments),
        )

    ordered = {str(k): results[str(k)] for k in sorted(int(k) for k in results.keys())}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)

    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)

    logging.info("Готово: %s сегментов записано в %s", len(ordered), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
