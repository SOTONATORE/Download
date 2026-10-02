#!/usr/bin/env python3
"""
generate_queries.py

Генерирует requests.json с поисковыми запросами (стоковые/архивные видео и фото)
для каждого сегмента SRT-файла, используя Gemini API (structured output через
response_schema).

Использование:
    python generate_queries.py --input input.srt --output requests.json
    python generate_queries.py --sources-mode 2 --strict 1

Переменные окружения:
    GEMINI_API_KEY       - обязателен, ключ Gemini API
    GENQ_MODEL           - опционально, основная модель (по умолчанию gemini-3.5-flash-lite)
    GENQ_FALLBACK_MODELS - опционально, через запятую - модели для переключения при
                           исчерпании дневного лимита основной (по умолчанию
                           "gemini-3.1-flash-lite" - RPD 500, подтверждено на реальном
                           аккаунте; gemini-3.8-flash из цепочки исключена намеренно -
                           у неё RPD всего 20, что слишком мало для батчевой генерации)
    GENQ_BATCH_SIZE       - опционально, число сегментов в одном вызове (по умолчанию 100)
    GENQ_HTTP_TIMEOUT_SECONDS - опционально, таймаут одного HTTP-запроса к Gemini в секундах
                           (по умолчанию 60; в SDK передаётся в миллисекундах)
    GENQ_SOURCES_MODE    - опционально, режим источников: 1 (только архив: wikimedia, loc, nasa),
                           2 (микс, по умолчанию), 3 (только сток: pexels, pixabay)
    GENQ_STRICT          - опционально, строгость проверки запросов: 1 (калибровка, по умолчанию -
                           падение с кодом 1 при нарушениях после повтора), 2 (мягко - warning в логе,
                           requests.json записан, код 0)

Формат requests.json:
    Словарь "номер сегмента" (строка) -> запись с полями: scene (одно английское предложение
    о том, что видно в кадре), sites (список источников в порядке приоритета), query_narrow,
    query_medium, query_broad (строка или null), type ("image"/"video"), is_entity (bool, выводится
    как bool(entity_keywords)), entity_keywords (список строк, английское написание первым). Старых полей query и fallback_query в записи НЕТ.
    Записи перед сохранением проходят _normalize_entry (детерминированная починка без
    повторных вызовов Gemini).

Возвращаемые коды:
    0 - requests.json успешно записан целиком (в т.ч. при предупреждениях в мягком режиме GENQ_STRICT=2)
    1 - структурная ошибка (битый SRT, невалидный запрос/schema, ключ) ИЛИ неудачная валидация
        запросов при GENQ_STRICT=1 после повторного запроса REPAIR (requests.json при этом
        сохраняется на диск для проверки, чекпоинт не удаляется)
    3 - дневной лимит ПОДТВЕРЖДЁН у всех моделей из списка (основной + fallback) - НЕ баг,
        нужно либо подождать сброса квоты (полночь по тихоокеанскому времени), либо
        включить billing, либо добавить ещё моделей в --fallback-models. Прогресс
        сохранён в чекпоинте, повторный запуск продолжит с прерванного места.
        Возвращается ТОЛЬКО когда pick_working_model подтвердила квоту у каждой модели.
    4 - preflight: ни одна модель не отвечает по ВРЕМЕННОЙ причине (5xx, таймаут, сетевой
        сбой - после 3 попыток с бэкоффом), либо часть моделей без квоты, а остальные
        недоступны. Это НЕ квота и НЕ структурная ошибка: достаточно перезапустить позже.
        Чекпоинт сохранён.
    Любой иной сбой preflight (неверное имя модели/ключ, неожиданное исключение) - код 1,
    с полным traceback в логе.

Зависимости:
    pip install google-genai
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
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
# Круг 2 REPAIR: широкое окно соседей (только текст SRT, ближайшие первыми, с пометкой расстояния).
REPAIR2_CONTEXT_WINDOW = 10
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
# Версия формата записи сегмента (набор полей ответа модели). Пишется в чекпоинт; при
# несовпадении старый чекпоинт игнорируется, чтобы записи разных схем не смешались в
# одном requests.json. ПОВЫШАТЬ при любом изменении полей SEGMENT_ENTRY_SCHEMA.
# 1 - query/fallback_query; 2 - scene + query_narrow/medium/broad.
SCHEMA_VERSION = 2
# Таймаут HTTP-запроса к Gemini (сек). В google-genai HttpOptions.timeout задаётся в
# МИЛЛИСЕКУНДАХ, поэтому при создании клиента переводим секунды в мс.
DEFAULT_HTTP_TIMEOUT_SECONDS = 60
# Preflight: ретраи транзиентных сбоев (5xx/таймаут/сеть) на одной модели.
PREFLIGHT_MAX_ATTEMPTS = 3
PREFLIGHT_BACKOFF_BASE_SECONDS = 2
SITES = ["pexels", "pixabay", "wikimedia", "nasa", "loc"]

DEFAULT_SOURCES_MODE = 2
DEFAULT_STRICT = 1

# Мусорные типы кадра (не должны присутствовать ни в scene, ни в запросах).
# Базовый список слов в единственном числе в ОДНОЙ константе; регулярное выражение
# строится динамически кодом через build_junk_kind_re.
JUNK_KIND_WORDS = (
    "map", "calendar", "document", "chart", "diagram", "infographic",
    "flag", "emblem", "coat of arms", "newspaper", "ID card", "passport",
)


def build_junk_kind_re(words: tuple[str, ...]) -> re.Pattern:
    """Динамически собирает регулярное выражение для поиска мусорных типов кадра
    по границам слов, с учётом множественного числа и без учёта регистра."""
    patterns = []
    for w in words:
        parts = w.split()
        if w.lower() == "coat of arms":
            patterns.append(r"coats?\s+of\s+arms")
        elif len(parts) > 1:
            escaped_lead = r"[\s-]+".join(re.escape(p) for p in parts[:-1])
            escaped_last = re.escape(parts[-1]) + r"s?"
            patterns.append(escaped_lead + r"[\s-]+" + escaped_last)
        else:
            patterns.append(re.escape(w) + r"s?")
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)


JUNK_KIND_RE = build_junk_kind_re(JUNK_KIND_WORDS)


def parse_sources_mode(cli_val: Optional[str | int], env_val: Optional[str | int]) -> int:
    """Парсит режим источников (1=архив, 2=микс, 3=сток). CLI в приоритете.
    Любое невалидное значение игнорируется с WARNING и берётся 2."""
    val = cli_val if cli_val is not None else env_val
    if val is None:
        return DEFAULT_SOURCES_MODE
    val_str = str(val).strip()
    if val_str in ("1", "2", "3"):
        return int(val_str)
    logging.warning(
        "Некорректный режим источников %r (допустимо 1, 2, 3) - использую по умолчанию %s.",
        val, DEFAULT_SOURCES_MODE,
    )
    return DEFAULT_SOURCES_MODE


def parse_strict_mode(cli_val: Optional[str | int], env_val: Optional[str | int]) -> int:
    """Парсит строгость проверки (1=калибровка, 2=мягко). CLI в приоритете.
    Любое невалидное значение игнорируется с WARNING и берётся 1."""
    val = cli_val if cli_val is not None else env_val
    if val is None:
        return DEFAULT_STRICT
    val_str = str(val).strip()
    if val_str in ("1", "2"):
        return int(val_str)
    logging.warning(
        "Некорректная строгость проверки %r (допустимо 1, 2) - использую по умолчанию %s.",
        val, DEFAULT_STRICT,
    )
    return DEFAULT_STRICT


# Слова-наполнители, которые безусловно вырезаются из запросов (см. _normalize_entry).
# video/stock/historic/vintage сюда НЕ входят: они допустимы, если часть предмета
# ("stock market", "video game", "vintage car"). Фразы идут раньше одиночных слов.
FORBIDDEN_QUERY_PHRASES = [
    "stock footage", "stock photos", "stock photo", "stock images", "stock image",
    "b-roll", "broll", "cinematic", "footage", "HD", "4K",
]
# Пробел во фразе матчится и как дефис ("stock-photo"); границы слов - чтобы не задеть,
# например, "hd" внутри другого слова.
FORBIDDEN_QUERY_RE = re.compile(
    r"\b(?:"
    + "|".join(r"[\s-]+".join(re.escape(w) for w in p.split()) for p in FORBIDDEN_QUERY_PHRASES)
    + r")\b",
    re.IGNORECASE,
)
# Чистка краёв запроса после вырезания наполнителей (см. _strip_forbidden): остаются
# висящие предлоги/союзы и знаки препинания ("footage of crowd" -> "of crowd").
# Наборы для начала и конца РАЗНЫЕ: артикли the/a/an в начале не трогаем, они бывают
# частью названий ("The Hague", "The Beatles"), а в конце висящий артикль - всегда мусор.
EDGE_STRIP_CHARS = ",;:.-" + " \t\r\n"
LEADING_STOP_WORDS = frozenset({"of", "in", "on", "at", "with", "for", "and", "to"})
TRAILING_STOP_WORDS = LEADING_STOP_WORDS | {"the", "a", "an"}
# Обязательные ключи записи ответа модели (после извлечения segment_index).
REQUIRED_ENTRY_KEYS = [
    "scene", "sites", "query_narrow", "query_medium", "query_broad",
    "type", "is_entity", "entity_keywords",
]
# Потолок отдельных warning нормализации за весь запуск, дальше - только итоговые счётчики.
MAX_NORMALIZE_WARNINGS = 20


def build_system_instruction(
    mode: int = DEFAULT_SOURCES_MODE, junk_words: tuple[str, ...] = JUNK_KIND_WORDS
) -> str:
    """Генерирует системный промпт с учётом выбранного режима источников (1/2/3)
    и запрещённых типов кадра из junk_words."""
    junk_list_str = ", ".join(junk_words)

    mode_blocks = {
        1: (
            "SOURCE MODE RULES (MODE 1: ARCHIVE ONLY):\n"
            "- All segments MUST use archival sites ONLY: [\"wikimedia\", \"loc\"] (use \"nasa\" first only when explicitly about space/astronomy/NASA missions). NEVER include \"pexels\" or \"pixabay\".\n"
            "- All search queries must follow the archival search style (concise proper nouns, literal title/caption matches).\n"
            "- For abstract or general segments without a specific named entity: derive a generalized ARCHIVAL query for the depicted place/era from the nearest neighbor segments, without people names, and strictly without any forbidden visual types.\n"
        ),
        2: (
            "SOURCE MODE RULES (MODE 2: MIXED ARCHIVE AND STOCK):\n"
            "- SITES DEFAULT: Default to stock sites [\"pexels\", \"pixabay\"]. Use archival sites [\"wikimedia\", \"loc\"] ONLY when the segment's narration explicitly names a specific real person, a specific building, or a specific physical object that can actually be photographed (not just a date, country, or era). Add \"nasa\" first only for space/astronomy/NASA missions.\n"
            "- When genuinely unsure, use stock sites [\"pexels\", \"pixabay\"], NOT both and NOT archive.\n"
            "- For abstract or general segments: follow the ABSTRACT / GENERAL SEGMENTS rule below (use stock sites).\n"
        ),
        3: (
            "SOURCE MODE RULES (MODE 3: STOCK ONLY):\n"
            "- All segments MUST use stock sites ONLY: [\"pexels\", \"pixabay\"]. NEVER include \"wikimedia\", \"loc\", or \"nasa\".\n"
            "- All search queries must follow the stock search style: query_medium (2-4 words, object + context), query_narrow (4-6 words, slightly more specific), query_broad (1-2 words, general image).\n"
            "- Replace any proper names with visual generalizations (e.g. \"Mehmed VI\" / \"sultan\" -> \"man in traditional robe\", \"Topkapi Palace\" -> \"old palace courtyard\").\n"
            "- In Mode 3, entity_keywords MUST ALWAYS be an empty list [], and is_entity MUST ALWAYS be false for ALL segments.\n"
        ),
    }
    mode_rule = mode_blocks.get(mode, mode_blocks[2])

    instruction = f"""\
You generate search-query instructions for stock/archival video and photo sourcing for a video's \
scenes. You receive numbered scene segments (number = the segment's sequential position in the \
original SRT, plus its timing and on-screen text).

Return a JSON ARRAY of objects - EXACTLY one object per segment listed under "Segments that need \
a response" (including silent/empty segments), with no gaps and no duplicates. Every object MUST \
contain a segment_index field (an integer, exactly matching the number from "### Segment N") plus \
the remaining fields defined by the schema. The order of objects in the array does not matter.

{mode_rule}
FIELD RULES:

1. scene - ONE English sentence: what the viewer should SEE in the frame for this segment. Choose it \
by answering three questions in order: (1) WHERE does this happen - the place named in the segment's \
text, or, if the text names none, the place implied by the neighboring segments; (2) WHO is there - the \
people by role or group, without personal names when the sites are stock; (3) WHAT of this can a camera \
film as a solid, living subject - a building, a hall, a street, a landscape, people, a vehicle, a tool, \
a physical object - rather than a flat sheet. Flat objects that carry text (paper, page, sheet, \
parchment, letter, manuscript, scroll) are undesirable as the MAIN subject of the frame. \
Never use or depict these shot types in scene or in any query: {junk_list_str} (and plural forms); \
a date or number is never a calendar, and geopolitics or wars are never a map. \
Write scene BEFORE the queries and derive all three queries from it. \
If the only thing the segment's text gives you to show is one of those forbidden shot types or a flat \
text-bearing object (a treaty, decree, letter, newspaper, map, date, flag, and the like), treat the \
segment as abstract and apply the ABSTRACT / GENERAL SEGMENTS rule below: a generalized frame built \
from the neighbors is not an invented detail. Otherwise do NOT invent details that are not in the \
segment's text (no extra people, moods, weather, time of day, or settings the text does not mention).

2. sites - ordered list of source sites, in priority order for this segment. Allowed values: \
"pexels", "pixabay", "wikimedia", "nasa", "loc". This list is fixed - never invent other sources. \
Follow the SOURCE MODE RULES above.

3. query_narrow, query_medium, query_broad - three English search queries for the SAME scene, from \
most specific to most general. They are tried in this order, so each must be a realistic search \
phrase on its own. The style depends on the FIRST site in "sites":
   - If the first site is an archive (wikimedia / loc / nasa):
     * query_narrow: SHORT, 2-4 words ONLY - the exact proper noun(s) that a real file title or \
caption on these sites would actually contain: a person's full name, OR a specific place name, OR a \
named event, optionally with a short refinement of the object (e.g. "Topkapi Palace gate", "Mehmed \
VI", "Siege of Vienna"). Do NOT append a year unless it is named in the segment's text (see the YEARS \
rule). If a real name genuinely needs more than 4 words, that's fine - the limit is about cutting \
padding, not truncating a proper noun. For nasa: exact mission/object names \
and dates, as concise as the name requires.
     * query_medium: ONLY the name OR ONLY the place, WITHOUT a year (e.g. "Mehmed VI", "Vienna").
     * query_broad: an ordinary plain-language phrasing of the same visual scene for \
pexels/pixabay, 2-4 words, no proper nouns (e.g. "old harbor", "wooden desk").
     MediaWiki (wikimedia) and LOC search match literal file titles/captions, which are short and \
factual, so descriptive or stylistic padding only dilutes the match. BAD -> GOOD: "Villa Magnolia \
San Remo interior 1926 archive" -> "Villa Magnolia San Remo"; "Prince Ertugrul Ottoman prince \
historical photo" -> "Ertuğrul Osman" (or the exact name given in the segment).
   - If the first site is stock (pexels / pixabay):
     * query_medium: 2-4 words, main object + context (e.g. "winding mountain road").
     * query_narrow: slightly more specific than medium, 4-6 words (e.g. "winding mountain road \
with pines").
     * query_broad: 1-2 words, the general image (e.g. "mountains").
   - query_broad = null ONLY for segments with no visual scene at all - e.g. silence, a black \
screen, a title/credits card with no depicted content, or on-screen text with nothing else \
happening. When in doubt, fill it in rather than returning null. query_narrow and query_medium are \
never null.

4. type - "image" or "video", whichever fits the described scene better (a static portrait -> \
"image"; a dynamic action or a generic/modern/abstract scene -> "video").

5. entity_keywords - the MAIN entity field. List EVERY proper name (person, place, event, \
organization, treaty, building) that appears in your query_narrow or query_medium, each in TWO \
variants: the English spelling and the Russian spelling, as consecutive pairs, for example \
["Vienna", "Вена", "Topkapi Palace", "Дворец Топкапы"]. The ENGLISH spelling of the most important \
name MUST be the FIRST element of the list (the search script uses it as the lookup query). Keep \
each name as one element (do not split "Topkapi Palace" into two words). Generic things - religions, \
ideologies, nationalities, professions, titles without a name, emotions, general themes ("Islam", \
"a caliph", "war", "monarchy") are NOT proper names. An EMPTY list [] means "no proper names in \
the queries". Never leave it empty when a query contains a proper name, even if the frame looks \
generic (a plain city view or a coastline of a named city is still an entity scene).

6. is_entity - true when entity_keywords is non-empty, false when it is empty. Fill it consistently \
with entity_keywords (the script recomputes it from that list anyway).

ABSTRACT / GENERAL SEGMENTS:
When a segment lacks a concrete visible physical subject (such as narrator evaluations, conclusions, \
transitions, abstract concepts, emotions, numbers, or dates without physical objects):
- Look at the nearest neighboring segments (2 segments before and 2 segments after, in the batch \
and in the Context BEFORE/AFTER sections).
- Derive a general visual theme from them.
- Formulate the scene and all three queries as a GENERALIZED stock shot fitting that visual theme \
(without proper nouns, even if neighbors name them: e.g. "Topkapi Palace" -> "old palace interior").
- Set sites = ["pexels", "pixabay"], entity_keywords = [], is_entity = false (unless running in Mode 1).

GENERAL RULES:
- Queries describe what is VISIBLE in the frame. They do not retell or paraphrase the narrator's \
words.
- NEVER use these filler words in any query: b-roll, cinematic, footage, HD, 4K, historical, and the \
phrases "stock footage", "stock photo", "stock image". Also NEVER use forbidden visual types in \
any query: {junk_list_str} (and plural forms). The words video, stock, vintage are also banned as \
filler (style padding added to a query), but ALLOWED when they are part of the depicted subject \
itself (e.g. "stock market", "video game", "video call", "vintage car").
- NEVER use mood adjectives (sad, empty, mysterious, dramatic, lonely, gloomy, etc.) in a query \
unless that exact mood is stated in the segment's text.
- YEARS: put a year (also a decade like "1920s" or a range like "1914-1918") in a query ONLY if that \
exact year is named in the segment's text; otherwise the query contains no year at all. Never guess or \
add a year from your own knowledge. Example: the text says "In 1918 Mehmed VI became sultan" -> the \
year comes from the text, so "Mehmed VI 1918" is allowed; the text says only "Mehmed VI became \
sultan" -> use "Mehmed VI". Archive sites match every word, so an invented year returns nothing.
- Never include "creative commons", "free", or "no copyright" in a query - these are not effective \
search terms; licensing is filtered separately downstream, not through the query text.
- Do not invent scene details beyond what the segment's text actually says (a generalized frame for an \
abstract segment, per the scene and ABSTRACT rules, is not an invention).
- Always include silent/empty segments in the output with a neutral scene and queries inferred from \
neighboring segments' context (query_broad may be null for them) - never skip a segment number.

You may also be given extra CONTEXT - neighboring segments before and/or after the main list, under \
separate "Context BEFORE" / "Context AFTER" headers. Use this context only to understand the \
narrative (for example, to resolve a pronoun or continue a thought from the current segment) - do \
NOT create response objects for these context segment numbers. Only answer for the segments listed \
under "Segments that need a response", each marked as "### Segment N".
"""
    return instruction


SYSTEM_INSTRUCTION = build_system_instruction(DEFAULT_SOURCES_MODE)

# Порядок полей важен: scene идёт первым после segment_index, чтобы модель сначала
# описывала кадр, а уже потом строила по нему запросы. Gemini не гарантирует порядок по
# порядку ключей в dict, поэтому он задан явно через property_ordering.
SEGMENT_ENTRY_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "segment_index": types.Schema(type=types.Type.INTEGER),
        "scene": types.Schema(
            type=types.Type.STRING,
            description=(
                "ONE English sentence: what the viewer should see in the frame. For abstract "
                "phrases use concrete objects/places of the topic (economy -> factory, cargo "
                "port, banknotes), not emotions. Do not invent details that are not in the segment text."
            ),
        ),
        "sites": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING, enum=SITES),
        ),
        "query_narrow": types.Schema(
            type=types.Type.STRING,
            description=(
                "Most specific query, English. First site archive (wikimedia/loc/nasa): exact "
                "name/place/event (+ short object refinement), 2-4 words, with NO year unless the "
                "year is named in the segment text (a proper name that needs more than 4 words "
                "is fine - the limit is about padding, not truncating a name). First site stock "
                "(pexels/pixabay): 4-6 words, a bit more specific than query_medium. No filler "
                "b-roll/cinematic/footage/HD/4K or 'stock footage/photo/image'; video/stock/"
                "historic/vintage only when part of the subject itself (e.g. 'stock market'); no "
                "mood adjectives absent from the text."
            ),
        ),
        "query_medium": types.Schema(
            type=types.Type.STRING,
            description=(
                "Medium query, English. First site archive: ONLY the name or ONLY the place, no "
                "year. First site stock: 2-4 words, object + context. Same filler-word rules as "
                "query_narrow."
            ),
        ),
        "query_broad": types.Schema(
            type=types.Type.STRING,
            nullable=True,
            description=(
                "Broadest query, English. First site archive: ordinary stock phrasing of the same "
                "scene for pexels/pixabay, 2-4 words. First site stock: 1-2 words, the general "
                "image. null ONLY when the segment has no visible scene at all (silence, black "
                "screen, title/credits card, bare on-screen text)."
            ),
        ),
        "type": types.Schema(type=types.Type.STRING, enum=["image", "video"]),
        "is_entity": types.Schema(
            type=types.Type.BOOLEAN,
            description=(
                "true if entity_keywords is non-empty, false if it is empty (the script "
                "recomputes this from entity_keywords)."
            ),
        ),
        "entity_keywords": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description=(
                "ALL proper names (person, place, event, organization, treaty, building) from "
                "query_narrow/query_medium, each as English spelling then Russian spelling. "
                "English spelling of the main name FIRST. Empty list = no proper names."
            ),
        ),
    },
    property_ordering=[
        "segment_index",
        "scene",
        "sites",
        "query_narrow",
        "query_medium",
        "query_broad",
        "type",
        "is_entity",
        "entity_keywords",
    ],
    required=[
        "segment_index",
        "scene",
        "sites",
        "query_narrow",
        "query_medium",
        "query_broad",
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


class AllModelsQuotaExhaustedError(RuntimeError):
    """pick_working_model: у КАЖДОЙ модели из списка подтверждена исчерпанная дневная квота."""


class ModelUnavailableError(Exception):
    """pick_working_model: рабочей модели не нашлось, но причина не (только) квота -
    5xx/таймаут/сеть после ретраев. reasons: model -> описание последней ошибки."""

    def __init__(self, reasons: dict[str, str], quota_models: list[str]):
        self.reasons = reasons
        self.quota_models = quota_models
        super().__init__(
            "Модели недоступны по временной причине: "
            + "; ".join(f"{m}: {r}" for m, r in reasons.items())
            + (f". Квота исчерпана у: {', '.join(quota_models)}" if quota_models else "")
        )


# Транзиентные сбои preflight: 5xx, таймауты, сетевые ошибки, пустой ответ.
_PREFLIGHT_TRANSIENT_ERRORS = (
    genai_errors.ServerError,
    httpx.HTTPError,  # TimeoutException, TransportError и т.п.
    TimeoutError,
    OSError,  # ConnectionError, ssl.SSLError
    ValueError,  # пустой ответ на проверочный запрос
)


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
    text = s.text if s.text else "(silence / no text)"
    return f"[{s.index}] {text}"


def _repair_round_rule(mode: int) -> str:
    """Дополнительное правило круга 2 REPAIR (только для него), зависит от режима источников."""
    base = (
        "- ROUND 2 ONLY: if the segment is about a specific document, treaty, decree, letter, map, date or "
        "the like, and without it the text gives nothing that a camera can shoot, replace it with a "
        "generalized frame derived from the neighbors (the place and the people, no personal names). "
    )
    if mode == 1:
        return base + (
            "Keep archival sites according to the Mode 1 rules; write a generalized ARCHIVAL query for the "
            "place or era taken from the neighbors, with no personal names and no shot-type words."
        )
    return base + (
        'Set sites = ["pexels", "pixabay"], entity_keywords = [], is_entity = false.'
    )


def build_prompt(
    batch: list[Segment],
    context_before: list[Segment] | None = None,
    context_after: list[Segment] | None = None,
    repair_info: dict[int, dict] | None = None,
) -> str:
    lines: list[str] = []

    if context_before:
        lines.append("### Context BEFORE (do not create response objects for these segment numbers - context only)")
        lines.extend(_format_context_line(s) for s in context_before)
        lines.append("")

    if repair_info:
        lines.append("### Segments that need a response (REPAIR MODE)")
        lines.append(
            "The following segments had validation issues in their previously generated data. "
            "Fix the specific issues listed for each segment."
        )
        lines.append("")
        rep_round = max((int(r.get("round", 1)) for r in repair_info.values()), default=1)
        rep_mode = next((r["mode"] for r in repair_info.values() if r.get("mode")), DEFAULT_SOURCES_MODE)
        junk_list = ", ".join(JUNK_KIND_WORDS)
        if rep_round >= 2:
            lines.append(
                "This is REPAIR ROUND 2 of 2 (the last one): the first attempt did not remove the problems below."
            )
        lines.append("REPAIR RULES:")
        lines.append(
            "- Fix ONLY the listed issues. If an issue is only about sites / entity_keywords / is_entity, "
            "keep scene and queries as they are."
        )
        lines.append(
            f"- If an issue says a shot type is forbidden ({junk_list}): do NOT edit the old queries. "
            "First write a NEW scene from scratch, using only this segment's narration text and the text of "
            "its neighbors, by answering three questions in order: (1) WHERE does this happen; (2) WHO is "
            "there (people by role, no personal names when the sites are stock); (3) WHAT of this can a "
            "camera film as a solid, living subject (a building, a hall, a street, a landscape, people, a "
            "vehicle, a physical object) rather than a flat sheet. Only then derive query_narrow, "
            "query_medium and query_broad from that new scene. Flat objects that carry text (paper, page, "
            "sheet, parchment, letter, manuscript, scroll) must not be the main subject of the frame. "
            "The forbidden word AND its near-synonyms or reformulations (sheet, page, leaf, illustration, "
            "print, chart-like, etc.) must not appear in scene or in any of query_narrow / query_medium / "
            "query_broad. Do not just rephrase."
        )
        lines.append("- A date or a number is NEVER shown as a calendar; geopolitics and wars are NEVER shown as a map.")
        lines.append(
            "- For an abstract segment (a date, a number, an assessment, a transition), or one whose text "
            "offers nothing to show except a forbidden shot type, use the generalized "
            "stock shot rule from the system prompt: infer the common visual theme from the neighbors and "
            "write a generic frame (no proper names)."
        )
        lines.append("- Fix sites / entity_keywords issues according to the current sources mode.")
        if rep_round >= 2:
            lines.append(_repair_round_rule(rep_mode))
        lines.append("")
        for s in batch:
            text = s.text if s.text else "(silence / no text)"
            rep = repair_info.get(s.index, {})
            prev = rep.get("entry", {})
            issues_list = rep.get("issues", [])
            issues_formatted = "\n".join(f"  - {iss}" for iss in issues_list) if issues_list else "  - (no specific issues)"
            neighbors_str = rep.get("neighbors", "(no neighbor context)")

            lines.append(f"### Segment {s.index}")
            lines.append(f"Timing: {s.start} --> {s.end}")
            lines.append(f"Narration text: {text}")
            # Для сегментов с мусорным словом прежний кадр не показываем: модель на нём якорится
            has_junk = any(
                "forbidden shot type" in iss or "запрещённый тип кадра" in iss for iss in issues_list
            )
            if not has_junk:
                lines.append(f"Previous scene: {prev.get('scene', '')}")
                lines.append(f"Previous sites: {json.dumps(prev.get('sites', []))}")
                lines.append(f"Previous query_narrow: {prev.get('query_narrow', '')}")
                lines.append(f"Previous query_medium: {prev.get('query_medium', '')}")
                lines.append(f"Previous query_broad: {prev.get('query_broad', '')}")
            lines.append("Validation issues to fix:")
            lines.append(issues_formatted)
            lines.append(f"Neighbors (context):\n{neighbors_str}")
            lines.append("Instruction: Fix ONLY the validation issues above; do NOT introduce forbidden words or sites.")
            lines.append("")
        lines.append(
            "SELF-CHECK before answering: make sure that none of scene, query_narrow, query_medium, "
            f"query_broad contains a word from this list: {', '.join(JUNK_KIND_WORDS)}, nor its forms "
            "(plural, hyphenated, compound) or paraphrases."
        )
        lines.append("")
    else:
        lines.append("### Segments that need a response")
        for s in batch:
            text = s.text if s.text else "(silence / no text)"
            lines.append(f"### Segment {s.index}\nTiming: {s.start} --> {s.end}\nText: {text}\n")

    if context_after:
        lines.append("### Context AFTER (do not create response objects for these segment numbers - context only)")
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
# Нормализация записей ответа модели
# ---------------------------------------------------------------------------

# Остаток лимита отдельных warning (список - чтобы менять без global).
_normalize_warn_left = [MAX_NORMALIZE_WARNINGS]


def _warn_limited(msg: str, *args) -> None:
    """Warning с общим потолком на запуск, чтобы на 1000+ сегментов лог не заспамился."""
    if _normalize_warn_left[0] <= 0:
        return
    _normalize_warn_left[0] -= 1
    logging.warning(msg, *args)
    if _normalize_warn_left[0] == 0:
        logging.warning(
            "Достигнут лимит отдельных предупреждений нормализации (%s) - дальше только "
            "итоговые счётчики по батчам.", MAX_NORMALIZE_WARNINGS,
        )


def _trim_query_edges(text: str) -> str:
    """Чистит края запроса после вырезания наполнителей.

    В цикле до стабильного результата: обрезает пробелы и ",;:.-" по краям, убирает
    первое слово из LEADING_STOP_WORDS и последнее из TRAILING_STOP_WORDS (без учёта
    регистра). Регистр остальных слов не меняется; результат может стать пустым."""
    # внутри запроса: пробел перед запятой и двойные запятые ("crowd , , street")
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    while True:
        prev = text
        text = text.strip(EDGE_STRIP_CHARS)
        words = text.split(" ")
        if words and words[0].lower() in LEADING_STOP_WORDS:
            words = words[1:]
        if words and words[-1].lower() in TRAILING_STOP_WORDS:
            words = words[:-1]
        text = " ".join(words)
        if text == prev:
            return text


def _strip_forbidden(text: str) -> str:
    """Вырезает слова-наполнители, схлопывает пробелы, обрезает края.

    Если что-то реально вырезано, дополнительно чистит края от висящих предлогов и
    знаков препинания ("footage of crowd" -> "crowd"). Если ничего не вырезано,
    запрос не трогается, кроме схлопывания пробелов (чтобы не портить нормальные)."""
    cut, n_cut = FORBIDDEN_QUERY_RE.subn(" ", text)
    cut = re.sub(r"\s+", " ", cut).strip()
    return _trim_query_edges(cut) if n_cut else cut


# "historic" как наполнитель ("historic palace" -> "palace"). Одиночное слово, не входит в
# FORBIDDEN_QUERY_PHRASES: "historic district" допустимо, если эта пара слов есть в тексте
# сегмента (часть названия). "historical" не затрагивается (граница слова).
_HISTORIC_RE = re.compile(r"\bhistoric\b(?:[\s-]+(?P<next>[^\W\d_]+))?", re.IGNORECASE)


def strip_historic_filler(query: str, segment_text: Optional[str] = None) -> tuple[str, int]:
    """Вырезает слово "historic" из запроса, кроме случая, когда пара "historic <слово>"
    дословно есть в тексте сегмента (без учёта регистра). segment_text=None - режутся все.
    Возвращает (запрос, сколько вхождений вырезано)."""
    if not query:
        return query, 0
    text_l = segment_text.lower() if segment_text else ""
    n_cut = 0

    def _repl(m: re.Match) -> str:
        nonlocal n_cut
        nxt = m.group("next")
        if nxt and re.search(r"\bhistoric[\s-]+" + re.escape(nxt.lower()) + r"\b", text_l):
            return m.group(0)
        n_cut += 1
        return (nxt or "") if nxt else " "

    out = _HISTORIC_RE.sub(_repl, query)
    if not n_cut:
        return query, 0
    out = re.sub(r"\s+", " ", out).strip()
    return _trim_query_edges(out), n_cut


# Годы в запросах: Wikimedia/LOC требуют все слова сразу, выдуманный год даёт пустую выдачу.
# Год допустим, только если он назван в тексте (субтитрах) этого же сегмента.
_YEAR_PAT = r"(?:1[0-9]{3}|20[0-9]{2})"
# Вводные слова перед годом режутся вместе с ним (\"early 1900s\", \"circa 1900\", \"in 1920\").
_YEAR_LEAD_WORDS = (
    "circa", "ca", "around", "about", "early", "late", "mid",
    "in", "from", "since", "until", "by", "before", "after", "during", "between",
)
_YEAR_RE = re.compile(
    r"(?P<pre>(?:\b(?:" + "|".join(_YEAR_LEAD_WORDS) + r")\.?\s+)?)"
    r"(?:"
    # диапазон: 1914-1918 / 1914 - 1918 / 1914-18
    r"(?P<ra>\b" + _YEAR_PAT + r")(?:\s*[-\u2013\u2014]\s*(?P<rb>" + _YEAR_PAT + r")\b|[-\u2013\u2014](?P<rc>\d{2})\b)"
    r"|(?P<dec>\b" + _YEAR_PAT + r")s\b"        # десятилетие: 1920s
    r"|(?P<yr>\b" + _YEAR_PAT + r"\b)"          # одиночный год
    r")",
    re.IGNORECASE,
)
_TEXT_YEAR_RE = re.compile(r"(?<!\d)" + _YEAR_PAT + r"(?!\d)")


def strip_unsupported_years(query: str, segment_text: Optional[str] = None) -> tuple[str, int]:
    """Вырезает из запроса годы (1900), десятилетия (1920s) и диапазоны (1914-1918),
    которых нет в тексте сегмента. Возвращает (очищенный_запрос, сколько_вырезано).

    Год остаётся, только если это же число есть в segment_text. Диапазон остаётся, только
    если в тексте есть оба конца; иначе режется целиком. Десятилетие 1920s проверяется по
    числу 1920. segment_text=None (или пустой) - в тексте лет нет, режутся все. Один
    вырезанный год/десятилетие/диапазон считается за 1. Если что-то вырезано, края чистит
    _trim_query_edges (\"Treaty of Sevres in 1920\" -> \"Treaty of Sevres\"); если нет -
    запрос возвращается как есть."""
    known = set(_TEXT_YEAR_RE.findall(segment_text)) if segment_text else set()
    n_cut = 0

    def _repl(m: re.Match) -> str:
        nonlocal n_cut
        if m.group("ra"):
            end = m.group("rb") or (m.group("ra")[:2] + m.group("rc"))
            keep = m.group("ra") in known and end in known
        else:
            keep = (m.group("dec") or m.group("yr")) in known
        if keep:
            return m.group(0)
        n_cut += 1
        return " "

    cut = _YEAR_RE.sub(_repl, query)
    if not n_cut:
        return query, 0
    cut = re.sub(r"\(\s*\)|\[\s*\]", " ", cut)  # опустевшие скобки: \"Vienna (1900)\"
    cut = re.sub(r"\s+", " ", cut).strip()
    return _trim_query_edges(cut), n_cut


# Общие слова, которые сами по себе не имя собственное (fallback_entity_keywords).
_GENERIC_CAPS_WORDS = frozenset({
    "the", "a", "an", "old", "new", "city", "royal", "ancient", "modern", "great", "grand",
    "national", "imperial", "historic", "historical", "medieval", "classic",
    "central", "main", "public", "traditional", "european", "asian", "inner", "outer",
})


# Маленькие служебные слова, которые не рвут имя в середине ("Treaty of Sevres").
_NAME_CONNECTORS = frozenset({"of", "de", "von", "van", "the"})

# Архивные источники: для них fallback работает и при is_entity=false от Gemini.
_ARCHIVE_SITES = frozenset({"wikimedia", "loc", "nasa"})


def fallback_entity_keywords(query: str) -> list[str]:
    """Имена собственные из query_medium без сети (когда Gemini сказал is_entity=true, или
    is_entity=false при архивном первом сайте, но дал пустой entity_keywords).

    Берутся слова с заглавной буквы, кроме общих (The, Old, City, Royal...); соседние такие
    слова склеиваются в одно имя (\"Topkapi Palace\"). Строчные of / de / von / van / the между
    заглавными словами не рвут цепочку (\"Treaty of Sevres\"); в начале и в конце цепочки они
    не включаются. Общее слово или знак препинания разрывают цепочку. Запрос делится на части
    по запятой и точке с запятой. Первое слово части заглавное по правилам письма, поэтому
    цепочка, начинающаяся с него, считается именем, только если в ней 2+ слов или она
    занимает всю часть (\"Vienna\", \"Mehmed VI\", \"Vienna, Istanbul\"); одиночное первое слово
    части, где есть другие слова (\"Vienna street, Istanbul\"), пропускается.
    Нет имён - пустой список."""
    tokens = (query or "").split()
    n_tok = len(tokens)
    # Границы частей запроса: часть закрывает токен с запятой / точкой с запятой на конце
    # (отдельный токен-разделитель закрывает предыдущую часть).
    part_last = [False] * n_tok
    for i, tok in enumerate(tokens):
        core = tok.rstrip(",;:.")
        if "," in tok[len(core):] or ";" in tok[len(core):]:
            part_last[i] = True
            if not core and i > 0:
                part_last[i - 1] = True
    part_last = [last or i == n_tok - 1 for i, last in enumerate(part_last)]
    names: list[str] = []
    run: list[str] = []
    pending: list[str] = []  # служебные слова после run, ещё не подтверждённые заглавным словом
    run_start = -1
    run_end = -1

    def _flush() -> None:
        nonlocal run, pending, run_start, run_end
        if run:
            at_part_start = run_start == 0 or part_last[run_start - 1]
            whole_part = at_part_start and part_last[run_end]
            skip = at_part_start and len(run) < 2 and not whole_part
            if not skip:
                name = " ".join(run)
                if name.casefold() not in {n.casefold() for n in names}:
                    names.append(name)
        run, pending, run_start, run_end = [], [], -1, -1

    for i, tok in enumerate(tokens):
        word = tok.strip(EDGE_STRIP_CHARS + "()[]\"'")
        ends_run = tok != tok.rstrip(",;:.")  # знак препинания после слова закрывает цепочку
        is_cap = (
            bool(word) and word[0].isupper() and word.casefold() not in _GENERIC_CAPS_WORDS
        )
        if is_cap:
            if not run:
                run_start = i
            run.extend(pending)
            pending = []
            run.append(word)
            run_end = i
            if ends_run:
                _flush()
        elif run and word in _NAME_CONNECTORS and not ends_run:
            pending.append(word)
        else:
            _flush()
    _flush()
    return names


def _clean_keywords(raw) -> list[str]:
    """strip, без нестрок/пустых/дублей (дубли - без учёта регистра, остаётся первое
    написание); регистр самих элементов сохраняется."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for kw in raw:
        if not isinstance(kw, str):
            continue
        kw = kw.strip()
        if kw and kw.casefold() not in seen:
            seen.add(kw.casefold())
            out.append(kw)
    return out


def _normalize_entry(
    item: dict, seg_index: int, stats: Optional[Counter] = None,
    segment_text: Optional[str] = None,
) -> dict:
    """Детерминированно проверяет и чинит запись сегмента (без новых вызовов Gemini).

    Структурно битая запись (не объект / нет обязательных ключей) - ValueError, чтобы
    сработали ретраи батча. Остальное чинится на месте; причины правок копятся в stats
    (Counter): по каждой причине - число записей, плюс "_entries" - сколько записей
    исправлено хотя бы раз. Замечания без правки идут в stats с префиксом "note:".

    segment_text - субтитры этого сегмента: годы в запросах остаются, только если названы
    в нём (None - вырезаются все). Число вырезанных годов копится в stats["_years_cut"].

    is_entity всегда выводится: bool(entity_keywords после чистки); значение от Gemini
    игнорируется. Пустой список при is_entity=true от Gemini -> fallback_entity_keywords
    по query_medium; то же при is_entity=false, если первый сайт архивный
    (wikimedia / loc / nasa), для стоковых сегментов - как раньше. Случаи, когда итог != ответу Gemini, копятся в stats["_entity_derived"]."""
    if not isinstance(item, dict):
        raise ValueError(f"Сегмент {seg_index}: запись ответа не объект: {str(item)[:200]}")
    missing = [k for k in REQUIRED_ENTRY_KEYS if k not in item]
    if missing:
        raise ValueError(f"Сегмент {seg_index}: в записи нет обязательных полей {missing}")

    fixes: list[str] = []
    notes: list[str] = []

    # sites: только значения из SITES, без дублей, порядок сохраняется
    raw_sites = item["sites"] if isinstance(item["sites"], list) else []
    sites: list[str] = []
    for site in raw_sites:
        if site in SITES and site not in sites:
            sites.append(site)
    if not sites:
        sites = ["pexels", "pixabay"]
        fixes.append("sites_empty")
        _warn_limited("Сегмент %s: пустой/невалидный sites %r - поставил %s.", seg_index, item["sites"], sites)
    elif sites != raw_sites:
        fixes.append("sites_cleaned")

    # запросы: вырезаем слова-наполнители
    scene = item["scene"].strip() if isinstance(item["scene"], str) else ""
    queries: dict[str, Optional[str]] = {}
    years_cut = 0
    for key in ("query_narrow", "query_medium", "query_broad"):
        raw = item[key]
        if raw is None and key == "query_broad":
            queries[key] = None
            continue
        if not isinstance(raw, str):
            raw = ""
        if FORBIDDEN_QUERY_RE.search(raw):
            fixes.append("forbidden_words_removed")
        cleaned, n_hist = strip_historic_filler(_strip_forbidden(raw), segment_text)
        if n_hist:
            fixes.append("historic_removed")
        cleaned, n_years = strip_unsupported_years(cleaned, segment_text)
        if n_years:
            years_cut += n_years
            fixes.append("unsupported_years_removed")
        queries[key] = cleaned

    narrow, medium, broad = queries["query_narrow"], queries["query_medium"], queries["query_broad"]
    if not narrow or not medium:
        if not narrow and not medium:
            narrow, n_hist = strip_historic_filler(_strip_forbidden(scene), segment_text)
            if n_hist:
                fixes.append("historic_removed")
            narrow, n_years = strip_unsupported_years(narrow, segment_text)
            medium = narrow
            years_cut += n_years
            if not narrow:
                raise ValueError(f"Сегмент {seg_index}: пусты query_narrow, query_medium и scene")
        elif not narrow:
            narrow = medium
        else:
            medium = narrow
        fixes.append("narrow_medium_empty")
        _warn_limited("Сегмент %s: пустой query_narrow/query_medium - подставил замену.", seg_index)
    if broad is not None and not broad:
        broad = None
        fixes.append("broad_emptied")

    # entity_keywords - главное поле; is_entity выводится из него (ответ Gemini игнорируется)
    gemini_is_entity = item["is_entity"]
    raw_keywords = item["entity_keywords"]
    keywords = _clean_keywords(raw_keywords)
    if keywords != raw_keywords:
        fixes.append("keywords_cleaned")
    if not keywords and (gemini_is_entity is True or sites[0] in _ARCHIVE_SITES):
        keywords = fallback_entity_keywords(medium)
        if keywords:
            fixes.append("keywords_fallback")
    is_entity = bool(keywords)
    entity_derived = is_entity is not gemini_is_entity

    # type
    seg_type = item["type"]
    if seg_type not in ("image", "video"):
        _warn_limited("Сегмент %s: недопустимый type %r - поставил 'video'.", seg_index, seg_type)
        seg_type = "video"
        fixes.append("type_fixed")

    if stats is not None:
        for reason in set(fixes):
            stats[reason] += 1
        for note in set(notes):
            stats[note] += 1
        if fixes:
            stats["_entries"] += 1
        if years_cut:
            stats["_years_cut"] += years_cut
        if entity_derived:
            stats["_entity_derived"] += 1

    return {
        "scene": scene,
        "sites": sites,
        "query_narrow": narrow,
        "query_medium": medium,
        "query_broad": broad,
        "type": seg_type,
        "is_entity": is_entity,
        "entity_keywords": keywords,
    }


# ---------------------------------------------------------------------------
# Валидация записей и подготовка повторного запроса
# ---------------------------------------------------------------------------

def _collect_entry_issues(
    entry: dict, mode: int, rx_check: re.Pattern
) -> list[tuple[str, str]]:
    """Собирает проблемы одной записи как пары (текст_ru, текст_en): русский - для лога,
    английский - для промпта REPAIR. Условия проверки заданы в одном месте."""
    archive_sites = frozenset({"wikimedia", "loc", "nasa"})
    stock_sites = frozenset({"pexels", "pixabay"})
    found_issues: list[tuple[str, str]] = []

    # 1. Запрещённые слова-типы в полях scene, query_narrow, query_medium, query_broad
    for field in ("scene", "query_narrow", "query_medium", "query_broad"):
        val = entry.get(field)
        if val and isinstance(val, str):
            found = rx_check.findall(val)
            if found:
                words_uniq = ", ".join(sorted(set(w.lower() for w in found)))
                found_issues.append((
                    f"в поле {field} запрещённый тип кадра: {words_uniq}",
                    f"forbidden shot type '{words_uniq}' in field {field}",
                ))

    # 2. Соответствие sites режиму
    sites = entry.get("sites") or []
    if not sites:
        found_issues.append(("список sites пуст", "sites list is empty"))
    else:
        if mode == 1:
            stock_present = ", ".join(x for x in sites if x in stock_sites)
            if stock_present:
                found_issues.append((
                    f"в режиме 1 (архив) недопустимы стоковые сайты: {stock_present}",
                    f"in mode 1 (archive) stock sites are not allowed: {stock_present}",
                ))
        elif mode == 3:
            arch_present = ", ".join(x for x in sites if x in archive_sites)
            if arch_present:
                found_issues.append((
                    f"в режиме 3 (сток) недопустимы архивные сайты: {arch_present}",
                    f"in mode 3 (stock) archive sites are not allowed: {arch_present}",
                ))

    # 3. Режим 3: is_entity != false ИЛИ entity_keywords непустой
    if mode == 3:
        is_ent = entry.get("is_entity", False)
        kw = entry.get("entity_keywords") or []
        if is_ent is not False:
            found_issues.append((
                f"в режиме 3 (сток) is_entity должен быть false, получено: {is_ent}",
                f"in mode 3 (stock) is_entity must be false, got: {is_ent}",
            ))
        if kw:
            found_issues.append((
                f"в режиме 3 (сток) entity_keywords должен быть пустым, найдено: {kw}",
                f"in mode 3 (stock) entity_keywords must be empty, found: {kw}",
            ))

    return found_issues


def validate_entries(
    entries: dict[str, dict],
    mode: int,
    junk_re: Optional[re.Pattern] = None,
    lang: str = "ru",
) -> dict[str, list[str]]:
    """Пост-проверка записей сегментов на соблюдение ограничений режима и запрещённых типов кадра.
    Возвращает словарь {номер_сегмента: [описания_проблем]}. lang="ru" (по умолчанию) - тексты
    для лога, lang="en" - те же проблемы по-английски для промпта REPAIR."""
    rx_check = junk_re if junk_re is not None else JUNK_KIND_RE
    pos = 1 if lang == "en" else 0
    issues: dict[str, list[str]] = {}

    for idx_str, entry in entries.items():
        seg_issues = [pair[pos] for pair in _collect_entry_issues(entry, mode, rx_check)]
        if seg_issues:
            issues[idx_str] = seg_issues

    return issues


def chunk_repair_indices(indices: list[int], max_size: int = 125) -> list[list[int]]:
    """Разбивает список индексов для повторного запроса (REPAIR) поровну:
    при n <= 125 - один запрос, иначе k = ceil(n/125) запросов по ~n/k."""
    n = len(indices)
    if n == 0:
        return []
    if n <= max_size:
        return [indices]
    k = math.ceil(n / max_size)
    chunk_size = math.ceil(n / k)
    chunks = []
    for i in range(0, n, chunk_size):
        chunks.append(indices[i : i + chunk_size])
    return chunks


def format_neighbors_context(
    target_idx: int,
    segments: list[Segment],
    results: Optional[dict[str, dict]] = None,
    window: int = CONTEXT_WINDOW,
    with_distance: bool = False,
) -> str:
    """Форматирует контекст соседей +-window для блока REPAIR. Источник - ТОЛЬКО текст SRT:
    scene и любые значения из results не показываются (параметр results оставлен ради
    совместимости вызовов и не используется). with_distance=False (круг 1): блоки Context
    BEFORE / AFTER. with_distance=True (круг 2): компактный список, ближайшие первыми, у каждого
    соседа пометка distance N; дальние соседи идут как фон."""
    pos = None
    for i, s in enumerate(segments):
        if s.index == target_idx:
            pos = i
            break
    if pos is None:
        return "(no neighbor context available)"

    def _txt(s: Segment) -> str:
        return " ".join((s.text or "").split()) or "(empty)"

    before = segments[max(0, pos - window) : pos]
    after = segments[pos + 1 : pos + 1 + window]

    lines = []
    if with_distance:
        # before[-d] - сосед слева на расстоянии d, after[d-1] - сосед справа на расстоянии d
        for d in range(1, window + 1):
            if d <= len(before):
                s = before[-d]
                lines.append(f"  distance {d} | before [{s.index}]: {_txt(s)}")
            if d <= len(after):
                s = after[d - 1]
                lines.append(f"  distance {d} | after [{s.index}]: {_txt(s)}")
        if lines:
            lines.insert(
                0,
                f"Neighbors by distance (nearest first; distance 1-{CONTEXT_WINDOW} is the immediate "
                f"surroundings, farther ones are background):",
            )
    else:
        if before:
            lines.append("Context BEFORE:")
            for s in before:
                lines.append(f"  [{s.index}] Text: {_txt(s)}")
        if after:
            lines.append("Context AFTER:")
            for s in after:
                lines.append(f"  [{s.index}] Text: {_txt(s)}")
    return "\n".join(lines) if lines else "(no neighbors)"


# ---------------------------------------------------------------------------
# Вызов Gemini с ретраями
# ---------------------------------------------------------------------------

def call_gemini_batch(
    client: "genai.Client",
    model: str,
    batch: list[Segment],
    context_before: list[Segment] | None = None,
    context_after: list[Segment] | None = None,
    mode: int = DEFAULT_SOURCES_MODE,
    repair_info: dict[int, dict] | None = None,
) -> dict:
    prompt = build_prompt(batch, context_before, context_after, repair_info=repair_info)

    config = types.GenerateContentConfig(
        system_instruction=build_system_instruction(mode),
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
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
            fix_stats: Counter = Counter()
            seg_texts = {s.index: s.text for s in batch}
            for item in parsed:
                if not isinstance(item, dict) or "segment_index" not in item:
                    raise ValueError(f"В элементе ответа нет segment_index: {item}")
                idx = item.pop("segment_index")
                result[str(idx)] = _normalize_entry(item, idx, fix_stats, seg_texts.get(idx))

            got_indices = {int(k) for k in result.keys()}
            missing = expected_indices - got_indices
            extra = got_indices - expected_indices
            if missing or extra:
                raise ValueError(
                    f"Несовпадение номеров сегментов в ответе батча [{batch[0].index}.."
                    f"{batch[-1].index}]: не хватает {sorted(missing)}, лишние {sorted(extra)}"
                )

            fixed_entries = fix_stats.pop("_entries", 0)
            years_cut_total = fix_stats.pop("_years_cut", 0)
            entity_derived_total = fix_stats.pop("_entity_derived", 0)
            logging.info(
                "Нормализация батча [%s..%s]: исправлено записей %s из %s, вырезано годов: %s, "
                "is_entity выведен из ключевых слов: %s%s",
                batch[0].index, batch[-1].index, fixed_entries, len(result), years_cut_total,
                entity_derived_total,
                ("; причины: " + ", ".join(f"{k}={v}" for k, v in sorted(fix_stats.items())))
                if fix_stats else "",
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
    Мутирует exhausted_models при обнаружении новой исчерпанной модели.

    Транзиентные сбои (5xx/таймаут/сеть) ретраятся PREFLIGHT_MAX_ATTEMPTS раз с
    экспоненциальным бэкоффом и джиттером; если не помогло - модель считается временно
    недоступной и берётся следующая. ClientError с дневной 429 - без ретраев.

    Исключения:
      AllModelsQuotaExhaustedError - у ВСЕХ моделей подтверждена дневная квота (код 3);
      ModelUnavailableError - рабочей модели нет, и не только из-за квоты (код 4);
      genai_errors.ClientError (не дневная 429) - ключ/имя модели/доступ, пробрасывается (код 1).
    """
    remaining = list(candidates)
    quota_models: list[str] = []
    unavailable: dict[str, str] = {}
    while remaining:
        model = remaining[0]
        known_exhausted_at = exhausted_models.get(model)
        if known_exhausted_at and _hours_since(known_exhausted_at) < EXHAUSTED_MODEL_TTL_HOURS:
            logging.info(
                "Модель %s уже отмечена исчерпанной %.1fч назад - пропускаю без запроса.",
                model, _hours_since(known_exhausted_at),
            )
            quota_models.append(model)
            remaining.pop(0)
            continue

        logging.info("Preflight: проверяю модель %s (без response_schema)...", model)
        quota_hit = False
        last_error: Optional[BaseException] = None
        for attempt in range(1, PREFLIGHT_MAX_ATTEMPTS + 1):
            attempt_start = time.monotonic()
            try:
                response = client.models.generate_content(
                    model=model,
                    contents="Ответь одним словом: OK",
                    config=types.GenerateContentConfig(
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                )
                if not response.text:
                    raise ValueError("Пустой ответ на проверочный запрос без schema")
                logging.info(
                    "Модель %s доступна (preflight-вызов %.2fs, попытка %s/%s). Ответ: %r",
                    model, time.monotonic() - attempt_start, attempt, PREFLIGHT_MAX_ATTEMPTS,
                    response.text.strip()[:50],
                )
                return model, remaining[1:]
            except genai_errors.ClientError as e:
                elapsed = time.monotonic() - attempt_start
                if _extract_status_code(e) == 429 and _is_daily_quota_error(e):
                    logging.warning(
                        "У модели %s уже исчерпан дневной лимит (preflight-вызов %.2fs): %s",
                        model, elapsed, e,
                    )
                    exhausted_models[model] = _now_iso()
                    quota_hit = True
                    break
                logging.error(
                    "Preflight не пройден для модели %s (код %s, НЕ дневная квота, %.2fs) - "
                    "похоже проблема в ключе/имени модели/доступе, а не в лимитах. "
                    "Полный ответ: %s",
                    model, _extract_status_code(e), elapsed, e,
                )
                raise
            except _PREFLIGHT_TRANSIENT_ERRORS as e:
                last_error = e
                logging.warning(
                    "Preflight модели %s: попытка %s/%s не удалась за %.2fs (%s: %s).",
                    model, attempt, PREFLIGHT_MAX_ATTEMPTS, time.monotonic() - attempt_start,
                    type(e).__name__, e,
                )
                if attempt < PREFLIGHT_MAX_ATTEMPTS:
                    base = PREFLIGHT_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    time.sleep(base * random.uniform(0.5, 1.5))  # джиттер +-50%

        remaining.pop(0)
        if quota_hit:
            quota_models.append(model)
            continue
        unavailable[model] = f"{type(last_error).__name__}: {last_error}"
        logging.error(
            "Модель %s временно недоступна после %s попыток (%s) - пробую следующую, если есть.",
            model, PREFLIGHT_MAX_ATTEMPTS, unavailable[model],
        )

    if unavailable:
        raise ModelUnavailableError(unavailable, quota_models)
    raise AllModelsQuotaExhaustedError(
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
    exhausted = data.get("exhausted_models", {})
    if data.get("schema_version") != SCHEMA_VERSION:
        # Формат записей изменился (или чекпоинт старый, без версии) - результаты из него
        # не подходят. Отбрасываем их, но помеченные исчерпанные модели сохраняем: квота
        # от формата не зависит.
        logging.warning(
            "Чекпоинт %s сохранён со схемой v%s, текущая схема v%s - старые результаты "
            "(%s сегментов) игнорирую, генерация начнётся с нуля.",
            path, data.get("schema_version", "?"), SCHEMA_VERSION, len(data.get("results", {})),
        )
        return {}, exhausted
    results = data.get("results", {})
    if results:
        logging.info("Найден чекпоинт: %s сегментов уже обработано ранее.", len(results))
    return results, exhausted


def save_checkpoint(path: str, src_hash: str, results: dict, exhausted_models: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_hash": src_hash,
                "schema_version": SCHEMA_VERSION,
                "results": results,
                "exhausted_models": exhausted_models,
            },
            f, ensure_ascii=False,
        )
    os.replace(tmp_path, path)  # атомарная замена, не оставляет битый файл при сбое на записи


# ---------------------------------------------------------------------------
# Повторный запрос REPAIR и завершение
# ---------------------------------------------------------------------------

def _run_repair_round(
    client: "genai.Client",
    current_model: str,
    fallback_queue: list[str],
    segments: list[Segment],
    results: dict[str, dict],
    exhausted_models: dict[str, str],
    checkpoint_path: str,
    src_hash: str,
    sources_mode: int,
    indices: list[int],
    round_num: int,
    call_batch_fn=call_gemini_batch,
) -> tuple[Optional[int], str]:
    """Один круг REPAIR по номерам indices (чанки по chunk_repair_indices). Обновляет results,
    fallback_queue, exhausted_models и чекпоинт на месте. Возвращает (код, модель): код None -
    круг завершён, иначе код возврата процесса (3 - квоты исчерпаны, 1 - ошибка вызова)."""
    seg_by_idx = {s.index: s for s in segments}
    repair_chunks = chunk_repair_indices(indices, max_size=125)
    # Круг 1: +-CONTEXT_WINDOW с блоками Context BEFORE/AFTER; круг 2: +-REPAIR2_CONTEXT_WINDOW
    # с пометками расстояния. В обоих кругах соседи - только текст SRT.
    neighbors_window = REPAIR2_CONTEXT_WINDOW if round_num == 2 else CONTEXT_WINDOW

    for chunk_num, chunk_indices in enumerate(repair_chunks, start=1):
        chunk_segs = [seg_by_idx[i] for i in chunk_indices]
        first_idx, last_idx = chunk_segs[0].index, chunk_segs[-1].index
        first_pos = next(i for i, s in enumerate(segments) if s.index == first_idx)
        last_pos = next(i for i, s in enumerate(segments) if s.index == last_idx)
        ctx_before = segments[max(0, first_pos - CONTEXT_WINDOW) : first_pos]
        ctx_after = segments[last_pos + 1 : last_pos + 1 + CONTEXT_WINDOW]

        # В промпт уходят английские формулировки проблем, в лог - русские
        issues_en = validate_entries(
            {str(i): results[str(i)] for i in chunk_indices}, sources_mode, lang="en"
        )
        chunk_repair_info = {}
        for s in chunk_segs:
            chunk_repair_info[s.index] = {
                "entry": results.get(str(s.index), {}),
                "issues": issues_en.get(str(s.index), []),
                "neighbors": format_neighbors_context(
                    s.index, segments, None, neighbors_window, with_distance=(round_num == 2)
                ),
                "round": round_num,
                "mode": sources_mode,
            }

        logging.info(
            "REPAIR круг %s/2: чанк %s/%s: %s сегментов [%s..%s] (модель: %s)",
            round_num, chunk_num, len(repair_chunks), len(chunk_segs), first_idx, last_idx, current_model,
        )

        while True:
            try:
                repaired_batch = call_batch_fn(
                    client, current_model, chunk_segs, ctx_before, ctx_after,
                    mode=sources_mode, repair_info=chunk_repair_info,
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
                        "Дневной лимит исчерпан во время REPAIR, а запасных моделей больше нет. "
                        "Прогресс сохранён в чекпоинте %s.", checkpoint_path,
                    )
                    return 3, current_model
                current_model = fallback_queue.pop(0)
                logging.warning("Переключаюсь на запасную модель: %s", current_model)
                continue
            except Exception as e:
                logging.error(
                    "REPAIR круг %s/2, чанк %s..%s не обработан после %s попыток: %s. Прерываю выполнение, "
                    "requests.json НЕ будет записан (чекпоинт сохранён в %s).",
                    round_num, first_idx, last_idx, MAX_RETRIES, e, checkpoint_path,
                )
                save_checkpoint(checkpoint_path, src_hash, results, exhausted_models)
                return 1, current_model

        results.update(repaired_batch)
        save_checkpoint(checkpoint_path, src_hash, results, exhausted_models)

    return None, current_model


def run_repair_cycle(
    client: "genai.Client",
    current_model: str,
    fallback_queue: list[str],
    segments: list[Segment],
    results: dict[str, dict],
    exhausted_models: dict[str, str],
    checkpoint_path: str,
    src_hash: str,
    sources_mode: int,
    strict_mode: int,
    output_path: str,
    call_batch_fn=call_gemini_batch,
) -> int:
    """Выполняет пост-проверку записей, до двух кругов исправления REPAIR через Gemini
    (круг 2 - только для номеров, оставшихся с нарушениями после круга 1; третьего круга нет),
    логирует статистику источников и сохраняет requests.json."""
    initial_issues = validate_entries(results, sources_mode)
    problem_indices = sorted(int(k) for k in initial_issues.keys())
    post_issues_1: dict[str, list[str]] = {}
    post_issues_2: dict[str, list[str]] = {}
    round2_ran = False

    if problem_indices:
        logging.warning(
            "Обнаружены ошибки валидации в %s сегментах. Запускаю повторный запрос (REPAIR)...",
            len(problem_indices),
        )
        code, current_model = _run_repair_round(
            client, current_model, fallback_queue, segments, results, exhausted_models,
            checkpoint_path, src_hash, sources_mode, problem_indices, 1, call_batch_fn,
        )
        if code is not None:
            return code

        # Перепроверка только исправленных номеров круга 1
        post_issues_1 = validate_entries({str(i): results[str(i)] for i in problem_indices}, sources_mode)

        if post_issues_1:
            round2_indices = sorted(int(k) for k in post_issues_1.keys())
            logging.warning(
                "После круга 1 остались нарушения в %s сегментах. Запускаю круг 2 REPAIR (последний)...",
                len(round2_indices),
            )
            code, current_model = _run_repair_round(
                client, current_model, fallback_queue, segments, results, exhausted_models,
                checkpoint_path, src_hash, sources_mode, round2_indices, 2, call_batch_fn,
            )
            if code is not None:
                return code
            round2_ran = True
            # Перепроверка только номеров круга 2
            post_issues_2 = validate_entries({str(i): results[str(i)] for i in round2_indices}, sources_mode)

    post_issues = post_issues_2 if round2_ran else post_issues_1

    if round2_ran:
        logging.info(
            "Валидация: проблемных сегментов до повтора: %s, после круга 1: %s, после круга 2: %s",
            len(initial_issues), len(post_issues_1), len(post_issues_2),
        )
    else:
        logging.info(
            "Валидация: проблемных сегментов до повтора: %s, после круга 1: %s, круг 2 не требовался",
            len(initial_issues), len(post_issues_1),
        )

    archive_count = sum(
        1 for e in results.values()
        if e.get("sites") and e["sites"][0] in ("wikimedia", "loc", "nasa")
    )
    stock_count = sum(
        1 for e in results.values()
        if e.get("sites") and e["sites"][0] in ("pexels", "pixabay")
    )
    total_entries = len(results)
    arch_pct = (archive_count / total_entries * 100.0) if total_entries else 0.0
    stock_pct = (stock_count / total_entries * 100.0) if total_entries else 0.0
    logging.info(
        "Источники: архив %s (%.1f%%), сток %s (%.1f%%)",
        archive_count, arch_pct, stock_count, stock_pct,
    )

    ordered = {str(k): results[str(k)] for k in sorted(int(k) for k in results.keys())}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)

    if post_issues:
        problem_items = sorted(post_issues.items(), key=lambda x: int(x[0]))
        for idx_str, problems in problem_items[:MAX_NORMALIZE_WARNINGS]:
            msg = f"Сегмент {idx_str}: {'; '.join(problems)}"
            if strict_mode == 1:
                logging.error("%s", msg)
            else:
                logging.warning("%s", msg)

        if len(problem_items) > MAX_NORMALIZE_WARNINGS:
            limit_msg = (
                f"Достигнут лимит отдельных сообщений ({len(problem_items)} сегментов с ошибками) - "
                f"показаны первые {MAX_NORMALIZE_WARNINGS}."
            )
            if strict_mode == 1:
                logging.error("%s", limit_msg)
            else:
                logging.warning("%s", limit_msg)

        summary_msg = (
            f"Остались нарушения валидации в {len(problem_items)} сегментах: "
            f"{sorted(int(k) for k in post_issues.keys())}"
        )
        if strict_mode == 1:
            logging.error("%s", summary_msg)
            logging.error(
                "GENQ_STRICT=1: валидация не пройдена. requests.json записан, "
                "чекпоинт НЕ удалён (%s).", checkpoint_path,
            )
            return 1
        else:
            logging.warning("%s", summary_msg)
            logging.warning(
                "GENQ_STRICT=2: есть нарушения валидации, но включён мягкий режим. "
                "requests.json записан, завершаю с кодом 0.",
            )
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
            return 0

    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)

    logging.info("Готово: %s сегментов записано в %s", len(ordered), output_path)
    return 0


# ---------------------------------------------------------------------------
# Разбиение на батчи и self-tests
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


def run_self_tests() -> int:
    """Self-tests без сети: python generate_queries.py --self-test. 0 - все прошли, 1 - есть падения."""
    failures: list[str] = []

    def check(name: str, got, want) -> None:
        ok = got == want
        print(("PASS" if ok else "FAIL") + f"  {name}" + ("" if ok else f"\n      got:  {got!r}\n      want: {want!r}"))
        if not ok:
            failures.append(name)

    def entry(**over) -> dict:
        base = {
            "scene": "A view.", "sites": ["pexels"], "query_narrow": "Topkapi Palace gate",
            "query_medium": "Topkapi Palace", "query_broad": "palace", "type": "image",
            "is_entity": False, "entity_keywords": [],
        }
        base.update(over)
        return base

    saved_warn = _normalize_warn_left[0]
    _normalize_warn_left[0] = 0  # тесты не должны тратить лимит warning
    try:
        # --- годы (E1) ---
        check("years: год не из текста вырезается",
              strip_unsupported_years("Treaty of Sevres in 1920"), ("Treaty of Sevres", 1))
        check("years: год из текста остаётся",
              strip_unsupported_years("Treaty of Sevres 1920", "в 1920 году"), ("Treaty of Sevres 1920", 0))
        check("years: диапазон без текста",
              strip_unsupported_years("trenches 1914-1918"), ("trenches", 1))
        check("years: десятилетие и скобки",
              strip_unsupported_years("Vienna (1900) street early 1920s"), ("Vienna street", 2))
        st: Counter = Counter()
        e = _normalize_entry(entry(query_medium="Vienna 1900", query_narrow="Vienna 1900",
                                   entity_keywords=["Vienna"]), 1, st, None)
        check("years: _normalize_entry режет год и считает", (e["query_medium"], st["_years_cut"]), ("Vienna", 2))

        # --- is_entity / entity_keywords (E2) ---
        st = Counter()
        e = _normalize_entry(entry(is_entity=False, entity_keywords=["Istanbul", "Стамбул"]), 1, st)
        check("ключевые слова есть, Gemini=false -> true",
              (e["is_entity"], e["entity_keywords"], st["_entity_derived"]),
              (True, ["Istanbul", "Стамбул"], 1))

        st = Counter()
        e = _normalize_entry(entry(is_entity=True, entity_keywords=[], query_medium="Topkapi Palace"), 2, st)
        check("пустой список, Gemini=true, medium 'Topkapi Palace' -> fallback",
              (e["is_entity"], e["entity_keywords"], st["_entity_derived"]),
              (True, ["Topkapi Palace"], 0))

        st = Counter()
        e = _normalize_entry(entry(is_entity=True, entity_keywords=[], query_narrow="royal throne room",
                                   query_medium="royal throne room"), 3, st)
        check("пустой список, Gemini=true, 'royal throne room' -> false",
              (e["is_entity"], e["entity_keywords"], st["_entity_derived"]), (False, [], 1))

        st = Counter()
        e = _normalize_entry(entry(is_entity=True, entity_keywords=["  Vienna ", "", "  ", "Вена", "vienna", "Вена", "NATO"]), 4, st)
        check("дубли/пустые чистятся, регистр сохраняется",
              (e["entity_keywords"], e["is_entity"]), (["Vienna", "Вена", "NATO"], True))

        e = _normalize_entry(entry(is_entity=True, entity_keywords=["Vienna"]), 5, Counter())
        check("is_entity=true и список непуст -> без изменений",
              (e["is_entity"], e["entity_keywords"]), (True, ["Vienna"]))

        e = _normalize_entry(entry(is_entity=False, entity_keywords=[], query_medium="crowd street"), 6, Counter())
        check("false и пустой список -> false", (e["is_entity"], e["entity_keywords"]), (False, []))

        # --- fallback_entity_keywords ---
        check("fallback: склейка соседних", fallback_entity_keywords("Topkapi Palace"), ["Topkapi Palace"])
        check("fallback: общие слова пропускаются", fallback_entity_keywords("The Old City Istanbul skyline"), ["Istanbul"])
        check("fallback: одиночное первое слово пропускается", fallback_entity_keywords("Street in Vienna"), ["Vienna"])
        check("fallback: два имени", fallback_entity_keywords("Sultan Abdulmecid II and Istanbul"), ["Sultan Abdulmecid II", "Istanbul"])
        check("fallback: нет имён", fallback_entity_keywords("royal throne room"), [])
        check("fallback: пусто", fallback_entity_keywords(""), [])
        check("fallback: одиночное слово-запрос", fallback_entity_keywords("Vienna"), ["Vienna"])
        check("fallback: одиночное слово-запрос 2", fallback_entity_keywords("Istanbul"), ["Istanbul"])
        check("fallback: Treaty of Sevres", fallback_entity_keywords("Treaty of Sevres"), ["Treaty of Sevres"])
        check("fallback: Battle of Vienna", fallback_entity_keywords("Battle of Vienna"), ["Battle of Vienna"])
        check("fallback: Duke of Wellington", fallback_entity_keywords("Duke of Wellington"), ["Duke of Wellington"])
        check("fallback: Vienna street -> []", fallback_entity_keywords("Vienna street"), [])
        check("fallback: Mehmed VI", fallback_entity_keywords("Mehmed VI"), ["Mehmed VI"])
        check("fallback: Topkapi Palace", fallback_entity_keywords("Topkapi Palace"), ["Topkapi Palace"])
        check("fallback: предлог в конце не включается", fallback_entity_keywords("Treaty of"), [])
        check("fallback: предлог в начале не включается", fallback_entity_keywords("the Sevres treaty gate"), ["Sevres"])
        check("fallback: части через запятую", fallback_entity_keywords("Vienna, Istanbul"), ["Vienna", "Istanbul"])
        check("fallback: части через ;", fallback_entity_keywords("Vienna; Treaty of Sevres"), ["Vienna", "Treaty of Sevres"])
        check("fallback: часть с другими словами", fallback_entity_keywords("Vienna street, Istanbul"), ["Istanbul"])
        check("fallback: первое слово части + имя", fallback_entity_keywords("Street in Vienna, Istanbul"), ["Vienna", "Istanbul"])
        e = _normalize_entry(entry(is_entity=False, entity_keywords=[], sites=["wikimedia", "pexels"],
                                   query_medium="Topkapi Palace"), 7, Counter())
        check("архивный сегмент, is_entity=false, пусто -> true",
              (e["is_entity"], e["entity_keywords"]), (True, ["Topkapi Palace"]))
        e = _normalize_entry(entry(is_entity=False, entity_keywords=[], sites=["pexels", "wikimedia"],
                                   query_medium="Topkapi Palace"), 8, Counter())
        check("стоковый первый сайт, is_entity=false -> false",
              (e["is_entity"], e["entity_keywords"]), (False, []))

        # --- _trim_query_edges (13; восстановлены по докстрингу, см. допущения) ---
        for i, (src, want) in enumerate([
            ("of crowd", "crowd"), ("  crowd  ", "crowd"), (", crowd ;", "crowd"),
            ("crowd of", "crowd"), ("crowd the", "crowd"), ("the Hague", "the Hague"),
            ("The Beatles", "The Beatles"), ("of and in", ""), ("crowd , , street", "crowd, street"),
            ("In The Crowd", "The Crowd"), ("crowd, street.", "crowd, street"),
            ("", ""), ("Treaty of Sevres", "Treaty of Sevres"),
        ], 1):
            check(f"trim {i}: {src!r}", _trim_query_edges(src), want)

        # --- historic (E3) ---
        check("historic: palace", strip_historic_filler("historic palace"), ("palace", 1))
        check("historic: city gate", strip_historic_filler("historic city gate"), ("city gate", 1))
        check("historic: в середине, регистр", strip_historic_filler("Vienna Historic skyline"), ("Vienna skyline", 1))
        check("historic: district нет в тексте -> режется",
              strip_historic_filler("historic district Vienna", "Мы гуляли по Вене"), ("district Vienna", 1))
        check("historic: district есть в тексте -> остаётся",
              strip_historic_filler("historic district Vienna", "the Historic District of Vienna"),
              ("historic district Vienna", 0))
        check("historic: palace не из текста, а district из текста",
              strip_historic_filler("historic palace", "historic district"), ("palace", 1))
        check("historic: historical не трогается", strip_historic_filler("historical palace"), ("historical palace", 0))
        check("historic: без вхождений", strip_historic_filler("Topkapi Palace"), ("Topkapi Palace", 0))
        check("historic: запрос только из слова", strip_historic_filler("historic"), ("", 1))
        st = Counter()
        e = _normalize_entry(entry(query_narrow="historic city gate", query_medium="historic palace",
                                   query_broad="historic palace"), 9, st, None)
        check("historic: _normalize_entry чистит и считает",
              (e["query_narrow"], e["query_medium"], e["query_broad"], st["historic_removed"]),
              ("city gate", "palace", "palace", 1))

        # --- сквозь call_gemini_batch: строка "Нормализация батча" и лимит warning (E3) ---
        import types as _t
        payload = [dict(entry(query_medium="historic palace 1900", query_narrow="historic palace 1900",
                              is_entity=False, entity_keywords=["Vienna"], sites=["bogus"],
                              type="weird"), segment_index=i) for i in range(1, 31)]
        fake = _t.SimpleNamespace(models=_t.SimpleNamespace(
            generate_content=lambda **kw: _t.SimpleNamespace(text=json.dumps(payload))))
        segs = [Segment(i, "0", "1", "text") for i in range(1, 31)]

        class _Cap(logging.Handler):
            def __init__(self):
                super().__init__()
                self.recs = []

            def emit(self, rec):
                self.recs.append((rec.levelno, rec.getMessage()))

        cap = _Cap()
        root = logging.getLogger()
        old_level = root.level
        root.addHandler(cap)
        root.setLevel(logging.INFO)
        _normalize_warn_left[0] = MAX_NORMALIZE_WARNINGS  # реальный счётчик для этого теста
        try:
            call_gemini_batch(fake, "m", segs)
        finally:
            root.removeHandler(cap)
            root.setLevel(old_level)
            _normalize_warn_left[0] = 0
        norm = [m for lv, m in cap.recs if m.startswith("Нормализация батча")]
        check("лог: одна строка нормализации", len(norm), 1)
        line = norm[0] if norm else ""
        check("лог: оба новых счётчика",
              ("вырезано годов: 60" in line, "is_entity выведен из ключевых слов: 30" in line), (True, True))
        check("лог: прежний формат", line.startswith("Нормализация батча [1..30]: исправлено записей 30 из 30, вырезано годов:"), True)
        check("лог: причины (в т.ч. historic_removed)",
              ("; причины: " in line, "historic_removed=30" in line, "sites_empty=30" in line), (True, True, True))
        warns = [m for lv, m in cap.recs if lv == logging.WARNING]
        n_limit_notes = sum(m.startswith("Достигнут лимит") for m in warns)
        check("лимит: отдельных warning ровно MAX_NORMALIZE_WARNINGS",
              len(warns) - n_limit_notes, MAX_NORMALIZE_WARNINGS)
        check("лимит: одно итоговое уведомление о лимите", n_limit_notes, 1)

        # ===================================================================
        # Новые тесты: JUNK_KIND_RE, validate_entries, REPAIR, чанки, режимы
        # ===================================================================

        # 1. JUNK_KIND_RE: границы слов и множественное число
        check("junk_re: documentary не совпадает", bool(JUNK_KIND_RE.search("a documentary film")), False)
        check("junk_re: documentation не совпадает", bool(JUNK_KIND_RE.search("ancient documentation")), False)
        check("junk_re: Maps совпадает", bool(JUNK_KIND_RE.search("Historic Maps of Europe")), True)
        check("junk_re: map в единственном числе", bool(JUNK_KIND_RE.search("a road map")), True)
        check("junk_re: calendar совпадает", bool(JUNK_KIND_RE.search("calendar page")), True)
        check("junk_re: calendars совпадает", bool(JUNK_KIND_RE.search("wall calendars")), True)
        check("junk_re: coat of arms совпадает", bool(JUNK_KIND_RE.search("royal coat of arms")), True)
        check("junk_re: coats of arms совпадает", bool(JUNK_KIND_RE.search("several coats of arms")), True)
        check("junk_re: ID card совпадает", bool(JUNK_KIND_RE.search("driver ID card")), True)
        check("junk_re: ID cards совпадает", bool(JUNK_KIND_RE.search("two id cards")), True)
        check("junk_re: passport совпадает", bool(JUNK_KIND_RE.search("open passport")), True)
        check("junk_re: passports совпадает", bool(JUNK_KIND_RE.search("foreign passports")), True)

        # 2. Динамическая сборка JUNK_KIND_RE и промпта из JUNK_KIND_WORDS (правка 2)
        test_junk_words = JUNK_KIND_WORDS + ("poster",)
        test_re = build_junk_kind_re(test_junk_words)
        test_prompt = build_system_instruction(2, junk_words=test_junk_words)
        check("динамический junk: poster найден в test_re", bool(test_re.search("vintage posters on wall")), True)
        check("динамический junk: poster отсутствует в штатном JUNK_KIND_RE", bool(JUNK_KIND_RE.search("vintage posters on wall")), False)
        check("динамический junk: poster присутствует в test_prompt", "poster" in test_prompt, True)

        # 3. validate_entries: проверка полей и слов-типов
        v_entries = {
            "101": entry(scene="A map of London", sites=["pexels"]),
            "102": entry(query_narrow="calendar 1920", sites=["pexels"]),
            "103": entry(query_medium="coat of arms", sites=["pexels"]),
            "104": entry(query_broad="passports", sites=["pexels"]),
            "105": entry(scene="Clean street", query_narrow="street", query_medium="street", query_broad="street"),
        }
        v_res = validate_entries(v_entries, mode=2)
        check("validate: 101 scene map", "101" in v_res and any("scene" in s and "map" in s for s in v_res["101"]), True)
        check("validate: 102 query_narrow calendar", "102" in v_res and any("query_narrow" in s and "calendar" in s for s in v_res["102"]), True)
        check("validate: 103 query_medium coat of arms", "103" in v_res and any("query_medium" in s and "coat of arms" in s for s in v_res["103"]), True)
        check("validate: 104 query_broad passports", "104" in v_res and any("query_broad" in s and "passports" in s for s in v_res["104"]), True)
        check("validate: 105 без ошибок", "105" in v_res, False)

        # 4. validate_entries: соответствие sites режимам 1 и 3
        v_sites = {
            "201": entry(sites=["pexels", "wikimedia"]),
            "202": entry(sites=["wikimedia", "loc"]),
            "203": entry(sites=["wikimedia", "pexels"]),
            "204": entry(sites=["pexels", "pixabay"]),
            "205": entry(sites=[]),
        }
        res_m1 = validate_entries(v_sites, mode=1)
        check("validate режим 1: pexels запрещён", "201" in res_m1 and any("режиме 1" in s for s in res_m1["201"]), True)
        check("validate режим 1: wikimedia разрешён", "202" in res_m1, False)
        check("validate список sites пуст", "205" in res_m1 and any("список sites пуст" in s for s in res_m1["205"]), True)

        res_m3 = validate_entries(v_sites, mode=3)
        check("validate режим 3: wikimedia запрещён", "203" in res_m3 and any("режиме 3" in s for s in res_m3["203"]), True)
        check("validate режим 3: pexels разрешён", "204" in res_m3, False)

        # 5. validate_entries: режим 3 и entity_keywords / is_entity (правка 1)
        v_mode3 = {
            "301": entry(sites=["pexels"], is_entity=False, entity_keywords=[]),
            "302": entry(sites=["pexels"], is_entity=False, entity_keywords=["Paris"]),
            "303": entry(sites=["pexels"], is_entity=True, entity_keywords=[]),
            "304": entry(sites=["pexels"], is_entity=True, entity_keywords=["London"]),
        }
        res_m3_ent = validate_entries(v_mode3, mode=3)
        check("validate режим 3: is_entity=false и keywords=[] -> OK", "301" in res_m3_ent, False)
        check("validate режим 3: keywords непустой -> нарушение", "302" in res_m3_ent and any("entity_keywords" in s for s in res_m3_ent["302"]), True)
        check("validate режим 3: is_entity=true при пустом keywords -> нарушение (правка 1)",
              "303" in res_m3_ent and any("is_entity" in s for s in res_m3_ent["303"]), True)
        check("validate режим 3: оба поля нарушены", "304" in res_m3_ent and len(res_m3_ent["304"]) >= 2, True)

        # 6. chunk_repair_indices: разбиение проблемных номеров
        check("чанки: 10 номеров -> 1 запрос", [len(c) for c in chunk_repair_indices(list(range(10)))], [10])
        check("чанки: 125 номеров -> 1 запрос", [len(c) for c in chunk_repair_indices(list(range(125)))], [125])
        check("чанки: 126 номеров -> 2 запроса по 63", [len(c) for c in chunk_repair_indices(list(range(126)))], [63, 63])
        check("чанки: 230 номеров -> 2 запроса по 115", [len(c) for c in chunk_repair_indices(list(range(230)))], [115, 115])
        check("чанки: 300 номеров -> 3 запроса по 100", [len(c) for c in chunk_repair_indices(list(range(300)))], [100, 100, 100])
        check("чанки: пустой список", chunk_repair_indices([]), [])

        # 7. build_system_instruction для каждого режима
        p1 = build_system_instruction(1)
        p2 = build_system_instruction(2)
        p3 = build_system_instruction(3)
        check("промпт режим 1: содержит MODE 1: ARCHIVE ONLY", "MODE 1: ARCHIVE ONLY" in p1, True)
        check("промпт режим 1: содержит запрещённые слова", "calendar" in p1 and "map" in p1, True)
        check("промпт режим 2: содержит MODE 2: MIXED", "MODE 2: MIXED" in p2, True)
        check("промпт режим 3: содержит MODE 3: STOCK ONLY", "MODE 3: STOCK ONLY" in p3, True)

        # 8. build_prompt: блок REPAIR и соседи
        seg_rep = Segment(42, "00:01:00,000", "00:01:05,000", "Reviewing war maps")
        rep_data = {
            42: {
                "entry": entry(scene="Looking at a map of battles", query_narrow="battle map"),
                "issues": ["forbidden shot type 'map' in field scene"],
                "neighbors": "  [41] Text: Before\n  [43] Text: After",
            }
        }
        rep_prompt = build_prompt([seg_rep], repair_info=rep_data)
        check("prompt REPAIR: заголовок режима", "REPAIR MODE" in rep_prompt, True)
        check("prompt REPAIR: мусорный сегмент без предыдущей сцены", "Looking at a map of battles" in rep_prompt, False)
        check("prompt REPAIR: содержит проблему", "forbidden shot type 'map' in field scene" in rep_prompt, True)
        check("prompt REPAIR: содержит соседей (текст SRT)", "[41] Text: Before" in rep_prompt, True)

        # 8а. validate_entries(lang="en") и lang по умолчанию
        en_in = {
            "1": entry(scene="A calendar page", sites=["pexels"]),
            "2": entry(sites=["pexels"]),
            "3": entry(sites=["wikimedia"], is_entity=True, entity_keywords=["Rome"]),
            "4": entry(sites=[]),
        }
        en_m1 = validate_entries(en_in, mode=1, lang="en")
        en_m3 = validate_entries(en_in, mode=3, lang="en")
        ru_m1 = validate_entries(en_in, mode=1)
        check("validate en: слово-тип", "forbidden shot type 'calendar' in field scene" in en_m1["1"], True)
        check("validate en: режим 1", "in mode 1 (archive) stock sites are not allowed: pexels" in en_m1["2"], True)
        check("validate en: режим 3 архивные сайты",
              "in mode 3 (stock) archive sites are not allowed: wikimedia" in en_m3["3"], True)
        check("validate en: режим 3 is_entity", "in mode 3 (stock) is_entity must be false, got: True" in en_m3["3"], True)
        check("validate en: режим 3 entity_keywords",
              "in mode 3 (stock) entity_keywords must be empty, found: ['Rome']" in en_m3["3"], True)
        check("validate en: пустой sites", "sites list is empty" in en_m1["4"], True)
        check("validate ru по умолчанию: слово-тип", "в поле scene запрещённый тип кадра: calendar" in ru_m1["1"], True)
        check("validate ru по умолчанию: режим 1",
              "в режиме 1 (архив) недопустимы стоковые сайты: pexels" in ru_m1["2"], True)
        check("validate ru по умолчанию: пустой sites", "список sites пуст" in ru_m1["4"], True)
        check("validate: ru и en дают одинаковые номера и число проблем",
              {k: len(v) for k, v in en_m3.items()}, {k: len(v) for k, v in validate_entries(en_in, mode=3).items()})

        # 8б. build_prompt REPAIR: правила, английские проблемы, скрытие прежнего кадра
        seg_a = Segment(57, "00:02:00,000", "00:02:03,000", "In 1920 everything changed")
        seg_b = Segment(58, "00:02:03,000", "00:02:06,000", "The market opened")
        old_a = entry(scene="A historic calendar page", query_narrow="calendar sheet wood",
                      query_medium="calendar sheet", query_broad="calendar")
        old_b = entry(scene="Busy bazaar stalls", sites=["wikimedia"], query_narrow="bazaar stalls old",
                      query_medium="bazaar stalls", query_broad="bazaar")
        iss_en = validate_entries({"57": old_a, "58": old_b}, mode=3, lang="en")
        rep2 = {
            57: {"entry": old_a, "issues": iss_en["57"], "neighbors": "  [56] Text: X"},
            58: {"entry": old_b, "issues": iss_en["58"], "neighbors": "  [57] Text: Z"},
        }
        rp = build_prompt([seg_a, seg_b], repair_info=rep2)
        check("prompt REPAIR: английская проблема", "forbidden shot type 'calendar' in field scene" in rp, True)
        check("prompt REPAIR: нет русских слов проблем", "запрещённый" in rp or "недопустимы" in rp, False)
        check("prompt REPAIR: все слова JUNK_KIND_WORDS", all(w in rp for w in JUNK_KIND_WORDS), True)
        check("prompt REPAIR: фраза про near-synonyms", "near-synonyms" in rp, True)
        check("prompt REPAIR: мусорный сегмент без старой scene", "A historic calendar page" in rp, False)
        check("prompt REPAIR: мусорный сегмент без старых query",
              "calendar sheet wood" in rp or "Previous query_broad: calendar" in rp, False)
        check("prompt REPAIR: мусорный сегмент сохраняет текст и соседей",
              "In 1920 everything changed" in rp and "[56] Text: X" in rp, True)
        check("prompt REPAIR: сегмент только с sites сохраняет старые значения",
              "Busy bazaar stalls" in rp and "Previous query_narrow: bazaar stalls old" in rp, True)

        # 7а. Системный промпт: три вопроса, подсказка про плоские листы, слова из константы
        for md, pm in ((1, p1), (2, p2), (3, p3)):
            check(f"системный промпт режим {md}: три вопроса (где / кто / что снимаемое)",
                  all(q in pm for q in ("(1) WHERE", "(2) WHO", "(3) WHAT")), True)
            check(f"системный промпт режим {md}: подсказка про плоские листы с текстом",
                  all(w in pm for w in ("paper", "parchment", "manuscript", "scroll")) and "MAIN subject" in pm, True)
            check(f"системный промпт режим {md}: все слова JUNK_KIND_WORDS из константы",
                  all(w in pm for w in JUNK_KIND_WORDS), True)
            check(f"системный промпт режим {md}: ссылка на ABSTRACT / GENERAL SEGMENTS из правила scene",
                  pm.index("ABSTRACT / GENERAL SEGMENTS") < pm.index("2. sites"), True)
        check("плоские листы не добавлены в валидатор",
              JUNK_KIND_RE.search("old paper scroll letter page parchment manuscript"), None)

        # 8в. format_neighbors_context: только текст SRT, без scene и без значений results
        nb_segs = [Segment(i, "00:00:00,000", "00:00:01,000", f"Text{i}") for i in range(1, 31)]
        nb_res = {str(i): entry(scene=f"SCENE_OF_{i}") for i in range(1, 31)}
        nb1 = format_neighbors_context(15, nb_segs, nb_res, CONTEXT_WINDOW)
        check("neighbors круг 1: текст SRT соседей", "[12] Text: Text12" in nb1 and "[18] Text: Text18" in nb1, True)
        check("neighbors круг 1: окно +-3", "[11]" not in nb1 and "[19]" not in nb1, True)
        check("neighbors круг 1: нет scene", "Scene" not in nb1 and "SCENE_OF" not in nb1, True)
        check("neighbors круг 1: блоки BEFORE/AFTER", "Context BEFORE:" in nb1 and "Context AFTER:" in nb1, True)
        check("константы окон: круг 1 = 3, круг 2 = 10", (CONTEXT_WINDOW, REPAIR2_CONTEXT_WINDOW), (3, 10))
        nb2 = format_neighbors_context(15, nb_segs, nb_res, REPAIR2_CONTEXT_WINDOW, with_distance=True)
        check("neighbors круг 2: нет scene", "Scene" not in nb2 and "SCENE_OF" not in nb2, True)
        check("neighbors круг 2: окно 10",
              "[5]" in nb2 and "[25]" in nb2 and "[4]" not in nb2 and "[26]" not in nb2 and "[15]" not in nb2, True)
        check("neighbors круг 2: ближайшие первыми",
              nb2.index("distance 1 |") < nb2.index("distance 2 |") < nb2.index("distance 10 |"), True)
        check("neighbors круг 2: пометка расстояния и текст SRT",
              "distance 1 | before [14]: Text14" in nb2 and "distance 1 | after [16]: Text16" in nb2, True)
        nb2_edge = format_neighbors_context(1, nb_segs, nb_res, REPAIR2_CONTEXT_WINDOW, with_distance=True)
        check("neighbors круг 2: край файла без before", "before" not in nb2_edge and "after [2]" in nb2_edge, True)

        # 8г. Промпт REPAIR: круг 1 и круг 2 в режимах 1 / 2 / 3
        seg_r2 = Segment(70, "00:03:00,000", "00:03:03,000", "The treaty was signed")
        old_r2 = entry(scene="Treaty on a desk", query_narrow="treaty document", query_medium="treaty document",
                       query_broad="document")
        nb_r2 = format_neighbors_context(15, nb_segs, nb_res, REPAIR2_CONTEXT_WINDOW, with_distance=True)
        for md in (1, 2, 3):
            iss_r2 = validate_entries({"70": old_r2}, mode=md, lang="en")["70"]
            info1 = {70: {"entry": old_r2, "issues": iss_r2, "neighbors": nb1, "round": 1, "mode": md}}
            info2 = {70: {"entry": old_r2, "issues": iss_r2, "neighbors": nb_r2, "round": 2, "mode": md}}
            pr1 = build_prompt([seg_r2], repair_info=info1)
            pr2 = build_prompt([seg_r2], repair_info=info2)
            check(f"REPAIR круг 1 режим {md}: три вопроса и плоские листы",
                  all(q in pr1 for q in ("WHERE", "WHO", "WHAT", "scroll", "NEW scene")), True)
            check(f"REPAIR круг 1 режим {md}: нет правила круга 2", "ROUND 2" not in pr1, True)
            check(f"REPAIR круг 1 режим {md}: самопроверка в конце блока со словами из константы",
                  "SELF-CHECK" in pr1 and all(w in pr1[pr1.index("SELF-CHECK"):] for w in JUNK_KIND_WORDS), True)
            check(f"REPAIR круг 1 режим {md}: самопроверка после списка сегментов",
                  pr1.index("SELF-CHECK") > pr1.index("### Segment 70"), True)
            check(f"REPAIR круг 2 режим {md}: правило круга 2 и самопроверка", "ROUND 2 ONLY" in pr2 and "SELF-CHECK" in pr2, True)
            check(f"REPAIR круг 2 режим {md}: старая scene мусорного сегмента не показана",
                  "Treaty on a desk" not in pr2 and "Previous query" not in pr2, True)
            check(f"REPAIR круг 2 режим {md}: соседи с расстоянием, без scene",
                  "distance 1 |" in pr2 and "Scene" not in pr2, True)
            if md == 1:
                check("REPAIR круг 2 режим 1: архивные sites, без инструкции про сток",
                      "Keep archival sites" in pr2 and 'sites = ["pexels", "pixabay"]' not in pr2
                      and "is_entity = false" not in pr2, True)
            else:
                check(f"REPAIR круг 2 режим {md}: sites = pexels/pixabay, пустые entity_keywords, is_entity=false",
                      'sites = ["pexels", "pixabay"]' in pr2 and "entity_keywords = []" in pr2
                      and "is_entity = false" in pr2, True)

        # 9. Повторный запрос на заглушке (REPAIR cycle):
        # а) Исправление проходит -> код 0, файл записан
        # б) Исправление не проходит, GENQ_STRICT=1 -> код 1, файл записан, чекпоинт не удалён
        # в) Исправление не проходит, GENQ_STRICT=2 -> код 0, файл записан, чекпоинт удалён
        import tempfile
        test_segs = [
            Segment(1, "00:00:00,000", "00:00:02,000", "A soldier stands guard"),
            Segment(2, "00:00:02,000", "00:00:04,000", "Map of the border"),
        ]

        with tempfile.TemporaryDirectory() as td:
            out_file = os.path.join(td, "requests.json")
            cp_file = checkpoint_path_for(out_file)

            # Кейс А: успешное исправление
            with open(cp_file, "w") as f:
                f.write("{}")
            initial_res_a = {
                "1": entry(scene="Soldier standing guard", sites=["pexels"], query_narrow="soldier guard"),
                "2": entry(scene="A military map", sites=["pexels"], query_narrow="military map"),
            }
            def mock_call_success(*args, **kwargs):
                return {
                    "2": entry(scene="A fortress wall at the border", sites=["pexels"],
                               query_narrow="fortress wall border", query_medium="fortress wall", query_broad="fortress")
                }
            code_a = run_repair_cycle(
                client=None, current_model="test-model", fallback_queue=[],
                segments=test_segs, results=initial_res_a, exhausted_models={},
                checkpoint_path=cp_file, src_hash="hash", sources_mode=2, strict_mode=1,
                output_path=out_file, call_batch_fn=mock_call_success,
            )
            check("repair mock: успех -> код 0", code_a, 0)
            check("repair mock: успех -> файл записан", os.path.isfile(out_file), True)
            check("repair mock: успех -> чекпоинт удалён", os.path.isfile(cp_file), False)
            with open(out_file) as f:
                saved_data_a = json.load(f)
            check("repair mock: успех -> сегмент 2 исправлен", "fortress wall" in saved_data_a["2"]["scene"], True)

            # Кейс Б: неуспех, strict=1 -> код 1, файл записан, чекпоинт не удалён
            with open(cp_file, "w") as f:
                f.write("{}")
            initial_res_b = {
                "1": entry(scene="Soldier standing guard", sites=["pexels"], query_narrow="soldier guard"),
                "2": entry(scene="A military map", sites=["pexels"], query_narrow="military map"),
            }
            def mock_call_fail(*args, **kwargs):
                # Модель снова вернула слово map
                return {
                    "2": entry(scene="Another strategic map", sites=["pexels"], query_narrow="strategic map")
                }
            code_b = run_repair_cycle(
                client=None, current_model="test-model", fallback_queue=[],
                segments=test_segs, results=initial_res_b, exhausted_models={},
                checkpoint_path=cp_file, src_hash="hash", sources_mode=2, strict_mode=1,
                output_path=out_file, call_batch_fn=mock_call_fail,
            )
            check("repair mock: неуспех strict=1 -> код 1", code_b, 1)
            check("repair mock: неуспех strict=1 -> файл записан", os.path.isfile(out_file), True)
            check("repair mock: неуспех strict=1 -> чекпоинт сохранён (не удалён)", os.path.isfile(cp_file), True)

            # Кейс В: неуспех, strict=2 -> код 0, файл записан, чекпоинт удалён
            code_c = run_repair_cycle(
                client=None, current_model="test-model", fallback_queue=[],
                segments=test_segs, results=initial_res_b, exhausted_models={},
                checkpoint_path=cp_file, src_hash="hash", sources_mode=2, strict_mode=2,
                output_path=out_file, call_batch_fn=mock_call_fail,
            )
            check("repair mock: неуспех strict=2 -> код 0", code_c, 0)
            check("repair mock: неуспех strict=2 -> файл записан", os.path.isfile(out_file), True)
            check("repair mock: неуспех strict=2 -> чекпоинт удалён", os.path.isfile(cp_file), False)

            # Кейс Г: в call_batch_fn уходят английские проблемы, а лог остаётся русским
            captured: dict = {}
            def mock_call_capture(*args, **kwargs):
                captured.update(kwargs.get("repair_info") or {})
                return {"2": entry(scene="Another strategic map", sites=["pexels"], query_narrow="strategic map")}
            class _ListHandler(logging.Handler):
                def __init__(self):
                    super().__init__()
                    self.msgs: list[str] = []
                def emit(self, record):
                    self.msgs.append(record.getMessage())
            lh = _ListHandler()
            logging.getLogger().addHandler(lh)
            try:
                with open(cp_file, "w") as f:
                    f.write("{}")
                run_repair_cycle(
                    client=None, current_model="test-model", fallback_queue=[],
                    segments=test_segs, results=dict(initial_res_b), exhausted_models={},
                    checkpoint_path=cp_file, src_hash="hash", sources_mode=2, strict_mode=2,
                    output_path=out_file, call_batch_fn=mock_call_capture,
                )
            finally:
                logging.getLogger().removeHandler(lh)
            check("repair mock: в call_batch_fn английские проблемы",
                  captured.get(2, {}).get("issues"), ["forbidden shot type 'map' in field scene",
                                                      "forbidden shot type 'map' in field query_narrow"])
            check("repair mock: лог по-прежнему русский (Сегмент N: ...)",
                  any(m.startswith("Сегмент 2: в поле scene запрещённый тип кадра: map") for m in lh.msgs), True)

            # Кейс Д: два круга REPAIR на заглушке (число вызовов, итоговые коды, квота и исключение)
            cyc_segs = [Segment(i, "00:00:00,000", "00:00:01,000", "Map of the border" if i == 7 else f"Narration {i}")
                        for i in range(1, 15)]

            def good_entry():
                return entry(scene="A fortress wall at the border", query_narrow="fortress wall border",
                             query_medium="fortress wall", query_broad="fortress")

            def bad_entry():
                return entry(scene="Another strategic map", query_narrow="strategic map")

            def make_results():
                r = {str(s.index): entry(scene="Calm street") for s in cyc_segs}
                r["7"] = entry(scene="A military map", query_narrow="military map")
                return r

            def make_mock(seq, calls):
                def _m(client, model, chunk, cb, ca, mode=2, repair_info=None):
                    calls.append((model, dict(repair_info or {})))
                    kind = seq[min(len(calls) - 1, len(seq) - 1)]
                    if isinstance(kind, Exception):
                        raise kind
                    return {"7": good_entry() if kind == "good" else bad_entry()}
                return _m

            def run_cyc(seq, strict=1, fallback=None, out=None):
                calls: list = []
                with open(cp_file, "w") as f:
                    f.write("{}")
                exhausted: dict = {}
                code = run_repair_cycle(
                    client=None, current_model="test-model", fallback_queue=list(fallback or []),
                    segments=cyc_segs, results=make_results(), exhausted_models=exhausted,
                    checkpoint_path=cp_file, src_hash="hash", sources_mode=2, strict_mode=strict,
                    output_path=out or out_file, call_batch_fn=make_mock(seq, calls),
                )
                return code, calls, exhausted

            root_logger = logging.getLogger()
            saved_level = root_logger.level
            root_logger.setLevel(logging.INFO)
            lh2 = _ListHandler()
            root_logger.addHandler(lh2)
            try:
                # Д1. Исправлено в круге 1 -> круг 2 не вызывается
                if os.path.isfile(out_file):
                    os.remove(out_file)
                code_d1, calls_d1, _ = run_cyc(["good"])
                check("repair 2 круга: исправлено в круге 1 -> код 0", code_d1, 0)
                check("repair 2 круга: исправлено в круге 1 -> один вызов (круг 2 не вызывался)", len(calls_d1), 1)
                check("repair 2 круга: круг 1 помечен round=1", [v["round"] for v in calls_d1[0][1].values()], [1])
                check("repair 2 круга: круг 1 -> соседи без scene и без distance",
                      "Scene" not in calls_d1[0][1][7]["neighbors"] and "distance" not in calls_d1[0][1][7]["neighbors"]
                      and "[6] Text: Narration 6" in calls_d1[0][1][7]["neighbors"], True)
                check("repair 2 круга: лог 'круг 2 не требовался'",
                      any("круг 2 не требовался" in m for m in lh2.msgs), True)

                # Д2. Исправлено в круге 2 -> код 0, файл записан, чекпоинт удалён
                lh2.msgs.clear()
                if os.path.isfile(out_file):
                    os.remove(out_file)
                code_d2, calls_d2, _ = run_cyc(["bad", "good"])
                check("repair 2 круга: исправлено в круге 2 -> код 0", code_d2, 0)
                check("repair 2 круга: исправлено в круге 2 -> два вызова", len(calls_d2), 2)
                check("repair 2 круга: исправлено в круге 2 -> файл записан", os.path.isfile(out_file), True)
                check("repair 2 круга: исправлено в круге 2 -> чекпоинт удалён", os.path.isfile(cp_file), False)
                with open(out_file) as f:
                    saved_d2 = json.load(f)
                check("repair 2 круга: сегмент 7 исправлен кругом 2", "fortress wall" in saved_d2["7"]["scene"], True)
                info_r2 = calls_d2[1][1][7]
                check("repair 2 круга: круг 2 помечен round=2", info_r2["round"], 2)
                check("repair 2 круга: круг 2 -> окно 10, ближайшие первыми, пометка distance, без scene",
                      "distance 6 | before [1]" in info_r2["neighbors"] and "distance 7 | after [14]" in info_r2["neighbors"]
                      and "distance 8 |" not in info_r2["neighbors"]
                      and info_r2["neighbors"].index("distance 1 |") < info_r2["neighbors"].index("distance 2 |")
                      and "Scene" not in info_r2["neighbors"] and "Calm street" not in info_r2["neighbors"], True)
                check("repair 2 круга: лог статистики по кругам",
                      any("после круга 1: 1, после круга 2: 0" in m for m in lh2.msgs), True)
                check("repair 2 круга: по строке REPAIR на чанк в каждом круге",
                      sum(1 for m in lh2.msgs if m.startswith("REPAIR круг 1/2: чанк 1/1: 1 сегментов [7..7]")) == 1
                      and sum(1 for m in lh2.msgs if m.startswith("REPAIR круг 2/2: чанк 1/1: 1 сегментов [7..7]")) == 1, True)
                check("repair 2 круга: строка 'Источники' осталась", any(m.startswith("Источники: архив") for m in lh2.msgs), True)

                # Д3. Не исправлено после круга 2: strict=1 -> код 1, strict=2 -> код 0; не больше двух кругов
                lh2.msgs.clear()
                code_d3, calls_d3, _ = run_cyc(["bad"], strict=1)
                check("repair 2 круга: не исправлено, strict=1 -> код 1", code_d3, 1)
                check("repair 2 круга: не исправлено -> ровно два вызова (третьего круга нет)", len(calls_d3), 2)
                check("repair 2 круга: не исправлено, strict=1 -> файл записан", os.path.isfile(out_file), True)
                check("repair 2 круга: не исправлено, strict=1 -> чекпоинт сохранён", os.path.isfile(cp_file), True)
                check("repair 2 круга: ERROR-строка с номером сегмента",
                      any(m.startswith("Сегмент 7: в поле scene запрещённый тип кадра: map") for m in lh2.msgs), True)
                code_d4, calls_d4, _ = run_cyc(["bad"], strict=2)
                check("repair 2 круга: не исправлено, strict=2 -> код 0", code_d4, 0)
                check("repair 2 круга: не исправлено, strict=2 -> два вызова", len(calls_d4), 2)
                check("repair 2 круга: не исправлено, strict=2 -> файл записан", os.path.isfile(out_file), True)
                check("repair 2 круга: не исправлено, strict=2 -> чекпоинт удалён", os.path.isfile(cp_file), False)

                # Д4. Квота во втором круге: запасная модель, затем код 3
                out_q = os.path.join(td, "out_q.json")
                code_q1, calls_q1, exh_q1 = run_cyc(
                    ["bad", DailyQuotaExceededError("test-model", "quota"), "good"], fallback=["fb-model"], out=out_q)
                check("repair круг 2: квота -> переключение на запасную модель, код 0", code_q1, 0)
                check("repair круг 2: квота -> вызовы test-model, test-model, fb-model",
                      [c[0] for c in calls_q1], ["test-model", "test-model", "fb-model"])
                check("repair круг 2: квота -> модель помечена исчерпанной", "test-model" in exh_q1, True)
                check("repair круг 2: после переключения остаётся round=2", calls_q1[2][1][7]["round"], 2)
                if os.path.isfile(out_q):
                    os.remove(out_q)
                code_q2, calls_q2, _ = run_cyc(
                    ["bad", DailyQuotaExceededError("test-model", "quota")], fallback=[], out=out_q)
                check("repair круг 2: квота без запасных моделей -> код 3", code_q2, 3)
                check("repair круг 2: код 3 -> requests.json не записан", os.path.isfile(out_q), False)
                check("repair круг 2: код 3 -> чекпоинт сохранён", os.path.isfile(cp_file), True)

                # Д5. Исключение во втором круге -> код 1, requests.json не записан, чекпоинт сохранён
                code_x, calls_x, _ = run_cyc(["bad", RuntimeError("boom")], out=out_q)
                check("repair круг 2: исключение -> код 1", code_x, 1)
                check("repair круг 2: исключение -> requests.json не записан", os.path.isfile(out_q), False)
                check("repair круг 2: исключение -> чекпоинт сохранён", os.path.isfile(cp_file), True)
                check("repair круг 2: исключение -> два вызова", len(calls_x), 2)
            finally:
                root_logger.removeHandler(lh2)
                root_logger.setLevel(saved_level)

        # 10. parse_sources_mode и parse_strict_mode (валидация env и CLI)
        check("parse mode: CLI валидный", parse_sources_mode("1", "3"), 1)
        check("parse mode: env валидный", parse_sources_mode(None, "3"), 3)
        check("parse mode: невалидный -> дефолт 2", parse_sources_mode("invalid", None), 2)
        check("parse mode: None -> дефолт 2", parse_sources_mode(None, None), 2)
        check("parse strict: CLI валидный", parse_strict_mode("2", "1"), 2)
        check("parse strict: env валидный", parse_strict_mode(None, "2"), 2)
        check("parse strict: невалидный -> дефолт 1", parse_strict_mode("bad", None), 1)
        check("parse strict: None -> дефолт 1", parse_strict_mode(None, None), 1)

    finally:
        _normalize_warn_left[0] = saved_warn

    print(f"\nИтого: {'все тесты прошли' if not failures else 'ПАДЕНИЯ: ' + ', '.join(failures)}")
    return 1 if failures else 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if "--self-test" in sys.argv[1:]:
        return run_self_tests()

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
        "--sources-mode", default=None,
        help="Режим источников: 1 (только архив), 2 (микс, по умолчанию), 3 (только сток).",
    )
    parser.add_argument(
        "--strict", default=None,
        help="Строгость проверки запросов: 1 (калибровка/код 1 при ошибках), 2 (мягко/warning, код 0).",
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

    sources_mode = parse_sources_mode(args.sources_mode, os.environ.get("GENQ_SOURCES_MODE"))
    strict_mode = parse_strict_mode(args.strict, os.environ.get("GENQ_STRICT"))

    mode_names = {1: "только архив", 2: "микс", 3: "только сток"}
    logging.info(
        "Режим источников: %s (%s), строгость проверки: %s (%s)",
        sources_mode, mode_names[sources_mode], strict_mode,
        "калибровка / код 1" if strict_mode == 1 else "мягко / код 0",
    )

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

    try:
        http_timeout_s = float(
            os.environ.get("GENQ_HTTP_TIMEOUT_SECONDS", DEFAULT_HTTP_TIMEOUT_SECONDS)
        )
        if http_timeout_s <= 0:
            raise ValueError("должен быть > 0")
    except ValueError as e:
        logging.error("Некорректный GENQ_HTTP_TIMEOUT_SECONDS: %s", e)
        return 1
    # HttpOptions.timeout в google-genai - миллисекунды. Действует на все вызовы клиента,
    # включая call_gemini_batch.
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(http_timeout_s * 1000)),
    )
    logging.info("HTTP-таймаут запроса к Gemini: %.1fs", http_timeout_s)

    fallback_models = [m.strip() for m in args.fallback_models.split(",") if m.strip()]
    candidates = [args.model] + [m for m in fallback_models if m != args.model]

    try:
        current_model, fallback_queue = pick_working_model(client, candidates, exhausted_models)
    except AllModelsQuotaExhaustedError as e:
        save_checkpoint(checkpoint_path, src_hash, results, exhausted_models)
        logging.error(
            "Не удалось найти рабочую модель среди %s - дневной лимит исчерпан у всех (%s). "
            "Прогресс (%s из %s сегментов) сохранён в чекпоинте %s. Запустите скрипт "
            "повторно позже (лимит сбрасывается в полночь по тихоокеанскому времени) или "
            "добавьте больше моделей в --fallback-models / GENQ_FALLBACK_MODELS.",
            candidates, e, len(results), len(segments), checkpoint_path,
        )
        return 3
    except ModelUnavailableError as e:
        save_checkpoint(checkpoint_path, src_hash, results, exhausted_models)
        logging.error(
            "Не удалось найти рабочую модель среди %s - это НЕ подтверждённая дневная квота, "
            "а временная недоступность: %s. Прогресс (%s из %s сегментов) сохранён в "
            "чекпоинте %s. Повторите запуск позже.",
            candidates, e, len(results), len(segments), checkpoint_path,
        )
        return 4
    except Exception:
        save_checkpoint(checkpoint_path, src_hash, results, exhausted_models)
        logging.exception(
            "Preflight модели завершился неожиданной ошибкой (не квота). Чекпоинт сохранён в %s.",
            checkpoint_path,
        )
        return 1

    if not args.skip_schema_preflight:
        # Явный тайминг preflight-звонка (логику намеренно не трогаем - см. обсуждение):
        # пока не ясно, разовая ли это задержка Gemini или повторяющаяся аномалия, поэтому
        # просто фиксируем цифру на каждом прогоне и смотрим, повторяется ли она дальше.
        preflight_start = time.monotonic()
        try:
            schema_preflight_check(client, current_model)
        except Exception as e:
            logging.error(
                "Preflight по response_schema не пройден за %.2fs: %s. Основной цикл не запускается.",
                time.monotonic() - preflight_start, e,
            )
            return 1
        logging.info("Preflight по response_schema занял %.2fs.", time.monotonic() - preflight_start)

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
                    client, current_model, batch, context_before, context_after,
                    mode=sources_mode,
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

    # Пост-проверка, исправление REPAIR, статистика и завершение
    return run_repair_cycle(
        client=client,
        current_model=current_model,
        fallback_queue=fallback_queue,
        segments=segments,
        results=results,
        exhausted_models=exhausted_models,
        checkpoint_path=checkpoint_path,
        src_hash=src_hash,
        sources_mode=sources_mode,
        strict_mode=strict_mode,
        output_path=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())
