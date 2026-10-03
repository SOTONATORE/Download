#!/usr/bin/env python3
"""
generate_queries.py

Генерирует requests.json с поисковыми запросами (стоковые/архивные видео и фото)
для каждого сегмента SRT-файла, используя Gemini API (structured output через
response_schema).

Использование:
    python generate_queries.py --input input.srt --output requests.json
    python generate_queries.py --sources-mode 1 --strict 1

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
    GENQ_SOURCES_MODE    - опционально, режим источников: 1 (микс, по умолчанию),
                           2 (только архив: wikimedia, loc, nasa), 3 (только сток: pexels, pixabay)
    GENQ_STRICT          - опционально, строгость проверки запросов: 1 (калибровка - падение
                           с кодом 1 при нарушениях после повтора), 2 (мягко, по умолчанию - warning в логе,
                           requests.json записан, код 0)

Формат requests.json:
    Словарь "номер сегмента" (строка) -> запись с полями: scene (одно английское предложение
    о том, что видно в кадре), sites (список источников в порядке приоритета), query_narrow,
    query_medium, query_broad (строка или null), type ("image"/"video"), is_entity (bool, выводится
    как bool(entity_keywords)), entity_keywords (список строк, английское написание первым), visual_value (целое 0-100: насколько
    реплика выигрывает от картинки; ставит Gemini по тексту и соседним репликам, вне 0-100 зажимается,
    нечисловое/отсутствующее - ошибка ответа с повтором батча; при REPAIR не меняется; в записи идёт
    последним полем). Старых полей query и fallback_query в записи НЕТ.
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
from functools import partial
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
# Окна контекста в блоке сегмента (build_segment_block): набираются целыми предложениями,
# пока сумма слов не достигнет порога; верхней границы нет.
CONTEXT_BEFORE_WORDS = 40
CONTEXT_AFTER_WORDS = 25
SHORT_SEGMENT_MAX_WORDS = 3
SHORT_SEGMENT_NOTE = (
    "Note: this fragment is too short to carry a picture on its own; use the setting of the "
    "sentence unless the fragment itself names a specific place, person or object."
)
# Круг 2 REPAIR: широкое окно соседей (только текст SRT, ближайшие первыми, с пометкой расстояния).
REPAIR2_CONTEXT_WINDOW = 10
# Предохранитель расширения окна соседей REPAIR до границ предложений: максимум добавленных
# сегментов на каждую сторону (на текст без пунктуации предложение может быть бесконечным).
REPAIR_MAX_EXPAND_SEGMENTS = 20
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
# 1 - query/fallback_query; 2 - scene + query_narrow/medium/broad; 3 - + visual_value.
SCHEMA_VERSION = 3
# Таймаут HTTP-запроса к Gemini (сек). В google-genai HttpOptions.timeout задаётся в
# МИЛЛИСЕКУНДАХ, поэтому при создании клиента переводим секунды в мс.
DEFAULT_HTTP_TIMEOUT_SECONDS = 60
# Preflight: ретраи транзиентных сбоев (5xx/таймаут/сеть) на одной модели.
PREFLIGHT_MAX_ATTEMPTS = 3
PREFLIGHT_BACKOFF_BASE_SECONDS = 2
SITES = ["pexels", "pixabay", "wikimedia", "nasa", "loc"]

# Режимы источников (env GENQ_SOURCES_MODE / --sources-mode): 1 = микс (по умолчанию),
# 2 = только архив, 3 = только сток. Числа 1/2/3 в коде не используются, только эти константы.
SOURCES_MIX = 1
SOURCES_ARCHIVE = 2
SOURCES_STOCK = 3
SOURCES_MODES = (SOURCES_MIX, SOURCES_ARCHIVE, SOURCES_STOCK)
DEFAULT_SOURCES_MODE = SOURCES_MIX
DEFAULT_STRICT = 2
# Режим типа медиа (env MEDIA_MODE): 1 = смешанный (тип выбирает Gemini), 2 = только видео,
# 3 = только фото. Не зависит от режима источников.
DEFAULT_MEDIA_MODE = 1
MEDIA_MODE_FORCED_TYPE = {2: "video", 3: "image"}
MEDIA_MODE_NAMES = {1: "смешанный", 2: "только видео", 3: "только фото"}

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
    """Парсит режим источников (1=микс, 2=архив, 3=сток). CLI в приоритете.
    Любое невалидное значение игнорируется с WARNING и берётся режим по умолчанию (микс)."""
    val = cli_val if cli_val is not None else env_val
    if val is None:
        return DEFAULT_SOURCES_MODE
    val_str = str(val).strip()
    if val_str in tuple(str(m) for m in SOURCES_MODES):
        return int(val_str)
    logging.warning(
        "Некорректный режим источников %r (допустимо %s) - использую по умолчанию %s.",
        val, ", ".join(str(m) for m in SOURCES_MODES), DEFAULT_SOURCES_MODE,
    )
    return DEFAULT_SOURCES_MODE


def parse_media_mode(env_val: Optional[str | int]) -> int:
    """Режим типа медиа из env MEDIA_MODE: \"1\"/\"2\"/\"3\" (пробелы обрезаются).
    None или пустая строка - 1 (смешанный). Любое другое значение - ValueError с понятным
    сообщением (без молчаливой подстановки по умолчанию)."""
    if env_val is None:
        return DEFAULT_MEDIA_MODE
    val_str = str(env_val).strip()
    if val_str == "":
        return DEFAULT_MEDIA_MODE
    if val_str in ("1", "2", "3"):
        return int(val_str)
    raise ValueError(
        f"Некорректное значение MEDIA_MODE: {env_val!r}. Допустимо: 1 (смешанный, тип выбирает "
        f"Gemini), 2 (только видео), 3 (только фото); пустое значение или не задано = 1."
    )


def parse_strict_mode(cli_val: Optional[str | int], env_val: Optional[str | int]) -> int:
    """Парсит строгость проверки (1=калибровка, 2=мягко). CLI в приоритете.
    Любое невалидное значение игнорируется с WARNING и берётся DEFAULT_STRICT (2)."""
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
    "type", "is_entity", "entity_keywords", "visual_value",
]
# Потолок отдельных warning нормализации за весь запуск, дальше - только итоговые счётчики.
MAX_NORMALIZE_WARNINGS = 20


def build_system_instruction(
    mode: int = DEFAULT_SOURCES_MODE, junk_words: tuple[str, ...] = JUNK_KIND_WORDS,
    media_mode: int = DEFAULT_MEDIA_MODE,
) -> str:
    """Генерирует системный промпт с учётом выбранного режима источников (1/2/3)
    и запрещённых типов кадра из junk_words."""
    junk_list_str = ", ".join(junk_words)

    mode_blocks = {
        SOURCES_ARCHIVE: (
            f"SOURCE MODE RULES (MODE {SOURCES_ARCHIVE}: ARCHIVE ONLY):\n"
            "- All segments MUST use archival sites ONLY: [\"wikimedia\", \"loc\"] (use \"nasa\" first only when explicitly about space/astronomy/NASA missions). NEVER include \"pexels\" or \"pixabay\".\n"
            "- All search queries must follow the archival search style (concise proper nouns, literal title/caption matches).\n"
            "- For abstract or general segments without a specific named entity: derive a generalized ARCHIVAL query for the depicted place/era taken by the SOURCE LADDER (steps b, c and e), without people names, and strictly without any forbidden visual types.\n"
        ),
        SOURCES_MIX: (
            f"SOURCE MODE RULES (MODE {SOURCES_MIX}: MIXED ARCHIVE AND STOCK):\n"
            "- SITES DEFAULT: Default to stock sites [\"pexels\", \"pixabay\"]. Use archival sites [\"wikimedia\", \"loc\"] ONLY when the words of the segment explicitly name a specific real person, a specific building, or a specific physical object that can actually be photographed (not just a date, country, or era). Add \"nasa\" first only for space/astronomy/NASA missions.\n"
            "- When genuinely unsure, use stock sites [\"pexels\", \"pixabay\"], NOT both and NOT archive.\n"
            "- For abstract or general segments: follow the ABSTRACT / GENERAL SEGMENTS rule below (use stock sites).\n"
        ),
        SOURCES_STOCK: (
            f"SOURCE MODE RULES (MODE {SOURCES_STOCK}: STOCK ONLY):\n"
            "- All segments MUST use stock sites ONLY: [\"pexels\", \"pixabay\"]. NEVER include \"wikimedia\", \"loc\", or \"nasa\".\n"
            "- All search queries must follow the stock search style: query_medium (2-4 words, object + context), query_narrow (4-6 words, slightly more specific), query_broad (1-2 words, general image).\n"
            "- Replace any proper names with plain visual generalizations (a person by role or appearance, a place or building by its type).\n"
            f"- In Mode {SOURCES_STOCK}, entity_keywords MUST ALWAYS be an empty list [], and is_entity MUST ALWAYS be false for ALL segments.\n"
        ),
    }
    mode_rule = mode_blocks.get(mode, mode_blocks[SOURCES_MIX])

    # Подсказка режима типа медиа (MEDIA_MODE): только при 2 или 3, в режиме 1 текст не меняется.
    media_hint = {
        2: (
            " MEDIA MODE: every segment will be sourced as VIDEO only, so write scene as a MOVING "
            "shot: describe an action and the movement of the camera or of the subject."
        ),
        3: (
            " MEDIA MODE: every segment will be sourced as PHOTO only, so write scene as a STILL "
            "frame: describe the composition, with no action unfolding in time."
        ),
    }.get(media_mode, "")

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

1. scene - ONE English sentence: what the viewer should SEE in the frame for this segment. Segments \
are fragments of continuous speech, so the scene shows what the words of this segment itself talk about. \
Choose it by answering three questions in order: (1) WHERE does this happen - the place and situation \
given by the SOURCE LADDER below; (2) WHO is there - the people by role or group, without personal \
names when the sites are stock; (3) WHAT of this can a camera film as a solid, living subject - a \
building, a hall, a street, a landscape, people, a vehicle, a tool, a physical object - rather than a \
flat sheet. Flat objects that carry text (paper, page, sheet, parchment, letter, manuscript, scroll) \
are undesirable as the MAIN subject of the frame.
SOURCE LADDER (the same for every segment, it always ends with a result):
   a) if the words of the segment name something a camera can film, show that;
   b) otherwise take the place and situation from the whole sentence (the "Sentence" line; the words \
after << belong to it and may be used);
   c) if the sentence gives too little, take only the place, the time and the participants by role \
from the "Before" line;
   d) the "After" line and any segment that follows the current one serve only to understand the text \
(a pronoun, a continued thought) and are NEVER a source for the scene;
   e) if nothing was found, scene is still REQUIRED for every segment in every mode, including REPAIR: \
a general view of the place or era of the sentence's topic; an empty string, null or a refusal for scene \
is never allowed.
Weak fragments (function words, abstraction, a state, a negation, an enumeration) are handled by steps \
b and c, with people shown by role.
Never use or depict these shot types in scene or in any query: {junk_list_str} (and plural forms); \
a date or number is never a calendar, and geopolitics or wars are never a map. \
Write scene BEFORE the queries and derive all three queries from it. \
If the only thing the segment's words give you to show is one of those forbidden shot types or a flat \
text-bearing object (a treaty, decree, letter, newspaper, map, date, flag, and the like), treat the \
segment as abstract and apply the ABSTRACT / GENERAL SEGMENTS rule below.

2. sites - ordered list of source sites, in priority order for this segment. Allowed values: \
"pexels", "pixabay", "wikimedia", "nasa", "loc". This list is fixed - never invent other sources. \
Follow the SOURCE MODE RULES above.

3. query_narrow, query_medium, query_broad - three English search queries for the SAME scene, from \
most specific to most general. They are tried in this order, so each must be a realistic search \
phrase on its own. The style depends on the FIRST site in "sites":
   - If the first site is an archive (wikimedia / loc / nasa):
     * query_narrow: SHORT, 2-4 words ONLY - the exact proper noun(s) that a real file title or \
caption on these sites would actually contain: a person's full name, OR a specific place name, OR a \
named event, optionally with a short refinement of the object. Do NOT append a year unless it is named in the segment's text (see the YEARS \
rule). If a real name genuinely needs more than 4 words, that's fine - the limit is about cutting \
padding, not truncating a proper noun. For nasa: exact mission/object names \
and dates, as concise as the name requires.
     * query_medium: ONLY the name OR ONLY the place, WITHOUT a year.
     * query_broad: an ordinary plain-language phrasing of the setting or place of the same visual scene for \
pexels/pixabay, 2-4 words, no proper nouns, not a portrait of a person.
     MediaWiki (wikimedia) and LOC search match literal file titles/captions, which are short and \
factual, so descriptive or stylistic padding only dilutes the match: use the bare name or place \
exactly as the segment gives it, without added descriptors, years or archive words.
   - If the first site is stock (pexels / pixabay):
     * query_medium: 2-4 words, main object + context.
     * query_narrow: slightly more specific than medium, 4-6 words.
     * query_broad: 1-2 words, the general image; prefer the setting or place over a portrait of a person.
   - query_broad = null ONLY for segments with no visual scene at all - e.g. silence, a black \
screen, a title/credits card with no depicted content, or on-screen text with nothing else \
happening. When in doubt, fill it in rather than returning null. query_narrow and query_medium are \
never null. This rule is about query_broad only.

4. type - "image" or "video", whichever fits the described scene better (a still, motionless view -> \
"image"; a dynamic action or a generic/modern/abstract scene -> "video").{media_hint}

5. entity_keywords - the MAIN entity field. List EVERY proper name (person, place, event, \
organization, treaty, building) that appears in your query_narrow or query_medium, each in TWO \
variants: the English spelling and the Russian spelling, as consecutive pairs, in the form \
["<English name 1>", "<Russian name 1>", "<English name 2>", "<Russian name 2>"]. The ENGLISH spelling of the most important \
name MUST be the FIRST element of the list (the search script uses it as the lookup query). Keep \
each name as one element (do not split a multi-word name into separate words). Generic things - religions, \
ideologies, nationalities, professions, titles without a name, emotions, general themes are NOT \
proper names. An EMPTY list [] means "no proper names in \
the queries". Never leave it empty when a query contains a proper name, even if the frame looks \
generic (a plain city view or a coastline of a named city is still an entity scene).

6. is_entity - true when entity_keywords is non-empty, false when it is empty. Fill it consistently \
with entity_keywords (the script recomputes it from that list anyway).

7. visual_value - an INTEGER from 0 to 100: how much the words of this segment gain from a picture. \
Judge ONLY by the text of this segment and the context of the neighboring sentences (for this field the \
"Sentence", "Before" and "After" lines may all be used, even though the After line is never a source \
for the scene). Do NOT judge by the scene or queries you wrote and do not raise the score just because \
a stock shot for the topic exists. Anchors:
   - 80-100: something concrete and visible that a camera can film - a place, an object, an event, a \
person or an organization;
   - 50-79: concrete but general ("people discussed", "in those years"): any shot on the topic would fit;
   - 20-49: abstract or reasoning; a picture is possible only as a metaphor;
   - 0-19: greetings, linking phrases, evaluations, calls to action, empty talk. A segment with no \
text (silence) ALWAYS gets 0.
Use the WHOLE range and tell the segments of this response apart: giving 70-90 to almost all segments \
is wrong. The scores are used only to compare segments of ONE video with each other, so the most \
picture-worthy segments must get the highest values and the emptiest ones the lowest. In REPAIR mode \
copy the "Previous visual_value" unchanged.

ABSTRACT / GENERAL SEGMENTS:
When a segment lacks a concrete visible physical subject (such as narrator evaluations, conclusions, \
transitions, abstract concepts, emotions, numbers, or dates without physical objects):
- Take the place and situation by the SOURCE LADDER: the "Sentence" line first (step b), then the \
"Before" line (step c), then a general view of the place or era of the topic (step e). The "After" \
line and the segments after the current one are not used for the scene.
- Formulate the scene and all three queries as a GENERALIZED stock shot of that place or situation \
(without proper nouns, even if the context names them).
- Set sites = ["pexels", "pixabay"], entity_keywords = [], is_entity = false (unless running in Mode {SOURCES_ARCHIVE}).

GENERAL RULES:
- Queries describe what is VISIBLE in the frame. They do not retell or paraphrase the narrator's \
words.
- NEVER use these filler words in any query: b-roll, cinematic, footage, HD, 4K, historical, and the \
phrases "stock footage", "stock photo", "stock image". Also NEVER use forbidden visual types in \
any query: {junk_list_str} (and plural forms). The words video, stock, vintage are also banned as \
filler (style padding added to a query), but ALLOWED when they are part of the depicted subject \
of the segment itself.
- NEVER use mood adjectives (sad, empty, mysterious, dramatic, lonely, gloomy, etc.) in a query \
unless that exact mood is stated in the segment's text.
- YEARS: put a year (also a decade like "1920s" or a range like "1914-1918") in a query ONLY if that \
exact year is named in the segment's text; otherwise the query contains no year at all. Never guess or \
add a year from your own knowledge: a year that the text itself names may be used, a year that only \
you know may not. Archive sites match every word, so an invented year returns nothing.
- Never include "creative commons", "free", or "no copyright" in a query - these are not effective \
search terms; licensing is filtered separately downstream, not through the query text.
- Do not invent scene details: the only details allowed are those of the words of the segment, the \
place, time and participants taken from the "Sentence" and "Before" lines by the SOURCE LADDER, and a \
general view of the place or era (step e); everything else is an invention (extra people, moods, \
weather, time of day or settings that none of these sources gives).
- Do not replace an idea with an object that symbolizes it when the words do not mention that object; \
show the place or situation of the sentence instead.
- VARIETY: the scene of a segment must not repeat the scenes you wrote for the previous three segments \
of this response, and its query_narrow must not repeat their query_narrow. When neighboring segments \
are about the same subject, vary the scene by another aspect of the same place or event that the \
words support, never by another subject and never by a changing frame size. For a segment about a \
named entity from an archive source, the name in query_medium and entity_keywords stays exactly as the \
rules above require; only the scene and query_narrow may vary. Do not invent events or details \
beyond what the scene rules above allow.
- Always include silent/empty segments in the output (query_broad may be null for them), with scene \
and queries taken by the SOURCE LADDER: for a segment without text the "Sentence" line if it lies \
inside a sentence, otherwise the "Before" line, otherwise a general view (step e) - never skip a \
segment number.

You may also be given extra CONTEXT inside each segment block: a "Sentence" line (the full sentence \
this segment belongs to, with the words of this segment between >> and <<), a "Before (context only)" \
line and an "After (context only)" line (the neighboring sentences). In REPAIR mode a "Neighbors \
(context)" block is added: in round 1 it lists neighbor segments under "Context BEFORE:" and \
"Context AFTER:" labels, in round 2 it lists neighbors by distance, nearest first. Use this context \
to understand the narrative (for example, to resolve a pronoun or continue a thought from the \
current segment); what of it may enter the scene is set by the SOURCE LADDER, where Neighbors \
entries before the segment count as the Before line and entries after it as the After line. \
Do NOT create response objects for context. Only answer for the segments listed \
under "Segments that need a response", each marked as "### Segment N".
"""
    return instruction


SYSTEM_INSTRUCTION = build_system_instruction(DEFAULT_SOURCES_MODE)

# Порядок полей важен: scene идёт первым после segment_index, чтобы модель сначала
# описывала кадр, а уже потом строила по нему запросы. Gemini не гарантирует порядок по
# порядку ключей в dict, поэтому он задан явно через property_ordering. Исключение - visual_value:
# он стоит сразу после segment_index, до scene, чтобы оценка не якорилась на уже написанном кадре
# (scene всегда конкретный, даже для абстрактных реплик). Порядок в requests.json от этого не зависит.
SEGMENT_ENTRY_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "segment_index": types.Schema(type=types.Type.INTEGER),
        "scene": types.Schema(
            type=types.Type.STRING,
            description=(
                "ONE English sentence: what the viewer should see in the frame. For abstract "
                "phrases use concrete objects/places of the topic, not emotions. Do not invent "
                "details that are not in the segment text."
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
                "historic/vintage only when part of the subject itself; no "
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
        "visual_value": types.Schema(
            type=types.Type.INTEGER,
            description=(
                "Integer 0-100: how much the segment's words gain from a picture, judged only "
                "by the segment text and neighboring sentences. 80-100 concrete visible "
                "(place/object/event/person/organization); 50-79 concrete but general; "
                "20-49 abstract, metaphor only; 0-19 greeting/link/evaluation/call/empty. "
                "Silence = 0. Use the whole range; do not give 70-90 to almost everything."
            ),
        ),
    },
    property_ordering=[
        "segment_index",
        "visual_value",
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
        "visual_value",
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
# Склейка сегментов в предложения (только данные; в промпт и в main пока не подключена)
# ---------------------------------------------------------------------------

# Знаки, которыми может кончаться предложение (многоточие «...» из трёх точек тоже
# заканчивается на "." и отдельно распознаётся в _ends_sentence).
SENTENCE_END_CHARS = ".?!\u2026"
# Закрывающие кавычки/скобки, которые могут стоять ПОСЛЕ знака конца: He said "stop." / (so.)
SENTENCE_CLOSERS = "\"'\u00bb\u201d\u2019)]"
# Открывающие кавычки/скобки в начале следующего сегмента: их пропускаем при проверке заглавной.
SENTENCE_OPENERS = "\"'\u00ab\u201c\u2018\u201e(["
# Сокращения, после точки которых предложение НЕ заканчивается (сравнение регистрозависимое,
# чтобы "no." в конце обычной фразы осталось концом предложения). "No." остаётся в наборе, но
# _ends_sentence обрабатывает его отдельно: оно сокращение только перед цифрой.
NO_BREAK_ABBREVIATIONS = frozenset(
    {"Mr.", "Mrs.", "Ms.", "Dr.", "St.", "No.", "Jr.", "Sr.", "vs."}
)
# "etc." - особый случай: может и закончить предложение, поэтому точка после неё считается
# продолжением только если следующий сегмент начинается со строчной буквы.
LOWERCASE_CONTINUES_ABBREVIATIONS = frozenset({"etc."})


@dataclass
class Sentence:
    number: int                         # порядковый номер предложения, с 1
    seg_indices: list[int]              # Segment.index входящих сегментов, по порядку
    text: str                           # полный текст (тексты сегментов через один пробел)
    spans: dict[int, tuple[int, int]]   # Segment.index -> (start, end) внутри text; text[start:end] = текст сегмента

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _strip_closers(text: str) -> str:
    return text.rstrip(SENTENCE_CLOSERS + " ")


def _ends_sentence(text: str, next_text: Optional[str]) -> bool:
    """Заканчивается ли предложение на этом сегменте. text - непустой нормализованный текст
    сегмента, next_text - текст следующего НЕпустого сегмента (None, если его нет)."""
    core = _strip_closers(text)
    if not core or core[-1] not in SENTENCE_END_CHARS:
        return False
    nxt = next_text.lstrip(SENTENCE_OPENERS + " ") if next_text else ""

    if core.endswith("\u2026") or core.endswith("..."):
        # Многоточие - конец только перед заглавной буквой; перед строчной (и перед цифрой,
        # знаком и т.п.) это продолжение фразы. В самом конце текста - конец.
        return next_text is None or nxt[:1].isupper()

    if core[-1] in "?!":
        return True

    # Остался случай "."
    token = core.split()[-1].lstrip(SENTENCE_OPENERS)
    if token == "No.":
        # «No.» - сокращение (номер) только перед цифрой («No.» + «5 apples.»); слово-число
        # («Five apples.») цифрой не считается, там граница; реплика «No.» - тоже граница.
        return not re.match(r"\d", nxt)
    if token in NO_BREAK_ABBREVIATIONS:
        return False
    if token in LOWERCASE_CONTINUES_ABBREVIATIONS:
        return not nxt[:1].islower()
    if re.search(r"\d\.$", token) and nxt[:1].isdigit():
        # число, разрезанное между сегментами: "1." + "5 percent"
        return False
    return True


def merge_segments_into_sentences(
    segments: list[Segment],
) -> tuple[list[Sentence], dict[int, Sentence]]:
    """Склеивает сегменты SRT в предложения. Чистая функция: без сети и без побочных эффектов,
    вызывается один раз по всему списку (до разбиения на батчи).

    Возвращает (sentences, by_segment): список предложений по порядку и словарь
    Segment.index -> Sentence, в которое входит сегмент (ссылка на тот же объект).

    Правила конца предложения см. _ends_sentence. Дополнительно:
    - последнее предложение без финального знака - тоже предложение;
    - пустой сегмент никогда не разрывает предложение: если предложение открыто, пустой
      сегмент входит в него с пустой границей (start == end, пробел в текст не добавляется);
      если открытого предложения нет (начало файла или тишина между предложениями), пустой
      сегмент образует отдельное предложение с text == \"\", чтобы не примешиваться к соседям;
    - тексты сегментов перед склейкой нормализуются (пробелы схлопываются)."""
    texts = [" ".join((s.text or "").split()) for s in segments]

    # next_nonempty[i] - текст ближайшего непустого сегмента строго после i
    next_nonempty: list[Optional[str]] = [None] * len(segments)
    upcoming: Optional[str] = None
    for i in range(len(segments) - 1, -1, -1):
        next_nonempty[i] = upcoming
        if texts[i]:
            upcoming = texts[i]

    sentences: list[Sentence] = []
    by_segment: dict[int, Sentence] = {}
    current: Optional[Sentence] = None

    for i, seg in enumerate(segments):
        t = texts[i]
        if not t:
            if current is None:
                lone = Sentence(len(sentences) + 1, [seg.index], "", {seg.index: (0, 0)})
                sentences.append(lone)
                by_segment[seg.index] = lone
            else:
                pos = len(current.text)
                current.seg_indices.append(seg.index)
                current.spans[seg.index] = (pos, pos)
                by_segment[seg.index] = current
            continue

        if current is None:
            current = Sentence(len(sentences) + 1, [], "", {})
            sentences.append(current)
        if current.text:
            current.text += " "
        start = len(current.text)
        current.text += t
        current.seg_indices.append(seg.index)
        current.spans[seg.index] = (start, len(current.text))
        by_segment[seg.index] = current

        if _ends_sentence(t, next_nonempty[i]):
            current = None

    return sentences, by_segment


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _collect_context_sentences(
    sentences: list[Sentence], start: int, step: int, window_words: int
) -> list[Sentence]:
    """Целые предложения от ближайшего к сегменту наружу (step -1 - назад, +1 - вперёд), пока
    сумма слов не достигнет window_words. Предложение, переваливающее за порог, берётся целиком;
    минимум одно предложение, если оно есть. Предложения с пустым текстом пропускаются.
    Возвращает предложения в порядке от ближнего к дальнему."""
    out: list[Sentence] = []
    total = 0
    i = start
    while 0 <= i < len(sentences):
        sen = sentences[i]
        i += step
        if not sen.text:
            continue
        out.append(sen)
        total += sen.word_count
        if total >= window_words:
            break
    return out


def build_segment_block(
    seg: Segment,
    sentences: list[Sentence],
    by_segment: dict[int, Sentence],
    before_words: int = CONTEXT_BEFORE_WORDS,
    after_words: int = CONTEXT_AFTER_WORDS,
) -> str:
    """Блок одного сегмента для пользовательского промпта (без хвостовой пустой строки).
    Не зависит от build_prompt - его же будет использовать REPAIR. sentences/by_segment - результат
    merge_segments_into_sentences по ВСЕМУ списку сегментов (предложение через границу батча
    берётся целиком). Пустые строки (нет Before у первого / After у последнего предложения) не
    печатаются.

    Пустой сегмент: Text = (silence / no text), без маркера >>...<<; если он стоит внутри
    непустого предложения, оно выводится в строке Sentence без маркера."""
    sen = by_segment[seg.index]
    lines = [f"### Segment {seg.index}", f"Timing: {seg.start} --> {seg.end}"]
    text = " ".join((seg.text or "").split())

    if text:
        a, b = sen.spans[seg.index]
        marked = f"{sen.text[:a]}>>{sen.text[a:b]}<<{sen.text[b:]}"
        lines.append(f"Text (what this segment says): {text}")
        lines.append(f"Sentence: {marked}")
        if len(text.split()) <= SHORT_SEGMENT_MAX_WORDS:
            lines.append(SHORT_SEGMENT_NOTE)
    else:
        lines.append("Text (what this segment says): (silence / no text)")
        if sen.text:
            lines.append(f"Sentence: {sen.text}")

    pos = sen.number - 1
    before = _collect_context_sentences(sentences, pos - 1, -1, before_words)
    after = _collect_context_sentences(sentences, pos + 1, +1, after_words)
    if before:
        lines.append("Before (context only): " + " ".join(x.text for x in reversed(before)))
    if after:
        lines.append("After (context only): " + " ".join(x.text for x in after))
    return "\n".join(lines)


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
    if mode == SOURCES_ARCHIVE:
        return base + (
            f"Keep archival sites according to the Mode {SOURCES_ARCHIVE} rules; write a generalized ARCHIVAL query for the "
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
    sentence_index: tuple[list[Sentence], dict[int, Sentence]] | None = None,
) -> str:
    """Обычный режим: блоки сегментов строит build_segment_block по sentence_index (результат
    merge_segments_into_sentences по всему списку; без него - запасной вариант: склейка только по
    batch, для вызовов без полного списка вроде preflight). context_before/context_after больше
    не используются ни в обычном режиме, ни в REPAIR (контекст теперь внутри блока сегмента:
    Sentence / Before / After по целым предложениям); параметры оставлены ради совместимости
    сигнатуры. В REPAIR блок каждого сегмента тоже строит build_segment_block по sentence_index
    (посчитан один раз по всему списку сегментов); запасной вариант без него - склейка по batch."""
    lines: list[str] = []

    if sentence_index is None:
        sentence_index = merge_segments_into_sentences(batch)
    sentences, by_segment = sentence_index

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
            "First write a NEW scene from scratch by answering three questions in order, taking the place, "
            "the situation and the participants by the SOURCE LADDER of the system prompt (the words of this "
            "segment, then the Sentence line, then the Before line, then a general view of the place or era); "
            "the After line and the Neighbors entries after the segment are for understanding only and are "
            "NEVER a source for the scene, and this holds wherever the rules below mention neighbors: "
            "(1) WHERE does this happen; (2) WHO is "
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
        lines.append(
            "- visual_value: copy the 'Previous visual_value' of each segment unchanged; do NOT re-score, "
            "even when the scene is rewritten."
        )
        if rep_round >= 2:
            lines.append(_repair_round_rule(rep_mode))
        lines.append("")
        for s in batch:
            rep = repair_info.get(s.index, {})
            prev = rep.get("entry", {})
            issues_list = rep.get("issues", [])
            issues_formatted = "\n".join(f"  - {iss}" for iss in issues_list) if issues_list else "  - (no specific issues)"
            neighbors_str = rep.get("neighbors", "(no neighbor context)")

            # Заголовок, Timing, Text, Sentence с маркером, Note и Before/After - как в обычном режиме
            lines.append(build_segment_block(s, sentences, by_segment))
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
            # visual_value показываем всегда (и для junk-сегментов): его ремонт не меняет
            lines.append(f"Previous visual_value: {prev.get('visual_value', '')}")
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
            lines.append(build_segment_block(s, sentences, by_segment))
            lines.append("")

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


def _parse_visual_value(raw, seg_index: int) -> tuple[int, bool]:
    """Разбор visual_value: целое (int, не bool; float допустим только с целым значением).
    Возвращает (значение, зажато_ли). Нечисловое, None, bool, строка, дробное - ValueError, чтобы
    сработал повтор батча (как у других невалидных полей); молча ничего не подставляется.
    Вне 0-100 - зажим в этот диапазон (причина копится в stats как visual_value_clamped)."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"Сегмент {seg_index}: visual_value должно быть целым числом, получено {raw!r}")
    if isinstance(raw, float):
        if not raw.is_integer():
            raise ValueError(f"Сегмент {seg_index}: visual_value должно быть целым, получено {raw!r}")
        raw = int(raw)
    clamped = min(100, max(0, raw))
    return clamped, clamped != raw


def _normalize_entry(
    item: dict, seg_index: int, stats: Optional[Counter] = None,
    segment_text: Optional[str] = None, media_mode: int = DEFAULT_MEDIA_MODE,
) -> dict:
    """Детерминированно проверяет и чинит запись сегмента (без новых вызовов Gemini).

    Структурно битая запись (не объект / нет обязательных ключей / нецелое visual_value) - ValueError, чтобы
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

    # visual_value: обязательное целое; нечисловое -> ValueError (повтор батча), вне 0-100 -> зажим
    visual_value, vv_clamped = _parse_visual_value(item["visual_value"], seg_index)
    if vv_clamped:
        fixes.append("visual_value_clamped")
        _warn_limited("Сегмент %s: visual_value %r вне 0-100 - зажал до %s.",
                      seg_index, item["visual_value"], visual_value)

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

    # type: режимы 2/3 (MEDIA_MODE) принудительно задают тип независимо от ответа Gemini
    # (единственное место); режим 1 - как раньше: валидация и запасная ветка.
    if media_mode in MEDIA_MODE_FORCED_TYPE:
        seg_type = MEDIA_MODE_FORCED_TYPE[media_mode]
    else:
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
        "visual_value": visual_value,
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
        if mode == SOURCES_ARCHIVE:
            stock_present = ", ".join(x for x in sites if x in stock_sites)
            if stock_present:
                found_issues.append((
                    f"в режиме {SOURCES_ARCHIVE} (архив) недопустимы стоковые сайты: {stock_present}",
                    f"in mode {SOURCES_ARCHIVE} (archive) stock sites are not allowed: {stock_present}",
                ))
        elif mode == SOURCES_STOCK:
            arch_present = ", ".join(x for x in sites if x in archive_sites)
            if arch_present:
                found_issues.append((
                    f"в режиме {SOURCES_STOCK} (сток) недопустимы архивные сайты: {arch_present}",
                    f"in mode {SOURCES_STOCK} (stock) archive sites are not allowed: {arch_present}",
                ))

    # 3. Режим 3: is_entity != false ИЛИ entity_keywords непустой
    if mode == SOURCES_STOCK:
        is_ent = entry.get("is_entity", False)
        kw = entry.get("entity_keywords") or []
        if is_ent is not False:
            found_issues.append((
                f"в режиме {SOURCES_STOCK} (сток) is_entity должен быть false, получено: {is_ent}",
                f"in mode {SOURCES_STOCK} (stock) is_entity must be false, got: {is_ent}",
            ))
        if kw:
            found_issues.append((
                f"в режиме {SOURCES_STOCK} (сток) entity_keywords должен быть пустым, найдено: {kw}",
                f"in mode {SOURCES_STOCK} (stock) entity_keywords must be empty, found: {kw}",
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


# Страховка для query_broad (этап 3в): у архивного сегмента с именем каскад при пустом архиве
# падает на query_broad, и слова portrait/man/woman/boy/girl дают студийный портрет не по теме.
# Набор слов фиксирован: формы (portraits, men, women) и составные не считаются.
BROAD_BANNED_WORDS = ("portrait", "man", "woman", "boy", "girl")
# Целое слово без учёта регистра. Слева и справа не допускаются буква/цифра/подчёркивание и дефис
# (man-made, manual, human, boyfriend не срабатывают). Апостроф (обычный и типографский) границу
# слова не нарушает: "woman's" и "woman\u2019s" - это слово woman.
BROAD_BANNED_RE = re.compile(
    r"(?<![\w-])(" + "|".join(BROAD_BANNED_WORDS) + r")(?![\w-])", re.IGNORECASE
)


def is_archive_named_entry(entry: dict) -> bool:
    """Архивный сегмент с именем: первый сайт архивный (как в _normalize_entry и в статистике
    источников: sites[0] in wikimedia/loc/nasa) И entity_keywords непустой (is_entity выводится
    из него же). Стоковый первый сайт или пустые entity_keywords - не такой сегмент."""
    sites = entry.get("sites")
    kw = entry.get("entity_keywords")
    if not isinstance(sites, list) or not sites or sites[0] not in _ARCHIVE_SITES:
        return False
    return isinstance(kw, list) and any(isinstance(k, str) and k.strip() for k in kw)


def check_archive_broad_words(entries: dict[str, dict], lang: str = "ru") -> dict[str, list[str]]:
    """Страховка кодом после ответа модели: если сегмент архивный с именем и в query_broad есть
    слово из BROAD_BANNED_WORDS, возвращает проблему (формат как у validate_entries). query_broad
    не меняется, слова не вычищаются; остальные сегменты не затрагиваются."""
    issues: dict[str, list[str]] = {}
    for idx_str, entry in entries.items():
        if not is_archive_named_entry(entry):
            continue
        broad = entry.get("query_broad")
        if not isinstance(broad, str) or not broad:
            continue
        found = sorted({m.group(1).lower() for m in BROAD_BANNED_RE.finditer(broad)})
        if not found:
            continue
        words = ", ".join(found)
        if lang == "en":
            issues[idx_str] = [
                f"query_broad contains the word '{words}', which is not allowed in query_broad of an "
                f"archive segment with a named entity; rewrite query_broad without any of: "
                f"{', '.join(BROAD_BANNED_WORDS)}"
            ]
        else:
            issues[idx_str] = [
                f"в query_broad архивного сегмента с именем запрещённое слово: {words}"
            ]
    return issues


def validate_with_broad_guard(
    entries: dict[str, dict], mode: int, junk_re: Optional[re.Pattern] = None, lang: str = "ru",
) -> dict[str, list[str]]:
    """validate_entries + check_archive_broad_words в один словарь {номер: [проблемы]}.
    Сегмент с обеими причинами попадает один раз (причины склеиваются), поэтому в REPAIR он
    не дублируется."""
    issues = validate_entries(entries, mode, junk_re, lang)
    for idx_str, extra in check_archive_broad_words(entries, lang).items():
        issues.setdefault(idx_str, []).extend(extra)
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


def _neighbor_window(
    pos: int,
    segments: list[Segment],
    window: int,
    sentence_index: tuple[list[Sentence], dict[int, Sentence]] | None,
    target_idx: int = 0,
) -> tuple[list[Segment], list[Segment]]:
    """Окно соседей вокруг позиции pos: сегменты в радиусе +-window, расширенные в обе стороны до
    границ целых предложений (по sentence_index: первый сегмент предложения для левого края,
    последний - для правого). Добавлено не более REPAIR_MAX_EXPAND_SEGMENTS сегментов на сторону;
    при срабатывании лимита окно режется по нему и пишется предупреждение. Без sentence_index
    расширения нет (прежнее поведение). Возвращает (before, after) в хронологическом порядке."""
    start = max(0, pos - window)
    end = min(len(segments), pos + 1 + window)  # не включительно
    if sentence_index is not None:
        _, by_segment = sentence_index
        pos_of = {sg.index: i for i, sg in enumerate(segments)}
        if start < pos:
            sen = by_segment.get(segments[start].index)
            first = pos_of.get(sen.seg_indices[0]) if sen else None
            if first is not None and first < start:
                new_start = max(first, start - REPAIR_MAX_EXPAND_SEGMENTS)
                if new_start > first:
                    logging.warning(
                        "REPAIR: окно соседей сегмента %s слева обрезано по лимиту расширения (%s сегментов), "
                        "предложение не доведено до начала.", target_idx, REPAIR_MAX_EXPAND_SEGMENTS,
                    )
                start = new_start
        if end - 1 > pos:
            sen = by_segment.get(segments[end - 1].index)
            last = pos_of.get(sen.seg_indices[-1]) if sen else None
            if last is not None and last > end - 1:
                new_last = min(last, end - 1 + REPAIR_MAX_EXPAND_SEGMENTS)
                if new_last < last:
                    logging.warning(
                        "REPAIR: окно соседей сегмента %s справа обрезано по лимиту расширения (%s сегментов), "
                        "предложение не доведено до конца.", target_idx, REPAIR_MAX_EXPAND_SEGMENTS,
                    )
                end = new_last + 1
    return segments[start:pos], segments[pos + 1 : end]


def format_neighbors_context(
    target_idx: int,
    segments: list[Segment],
    results: Optional[dict[str, dict]] = None,
    window: int = CONTEXT_WINDOW,
    with_distance: bool = False,
    sentence_index: tuple[list[Sentence], dict[int, Sentence]] | None = None,
) -> str:
    """Форматирует контекст соседей +-window для блока REPAIR. Источник - ТОЛЬКО текст SRT:
    scene и любые значения из results не показываются (параметр results оставлен ради
    совместимости вызовов и не используется). with_distance=False (круг 1): блоки Context
    BEFORE / AFTER. with_distance=True (круг 2): компактный список, ближайшие первыми, у каждого
    соседа пометка distance N; дальние соседи идут как фон.
    sentence_index (по всему списку сегментов): окно +-window расширяется до границ целых
    предложений (см. _neighbor_window); distance считается от проверяемого сегмента, в том числе
    для добавленных. Без sentence_index окно строго +-window, как раньше."""
    pos = None
    for i, s in enumerate(segments):
        if s.index == target_idx:
            pos = i
            break
    if pos is None:
        return "(no neighbor context available)"

    def _txt(s: Segment) -> str:
        return " ".join((s.text or "").split()) or "(empty)"

    before, after = _neighbor_window(pos, segments, window, sentence_index, target_idx)

    lines = []
    if with_distance:
        # before[-d] - сосед слева на расстоянии d, after[d-1] - сосед справа на расстоянии d
        for d in range(1, max(len(before), len(after)) + 1):
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
    sentence_index: tuple[list[Sentence], dict[int, Sentence]] | None = None,
    media_mode: int = DEFAULT_MEDIA_MODE,
) -> dict:
    prompt = build_prompt(
        batch, context_before, context_after, repair_info=repair_info, sentence_index=sentence_index
    )

    config = types.GenerateContentConfig(
        system_instruction=build_system_instruction(mode, media_mode=media_mode),
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
                result[str(idx)] = _normalize_entry(item, idx, fix_stats, seg_texts.get(idx), media_mode=media_mode)

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
    sentence_index: tuple[list[Sentence], dict[int, Sentence]] | None = None,
) -> tuple[Optional[int], str]:
    """Один круг REPAIR по номерам indices (чанки по chunk_repair_indices). Обновляет results,
    fallback_queue, exhausted_models и чекпоинт на месте. Возвращает (код, модель): код None -
    круг завершён, иначе код возврата процесса (3 - квоты исчерпаны, 1 - ошибка вызова)."""
    seg_by_idx = {s.index: s for s in segments}
    repair_chunks = chunk_repair_indices(indices, max_size=125)
    # Строка соседей (format_neighbors_context): круг 1 - +-CONTEXT_WINDOW, подписи Context
    # BEFORE/AFTER внутри строки; круг 2 - +-REPAIR2_CONTEXT_WINDOW с пометками distance N. В обоих
    # кругах окно расширено до границ целых предложений (sentence_index), соседи - только текст SRT.
    neighbors_window = REPAIR2_CONTEXT_WINDOW if round_num == 2 else CONTEXT_WINDOW

    for chunk_num, chunk_indices in enumerate(repair_chunks, start=1):
        chunk_segs = [seg_by_idx[i] for i in chunk_indices]
        first_idx, last_idx = chunk_segs[0].index, chunk_segs[-1].index
        first_pos = next(i for i, s in enumerate(segments) if s.index == first_idx)
        last_pos = next(i for i, s in enumerate(segments) if s.index == last_idx)
        ctx_before = segments[max(0, first_pos - CONTEXT_WINDOW) : first_pos]
        ctx_after = segments[last_pos + 1 : last_pos + 1 + CONTEXT_WINDOW]

        # В промпт уходят английские формулировки проблем, в лог - русские
        issues_en = validate_with_broad_guard(
            {str(i): results[str(i)] for i in chunk_indices}, sources_mode, lang="en"
        )
        chunk_repair_info = {}
        for s in chunk_segs:
            chunk_repair_info[s.index] = {
                "entry": results.get(str(s.index), {}),
                "issues": issues_en.get(str(s.index), []),
                "neighbors": format_neighbors_context(
                    s.index, segments, None, neighbors_window, with_distance=(round_num == 2),
                    sentence_index=sentence_index,
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
                    sentence_index=sentence_index,
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

        # visual_value ремонт не меняет: сохраняем прежнюю оценку (модель обязана вернуть поле
        # по схеме, но её новое значение отбрасывается, если у сегмента оценка уже была)
        for idx_str, new_entry in repaired_batch.items():
            old_vv = results.get(idx_str, {}).get("visual_value")
            if old_vv is not None:
                new_entry["visual_value"] = old_vv
        results.update(repaired_batch)
        save_checkpoint(checkpoint_path, src_hash, results, exhausted_models)

    return None, current_model


def summarize_visual_values(entries: dict[str, dict]) -> str:
    """Одна строка для лога: min / медиана / max visual_value и число сегментов в корзинах
    0-19, 20-49, 50-79, 80-100 (чтобы по прогону видеть, не ставит ли модель всем одно и то же)."""
    vals = sorted(e["visual_value"] for e in entries.values() if isinstance(e.get("visual_value"), int))
    if not vals:
        return "visual_value: нет оценок"
    n = len(vals)
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    buckets = [sum(1 for v in vals if lo <= v <= hi) for lo, hi in ((0, 19), (20, 49), (50, 79), (80, 100))]
    return (
        f"visual_value: сегментов {n}, min {vals[0]}, медиана {median:g}, max {vals[-1]}; "
        f"корзины 0-19: {buckets[0]}, 20-49: {buckets[1]}, 50-79: {buckets[2]}, 80-100: {buckets[3]}"
    )


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
    sentence_index: tuple[list[Sentence], dict[int, Sentence]] | None = None,
) -> int:
    """Выполняет пост-проверку записей, до двух кругов исправления REPAIR через Gemini
    (круг 2 - только для номеров, оставшихся с нарушениями после круга 1; третьего круга нет),
    логирует статистику источников и сохраняет requests.json."""
    # Склейка в предложения - один раз по всему списку (main передаёт готовый индекс)
    if sentence_index is None:
        sentence_index = merge_segments_into_sentences(segments)
    initial_issues = validate_with_broad_guard(results, sources_mode)
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
            sentence_index,
        )
        if code is not None:
            return code

        # Перепроверка только исправленных номеров круга 1
        post_issues_1 = validate_with_broad_guard({str(i): results[str(i)] for i in problem_indices}, sources_mode)

        if post_issues_1:
            round2_indices = sorted(int(k) for k in post_issues_1.keys())
            logging.warning(
                "После круга 1 остались нарушения в %s сегментах. Запускаю круг 2 REPAIR (последний)...",
                len(round2_indices),
            )
            code, current_model = _run_repair_round(
                client, current_model, fallback_queue, segments, results, exhausted_models,
                checkpoint_path, src_hash, sources_mode, round2_indices, 2, call_batch_fn,
                sentence_index,
            )
            if code is not None:
                return code
            round2_ran = True
            # Перепроверка только номеров круга 2
            post_issues_2 = validate_with_broad_guard({str(i): results[str(i)] for i in round2_indices}, sources_mode)

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
    logging.info("%s", summarize_visual_values(results))

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
            "is_entity": False, "entity_keywords": [], "visual_value": 50,
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
        test_prompt = build_system_instruction(SOURCES_MIX, junk_words=test_junk_words)
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
        v_res = validate_entries(v_entries, mode=SOURCES_MIX)
        check("validate: 101 scene map", "101" in v_res and any("scene" in s and "map" in s for s in v_res["101"]), True)
        check("validate: 102 query_narrow calendar", "102" in v_res and any("query_narrow" in s and "calendar" in s for s in v_res["102"]), True)
        check("validate: 103 query_medium coat of arms", "103" in v_res and any("query_medium" in s and "coat of arms" in s for s in v_res["103"]), True)
        check("validate: 104 query_broad passports", "104" in v_res and any("query_broad" in s and "passports" in s for s in v_res["104"]), True)
        check("validate: 105 без ошибок", "105" in v_res, False)

        # 4. validate_entries: соответствие sites режимам архив и сток
        v_sites = {
            "201": entry(sites=["pexels", "wikimedia"]),
            "202": entry(sites=["wikimedia", "loc"]),
            "203": entry(sites=["wikimedia", "pexels"]),
            "204": entry(sites=["pexels", "pixabay"]),
            "205": entry(sites=[]),
        }
        res_arch = validate_entries(v_sites, mode=SOURCES_ARCHIVE)
        check("validate режим архив: pexels запрещён", "201" in res_arch and any(f"режиме {SOURCES_ARCHIVE}" in s for s in res_arch["201"]), True)
        check("validate режим архив: wikimedia разрешён", "202" in res_arch, False)
        check("validate список sites пуст", "205" in res_arch and any("список sites пуст" in s for s in res_arch["205"]), True)

        res_stock = validate_entries(v_sites, mode=SOURCES_STOCK)
        check("validate режим сток: wikimedia запрещён", "203" in res_stock and any(f"режиме {SOURCES_STOCK}" in s for s in res_stock["203"]), True)
        check("validate режим сток: pexels разрешён", "204" in res_stock, False)

        # 5. validate_entries: режим 3 и entity_keywords / is_entity (правка 1)
        v_mode3 = {
            "301": entry(sites=["pexels"], is_entity=False, entity_keywords=[]),
            "302": entry(sites=["pexels"], is_entity=False, entity_keywords=["Paris"]),
            "303": entry(sites=["pexels"], is_entity=True, entity_keywords=[]),
            "304": entry(sites=["pexels"], is_entity=True, entity_keywords=["London"]),
        }
        res_m3_ent = validate_entries(v_mode3, mode=SOURCES_STOCK)
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
        p_mix = build_system_instruction(SOURCES_MIX)
        p_arch = build_system_instruction(SOURCES_ARCHIVE)
        p_stock = build_system_instruction(SOURCES_STOCK)
        _blk_mix = f"MODE {SOURCES_MIX}: MIXED ARCHIVE AND STOCK"
        _blk_arch = f"MODE {SOURCES_ARCHIVE}: ARCHIVE ONLY"
        _blk_stock = f"MODE {SOURCES_STOCK}: STOCK ONLY"
        check("промпт микс: блок микса и только он", (_blk_mix in p_mix, _blk_arch in p_mix, _blk_stock in p_mix), (True, False, False))
        check("промпт архив: блок архива и только он", (_blk_mix in p_arch, _blk_arch in p_arch, _blk_stock in p_arch), (False, True, False))
        check("промпт сток: блок стока и только он", (_blk_mix in p_stock, _blk_arch in p_stock, _blk_stock in p_stock), (False, False, True))
        check("промпт: режим по умолчанию даёт блок микса", build_system_instruction(), p_mix)
        check("промпт: SYSTEM_INSTRUCTION (по умолчанию) = микс", SYSTEM_INSTRUCTION, p_mix)
        check("промпт: неизвестный режим -> запасной блок микса", build_system_instruction(99), p_mix)
        check("промпт: константы режимов (микс 1, архив 2, сток 3), умолчание = микс",
              (SOURCES_MIX, SOURCES_ARCHIVE, SOURCES_STOCK, DEFAULT_SOURCES_MODE), (1, 2, 3, SOURCES_MIX))
        check("промпт архив: в тексте нет упоминаний режима микса/стока",
              ("Mode %d" % SOURCES_STOCK in p_arch, "Mode %d" % SOURCES_MIX in p_arch), (False, False))
        check("промпт микс/сток: ссылка 'unless running in Mode' ведёт на архив",
              all(f"(unless running in Mode {SOURCES_ARCHIVE})." in x for x in (p_mix, p_arch, p_stock)), True)
        check("промпт архив: содержит запрещённые слова", "calendar" in p_arch and "map" in p_arch, True)

        # 7а. Этап 3а: формулировки про контекст соответствуют реальному формату сегмента,
        # в правилах нет конкретных примеров с именами, сюжетами и предметами
        for md, pm in ((SOURCES_MIX, p_mix), (SOURCES_ARCHIVE, p_arch), (SOURCES_STOCK, p_stock)):
            check(f"этап 3а режим {md}: нет устаревших упоминаний чанк-блоков и 'narration text'",
                  [x for x in ("### Context BEFORE", "### Context AFTER", "Context BEFORE/AFTER",
                               "Context BEFORE\" / \"Context AFTER", "Context AFTER sections",
                               "narration text", "Narration text") if x in pm], [])
            check(f"этап 3а режим {md}: описаны строки Sentence / Before / After и маркеры >> <<",
                  all(x in pm for x in ("Sentence", "Before (context only)", "After (context only)", ">>", "<<")), True)
            check(f"этап 3а режим {md}: Neighbors - подписи круга 1 и список по расстоянию круга 2",
                  all(x in pm for x in ("Neighbors (context)", "\"Context BEFORE:\"", "\"Context AFTER:\"", "by distance")), True)
            banned = ["Topkapi", "Mehmed", "Vienna", "Вена", "Топкап", "Дворец", "Villa Magnolia", "San Remo",
                      "Ertu", "Siege of", "old harbor", "wooden desk", "winding", "with pines", "mountains",
                      "Islam", "caliph", "sultan", "monarchy", "palace", "courtyard",
                      "stock market", "video game", "video call", "vintage car",
                      "economy", "factory", "cargo port", "banknotes"]
            pm_low = pm.lower()
            check(f"этап 3а режим {md}: нет имён и сюжетных примеров",
                  [w for w in banned if w.lower() in pm_low], [])
        # те же запретные подстроки - в тексте всех description SEGMENT_ENTRY_SCHEMA
        _descs = [pr.description or "" for pr in SEGMENT_ENTRY_SCHEMA.properties.values()]
        _descs += [pr.items.description or "" for pr in SEGMENT_ENTRY_SCHEMA.properties.values() if pr.items is not None]
        _descs_low = " ".join(_descs).lower()
        check("этап 3а: нет имён и сюжетных примеров в description схемы",
              [w for w in banned if w.lower() in _descs_low], [])
        check("этап 3а: description схемы не пустые у scene/query_narrow (тест не пустой)",
              len(_descs_low) > 200, True)
        # подписи в тексте промпта совпадают с реальным выводом format_neighbors_context
        _nb_t = [Segment(index=i, start="0", end="1", text=f"T{i}") for i in range(1, 6)]
        _nb1 = format_neighbors_context(3, _nb_t, None, 1)
        _nb2 = format_neighbors_context(3, _nb_t, None, 1, with_distance=True)
        check("этап 3а: подписи Neighbors из промпта есть в реальном выводе",
              "Context BEFORE:" in _nb1 and "Context AFTER:" in _nb1 and "Neighbors by distance" in _nb2, True)
        # REPAIR-правило: нет 'narration text', есть ссылка на реальные источники контекста
        _rp = build_prompt(
            [_nb_t[2]],
            repair_info={3: {"entry": {}, "issues": ["forbidden shot type: map"], "neighbors": "NB", "round": 1, "mode": SOURCES_MIX}},
        )
        check("этап 3а: REPAIR-правило без 'narration text', со ссылкой на Sentence/Neighbors",
              ("narration text" in _rp, "Sentence, Before, After lines and the Neighbors block" in _rp), (False, False))
        # этап 3б (изменён прежний тест: старая ссылка на After/Neighbors как источник сцены заменена лесенкой)
        check("этап 3б: REPAIR круг 1 и 2 ссылаются на лесенку и запрет After, старой формулировки нет",
              [("SOURCE LADDER" in x and "NEVER a source for the scene" in x and "using only the words of this segment and the context" not in x)
               for x in (_rp, build_prompt([_nb_t[2]], repair_info={3: {"entry": {}, "issues": ["forbidden shot type: map"], "neighbors": "NB", "round": 2, "mode": SOURCES_ARCHIVE}}))], [True, True])

        # 7б. VARIETY: пункт в промпте всех режимов, без людей и имён, на своём месте
        for md, pm in ((SOURCES_MIX, p_mix), (SOURCES_ARCHIVE, p_arch), (SOURCES_STOCK, p_stock)):
            check(f"VARIETY режим {md}: строка 'VARIETY:' ровно один раз", pm.count("VARIETY:"), 1)
            v_start = pm.index("VARIETY: ") + len("VARIETY: ")
            v_item = pm[v_start:pm.index("\n", v_start)]
            v_low = v_item.lower()
            check(f"VARIETY режим {md}: нет people / crowd / reaction",
                  [w for w in ("people", "crowd", "reaction") if w in v_low], [])
            v_caps = []
            for v_sent in re.split(r"(?<=[.;])\s+", v_item):
                for v_word in v_sent.split()[1:]:
                    v_clean = v_word.strip(".,;:()\"'")
                    # ALL-CAPS (MUST) - модальность, а не имя собственное
                    if v_clean[:1].isupper() and not v_clean.isupper():
                        v_caps.append(v_clean)
            check(f"VARIETY режим {md}: нет слов с заглавной (имён собственных)", v_caps, [])
            check(f"VARIETY режим {md}: после 'Do not invent scene details', до 'Always include silent/empty'",
                  pm.index("Do not invent scene details") < pm.index("VARIETY:") < pm.index("Always include silent/empty"), True)


        # 3б: новые правила во всех трёх режимах
        _KEY3B = {'rule1': 'fragments of continuous speech', 'ladder': 'SOURCE LADDER', 'a': 'a) if the words of the segment name something a camera can film', 'b': 'b) otherwise take the place and situation from the whole sentence', 'c': 'c) if the sentence gives too little', 'd_after': 'NEVER a source for the scene', 'e': 'scene is still REQUIRED for every segment in every mode, including REPAIR', 'variety': 'another aspect of the same place or event', 'symbol': 'Do not replace an idea with an object that symbolizes it', 'broad': 'not a portrait of a person', 'broad_soft': 'prefer the setting or place over a portrait of a person'}
        _OLD3B = ['implied by the neighboring segments', 'built from the neighbors', "inferred from neighboring segments' context", 'from the nearest neighbor segments', "segment's narration", '(wide shot, medium shot, close-up, detail)', '2 segments before and 2 segments after', 'derive a general visual theme', 'a static portrait']
        for md, pm in ((SOURCES_MIX, p_mix), (SOURCES_ARCHIVE, p_arch), (SOURCES_STOCK, p_stock)):
            for kn, kv in _KEY3B.items():
                check(f"этап 3б режим {md}: ключевая фраза '{kn}'", kv in pm, True)
            _va = pm.index("VARIETY: ") + 9
            _vi = pm[_va:pm.index("\n", _va)].lower()
            check(f"этап 3б режим {md}: VARIETY без close-up/close up/wide shot/medium shot/detail",
                  re.findall(r"close-up|close up|wide shot|medium shot|\bdetail\b", _vi), [])
            check(f"этап 3б режим {md}: старых формулировок A1-A7 нет", [o for o in _OLD3B if o in pm], [])
            check(f"этап 3б режим {md}: лесенка в тексте один раз", pm.count("SOURCE LADDER (the same"), 1)
            _nw = []
            for _ln in pm[pm.index("SOURCE LADDER (the same"):pm.index("2. sites")].split("\n"):
                for _sent in re.split(r"(?<=[.;:])\s+", _ln):
                    for _w in _sent.split()[1:]:
                        _c = _w.strip(".,;:()\"'")
                        if _c[:1].isupper() and not _c.isupper() and _c not in ("Sentence", "Before", "After"):
                            _nw.append(_c)
            check(f"этап 3б режим {md}: в блоке лесенки нет слов с заглавной (кроме названий строк)", _nw, [])

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
        en_arch = validate_entries(en_in, mode=SOURCES_ARCHIVE, lang="en")
        en_stock = validate_entries(en_in, mode=SOURCES_STOCK, lang="en")
        ru_arch = validate_entries(en_in, mode=SOURCES_ARCHIVE)
        check("validate en: слово-тип", "forbidden shot type 'calendar' in field scene" in en_arch["1"], True)
        check("validate en: режим архив", f"in mode {SOURCES_ARCHIVE} (archive) stock sites are not allowed: pexels" in en_arch["2"], True)
        check("validate en: режим сток архивные сайты",
              f"in mode {SOURCES_STOCK} (stock) archive sites are not allowed: wikimedia" in en_stock["3"], True)
        check("validate en: режим сток is_entity", f"in mode {SOURCES_STOCK} (stock) is_entity must be false, got: True" in en_stock["3"], True)
        check("validate en: режим сток entity_keywords",
              f"in mode {SOURCES_STOCK} (stock) entity_keywords must be empty, found: ['Rome']" in en_stock["3"], True)
        check("validate en: пустой sites", "sites list is empty" in en_arch["4"], True)
        check("validate ru по умолчанию: слово-тип", "в поле scene запрещённый тип кадра: calendar" in ru_arch["1"], True)
        check("validate ru по умолчанию: режим архив",
              f"в режиме {SOURCES_ARCHIVE} (архив) недопустимы стоковые сайты: pexels" in ru_arch["2"], True)
        check("validate ru по умолчанию: пустой sites", "список sites пуст" in ru_arch["4"], True)
        check("validate: ru и en дают одинаковые номера и число проблем",
              {k: len(v) for k, v in en_stock.items()}, {k: len(v) for k, v in validate_entries(en_in, mode=SOURCES_STOCK).items()})

        # 8б. build_prompt REPAIR: правила, английские проблемы, скрытие прежнего кадра
        seg_a = Segment(57, "00:02:00,000", "00:02:03,000", "In 1920 everything changed")
        seg_b = Segment(58, "00:02:03,000", "00:02:06,000", "The market opened")
        old_a = entry(scene="A historic calendar page", query_narrow="calendar sheet wood",
                      query_medium="calendar sheet", query_broad="calendar")
        old_b = entry(scene="Busy bazaar stalls", sites=["wikimedia"], query_narrow="bazaar stalls old",
                      query_medium="bazaar stalls", query_broad="bazaar")
        iss_en = validate_entries({"57": old_a, "58": old_b}, mode=SOURCES_STOCK, lang="en")
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
        for md, pm in ((SOURCES_MIX, p_mix), (SOURCES_ARCHIVE, p_arch), (SOURCES_STOCK, p_stock)):
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

        # 8г. Промпт REPAIR: круг 1 и круг 2 во всех режимах источников
        seg_r2 = Segment(70, "00:03:00,000", "00:03:03,000", "The treaty was signed")
        old_r2 = entry(scene="Treaty on a desk", query_narrow="treaty document", query_medium="treaty document",
                       query_broad="document")
        nb_r2 = format_neighbors_context(15, nb_segs, nb_res, REPAIR2_CONTEXT_WINDOW, with_distance=True)
        for md in SOURCES_MODES:
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
            if md == SOURCES_ARCHIVE:
                check("REPAIR круг 2 режим архив: архивные sites, без инструкции про сток",
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
                checkpoint_path=cp_file, src_hash="hash", sources_mode=SOURCES_MIX, strict_mode=1,
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
                checkpoint_path=cp_file, src_hash="hash", sources_mode=SOURCES_MIX, strict_mode=1,
                output_path=out_file, call_batch_fn=mock_call_fail,
            )
            check("repair mock: неуспех strict=1 -> код 1", code_b, 1)
            check("repair mock: неуспех strict=1 -> файл записан", os.path.isfile(out_file), True)
            check("repair mock: неуспех strict=1 -> чекпоинт сохранён (не удалён)", os.path.isfile(cp_file), True)

            # Кейс В: неуспех, strict=2 -> код 0, файл записан, чекпоинт удалён
            code_c = run_repair_cycle(
                client=None, current_model="test-model", fallback_queue=[],
                segments=test_segs, results=initial_res_b, exhausted_models={},
                checkpoint_path=cp_file, src_hash="hash", sources_mode=SOURCES_MIX, strict_mode=2,
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
                    checkpoint_path=cp_file, src_hash="hash", sources_mode=SOURCES_MIX, strict_mode=2,
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
                def _m(client, model, chunk, cb, ca, mode=SOURCES_MIX, repair_info=None, sentence_index=None):
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
                    checkpoint_path=cp_file, src_hash="hash", sources_mode=SOURCES_MIX, strict_mode=strict,
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

        # 9. merge_segments_into_sentences: склейка сегментов в предложения
        def _mk(*texts: str) -> list[Segment]:
            return [Segment(i + 1, "00:00:00,000", "00:00:01,000", t) for i, t in enumerate(texts)]

        def _groups(*texts: str) -> list[list[int]]:
            return [x.seg_indices for x in merge_segments_into_sentences(_mk(*texts))[0]]

        # 9а. реальный SRT (result.srt лежит рядом со скриптом или в текущей папке)
        real_srt = next(
            (c for c in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.srt"), "result.srt")
             if os.path.isfile(c)),
            None,
        )
        if real_srt is None:
            print("SKIP  merge: result.srt не найден рядом со скриптом - тесты на реальном SRT пропущены")
        else:
            real_sen, real_by = merge_segments_into_sentences(parse_srt(real_srt))
            check("merge real: ровно 41 предложение", len(real_sen), 41)
            check("merge real: сегменты 100-101 склеены", real_by[100] is real_by[101], True)
            check("merge real: предложение 100-101 состоит ровно из них",
                  real_by[100].seg_indices, [100, 101])
            check("merge real: сегменты 117-124 склеены",
                  real_by[117].seg_indices, list(range(117, 125)))
            check("merge real: 125-126 склеены без финальной точки",
                  (real_by[125] is real_by[126], real_by[126].text[-1] in ".?!\u2026"), (True, False))
            check("merge real: не более 48 слов в предложении", max(x.word_count for x in real_sen) <= 48, True)
            check("merge real: не более 8 сегментов в предложении",
                  max(len(x.seg_indices) for x in real_sen) <= 8, True)
            check("merge real: каждый сегмент ровно в одном предложении",
                  sorted(i for x in real_sen for i in x.seg_indices), list(range(1, 127)))
            check("merge real: границы spans соответствуют тексту сегментов",
                  all(
                      x.text[a:b] == next(s.text for s in parse_srt(real_srt) if s.index == i)
                      for x in real_sen for i, (a, b) in x.spans.items()
                  ),
                  True)

        # 9б. многоточие
        check("merge: многоточие + строчная -> склеивает", _groups("He died there\u2026", "of a heart condition."), [[1, 2]])
        check("merge: многоточие + заглавная -> разрывает", _groups("He died there\u2026", "Of course."), [[1], [2]])
        check("merge: три точки + строчная -> склеивает", _groups("He died there...", "of a heart condition."), [[1, 2]])
        check("merge: три точки + заглавная -> разрывает", _groups("He died there...", "Of course."), [[1], [2]])
        check("merge: многоточие в самом конце -> конец", _groups("Well\u2026"), [[1]])
        check("merge: многоточие + открывающая кавычка + заглавная -> разрывает",
              _groups("He died there\u2026", "\"Of course.\""), [[1], [2]])
        check("merge: многоточие, закрывающая кавычка, строчная -> склеивает",
              _groups("He said \"wait\u2026\"", "and left."), [[1, 2]])
        check("merge: многоточие, пустой сегмент, строчная -> склеивает вместе с пустым",
              _groups("He died there\u2026", "", "of a heart condition."), [[1, 2, 3]])

        # 9в. закрывающие кавычка/скобка после знака
        for closer in ['"', "'", "\u00bb", "\u201d", "\u2019", ")", "]"]:
            for mark in [".", "?", "!", "\u2026"]:
                check(f"merge: знак {mark!r} + закрывающий {closer!r} -> конец",
                      _groups(f"He said stop{mark}{closer}", "Then left."), [[1], [2]])
        check("merge: без знака конца - склеивает", _groups("He said", "stop."), [[1, 2]])
        check("merge: запятая - не конец", _groups("He said,", "stop."), [[1, 2]])

        # 9г. пустой сегмент
        check("merge: пустой сегмент внутри предложения не ломает склейку",
              _groups("Hello there", "", "my friend."), [[1, 2, 3]])
        e_sen, e_by = merge_segments_into_sentences(_mk("Hello there", "", "my friend."))
        check("merge: пустая граница start == end, пробелов не добавляет",
              (e_sen[0].text, e_sen[0].spans[2]), ("Hello there my friend.", (11, 11)))
        check("merge: пустой сегмент между предложениями - отдельное предложение с пустым текстом",
              [(x.seg_indices, x.text) for x in merge_segments_into_sentences(_mk("One.", "", "Two."))[0]],
              [([1], "One."), ([2], ""), ([3], "Two.")])
        check("merge: пустой сегмент в начале", _groups("", "Hello."), [[1], [2]])
        check("merge: пустой сегмент в конце после конца предложения", _groups("Hello.", ""), [[1], [2]])
        check("merge: только пустые сегменты", _groups("", ""), [[1], [2]])
        check("merge: пустой список", merge_segments_into_sentences([]), ([], {}))

        # 9д. один сегмент из одного слова
        check("merge: одно слово с точкой", _groups("Sultans."), [[1]])
        check("merge: одно слово без знака", _groups("Sultans"), [[1]])
        check("merge: одно слово склеивается с соседями",
              _groups("A throne for", "Sultans", "and kings."), [[1, 2, 3]])

        # 9е. аббревиатуры и числа в конце сегмента не дают границы
        for abbr in ["Mr.", "Mrs.", "Ms.", "Dr.", "St.", "Jr.", "Sr.", "vs."]:  # "No." - отдельно, ниже
            check(f"merge: {abbr} в конце сегмента - не граница",
                  _groups(f"He met {abbr}", "Smith in town."), [[1, 2]])
            check(f"merge: {abbr} в конце сегмента + заглавная - не граница",
                  _groups(f"He met {abbr}", "Smith. He left."), [[1, 2]])
        check("merge: аббревиатура в скобках - не граница", _groups("(see Mr.)", "Smith."), [[1, 2]])
        check("merge: etc. + строчная - не граница", _groups("flags, drums, etc.", "were carried."), [[1, 2]])
        check("merge: etc. + заглавная - граница", _groups("flags, drums, etc.", "Then it ended."), [[1], [2]])
        check("merge: etc. в самом конце - граница", _groups("flags, drums, etc."), [[1]])
        check("merge: число 1.5 в конце сегмента - не граница", _groups("it grew by 1.5", "percent."), [[1, 2]])
        check("merge: число, разрезанное точкой (1. + 5) - не граница", _groups("it grew by 1.", "5 percent."), [[1, 2]])
        check("merge: год с точкой + заглавная - граница", _groups("He died in 1926.", "Turkey refused."), [[1], [2]])
        check("merge: обычное слово в нижнем регистре no. - граница", _groups("He said no.", "Then left."), [[1], [2]])
        # 9е-1. «No.»: сокращение только перед символом-цифрой
        check("merge: реплика «No.» + «Then left.» - две группы", _groups("No.", "Then left."), [[1], [2]])
        check("merge: «No.» + «5 apples.» - склеивает", _groups("No.", "5 apples."), [[1, 2]])
        check("merge: «No.» + «Five apples.» - две группы", _groups("No.", "Five apples."), [[1], [2]])
        check("merge: «He met No.» + «Smith in town.» - граница (не цифра)",
              _groups("He met No.", "Smith in town."), [[1], [2]])
        check("merge: «No.» + пустой + «5 apples.» - склеивает", _groups("No.", "", "5 apples."), [[1, 2, 3]])
        check("merge: «No.» в самом конце - граница", _groups("No."), [[1]])
        check("merge: «(No.» + «5 apples.» - склеивает", _groups("(No.", "5 apples."), [[1, 2]])

        # 9з. build_segment_block и подключение к build_prompt
        def _blk(texts, i, bw=CONTEXT_BEFORE_WORDS, aw=CONTEXT_AFTER_WORDS):
            segs = _mk(*texts)
            sen, by = merge_segments_into_sentences(segs)
            return build_segment_block(segs[i - 1], sen, by, bw, aw).split("\n")

        check("window: константы 40/25", (CONTEXT_BEFORE_WORDS, CONTEXT_AFTER_WORDS), (40, 25))
        # маркер ровно вокруг своего сегмента
        bl = _blk(["He died there", "of a heart", "condition."], 2)
        check("block: маркер вокруг текста сегмента",
              "Sentence: He died there >>of a heart<< condition." in bl, True)
        bl = _blk(["He died there", "of a heart", "condition."], 1)
        check("block: маркер на первом сегменте предложения", "Sentence: >>He died there<< of a heart condition." in bl, True)
        bl = _blk(["He died there", "of a heart", "condition."], 3)
        check("block: маркер на последнем сегменте предложения", "Sentence: He died there of a heart >>condition.<<" in bl, True)
        bl = _blk(["Hello there", "", "my friend."], 3)
        check("block: маркер при пустом сегменте внутри предложения", "Sentence: Hello there >>my friend.<<" in bl, True)
        check("block: заголовок и тайминг", (bl[0], bl[1]), ("### Segment 3", "Timing: 00:00:00,000 --> 00:00:01,000"))
        check("block: строка Text", bl[2], "Text (what this segment says): my friend.")
        # первое / последнее предложение, пустых строк нет
        bl = _blk(["One two.", "Three four.", "Five six."], 1)
        check("block: у первого предложения нет Before", [x for x in bl if x.startswith("Before")], [])
        check("block: After первого = ближайшие целые предложения",
              [x for x in bl if x.startswith("After")], ["After (context only): Three four. Five six."])
        bl = _blk(["One two.", "Three four.", "Five six."], 3)
        check("block: у последнего предложения нет After", [x for x in bl if x.startswith("After")], [])
        check("block: Before последнего в хронологическом порядке",
              [x for x in bl if x.startswith("Before")], ["Before (context only): One two. Three four."])
        check("block: пустых строк внутри блока нет", all(x.strip() for x in bl), True)
        bl = _blk(["Only one sentence here."], 1)
        check("block: единственное предложение - ни Before, ни After",
              [x for x in bl if x.startswith(("Before", "After"))], [])
        # окно набирается целыми предложениями, может превышать порог
        big = ["x " * 29 + "xx.", "tail one two three four."]   # 30 слов + 5 слов
        bl = _blk(big + ["Mid one.", "Cur seg now."], 4, bw=40)
        bb = next(x for x in bl if x.startswith("Before"))
        check("window: 'Before' переваливает за 40 целиком", len(bb.split(": ", 1)[1].split()), 30 + 5 + 2)
        bl = _blk(["a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e f g h i j k l m n o p q r s t u v w x y.",
                   "Cur seg now."], 2, bw=40)
        bb = next(x for x in bl if x.startswith("Before"))
        check("window: единственное длинное предложение берётся целиком (минимум одно)", len(bb.split(": ", 1)[1].split()), 51)
        bl = _blk(["S1 a b c d e f g h i.", "S2 a b c d e f g h i.", "S3 a b c d e f g h i.", "S4 a b c d e f g h i.",
                   "S5 a b c d e f g h i.", "Cur seg now."], 6, bw=40)
        bb = next(x for x in bl if x.startswith("Before"))
        check("window: ровно 40 слов - останавливается (S2..S5)", bb.startswith("Before (context only): S2 ") and "S1" not in bb, True)
        bl = _blk(["Cur seg now.", "A b c d e f g h i j.", "K l m n o p q r s t.", "U v w x y z a b c d.",
                   "E f g h i j k l m n.", "O p q r s t u v w x."], 1, aw=25)
        ab = next(x for x in bl if x.startswith("After"))
        check("window: After 25 слов -> три предложения (30 слов, переваливает целиком)", len(ab.split(": ", 1)[1].split()), 30)
        # пустые предложения пропускаются
        bl = _blk(["Left one.", "", "Left two.", "", "Cur seg now."], 5, bw=1)
        check("window: пустые предложения пропущены, берётся ближайшее непустое",
              [x for x in bl if x.startswith("Before")], ["Before (context only): Left two."])
        bl = _blk(["Cur seg now.", "", "", "Next one."], 1, aw=1)
        check("window: пустые предложения после - пропущены",
              [x for x in bl if x.startswith("After")], ["After (context only): Next one."])
        bl = _blk(["", "Cur seg now."], 2)
        check("window: только пустое предложение перед - Before нет", [x for x in bl if x.startswith("Before")], [])
        # Note для коротких сегментов
        for n_words, txt in [(1, "condition."), (2, "Hello there."), (3, "He left early.")]:
            bl = _blk([txt, "Some longer sentence follows here today."], 1)
            check(f"note: сегмент из {n_words} слов - есть Note", SHORT_SEGMENT_NOTE in bl, True)
        bl = _blk(["He left very early.", "Next."], 1)
        check("note: сегмент из 4 слов - нет Note", any(x.startswith("Note:") for x in bl), False)
        check("note: текст дословно",
              SHORT_SEGMENT_NOTE,
              "Note: this fragment is too short to carry a picture on its own; use the setting of the "
              "sentence unless the fragment itself names a specific place, person or object.")
        bl = _blk(["Hello there", "", "my friend."], 2)
        check("block: пустой сегмент - (silence / no text), без маркера и Note",
              (bl[2], any(">>" in x for x in bl), any(x.startswith("Note:") for x in bl)),
              ("Text (what this segment says): (silence / no text)", False, False))
        bl = _blk(["One.", "", "Two."], 2)
        check("block: пустой сегмент между предложениями - Before/After соседей",
              bl[2:], ["Text (what this segment says): (silence / no text)",
                       "Before (context only): One.", "After (context only): Two."])
        # весь блок только на английском (кроме текста сегмента) и без старых меток
        check("block: нет русских символов в служебных строках",
              not re.search("[а-яё]", "\n".join(_blk(["He left very early.", "Next."], 1)), re.I), True)

        # предложение через границу батча: целиком, принадлежит батчу своего сегмента
        bsegs = _mk("He died there", "of a heart", "condition.", "Turkey refused", "to take it.")
        b_idx = merge_segments_into_sentences(bsegs)
        bts = make_batches(bsegs, 2, CONTEXT_WINDOW, min_last_batch_ratio=0.0)
        check("batch: make_batches по-прежнему возвращает тройки",
              [(len(b), len(cb), len(ca)) for b, cb, ca in bts], [(2, 0, 3), (3, 2, 0)])  # хвост из 3 сегментов слит по правилу min_size
        p_b1 = build_prompt(bts[0][0], bts[0][1], bts[0][2], sentence_index=b_idx)
        p_b2 = build_prompt(bts[1][0], bts[1][1], bts[1][2], sentence_index=b_idx)
        check("batch: предложение через границу в первом батче - целиком, маркер на своём сегменте",
              "Sentence: He died there >>of a heart<< condition." in p_b1, True)
        check("batch: во втором батче то же предложение целиком, маркер на сегменте 3",
              "Sentence: He died there of a heart >>condition.<<" in p_b2, True)
        check("batch: сегмент принадлежит только своему батчу",
              ("### Segment 3" in p_b1, "### Segment 3" in p_b2, "### Segment 2" in p_b2), (False, True, False))
        check("batch: старых блоков Context BEFORE/AFTER в обычном режиме нет",
              ("Context BEFORE" in p_b2, "Context AFTER" in p_b2, "\nText: " in p_b2), (False, False, False))
        p_old = build_prompt(bts[1][0], bsegs[:1], bsegs[-1:], repair_info={3: {"entry": {}, "issues": ["x"]}})
        check("batch: в REPAIR-режиме блоков Context BEFORE/AFTER на уровне чанка больше нет",
              ("### Context BEFORE" in p_old, "### Context AFTER" in p_old), (False, False))
        # REPAIR по общему sentence_index: окна по предложениям, старого формата сегмента нет
        rp_ent = {3: {"entry": {"scene": "OLD_SCENE"}, "issues": ["x"], "neighbors": "NB"}}
        p_r3 = build_prompt(bts[1][0][:1], bsegs[:1], bsegs[-1:], repair_info=rp_ent, sentence_index=b_idx)
        check("repair block: предложение через границу батча целиком, маркер на сегменте 3",
              "Sentence: He died there of a heart >>condition.<<" in p_r3, True)
        check("repair block: After - следующее предложение целиком",
              "After (context only): Turkey refused to take it." in p_r3, True)
        check("repair block: нет 'Narration text:' и 'Context BEFORE/AFTER'",
              ("Narration text:" in p_r3, "Context BEFORE" in p_r3, "Context AFTER" in p_r3), (False, False, False))
        check("repair block: Note для короткого сегмента",
              SHORT_SEGMENT_NOTE in p_r3, True)
        check("repair block: Previous scene и Neighbors остались",
              ("Previous scene: OLD_SCENE" in p_r3, "Neighbors (context):\nNB" in p_r3), (True, True))
        # sentence_index доходит до call_batch_fn в обоих кругах
        _si_seen: list = []
        def _m_si(client, model, chunk, cb, ca, mode=SOURCES_MIX, repair_info=None, sentence_index=None):
            _si_seen.append(sentence_index)
            return {"1": entry(scene="Calm street", sites=["pexels"], query_narrow="calm street",
                              query_medium="street", query_broad="street")}
        _si_segs = _mk("Map of the border", "ends here.")
        _si_idx = merge_segments_into_sentences(_si_segs)
        with tempfile.TemporaryDirectory() as _td:
            _o = os.path.join(_td, "r.json")
            run_repair_cycle(
                client=None, current_model="m", fallback_queue=[], segments=_si_segs,
                results={"1": entry(scene="A map", sites=["pexels"], query_narrow="map"), "2": entry(scene="Calm street")},
                exhausted_models={}, checkpoint_path=os.path.join(_td, "c.json"), src_hash="h",
                sources_mode=SOURCES_MIX, strict_mode=2, output_path=_o, call_batch_fn=_m_si, sentence_index=_si_idx,
            )
        check("repair: sentence_index из run_repair_cycle доходит до call_batch_fn без пересчёта",
              len(_si_seen) >= 1 and all(x is _si_idx for x in _si_seen), True)

        if real_srt is not None:
            rsegs = parse_srt(real_srt)
            r_idx = merge_segments_into_sentences(rsegs)
            rb = make_batches(rsegs, DEFAULT_BATCH_SIZE, CONTEXT_WINDOW)
            pr1 = build_prompt(rb[0][0], rb[0][1], rb[0][2], sentence_index=r_idx)
            pr2 = build_prompt(rb[1][0], rb[1][1], rb[1][2], sentence_index=r_idx)
            check("real batch: сегмент 100 в батче 1, 101 в батче 2",
                  ("### Segment 100\n" in pr1, "### Segment 101\n" in pr1, "### Segment 101\n" in pr2), (True, False, True))
            check("real batch: предложение 100-101 целиком в обоих батчах",
                  ("<<" in pr1 and r_idx[1][100].text in pr1.replace(">>", "").replace("<<", ""),
                   r_idx[1][101].text in pr2.replace(">>", "").replace("<<", "")), (True, True))
            check("real batch: в блоках нет пустых служебных строк вида 'Before (context only): '",
                  not re.search(r"(Before|After) \(context only\): *$", pr1 + pr2, re.M), True)

        # 9ж. ссылки на предложение и чистота
        l_sen, l_by = merge_segments_into_sentences(_mk("A b", "c.", "D e."))
        check("merge: by_segment ссылается на тот же объект", (l_by[1] is l_sen[0], l_by[2] is l_sen[0], l_by[3] is l_sen[1]), (True, True, True))
        check("merge: номера предложений с 1", [x.number for x in l_sen], [1, 2])
        check("merge: spans", (l_sen[0].text, l_sen[0].spans), ("A b c.", {1: (0, 3), 2: (4, 6)}))
        before = [(s.index, s.text) for s in _mk("A b", "c.")]
        merge_segments_into_sentences(_mk("A b", "c."))
        check("merge: функция не меняет сегменты", [(s.index, s.text) for s in _mk("A b", "c.")], before)

        # 9и. Окна соседей REPAIR расширяются до границ предложений
        class _WarnCap(logging.Handler):
            def __init__(self):
                super().__init__(level=logging.WARNING)
                self.msgs: list[str] = []
            def emit(self, record):
                self.msgs.append(record.getMessage())

        def _nw(segs, tgt, win, with_idx=True):
            sidx = merge_segments_into_sentences(segs) if with_idx else None
            pos = next(i for i, x in enumerate(segs) if x.index == tgt)
            b, a = _neighbor_window(pos, segs, win, sidx, tgt)
            return [x.index for x in b], [x.index for x in a]

        check("константа: REPAIR_MAX_EXPAND_SEGMENTS = 20", REPAIR_MAX_EXPAND_SEGMENTS, 20)
        # 12 сегментов, предложения: [1-4] [5-6] [7-12]
        ns = _mk("a b", "c d", "e f", "g h.", "i j", "k l.", "m n", "o p", "q r", "s t", "u v", "w x.")
        # граница радиуса уже на границе предложения: окно совпадает со старым
        old_b, old_a = _nw(ns, 7, 2, with_idx=False)
        new_b, new_a = _nw(ns, 7, 2)
        check("neighbors-window: сегмент 7, радиус 2 - слева граница совпала, окно как раньше", (old_b, new_b), ([5, 6], [5, 6]))
        check("neighbors-window: справа расширено ровно на недостающие сегменты", (old_a, new_a), ([8, 9], [8, 9, 10, 11, 12]))
        old_b, old_a = _nw(ns, 9, 1, with_idx=False)
        new_b, new_a = _nw(ns, 9, 1)
        check("neighbors-window: сегмент 9, радиус 1 - слева +1 до начала предложения 7..12",
              (old_b, new_b), ([8], [7, 8]))
        check("neighbors-window: сегмент 9, справа расширено до конца предложения",
              (old_a, new_a), ([10], [10, 11, 12]))
        old_b, old_a = _nw(ns, 3, 2, with_idx=False)
        new_b, new_a = _nw(ns, 3, 2)
        check("neighbors-window: слева край файла, справа конец предложения [1-4]",
              (old_b, new_b, old_a, new_a), ([1, 2], [1, 2], [4, 5], [4, 5, 6]))
        # окно не режет предложения: начало окна = начало предложения, конец = конец
        sen_n, by_n = merge_segments_into_sentences(ns)
        bad_cut = []
        for tgt in range(1, 13):
            for win in (1, 2, 3, 10):
                b_, a_ = _nw(ns, tgt, win)
                if b_ and by_n[b_[0]].seg_indices[0] != b_[0]:
                    bad_cut.append((tgt, win, "L"))
                if a_ and by_n[a_[-1]].seg_indices[-1] != a_[-1]:
                    bad_cut.append((tgt, win, "R"))
        check("neighbors-window: окно начинается и кончается на границах предложений", bad_cut, [])
        # формат: круг 2, distance от проверяемого сегмента, в том числе для добавленных
        nb_x = format_neighbors_context(9, ns, None, 1, with_distance=True, sentence_index=(sen_n, by_n))
        check("neighbors-window круг 2: distance 1..3, добавленные сегменты с расстоянием от сегмента 9",
              all(t in nb_x for t in ("distance 1 | before [8]: o p", "distance 2 | before [7]: m n",
                                      "distance 1 | after [10]: s t", "distance 3 | after [12]: w x.")), True)
        check("neighbors-window круг 2: нет distance 4", "distance 4" in nb_x, False)
        nb_c1 = format_neighbors_context(9, ns, None, 1, sentence_index=(sen_n, by_n))
        check("neighbors-window круг 1: блоки BEFORE/AFTER с расширением",
              "Context BEFORE:" in nb_c1 and "[7] Text: m n" in nb_c1 and "[12] Text: w x." in nb_c1, True)
        check("neighbors-window: без sentence_index - прежний вывод",
              format_neighbors_context(9, ns, None, 1) == format_neighbors_context(9, ns, None, 1, sentence_index=None)
              and "[7]" not in format_neighbors_context(9, ns, None, 1), True)
        # предохранитель: текст без пунктуации
        long_segs = _mk(*[f"w{i} x{i}" for i in range(1, 101)])
        wc = _WarnCap()
        logging.getLogger().addHandler(wc)
        try:
            lb, la = _nw(long_segs, 50, 3)
        finally:
            logging.getLogger().removeHandler(wc)
        check("лимит: на сторону добавлено не больше 20 (слева 3+20, справа 3+20)", (len(lb), len(la)), (23, 23))
        check("лимит: окно обрезано ровно по лимиту", (lb[0], la[-1]), (50 - 23, 50 + 23))
        check("лимит: предупреждение в логе (слева и справа)",
              (sum("слева" in m for m in wc.msgs), sum("справа" in m for m in wc.msgs)), (1, 1))
        # реальный SRT: лимит не срабатывает ни для одного сегмента, оба круга
        if real_srt is not None:
            wc = _WarnCap()
            logging.getLogger().addHandler(wc)
            max_add = 0
            try:
                for rs in rsegs:
                    for win in (CONTEXT_WINDOW, REPAIR2_CONTEXT_WINDOW):
                        pos_ = rs.index - rsegs[0].index
                        ob, oa = _neighbor_window(pos_, rsegs, win, None, rs.index)
                        nb_, na_ = _neighbor_window(pos_, rsegs, win, r_idx, rs.index)
                        max_add = max(max_add, len(nb_) - len(ob), len(na_) - len(oa))
            finally:
                logging.getLogger().removeHandler(wc)
            check("real SRT: предупреждений о лимите нет", wc.msgs, [])
            check("real SRT: максимальное добавление на сторону <= 20", max_add <= REPAIR_MAX_EXPAND_SEGMENTS, True)
            print(f"INFO  real SRT: максимум добавленных сегментов на сторону = {max_add}")

        # 9в. Страховка query_broad у архивного сегмента с именем (этап 3в)
        def arch(broad, **over):
            return entry(sites=["wikimedia", "loc"], is_entity=True, entity_keywords=["Mehmed VI"],
                         query_broad=broad, **over)

        def broad_repair_run(res, strict=2):
            """Прогон run_repair_cycle с заглушкой: возвращает (номера в REPAIR по кругам, repair_info кругов)."""
            segs = [Segment(i, "00:00:00,000", "00:00:01,000", f"text {i}") for i in sorted(int(k) for k in res)]
            calls: list[tuple[list[int], dict]] = []

            def mock(client, model, chunk, cb, ca, mode=SOURCES_MIX, repair_info=None, sentence_index=None):
                calls.append(([x.index for x in chunk], repair_info))
                return {str(x.index): res[str(x.index)] for x in chunk}  # модель ничего не меняет

            with tempfile.TemporaryDirectory() as td2:
                out2 = os.path.join(td2, "r.json")
                code2 = run_repair_cycle(
                    client=None, current_model="m", fallback_queue=[], segments=segs, results=dict(res),
                    exhausted_models={}, checkpoint_path=checkpoint_path_for(out2), src_hash="h",
                    sources_mode=SOURCES_MIX, strict_mode=strict, output_path=out2, call_batch_fn=mock,
                )
            return code2, calls

        for w in BROAD_BANNED_WORDS:
            res_w = {"1": arch(f"old {w} sitting"), "2": entry(), "3": arch("palace hall")}
            code_w, calls_w = broad_repair_run(res_w)
            check(f"broad: слово {w} -> в REPAIR только архивный с именем",
                  [c[0] for c in calls_w][:1], [[1]])
        not_hit = ["manual labor", "human figure", "boyfriend", "man-made lake", "portraits of kings",
                   "men at work", "women", "palace hall", "super-man", None]
        for q in not_hit:
            check(f"broad: не срабатывает на {q!r}", check_archive_broad_words({"1": arch(q)}), {})
        for q in ["woman's hat", "woman\u2019s hat", "WOMAN", "Portrait", "the Boy.", "'girl'"]:
            check(f"broad: срабатывает на {q!r}", list(check_archive_broad_words({"1": arch(q)})), ["1"])
        for src in (["pexels", "pixabay"],):
            check("broad: стоковый сегмент с запрещённым словом не затронут",
                  check_archive_broad_words({"1": entry(sites=src, is_entity=True, entity_keywords=["X"],
                                                        query_broad="man portrait")}), {})
        check("broad: архивный без имени не затронут",
              check_archive_broad_words({"1": entry(sites=["wikimedia"], query_broad="royal man")}), {})
        iss_b = check_archive_broad_words({"1": arch("a woman portrait")}, lang="en")["1"][0]
        check("broad: причина на английском с найденными словами",
              ("portrait, woman" in iss_b) and iss_b.isascii(), True)
        # причина доходит до repair_info; сегмент с двумя причинами в REPAIR один раз
        res_d = {"1": arch("man portrait", scene="A map on a desk")}
        code_d, calls_d = broad_repair_run(res_d)
        ids_d = [i for c in calls_d for i in c[0]]
        check("broad: сегмент с двумя причинами не дублируется (круг 1 и 2 по одному разу)", ids_d, [1, 1])
        iss_d = calls_d[0][1][1]["issues"]
        check("broad: в issues и junk-причина, и причина query_broad",
              (len(iss_d), any("query_broad contains" in x for x in iss_d),
               any("forbidden shot type" in x for x in iss_d)), (2, True, True))
        check("broad: без чистых сегментов REPAIR не вызывается",
              broad_repair_run({"1": arch("palace"), "2": entry()})[1], [])
        code_s1, _ = broad_repair_run({"1": arch("man")}, strict=1)
        code_s2, _ = broad_repair_run({"1": arch("man")}, strict=2)
        check("broad: после круга 2 слово осталось -> штатно strict=1 код 1, strict=2 код 0", (code_s1, code_s2), (1, 0))

        # 9г. visual_value: разбор, зажим, ошибки, ремонт, чекпоинт, лог
        import tempfile as _tf
        st_v: Counter = Counter()
        e_v = _normalize_entry(entry(visual_value=85), 1, st_v)
        check("visual_value: валидное проходит, поле последнее", (e_v["visual_value"], list(e_v)[-1]), (85, "visual_value"))
        e_v = _normalize_entry(entry(visual_value=150), 1, st_v)
        check("visual_value: 150 зажимается до 100 + причина в stats", (e_v["visual_value"], st_v["visual_value_clamped"]), (100, 1))
        check("visual_value: -5 -> 0", _normalize_entry(entry(visual_value=-5), 1)["visual_value"], 0)
        check("visual_value: 85.0 -> 85", _normalize_entry(entry(visual_value=85.0), 1)["visual_value"], 85)
        _no_vv = entry()
        del _no_vv["visual_value"]
        for label, bad in (("нет поля", _no_vv), ("None", entry(visual_value=None)), ("строка", entry(visual_value="high")),
                           ("bool", entry(visual_value=True)), ("дробное", entry(visual_value=85.5))):
            try:
                _normalize_entry(bad, 7)
                raised = False
            except ValueError:
                raised = True
            check(f"visual_value: {label} -> ValueError (повтор батча)", raised, True)
        check("visual_value: в required схемы и в REQUIRED_ENTRY_KEYS",
              ("visual_value" in SEGMENT_ENTRY_SCHEMA.required, "visual_value" in REQUIRED_ENTRY_KEYS,
               "visual_value" in SEGMENT_ENTRY_SCHEMA.property_ordering), (True, True, True))
        check("visual_value: правило в системной инструкции", "7. visual_value" in build_system_instruction(SOURCES_MIX), True)
        with _tf.TemporaryDirectory() as td_v:
            out_v = os.path.join(td_v, "r.json")
            segs_v = [Segment(1, "00:00:00,000", "00:00:01,000", "a"), Segment(2, "00:00:01,000", "00:00:02,000", "b")]
            res_v = {"1": entry(visual_value=30), "2": entry(scene="A map", query_narrow="battle map", visual_value=42)}
            prompts_v: list[str] = []

            def mock_v(client, model, chunk, cb, ca, mode=SOURCES_MIX, repair_info=None, sentence_index=None):
                prompts_v.append(build_prompt(chunk, repair_info=repair_info, sentence_index=sentence_index))
                return {"2": entry(scene="A fortress wall", query_narrow="fortress wall", visual_value=99)}

            run_repair_cycle(client=None, current_model="m", fallback_queue=[], segments=segs_v, results=res_v,
                             exhausted_models={}, checkpoint_path=checkpoint_path_for(out_v), src_hash="h",
                             sources_mode=SOURCES_MIX, strict_mode=2, output_path=out_v, call_batch_fn=mock_v)
            with open(out_v) as f_v:
                saved_v = json.load(f_v)
            check("visual_value: REPAIR не меняет прежнюю оценку", (saved_v["1"]["visual_value"], saved_v["2"]["visual_value"], "fortress" in saved_v["2"]["scene"]), (30, 42, True))
            check("visual_value: Previous visual_value в промпте REPAIR", bool(prompts_v) and "Previous visual_value: 42" in prompts_v[0], True)
            # старый чекпоинт (v2) игнорируется
            cp_v = os.path.join(td_v, "old.checkpoint.json")
            with open(cp_v, "w") as f_v:
                json.dump({"source_hash": "h", "schema_version": 2, "results": {"1": entry()}, "exhausted_models": {}}, f_v)
            check("visual_value: чекпоинт v2 без оценок не подхватывается", load_checkpoint(cp_v, "h")[0], {})
        check("visual_value: строка распределения",
              summarize_visual_values({"1": {"visual_value": 0}, "2": {"visual_value": 19}, "3": {"visual_value": 20},
                                       "4": {"visual_value": 80}, "5": {"visual_value": 100}}),
              "visual_value: сегментов 5, min 0, медиана 20, max 100; корзины 0-19: 2, 20-49: 1, 50-79: 0, 80-100: 2")

        # 10. parse_sources_mode и parse_strict_mode (валидация env и CLI)
        check("parse mode: CLI валидный (1 микс), приоритет над env", parse_sources_mode("1", "3"), SOURCES_MIX)
        check("parse mode: env валидный (3 сток)", parse_sources_mode(None, "3"), SOURCES_STOCK)
        check("parse mode: '2' -> архив", parse_sources_mode("2", None), SOURCES_ARCHIVE)
        check("parse mode: '3' -> сток", parse_sources_mode("3", None), SOURCES_STOCK)
        check("parse mode: 2 (int) из env -> архив", parse_sources_mode(None, 2), SOURCES_ARCHIVE)
        check("parse mode: невалидный -> микс", parse_sources_mode("invalid", None), SOURCES_MIX)
        check("parse mode: None -> микс", parse_sources_mode(None, None), SOURCES_MIX)
        check("parse mode: пустая строка -> микс", parse_sources_mode("", None), SOURCES_MIX)
        check("parse mode: '4' и '0' -> микс", (parse_sources_mode("4", None), parse_sources_mode("0", None)), (SOURCES_MIX, SOURCES_MIX))
        check("parse strict: CLI валидный", parse_strict_mode("2", "1"), 2)
        check("parse strict: env валидный", parse_strict_mode(None, "2"), 2)
        check("parse strict: невалидный -> дефолт 2", parse_strict_mode("bad", None), 2)
        check("parse strict: None -> дефолт 2", parse_strict_mode(None, None), 2)

        # 11. MEDIA_MODE: разбор, принудительный тип, подсказка в инструкции
        check("media_mode: None -> 1", parse_media_mode(None), 1)
        check("media_mode: пусто и пробелы -> 1", (parse_media_mode(""), parse_media_mode("  ")), (1, 1))
        check("media_mode: 1/2/3 с пробелами", [parse_media_mode(v) for v in ("1", " 2 ", "3\n")], [1, 2, 3])
        for bad_mm in ("0", "4", "video", "1.0", "1,2", "-1"):
            try:
                parse_media_mode(bad_mm)
                got_mm = "нет ошибки"
            except ValueError as e_mm:
                got_mm = "ошибка" if (repr(bad_mm) in str(e_mm) and "1" in str(e_mm)) else "плохое сообщение"
            check(f"media_mode: мусор {bad_mm!r} -> ошибка", got_mm, "ошибка")
        check("media_mode: режим 2 -> video при ответе image", _normalize_entry(entry(type="image"), 1, media_mode=2)["type"], "video")
        check("media_mode: режим 3 -> image при ответе video", _normalize_entry(entry(type="video"), 1, media_mode=3)["type"], "image")
        check("media_mode: режим 2/3 игнорируют битый type",
              (_normalize_entry(entry(type="gif"), 1, media_mode=2)["type"], _normalize_entry(entry(type="gif"), 1, media_mode=3)["type"]),
              ("video", "image"))
        st_mm = Counter()
        _normalize_entry(entry(type="gif"), 1, st_mm, media_mode=2)
        check("media_mode: режим 2 не пишет type_fixed", st_mm.get("type_fixed", 0), 0)
        check("media_mode: режим 1 оставляет ответ Gemini", _normalize_entry(entry(type="image"), 1)["type"], "image")
        check("media_mode: режим 1 битый type -> video", _normalize_entry(entry(type="gif"), 1)["type"], "video")
        base_si = build_system_instruction(SOURCES_MIX)
        check("media_mode: режим 1 не меняет инструкцию", build_system_instruction(SOURCES_MIX, media_mode=1), base_si)
        check("media_mode: подсказка только в 2/3",
              ("MOVING" in build_system_instruction(SOURCES_MIX, media_mode=2), "STILL" in build_system_instruction(SOURCES_MIX, media_mode=3),
               "MEDIA MODE" in base_si), (True, True, False))

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
        help="Режим источников: 1 (микс, по умолчанию), 2 (только архив), 3 (только сток).",
    )
    parser.add_argument(
        "--strict", default=None,
        help="Строгость проверки запросов: 1 (калибровка/код 1 при ошибках), 2 (мягко/warning, код 0; по умолчанию).",
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

    try:
        media_mode = parse_media_mode(os.environ.get("MEDIA_MODE"))
    except ValueError as e:
        logging.error("%s", e)
        return 1
    # Режим типа медиа доходит до call_gemini_batch через partial (и в основном цикле, и в REPAIR)
    call_batch = partial(call_gemini_batch, media_mode=media_mode)
    logging.info("Режим типа медиа: %s (%s)", media_mode, MEDIA_MODE_NAMES[media_mode])

    mode_names = {SOURCES_MIX: "микс", SOURCES_ARCHIVE: "только архив", SOURCES_STOCK: "только сток"}
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

    # Склейка в предложения - один раз по всему списку, до разбиения на батчи
    sentence_index = merge_segments_into_sentences(segments)
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
                batch_result = call_batch(
                    client, current_model, batch, context_before, context_after,
                    mode=sources_mode, sentence_index=sentence_index,
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
        call_batch_fn=call_batch,
        sentence_index=sentence_index,
    )


if __name__ == "__main__":
    sys.exit(main())
