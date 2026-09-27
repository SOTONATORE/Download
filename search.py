#!/usr/bin/env python3
"""
search.py

Недостающее звено между generate_queries.py и download.py: читает requests.json
(сгенерированный generate_queries.py) и для каждого сегмента подбирает
подходящее стоковое/архивное фото или видео на pexels/pixabay/wikimedia/nasa/loc,
используя лицензионный фильтр, текстовый фильтр по сущностям и CLIP-ранжирование
по схожести превью с текстом запроса.

На выходе:
    links.txt   - строки "номер: URL" в формате, который уже понимает download.py
                  (передаётся туда как переменная окружения INPUT_LINKS)
    missing.txt - номера сегментов, для которых ничего подходящего не нашлось

Использование:
    python search.py --input requests.json --links-output links.txt --missing-output missing.txt

Переменные окружения:
    PEXELS_API_KEY, PIXABAY_API_KEY   - обязательны (в т.ч. для fallback_query)
    SEARCH_SEM_PEXELS / _PIXABAY / _WIKIMEDIA / _NASA / _LOC
        - per-site семафоры одновременных запросов (умолч. 5/5/10/10/10)
    SEARCH_LOC_MIN_INTERVAL_SECONDS
        - глобальный (на весь запуск, не per-задача) минимальный интервал между
          ПОСЛЕДОВАТЕЛЬНЫМИ запросами к LOC, вне зависимости от того, сколько
          сегментов обрабатывается параллельно. Это отдельный механизм от
          SEARCH_SEM_LOC: семафор ограничивает только конкурентность (сколько
          запросов летит ОДНОВРЕМЕННО), а не частоту (сколько запросов в секунду
          в принципе уходит) - при высокой глобальной параллельности сегментов
          семафор сам по себе не мешает 10 запросам уйти почти синхронно, а затем
          ещё 10 через долю секунды, и LOC отвечает 429 почти на всё подряд.
          Умолч. 5.0 сек (~12 запросов/мин) - подобрано по ОФИЦИАЛЬНОЙ документации
          LOC (https://www.loc.gov/apis/json-and-yaml/working-within-limits/):
          лимит JSON/YAML API - 20 запросов/мин, при превышении - блокировка IP на
          1 ЧАС (не на минуту!). 5с даёт ~40% запас против этого потолка - выбран
          сознательно с большим запасом, а не впритык к 20/мин, т.к. сам LOC
          предупреждает, что при высокой нагрузке на их стороне лимит может
          эффективно снижаться и ниже заявленного, а цена ошибки - часовой бан,
          а не просто лишняя секунда ожидания на сегмент. Если 429 всё равно
          появляются - увеличивайте ещё (7-10 сек и выше).
    SEARCH_GLOBAL_CONCURRENCY  - сколько сегментов обрабатывать параллельно (умолч. 40)
    SEARCH_CLIP_CONCURRENCY   - сколько CLIP-инференсов одновременно (умолч. 2, CPU-bound)
    SEARCH_CANDIDATES_PER_SITE - сколько топ-кандидатов с сайта пускать под CLIP (умолч. 5)
    SEARCH_CLIP_MODEL / SEARCH_CLIP_PRETRAINED - модель open_clip (умолч. ViT-B-32-quickgelu / openai)
    SEARCH_SIM_MIN_THRESHOLD   - минимальный raw CLIP cosine similarity, чтобы кандидат
        вообще прошёл в пул (умолч. 0.21 - см. "Шестое уточнение" в докстринге ниже)
    SEARCH_SIM_ACCEPT_THRESHOLD - similarity, при которой прекращаем перебор сайтов
        и берём кандидата сразу (умолч. 0.30 - см. там же)

Возвращаемые коды:
    0 - links.txt и missing.txt успешно записаны (даже если часть/все сегменты в missing)
    1 - структурная ошибка (невалидный API-ключ, битый requests.json и т.п.)

Зависимости:
    pip install aiohttp pillow open_clip_torch
    (torch лучше ставить отдельно, CPU-only wheel, см. комментарий в конце файла)

ВАЖНОЕ ДОПУЩЕНИЕ по интерпретации п.6-7 исходного ТЗ (спецификация была неоднозначна
в этом месте): "лучший кандидат сайта >= порога accept" останавливает перебор сайтов, но
выбор и дедуп-бронирование всегда идёт по НАКОПЛЕННОМУ пулу кандидатов (>= порога min) со
всех уже пройденных сайтов, отсортированному по убыванию similarity - а не только по
кандидатам текущего (последнего) сайта. Это единственное прочтение, совместимое одновременно
с "не пробовать остальные сайты" (п.6) и "среди кандидатов ПО ВСЕМ ПЕРЕБРАННЫМ САЙТАМ" (п.7).
Если имелось в виду не так - легко поменять в try_claim_pool/process_segment.

Второе допущение: правило "нет превью для CLIP -> дисквалифицировать именно этого
кандидата, не весь сайт" (изначально уточнено для видео на Wikimedia) применено ко ВСЕМ
сайтам единообразно в fetch_preview_bytes/score_candidates - это строго безопаснее и
не создаёт особых случаев.

Третье допущение (rate-limiting LOC): семафор SEARCH_SEM_LOC и rate-limiter
SEARCH_LOC_MIN_INTERVAL_SECONDS решают РАЗНЫЕ задачи и работают одновременно -
семафор по-прежнему ограничивает, сколько запросов к LOC могут физически висеть
в полёте одновременно, а rate-limiter поверх этого гарантирует минимальный зазор
по времени между началом двух последовательных запросов (глобально по всему
запуску, а не per-сегмент/per-задача).

Четвёртое (важное) уточнение по LOC, добавленное после реального прогона: по
официальной документации LOC (working-within-limits) превышение лимита JSON/YAML
API (20 запросов/мин) приводит к блокировке IP на ЦЕЛЫЙ ЧАС, а не к обычному
кратковременному 429. Это значит, что как только LOC один раз ответил 429,
дальнейшие ретраи с exponential backoff (секунды-десятки секунд) внутри ТЕКУЩЕГО
запуска бессмысленны - блокировка всё равно не снимется за время работы CI-джобы.
Поэтому search_loc теперь вызывает http_get_json с treat_429_as_exhaustion=True
(как pexels/pixabay) - первый же 429 сразу помечает "loc" исчерпанным на весь
остаток запуска, вместо повторных попыток достучаться до сайта, который уже точно
не ответит. Сегменты, где loc стоит не последним в sites, просто продолжают перебор
остальных сайтов - это поведение уже было и не менялось.

Также LOC может отдать вместо JSON html-страницу с CAPTCHA при перегрузке на своей
стороне (см. ту же страницу документации: "users may encounter ... HTML pages with
CAPTCHAs even when operating below the rates listed above") - это ловится отдельно
как aiohttp.ContentTypeError при попытке resp.json() и обрабатывается так же, как
429 (тот же treat_429_as_exhaustion), т.к. по сути это тот же сигнал "нас блокируют".

Пятое (критичное) уточнение - баг конфигурации CLIP, найденный после прогона с
0 найденных из 126 сегментов СРАЗУ ПО ВСЕМ сайтам (не только loc): модель бралась
как SEARCH_CLIP_MODEL=ViT-B-32 (без суффикса) с pretrained=openai. Это известный
баг open_clip (https://github.com/mlfoundations/open_clip/issues/771): чекпоинт
"openai" для B/32 обучен с QuickGELU-активацией, но конфиг архитектуры "ViT-B-32"
(без суффикса) по умолчанию использует обычный GELU - в логах это видно как warning
"QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag
'openai' (quick_gelu=True)". Из-за этого модель технически загружается и работает
без ошибок, но выдаёт бессмысленные эмбеддинги - и КАЖДЫЙ кандидат на КАЖДОМ сайте
получает around-случайный/заниженный similarity, падающий ниже порога. Это системная
причина сразу для всех сайтов одновременно, не связанная с лицензиями, запросами или
сущностями. Исправлено: дефолт SEARCH_CLIP_MODEL сменён на ViT-B-32-quickgelu (тот же
pretrained=openai, но с правильной активацией).

Шестое уточнение (по итогам прогона на 126 сегментах после фикса QuickGELU) - пороги
similarity были подобраны "на глаз" и оказались нереалистично высокими для СЫРОГО (без
температурного скейлинга/софтмакса) косинусного сходства CLIP: у настоящих релевантных
пар текст-картинка raw cosine similarity типично лежит в диапазоне ~0.2-0.35, а не
0.5-0.85, как было выставлено изначально. Сводка по прогону это подтвердила напрямую:
pexels/loc присылали в CLIP сотни нормальных кандидатов с avg similarity 0.20-0.28 и
best 0.33-0.36 - это здоровые значения для настоящих совпадений, просто ниже прежнего
порога отсечения 0.5, из-за чего пул почти всегда оказывался пуст. Пороги пересчитаны
под этот диапазон (SIM_MIN_THRESHOLD 0.5->0.21, SIM_ACCEPT_THRESHOLD 0.85->0.30) и
вынесены в переменные окружения SEARCH_SIM_MIN_THRESHOLD / SEARCH_SIM_ACCEPT_THRESHOLD,
чтобы их можно было донастроить по факту (например по перцентилю на своей выборке
сегментов), не трогая код.

Седьмое уточнение - у pixabay и wikimedia в тестовом прогоне 100% превью не скачивались
(fetch_preview_bytes возвращал None для всех кандидатов, 0 ушло в CLIP), при этом сам
поиск (raw/license/keyword) отрабатывал нормально. Причина - типичная защита CDN от
хотлинкинга: запрос к самому медиафайлу без Referer (иногда и Origin), указывающего на
страницу-источник, отклоняется (403/406 и т.п.), даже если User-Agent в порядке (User-
Agent уже был поправлен раньше для API-запросов к Wikimedia, но не для скачивания самих
превью-картинок). Исправлено: fetch_preview_bytes теперь подставляет Referer (и Origin,
выведенный из него) по каждому сайту - для wikimedia используется page_url конкретного
кандидата (страница файла), если он есть, иначе общий https://commons.wikimedia.org/;
для остальных сайтов - их основной домен. Также при неуспехе (status != 200 или сетевая
ошибка) теперь логируется DEBUG-строка с сайтом, id кандидата, HTTP-статусом и превью
тела ответа - раньше fetch_preview_bytes молча возвращал None без единой детали, что и
не давало отличить блокировку по Referer от любой другой причины.

Восьмое уточнение - text_matches_keywords делал точное вхождение подстроки, из-за чего
разные способы транслитерации одного и того же имени (например "Abdulmecid" в
entity_keywords против "Abdülmecid" в тексте кандидата, или "Abdul Mejid" против
"Abdulmecid") не совпадали, хотя семантически это один и тот же человек. Не подключая
внешних fuzzy-библиотек, добавлены два дешёвых слоя поверх прежней точной проверки:
(1) нормализация через unicodedata (NFKD + снятие комбинирующих диакритических знаков),
которая сама по себе схлопывает "Abdülmecid" -> "abdulmecid"; (2) до-проверка на уровне
отдельных слов текста через стандартный difflib.SequenceMatcher (стандартная библиотека,
без новых зависимостей) с порогом схожести - ловит близкие, но не идентичные варианты
написания вроде "Mejid"/"Mecid". Это осознанно не полноценный fuzzy-matching (без
rapidfuzz и т.п.) - как и просили, достаточно "чего-то попроще" поверх точного совпадения.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import functools
import io
import json
import logging
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

try:
    import aiohttp
except ImportError:
    print("Не найден aiohttp. Установите: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

try:
    import torch
    import open_clip
    from PIL import Image
except ImportError:
    print(
        "Не найдены torch/open_clip/pillow. Установите (CPU-only torch, чтобы не тянуть "
        "CUDA-сборку в CI):\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "  pip install open_clip_torch pillow\n",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Константы и настройки из окружения
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_INPUT = "requests.json"
DEFAULT_LINKS_OUTPUT = "links.txt"
DEFAULT_MISSING_OUTPUT = "missing.txt"

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 4
MAX_BACKOFF_SECONDS = 60

# См. "Шестое уточнение" в докстринге модуля: raw CLIP cosine similarity для реально
# релевантных пар текст-картинка типично лежит в диапазоне ~0.2-0.35, а не 0.5-0.85 -
# пороги пересчитаны под это и вынесены в окружение, чтобы их можно было донастроить
# без правки кода (например по перцентилю на собственной выборке сегментов).
SIM_ACCEPT_THRESHOLD = float(os.environ.get("SEARCH_SIM_ACCEPT_THRESHOLD", 0.30))
SIM_MIN_THRESHOLD = float(os.environ.get("SEARCH_SIM_MIN_THRESHOLD", 0.21))

CANDIDATES_PER_SITE = int(os.environ.get("SEARCH_CANDIDATES_PER_SITE", 5))

CLIP_MODEL_NAME = os.environ.get("SEARCH_CLIP_MODEL", "ViT-B-32-quickgelu")
CLIP_PRETRAINED = os.environ.get("SEARCH_CLIP_PRETRAINED", "openai")
CLIP_CONCURRENCY = int(os.environ.get("SEARCH_CLIP_CONCURRENCY", 2))

SEMAPHORE_DEFAULTS = {
    "pexels": int(os.environ.get("SEARCH_SEM_PEXELS", 5)),
    "pixabay": int(os.environ.get("SEARCH_SEM_PIXABAY", 5)),
    "wikimedia": int(os.environ.get("SEARCH_SEM_WIKIMEDIA", 10)),
    "nasa": int(os.environ.get("SEARCH_SEM_NASA", 10)),
    "loc": int(os.environ.get("SEARCH_SEM_LOC", 10)),
}
GLOBAL_SEGMENT_CONCURRENCY = int(os.environ.get("SEARCH_GLOBAL_CONCURRENCY", 40))

# Отдельный от семафора механизм - см. докстринг модуля, раздел "Третье"/"Четвёртое"
# допущение. Дефолт подобран по официальной документации LOC (working-within-limits:
# 20 запросов/мин у JSON/YAML API, час блокировки при превышении), с большим запасом
# (5с = 12 запросов/мин, ~40% ниже потолка) - цена ошибки высокая (часовой бан), поэтому
# лучше перестраховаться, чем экономить секунды на сегмент.
LOC_MIN_INTERVAL_SECONDS = float(os.environ.get("SEARCH_LOC_MIN_INTERVAL_SECONDS", 5.0))

PREVIEW_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
}

# См. "Седьмое уточнение" в докстринге модуля: без Referer (и часто Origin) многие CDN
# отдают 403/406 на запрос к самому медиафайлу, даже если User-Agent в порядке. Значения
# ниже - "страница-источник по умолчанию" для сайтов, где у кандидата нет собственного
# page_url; для wikimedia в fetch_preview_bytes используется page_url конкретного
# кандидата, если он задан (это надёжнее общего домена).
PREVIEW_REFERERS = {
    "pexels": "https://www.pexels.com/",
    "pixabay": "https://pixabay.com/",
    "wikimedia": "https://commons.wikimedia.org/",
    "nasa": "https://images.nasa.gov/",
    "loc": "https://www.loc.gov/",
}

# Wikimedia (и вообще большинство API) с некоторых пор жёстко требуют внятный
# User-Agent с указанием, что это за инструмент и как с ним связаться - иначе 403
# ("Please set a user-agent and respect our robot policy"). Это НЕ временная ошибка,
# ретраить её бессмысленно - см. правку в http_get_json ниже. Подставьте сюда свой
# реальный контакт/ссылку на репозиторий - Wikimedia может ужесточить проверку и на
# осмысленность значения, не только на его наличие.
SESSION_USER_AGENT = (
    "MediaSearchPipeline/1.0 "
    "(https://github.com/SOTONATORE/Download; contact: fordlababit@gmail.com)"
)

# Белый список LicenseShortName для Wikimedia Commons (регистронезависимо, по префиксу).
# "pd" матчится только как отдельное "слово" (PD, PD-old, PD-US, ...), чтобы не словить
# случайные ложные совпадения.
_WM_PD_RE = re.compile(r"^pd([-\s]|$)", re.IGNORECASE)
_WM_FREE_PREFIXES = ("cc0", "cc-zero", "cc by", "public domain", "fal", "gfdl")


def wikimedia_license_ok(license_short_name: str) -> bool:
    if not license_short_name:
        return False
    v = license_short_name.strip().lower()
    if _WM_PD_RE.match(v):
        return True
    return any(v.startswith(p) for p in _WM_FREE_PREFIXES)


def nasa_license_ok(item_data: dict) -> bool:
    return not item_data.get("copyright")


def loc_license_ok(item: dict) -> bool:
    """Эвристика: LOC не даёт единого чистого поля "свободно/не свободно", поэтому
    консервативно исключаем всё, что явно помечено access_restricted или содержит
    rights_advisory без явной фразы про отсутствие ограничений/public domain."""
    if item.get("access_restricted") is True:
        return False
    advisory = item.get("rights_advisory")
    if isinstance(advisory, list):
        advisory_text = " ".join(str(a) for a in advisory).lower()
    else:
        advisory_text = str(advisory or "").lower()
    if advisory_text:
        if "no known restriction" in advisory_text or "public domain" in advisory_text:
            return True
        return False
    return True


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "").strip()


# ---------------------------------------------------------------------------
# Текстовый фильтр по сущностям (см. "Восьмое уточнение" в докстринге модуля)
# ---------------------------------------------------------------------------

# Порог схожести слов для difflib.SequenceMatcher.ratio(). Подобран консервативно (не
# слишком низко, чтобы не плодить ложные совпадения на коротких словах): ловит замены
# 1-2 символов в словах длиной от ~5 символов (например "mecid"/"mejid" -> ratio 0.80),
# но не сводит вместе произвольные разные короткие слова.
_FUZZY_WORD_RATIO_THRESHOLD = float(os.environ.get("SEARCH_FUZZY_KEYWORD_RATIO", 0.78))
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _normalize_for_match(s: str) -> str:
    """NFKD + снятие комбинирующих диакритических знаков - схлопывает разные способы
    записи одного и того же имени (например "Abdülmecid" -> "abdulmecid"), плюс lower()."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


_MIN_FUZZY_WORD_LEN = 3  # короче - слишком много случайных подстрочных совпадений (например "a", "i", "тон")


def _fuzzy_word_match(keyword_norm: str, text_words: list[str]) -> bool:
    if len(keyword_norm) < _MIN_FUZZY_WORD_LEN:
        return False
    for w in text_words:
        if len(w) < _MIN_FUZZY_WORD_LEN:
            continue
        if keyword_norm in w or w in keyword_norm:
            return True
        if difflib.SequenceMatcher(None, keyword_norm, w).ratio() >= _FUZZY_WORD_RATIO_THRESHOLD:
            return True
    return False


def text_matches_keywords(text: str, keywords: list[str]) -> bool:
    """Сначала точное вхождение подстроки (как раньше, самый дешёвый и надёжный случай),
    затем - если оно не сработало - два дешёвых fuzzy-слоя поверх нормализованного текста:
    снятие диакритики и приблизительное совпадение отдельных слов через difflib. Осознанно
    без внешних библиотек (rapidfuzz и т.п.) - см. "Восьмое уточнение" в докстринге модуля."""
    if not keywords:
        return True
    t_raw = (text or "").lower()
    if any(kw.lower() in t_raw for kw in keywords if kw):
        return True

    norm_text = _normalize_for_match(text)
    if not norm_text:
        return False
    text_words = _WORD_RE.findall(norm_text)

    for kw in keywords:
        if not kw:
            continue
        kw_norm = _normalize_for_match(kw)
        if len(kw_norm) >= _MIN_FUZZY_WORD_LEN and kw_norm in norm_text:
            return True
        if _fuzzy_word_match(kw_norm, text_words):
            return True
    return False


class FatalConfigError(RuntimeError):
    """Структурная ошибка конфигурации (невалидный API-ключ и т.п.) - не ретраится,
    приводит к остановке всего скрипта с кодом 1."""


# ---------------------------------------------------------------------------
# Диагностика воронки (raw -> лицензия -> сущности -> CLIP) по каждому сайту
# ---------------------------------------------------------------------------

@dataclass
class SiteStats:
    """Считает, сколько кандидатов на каждом этапе фильтрации осталось - чтобы по
    финальной сводке было сразу видно, на каком именно шаге воронка обнуляется
    (сырой поиск / лицензия / текстовый фильтр сущностей / отсутствие превью /
    сам CLIP-скоринг), а не только итоговое "найдено 0 из N"."""

    segments_attempted: int = 0
    raw_total: int = 0
    license_ok_total: int = 0
    keyword_ok_total: int = 0
    sent_to_clip_total: int = 0
    preview_missing_total: int = 0
    clip_error_total: int = 0
    clip_scored_total: int = 0
    clip_passed_total: int = 0   # similarity >= SIM_MIN_THRESHOLD
    clip_accept_total: int = 0  # similarity >= SIM_ACCEPT_THRESHOLD
    score_sum: float = 0.0
    best_score: float = 0.0

    def record_score(self, sim: float) -> None:
        self.clip_scored_total += 1
        self.score_sum += sim
        if sim > self.best_score:
            self.best_score = sim
        if sim >= SIM_ACCEPT_THRESHOLD:
            self.clip_accept_total += 1
        if sim >= SIM_MIN_THRESHOLD:
            self.clip_passed_total += 1

    @property
    def avg_score(self) -> float:
        return (self.score_sum / self.clip_scored_total) if self.clip_scored_total else 0.0


def log_site_stats_summary(site_stats: dict) -> None:
    if not site_stats:
        return
    logging.info("=" * 100)
    logging.info("СВОДКА ПО ВОРОНКЕ ФИЛЬТРАЦИИ (диагностика, откуда берутся нули):")
    logging.info(
        "%-18s %6s %7s %8s %9s %8s %9s %8s %8s %8s %7s %7s",
        "сайт", "сегм.", "raw", "лиценз.", "keyword", "->CLIP", "нет прев.",
        "scored", f">={SIM_MIN_THRESHOLD:.2f}", f">={SIM_ACCEPT_THRESHOLD:.2f}", "avg", "best",
    )
    for key in sorted(site_stats.keys()):
        s = site_stats[key]
        logging.info(
            "%-18s %6d %7d %8d %9d %8d %9d %8d %8d %8d %7.3f %7.3f",
            key, s.segments_attempted, s.raw_total, s.license_ok_total,
            s.keyword_ok_total, s.sent_to_clip_total, s.preview_missing_total,
            s.clip_scored_total, s.clip_passed_total, s.clip_accept_total,
            s.avg_score, s.best_score,
        )
    logging.info(
        "Как читать: raw=0 -> сайт вообще ничего не вернул по запросу (сеть/сам API/лимит). "
        "лиценз.=0 при raw>0 -> все кандидаты отсеяны лицензионным фильтром. "
        "keyword=0 при лиценз.>0 -> entity_keywords/is_entity слишком узкие или не совпадают "
        "с текстом кандидатов. нет_прев.=raw (или близко) -> превью не скачиваются (сайт "
        "блокирует PREVIEW_HEADERS/Referer/хотлинкинг - включите DEBUG-логирование, чтобы "
        "увидеть точный статус-код и тело ответа по каждому провалу). scored>0, но "
        "avg/best низкие (ниже SIM_MIN_THRESHOLD) -> CLIP отрабатывает, но ничего не "
        "совпадает по смыслу - либо сам CLIP настроен неверно (см. 'Пятое уточнение' в "
        "докстринге модуля про QuickGELU), либо запросы от generate_queries.py слишком "
        "специфичны/не по делу. Учтите: raw CLIP similarity для здоровых совпадений обычно "
        "лежит в диапазоне ~0.2-0.35 (см. 'Шестое уточнение') - это НЕ то же самое, что "
        "similarity софтмакса/температурного скейлинга."
    )
    logging.info("=" * 100)


# ---------------------------------------------------------------------------
# Глобальный rate-limiter (минимальный интервал между запросами)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Гарантирует минимальный интервал между НАЧАЛОМ двух последовательных запросов,
    глобально на весь запуск - в отличие от asyncio.Semaphore, который ограничивает
    только число одновременно летящих запросов, но не мешает им уйти пачкой один за
    другим. Нужен для сайтов вроде LOC, которые банят по частоте (req/sec), а не
    только по конкурентности.

    Реализация: единый asyncio.Lock сериализует "вход" в лимитер, так что даже при
    большом числе параллельных корутин (до GLOBAL_SEGMENT_CONCURRENCY штук) фактические
    запросы к сайту физически не могут стартовать чаще, чем раз в min_interval секунд,
    независимо от того, сколько сегментов обрабатывается одновременно."""

    def __init__(self, min_interval_seconds: float):
        self.min_interval = max(0.0, min_interval_seconds)
        self._lock = asyncio.Lock()
        self._last_start_ts: Optional[float] = None

    async def wait_turn(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._last_start_ts is not None:
                elapsed = now - self._last_start_ts
                remaining = self.min_interval - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
                    now = loop.time()
            self._last_start_ts = now


# ---------------------------------------------------------------------------
# Модели данных
# ---------------------------------------------------------------------------

@dataclass
class SegmentSpec:
    index: int
    sites: list[str]
    query: str
    fallback_query: Optional[str]
    type: str  # "image" | "video"
    is_entity: bool
    entity_keywords: list[str]


@dataclass
class Candidate:
    site: str
    cand_id: str
    text: str
    license_ok: bool
    preview_url: Optional[str]
    page_url: Optional[str]
    # Если задан - вызывается для получения финального URL (для сайтов, где
    # найденный на этапе поиска URL ещё не тот, что нужно download.py, например NASA).
    final_url_resolver: Optional[Callable[["Context"], Awaitable[Optional[str]]]] = None
    similarity: float = 0.0


@dataclass
class Context:
    session: "aiohttp.ClientSession"
    pexels_api_key: str
    pixabay_api_key: str
    site_semaphores: dict
    global_semaphore: asyncio.Semaphore
    clip_semaphore: asyncio.Semaphore
    used_files_lock: asyncio.Lock
    used_files: set = field(default_factory=set)
    exhausted_sites: set = field(default_factory=set)
    search_cache: dict = field(default_factory=dict)
    search_cache_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    clip: "ClipScorer" = None
    # per-site глобальные rate-limiter'ы (минимальный интервал между запросами).
    # По умолчанию заполнен только для "loc" - см. LOC_MIN_INTERVAL_SECONDS. Сайты,
    # для которых лимитера нет в словаре, им попросту не ограничиваются (используют
    # только семафор конкурентности, как и раньше).
    rate_limiters: dict = field(default_factory=dict)
    # Диагностика воронки фильтрации по каждому сайту (и отдельно "<site>_fallback"
    # для запросов через try_fallback) - см. SiteStats/log_site_stats_summary.
    site_stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

class ClipScorer:
    def __init__(self, model_name: str, pretrained: str):
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._load_lock = asyncio.Lock()

    async def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            logging.info(
                "Загружаю CLIP-модель %s (%s) - при первом запуске без кэша качается "
                "с интернета, дальше держите её в actions/cache.",
                self.model_name, self.pretrained,
            )
            loop = asyncio.get_running_loop()
            model, _, preprocess = await loop.run_in_executor(
                None,
                functools.partial(
                    open_clip.create_model_and_transforms,
                    self.model_name, pretrained=self.pretrained,
                ),
            )
            model.eval()
            tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model, self._preprocess, self._tokenizer = model, preprocess, tokenizer
            logging.info("CLIP-модель загружена.")

    async def score(self, image_bytes: bytes, text: str) -> float:
        await self.ensure_loaded()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._score_sync, image_bytes, text)

    def _score_sync(self, image_bytes: bytes, text: str) -> float:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_input = self._preprocess(image).unsqueeze(0)
        text_input = self._tokenizer([text[:300]])
        with torch.no_grad():
            image_features = self._model.encode_image(image_input)
            text_features = self._model.encode_text(text_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
        return similarity


# ---------------------------------------------------------------------------
# HTTP-хелпер с ретраями/бэкоффом (в духе generate_queries.py) + учёт 429-исчерпания
# ---------------------------------------------------------------------------

async def http_get_json(
    ctx: Context,
    site: str,
    url: str,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    treat_429_as_exhaustion: bool = False,
) -> Any:
    if site in ctx.exhausted_sites:
        return None

    rate_limiter = ctx.rate_limiters.get(site)

    last_error: Optional[BaseException] = None
    for attempt in range(1, MAX_RETRIES + 1):
        async with ctx.site_semaphores[site]:
            # Семафор выше уже ограничил конкурентность (сколько запросов к этому
            # сайту летят ОДНОВРЕМЕННО). rate_limiter, если задан для сайта, отдельно
            # гарантирует минимальный интервал между СТАРТАМИ последовательных
            # запросов - это защищает от ситуации, когда очередная "порция" из
            # N параллельных задач всё равно уходит почти синхронно с предыдущей.
            if rate_limiter is not None:
                await rate_limiter.wait_turn()
            try:
                async with ctx.session.get(
                    url, headers=headers, params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    status = resp.status

                    if status == 429:
                        if treat_429_as_exhaustion:
                            logging.warning(
                                "%s: получен 429 - помечаю сайт исчерпанным до конца "
                                "текущего запуска.", site,
                            )
                            ctx.exhausted_sites.add(site)
                            return None
                        delay = min(INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
                        logging.warning(
                            "%s: 429, попытка %s/%s, жду %.1fs.", site, attempt, MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay + random.uniform(0, 1))
                        last_error = RuntimeError("429 Too Many Requests")
                        continue

                    if status in (401, 403):
                        text = await resp.text()
                        if site in ("pexels", "pixabay"):
                            raise FatalConfigError(
                                f"{site}: HTTP {status} - похоже на невалидный API-ключ. "
                                f"Тело ответа: {text[:300]}"
                            )
                        # Для остальных сайтов 401/403 - НЕ транзиентная ошибка (неверный
                        # User-Agent, política робота и т.п.) - ретраить бессмысленно,
                        # отдаём пустой результат сразу, чтобы не жечь минуты на 5 попыток.
                        logging.error(
                            "%s: HTTP %s - не ретраю (не временная ошибка). Тело: %s",
                            site, status, text[:300],
                        )
                        return None

                    if status >= 500:
                        delay = min(INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
                        logging.warning(
                            "%s: HTTP %s (попытка %s/%s), жду %.1fs.", site, status, attempt, MAX_RETRIES, delay,
                        )
                        last_error = RuntimeError(f"HTTP {status}")
                        await asyncio.sleep(delay)
                        continue

                    if status != 200:
                        text = await resp.text()
                        logging.warning(
                            "%s: неожиданный статус %s (попытка %s/%s): %s",
                            site, status, attempt, MAX_RETRIES, text[:200],
                        )
                        last_error = RuntimeError(f"HTTP {status}: {text[:300]}")
                        delay = min(INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
                        await asyncio.sleep(delay)
                        continue

                    try:
                        return await resp.json(content_type=None)
                    except aiohttp.ContentTypeError:
                        # Статус 200, но тело не парсится как JSON - на практике это
                        # HTML-страница вместо ожидаемого ответа. У LOC это официально
                        # задокументированный побочный эффект перегрузки на их стороне
                        # ("HTML pages with CAPTCHAs even when operating below the rates
                        # listed above") - по сути тот же сигнал блокировки, что и 429,
                        # поэтому обрабатываем его так же (включая treat_429_as_exhaustion).
                        text_preview = (await resp.text())[:200]
                        if treat_429_as_exhaustion:
                            logging.warning(
                                "%s: получен не-JSON ответ (похоже на CAPTCHA/rate-limit "
                                "страницу вместо API-ответа) - помечаю сайт исчерпанным до "
                                "конца текущего запуска. Превью тела: %s",
                                site, text_preview,
                            )
                            ctx.exhausted_sites.add(site)
                            return None
                        logging.warning(
                            "%s: не-JSON ответ при статусе 200 (попытка %s/%s), похоже на "
                            "CAPTCHA/перегрузку. Превью: %s", site, attempt, MAX_RETRIES, text_preview,
                        )
                        last_error = RuntimeError(f"non-JSON 200 response: {text_preview}")
                        delay = min(INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
                        await asyncio.sleep(delay)
                        continue

            except FatalConfigError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                delay = min(INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
                logging.warning(
                    "%s: сетевая ошибка (попытка %s/%s): %s. Жду %.1fs.",
                    site, attempt, MAX_RETRIES, e, delay,
                )
                await asyncio.sleep(delay)
                continue

    logging.error("%s: запрос не удался после %s попыток: %s", site, MAX_RETRIES, last_error)
    return None


async def cached_search(
    ctx: Context, site: str, media_type: str, query: str,
    fetch_coro_factory: Callable[[], Awaitable[list]],
) -> list:
    """Single-flight кэш по (site, media_type, query): если запрос уже в процессе -
    ждём его же, а не дублируем (важно для узких лимитов вроде Pexels 200/час)."""
    key = (site, media_type, query)
    async with ctx.search_cache_lock:
        task = ctx.search_cache.get(key)
        if task is None:
            task = asyncio.ensure_future(fetch_coro_factory())
            ctx.search_cache[key] = task
    return await task


# ---------------------------------------------------------------------------
# Поиск по сайтам
# ---------------------------------------------------------------------------

async def search_pexels(ctx: Context, query: str, media_type: str) -> list[Candidate]:
    async def _do() -> list[Candidate]:
        if "pexels" in ctx.exhausted_sites:
            return []
        url = "https://api.pexels.com/videos/search" if media_type == "video" else "https://api.pexels.com/v1/search"
        headers = {"Authorization": ctx.pexels_api_key}
        params = {"query": query, "per_page": 15}
        data = await http_get_json(ctx, "pexels", url, headers=headers, params=params, treat_429_as_exhaustion=True)
        result: list[Candidate] = []
        if not data:
            return result
        if media_type == "video":
            for v in data.get("videos", []):
                preview = v.get("image")
                if not preview:
                    pics = v.get("video_pictures") or []
                    preview = pics[0]["picture"] if pics else None
                result.append(Candidate(
                    site="pexels", cand_id=str(v["id"]), text="",
                    license_ok=True, preview_url=preview, page_url=v.get("url"),
                ))
        else:
            for p in data.get("photos", []):
                src = p.get("src") or {}
                preview = src.get("medium") or src.get("small") or src.get("original")
                result.append(Candidate(
                    site="pexels", cand_id=str(p["id"]), text=p.get("alt") or "",
                    license_ok=True, preview_url=preview, page_url=p.get("url"),
                ))
        return result

    return await cached_search(ctx, "pexels", media_type, query, _do)


async def search_pixabay(ctx: Context, query: str, media_type: str) -> list[Candidate]:
    async def _do() -> list[Candidate]:
        if "pixabay" in ctx.exhausted_sites:
            return []
        url = "https://pixabay.com/api/videos/" if media_type == "video" else "https://pixabay.com/api/"
        params = {"key": ctx.pixabay_api_key, "q": query, "per_page": 20}
        data = await http_get_json(ctx, "pixabay", url, params=params, treat_429_as_exhaustion=True)
        result: list[Candidate] = []
        if not data:
            return result
        for hit in data.get("hits", []):
            page_url = hit.get("pageURL")
            tags = hit.get("tags", "")
            if media_type == "video":
                picture_id = hit.get("picture_id")
                preview = f"https://i.vimeocdn.com/video/{picture_id}_200x150.jpg" if picture_id else None
            else:
                preview = hit.get("previewURL") or hit.get("webformatURL")
            result.append(Candidate(
                site="pixabay", cand_id=str(hit["id"]), text=tags,
                license_ok=True, preview_url=preview, page_url=page_url,
            ))
        return result

    return await cached_search(ctx, "pixabay", media_type, query, _do)


async def search_wikimedia(ctx: Context, query: str, media_type: str) -> list[Candidate]:
    async def _do() -> list[Candidate]:
        search_query = f"{query} filetype:video" if media_type == "video" else query
        params = {
            "action": "query", "list": "search", "srsearch": search_query,
            "srnamespace": 6, "srlimit": 20, "format": "json",
        }
        data = await http_get_json(ctx, "wikimedia", "https://commons.wikimedia.org/w/api.php", params=params)
        if not data:
            return []
        hits = data.get("query", {}).get("search", [])
        video_ext_re = re.compile(r"\.(ogv|webm|mp4|mpg|mpeg)$", re.IGNORECASE)
        if media_type == "video":
            hits = [h for h in hits if video_ext_re.search(h.get("title", ""))]
        else:
            hits = [h for h in hits if not video_ext_re.search(h.get("title", ""))]
        if not hits:
            return []

        titles = [h["title"] for h in hits[:20]]
        snippet_by_title = {h["title"]: strip_html(h.get("snippet", "")) for h in hits}
        iiparams = {
            "action": "query", "prop": "imageinfo",
            "titles": "|".join(titles),
            "iiprop": "url|extmetadata",
            "iiurlwidth": 800,
            "format": "json",
        }
        info_data = await http_get_json(ctx, "wikimedia", "https://commons.wikimedia.org/w/api.php", params=iiparams)
        result: list[Candidate] = []
        if not info_data:
            return result
        pages = info_data.get("query", {}).get("pages", {}) or {}
        for _, page in pages.items():
            title = page.get("title")
            infos = page.get("imageinfo") or []
            if not title or not infos:
                continue
            info = infos[0]
            direct_url = info.get("url")
            thumb_url = info.get("thumburl")
            if not direct_url:
                continue
            if media_type == "video" and not thumb_url:
                # Не удалось получить превью для CLIP у этого конкретного файла -
                # дисквалифицируем именно его, не весь сайт (см. уточнение в докстринге).
                continue
            extm = info.get("extmetadata", {}) or {}
            license_short = (extm.get("LicenseShortName") or {}).get("value", "")
            description = strip_html((extm.get("ImageDescription") or {}).get("value", ""))
            text = " ".join(filter(None, [title, snippet_by_title.get(title, ""), description]))
            # page_url здесь - страница описания файла на Commons (index.php?...),
            # используется в т.ч. как Referer при скачивании превью - см. "Седьмое
            # уточнение" в докстринге модуля. Формируем её отдельно от direct_url
            # (прямая ссылка на сам файл на upload.wikimedia.org - её Referer'ом
            # ставить бессмысленно, это не HTML-страница).
            wiki_page_url = "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_")
            result.append(Candidate(
                site="wikimedia", cand_id=title, text=text,
                license_ok=wikimedia_license_ok(license_short),
                preview_url=thumb_url or direct_url, page_url=wiki_page_url,
            ))
        return result

    return await cached_search(ctx, "wikimedia", media_type, query, _do)


def _nasa_manifest_resolver(manifest_href: Optional[str], media_type: str):
    if not manifest_href:
        return None

    async def _resolve(ctx: Context) -> Optional[str]:
        files = await http_get_json(ctx, "nasa", manifest_href)
        if not files or not isinstance(files, list):
            return None
        if media_type == "video":
            candidates = [f for f in files if isinstance(f, str) and f.lower().endswith(".mp4")]
        else:
            candidates = [
                f for f in files
                if isinstance(f, str) and re.search(r"\.(jpg|jpeg|png|tif|tiff)$", f, re.IGNORECASE)
            ]
        if not candidates:
            return None

        def rank(f: str) -> int:
            fl = f.lower()
            if "~orig" in fl:
                return 3
            if "~large" in fl:
                return 2
            if "~medium" in fl:
                return 1
            return 0

        candidates.sort(key=rank, reverse=True)
        return candidates[0]

    return _resolve


async def search_nasa(ctx: Context, query: str, media_type: str) -> list[Candidate]:
    async def _do() -> list[Candidate]:
        params = {"q": query, "media_type": media_type}
        data = await http_get_json(ctx, "nasa", "https://images-api.nasa.gov/search", params=params)
        result: list[Candidate] = []
        if not data:
            return result
        items = (data.get("collection") or {}).get("items", []) or []
        for item in items:
            datas = item.get("data") or []
            if not datas:
                continue
            d0 = datas[0]
            links = item.get("links") or []
            preview = None
            for l in links:
                if l.get("rel") == "preview":
                    preview = l.get("href")
                    break
            if not preview and links:
                preview = links[0].get("href")
            text = " ".join(filter(None, [
                d0.get("title", ""), d0.get("description", ""),
                " ".join(d0.get("keywords") or []),
            ]))
            manifest_href = item.get("href")
            nasa_id = d0.get("nasa_id") or manifest_href
            if not nasa_id:
                continue
            result.append(Candidate(
                site="nasa", cand_id=nasa_id, text=text,
                license_ok=nasa_license_ok(d0), preview_url=preview, page_url=None,
                final_url_resolver=_nasa_manifest_resolver(manifest_href, media_type),
            ))
        return result

    return await cached_search(ctx, "nasa", media_type, query, _do)


async def search_loc(ctx: Context, query: str, media_type: str) -> list[Candidate]:
    async def _do() -> list[Candidate]:
        params: dict = {"q": query, "fo": "json", "c": 20}
        if media_type == "video":
            # Best-effort фильтр LOC по формату - на практике видео на LOC редки и
            # официальной чистой поддержки "только видео" в search API нет, поэтому
            # это не строгая гарантия, а сужение выдачи (подтверждено как некритично).
            params["fa"] = "partof:online video"
        # treat_429_as_exhaustion=True: у LOC превышение лимита JSON API (20/мин) даёт
        # блокировку IP на 1 час (см. докстринг модуля, "Четвёртое уточнение"), поэтому
        # ретраить 429 в рамках одного запуска CI бессмысленно - сразу помечаем сайт
        # исчерпанным, как pexels/pixabay при их часовых/дневных лимитах.
        data = await http_get_json(
            ctx, "loc", "https://www.loc.gov/search/", params=params,
            treat_429_as_exhaustion=True,
        )
        result: list[Candidate] = []
        if not data:
            return result
        for item in data.get("results", []) or []:
            item_id = item.get("id")
            if not item_id:
                continue
            title = item.get("title", "")
            desc = item.get("description")
            if isinstance(desc, list):
                desc = " ".join(str(x) for x in desc)
            text = " ".join(filter(None, [title, desc or ""]))
            images = item.get("image_url") or []
            preview = images[0] if images else None
            result.append(Candidate(
                site="loc", cand_id=item_id, text=text,
                license_ok=loc_license_ok(item), preview_url=preview, page_url=item_id,
            ))
        return result

    return await cached_search(ctx, "loc", media_type, query, _do)


SITE_SEARCH_FUNCS: dict = {
    "pexels": search_pexels,
    "pixabay": search_pixabay,
    "wikimedia": search_wikimedia,
    "nasa": search_nasa,
    "loc": search_loc,
}


# ---------------------------------------------------------------------------
# Превью + CLIP-скоринг + финализация URL
# ---------------------------------------------------------------------------

async def fetch_preview_bytes(ctx: Context, cand: Candidate) -> Optional[bytes]:
    if not cand.preview_url:
        return None

    # См. "Седьмое уточнение" в докстринге модуля: без Referer (часто и Origin) CDN
    # многих сайтов отдают 403/406 на прямой запрос к медиафайлу (защита от хотлинкинга),
    # даже когда User-Agent в порядке. Для wikimedia предпочитаем page_url конкретного
    # кандидата (страница файла на Commons) - он надёжнее общего домена, т.к. некоторые
    # CDN сверяют не только домен, но и правдоподобие самой страницы-источника.
    headers = dict(PREVIEW_HEADERS)
    referer = cand.page_url if (cand.site == "wikimedia" and cand.page_url) else PREVIEW_REFERERS.get(cand.site)
    if referer:
        headers["Referer"] = referer
        parsed = urlparse(referer)
        if parsed.scheme and parsed.netloc:
            headers["Origin"] = f"{parsed.scheme}://{parsed.netloc}"

    last_status: Optional[int] = None
    last_body_preview = ""
    for attempt in range(1, 3):
        try:
            async with ctx.session.get(
                cand.preview_url, headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status != 200:
                    last_status = resp.status
                    try:
                        last_body_preview = (await resp.text())[:200]
                    except Exception:
                        last_body_preview = "<не текст/не удалось прочитать тело>"
                    logging.debug(
                        "Превью %s/%s: HTTP %s при GET %s (попытка %s/2, Referer=%s). Тело: %s",
                        cand.site, cand.cand_id, resp.status, cand.preview_url,
                        attempt, referer, last_body_preview,
                    )
                    return None
                return await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.debug(
                "Превью %s/%s: сетевая ошибка (попытка %s/2) при GET %s: %s",
                cand.site, cand.cand_id, attempt, cand.preview_url, e,
            )
            await asyncio.sleep(1.0 * attempt)

    if last_status is not None:
        logging.debug(
            "Превью %s/%s: не удалось скачать после всех попыток, последний статус %s, тело: %s",
            cand.site, cand.cand_id, last_status, last_body_preview,
        )
    return None


async def score_candidates(
    ctx: Context, candidates: list[Candidate], query_text: str,
    seg_index: Optional[int] = None, stats_key: Optional[str] = None,
) -> list[Candidate]:
    key = stats_key or (candidates[0].site if candidates else "unknown")
    stats = ctx.site_stats.setdefault(key, SiteStats())

    scored: list[Candidate] = []
    for cand in candidates:
        preview_bytes = await fetch_preview_bytes(ctx, cand)
        if preview_bytes is None:
            # Превью недоступно - дисквалифицируем именно этого кандидата, идём дальше.
            stats.preview_missing_total += 1
            continue
        stats.sent_to_clip_total += 1
        async with ctx.clip_semaphore:
            try:
                sim = await ctx.clip.score(preview_bytes, query_text)
            except Exception as e:
                logging.debug("CLIP не смог оценить %s/%s: %s", cand.site, cand.cand_id, e)
                stats.clip_error_total += 1
                continue
        stats.record_score(sim)
        if sim >= SIM_MIN_THRESHOLD:
            cand.similarity = sim
            scored.append(cand)
    scored.sort(key=lambda c: c.similarity, reverse=True)

    if candidates and not scored:
        if stats.sent_to_clip_total == 0:
            logging.info(
                "Сегмент %s/%s: %s кандидатов, но ни для одного не скачалось превью "
                "(0 ушло в CLIP) - похоже, сайт блокирует запросы превью (см. DEBUG-лог "
                "с точным статусом/телом ответа выше), а не проблема со смыслом/CLIP.",
                seg_index, key, len(candidates),
            )
        else:
            logging.info(
                "Сегмент %s/%s: %s кандидатов ушло в CLIP, ни один не набрал >= %.2f "
                "(лучший скор в этом сегменте см. в общей сводке по сайту в конце лога).",
                seg_index, key, stats.sent_to_clip_total, SIM_MIN_THRESHOLD,
            )
    return scored


async def finalize_candidate(ctx: Context, cand: Candidate) -> Optional[str]:
    if cand.final_url_resolver is not None:
        try:
            return await cand.final_url_resolver(ctx)
        except Exception as e:
            logging.warning("Не удалось финализировать %s/%s: %s", cand.site, cand.cand_id, e)
            return None
    return cand.page_url


async def try_claim_pool(ctx: Context, pool: list[Candidate]) -> Optional[str]:
    for cand in pool:  # уже отсортирован по убыванию similarity
        key = (cand.site, cand.cand_id)
        async with ctx.used_files_lock:
            if key in ctx.used_files:
                continue
            ctx.used_files.add(key)  # бронируем сразу внутри лока, в момент выбора
        final_url = await finalize_candidate(ctx, cand)
        if final_url:
            return final_url
        # Финализация не удалась (например NASA-манифест не дал нужного файла) -
        # кандидат уже забронирован, но он всё равно непригоден никому - пробуем следующего.
    return None


async def fetch_and_filter(ctx: Context, site: str, seg: SegmentSpec) -> list[Candidate]:
    stats = ctx.site_stats.setdefault(site, SiteStats())
    stats.segments_attempted += 1

    raw = await SITE_SEARCH_FUNCS[site](ctx, seg.query, seg.type)
    stats.raw_total += len(raw)
    if not raw:
        logging.info(
            "Сегмент %s/%s: 0 сырых кандидатов по запросу %r - сайт ничего не вернул "
            "(проверьте сеть/сам API/лимиты для этого сайта).", seg.index, site, seg.query,
        )
        return []

    licensed = [c for c in raw if c.license_ok]
    stats.license_ok_total += len(licensed)
    if not licensed:
        logging.info(
            "Сегмент %s/%s: %s сырых кандидатов, но 0 прошло лицензионный фильтр.",
            seg.index, site, len(raw),
        )
        return []

    skip_keyword_filter = site == "pexels" and seg.type == "video"  # там нет текстовых полей
    if seg.is_entity and not skip_keyword_filter:
        before = len(licensed)
        licensed = [c for c in licensed if text_matches_keywords(c.text, seg.entity_keywords)]
        if not licensed:
            logging.info(
                "Сегмент %s/%s: %s кандидатов прошли лицензию, но 0 после фильтра сущностей "
                "%r - ключевые слова слишком узкие/не совпадают с текстом кандидатов.",
                seg.index, site, before, seg.entity_keywords,
            )
            return []
    stats.keyword_ok_total += len(licensed)
    return licensed


async def try_fallback(ctx: Context, seg: SegmentSpec) -> Optional[str]:
    pool: list[Candidate] = []
    for site in ("pexels", "pixabay"):
        if site in ctx.exhausted_sites:
            continue
        stats_key = f"{site}_fallback"
        stats = ctx.site_stats.setdefault(stats_key, SiteStats())
        stats.segments_attempted += 1
        raw = await SITE_SEARCH_FUNCS[site](ctx, seg.fallback_query, seg.type)
        stats.raw_total += len(raw)
        if not raw:
            continue
        top = raw[:CANDIDATES_PER_SITE]
        stats.license_ok_total += len(top)
        stats.keyword_ok_total += len(top)
        scored = await score_candidates(
            ctx, top, seg.fallback_query, seg_index=seg.index, stats_key=stats_key,
        )
        pool.extend(scored)
    if not pool:
        return None
    pool.sort(key=lambda c: c.similarity, reverse=True)
    return await try_claim_pool(ctx, pool)


async def process_segment_inner(ctx: Context, seg: SegmentSpec) -> Optional[str]:
    pool: list[Candidate] = []
    for site in seg.sites:
        if site in ctx.exhausted_sites:
            continue
        licensed = await fetch_and_filter(ctx, site, seg)
        if not licensed:
            continue  # сайт не дал ни одного кандидата после лицензии/ключевых слов
        top = licensed[:CANDIDATES_PER_SITE]
        scored = await score_candidates(ctx, top, seg.query, seg_index=seg.index, stats_key=site)
        if not scored:
            continue  # сайт дал пустой результат по CLIP-порогу (< SIM_MIN_THRESHOLD либо все превью недоступны)
        pool.extend(scored)
        pool.sort(key=lambda c: c.similarity, reverse=True)
        if scored[0].similarity >= SIM_ACCEPT_THRESHOLD:
            break  # ранний выход - дальше сайты не пробуем (см. допущение в докстринге)
        # иначе - similarity в [SIM_MIN_THRESHOLD, SIM_ACCEPT_THRESHOLD), пробуем следующий сайт

    if pool:
        url = await try_claim_pool(ctx, pool)
        if url:
            return url

    if seg.fallback_query:
        return await try_fallback(ctx, seg)

    return None


async def process_segment(ctx: Context, seg: SegmentSpec) -> tuple[int, Optional[str]]:
    async with ctx.global_semaphore:
        url = await process_segment_inner(ctx, seg)
    return seg.index, url


# ---------------------------------------------------------------------------
# Оркестрация запуска
# ---------------------------------------------------------------------------

async def run_search(ctx: Context, segments: list[SegmentSpec]) -> tuple[dict, list]:
    total = len(segments)
    done_count = 0
    results: dict = {}
    missing: list = []

    async def wrapped(seg: SegmentSpec):
        nonlocal done_count
        idx, url = await process_segment(ctx, seg)
        done_count += 1
        pct = done_count / total * 100
        logging.info(
            "Сегмент %s обработан (%s/%s, %.0f%%): %s",
            idx, done_count, total, pct, "найдено" if url else "НЕ найдено",
        )
        return idx, url

    tasks = [asyncio.ensure_future(wrapped(seg)) for seg in segments]
    try:
        for coro in asyncio.as_completed(tasks):
            idx, url = await coro
            if url:
                results[idx] = url
            else:
                missing.append(idx)
    except FatalConfigError:
        for t in tasks:
            t.cancel()
        raise

    return results, missing


def load_requests(path: str) -> list[SegmentSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError("requests.json пуст или имеет неверную структуру (ожидался объект-словарь)")

    specs: list[SegmentSpec] = []
    for k, v in data.items():
        try:
            idx = int(k)
            sites = list(v["sites"])
            if not sites:
                raise ValueError("пустой список sites")
            specs.append(SegmentSpec(
                index=idx,
                sites=sites,
                query=str(v["query"]),
                fallback_query=v.get("fallback_query"),
                type=str(v["type"]),
                is_entity=bool(v["is_entity"]),
                entity_keywords=list(v.get("entity_keywords") or []),
            ))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Сегмент {k!r} в requests.json имеет некорректную структуру: {e}") from e

    specs.sort(key=lambda s: s.index)
    return specs


async def amain(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.input):
        logging.error("Входной файл не найден: %s", args.input)
        return 1

    try:
        segments = load_requests(args.input)
    except (ValueError, json.JSONDecodeError) as e:
        logging.error("Ошибка чтения %s: %s", args.input, e)
        return 1

    logging.info("Загружено сегментов: %s", len(segments))
    logging.info(
        "Пороги CLIP similarity: SIM_MIN_THRESHOLD=%.3f, SIM_ACCEPT_THRESHOLD=%.3f "
        "(настраиваются через SEARCH_SIM_MIN_THRESHOLD / SEARCH_SIM_ACCEPT_THRESHOLD).",
        SIM_MIN_THRESHOLD, SIM_ACCEPT_THRESHOLD,
    )

    pexels_key = os.environ.get("PEXELS_API_KEY", "")
    pixabay_key = os.environ.get("PIXABAY_API_KEY", "")
    if not pexels_key or not pixabay_key:
        logging.error(
            "PEXELS_API_KEY и/или PIXABAY_API_KEY не заданы - без них поиск невозможен "
            "(в т.ч. fallback_query всегда идёт через pexels/pixabay)."
        )
        return 1

    if LOC_MIN_INTERVAL_SECONDS > 0:
        logging.info(
            "LOC rate-limiter активен: минимум %.2fs между последовательными запросами "
            "(%.1f запросов/мин; официальный лимит LOC - 20/мин с часовой блокировкой при "
            "превышении). Настраивается через SEARCH_LOC_MIN_INTERVAL_SECONDS. При первом "
            "же 429 (или CAPTCHA-ответе) сайт LOC помечается исчерпанным на весь остаток "
            "запуска - повторные попытки в рамках часовой блокировки не имеют смысла.",
            LOC_MIN_INTERVAL_SECONDS, 60.0 / LOC_MIN_INTERVAL_SECONDS,
        )

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(
        connector=connector, headers={"User-Agent": SESSION_USER_AGENT},
    ) as session:
        ctx = Context(
            session=session,
            pexels_api_key=pexels_key,
            pixabay_api_key=pixabay_key,
            site_semaphores={s: asyncio.Semaphore(v) for s, v in SEMAPHORE_DEFAULTS.items()},
            global_semaphore=asyncio.Semaphore(GLOBAL_SEGMENT_CONCURRENCY),
            clip_semaphore=asyncio.Semaphore(CLIP_CONCURRENCY),
            used_files_lock=asyncio.Lock(),
            rate_limiters={"loc": RateLimiter(LOC_MIN_INTERVAL_SECONDS)},
        )
        ctx.clip = ClipScorer(CLIP_MODEL_NAME, CLIP_PRETRAINED)

        try:
            results, missing = await run_search(ctx, segments)
        except FatalConfigError as e:
            logging.error("Структурная ошибка конфигурации: %s", e)
            log_site_stats_summary(ctx.site_stats)
            return 1

        log_site_stats_summary(ctx.site_stats)

    with open(args.links_output, "w", encoding="utf-8") as f:
        for idx in sorted(results):
            f.write(f"{idx}: {results[idx]}\n")

    with open(args.missing_output, "w", encoding="utf-8") as f:
        for idx in sorted(missing):
            f.write(f"{idx}\n")

    logging.info(
        "Готово: найдено %s из %s сегментов, не найдено %s. Результаты: %s, пропуски: %s",
        len(results), len(segments), len(missing), args.links_output, args.missing_output,
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Подбор медиа для сегментов requests.json")
    parser.add_argument("--input", default=os.environ.get("SEARCH_INPUT", DEFAULT_SEARCH_INPUT))
    parser.add_argument("--links-output", default=os.environ.get("SEARCH_LINKS_OUTPUT", DEFAULT_LINKS_OUTPUT))
    parser.add_argument("--missing-output", default=os.environ.get("SEARCH_MISSING_OUTPUT", DEFAULT_MISSING_OUTPUT))
    args = parser.parse_args()

    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
