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
    SEARCH_GLOBAL_CONCURRENCY  - сколько сегментов обрабатывать параллельно (умолч. 40)
    SEARCH_CLIP_CONCURRENCY   - сколько CLIP-инференсов одновременно (умолч. 2, CPU-bound)
    SEARCH_CANDIDATES_PER_SITE - сколько топ-кандидатов с сайта пускать под CLIP (умолч. 5)
    SEARCH_CLIP_MODEL / SEARCH_CLIP_PRETRAINED - модель open_clip (умолч. ViT-B-32 / openai)

Возвращаемые коды:
    0 - links.txt и missing.txt успешно записаны (даже если часть/все сегменты в missing)
    1 - структурная ошибка (невалидный API-ключ, битый requests.json и т.п.)

Зависимости:
    pip install aiohttp pillow open_clip_torch
    (torch лучше ставить отдельно, CPU-only wheel, см. комментарий в конце файла)

ВАЖНОЕ ДОПУЩЕНИЕ по интерпретации п.6-7 исходного ТЗ (спецификация была неоднозначна
в этом месте): "лучший кандидат сайта >= 0.85" останавливает перебор сайтов, но выбор
и дедуп-бронирование всегда идёт по НАКОПЛЕННОМУ пулу кандидатов (>=0.5) со всех уже
пройденных сайтов, отсортированному по убыванию similarity - а не только по кандидатам
текущего (последнего) сайта. Это единственное прочтение, совместимое одновременно с
"не пробовать остальные сайты" (п.6) и "среди кандидатов ПО ВСЕМ ПЕРЕБРАННЫМ САЙТАМ"
(п.7). Если имелось в виду не так - легко поменять в try_claim_pool/process_segment.

Второе допущение: правило "нет превью для CLIP -> дисквалифицировать именно этого
кандидата, не весь сайт" (изначально уточнено для видео на Wikimedia) применено ко ВСЕМ
сайтам единообразно в fetch_preview_bytes/score_candidates - это строго безопаснее и
не создаёт особых случаев.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import io
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

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

SIM_ACCEPT_THRESHOLD = 0.85
SIM_MIN_THRESHOLD = 0.5

CANDIDATES_PER_SITE = int(os.environ.get("SEARCH_CANDIDATES_PER_SITE", 5))

CLIP_MODEL_NAME = os.environ.get("SEARCH_CLIP_MODEL", "ViT-B-32")
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

PREVIEW_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
}

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


def text_matches_keywords(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    t = (text or "").lower()
    return any(kw.lower() in t for kw in keywords if kw)


class FatalConfigError(RuntimeError):
    """Структурная ошибка конфигурации (невалидный API-ключ и т.п.) - не ретраится,
    приводит к остановке всего скрипта с кодом 1."""


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

    last_error: Optional[BaseException] = None
    for attempt in range(1, MAX_RETRIES + 1):
        async with ctx.site_semaphores[site]:
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

                    if status in (401, 403) and site in ("pexels", "pixabay"):
                        text = await resp.text()
                        raise FatalConfigError(
                            f"{site}: HTTP {status} - похоже на невалидный API-ключ. "
                            f"Тело ответа: {text[:300]}"
                        )

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

                    return await resp.json(content_type=None)

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
            result.append(Candidate(
                site="wikimedia", cand_id=title, text=text,
                license_ok=wikimedia_license_ok(license_short),
                preview_url=thumb_url or direct_url, page_url=direct_url,
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
        data = await http_get_json(ctx, "loc", "https://www.loc.gov/search/", params=params)
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
    for attempt in range(1, 3):
        try:
            async with ctx.session.get(
                cand.preview_url, headers=PREVIEW_HEADERS,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status != 200:
                    return None
                return await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(1.0 * attempt)
    return None


async def score_candidates(ctx: Context, candidates: list[Candidate], query_text: str) -> list[Candidate]:
    scored: list[Candidate] = []
    for cand in candidates:
        preview_bytes = await fetch_preview_bytes(ctx, cand)
        if preview_bytes is None:
            # Превью недоступно - дисквалифицируем именно этого кандидата, идём дальше.
            continue
        async with ctx.clip_semaphore:
            try:
                sim = await ctx.clip.score(preview_bytes, query_text)
            except Exception as e:
                logging.debug("CLIP не смог оценить %s/%s: %s", cand.site, cand.cand_id, e)
                continue
        if sim >= SIM_MIN_THRESHOLD:
            cand.similarity = sim
            scored.append(cand)
    scored.sort(key=lambda c: c.similarity, reverse=True)
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
    raw = await SITE_SEARCH_FUNCS[site](ctx, seg.query, seg.type)
    if not raw:
        return []
    licensed = [c for c in raw if c.license_ok]
    if not licensed:
        return []
    skip_keyword_filter = site == "pexels" and seg.type == "video"  # там нет текстовых полей
    if seg.is_entity and not skip_keyword_filter:
        licensed = [c for c in licensed if text_matches_keywords(c.text, seg.entity_keywords)]
    return licensed


async def try_fallback(ctx: Context, seg: SegmentSpec) -> Optional[str]:
    pool: list[Candidate] = []
    for site in ("pexels", "pixabay"):
        if site in ctx.exhausted_sites:
            continue
        raw = await SITE_SEARCH_FUNCS[site](ctx, seg.fallback_query, seg.type)
        if not raw:
            continue
        top = raw[:CANDIDATES_PER_SITE]
        scored = await score_candidates(ctx, top, seg.fallback_query)
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
        scored = await score_candidates(ctx, top, seg.query)
        if not scored:
            continue  # сайт дал пустой результат по CLIP-порогу (< 0.5 либо все превью недоступны)
        pool.extend(scored)
        pool.sort(key=lambda c: c.similarity, reverse=True)
        if scored[0].similarity >= SIM_ACCEPT_THRESHOLD:
            break  # ранний выход - дальше сайты не пробуем (см. допущение в докстринге)
        # иначе - similarity в [0.5, 0.85), пробуем следующий сайт из списка

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

    pexels_key = os.environ.get("PEXELS_API_KEY", "")
    pixabay_key = os.environ.get("PIXABAY_API_KEY", "")
    if not pexels_key or not pixabay_key:
        logging.error(
            "PEXELS_API_KEY и/или PIXABAY_API_KEY не заданы - без них поиск невозможен "
            "(в т.ч. fallback_query всегда идёт через pexels/pixabay)."
        )
        return 1

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        ctx = Context(
            session=session,
            pexels_api_key=pexels_key,
            pixabay_api_key=pixabay_key,
            site_semaphores={s: asyncio.Semaphore(v) for s, v in SEMAPHORE_DEFAULTS.items()},
            global_semaphore=asyncio.Semaphore(GLOBAL_SEGMENT_CONCURRENCY),
            clip_semaphore=asyncio.Semaphore(CLIP_CONCURRENCY),
            used_files_lock=asyncio.Lock(),
        )
        ctx.clip = ClipScorer(CLIP_MODEL_NAME, CLIP_PRETRAINED)

        try:
            results, missing = await run_search(ctx, segments)
        except FatalConfigError as e:
            logging.error("Структурная ошибка конфигурации: %s", e)
            return 1

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
