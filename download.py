import os
import re
import json
import sys
import time
import threading
import urllib.parse
from urllib.parse import urlparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from curl_cffi import requests as cffi_requests

# Живой прогресс в CI-логе: без этого stdout буферизуется блоками (не построчно, т.к.
# он не привязан к терминалу в GitHub Actions), и все print() из скрипта, скачивающего
# десятки файлов последовательно, могут появиться в логе одним куском в конце шага или
# с большой задержкой - выглядит как "зависло", хотя скрипт всё это время реально
# работал. sys.stdout.reconfigure(line_buffering=True) переключает стандартный вывод на
# построчный флаш немедленно при импорте модуля (до первого print), не заставляя
# добавлять flush=True в каждый print() по всему файлу. PYTHONUNBUFFERED=1 (см.
# download.yml, шаг "Run download script with input") решает ту же задачу ещё раньше -
# полностью отключает буферизацию на уровне интерпретатора, ДО того как этот код вообще
# успевает выполниться - оставлено здесь как подстраховка, если скрипт когда-нибудь
# запустят без этой переменной окружения (например локально).
sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = "downloaded_media"
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")
COVERR_API_KEY = os.environ.get("COVERR_API_KEY", "")

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
}

# ---------------------------------------------------------------------------
# Учёт провалов скачивания (см. download_failed.txt) - раньше каждый [ОШИБКА]
# просто печатался в stdout CI-лога и терялся безвозвратно: чтобы узнать, что и
# почему не скачалось, приходилось руками читать весь лог шага целиком. Теперь
# каждый провал ещё и запоминается в FAILED_ITEMS через fail(), а в конце main()
# при непустом списке пишется download_failed.txt ("номер: причина" построчно) -
# его дальше workflow подмешивает в missing.txt и (только если он не пуст)
# прикладывает к релизу.
# ---------------------------------------------------------------------------

FAILED_ITEMS: list[str] = []
_failed_items_lock = threading.Lock()  # раунд 5: FAILED_ITEMS теперь пишут несколько
                                        # потоков одновременно (list.append сам по себе
                                        # атомарен под GIL, но лок оставлен явно - дешевле
                                        # и понятнее, чем полагаться на детали реализации
                                        # CPython, если код когда-нибудь перенесут).


def fail(number: int, message: str) -> None:
    print(f"[ОШИБКА] {message}")
    with _failed_items_lock:
        FAILED_ITEMS.append(f"{number}: {message}")


# ---------------------------------------------------------------------------
# loc.gov: rate-limiter + "исчерпан после первого 429" + ретраи на сетевые сбои.
#
# См. реальный прогон на 108 файлах: с сегмента 27 (первый 429 от loc.gov) и до
# самого конца ВСЕ последующие обращения к loc.gov (39 штук) проваливались тем же
# 429 подряд, без единого успеха - потому что loc.gov, по официальной документации
# (https://www.loc.gov/apis/json-and-yaml/working-within-limits/), при превышении
# лимита JSON API (20 запросов/мин) банит IP на ЦЕЛЫЙ ЧАС, а не на минуту. download.py
# качает файлы последовательно, без пауз между запросами, и на каждый LOC-айтем делает
# ДВА запроса (сначала JSON-метаданные, потом сам файл) - на батче из нескольких
# десятков LOC-ссылок лимит выбивается быстро, после чего скрипт продолжает впустую
# долбиться в стену до конца прогона. Тот же принцип (rate-limiter + "не ретраить
# 429, сразу считать сайт исчерпанным") уже применён в search.py - см. его докстринг,
# "Четвёртое уточнение".
# ---------------------------------------------------------------------------

LOC_MIN_INTERVAL_SECONDS = float(os.environ.get("DOWNLOAD_LOC_MIN_INTERVAL_SECONDS", 3.2))
# 3.2с (~18.75 запросов/мин, ~6.25% запас от официального потолка 20/мин) - тот же
# пересчёт и то же обоснование округления В БОЛЬШУЮ сторону (не ровно 3.158с=5% запаса,
# чтобы дрожание таймингов не утащило фактическую частоту выше 19/мин), что и у
# SEARCH_LOC_MIN_INTERVAL_SECONDS в search.py - см. его докстринг. Раньше здесь тоже
# стояло 5.0с/~40% запаса; тронуто по итогам того же реального прогона (см. обсуждение),
# применяется к _loc_wait_turn() ПЕРЕД каждым запросом (включая ретраи внутри _loc_get),
# поэтому ретраи на сетевые сбои/таймауты по-прежнему сериализуются этим же интервалом,
# просто сам интервал теперь короче.
LOC_RETRIES = int(os.environ.get("DOWNLOAD_LOC_RETRIES", 3))

# ---------------------------------------------------------------------------
# Раунд 5 - конкурентность для НЕ-LOC скачиваний.
#
# По реальному прогону на 126 файлах: LOC (28/125 файлов) - avg 10.74с между
# скачиваниями (rate-limiter 3.2с + сама медленная сеть LOC), а ВСЁ ОСТАЛЬНОЕ
# (97/125: wikimedia avg 2.16с, pexels/pixabay avg 0.43с, видео через yt-dlp
# avg 0.80с) шло СТРОГО последовательно, хотя каждый сайт по отдельности явно
# выдерживает параллельные запросы. Раньше это был чистый sequential-цикл без
# единого потока/корутины - LOC был не единственной причиной долгого прогона,
# просто маскировал остальное.
#
# Решение: LOC остаётся полностью последовательным (не трогаем его логику
# вообще - она и так работает и завязана на глобальный rate-limiter), а всё
# остальное разбирается пулом потоков ThreadPoolExecutor, который запускается
# ОДНОВРЕМЕННО с последовательным циклом по LOC (не до и не после), чтобы время
# LOC и время всего остального перекрывались, а не суммировались - см. main().
#
# Внутри пула нужны ДВА отдельных thread-safe ограничителя, иначе параллелизм
# сам создаст новые проблемы:
#   1) Wikimedia - раньше здесь был точечный time.sleep(1.5) ПЕРЕД каждым
#      запросом ("чтобы сервера Викимедии не блочили по 429"), рассчитанный на
#      строго последовательный вызов. При параллельных потоках он ничего не
#      гарантирует: N потоков проснутся почти одновременно и всё равно ударят
#      в Wikimedia все разом. Заменён на WikimediaRateLimiter - тот же принцип,
#      что и RateLimiter в search.py (минимальный интервал между СТАРТАМИ
#      последовательных запросов), но на threading.Lock вместо asyncio.Lock.
#   2) Pexels/Pixabay - лимит у обоих ПО API-КЛЮЧУ, а не по IP/раннеру (в
#      отличие от LOC). Общий ThreadPoolExecutor по умолчанию даёт им такую же
#      конкурентность, как всем остальным (DOWNLOAD_CONCURRENCY потоков) - это
#      бьёт по одному и тому же ключу все сразу. Добавлен отдельный, более
#      узкий Semaphore именно для pexels+pixabay (не для видео/wikimedia/loc),
#      чтобы не спалить их лимит запросов даже при большом общем пуле потоков.
# ---------------------------------------------------------------------------

DOWNLOAD_CONCURRENCY = int(os.environ.get("DOWNLOAD_CONCURRENCY", 8))
WIKIMEDIA_MIN_INTERVAL_SECONDS = float(os.environ.get("DOWNLOAD_WIKIMEDIA_MIN_INTERVAL_SECONDS", 1.5))
PEXELS_PIXABAY_CONCURRENCY = int(os.environ.get("DOWNLOAD_PEXELS_PIXABAY_CONCURRENCY", 3))


class ThreadRateLimiter:
    """Thread-safe версия RateLimiter из search.py (там - на asyncio.Lock, здесь -
    на threading.Lock, т.к. download.py синхронный и качает файлы через
    ThreadPoolExecutor, а не asyncio). Гарантирует минимальный интервал между
    НАЧАЛОМ двух последовательных запросов, общий на все потоки сразу."""

    def __init__(self, min_interval_seconds: float):
        self.min_interval = max(0.0, min_interval_seconds)
        self._lock = threading.Lock()
        self._last_start_ts = 0.0

    def wait_turn(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last_start_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_start_ts = time.monotonic()


_wikimedia_rate_limiter = ThreadRateLimiter(WIKIMEDIA_MIN_INTERVAL_SECONDS)
_pexels_pixabay_semaphore = threading.Semaphore(PEXELS_PIXABAY_CONCURRENCY)


def _pexels_pixabay_throttled(func):
    """Декоратор вместо ручного 'with _pexels_pixabay_semaphore:' внутри каждой из
    четырёх функций download_pexels_*/download_pixabay_* - не нужно переотступать
    тело функции, ограничение вешается снаружи."""
    def wrapper(*args, **kwargs):
        with _pexels_pixabay_semaphore:
            return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

_loc_last_request_ts = 0.0
_loc_exhausted = False


class LocExhaustedError(RuntimeError):
    """LOC уже забанил нас в этом запуске (был 429) - дальнейшие попытки в рамках
    текущего прогона гарантированно провалятся тем же способом, ретраить бессмысленно."""


def _loc_wait_turn() -> None:
    global _loc_last_request_ts
    now = time.monotonic()
    wait = LOC_MIN_INTERVAL_SECONDS - (now - _loc_last_request_ts)
    if wait > 0:
        time.sleep(wait)
    _loc_last_request_ts = time.monotonic()


def _loc_get(url: str, timeout: int = 30, **kwargs):
    """Обёртка над cffi_requests.get специально для loc.gov: применяет rate-limiter
    перед КАЖДЫМ запросом (метаданные и сам файл считаются в один и тот же бюджет -
    оба домена loc.gov, дешевле перестраховаться, чем гадать про раздельные лимиты),
    ретраит транзиентные сетевые ошибки/таймауты, и при первом же HTTP 429 сразу
    помечает LOC исчерпанным до конца запуска - все следующие вызовы после этого
    момента падают мгновенно с LocExhaustedError, не тратя лишних запросов и времени
    сборки на заведомо бесполезные попытки."""
    global _loc_exhausted
    if _loc_exhausted:
        raise LocExhaustedError("LOC уже исчерпан в этом запуске (был 429 ранее)")

    last_exc: Exception | None = None
    for attempt in range(1, LOC_RETRIES + 1):
        _loc_wait_turn()
        try:
            resp = cffi_requests.get(url, impersonate="chrome", timeout=timeout, **kwargs)
        except Exception as e:
            last_exc = e
            time.sleep(2 * attempt)
            continue

        if resp.status_code == 429:
            _loc_exhausted = True
            raise LocExhaustedError("HTTP 429 - LOC помечен исчерпанным до конца текущего запуска")

        resp.raise_for_status()
        return resp

    raise last_exc or RuntimeError("не удалось выполнить запрос к loc.gov")


def parse_and_download_links(env_name: str) -> None:
    raw_text = os.environ.get(env_name, "")
    if not raw_text:
        print("Список ссылок пуст. Нечего скачивать.")
        return

    print("Начинаю разбор и скачивание файлов...")
    matches = re.findall(r'(\d+)\s*:\s*(https?://[^\s]+)', raw_text)

    if not matches:
        print("[ОШИБКА] Не удалось распознать ссылки формата 'номер: ссылка'")
        return

    print(f"Найдено ссылок для скачивания: {len(matches)}")

    # Раунд 5: LOC остаётся полностью последовательным (не трогаем его rate-limiter
    # логику), всё остальное идёт через ThreadPoolExecutor - см. подробное обоснование
    # в блоке комментариев "Раунд 5" в начале файла. Важно: LOC-цикл ниже запускается
    # СРАЗУ ПОСЛЕ submit() пула, а не после его завершения - это даёт перекрытие по
    # времени (пока LOC ждёт свои 3.2с между запросами, пул параллельно докачивает
    # всё остальное в фоне), а не сложение времени LOC + времени всего остального.
    loc_items = []
    other_items = []
    for num_str, url in matches:
        number = int(num_str)
        if "loc.gov" in url.lower():
            loc_items.append((number, url))
        else:
            other_items.append((number, url))

    print(
        f"[ИНФО] {len(other_items)} файлов пойдут параллельно (до {DOWNLOAD_CONCURRENCY} "
        f"потоков, pexels+pixabay внутри этого пула дополнительно ограничены "
        f"{PEXELS_PIXABAY_CONCURRENCY} потоками - общий ключ), {len(loc_items)} файлов "
        f"LOC - строго последовательно, тем же rate-limiter'ом, что и раньше, запущены "
        f"ОДНОВРЕМЕННО с пулом (не после него)."
    )

    with ThreadPoolExecutor(max_workers=DOWNLOAD_CONCURRENCY) as executor:
        futures = [executor.submit(download_media_item, number, url) for number, url in other_items]

        for number, url in loc_items:
            download_media_item(number, url)

        for future in as_completed(futures):
            future.result()  # пробрасываем неожиданные исключения - download_media_item
                              # сам ловит и логирует всё ожидаемое через fail()


def extract_id(url: str) -> str:
    match = re.search(r"(\d+)/?$", url.strip().rstrip("/"))
    return match.group(1) if match else ""


def extract_og_media_url(html: str) -> str:
    """Достаём прямую ссылку на фото/видео из og:image или og:video meta-тегов страницы."""
    match = re.search(
        r'<meta[^>]+property=["\']og:(?:image|video)(?::secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE
    )
    if match:
        return match.group(1)
    match = re.search(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:(?:image|video)(?::secure_url)?["\']',
        html, re.IGNORECASE
    )
    return match.group(1) if match else ""


def looks_like_html(content: bytes) -> bool:
    """Проверяем, не скачали ли мы случайно HTML-страницу вместо медиафайла."""
    snippet = content[:512].lstrip().lower()
    return snippet.startswith(b"<!doctype") or snippet.startswith(b"<html") or b"<head>" in snippet[:1024]


def download_media_item(number: int, url: str) -> None:
    url = url.strip()
    url_lower = url.lower()

    # 1. PEXELS ВИДЕО
    if "pexels.com" in url_lower and "/video/" in url_lower:
        video_id = extract_id(url)
        if video_id:
            download_pexels_video(number, video_id)
        else:
            fail(number, f"Pexels видео {number}: не удалось извлечь id из ссылки {url}")
        return

    # 2. PEXELS ФОТО
    if "pexels.com" in url_lower and "/photo/" in url_lower:
        photo_id = extract_id(url)
        if photo_id and PEXELS_API_KEY:
            download_pexels_photo(number, photo_id)
        else:
            fail(number, f"Pexels фото {number}: нет id ({photo_id!r}) или не задан PEXELS_API_KEY")
        return

    # 3. PIXABAY (ФОТО И ВИДЕО ЧЕРЕЗ ОФИЦИАЛЬНЫЙ API)
    if "pixabay.com" in url_lower:
        item_id = extract_id(url)
        if item_id and PIXABAY_API_KEY:
            if "/videos/" in url_lower:
                download_pixabay_video(number, item_id)
            else:
                download_pixabay_photo(number, item_id)
            return
        fail(number, f"Pixabay {number}: нет id ({item_id!r}) или не задан PIXABAY_API_KEY")
        return

    # 4. COVERR ВИДЕО
    if "coverr.co" in url_lower:
        download_coverr_video(number, url)
        return

    # 5. WIKIMEDIA COMMONS (wikimedia.org / wikipedia.org)
    if "wikimedia.org" in url_lower or "wikipedia.org" in url_lower:
        download_wikimedia_commons(number, url)
        return

    # 6. LIBRARY OF CONGRESS (loc.gov)
    if "loc.gov" in url_lower:
        download_loc_gov(number, url)
        return

    # 7. ПРЯМЫЕ ССЫЛКИ НА ВИДЕОФАЙЛЫ + СТРАНИЦЫ MIXKIT -> ЧЕРЕЗ yt-dlp
    if url_lower.endswith((".mp4", ".mov", ".avi")) or "mixkit.co" in url_lower:
        download_via_ytdlp(number, url)
        return

    # 8. ВСЁ ОСТАЛЬНОЕ - СКАЧИВАНИЕ ЧЕРЕЗ curl_cffi С ИМИТАЦИЕЙ БРАУЗЕРА
    download_direct_via_cffi(number, url)


def download_wikimedia_commons(number: int, url: str) -> None:
    """Скачивание оригинального файла с Wikimedia Commons с паузами от лимита 429."""
    # Раунд 5: раньше был просто time.sleep(1.5) - работало только при строго
    # последовательном вызове. Теперь несколько потоков могут звать эту функцию
    # одновременно, поэтому пауза стала общим thread-safe rate-limiter'ом (см.
    # блок "Раунд 5" в начале файла) - интервал тот же (1.5с), но теперь он
    # действительно минимальный интервал МЕЖДУ запросами всех потоков вместе,
    # а не просто пауза перед каждым отдельным вызовом.
    _wikimedia_rate_limiter.wait_turn()

    try:
        path = urlparse(url).path
        file_part = path.split("/wiki/")[-1] if "/wiki/" in path else path.split("/")[-1]
        file_title = urllib.parse.unquote(file_part)

        if not file_title.lower().startswith(("file:", "файл:")):
            file_title = "File:" + file_title

        api_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(file_title)}&prop=imageinfo&iiprop=url&format=json"

        data = None
        for attempt in range(3):
            resp = cffi_requests.get(api_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=20)
            if resp.status_code == 429:
                print(f"[ИНФО] Wikimedia 429 limit, пауза {4 * (attempt + 1)} сек для {number}...")
                time.sleep(4 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            break

        if not data:
            fail(number, f"Wikimedia {number}: лимит запросов 429 не сбросился")
            return

        pages = data.get("query", {}).get("pages", {})
        direct_url = None
        for _, page_data in pages.items():
            imageinfo = page_data.get("imageinfo", [])
            if imageinfo and "url" in imageinfo[0]:
                direct_url = imageinfo[0]["url"]
                break

        if not direct_url:
            page_resp = cffi_requests.get(url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=30)
            page_resp.raise_for_status()
            direct_url = extract_og_media_url(page_resp.text)

        if not direct_url:
            fail(number, f"Wikimedia {number}: не удалось извлечь ссылку ({url})")
            return

        # Скачивание файла с повторами
        content = None
        img_resp = None
        for attempt in range(3):
            img_resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=60)
            if img_resp.status_code == 429:
                time.sleep(4 * (attempt + 1))
                continue
            img_resp.raise_for_status()
            content = img_resp.content
            break

        if not content or looks_like_html(content):
            fail(number, f"Wikimedia {number}: не удалось скачать файл изображения")
            return

        ext = guess_extension(direct_url, img_resp.headers.get("Content-Type", ""))
        filepath = os.path.join(OUTPUT_DIR, f"{number}{ext}")
        with open(filepath, "wb") as f:
            f.write(content)
        print(f"[OK] WIKIMEDIA {number} ({number}{ext}) успешно скачано")
    except Exception as e:
        fail(number, f"Wikimedia ошибка {number}: {e}")


def download_direct_via_cffi(number: int, url: str) -> None:
    try:
        resp = cffi_requests.get(url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=30)
        resp.raise_for_status()
        content = resp.content

        if looks_like_html(content):
            media_url = extract_og_media_url(resp.text)
            if media_url:
                media_resp = cffi_requests.get(media_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=60)
                media_resp.raise_for_status()
                media_content = media_resp.content
                if not looks_like_html(media_content):
                    ext = guess_extension(media_url, media_resp.headers.get("Content-Type", ""))
                    filename = f"{number}{ext}"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    with open(filepath, "wb") as f:
                        f.write(media_content)
                    print(f"[OK] ФАЙЛ {number} ({filename}) извлечён из страницы")
                    return

            fail(number, f"{number}: страница не содержит медиафайла ({url})")
            return

        content_type = resp.headers.get("Content-Type", "")
        ext = guess_extension(url, content_type)
        filename = f"{number}{ext}"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(content)
        print(f"[OK] ФАЙЛ {number} ({filename}) успешно скачан")
    except Exception as e:
        fail(number, f"Не удалось скачать {number}: {e}")


@_pexels_pixabay_throttled
def download_pixabay_photo(number: int, photo_id: str) -> None:
    api_url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&id={photo_id}"
    try:
        resp = cffi_requests.get(api_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=20)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])
        if hits:
            direct_url = hits[0].get("largeImageURL") or hits[0].get("imageURL")
            if direct_url:
                img_resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=30)
                img_resp.raise_for_status()
                filepath = os.path.join(OUTPUT_DIR, f"{number}.jpg")
                with open(filepath, "wb") as f:
                    f.write(img_resp.content)
                print(f"[OK] ФОТО PIXABAY {number} (id={photo_id}) успешно скачано")
                return
        fail(number, f"Не удалось получить фото Pixabay {number}")
    except Exception as e:
        fail(number, f"Pixabay API ошибка {number}: {e}")


@_pexels_pixabay_throttled
def download_pixabay_video(number: int, video_id: str) -> None:
    api_url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&id={video_id}"
    try:
        resp = cffi_requests.get(api_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=20)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])
        if hits:
            videos = hits[0].get("videos", {})
            best_video = videos.get("large") or videos.get("medium") or videos.get("small")
            if best_video and "url" in best_video:
                direct_url = best_video["url"]
                vid_resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=60)
                vid_resp.raise_for_status()
                filepath = os.path.join(OUTPUT_DIR, f"{number}.mp4")
                with open(filepath, "wb") as f:
                    f.write(vid_resp.content)
                print(f"[OK] ВИДЕО PIXABAY {number} (id={video_id}) успешно скачано")
                return
        fail(number, f"Не удалось получить видео Pixabay {number}")
    except Exception as e:
        fail(number, f"Pixabay API ошибка {number}: {e}")


@_pexels_pixabay_throttled
def download_pexels_photo(number: int, photo_id: str) -> None:
    api_url = f"https://api.pexels.com/v1/photos/{photo_id}"
    try:
        resp = cffi_requests.get(api_url, headers={**BROWSER_HEADERS, "Authorization": PEXELS_API_KEY},
                                  impersonate="chrome", timeout=20)
        resp.raise_for_status()
        data = resp.json()
        direct_url = data.get("src", {}).get("original") or data.get("src", {}).get("large")
        if direct_url:
            img_resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=30)
            img_resp.raise_for_status()
            filepath = os.path.join(OUTPUT_DIR, f"{number}.jpg")
            with open(filepath, "wb") as f:
                f.write(img_resp.content)
            print(f"[OK] ФОТО PEXELS {number} (id={photo_id}) успешно скачано")
        else:
            # Раньше при пустом direct_url функция молча ничего не делала - провал
            # терялся без единого слова в логе. Теперь хотя бы попадает в отчёт.
            fail(number, f"Pexels фото {number} (id={photo_id}): в ответе API нет src.original/large")
    except Exception as e:
        fail(number, f"Ошибка скачивания фото Pexels {number}: {e}")


@_pexels_pixabay_throttled
def download_pexels_video(number: int, video_id: str) -> None:
    if not PEXELS_API_KEY:
        fail(number, f"Видео {number}: нет PEXELS_API_KEY.")
        return

    api_url = f"https://api.pexels.com/v1/videos/videos/{video_id}"
    try:
        resp = cffi_requests.get(api_url, headers={**BROWSER_HEADERS, "Authorization": PEXELS_API_KEY},
                                  impersonate="chrome", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        fail(number, f"Pexels API ошибка {number}: {e}")
        return

    video_files = data.get("video_files", [])
    if not video_files:
        fail(number, f"Pexels {number}: нет video_files в ответе API")
        return

    mp4_files = [f for f in video_files if f.get("file_type") == "video/mp4"]
    candidates = mp4_files if mp4_files else video_files
    best = max(candidates, key=lambda f: f.get("width") or 0)
    direct_url = best["link"]

    try:
        resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=60)
        resp.raise_for_status()
        filepath = os.path.join(OUTPUT_DIR, f"{number}.mp4")
        with open(filepath, "wb") as f:
            f.write(resp.content)
        print(f"[OK] ВИДЕО PEXELS {number} (id={video_id}) успешно скачано")
    except Exception as e:
        fail(number, f"Не удалось скачать файл видео {number}: {e}")


def download_coverr_video(number: int, url: str) -> None:
    try:
        page_resp = cffi_requests.get(url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=30)
        page_resp.raise_for_status()
        html = page_resp.text

        direct_url = extract_og_media_url(html)
        if not direct_url:
            fail(number, f"Coverr {number}: не нашли og:video на странице {url}")
            return

        vid_resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=60)
        vid_resp.raise_for_status()
        content = vid_resp.content

        if looks_like_html(content):
            fail(number, f"Coverr {number}: по ссылке пришла HTML-страница, а не видео")
            return

        filepath = os.path.join(OUTPUT_DIR, f"{number}.mp4")
        with open(filepath, "wb") as f:
            f.write(content)
        print(f"[OK] ВИДЕО COVERR {number} успешно скачано ({direct_url})")
    except Exception as e:
        fail(number, f"Coverr ошибка {number}: {e}")


def download_loc_gov(number: int, url: str) -> None:
    parsed = urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    query["fo"] = ["json"]
    json_url = parsed._replace(query=urllib.parse.urlencode(query, doseq=True)).geturl()

    try:
        resp = _loc_get(json_url, headers=BROWSER_HEADERS)
        data = resp.json()

        direct_url = None
        resource = data.get("resource", {}) or {}
        if isinstance(resource.get("files"), list):
            for file_group in resource["files"]:
                if isinstance(file_group, list) and file_group:
                    candidate = max(file_group, key=lambda f: f.get("height", 0) if isinstance(f, dict) else 0)
                    if isinstance(candidate, dict) and candidate.get("url"):
                        direct_url = candidate["url"]
                        break
        if not direct_url:
            item = data.get("item", {}) or {}
            image_url = item.get("image_url")
            if isinstance(image_url, list) and image_url:
                direct_url = image_url[-1]
            elif isinstance(image_url, str):
                direct_url = image_url

        if not direct_url:
            fail(number, f"loc.gov {number}: не удалось найти прямую ссылку на файл")
            return

        file_resp = _loc_get(direct_url, headers=BROWSER_HEADERS, timeout=60)
        content = file_resp.content

        if looks_like_html(content):
            fail(number, f"loc.gov {number}: похоже, скачалась HTML-страница")
            return

        ext = guess_extension(direct_url, file_resp.headers.get("Content-Type", ""))
        filepath = os.path.join(OUTPUT_DIR, f"{number}{ext}")
        with open(filepath, "wb") as f:
            f.write(content)
        print(f"[OK] LOC.GOV {number} успешно скачан")
    except LocExhaustedError as e:
        fail(number, f"loc.gov ошибка {number}: {e}")
    except Exception as e:
        fail(number, f"loc.gov ошибка {number}: {e}")


def download_via_ytdlp(number: int, url: str) -> None:
    try:
        cmd = [
            "yt-dlp",
            "--extractor-args", "generic:impersonate=chrome",
            "-o", f"{OUTPUT_DIR}/{number}.%(ext)s",
            "--format", "mp4/best",
            "--no-check-certificate",
            "--retries", "3",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] ВИДЕО {number} успешно скачано через yt-dlp")
        else:
            fail(number, f"Ошибка yt-dlp {number}: {result.stderr}")
    except Exception as ytdl_err:
        fail(number, f"Не удалось запустить yt-dlp для {number}: {ytdl_err}")


def guess_extension(url: str, content_type: str = "") -> str:
    path = urlparse(url).path
    match = re.search(r"\.(jpg|jpeg|png|webp|gif|mp4|mov)$", path, re.IGNORECASE)
    if match:
        return "." + match.group(1).lower()
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "png" in content_type:
        return ".png"
    if "mp4" in content_type:
        return ".mp4"
    return ".jpg"


def write_failed_report(path: str = "download_failed.txt") -> None:
    """Пишет download_failed.txt ТОЛЬКО если реально что-то не скачалось (FAILED_ITEMS
    непустой) - если всё скачалось успешно, файл вообще не создаётся, чтобы workflow
    мог проверять его существование/непустоту (`[ -s download_failed.txt ]`) и не
    прикладывать к релизу пустой шум."""
    if not FAILED_ITEMS:
        return
    with open(path, "w", encoding="utf-8") as f:
        for line in FAILED_ITEMS:
            f.write(line + "\n")
    print(f"[ИНФО] Не удалось скачать {len(FAILED_ITEMS)} файлов - записано в {path}")


def main():
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parse_and_download_links("INPUT_LINKS")
    write_failed_report()


if __name__ == "__main__":
    main()
