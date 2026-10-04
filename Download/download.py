import os
import re
import glob
import json
import sys
import random
import time
import threading
import urllib.parse
from urllib.parse import urlparse
import subprocess
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from curl_cffi import requests as cffi_requests
from media_formats import (
    ext_from_name, is_allowed_ext, kind_of_ext, sniff_format, describe_rejected, FormatRejectStats,
)

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
FORMAT_REJECTS = FormatRejectStats()
# Расширения в URL, которым мы "верим" как заявке формата: если оно есть и не согласуется
# с содержимым (sniff_format) - отказ.
_KNOWN_HINT_EXTS = frozenset({"jpg", "jpeg", "png", "gif", "webp", "tif", "tiff", "pdf",
                              "mp4", "mov", "avi", "webm", "ogv", "mpg", "mpeg", "svg"})
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")
COVERR_API_KEY = os.environ.get("COVERR_API_KEY", "")

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
}

# ---------------------------------------------------------------------------
# Wikimedia — отдельный User-Agent, а НЕ общий BROWSER_HEADERS.
#
# Для остальных сайтов (loc.gov, coverr, generic-cffi) имитация браузера в
# BROWSER_HEADERS - осознанный анти-antibot приём, там UA-браузер - это то, что
# нужно. У Wikimedia наоборот: их API Etiquette прямо требует содержательный,
# идентифицирующий User-Agent (https://www.mediawiki.org/wiki/API:Etiquette),
# а не маскировку под браузер - этот урок уже был извлечён в search.py после
# эпизода с 403 "Please set a user-agent and respect our robot policy" (см.
# SESSION_USER_AGENT там же) и специального оверрайда для wikimedia в
# PREVIEW_HEADERS. download.py этот урок не унаследовал: тут для wikimedia
# по-прежнему шёл общий BROWSER_HEADERS с фейковым Chrome UA - т.е. ровно тот
# профиль ("анонимный, замаскированный под браузер, скриптовый трафик"), по
# которому новый anti-automation rate-limit Wikimedia 2026 года бьёт жёстче
# (см. https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits). Используется
# для ОБОИХ запросов - и к api.php, и к самому файлу (upload.wikimedia.org) -
# т.к. в search.py генерик-UA на превью-запросах (тоже к upload/thumb.
# wikimedia.org) уже давал 403 именно по этой причине.
WIKIMEDIA_HEADERS = {
    "User-Agent": "MediaSearchPipeline/1.0 "
                  "(https://github.com/SOTONATORE/Download; contact: fordlababit@gmail.com)",
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


def log_429_details(site: str, context: str, resp) -> None:
    """Печатает Retry-After и обрезанное тело ответа при HTTP 429 - раньше на
    прогоне с реальным 429 у Wikimedia в логе не было ничего, кроме самого факта
    статус-кода: ни Retry-After, ни тела ответа код нигде не печатал, поэтому
    нельзя было сказать по факту, что именно ответил сервер (частота по IP,
    per-minute automation-лимит, что-то ещё). context - произвольная строка для
    привязки к конкретному номеру/запросу (у LOC на уровне _loc_get номер файла
    не всегда известен - это общая обёртка и для метаданных, и для файла, поэтому
    здесь не жёстко number, а строка). Работает и для LOC (там та же прореха в
    _loc_get), и для Wikimedia - единая точка, единый формат в логе."""
    retry_after = resp.headers.get("Retry-After", "<нет заголовка>") if resp is not None else "<нет ответа>"
    try:
        body_snippet = resp.text[:300].replace("\n", " ") if resp is not None else "<нет тела>"
    except Exception:
        body_snippet = "<не удалось прочитать тело>"
    print(
        f"[DEBUG] {site} 429 ({context}): Retry-After={retry_after!r}, "
        f"тело (первые 300 симв.): {body_snippet!r}"
    )


def _extract_retry_after_seconds(resp, default_backoff: float) -> float:
    """Извлекает Retry-After (в секундах) из заголовка ответа при 429.
    Если заголовок есть и корректен, спим max(retry_after + 1.0, default_backoff),
    добавляя 1 секунду запаса от дрожания таймингов."""
    raw = resp.headers.get("Retry-After") if resp is not None else None
    if raw:
        try:
            val = float(raw)
            return max(val + 1.0, default_backoff)
        except (ValueError, TypeError):
            pass
    return default_backoff


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
# стояло 5.0с/~40% запаса; тронуто по итогам того же реального прогона (см. обсуждение).

# ---------------------------------------------------------------------------
# Раунд 7 (история) и Раунд 10 (Этап 2) — отмена разделения интервалов,
# единый интервал LOC от момента ЗАВЕРШЕНИЯ запроса и полная потокобезопасность.
#
# В Раунде 7 попытались разделить лимитер на metadata (3.2с) и file (0.45с) на основе
# официальной документации (20 запр/мин против 150 запр/мин для /storage-services/).
# Однако реальный прогон на 126 сегментах показал катастрофический побочный эффект:
# интервалы отсчитывались от СТАРТА каждого типа запроса. Скачивание тяжелого медиафайла
# занимало несколько секунд (> 3.2с). Когда скрипт переходил к следующему сегменту,
# лимитер метаданных смотрел на таймстемп старта прошлого JSON, видел, что прошло > 3.2с,
# и стрелял следующим JSON через 0.003 секунды после завершения загрузки картинки!
# В итоге Cloudflare на внешнем периметре loc.gov (который защищает все домены loc.gov
# суммарно по IP) зафиксировал всплеск частоты (20 запросов за 38 секунд) и забанил
# IP раннера капчей HTTP 429 ("<title>Just a moment...</title>"). Скрипт выставил
# _loc_exhausted = True, и все последующие файлы с primary на LOC упали (потеряно 8 файлов).
#
# Вторая проблема: у сегментов из параллельного пула backup мог вести на LOC (15 таких
# сегментов). При падении primary рабочий поток из пула обращался к LOC параллельно
# с основным последовательным циклом, вызывая гонки данных по таймстемпам и одновременные
# запросы к LOC.
#
# Решение Раунда 10:
#   1) Отмена разделения: единый интервал LOC_MIN_INTERVAL_SECONDS (3.2с) применяется
#      ко ВСЕМ сетевым операциям с loc.gov (и к метаданным ?fo=json, и к файлам).
#   2) Интервал отсчитывается строго от момента ЗАВЕРШЕНИЯ (finally: _loc_last_finish_ts)
#      предыдущего запроса, исключая любые нулевые паузы после долгих скачиваний.
#   3) Все операции с LOC защищены мьютексом _loc_lock (concurrency = 1 гарантирована
#      на уровне процесса, даже если фоновые потоки обращаются к backup на LOC).
#   4) При получении не-JSON ответа (HTML) детектируется капча/Cloudflare (бан), а для
#      обычных статических страниц (выставки /exhibits/, блоги) сразу генерируется
#      ошибка без 3 пустых ретраев, мгновенно переводя сегмент на backup.
# ---------------------------------------------------------------------------

# Константа сохранена для обратной совместимости, но приравнена к LOC_MIN_INTERVAL_SECONDS:
LOC_FILE_MIN_INTERVAL_SECONDS = float(
    os.environ.get("DOWNLOAD_LOC_FILE_MIN_INTERVAL_SECONDS", LOC_MIN_INTERVAL_SECONDS)
)
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
#
# Раунд 6 - разбор реального 429 на прогоне с ThreadPoolExecutor(max_workers=8):
# ThreadRateLimiter сам по себе оказался технически исправен (один инстанс на
# процесс, лок честно охватывает read-modify-write), но он гасил только СТАРТ
# самого первого запроса на файл (metadata api.php) - мимо него шли: а) ретраи
# на 429 внутри retry-цикла metadata-запроса (свой time.sleep + новый запрос в
# обход wait_turn), б) отдельный, вообще ничем не ограниченный второй запрос
# за самим файлом на upload.wikimedia.org и его собственный retry-цикл. Т.е.
# лимитер видел долю реального трафика к Wikimedia, а не весь. Плюс, по
# официальной документации (https://wikitech.wikimedia.org/wiki/Robot_policy),
# лимит Wikimedia для анонимов задан в первую очередь как КОНКУРЕНТНОСТЬ (не
# больше 3 одновременных запросов к API, не больше 2 - для скачивания файлов),
# а не только как частота стартов - ThreadRateLimiter конкурентность не
# ограничивал вообще: сколько потоков одновременно держат открытое соединение к
# Wikimedia, зависело только от DOWNLOAD_CONCURRENCY=8. Добавлен
# _wikimedia_semaphore на WIKIMEDIA_CONCURRENCY=2 (официальный документированный
# потолок для скачивания файлов у анонимов - взят как есть, без запаса сверху,
# в отличие от LOC, где 3.2с была оценкой с запасом от заявленного лимита
# запросов/мин), оборачивающий download_wikimedia_commons целиком (то есть
# оба запроса на один номер), плюс wait_turn() перенесён внутрь ОБОИХ
# retry-циклов, чтобы каждый фактический HTTP-запрос (включая ретраи и второй
# запрос за файлом) проходил через общий rate-limiter, а не только первая
# попытка первого запроса.
# ---------------------------------------------------------------------------

DOWNLOAD_CONCURRENCY = int(os.environ.get("DOWNLOAD_CONCURRENCY", 8))

# ---------------------------------------------------------------------------
# Раунд 7 - интервал Wikimedia вниз, тоже подтверждено официальной документацией.
#
# Официальный лимит для анонимных запросов - 5 запросов/сек (robot policy,
# https://wikitech.wikimedia.org/wiki/Robot_policy). Прежнее значение 1.5с давало
# эффективный темп всего 0.667 запр/сек - меньше 15% от разрешённого бюджета: интервал
# был подобран ещё в раунде 6 "на глаз" как консервативная защита от 429, ДО того как
# в этом же раунде 6 был отдельно добавлен _wikimedia_semaphore (WIKIMEDIA_CONCURRENCY=2) -
# т.е. конкурентность (вторая, отдельная официальная граница) уже покрыта им, а интервал
# ниже неё избыточно душил ещё и частоту стартов запросов поверх лимита конкурентности.
# 0.25с (4 запр/сек, ~20% запас от 5/сек) - та же методика округления В БОЛЬШУЮ сторону
# от точного значения (1/5=0.2с ровно), что у LOC_MIN_INTERVAL_SECONDS выше, с запасом
# ближе к верхней границе по вкладу WIKIMEDIA_CONCURRENCY=2 - несколько потоков одновременно
# ждут этот же общий ThreadRateLimiter, и любая просадка сети чуть смещает моменты wait_turn()
# у каждого, поэтому взят более широкий запас, чем можно было бы при строго одиночном потоке.
WIKIMEDIA_MIN_INTERVAL_SECONDS = float(os.environ.get("DOWNLOAD_WIKIMEDIA_MIN_INTERVAL_SECONDS", 0.25))
WIKIMEDIA_CONCURRENCY = int(os.environ.get("DOWNLOAD_WIKIMEDIA_CONCURRENCY", 2))
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
_wikimedia_semaphore = threading.Semaphore(WIKIMEDIA_CONCURRENCY)
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


# ---------------------------------------------------------------------------
# Лимит Pixabay API (100 запросов за 60 с на ключ): скользящее окно + общая пауза после 429.
#
# Относится ТОЛЬКО к запросам к API Pixabay (pixabay.com/api/ и /api/videos/ по id). Прямые ссылки
# (cdn.pixabay.com, is_direct_media_url) и скачивание самого файла - не API, в окне не считаются.
# Pexels не затронут.
#
# Почему не декоратор: _pexels_pixabay_throttled держит семафор на ВСЮ функцию, а ожидание места в окне
# и пауза после 429 могут длиться десятки секунд - Pexels-скачивания встали бы в очередь за Pixabay.
# Поэтому download_pixabay_* больше не обёрнуты декоратором: место в окне и общая пауза берутся ДО
# входа в семафор, а сам семафор держится только на время одного HTTP-запроса (запрос к API и,
# отдельно, скачивание файла). Пауза после 429 тоже спится вне семафора.
# ---------------------------------------------------------------------------

PIXABAY_WINDOW_SECONDS = 60.0
PIXABAY_RPM_DEFAULT = 90                    # запас от официальных 100/мин
PIXABAY_429_MAX_RETRIES = 3                 # повторов того же запроса после 429
PIXABAY_429_FALLBACK_PAUSE_SECONDS = 61.0   # нет заголовка / значение вне 0...600
PIXABAY_429_RESET_MARGIN_SECONDS = 1.0      # запас к X-RateLimit-Reset
PIXABAY_429_RESET_MAX_SECONDS = 600.0       # Reset выше этого - мусор в заголовке


def load_pixabay_rpm(environ=None) -> int:
    """PIXABAY_REQUESTS_PER_MINUTE: пусто/не задано -> 90; не целое число или < 1 -> ValueError
    с названием переменной (вызывающий останавливает запуск)."""
    env = os.environ if environ is None else environ
    raw = env.get("PIXABAY_REQUESTS_PER_MINUTE")
    if raw is None or not raw.strip():
        return PIXABAY_RPM_DEFAULT
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"PIXABAY_REQUESTS_PER_MINUTE={raw!r}: ожидается целое число >= 1") from None
    if value < 1:
        raise ValueError(f"PIXABAY_REQUESTS_PER_MINUTE={raw!r}: значение должно быть не меньше 1")
    return value


def pixabay_reset_pause(raw) -> tuple:
    """(пауза в секундах, откуда взята) по заголовку X-RateLimit-Reset (секунды до сброса окна):
    значение + 1 с; нет заголовка / не число / вне 0...600 -> 61 с."""
    if raw is None:
        return PIXABAY_429_FALLBACK_PAUSE_SECONDS, "X-RateLimit-Reset отсутствует"
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return PIXABAY_429_FALLBACK_PAUSE_SECONDS, f"X-RateLimit-Reset={raw!r} не число"
    if not math.isfinite(v) or v < 0 or v > PIXABAY_429_RESET_MAX_SECONDS:
        return PIXABAY_429_FALLBACK_PAUSE_SECONDS, f"X-RateLimit-Reset={raw!r} вне 0...{PIXABAY_429_RESET_MAX_SECONDS:g}"
    return v + PIXABAY_429_RESET_MARGIN_SECONDS, f"X-RateLimit-Reset={raw} + {PIXABAY_429_RESET_MARGIN_SECONDS:g} с"


class ThreadSlidingWindow:
    """Не более max_requests взятий за любые window_seconds (скользящее окно), потокобезопасно.
    acquire() берёт место, а если мест нет - спит (sleep_fn, без удержания собственной блокировки)
    до освобождения и берёт его. Возвращает True, если пришлось ждать. Время и сон передаются
    снаружи (now_fn/sleep_fn) - в selftest подменяются."""

    def __init__(self, max_requests: int, window_seconds: float,
                 now_fn=time.monotonic, sleep_fn=time.sleep) -> None:
        if not isinstance(max_requests, int) or isinstance(max_requests, bool) or max_requests < 1:
            raise ValueError(f"max_requests должен быть целым >= 1, получено {max_requests!r}")
        if not window_seconds > 0:
            raise ValueError(f"window_seconds должен быть > 0, получено {window_seconds!r}")
        self.max_requests = max_requests
        self.window_seconds = float(window_seconds)
        self._now = now_fn
        self._sleep = sleep_fn
        self._lock = threading.Lock()
        self._stamps: list = []

    def acquire(self) -> bool:
        waited = False
        while True:
            with self._lock:
                now = self._now()
                self._stamps = [t for t in self._stamps if t + self.window_seconds > now]
                if len(self._stamps) < self.max_requests:
                    self._stamps.append(now)
                    return waited
                wait = self._stamps[0] + self.window_seconds - now
            waited = True
            self._sleep(max(wait, 0.001))


class SharedPause:
    """Общая для всех потоков "пауза до": после 429 один раз выставляется, и ВСЕ потоки перед
    следующим запросом ждут её конца (иначе параллельные запросы получили бы 429 по очереди и
    сожгли бы свои повторы). Время и сон - снаружи."""

    def __init__(self, now_fn=time.monotonic, sleep_fn=time.sleep) -> None:
        self._now = now_fn
        self._sleep = sleep_fn
        self._lock = threading.Lock()
        self._until = 0.0

    def extend(self, seconds: float) -> None:
        with self._lock:
            self._until = max(self._until, self._now() + seconds)

    def wait(self) -> float:
        """Спит, пока пауза не кончится; возвращает сколько секунд проспал (0.0 - паузы нет)."""
        slept = 0.0
        while True:
            with self._lock:
                remaining = self._until - self._now()
            if remaining <= 0:
                return slept
            self._sleep(remaining)
            slept += remaining


class PixabayApiStats:
    """Потокобезопасные счётчики для итоговой строки."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.rate_limited = 0
        self.retries = 0
        self.window_waits = 0

    def add(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def summary_line(self) -> str:
        with self._lock:
            return (f"Pixabay API: запросов {self.requests}, получено 429: {self.rate_limited}, "
                    f"повторов: {self.retries}, ожиданий места в окне: {self.window_waits}")


class PixabayApiLimiter:
    """Окно + общая пауза + счётчики. before_request() вызывается ДО входа в семафор."""

    def __init__(self, rpm: int, now_fn=time.monotonic, sleep_fn=time.sleep) -> None:
        self.window = ThreadSlidingWindow(rpm, PIXABAY_WINDOW_SECONDS, now_fn, sleep_fn)
        self.pause = SharedPause(now_fn, sleep_fn)
        self.stats = PixabayApiStats()

    def before_request(self) -> None:
        self.pause.wait()
        if self.window.acquire():
            self.stats.add(window_waits=1)
        self.pause.wait()  # 429 мог прийти, пока ждали место в окне


_pixabay_limiter = None
_pixabay_limiter_lock = threading.Lock()


def get_pixabay_limiter() -> PixabayApiLimiter:
    """Один экземпляр на все потоки. Создаётся при первом обращении из PIXABAY_REQUESTS_PER_MINUTE
    (main() вызывает его в самом начале, чтобы неверное значение остановило запуск до скачивания)."""
    global _pixabay_limiter
    with _pixabay_limiter_lock:
        if _pixabay_limiter is None:
            _pixabay_limiter = PixabayApiLimiter(load_pixabay_rpm())
        return _pixabay_limiter


def set_pixabay_limiter(limiter) -> None:
    """Подмена экземпляра (selftest)."""
    global _pixabay_limiter
    with _pixabay_limiter_lock:
        _pixabay_limiter = limiter


def pixabay_api_get(number: int, api_url: str, what: str):
    """GET к API Pixabay с окном и обработкой 429. Возвращает ответ со статусом < 400.
    429: пауза до сброса окна (X-RateLimit-Reset + 1 с, иначе 61 с) и повтор того же запроса, до
    PIXABAY_429_MAX_RETRIES повторов; пауза > DIRECT_MAX_RETRY_WAIT_SECONDS не ждётся. Повторы тоже
    берут место в окне. Окончательный провал по 429 - RuntimeError с текстом "HTTP 429 ..." (так
    classify_failure_message не запускает внешний повтор, сразу backup); остальные ошибки HTTP -
    как раньше (raise_for_status). Вне семафора спит и ждёт место; под семафором - только сам запрос."""
    limiter = get_pixabay_limiter()
    retries = 0
    while True:
        limiter.before_request()
        with _pexels_pixabay_semaphore:
            resp = cffi_requests.get(api_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=20)
        limiter.stats.add(requests=1)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        limiter.stats.add(rate_limited=1)
        log_429_details("Pixabay", f"API {what}, номер {number}, попытка {retries + 1}", resp)
        pause, src = pixabay_reset_pause(resp.headers.get("X-RateLimit-Reset"))
        if retries >= PIXABAY_429_MAX_RETRIES:
            raise RuntimeError(f"HTTP 429: лимит Pixabay API не сбросился после {retries} повторов")
        if pause > DIRECT_MAX_RETRY_WAIT_SECONDS:
            raise RuntimeError(
                f"HTTP 429: пауза {pause:.0f} с ({src}) больше предела "
                f"{DIRECT_MAX_RETRY_WAIT_SECONDS:.0f} с, не жду"
            )
        retries += 1
        limiter.stats.add(retries=1)
        limiter.pause.extend(pause)
        print(f"[ИНФО] Pixabay API 429 ({what}, номер {number}): общая пауза {pause:.1f} с ({src}), "
              f"повтор {retries}/{PIXABAY_429_MAX_RETRIES}")


# ---------------------------------------------------------------------------
# Прямые ссылки на файлы Pexels/Pixabay (их CDN). Такая ссылка качается сразу по
# URL: без запроса к API за метаданными и без API-ключа. Все остальные ссылки
# Pexels/Pixabay считаются страницами и идут прежним путём через API.
# Сравнивается ИМЕННО hostname целиком (не подстрока), поэтому
# "images.pexels.com.evil.com" или "evil.com/images.pexels.com/x.jpg" - не прямые.
# ---------------------------------------------------------------------------
DIRECT_MEDIA_DOMAINS = frozenset({"images.pexels.com", "videos.pexels.com", "cdn.pixabay.com"})
DIRECT_DOWNLOAD_ATTEMPTS = 3            # попыток на одну прямую ссылку при 429
DIRECT_MAX_RETRY_WAIT_SECONDS = 120.0   # Retry-After больше этого - не ждём, сразу провал (-> backup)


def is_direct_media_url(url: str) -> bool:
    """True, если хост URL - один из DIRECT_MEDIA_DOMAINS."""
    try:
        host = (urlparse((url or "").strip()).hostname or "").lower()
    except ValueError:
        return False
    return host in DIRECT_MEDIA_DOMAINS


def classify_pexels_pixabay_source(url: str) -> str | None:
    """'direct' - прямая ссылка; 'api' - URL, который download_media_item отдаёт в
    download_pexels_*/download_pixabay_* (те же условия, что в диспетчере); иначе None."""
    if is_direct_media_url(url):
        return "direct"
    u = (url or "").strip().lower()
    if "pexels.com" in u and ("/video/" in u or "/photo/" in u):
        return "api"
    if "pixabay.com" in u:
        return "api"
    return None


class SourceStats:
    """Потокобезопасный счётчик УСПЕШНО скачанных (и прошедших проверку длительности)
    файлов Pexels/Pixabay: по прямой ссылке и через API. Backup-ссылки считаются так же."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.direct = 0
        self.api = 0

    def add(self, url: str) -> None:
        kind = classify_pexels_pixabay_source(url)
        with self._lock:
            if kind == "direct":
                self.direct += 1
            elif kind == "api":
                self.api += 1

    def summary_line(self) -> str:
        with self._lock:
            return f"Скачано по прямой ссылке: {self.direct}, через API: {self.api}"


SOURCE_STATS = SourceStats()

# Состояние LOC: лок гарантирует строгую последовательность (concurrency=1) даже
# если фоновые потоки ThreadPoolExecutor обращаются к backup на loc.gov.
_loc_lock = threading.Lock()
_loc_last_finish_ts = 0.0  # Раунд 10: интервал считаем строго от ЗАВЕРШЕНИЯ запроса
_loc_exhausted = False
# ПРИМЕЧАНИЕ: флаг "исчерпан" по-прежнему ОДИН на оба типа запроса, не разделяем -
# официальная документация явно подтверждает раздельные ЛИМИТЫ ЧАСТОТЫ (20/мин и
# 150/мин), но НЕ подтверждает раздельную длительность/область бана при 429 для
# каждого эндпоинта по отдельности. Раз это не подтверждённый факт, а предположение -
# перестраховываемся и при 429 на ЛЮБОМ из двух эндпоинтов считаем исчерпанным весь LOC
# целиком, как и раньше.


class LocExhaustedError(RuntimeError):
    """LOC уже забанил нас в этом запуске (был 429) - дальнейшие попытки в рамках
    текущего прогона гарантированно провалятся тем же способом, ретраить бессмысленно."""


def _loc_wait_turn(is_file_request: bool = False) -> None:
    """Раунд 10: единый интервал для ВСЕХ сетевых запросов к loc.gov (и метаданных,
    и файлов), отсчитываемый от момента ЗАВЕРШЕНИЯ прошлого запроса.
    Вызывается строго под _loc_lock."""
    global _loc_last_finish_ts
    now = time.monotonic()
    wait = LOC_MIN_INTERVAL_SECONDS - (now - _loc_last_finish_ts)
    if wait > 0:
        time.sleep(wait)


def _loc_raw_get(url: str, timeout: int = 30, **kwargs):
    """Выполняет один фактический HTTP-запрос к loc.gov строго под _loc_lock
    с соблюдением минимального интервала от момента ЗАВЕРШЕНИЯ прошлого сетевого
    запроса (finally: _loc_last_finish_ts = time.monotonic())."""
    global _loc_last_finish_ts, _loc_exhausted
    with _loc_lock:
        if _loc_exhausted:
            raise LocExhaustedError("LOC уже исчерпан в этом запуске (был 429 ранее)")
        _loc_wait_turn()
        try:
            return cffi_requests.get(url, impersonate="chrome", timeout=timeout, **kwargs)
        finally:
            _loc_last_finish_ts = time.monotonic()


def _loc_get(url: str, timeout: int = 30, is_file_request: bool = False, **kwargs):
    """Обёртка над cffi_requests.get специально для loc.gov: выполняет сетевые
    запросы строго последовательно через _loc_raw_get под _loc_lock с единым
    интервалом LOC_MIN_INTERVAL_SECONDS от момента ЗАВЕРШЕНИЯ прошлого запроса.
    Ретраит транзиентные сетевые ошибки/таймауты. При первом же HTTP 429 сразу
    помечает LOC исчерпанным до конца запуска (_loc_exhausted = True)."""
    global _loc_exhausted
    with _loc_lock:
        if _loc_exhausted:
            raise LocExhaustedError("LOC уже исчерпан в этом запуске (был 429 ранее)")

    last_exc: Exception | None = None
    for attempt in range(1, LOC_RETRIES + 1):
        has_next = attempt < LOC_RETRIES
        try:
            resp = _loc_raw_get(url, timeout=timeout, **kwargs)
        except LocExhaustedError:
            raise
        except Exception as e:
            last_exc = e
            print(
                f"[WARN] LOC: сбой запроса, попытка {attempt}/{LOC_RETRIES} "
                f"[{type(e).__name__}: {_loc_redact(str(e))[:200]}] {_loc_redact(url)}"
            )
            if has_next:
                time.sleep(2 * attempt)
            continue

        if resp.status_code == 429:
            log_429_details("LOC", url, resp)
            with _loc_lock:
                _loc_exhausted = True
            raise LocExhaustedError("HTTP 429 - LOC помечен исчерпанным до конца текущего запуска")

        resp.raise_for_status()
        return resp

    raise last_exc or RuntimeError("не удалось выполнить запрос к loc.gov")


def _loc_redact(text: str) -> str:
    """Вырезает значения key/api_key/token из текста перед выводом в лог."""
    return re.sub(r"(?i)\b(key|api_key|apikey|token|access_token)=[^&\s'\"]+", r"\1=***", text or "")


def _loc_get_json(url: str, **kwargs) -> dict:
    """Как _loc_get, но JSON парсится ВНУТРИ цикла ретраев (LOC_RETRIES).
    Все сетевые запросы выполняются строго последовательно через _loc_raw_get
    под _loc_lock с единым интервалом LOC_MIN_INTERVAL_SECONDS от ЗАВЕРШЕНИЯ прошлого.
    При получении страницы CAPTCHA/Cloudflare выставляет _loc_exhausted = True.
    При получении статического HTML (выставки, блоги, порталы вместо API) сразу
    выбрасывает исключение без бессмысленных ретраев, мгновенно переводя сегмент
    на backup без сжигания лимитов. 429 - LocExhaustedError."""
    global _loc_exhausted
    with _loc_lock:
        if _loc_exhausted:
            raise LocExhaustedError("LOC уже исчерпан в этом запуске (был 429 ранее)")

    timeout = kwargs.pop("timeout", 30)
    last_exc: Exception | None = None
    for attempt in range(1, LOC_RETRIES + 1):
        has_next = attempt < LOC_RETRIES
        try:
            resp = _loc_raw_get(url, timeout=timeout, **kwargs)
        except LocExhaustedError:
            raise
        except Exception as e:
            last_exc = e
            print(
                f"[WARN] LOC: сбой запроса, попытка {attempt}/{LOC_RETRIES} "
                f"[{type(e).__name__}: {_loc_redact(str(e))[:200]}] {_loc_redact(url)}"
            )
            if has_next:
                time.sleep(2 * attempt)
            continue

        if resp.status_code == 429:
            log_429_details("LOC", url, resp)
            with _loc_lock:
                _loc_exhausted = True
            raise LocExhaustedError("HTTP 429 - LOC помечен исчерпанным до конца текущего запуска")

        if resp.status_code in (500, 502, 503, 504):
            last_exc = RuntimeError(f"loc.gov: HTTP {resp.status_code}")
            print(f"[WARN] LOC: HTTP {resp.status_code}, попытка {attempt}/{LOC_RETRIES}")
            if has_next:
                time.sleep(2 * attempt + random.uniform(0, 1))
            continue

        resp.raise_for_status()

        # Проверка ответа: сначала ищем признаки капчи / Cloudflare challenge
        raw_text = resp.text or ""
        lower_text = raw_text.lower()
        if any(marker in lower_text for marker in (
            "just a moment...", "cf-chl", "cloudflare", "attention required", "captcha", "security check"
        )):
            log_429_details("LOC", f"CAPTCHA/Cloudflare ({url})", resp)
            with _loc_lock:
                _loc_exhausted = True
            raise LocExhaustedError(
                f"loc.gov вернул страницу проверки/CAPTCHA (Cloudflare) при статусе {resp.status_code} - "
                f"LOC помечен исчерпанным до конца текущего запуска"
            )

        # Проверяем, не является ли ответ статической HTML-страницей (выставки, блоги, порталы)
        ctype = resp.headers.get("Content-Type", "").lower()
        if ctype.startswith("text/html") or looks_like_html(resp.content):
            print(f"[WARN] LOC: страница отдаёт HTML вместо JSON (не API-эндпоинт): {_loc_redact(url)}")
            raise ValueError(f"loc.gov вернул HTML-страницу вместо JSON-метаданных (URL={_loc_redact(url)})")

        try:
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError(f"JSON не объект, а {type(data).__name__}")
        except Exception as e:
            try:
                body = _loc_redact(raw_text[:200].replace("\n", " "))
            except Exception:
                body = "<не удалось прочитать тело>"
            last_exc = RuntimeError(
                f"loc.gov: не-JSON ответ (status={resp.status_code}, Content-Type={ctype!r}): {type(e).__name__}"
            )
            print(
                f"[WARN] LOC: не-JSON ответ, попытка {attempt}/{LOC_RETRIES}, "
                f"status={resp.status_code}, Content-Type={ctype!r}, "
                f"тело (первые 200 симв.): {body!r}"
            )
            if has_next:
                time.sleep(2 * attempt + random.uniform(0, 1))
            continue

        if attempt > 1:
            print(f"[INFO] LOC: успех с попытки {attempt}/{LOC_RETRIES} ({_loc_redact(url)})")
        return data

    raise last_exc or RuntimeError("не удалось получить JSON от loc.gov")


def parse_links(raw_text: str) -> dict:
    """Общий разбор строк 'номер: URL' - используется и для links.txt/
    INPUT_LINKS, и для backup_links.txt/INPUT_BACKUP_LINKS (тот же формат,
    целые номера сегментов - см. докстринг search.py про backup_links.txt)."""
    matches = re.findall(r'(\d+)\s*:\s*(https?://[^\s]+)', raw_text)
    return {int(num_str): url for num_str, url in matches}


class ProgressCounter:
    """Счётчик прогресса скачивания для ОДНОГО вызова parse_and_download_links
    (создаётся внутри неё, глобального состояния не держит). Общий для пула и
    LOC-цикла: один Lock, печать и инкремент под ним - порядок строк совпадает
    с порядком счётчика, проценты не убывают."""

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.ok = 0
        self.failed = 0
        self._lock = threading.Lock()

    def run(self, number: int, primary_url: str, backup_url: str = "") -> None:
        """Обёртка вокруг download_number_with_backup: счётчик растёт в finally,
        исключение пробрасывается как раньше."""
        raised = True
        try:
            download_number_with_backup(number, primary_url, backup_url)
            raised = False
        finally:
            prefix = f"{number}: "
            with _failed_items_lock:
                is_failed = raised or any(item.startswith(prefix) for item in FAILED_ITEMS)
            with self._lock:
                self.done += 1
                if is_failed:
                    self.failed += 1
                else:
                    self.ok += 1
                status = "ОШИБКА" if is_failed else "скачан"
                print(
                    f"[ПРОГРЕСС] {self.done}/{self.total} "
                    f"({self.done * 100 / self.total:.0f}%) номер {number}: {status}",
                    flush=True,
                )

    def print_summary(self) -> None:
        if self.total == 0:
            return
        with self._lock:
            print(
                f"[ПРОГРЕСС] Готово: скачано {self.ok}, с ошибкой {self.failed} "
                f"из {self.total}",
                flush=True,
            )


def parse_and_download_links(env_name: str, backup_env_name: str = "") -> None:
    raw_text = os.environ.get(env_name, "")
    if not raw_text:
        print("Список ссылок пуст. Нечего скачивать.")
        return

    print("Начинаю разбор и скачивание файлов...")
    links = parse_links(raw_text)

    if not links:
        print("[ОШИБКА] Не удалось распознать ссылки формата 'номер: ссылка'")
        return

    # backup_env_name может быть пустой строкой (старый сценарий с ручными
    # ссылками, где backup-файла нет вообще) - тогда backup_links остаётся
    # пустым словарём, и download_number_with_backup для каждого номера просто
    # не находит backup (backup_url="") - это нормальный случай, не ошибка.
    backup_links = {}
    if backup_env_name:
        backup_raw_text = os.environ.get(backup_env_name, "")
        if backup_raw_text:
            backup_links = parse_links(backup_raw_text)

    print(
        f"Найдено ссылок для скачивания: {len(links)}"
        + (f" (с backup-ссылкой: {len(backup_links)})" if backup_env_name else "")
    )

    prepare_duration_check(links, backup_links)

    # Раунд 5: LOC остаётся полностью последовательным (не трогаем его rate-limiter
    # логику), всё остальное идёт через ThreadPoolExecutor - см. подробное обоснование
    # в блоке комментариев "Раунд 5" в начале файла. Важно: LOC-цикл ниже запускается
    # СРАЗУ ПОСЛЕ submit() пула, а не после его завершения - это даёт перекрытие по
    # времени (пока LOC ждёт свои 3.2с между запросами, пул параллельно докачивает
    # всё остальное в фоне), а не сложение времени LOC + времени всего остального.
    #
    # Бакет (LOC-последовательный / параллельный пул) определяется ТОЛЬКО по сайту
    # primary-ссылки, backup сюда не влияет - если backup окажется с другого сайта,
    # он всё равно физически безопасен для вызова из любого потока (все rate-limiter'ы
    # сайтов thread-safe сами по себе, см. ThreadRateLimiter/_wikimedia_semaphore/
    # _pexels_pixabay_semaphore выше, а LOC защищён _loc_lock), просто пойдёт туда же,
    # где обрабатывался provider primary для этого number.
    loc_items = []
    other_items = []
    for number, url in links.items():
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

    progress = ProgressCounter(len(other_items) + len(loc_items))

    with ThreadPoolExecutor(max_workers=DOWNLOAD_CONCURRENCY) as executor:
        futures = [
            executor.submit(progress.run, number, url, backup_links.get(number, ""))
            for number, url in other_items
        ]

        for number, url in loc_items:
            progress.run(number, url, backup_links.get(number, ""))

        for future in as_completed(futures):
            future.result()  # пробрасываем неожиданные исключения - download_number_with_backup
                              # сам ловит и логирует всё ожидаемое через fail()

    progress.print_summary()


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


def save_media(number: int, content: bytes, hint_url: str = "", content_type: str = "",
               site: str = "") -> bool:
    """Единственная точка записи N.<ext>. Формат - ТОЛЬКО по содержимому (sniff_format).
    True - файл сохранён; False - отказ (fail() уже вызван, файл не записан)."""
    fmt = sniff_format(content)
    if fmt is None:
        desc = describe_rejected(content)
        fail(number, f"неподдерживаемый формат: {desc}")
        FORMAT_REJECTS.add(site, desc)
        return False

    hint = ext_from_name(hint_url)
    hint_norm = "jpg" if hint == "jpeg" else hint
    # mp4/mov/avi между собой совместимы (бренд ftyp ненадёжен): сохраняем по содержимому.
    both_video = kind_of_ext(hint_norm) == "video" and kind_of_ext(fmt) == "video"
    if hint in _KNOWN_HINT_EXTS and hint_norm != fmt and not both_video:
        fail(number, f"неподдерживаемый формат: расширение .{hint} не совпадает с содержимым ({fmt})")
        FORMAT_REJECTS.add(site, hint)
        return False

    filename = f"{number}.{fmt}"
    with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
        f.write(content)
    print(f"[OK] {(site or 'файл').upper()} {number} ({filename}) успешно скачано")
    return True


def download_media_item(number: int, url: str) -> None:
    url = url.strip()
    url_lower = url.lower()

    # 0. ПРЯМАЯ ССЫЛКА PEXELS/PIXABAY (CDN): без API и без ключа. Должна идти ПЕРВОЙ:
    # cdn.pixabay.com / videos.pexels.com иначе совпали бы с проверками ниже по подстроке.
    if is_direct_media_url(url):
        download_direct_media(number, url)
        return

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

    # 7. ПРЯМЫЕ ССЫЛКИ .mp4/.mov + СТРАНИЦЫ MIXKIT -> ЧЕРЕЗ yt-dlp
    if url_lower.endswith((".mp4", ".mov")) or "mixkit.co" in url_lower:
        download_via_ytdlp(number, url)
        return

    # 8. ВСЁ ОСТАЛЬНОЕ - СКАЧИВАНИЕ ЧЕРЕЗ curl_cffi С ИМИТАЦИЕЙ БРАУЗЕРА
    download_direct_via_cffi(number, url)


# ---------------------------------------------------------------------------
# Backup-ссылки: retry на транзиентный сетевой сбой + переход на backup URL.
#
# Ни одна из 8 site-функций выше НЕ изменена: они как раньше сами решают,
# получилось скачать или нет, и сами же зовут fail() при провале. Вместо того
# чтобы переписывать восемь разных except-блоков на "бросай исключение вместо
# fail()", успех/провал конкретного вызова определяется СНАРУЖИ - по тому,
# появилась ли в FAILED_ITEMS новая запись с префиксом "{number}: " за время
# вызова. Это безопасно при параллельных потоках (ThreadPoolExecutor): каждый
# номер сегмента обрабатывается ровно одним вызовом download_number_with_backup
# за весь прогон (parse_and_download_links вызывает его по одному разу на
# number), поэтому записи с ЧУЖИМ number, добавляемые в это же время другими
# потоками, никак не пересекаются с проверкой "по префиксу этого number" -
# сравнивать пришлось бы длину всего списка, но не подмножество с нашим
# префиксом.
#
# Классификация сбоя (для решения "делать быстрый повтор или сразу backup")
# идёт по тексту сообщения, которое сама site-функция и так уже формирует в
# fail(number, f"...: {e}") - т.е. по str() исходного исключения. Раз функции
# не трогаем, точный тип исключения curl_cffi наружу не долетает, но текст
# исключения (timeout/connection reset/HTTP-2 stream reset и т.п.) долетает
# всегда - этого достаточно для грубой, но практичной классификации.
# ---------------------------------------------------------------------------

def classify_failure_message(message: str) -> str:
    """"429" / "transient" / "other" по тексту сообщения об ошибке.

    "429" - рейт-лимит, немедленный повтор того же запроса бессмысленен (ответ
    будет тем же) - НЕ делаем внешний retry, сразу переходим к backup (если
    есть). Это НЕ отменяет и не дублирует уже существующую 429-логику
    LOC/Wikimedia - для их URL внешний retry в любом случае не применяется
    (см. _url_gets_external_retry), этот случай здесь только на случай 429 у
    pexels/pixabay/coverr/generic-cffi, где своей 429-логики нет вообще.

    "transient" - именно то, ради чего затевалась вся задача: сетевой/
    транспортный сбой (timeout, connection reset, HTTP/2 stream reset и
    т.п.), который с большой вероятностью не повторится при немедленном
    повторном запросе. Получает ровно 1 быстрый повтор той же ссылки.

    "other" - логическая ошибка (нет нужного поля в ответе API, не нашли
    ссылку на странице, пришла HTML-страница вместо файла и т.п.) - повторный
    запрос даст тот же результат, поэтому сразу backup без бессмысленного
    повтора."""
    text = message.lower()
    if "429" in text or "too many requests" in text:
        return "429"
    transient_markers = (
        "timeout", "timed out", "connection reset", "connection aborted",
        "connection refused", "stream reset", "rst_stream", "reset by peer",
        "broken pipe", "eof occurred", "ssl error", "name or service not known",
        "could not resolve host", "temporary failure in name resolution",
        "failed to establish a new connection", "curl: (",
        "connectionerror", "http/2 stream", "network is unreachable",
        "socket.timeout", "connection closed",
    )
    if any(marker in text for marker in transient_markers):
        return "transient"
    return "other"


def _url_gets_external_retry(url: str) -> bool:
    """LOC/Wikimedia/yt-dlp-маршрут (см. те же условия, что и в
    download_media_item выше) уже имеют СВОЮ, проверенную боем ретрай-логику
    (429-ретраи у LOC/Wikimedia, --retries у yt-dlp) - модуль их не трогает
    вообще, и добавлять поверх ещё один внешний retry было бы избыточно и
    только тратило бы время. Для этих маршрутов backup применяется СРАЗУ по
    итогу их (нетронутой) внутренней логики, без дополнительного повтора.
    Всем остальным маршрутам (pexels/pixabay/coverr/generic-cffi - у них
    сейчас 0 внутренних ретраев) - положен новый внешний "1 повтор на
    транзиентный сбой"."""
    if is_direct_media_url(url):
        # Свой цикл только на 429; транзиентные сбои (как у pexels/pixabay-страниц)
        # получают внешний повтор. Без этой строки прямой .mp4 попал бы в yt-dlp-ветку ниже.
        return True
    url_lower = url.lower()
    if "wikimedia.org" in url_lower or "wikipedia.org" in url_lower:
        return False
    if "loc.gov" in url_lower:
        return False
    if url_lower.endswith((".mp4", ".mov")) or "mixkit.co" in url_lower:
        return False
    return True


# ---------------------------------------------------------------------------
# Строгая проверка реальной длительности скачанных ВИДЕО (фото не трогаем).
# Видео короче нужного даёт чёрные кадры в автомонтаже. Нужная длина берётся из
# min_durations.json {"номер сегмента": секунды} (артефакт search-links). Проверка
# встроена в _attempt_download, поэтому backup-ссылка скачивается и проверяется
# ровно тем же правилом, а итоговая причина попадает в download_failed.txt.
# Сравнение СТРОГОЕ: длительность >= нужной, без допусков и округлений.
# ---------------------------------------------------------------------------

FFPROBE_TIMEOUT_SECONDS = 30
MIN_DURATIONS: dict[int, float] | None = None  # None - файл не загружен


class FfprobeError(RuntimeError):
    """ffprobe завершился с ошибкой (ненулевой код)."""


def _run_ffprobe(args: list[str]) -> str:
    """Единственное место запуска ffprobe (в selftest подменяется через
    _FFPROBE_RUNNER). Возвращает stdout; бросает subprocess.TimeoutExpired,
    OSError (в т.ч. FileNotFoundError) или FfprobeError."""
    res = subprocess.run(args, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_SECONDS)
    if res.returncode != 0:
        raise FfprobeError(f"код {res.returncode}: {(res.stderr or '').strip()[:200]}")
    return res.stdout


_FFPROBE_RUNNER = _run_ffprobe


def parse_ffprobe_duration(text: str) -> float | None:
    """Число секунд из вывода ffprobe или None ("N/A", пусто, мусор, nan/inf, <= 0)."""
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = float(line)
        except ValueError:
            return None
        return value if math.isfinite(value) and value > 0 else None
    return None


def duration_ok(actual: float, required: float) -> bool:
    """Строго: равно проходит, на сколько угодно меньше - нет."""
    return actual >= required


def parse_min_durations(text: str) -> dict[int, float]:
    """{"номер": секунды} -> {int: float}. ValueError с русским текстом при любой ошибке."""
    try:
        raw = json.loads(text)
    except (ValueError, TypeError) as e:
        raise ValueError(f"не удалось разобрать JSON: {e}")
    if not isinstance(raw, dict):
        raise ValueError("ожидался объект вида {\"номер сегмента\": секунды}")
    result: dict[int, float] = {}
    for key, value in raw.items():
        k = str(key).strip()
        if not k.isdigit():
            raise ValueError(f"ключ {key!r} не является номером сегмента")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"значение для сегмента {k} не число: {value!r}")
        v = float(value)
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"значение для сегмента {k} некорректно: {value!r}")
        result[int(k)] = v
    return result


def load_min_durations() -> tuple[dict[int, float] | None, str]:
    """(данные, "") или (None, причина). Источник: INPUT_MIN_DURATIONS (путь к файлу;
    если значение начинается с "{" - это сам JSON, как INPUT_LINKS несёт сам текст),
    по умолчанию min_durations.json в текущей папке (туда же скачивается search-links
    с links.txt)."""
    value = os.environ.get("INPUT_MIN_DURATIONS", "").strip()
    if value.startswith("{"):
        source, text = "INPUT_MIN_DURATIONS", value
    else:
        path = value or "min_durations.json"
        source = path
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            return None, f"файл {path} не найден или не читается ({e})"
    try:
        return parse_min_durations(text), ""
    except ValueError as e:
        return None, f"{source}: {e}"


def measure_duration(path: str) -> tuple[float | None, str]:
    """(секунды, "") или (None, причина). Сначала видеопоток, запасной вариант - контейнер."""
    probes = (
        ("видеопотока", ["-select_streams", "v:0", "-show_entries", "stream=duration"]),
        ("контейнера", ["-show_entries", "format=duration"]),
    )
    for what, extra in probes:
        args = ["ffprobe", "-v", "error", *extra, "-of", "default=nw=1:nk=1", path]
        try:
            out = _FFPROBE_RUNNER(args)
        except subprocess.TimeoutExpired:
            return None, f"ffprobe: таймаут {FFPROBE_TIMEOUT_SECONDS} с при чтении длительности {what}"
        except FileNotFoundError:
            return None, "ffprobe не найден в PATH"
        except OSError as e:
            return None, f"не удалось запустить ffprobe: {e}"
        except FfprobeError as e:
            return None, f"ffprobe завершился с ошибкой при чтении {what} ({e})"
        value = parse_ffprobe_duration(out)
        if value is not None:
            return value, ""
    return None, "ffprobe не вернул длительность ни у видеопотока, ни у контейнера"


def check_video_duration(number: int, path: str,
                         min_durations: dict[int, float] | None) -> tuple[bool, str]:
    """(прошло, причина). Не прошло = любой случай, когда нельзя доказать длительность >= нужной."""
    if min_durations is None:
        return False, "проверка длительности невозможна: min_durations.json не загружен"
    if number not in min_durations:
        return False, f"в min_durations.json нет записи для сегмента {number}"
    required = min_durations[number]
    actual, reason = measure_duration(path)
    if actual is None:
        return False, f"не удалось измерить длительность видео: {reason}"
    if not duration_ok(actual, required):
        return False, f"видео короче нужного: {actual} с < {required} с"
    return True, ""


class DurationStats:
    """Потокобезопасные счётчики: проверено / отсеяно / заменено backup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.checked = 0
        self.rejected = 0
        self.replaced = 0
        self._rejected_numbers: set[int] = set()

    def add_check(self, number: int, ok: bool) -> None:
        with self._lock:
            self.checked += 1
            if not ok:
                self.rejected += 1
                self._rejected_numbers.add(number)

    def note_backup_success(self, number: int) -> None:
        """Backup спас номер, чей primary был отсеян проверкой длительности."""
        with self._lock:
            if number in self._rejected_numbers:
                self.replaced += 1

    def summary_line(self) -> str:
        with self._lock:
            return (f"Проверка длительности: проверено видео {self.checked}, "
                    f"отсеяно по длительности {self.rejected}, заменено backup {self.replaced}")


DURATION_STATS = DurationStats()


def _url_looks_video(url: str) -> bool:
    """Эвристика по URL (до скачивания): похоже ли, что там видео."""
    u = url.lower()
    if kind_of_ext(ext_from_name(url)) == "video":
        return True
    return (("pexels.com" in u and "/video/" in u) or ("pixabay.com" in u and "/videos/" in u)
            or "coverr.co" in u or "mixkit.co" in u)


def _stop(message: str) -> None:
    print(f"[КРИТИЧНО] {message}", file=sys.stderr)
    sys.exit(1)


def prepare_duration_check(links: dict, backup_links: dict) -> None:
    """Вызывается до скачивания. Если среди primary/backup-ссылок есть хоть одно видео -
    без min_durations.json или без ffprobe останавливаемся. Если видео не ожидается,
    файл грузится "по возможности" (на случай видео, не распознанного по URL); а если
    оно всё же скачается без данных - _attempt_download засчитает его неудачей."""
    global MIN_DURATIONS
    any_video = any(_url_looks_video(u) for u in [*links.values(), *backup_links.values()])
    data, err = load_min_durations()
    MIN_DURATIONS = data
    if not any_video:
        return
    if data is None:
        _stop(f"В списке ссылок есть видео, но min_durations.json не загружен: {err}. "
              f"Укажите путь в INPUT_MIN_DURATIONS или положите min_durations.json "
              f"рядом с links.txt (артефакт search-links).")
    if shutil.which("ffprobe") is None:
        _stop("В списке ссылок есть видео, но ffprobe не найден в PATH: "
              "установите ffmpeg (например: sudo apt-get install -y ffmpeg).")


def _is_video_file(path: str) -> bool:
    kind = kind_of_ext(ext_from_name(path))
    if kind is None:
        try:
            with open(path, "rb") as f:
                head = f.read(16)
        except OSError:
            return False
        kind = kind_of_ext(sniff_format(head) or "")
    return kind == "video"


def verify_downloaded_video(number: int) -> tuple[bool, str]:
    """Проверяет скачанные видео номера (фото пропускает, ffprobe для них не запускается).
    Короткое/непроверяемое видео удаляется. (True, "") если всё в порядке или видео нет."""
    paths = sorted(p for p in glob.glob(os.path.join(OUTPUT_DIR, f"{number}.*"))
                   if not p.endswith((".part", ".ytdl", ".temp")))
    for path in paths:
        if not _is_video_file(path):
            continue
        ok, reason = check_video_duration(number, path, MIN_DURATIONS)
        DURATION_STATS.add_check(number, ok)
        if not ok:
            for p in paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
            return False, reason
        print(f"[ИНФО] {number}: длительность видео подходит "
              f"(нужно {MIN_DURATIONS[number]} с)")
    return True, ""


def _attempt_download(number: int, url: str) -> tuple[bool, str]:
    """Один вызов существующего download_media_item (без изменений внутри
    него) - возвращает (успех, текст_ошибки). Текст ошибки берётся из
    последней записи FAILED_ITEMS с префиксом "{number}: ", появившейся за
    время этого вызова."""
    prefix = f"{number}: "
    with _failed_items_lock:
        before = sum(1 for item in FAILED_ITEMS if item.startswith(prefix))
    download_media_item(number, url)
    with _failed_items_lock:
        entries = [item for item in FAILED_ITEMS if item.startswith(prefix)]
    if len(entries) > before:
        return False, entries[-1][len(prefix):]
    ok, reason = verify_downloaded_video(number)
    if not ok:
        fail(number, reason)
        return False, reason
    SOURCE_STATS.add(url)
    return True, ""


def _clear_failed_entries_for(number: int) -> None:
    """Убирает из FAILED_ITEMS промежуточные записи об этом number (после
    неудачного retry/перед переходом на backup) - в конце для этого number
    должна остаться максимум ОДНА итоговая запись (если он так и не скачался),
    а не по одной на каждую промежуточную попытку. Безопасно при параллельных
    потоках - см. комментарий в начале блока: с этим номером в это время
    работает только вызвавший поток."""
    prefix = f"{number}: "
    with _failed_items_lock:
        FAILED_ITEMS[:] = [item for item in FAILED_ITEMS if not item.startswith(prefix)]


def download_number_with_backup(number: int, primary_url: str, backup_url: str = "") -> None:
    """Верхнеуровневая обёртка над download_media_item для одного сегмента:

    1. Пробуем primary_url.
    2. Если провал И маршрут без своей ретрай-логики (_url_gets_external_retry)
       И сбой классифицирован как "transient" - делаем РОВНО ОДИН быстрый
       повторный вызов той же primary_url (без долгого бэкоффа - сразу).
    3. Если всё ещё провал - при наличии backup_url пробуем её (та же логика
       диспетчеризации по сайту, что и для primary - LOC/Wikimedia/yt-dlp
       маршрут для backup отработает своей внутренней логикой точно так же).
    4. Если и backup не спас (или его не было) - одна итоговая запись в
       FAILED_ITEMS через fail(), как раньше делал сам download_media_item.

    Раздельная пометка в лог при успехе через backup - сама site-функция не в
    курсе, что её вызвали как backup (её "[OK] ..." не трогаем), поэтому
    отдельная строка печатается здесь."""
    ok, message = _attempt_download(number, primary_url)

    if not ok and _url_gets_external_retry(primary_url):
        if classify_failure_message(message) == "transient":
            print(
                f"[ИНФО] {number}: похоже на транзиентный сбой соединения "
                f"({message}) - один быстрый повтор той же ссылки перед backup..."
            )
            _clear_failed_entries_for(number)
            ok, message = _attempt_download(number, primary_url)

    if ok:
        return

    if backup_url:
        _clear_failed_entries_for(number)
        print(f"[ИНФО] {number}: primary-ссылка не скачалась ({message}) - пробую backup-ссылку...")
        ok, backup_message = _attempt_download(number, backup_url)
        if ok:
            print(f"[OK] {number}: файл успешно скачан [через backup] ({backup_url})")
            DURATION_STATS.note_backup_success(number)
            return
        message = f"primary не скачался ({message}); backup тоже не скачался ({backup_message})"

    _clear_failed_entries_for(number)
    fail(number, message)


def download_wikimedia_commons(number: int, url: str) -> None:
    """Скачивание оригинального файла с Wikimedia Commons с паузами от лимита 429.

    Раунд 6 - разбор реального 429 на прогоне с ThreadPoolExecutor: раньше
    _wikimedia_rate_limiter.wait_turn() вызывался РОВНО ОДИН РАЗ, перед первой
    попыткой metadata-запроса - мимо него шли ретраи (свой time.sleep + новый
    запрос без wait_turn) и целиком второй запрос (сам файл, upload.wikimedia.org)
    вместе с его собственным retry-циклом. Т.е. лимитер видел лишь часть
    реального трафика. Плюс официальная robot policy Wikimedia
    (https://wikitech.wikimedia.org/wiki/Robot_policy) задаёт лимит для анонимов
    в первую очередь как КОНКУРЕНТНОСТЬ (не больше 2 одновременных скачиваний
    файла), а не только частоту стартов - для этого добавлен _wikimedia_semaphore,
    оборачивающий всю функцию целиком (оба запроса на один номер - одна "единица"
    конкурентности). Теперь: 1) семафор ограничивает, сколько номеров вообще
    одновременно работают с Wikimedia (2, без запаса - официальный потолок, а не
    оценка), 2) wait_turn() вызывается перед КАЖДЫМ фактическим HTTP-запросом
    (обе retry-петли и fallback-запрос страницы), а не только один раз в начале.

    Раунд 10 (Этап 2) — честный учет заголовка Retry-After: при получении 429 от
    Wikimedia время сна вычисляется через _extract_retry_after_seconds как
    max(Retry-After + 1.0, backoff), предотвращая преждевременное пробуждение
    потоков до истечения окна блокировки."""
    with _wikimedia_semaphore:
        try:
            path = urlparse(url).path
            file_part = path.split("/wiki/")[-1] if "/wiki/" in path else path.split("/")[-1]
            file_title = urllib.parse.unquote(file_part)

            if not file_title.lower().startswith(("file:", "файл:")):
                file_title = "File:" + file_title

            api_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(file_title)}&prop=imageinfo&iiprop=url&format=json"

            data = None
            for attempt in range(3):
                _wikimedia_rate_limiter.wait_turn()
                resp = cffi_requests.get(api_url, headers=WIKIMEDIA_HEADERS, impersonate="chrome", timeout=20)
                if resp.status_code == 429:
                    log_429_details("Wikimedia", f"metadata, номер {number}, попытка {attempt + 1}", resp)
                    sleep_sec = _extract_retry_after_seconds(resp, float(4 * (attempt + 1)))
                    print(f"[ИНФО] Wikimedia 429 limit, пауза {sleep_sec:.1f} сек для {number}...")
                    time.sleep(sleep_sec)
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
                _wikimedia_rate_limiter.wait_turn()
                page_resp = cffi_requests.get(url, headers=WIKIMEDIA_HEADERS, impersonate="chrome", timeout=30)
                if page_resp.status_code == 429:
                    log_429_details("Wikimedia", f"page_fallback, номер {number}", page_resp)
                    sleep_sec = _extract_retry_after_seconds(page_resp, 5.0)
                    print(f"[ИНФО] Wikimedia 429 limit (page), пауза {sleep_sec:.1f} сек для {number}...")
                    time.sleep(sleep_sec)
                    _wikimedia_rate_limiter.wait_turn()
                    page_resp = cffi_requests.get(url, headers=WIKIMEDIA_HEADERS, impersonate="chrome", timeout=30)
                page_resp.raise_for_status()
                direct_url = extract_og_media_url(page_resp.text)

            if not direct_url:
                fail(number, f"Wikimedia {number}: не удалось извлечь ссылку ({url})")
                return

            # Скачивание файла с повторами
            content = None
            img_resp = None
            for attempt in range(3):
                _wikimedia_rate_limiter.wait_turn()
                img_resp = cffi_requests.get(direct_url, headers=WIKIMEDIA_HEADERS, impersonate="chrome", timeout=60)
                if img_resp.status_code == 429:
                    log_429_details("Wikimedia", f"файл, номер {number}, попытка {attempt + 1}", img_resp)
                    sleep_sec = _extract_retry_after_seconds(img_resp, float(4 * (attempt + 1)))
                    print(f"[ИНФО] Wikimedia 429 limit (файл), пауза {sleep_sec:.1f} сек для {number}...")
                    time.sleep(sleep_sec)
                    continue
                img_resp.raise_for_status()
                content = img_resp.content
                break

            if not content or looks_like_html(content):
                fail(number, f"Wikimedia {number}: не удалось скачать файл изображения")
                return

            save_media(number, content, direct_url, img_resp.headers.get("Content-Type", ""), "wikimedia")
        except Exception as e:
            fail(number, f"Wikimedia ошибка {number}: {e}")


def download_direct_via_cffi(number: int, url: str) -> None:
    site = "nasa" if "nasa.gov" in url.lower() else "generic"
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
                    save_media(number, media_content, media_url,
                               media_resp.headers.get("Content-Type", ""), site)
                    return

            fail(number, f"{number}: страница не содержит медиафайла ({url})")
            return

        content_type = resp.headers.get("Content-Type", "")
        save_media(number, content, url, content_type, site)
    except Exception as e:
        fail(number, f"Не удалось скачать {number}: {e}")


def download_pixabay_photo(number: int, photo_id: str) -> None:
    # Без декоратора _pexels_pixabay_throttled: окно/пауза - вне семафора (см. pixabay_api_get)
    api_url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&id={photo_id}"
    try:
        resp = pixabay_api_get(number, api_url, "фото")
        data = resp.json()
        hits = data.get("hits", [])
        if hits:
            direct_url = hits[0].get("largeImageURL") or hits[0].get("imageURL")
            if direct_url:
                with _pexels_pixabay_semaphore:
                    img_resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=30)
                    img_resp.raise_for_status()
                    save_media(number, img_resp.content, direct_url,
                               img_resp.headers.get("Content-Type", ""), "pixabay")
                return
        fail(number, f"Не удалось получить фото Pixabay {number}")
    except Exception as e:
        fail(number, f"Pixabay API ошибка {number}: {e}")


def download_pixabay_video(number: int, video_id: str) -> None:
    # Без декоратора _pexels_pixabay_throttled: окно/пауза - вне семафора (см. pixabay_api_get)
    api_url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&id={video_id}"
    try:
        resp = pixabay_api_get(number, api_url, "видео")
        data = resp.json()
        hits = data.get("hits", [])
        if hits:
            videos = hits[0].get("videos", {})
            best_video = videos.get("large") or videos.get("medium") or videos.get("small")
            if best_video and "url" in best_video:
                direct_url = best_video["url"]
                with _pexels_pixabay_semaphore:
                    vid_resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=60)
                    vid_resp.raise_for_status()
                    save_media(number, vid_resp.content, direct_url,
                               vid_resp.headers.get("Content-Type", ""), "pixabay")
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
            save_media(number, img_resp.content, direct_url,
                       img_resp.headers.get("Content-Type", ""), "pexels")
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

    mp4_files = [f for f in video_files if f.get("file_type") == "video/mp4" and f.get("link")]
    if not mp4_files:
        fail(number, f"Pexels {number}: в video_files нет video/mp4")
        return
    best = max(mp4_files, key=lambda f: f.get("width") or 0)
    direct_url = best["link"]

    try:
        resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=60)
        resp.raise_for_status()
        save_media(number, resp.content, direct_url, resp.headers.get("Content-Type", ""), "pexels")
    except Exception as e:
        fail(number, f"Не удалось скачать файл видео {number}: {e}")


def download_direct_media(number: int, url: str) -> None:
    """Прямая ссылка на файл Pexels/Pixabay (см. DIRECT_MEDIA_DOMAINS): один GET по URL,
    без API и без ключа. 429 -> повтор с учётом Retry-After (до DIRECT_DOWNLOAD_ATTEMPTS
    попыток; Retry-After > DIRECT_MAX_RETRY_WAIT_SECONDS - не ждём). Формат - по
    содержимому, имя N.<ext> - через save_media. Семафор держится только на время
    самого запроса, но не на паузе - новых ожиданий под семафором нет. Откат на API
    не делается: провал = обычный провал primary (дальше backup в download_number_with_backup)."""
    host = (urlparse(url).hostname or "").lower()
    site = "pexels" if "pexels" in host else "pixabay"
    timeout = 60 if _url_looks_video(url) else 30
    try:
        resp = None
        for attempt in range(DIRECT_DOWNLOAD_ATTEMPTS):
            with _pexels_pixabay_semaphore:
                resp = cffi_requests.get(url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=timeout)
            if resp.status_code != 429:
                break
            log_429_details(site.capitalize(), f"прямая ссылка, номер {number}, попытка {attempt + 1}", resp)
            if attempt == DIRECT_DOWNLOAD_ATTEMPTS - 1:
                break  # последняя попытка: спать уже незачем
            sleep_sec = _extract_retry_after_seconds(resp, float(4 * (attempt + 1)))
            if sleep_sec > DIRECT_MAX_RETRY_WAIT_SECONDS:
                fail(number, f"Прямая ссылка {site} {number}: 429, Retry-After слишком большой "
                             f"({sleep_sec:.0f} с) ({url})")
                return
            print(f"[ИНФО] {site.capitalize()} 429 limit (прямая ссылка), пауза {sleep_sec:.1f} сек для {number}...")
            time.sleep(sleep_sec)
        if resp.status_code == 429:
            fail(number, f"Прямая ссылка {site} {number}: лимит запросов 429 не сбросился ({url})")
            return
        resp.raise_for_status()
        save_media(number, resp.content, url, resp.headers.get("Content-Type", ""), site)
    except Exception as e:
        fail(number, f"Не удалось скачать по прямой ссылке {site} {number}: {e}")


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

        save_media(number, content, direct_url, vid_resp.headers.get("Content-Type", ""), "coverr")
    except Exception as e:
        fail(number, f"Coverr ошибка {number}: {e}")


def _loc_num(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _loc_is_photo(entry: dict) -> bool:
    """Файл LOC годен, если и расширение URL, и mimetype (когда они есть) - jpg/png."""
    ext = ext_from_name(entry.get("url") or "")
    mime = str(entry.get("mimetype") or "").lower().strip()
    if not ext and not mime:
        return False
    if ext and not is_allowed_ext(ext, "photo"):
        return False
    if mime and mime not in ("image/jpeg", "image/jpg", "image/png"):
        return False
    return True


def download_loc_gov(number: int, url: str) -> None:
    global _loc_exhausted
    parsed = urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    query["fo"] = ["json"]
    json_url = parsed._replace(query=urllib.parse.urlencode(query, doseq=True)).geturl()

    try:
        data = _loc_get_json(json_url, headers=BROWSER_HEADERS)

        direct_url = None
        resource = data.get("resource", {}) or {}
        candidates = []
        if isinstance(resource.get("files"), list):
            for file_group in resource["files"]:
                if isinstance(file_group, list):
                    candidates += [f for f in file_group if isinstance(f, dict) and _loc_is_photo(f)]
        if candidates:
            best = max(candidates, key=lambda f: (_loc_num(f.get("height")), _loc_num(f.get("width")),
                                                  _loc_num(f.get("size"))))
            direct_url = best["url"]
        else:
            item = data.get("item", {}) or {}
            image_url = item.get("image_url")
            urls = image_url if isinstance(image_url, list) else ([image_url] if isinstance(image_url, str) else [])
            urls = [u for u in urls if isinstance(u, str) and is_allowed_ext(ext_from_name(u), "photo")]
            if urls:
                direct_url = urls[-1]  # в image_url последний элемент - самый большой

        if not direct_url:
            fail(number, "loc.gov: нет jpg/png")
            return

        file_resp = _loc_get(direct_url, headers=BROWSER_HEADERS, timeout=60, is_file_request=True)
        content = file_resp.content

        if looks_like_html(content):
            # Проверяем, не скачалась ли капча Cloudflare вместо файла
            if any(m in content.lower() for m in (b"just a moment...", b"cf-chl", b"cloudflare", b"captcha")):
                log_429_details("LOC", f"file CAPTCHA ({direct_url})", file_resp)
                with _loc_lock:
                    _loc_exhausted = True
                raise LocExhaustedError("HTTP 429 / CAPTCHA при скачивании файла LOC - LOC помечен исчерпанным до конца запуска")
            fail(number, f"loc.gov {number}: похоже, скачалась HTML-страница")
            return

        save_media(number, content, direct_url, file_resp.headers.get("Content-Type", ""), "loc.gov")
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
            "--format", "b[ext=mp4]/mp4",
            "--no-check-certificate",
            "--retries", "3",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            fail(number, f"Ошибка yt-dlp {number}: {result.stderr}")
            return
        produced = [p for p in glob.glob(os.path.join(OUTPUT_DIR, f"{number}.*"))
                    if not p.endswith((".part", ".ytdl", ".temp"))]
        if not produced:
            fail(number, f"yt-dlp {number}: файл не появился")
            return
        path = produced[0]
        with open(path, "rb") as f:
            head = f.read(16)
        fmt = sniff_format(head)
        if fmt is None:
            desc = describe_rejected(head)
            for p in glob.glob(os.path.join(OUTPUT_DIR, f"{number}.*")):
                os.remove(p)
            fail(number, f"неподдерживаемый формат: {desc}")
            FORMAT_REJECTS.add("yt-dlp", desc)
            return
        final = os.path.join(OUTPUT_DIR, f"{number}.{fmt}")
        if path != final:
            os.replace(path, final)
        print(f"[OK] ВИДЕО {number} ({number}.{fmt}) успешно скачано через yt-dlp")
    except Exception as ytdl_err:
        fail(number, f"Не удалось запустить yt-dlp для {number}: {ytdl_err}")


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


def _selftest() -> None:
    """python download.py --selftest - проверка чистых частей без сети и без ffprobe."""
    global MIN_DURATIONS, _FFPROBE_RUNNER, OUTPUT_DIR
    import tempfile

    # разбор вывода ffprobe
    assert parse_ffprobe_duration("12.345000\n") == 12.345
    assert parse_ffprobe_duration("5") == 5.0
    assert parse_ffprobe_duration("N/A\n") is None
    assert parse_ffprobe_duration("") is None
    assert parse_ffprobe_duration("  \n") is None
    assert parse_ffprobe_duration("garbage") is None
    assert parse_ffprobe_duration("nan") is None and parse_ffprobe_duration("inf") is None
    assert parse_ffprobe_duration("0") is None and parse_ffprobe_duration("-3") is None

    # строгое сравнение
    assert duration_ok(5.0, 5.0)
    assert duration_ok(5.001, 5.0)
    assert not duration_ok(4.999, 5.0)
    assert not duration_ok(5.0 - 1e-9, 5.0)

    # разбор min_durations.json
    assert parse_min_durations('{"1": 3.5, "12": 4}') == {1: 3.5, 12: 4.0}
    for bad in ("", "not json", "[1,2]", '{"a": 1}', '{"1": "x"}', '{"1": true}',
                '{"1": -1}', '{"1": null}', '{"1": NaN}'):
        try:
            parse_min_durations(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"должно быть ValueError: {bad!r}")

    # measure_duration с подменой запуска ffprobe
    def fake(outputs):
        calls = []
        def runner(args):
            calls.append(args)
            r = outputs[len(calls) - 1]
            if isinstance(r, Exception):
                raise r
            return r
        return runner, calls

    saved = _FFPROBE_RUNNER
    try:
        runner, calls = fake(["7.5\n"])
        _FFPROBE_RUNNER = runner
        assert measure_duration("x.mp4") == (7.5, "") and len(calls) == 1
        assert "stream=duration" in calls[0] and "v:0" in calls[0]

        runner, calls = fake(["N/A\n", "6.0\n"])  # поток N/A -> контейнер
        _FFPROBE_RUNNER = runner
        assert measure_duration("x.mp4") == (6.0, "") and len(calls) == 2
        assert "format=duration" in calls[1]

        runner, calls = fake(["\n", "N/A"])
        _FFPROBE_RUNNER = runner
        v, why = measure_duration("x.mp4")
        assert v is None and "ни у видеопотока, ни у контейнера" in why

        for exc, text in ((subprocess.TimeoutExpired("ffprobe", 30), "таймаут"),
                          (FileNotFoundError(), "не найден"),
                          (PermissionError("x"), "не удалось запустить"),
                          (FfprobeError("код 1"), "ошибкой")):
            runner, _ = fake([exc])
            _FFPROBE_RUNNER = runner
            v, why = measure_duration("x.mp4")
            assert v is None and text in why, (exc, why)

        # check_video_duration: строго, нет записи, нет файла данных
        runner, _ = fake(["5.0\n"])
        _FFPROBE_RUNNER = runner
        assert check_video_duration(1, "x.mp4", {1: 5.0}) == (True, "")
        runner, _ = fake(["4.999\n"])
        _FFPROBE_RUNNER = runner
        ok, why = check_video_duration(1, "x.mp4", {1: 5.0})
        assert not ok and why == "видео короче нужного: 4.999 с < 5.0 с", why
        ok, why = check_video_duration(2, "x.mp4", {1: 5.0})
        assert not ok and "нет записи" in why
        ok, why = check_video_duration(1, "x.mp4", None)
        assert not ok and "не загружен" in why

        # verify_downloaded_video: фото не проверяется, короткое видео удаляется
        with tempfile.TemporaryDirectory() as tmp:
            OUTPUT_DIR = tmp
            MIN_DURATIONS = {1: 5.0, 2: 5.0}
            open(os.path.join(tmp, "1.jpg"), "wb").write(b"\xff\xd8\xff" + b"\0" * 20)
            runner, calls = fake([])
            _FFPROBE_RUNNER = runner
            assert verify_downloaded_video(1) == (True, "") and not calls  # ffprobe не вызван
            open(os.path.join(tmp, "2.mp4"), "wb").write(b"x")
            runner, _ = fake(["4.5\n"])
            _FFPROBE_RUNNER = runner
            ok, why = verify_downloaded_video(2)
            assert not ok and "короче" in why and not os.path.exists(os.path.join(tmp, "2.mp4"))
    finally:
        _FFPROBE_RUNNER = saved
        MIN_DURATIONS = None

    # счётчики
    st = DurationStats()
    st.add_check(1, False)
    st.add_check(2, True)
    st.note_backup_success(1)
    st.note_backup_success(2)  # 2 не отсеивался - не считается
    assert (st.checked, st.rejected, st.replaced) == (2, 1, 1)
    assert st.summary_line().endswith("проверено видео 2, отсеяно по длительности 1, заменено backup 1")

    # эвристика видео по URL
    assert _url_looks_video("https://x.org/a/clip.MP4?x=1")
    assert _url_looks_video("https://www.pexels.com/video/foo-123/")
    assert not _url_looks_video("https://x.org/a/photo.jpg")

    # прямые ссылки Pexels/Pixabay: определение по hostname
    for u in ("https://images.pexels.com/photos/1/pexels-photo-1.jpeg?auto=compress",
              "https://videos.pexels.com/video-files/1/1-hd.mp4",
              "https://cdn.pixabay.com/photo/2020/01/01/a.jpg",
              "https://cdn.pixabay.com/video/2020/01/01/clip.mp4",
              "HTTPS://Images.Pexels.COM/photos/1/a.png"):
        assert is_direct_media_url(u), u
        assert classify_pexels_pixabay_source(u) == "direct", u
    for u in ("https://www.pexels.com/photo/foo-123/", "https://www.pexels.com/video/foo-123/",
              "https://pixabay.com/photos/foo-123/", "https://pixabay.com/videos/foo-123/",
              "https://api.pexels.com/v1/photos/1", "https://commons.wikimedia.org/wiki/File:A.jpg",
              "https://images.pexels.com.evil.com/a.jpg", "https://evil.com/images.pexels.com/a.jpg",
              "https://evil.com/?u=cdn.pixabay.com", "", "not a url"):
        assert not is_direct_media_url(u), u
    assert classify_pexels_pixabay_source("https://www.pexels.com/photo/foo-123/") == "api"
    assert classify_pexels_pixabay_source("https://www.pexels.com/video/foo-123/") == "api"
    assert classify_pexels_pixabay_source("https://pixabay.com/videos/foo-123/") == "api"
    assert classify_pexels_pixabay_source("https://pixabay.com/photos/foo-123/") == "api"
    assert classify_pexels_pixabay_source("https://www.pexels.com/@user/") is None  # диспетчер отдаст в generic
    assert classify_pexels_pixabay_source("https://x.org/a.jpg") is None
    assert _url_gets_external_retry("https://videos.pexels.com/video-files/1/1-hd.mp4")  # не yt-dlp-ветка
    assert not _url_gets_external_retry("https://x.org/a.mp4")  # прежнее поведение

    st = SourceStats()
    assert st.summary_line() == "Скачано по прямой ссылке: 0, через API: 0"
    st.add("https://images.pexels.com/a.jpg")
    st.add("https://cdn.pixabay.com/a.jpg")
    st.add("https://www.pexels.com/photo/x-1/")
    st.add("https://commons.wikimedia.org/wiki/File:A.jpg")  # не считается
    assert st.summary_line() == "Скачано по прямой ссылке: 2, через API: 1"

    # скачивание по прямой ссылке, сеть заглушена (cffi_requests.get, time.sleep)
    class _Resp:
        def __init__(self, status=200, content=b"", headers=None):
            self.status_code, self.content, self.headers = status, content, headers or {}
            self.text = ""

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP Error {self.status_code}")

    jpg = b"\xff\xd8\xff\xe0" + b"\0" * 32
    png = b"\x89PNG\r\n\x1a\n" + b"\0" * 32
    global SOURCE_STATS
    saved_get, saved_sleep = cffi_requests.get, time.sleep
    saved_out, saved_stats = OUTPUT_DIR, SOURCE_STATS
    saved_keys = (globals()["PEXELS_API_KEY"], globals()["PIXABAY_API_KEY"])
    saved_failed = FAILED_ITEMS[:]
    saved_md = MIN_DURATIONS
    try:
        globals()["PEXELS_API_KEY"] = globals()["PIXABAY_API_KEY"] = ""  # ключ не нужен
        with tempfile.TemporaryDirectory() as tmp:
            OUTPUT_DIR = tmp
            MIN_DURATIONS = {}
            script, urls, sleeps = [], [], []

            def fake_get(url, **kw):
                urls.append(url)
                r = script.pop(0)
                if isinstance(r, Exception):
                    raise r
                return r

            cffi_requests.get = fake_get
            time.sleep = lambda sec: sleeps.append(sec)

            def reset(*responses):
                script[:] = list(responses)
                urls.clear()
                sleeps.clear()
                FAILED_ITEMS.clear()

            def fails(n):
                return [x for x in FAILED_ITEMS if x.startswith(f"{n}: ")]

            # успех: один GET по самому URL, без ключа, имя по содержимому
            reset(_Resp(200, jpg))
            u1 = "https://images.pexels.com/photos/1/pexels-photo-1.jpeg?auto=compress"
            download_media_item(7, u1)
            assert urls == [u1] and not fails(7) and os.path.exists(os.path.join(tmp, "7.jpg"))

            # формат по содержимому: расширение в URL не совпадает -> отказ, файла нет
            reset(_Resp(200, png))
            download_media_item(8, "https://cdn.pixabay.com/photo/a.jpg")
            assert fails(8) and "неподдерживаемый формат" in fails(8)[0]
            assert not glob.glob(os.path.join(tmp, "8.*"))

            # 429 с Retry-After: пауза max(RA+1, 4) и повтор
            reset(_Resp(429, headers={"Retry-After": "10"}), _Resp(200, jpg))
            download_media_item(9, "https://images.pexels.com/photos/9/a.jpg")
            assert sleeps == [11.0] and len(urls) == 2 and not fails(9)
            assert os.path.exists(os.path.join(tmp, "9.jpg"))

            # 429 на всех попытках: провал с "429" (внешнего повтора нет), после последней - без сна
            reset(*[_Resp(429) for _ in range(DIRECT_DOWNLOAD_ATTEMPTS)])
            download_media_item(10, "https://images.pexels.com/photos/10/a.jpg")
            assert len(urls) == DIRECT_DOWNLOAD_ATTEMPTS and len(sleeps) == DIRECT_DOWNLOAD_ATTEMPTS - 1
            assert fails(10) and classify_failure_message(fails(10)[-1]) == "429"

            # огромный Retry-After: не ждём
            reset(_Resp(429, headers={"Retry-After": "99999"}))
            download_media_item(11, "https://images.pexels.com/photos/11/a.jpg")
            assert not sleeps and len(urls) == 1 and fails(11)

            # провал primary (404) -> backup (тоже прямая ссылка); запросов к API нет
            SOURCE_STATS = SourceStats()
            reset(_Resp(404), _Resp(200, jpg))
            p, b = "https://images.pexels.com/photos/12/a.jpg", "https://cdn.pixabay.com/photo/b.jpg"
            download_number_with_backup(12, p, b)
            assert urls == [p, b] and not fails(12) and os.path.exists(os.path.join(tmp, "12.jpg"))
            assert (SOURCE_STATS.direct, SOURCE_STATS.api) == (1, 0)

            # провал primary и backup: одна итоговая запись, откат на API не делается
            reset(_Resp(404), _Resp(404))
            download_number_with_backup(13, p, b)
            assert urls == [p, b] and len(fails(13)) == 1 and "backup тоже не скачался" in fails(13)[0]
            assert (SOURCE_STATS.direct, SOURCE_STATS.api) == (1, 0)
            assert all(is_direct_media_url(x) for x in urls)
    finally:
        cffi_requests.get, time.sleep = saved_get, saved_sleep
        OUTPUT_DIR, SOURCE_STATS, MIN_DURATIONS = saved_out, saved_stats, saved_md
        globals()["PEXELS_API_KEY"], globals()["PIXABAY_API_KEY"] = saved_keys
        FAILED_ITEMS[:] = saved_failed

    # ---- лимит Pixabay API: окно, общая пауза, 429 (время и сон подменены, HTTP заглушен) ----
    assert load_pixabay_rpm({}) == 90 and load_pixabay_rpm({"PIXABAY_REQUESTS_PER_MINUTE": ""}) == 90
    assert load_pixabay_rpm({"PIXABAY_REQUESTS_PER_MINUTE": "   "}) == 90
    assert load_pixabay_rpm({"PIXABAY_REQUESTS_PER_MINUTE": " 50 "}) == 50
    for _bad in ("0", "-5", "abc", "1.5", "90 запросов"):
        try:
            load_pixabay_rpm({"PIXABAY_REQUESTS_PER_MINUTE": _bad})
        except ValueError as e:
            assert "PIXABAY_REQUESTS_PER_MINUTE" in str(e), e
        else:
            raise AssertionError(f"PIXABAY_REQUESTS_PER_MINUTE={_bad!r} должно останавливать")

    assert pixabay_reset_pause("5")[0] == 6.0 and pixabay_reset_pause("0")[0] == 1.0
    assert pixabay_reset_pause("600")[0] == 601.0 and pixabay_reset_pause("2.5")[0] == 3.5
    for _raw in (None, "", "abc", "9999", "601", "-1", "nan", "inf"):
        assert pixabay_reset_pause(_raw)[0] == 61.0, _raw

    class _Clk:
        def __init__(self):
            self.t = 1000.0

        def __call__(self):
            return self.t

    _ck = _Clk()
    _sl: list = []

    def _fsleep(sec):
        _sl.append(sec)
        _ck.t += sec

    _w = ThreadSlidingWindow(90, 60.0, now_fn=_ck, sleep_fn=_fsleep)
    assert [_w.acquire() for _ in range(90)] == [False] * 90 and _sl == []   # 90 проходят сразу
    assert _w.acquire() is True and _sl == [60.0]                           # 91-й ждёт 60 с
    assert _w.acquire() is False and _sl == [60.0]                          # окно освободилось
    for _args in ((0, 60.0), (-1, 60.0), (1.5, 60.0), (True, 60.0), (5, 0), (5, -1.0)):
        try:
            ThreadSlidingWindow(*_args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ThreadSlidingWindow{_args} должен падать")
    # частичное освобождение: ждём только до выхода ПЕРВОГО места из окна
    _ck2 = _Clk(); _sl2: list = []
    _w2 = ThreadSlidingWindow(2, 10.0, now_fn=_ck2, sleep_fn=lambda sec: (_sl2.append(sec), setattr(_ck2, "t", _ck2.t + sec)))
    _w2.acquire(); _ck2.t += 4; _w2.acquire()
    assert _w2.acquire() is True and _sl2 == [6.0], _sl2
    # потокобезопасность: 8 потоков x 10 взятий в окне на 80 - все проходят без сна, ровно 80 записей
    _ck3 = _Clk(); _sl3: list = []
    _w3 = ThreadSlidingWindow(80, 60.0, now_fn=_ck3, sleep_fn=_sl3.append)
    _ths = [threading.Thread(target=lambda: [_w3.acquire() for _ in range(10)]) for _ in range(8)]
    for _t in _ths:
        _t.start()
    for _t in _ths:
        _t.join()
    assert _sl3 == [] and len(_w3._stamps) == 80

    _ck4 = _Clk(); _sl4: list = []
    _sp = SharedPause(now_fn=_ck4, sleep_fn=lambda sec: (_sl4.append(sec), setattr(_ck4, "t", _ck4.t + sec)))
    assert _sp.wait() == 0.0 and _sl4 == []
    _sp.extend(5.0); _sp.extend(3.0)   # берётся максимум, а не сумма
    assert _sp.wait() == 5.0 and _sl4 == [5.0] and _sp.wait() == 0.0

    # --- download_pixabay_*: окно/429/семафор (cffi_requests.get заглушен) ---
    class _JResp(_Resp):
        def __init__(self, status=200, content=b"", headers=None, body=None):
            super().__init__(status, content, headers)
            self._body = body

        def json(self):
            return self._body

    _OK_PHOTO = {"hits": [{"largeImageURL": "https://cdn.pixabay.com/photo/a.jpg"}]}
    _saved_get2 = cffi_requests.get
    _saved_state2 = (globals()["PIXABAY_API_KEY"], OUTPUT_DIR, FAILED_ITEMS[:], _pixabay_limiter,
                     os.environ.get("PIXABAY_REQUESTS_PER_MINUTE"), sys.argv[:])
    try:
        globals()["PIXABAY_API_KEY"] = "K"
        with tempfile.TemporaryDirectory() as tmp:
            OUTPUT_DIR = tmp

            def _setup(script, rpm=90):
                clk, sleeps, sem_free, calls, held = _Clk(), [], [], [], []

                def _probe():
                    got = [_pexels_pixabay_semaphore.acquire(blocking=False) for _ in range(PEXELS_PIXABAY_CONCURRENCY)]
                    for g in got:
                        if g:
                            _pexels_pixabay_semaphore.release()
                    return all(got)  # True - семафор целиком свободен

                def sl(sec):
                    sem_free.append(_probe())
                    sleeps.append(sec)
                    clk.t += sec

                lim = PixabayApiLimiter(rpm, now_fn=clk, sleep_fn=sl)
                set_pixabay_limiter(lim)

                def fake_get(url, **kw):
                    calls.append(url)
                    held.append(not _probe())  # True - текущий поток держит семафор во время запроса
                    r = script.pop(0)
                    if isinstance(r, Exception):
                        raise r
                    return r

                cffi_requests.get = fake_get
                FAILED_ITEMS.clear()
                return lim, sleeps, sem_free, calls, held

            def _fails(n):
                return [x for x in FAILED_ITEMS if x.startswith(f"{n}: ")]

            # 429 с Reset: 5 -> пауза 6 с, повтор проходит; во сне семафор свободен, в запросе - занят
            lim, sleeps, sem_free, calls, held = _setup([
                _JResp(429, headers={"X-RateLimit-Reset": "5"}), _JResp(200, body=_OK_PHOTO), _Resp(200, jpg)])
            download_pixabay_photo(31, "123")
            assert sleeps == [6.0] and sem_free == [True] and not _fails(31), (sleeps, sem_free, FAILED_ITEMS)
            assert len(calls) == 3 and calls[0] == calls[1] and "pixabay.com/api/?key=K&id=123" in calls[0]
            assert held == [True, True, True], held
            assert os.path.exists(os.path.join(tmp, "31.jpg"))
            assert lim.stats.summary_line() == (
                "Pixabay API: запросов 2, получено 429: 1, повторов: 1, ожиданий места в окне: 0")

            # три 429 подряд и успех на 4-м запросе (3 повтора - допустимо)
            lim, sleeps, sem_free, calls, held = _setup(
                [_JResp(429, headers={"X-RateLimit-Reset": "1"})] * 3 + [_JResp(200, body=_OK_PHOTO), _Resp(200, jpg)])
            download_pixabay_photo(32, "5")
            assert sleeps == [2.0, 2.0, 2.0] and not _fails(32) and all(sem_free)
            assert (lim.stats.requests, lim.stats.rate_limited, lim.stats.retries) == (4, 3, 3)

            # 429 на всех 4 запросах (запрос + 3 повтора): ошибка со словом "429", сразу в backup
            lim, sleeps, sem_free, calls, held = _setup([_JResp(429, headers={"X-RateLimit-Reset": "1"})] * 4)
            download_pixabay_photo(33, "5")
            assert len(calls) == 4 and sleeps == [2.0, 2.0, 2.0] and all(sem_free)
            assert len(_fails(33)) == 1 and "429" in _fails(33)[0], FAILED_ITEMS
            assert classify_failure_message(_fails(33)[0][len("33: "):]) == "429"
            assert (lim.stats.requests, lim.stats.rate_limited, lim.stats.retries) == (4, 4, 3)

            # 429 без заголовка / с мусором (abc, 9999): пауза 61 с
            for _num, _hdr in ((34, {}), (35, {"X-RateLimit-Reset": "abc"}), (36, {"X-RateLimit-Reset": "9999"})):
                lim, sleeps, sem_free, calls, held = _setup(
                    [_JResp(429, headers=_hdr), _JResp(200, body=_OK_PHOTO), _Resp(200, jpg)])
                download_pixabay_photo(_num, "7")
                assert sleeps == [61.0] and not _fails(_num) and sem_free == [True], (_num, sleeps)

            # огромная пауза (Reset=300 -> 301 с > 120): немедленная ошибка без сна и без общей паузы
            lim, sleeps, sem_free, calls, held = _setup([_JResp(429, headers={"X-RateLimit-Reset": "300"})])
            download_pixabay_photo(37, "7")
            assert sleeps == [] and len(calls) == 1 and lim.pause.wait() == 0.0 and sleeps == []
            assert len(_fails(37)) == 1 and "429" in _fails(37)[0] and "не жду" in _fails(37)[0]
            assert lim.stats.retries == 0

            # общая пауза: после 429 в одном потоке следующий запрос (в т.ч. другого номера) ждёт её конца
            lim, sleeps, sem_free, calls, held = _setup([_JResp(200, body={"hits": []}), _JResp(200, body={"hits": []})])
            lim.pause.extend(7.0)
            download_pixabay_video(38, "9")
            assert sleeps == [7.0] and sem_free == [True] and "pixabay.com/api/videos/?key=K&id=9" in calls[0]
            assert len(_fails(38)) == 1 and "Не удалось получить видео Pixabay" in _fails(38)[0]
            download_pixabay_photo(39, "9")   # пауза уже кончилась - без сна
            assert sleeps == [7.0] and lim.stats.requests == 2

            # ожидание места в окне - вне семафора: окно на 1 запрос, второй вызов ждёт 60 с
            lim, sleeps, sem_free, calls, held = _setup(
                [_JResp(200, body=_OK_PHOTO), _Resp(200, jpg), _JResp(200, body=_OK_PHOTO), _Resp(200, jpg)], rpm=1)
            download_pixabay_photo(40, "1")
            assert sleeps == [] and lim.stats.window_waits == 0
            download_pixabay_photo(41, "2")
            assert sleeps == [60.0] and sem_free == [True] and lim.stats.window_waits == 1
            assert not _fails(40) and not _fails(41) and lim.stats.requests == 2

            # прямая ссылка cdn.pixabay.com в окне не считается: окно на 1 запрос, а API-запрос не ждёт
            lim, sleeps, sem_free, calls, held = _setup(
                [_Resp(200, jpg), _JResp(200, body=_OK_PHOTO), _Resp(200, jpg)], rpm=1)
            download_media_item(42, "https://cdn.pixabay.com/photo/2020/a.jpg")
            assert lim.stats.requests == 0 and not _fails(42) and os.path.exists(os.path.join(tmp, "42.jpg"))
            download_pixabay_photo(43, "3")
            assert sleeps == [] and lim.stats.requests == 1 and lim.stats.window_waits == 0

            # PIXABAY_REQUESTS_PER_MINUTE: лениво читается из окружения; неверное значение - остановка
            set_pixabay_limiter(None)
            os.environ["PIXABAY_REQUESTS_PER_MINUTE"] = ""
            assert get_pixabay_limiter().window.max_requests == 90
            assert get_pixabay_limiter() is get_pixabay_limiter()   # один экземпляр на все потоки
            for _bad in ("0", "-5", "abc"):
                set_pixabay_limiter(None)
                os.environ["PIXABAY_REQUESTS_PER_MINUTE"] = _bad
                try:
                    get_pixabay_limiter()
                except ValueError as e:
                    assert "PIXABAY_REQUESTS_PER_MINUTE" in str(e)
                else:
                    raise AssertionError(_bad)
                sys.argv = ["download.py"]   # main() должен остановиться ДО очистки папки и скачивания
                set_pixabay_limiter(None)
                try:
                    main()
                except SystemExit as e:
                    assert e.code == 1, e.code
                else:
                    raise AssertionError(f"main() не остановился при PIXABAY_REQUESTS_PER_MINUTE={_bad!r}")
                assert os.path.isdir(tmp)
    finally:
        cffi_requests.get = _saved_get2
        globals()["PIXABAY_API_KEY"], OUTPUT_DIR = _saved_state2[0], _saved_state2[1]
        FAILED_ITEMS[:] = _saved_state2[2]
        set_pixabay_limiter(_saved_state2[3])
        if _saved_state2[4] is None:
            os.environ.pop("PIXABAY_REQUESTS_PER_MINUTE", None)
        else:
            os.environ["PIXABAY_REQUESTS_PER_MINUTE"] = _saved_state2[4]
        sys.argv = _saved_state2[5]

    print("ok")


def main():
    if "--selftest" in sys.argv:
        _selftest()
        return
    try:
        get_pixabay_limiter()  # неверный PIXABAY_REQUESTS_PER_MINUTE - остановка до скачивания
    except ValueError as e:
        _stop(str(e))
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parse_and_download_links("INPUT_LINKS", "INPUT_BACKUP_LINKS")
    write_failed_report()
    print("[ИНФО] " + DURATION_STATS.summary_line())

    counts = {}
    for name in os.listdir(OUTPUT_DIR):
        e = ext_from_name(name)
        counts[e] = counts.get(e, 0) + 1
    saved = ", ".join(f"{e}: {counts.get(e, 0)}" for e in ("jpg", "png", "mp4", "mov", "avi"))
    rejects = FORMAT_REJECTS.summary_line()
    print("[ИНФО] " + (f"(до backup) {rejects}. " if rejects else "") + f"Сохранено: {saved}")
    print("[ИНФО] " + SOURCE_STATS.summary_line())
    print("[ИНФО] " + get_pixabay_limiter().stats.summary_line())


if __name__ == "__main__":
    main()
