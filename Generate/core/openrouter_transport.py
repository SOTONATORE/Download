"""OpenRouter: транспорт генерации промптов с планировщиком отправок по ключам.

Интерфейс совпадает с RealTransport (gemini_prompts): generate_json(model, system, user, schema) -> str,
close(), поддержка with. Ошибки приходят как GeminiRuntimeError с теми же полями (status, code,
daily_quota, retry_after, key_exhausted_only), поэтому оркестратор менять не нужно.

Ключи берутся из переменных OPENROUTER1..OPENROUTER10. Пул с замком на время запроса здесь не используется:
планировщик OpenRouterSendScheduler решает, можно ли отправить запрос на ключ сейчас, по трём правилам:
пауза между отправками на ключ, число отправок в скользящем окне и блокировка ключа после 429.
Запрос к модели выполняется без каких-либо замков. Лимиты аккаунта OpenRouter могут быть общими
для всех ключей; это здесь не учтено, 429 от провайдера обрабатывается по каждому ключу отдельно.

Значения ключей не попадают в логи, в repr() и в тексты исключений. Повторов, пауз и смены модели
здесь нет: это делает оркестратор.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from typing import Any, Callable, Mapping, Optional

try:
    from .gemini_prompts import GeminiInputError, GeminiRuntimeError
except ImportError:  # запуск из папки Generate как скрипта
    from gemini_prompts import GeminiInputError, GeminiRuntimeError  # type: ignore

log = logging.getLogger("Generate.openrouter_transport")

# Имена переменных собираются из частей, чтобы в коде не было литерала секретного имени.
KEY_ENV_PREFIX = "OPENROUTER"
KEY_ENV_COUNT = 10

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Старые лимиты RPM/RPD/TPM больше не применяются в пути запроса (их заменяют правила планировщика);
# константы оставлены, так как на них ссылаются тесты и конфиг.
DEFAULT_RPM = 20
DEFAULT_RPD = 1000
# OpenRouter не публикует TPM для бесплатных моделей; значение ниже — заглушка, его нужно проверить.
DEFAULT_TPM = 1_000_000
# лимит ответа по умолчанию; рассуждения моделей занимают основную часть лимита
DEFAULT_MAX_OUTPUT_TOKENS = 32000
DEFAULT_REASONING_EFFORT = "low"
# генерация с рассуждениями может занимать до нескольких минут
OPENROUTER_TIMEOUT_SEC = 600.0

PREVIEW_CHARS = 300        # сколько символов тела 429 попадает в лог

# Правила планировщика отправок по умолчанию (переопределяются ключами конфига).
SEND_GAP_SEC = 3.0          # минимальный промежуток между отправками на один ключ, сек
SENDS_PER_WINDOW = 10       # отправок на один ключ в скользящем окне
WINDOW_SEC = 60.0           # длина скользящего окна, сек
DEFAULT_BLOCK_SEC = 60.0    # блокировка ключа при 429 без Retry-After, сек
SECONDS_PER_DAY = 86400     # суточная квота: сутки по UTC, как в пуле Gemini

# Признак суточной квоты в теле 429: «perday», «per day», «per-day» или слово «daily» (слово «today» не подходит).
_DAILY_PATTERN = re.compile(r"per[\s-]?day|\bdaily\b")

# Похожее на ключ в теле ответа заменяется на *** до записи в лог.
_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{8,}"),
    re.compile(r"sk-or-[0-9A-Za-z_\-]+"),
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
)


def _pool_module():
    """Модуль пула подгружается при создании транспорта: gemini_pool сам импортирует gemini_prompts."""
    try:
        from . import gemini_pool as gpool
    except ImportError:  # плоский импорт, как в остальных модулях core
        import gemini_pool as gpool  # type: ignore
    return gpool


def _silence_http_logs() -> None:
    """httpx пишет адреса запросов на уровне INFO; для транспорта это лишний шум."""
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def parallel_sends_limit(key_count: int) -> int:
    """Число потоков для OpenRouter: ключи × SENDS_PER_WINDOW, не меньше 1.

    Потоков столько, чтобы каждая разрешённая в окне отправка имела свой поток. Сам по себе
    этот предел не ограничивает число запросов в полёте: их ограничивают правила планировщика.
    """
    return max(1, key_count) * SENDS_PER_WINDOW


def load_openrouter_keys(environ: Optional[Mapping[str, str]] = None) -> list[tuple[str, str]]:
    """Читает переменные OPENROUTER с номерами 1..10. Пустые значения пропускаются.

    Без аргумента читается os.environ: так ключи получает cli.py, не трогая окружение сам.
    Возвращает пары (имя переменной, значение) в порядке номеров.
    В лог пишется только количество ключей.
    """
    # чтение окружения здесь, а не в вызывающем коде: значения ключей нигде не печатаются
    source = os.environ if environ is None else environ
    keys: list[tuple[str, str]] = []
    for i in range(1, KEY_ENV_COUNT + 1):
        name = KEY_ENV_PREFIX + str(i)
        value = (source.get(name) or "").strip()
        if value:
            keys.append((name, value))
    if not keys:
        raise GeminiInputError(
            "Не задан ни один ключ OpenRouter: переменные OPENROUTER1…OPENROUTER10 пусты или отсутствуют.")
    log.info("Загружено ключей OpenRouter: %d", len(keys))
    return keys


def _redact(text: str, api_key: str) -> str:
    """Заменяет в тексте значение ключа и похожие на ключ фрагменты на ***."""
    out = text.replace(api_key, "***") if api_key else text
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("***", out)
    return out


def _body_text(response: Any) -> str:
    """Тело ответа как текст. Если его нет или оно не строка, возвращается пустая строка."""
    try:
        text = response.text
    except Exception:
        return ""
    return text if isinstance(text, str) else ""


def _retry_after(response: Any) -> Optional[float]:
    """Заголовок Retry-After, если он число больше нуля. Иначе None."""
    try:
        value = float(response.headers.get("Retry-After", ""))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def _extract_content(response: Any) -> Optional[str]:
    """choices[0].message.content из ответа 200. Если поля нет или это не строка, возвращается None."""
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None
    return content if isinstance(content, str) else None


def _wait_for_key(now: float, sends, last_send: Optional[float], blocked_until: float,
                  send_gap: float, per_window: int, window: float) -> float:
    """Сколько секунд ждать до разрешённой отправки на ключ. 0.0 означает: отправка разрешена сейчас.

    Три правила, все должны выполняться одновременно:
    - блокировка после 429: now >= blocked_until;
    - пауза: now - last_send >= send_gap;
    - окно: отправок с временем t > now - window меньше per_window.
    sends — времена отправок ключа по возрастанию; устаревшие в функции отбрасываются.
    Если окно заполнено, ждём, пока старейшая из отправок, которую нужно вытеснить, уйдёт из окна.
    """
    wait = 0.0
    if now < blocked_until:
        wait = max(wait, blocked_until - now)
    if last_send is not None and now - last_send < send_gap:
        wait = max(wait, last_send + send_gap - now)
    in_window = [t for t in sends if t > now - window]
    excess = len(in_window) - per_window + 1   # сколько самых старых отправок нужно вытеснить, чтобы стало можно
    if excess > 0:
        wait = max(wait, in_window[excess - 1] + window - now)
    return wait


class OpenRouterSendScheduler:
    """Планировщик отправок OpenRouter: решает, можно ли отправить запрос на ключ сейчас.

    Состояние по каждому ключу: очередь времён отправок для скользящего окна, время последней отправки,
    время конца блокировки после 429. Отдельно: пары (ключ, модель), исчерпавшие суточную квоту.
    Один замок на все поля. Запрос в полёте не учитывается: ключ не блокируется из-за ожидающего ответа.
    Ожидание выполняется без замка: замок отпускается, затем выполняется sleep и решение принимается заново.
    """

    def __init__(self, key_names, *, send_gap_sec: float = SEND_GAP_SEC,
                 sends_per_window: int = SENDS_PER_WINDOW, window_sec: float = WINDOW_SEC,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        names = list(key_names)
        if not names:
            raise GeminiInputError("Планировщик OpenRouter пуст: нет ни одного ключа.")
        if len(set(names)) != len(names):
            raise GeminiInputError("В планировщике OpenRouter есть повторяющиеся имена ключей.")
        if send_gap_sec < 0:
            raise GeminiInputError("Пауза между отправками OpenRouter должна быть не меньше 0.")
        if sends_per_window < 1:
            raise GeminiInputError("Число отправок в окне OpenRouter должно быть не меньше 1.")
        if window_sec <= 0:
            raise GeminiInputError("Длина окна OpenRouter должна быть больше 0.")
        self._names = names
        self._send_gap = float(send_gap_sec)
        self._per_window = int(sends_per_window)
        self._window = float(window_sec)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._sends: dict[str, deque] = {n: deque() for n in names}
        self._last_send: dict[str, Optional[float]] = {n: None for n in names}
        self._blocked_until: dict[str, float] = {n: 0.0 for n in names}
        self._day = -1
        self._day_exhausted: set[tuple[str, str]] = set()   # (ключ, модель) — квота на сутки исчерпана
        self._cursor = 0                                    # с какого ключа начинается поиск (round-robin)
        self._exhausted_error = _pool_module().AllKeysExhaustedError

    def __repr__(self) -> str:
        return (f"OpenRouterSendScheduler(keys={len(self._names)}, gap={self._send_gap}, "
                f"per_window={self._per_window}, window={self._window})")

    def _roll_day(self, now: float) -> None:
        """При смене суток сбрасывает отметки суточной квоты по всем парам."""
        day = int(now // SECONDS_PER_DAY)
        if day != self._day:
            self._day = day
            self._day_exhausted.clear()

    def _record_send(self, name: str, now: float) -> None:
        """Записывает отправку на ключ и отбрасывает отправки, вышедшие из окна. Вызывается под замком."""
        sends = self._sends[name]
        sends.append(now)
        while sends and sends[0] <= now - self._window:
            sends.popleft()
        self._last_send[name] = now

    def acquire_send(self, model: str) -> str:
        """Возвращает имя ключа, на который разрешена отправка сейчас; ждёт, если разрешения пока нет.

        Отправка записывается в состояние ключа в момент решения, под замком. Если все ключи исчерпали
        суточную квоту для модели, бросает AllKeysExhaustedError (ожидание не начинается).
        """
        while True:
            with self._lock:
                now = self._clock()
                self._roll_day(now)
                count = len(self._names)
                best_wait: Optional[float] = None
                for i in range(count):
                    idx = (self._cursor + i) % count
                    name = self._names[idx]
                    if (name, model) in self._day_exhausted:
                        continue
                    wait = _wait_for_key(now, self._sends[name], self._last_send[name],
                                         self._blocked_until[name], self._send_gap,
                                         self._per_window, self._window)
                    if wait <= 0:
                        self._record_send(name, now)
                        self._cursor = (idx + 1) % count
                        return name
                    if best_wait is None or wait < best_wait:
                        best_wait = wait
                if best_wait is None:
                    raise self._exhausted_error(
                        "Все ключи OpenRouter исчерпали суточную квоту для этой модели.")
            # ожидание без замка: другие потоки могут отправлять и блокировать ключи
            self._sleep(best_wait)

    def block_key(self, name: str, seconds: float) -> None:
        """Блокирует ключ на seconds секунд от текущего времени (после 429 с Retry-After или без него)."""
        with self._lock:
            if name not in self._blocked_until:
                raise RuntimeError(f"Ключ {name} не входит в планировщик OpenRouter.")
            now = self._clock()
            self._blocked_until[name] = max(self._blocked_until[name], now + float(seconds))
            log.info("Ключ %s заблокирован на %.0f с после HTTP 429.", name, float(seconds))

    def mark_day_exhausted(self, name: str, model: str) -> None:
        """Отмечает суточную квоту пары (ключ, модель) до конца суток. Ключ остаётся доступен для других моделей."""
        with self._lock:
            if name not in self._blocked_until:
                raise RuntimeError(f"Ключ {name} не входит в планировщик OpenRouter.")
            self._roll_day(self._clock())
            self._day_exhausted.add((name, model))
            log.info("Ключ %s, модель %s: суточная квота исчерпана до конца суток.", name, model)

    def all_exhausted(self, model: str) -> bool:
        """True, если суточная квота для модели исчерпана на всех ключах."""
        with self._lock:
            self._roll_day(self._clock())
            return all((n, model) in self._day_exhausted for n in self._names)


class OpenRouterTransport:
    """Транспорт OpenRouter на httpx поверх планировщика отправок OpenRouterSendScheduler.

    Перед каждым вызовом generate_json планировщик выдаёт имя ключа и момент отправки. Ключ уходит только
    в заголовок Authorization; в URL, логи и исключения он не попадает. Исход запроса влияет на
    планировщик через block_key (429 с Retry-After или без него) и mark_day_exhausted (суточная квота).
    """

    provider = "openrouter"  # читает gemini_prompts при расчёте числа потоков

    def __init__(self, *, keys: Any = None, environ: Any = None,
                 model_limits: Any = None, base_url: str = DEFAULT_BASE_URL, timeout: float = OPENROUTER_TIMEOUT_SEC,
                 client: Any = None, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
                 use_json_schema: bool = True, reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                 send_gap_sec: float = SEND_GAP_SEC, sends_per_window: int = SENDS_PER_WINDOW,
                 window_sec: float = WINDOW_SEC, clock: Any = None, sleep: Any = None):
        # model_limits принимается для совместимости с cli.py и не используется: лимиты RPM/RPD заменены планировщиком
        try:
            import httpx
        except ImportError:
            raise GeminiInputError("Не установлен пакет httpx.") from None
        self._httpx = httpx
        self._gpool = _pool_module()
        if keys is None:
            # окружение читается здесь, при создании; в самом запросе окружение не трогаем
            keys = load_openrouter_keys(environ)
        keys = list(keys)
        self._key_values: dict[str, str] = dict(keys)   # имя переменной -> значение; значения только здесь
        self._scheduler = OpenRouterSendScheduler(
            [name for name, _ in keys], send_gap_sec=send_gap_sec, sends_per_window=sends_per_window,
            window_sec=window_sec, clock=clock or time.time, sleep=sleep or time.sleep)
        self._max_output_tokens = max_output_tokens
        self._use_json_schema = use_json_schema
        self._reasoning_effort = reasoning_effort
        _silence_http_logs()
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._base_url = base_url.rstrip("/")

    def __repr__(self) -> str:
        return f"OpenRouterTransport(base_url={self._base_url!r})"

    def close(self) -> None:
        """Закрывает внутренний httpx.Client. Клиент, переданный снаружи, не закрывается.
        Повторный вызов безопасен."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OpenRouterTransport":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def generate_json(self, model: str, system: str, user: str, schema: dict,
                      reasoning_effort: "str | None" = None) -> str:
        gpool = self._gpool
        url = f"{self._base_url}/chat/completions"
        if self._use_json_schema:
            response_format = {"type": "json_schema", "json_schema": {"name": "segments", "schema": schema}}
        else:
            response_format = {"type": "json_object"}
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self._max_output_tokens,
            "response_format": response_format,
        }
        # рассуждения: уровень запроса, если задан, иначе уровень транспорта; при none ключа в запросе нет
        effort = self._reasoning_effort if reasoning_effort is None else reasoning_effort
        if effort != "none":
            body["reasoning"] = {"effort": effort}
        try:
            name = self._scheduler.acquire_send(model)
        except gpool.AllKeysExhaustedError:
            # все ключи исчерпали суточную квоту для модели: признак daily_quota, чтобы оркестратор сменил модель
            raise GeminiRuntimeError(
                "Все ключи OpenRouter исчерпали суточную квоту для этой модели.",
                status=429, code="RESOURCE_EXHAUSTED", daily_quota=True) from None

        # заголовок с ключом живёт только внутри этого вызова и никуда не передаётся
        headers = {"Authorization": f"Bearer {self._key_values[name]}", "Content-Type": "application/json"}
        log.info("OpenRouter: запрос к модели %s, max_tokens %s, ключ %s", model, self._max_output_tokens, name)
        start = time.monotonic()
        try:
            resp = self._client.post(url, headers=headers, json=body)
            log.info("OpenRouter: ответ за %.1f с, статус %s", time.monotonic() - start, resp.status_code)
        except (self._httpx.TransportError, self._httpx.TimeoutException):
            # ключ не виноват: планировщик ничего не меняет; в сообщение не попадает ничего из запроса
            raise GeminiRuntimeError("Сбой сети при обращении к OpenRouter.",
                                     status=None, code="NETWORK") from None
        except Exception as e:
            # любой другой сбой до ответа тоже без штрафа ключу
            raise GeminiRuntimeError(
                f"Непредвиденный сбой при обращении к OpenRouter ({type(e).__name__}).",
                status=None, code="UNEXPECTED") from None

        status = resp.status_code
        if status == 200:
            content = _extract_content(resp)
            if content is None:
                # пустой или неожиданный ответ: ключ не виноват, но ответа нет
                # диагностика: что вернул OpenRouter, без текста ответа и без заголовков
                _finish_for_retry = None
                try:
                    _data = resp.json()
                    _choice = _data.get("choices", [{}])[0] if isinstance(_data, dict) else {}
                    _msg = _choice.get("message") or {}
                    _finish = _choice.get("finish_reason") or "нет"
                    _finish_for_retry = _finish
                    _usage = _data.get("usage") if isinstance(_data, dict) else None
                    _usage_text = json.dumps(_usage, ensure_ascii=False) if _usage is not None else "нет"
                    if "reasoning" in _msg:
                        _reasoning = _msg.get("reasoning")
                    elif "reasoning_content" in _msg:
                        _reasoning = _msg.get("reasoning_content")
                    else:
                        _reasoning = None
                    _reasoning_len = len(_reasoning) if isinstance(_reasoning, str) else "нет"
                    _content_val = _msg.get("content")
                    _content_len = len(_content_val) if isinstance(_content_val, str) else "null"
                    log.warning("Пустой ответ OpenRouter, модель %s, finish_reason %s, usage %s, "
                                "длина reasoning %s, длина content %s",
                                model, _finish, _usage_text, _reasoning_len, _content_len)
                except Exception:
                    log.warning("Пустой ответ OpenRouter, модель %s, диагностика недоступна", model)
                if _finish_for_retry == "length":
                    # лимит вывода исчерпан рассуждениями: тот же запрос даст тот же пустой ответ,
                    # поэтому статус вне переходных (500/503/504), и оркестратор не повторяет запрос
                    raise GeminiRuntimeError("OpenRouter исчерпал лимит вывода без текста.",
                                             status=422, code="EMPTY_RESPONSE_LENGTH")
                # 500 переводит в переходные ошибки, и оркестратор повторит запрос
                raise GeminiRuntimeError("OpenRouter вернул ответ без текста.",
                                         status=500, code="EMPTY_RESPONSE")
            return content

        if status == 429:
            text = _body_text(resp)
            # день или минута по тексту тела: признак пока не проверен на реальных ответах OpenRouter
            lowered = text.lower()
            daily = _DAILY_PATTERN.search(lowered) is not None
            retry = None if daily else _retry_after(resp)
            if daily:
                # квота исчерпана только у этого ключа для этой модели до конца суток
                self._scheduler.mark_day_exhausted(name, model)
                label = "daily_quota"
            else:
                # блокируем только этот ключ: Retry-After, а без него — значение по умолчанию
                self._scheduler.block_key(name, retry if retry is not None else DEFAULT_BLOCK_SEC)
                label = "rate_limited"
            # в лог идёт только то, что вернул сам ответ: ключ заменяется на ***, длина обрезается
            log.warning("HTTP 429 от OpenRouter, модель %s, ключ %s, исход %s, тело ответа: %s",
                        model, name, label, _redact(text, self._key_values[name])[:PREVIEW_CHARS])
            err = GeminiRuntimeError(
                f"OpenRouter вернул ошибку: HTTP 429, исход {label}.",
                status=429, code="RESOURCE_EXHAUSTED", daily_quota=daily, retry_after=retry)
            if daily:
                err.key_exhausted_only = True
            raise err

        if status == 402:
            # ключ не виноват; баланс общий для аккаунта, поэтому прогон останавливается
            log.error("баланс аккаунта OpenRouter отрицательный или пуст, прогон остановлен.")
            raise GeminiRuntimeError("OpenRouter отклонил запрос: баланс аккаунта пуст (HTTP 402).",
                                     status=402, code="PAYMENT_REQUIRED", daily_quota=False)

        if status == 401:
            # в лог идёт имя переменной, значение ключа не выводится
            log.error("Ключ отклонён OpenRouter (HTTP 401): проверьте переменную %s. Значение ключа не выводится.",
                      name)
            raise GeminiRuntimeError("OpenRouter отклонил ключ (HTTP 401).",
                                     status=401, code="UNAUTHORIZED")

        # 500, 502, 503, 504 и прочие коды (в том числе 4xx): ключ не виноват, состояние не меняется
        code = "UNAVAILABLE" if status == 503 else str(status)
        raise GeminiRuntimeError(f"OpenRouter вернул ошибку: HTTP {status}.", status=status, code=code)
