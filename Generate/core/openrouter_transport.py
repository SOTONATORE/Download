"""OpenRouter: транспорт генерации промптов поверх общего пула ключей (gemini_pool).

Интерфейс совпадает с RealTransport (gemini_prompts): generate_json(model, system, user, schema) -> str,
close(), поддержка with. Ошибки приходят как GeminiRuntimeError с теми же полями (status, code,
daily_quota, retry_after, key_exhausted_only), поэтому оркестратор менять не нужно.

Ключи берутся из переменных OPENROUTER1..OPENROUTER10. Лимиты бесплатных моделей OpenRouter
действуют на аккаунт, а не на ключ: несколько ключей могут делить одну квоту. Это здесь не учтено.

Значения ключей не попадают в логи, в repr() и в тексты исключений. Повторов, пауз и смены модели
здесь нет: это делает оркестратор.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Mapping, Optional

try:
    from .gemini_prompts import GeminiInputError, GeminiRuntimeError
except ImportError:  # запуск из папки Generate как скрипта
    from gemini_prompts import GeminiInputError, GeminiRuntimeError  # type: ignore

log = logging.getLogger("Generate.openrouter_transport")

# Имена переменных собираются из частей, чтобы в коде не было литерала секретного имени.
KEY_ENV_PREFIX = "OPENROUTER"
KEY_ENV_COUNT = 10

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_RPM = 20
DEFAULT_RPD = 1000
# OpenRouter не публикует TPM для бесплатных моделей; значение ниже — заглушка, его нужно проверить.
DEFAULT_TPM = 1_000_000
DEFAULT_MAX_OUTPUT_TOKENS = 4000

POOL_CHARS_PER_TOKEN = 3   # грубая оценка токенов входа для TPM
PREVIEW_CHARS = 300        # сколько символов тела 429 попадает в лог

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


class OpenRouterTransport:
    """Транспорт OpenRouter на httpx поверх общего пула ключей (gemini_pool).

    На каждый вызов generate_json берётся ключ (lease) из пула. Ключ уходит только в заголовок
    Authorization; в URL, логи и исключения он не попадает. Исход запроса возвращается в пул.
    """

    def __init__(self, *, pool: Any = None, keys: Any = None, environ: Any = None,
                 model_limits: Any = None, base_url: str = DEFAULT_BASE_URL, timeout: float = 120.0,
                 client: Any = None, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
                 use_json_schema: bool = True):
        # Если пул не передан, он создаётся из ключей (переданных или прочитанных из окружения).
        try:
            import httpx
        except ImportError:
            raise GeminiInputError("Не установлен пакет httpx.") from None
        self._httpx = httpx
        self._gpool = _pool_module()
        if pool is None:
            if keys is None:
                # окружение читается здесь, при создании; в самом запросе окружение не трогаем
                keys = load_openrouter_keys(environ)
            pool = self._gpool.KeyPool(keys, rpm_limit=DEFAULT_RPM, tpm_limit=DEFAULT_TPM,
                                       rpd_limit=DEFAULT_RPD, model_limits=model_limits)
        self._pool = pool
        self._tpm = getattr(pool, "_tpm", DEFAULT_TPM)   # оценка токенов не должна превышать TPM пула
        self._max_output_tokens = max_output_tokens
        self._use_json_schema = use_json_schema
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

    def generate_json(self, model: str, system: str, user: str, schema: dict) -> str:
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
        tokens = min((len(system) + len(user)) // POOL_CHARS_PER_TOKEN, self._tpm)
        try:
            lease = self._pool.acquire(model, tokens=tokens)
        except gpool.AllKeysExhaustedError:
            # все ключи исчерпали суточную квоту для модели: признак daily_quota, чтобы оркестратор сменил модель
            raise GeminiRuntimeError(
                "Все ключи OpenRouter исчерпали суточную квоту для этой модели.",
                status=429, code="RESOURCE_EXHAUSTED", daily_quota=True) from None

        # заголовок с ключом живёт только внутри этого вызова и никуда не передаётся
        headers = {"Authorization": f"Bearer {lease.api_key}", "Content-Type": "application/json"}
        try:
            resp = self._client.post(url, headers=headers, json=body)
        except (self._httpx.TransportError, self._httpx.TimeoutException):
            # ключ не виноват: состояние не меняется; в сообщение не попадает ничего из запроса
            self._pool.release(lease, gpool.Outcome.NETWORK_ERROR, model)
            raise GeminiRuntimeError("Сбой сети при обращении к OpenRouter.",
                                     status=None, code="NETWORK") from None
        except Exception as e:
            # любой другой сбой до ответа тоже без штрафа ключу; ключ обязательно возвращается в пул
            self._pool.release(lease, gpool.Outcome.NETWORK_ERROR, model)
            raise GeminiRuntimeError(
                f"Непредвиденный сбой при обращении к OpenRouter ({type(e).__name__}).",
                status=None, code="UNEXPECTED") from None

        status = resp.status_code
        if status == 200:
            content = _extract_content(resp)
            if content is None:
                # пустой или неожиданный ответ: ключ не виноват, но ответа нет
                # диагностика: что вернул OpenRouter, без текста ответа и без заголовков
                try:
                    _data = resp.json()
                    _choice = _data.get("choices", [{}])[0] if isinstance(_data, dict) else {}
                    _msg = _choice.get("message") or {}
                    _finish = _choice.get("finish_reason") or "нет"
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
                self._pool.release(lease, gpool.Outcome.SERVER_ERROR, model)
                # 500 переводит в переходные ошибки, и оркестратор повторит запрос
                raise GeminiRuntimeError("OpenRouter вернул ответ без текста.",
                                         status=500, code="EMPTY_RESPONSE")
            self._pool.release(lease, gpool.Outcome.SUCCESS, model)
            return content

        if status == 429:
            text = _body_text(resp)
            # день или минута по тексту тела: признак пока не проверен на реальных ответах OpenRouter
            lowered = text.lower()
            daily = _DAILY_PATTERN.search(lowered) is not None
            retry = None if daily else _retry_after(resp)
            outcome = gpool.Outcome.DAILY_QUOTA if daily else gpool.Outcome.RATE_LIMITED
            self._pool.release(lease, outcome, model, retry_after=retry)
            # в лог идёт только то, что вернул сам ответ: ключ заменяется на ***, длина обрезается
            log.warning("HTTP 429 от OpenRouter, модель %s, исход %s, тело ответа: %s",
                        model, outcome.value, _redact(text, lease.api_key)[:PREVIEW_CHARS])
            err = GeminiRuntimeError(
                f"OpenRouter вернул ошибку: HTTP 429, исход {outcome.value}.",
                status=429, code="RESOURCE_EXHAUSTED", daily_quota=daily, retry_after=retry)
            if daily:
                # квота исчерпана только у этого ключа для этой модели: пул может выдать другой ключ
                err.key_exhausted_only = True
            raise err

        if status == 402:
            # ключ не виноват; баланс общий для аккаунта, поэтому прогон останавливается
            self._pool.release(lease, gpool.Outcome.SERVER_ERROR, model)
            log.error("баланс аккаунта OpenRouter отрицательный или пуст, прогон остановлен.")
            raise GeminiRuntimeError("OpenRouter отклонил запрос: баланс аккаунта пуст (HTTP 402).",
                                     status=402, code="PAYMENT_REQUIRED", daily_quota=False)

        if status == 401:
            self._pool.release(lease, gpool.Outcome.SERVER_ERROR, model)
            # в лог идёт имя переменной, значение ключа не выводится
            log.error("Ключ отклонён OpenRouter (HTTP 401): проверьте переменную %s. Значение ключа не выводится.",
                      lease.name)
            raise GeminiRuntimeError("OpenRouter отклонил ключ (HTTP 401).",
                                     status=401, code="UNAUTHORIZED")

        # 500, 502, 503, 504 и прочие коды (в том числе 4xx): ключ не виноват, состояние не меняется
        self._pool.release(lease, gpool.Outcome.SERVER_ERROR, model)
        code = "UNAVAILABLE" if status == 503 else str(status)
        raise GeminiRuntimeError(f"OpenRouter вернул ошибку: HTTP {status}.", status=status, code=code)
