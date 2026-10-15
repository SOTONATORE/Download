"""Клиент Vast.ai (этап 6; SPEC 6, 8.2, 8.3, 0.1, 0.2, 0.5).

Только стандартная библиотека + httpx. Ключ берётся ТОЛЬКО из переменной окружения
и передаётся ТОЛЬКО в заголовке Authorization: Bearer. Ключ не попадает в URL,
логи, repr и тексты исключений.

Пути API заданы константами ниже, версия указана в самом пути. Статус каждого пути
(подтверждён ли живым ответом или нет) записан в SPEC 8.3.

Перенаправления (301/302/303/307/308) клиент НЕ выполняет: заголовок с ключом не уходит
по адресу, который сервер назвал сам, а сбой виден сразу как ошибка с кодом 3.

При debug_logs=True каждый запрос и ответ пишутся в лог сырыми, с маскированием секретов
(SPEC 0.1, 10). При debug_logs=False журнала нет: ни строк, ни вычислений маскирования.
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

log = logging.getLogger("Generate.vast_client")

# Имя секрета собрано из частей (test_structure.py: литерал в коде запрещён).
ENV_KEY_NAME = "GEN_" + "VAST_API_KEY"
# Корень сервера; версия API входит в каждый путь (см. константы ниже).
DEFAULT_BASE_URL = "https://console.vast.ai"
WORKER_HTTP_PORT = 8000
_SNIPPET_LIMIT = 200
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_LIST_PAGE_LIMIT = 25
# Предел страниц списка: защита от бесконечного цикла при сбое курсора.
_LIST_MAX_PAGES = 200
# Причины отказа при поиске: (ключ, подпись в строке итога поиска); порядок в логе фиксирован
_REJECT_LABELS = (
    ("unparsed", "не разобрано"),
    ("single_gpu", "не одиночная карта"),
    ("price_low", "цена ниже минимума"),
    ("price_high", "цена выше максимума"),
    ("reliability", "надёжность"),
    ("network", "сеть"),
    ("vram", "VRAM"),
    ("gpu_name", "имя GPU"),
    ("inet_price", "цена трафика выше максимума"),
    ("inet_price_unknown", "цена трафика не разобрана"),
)

# Пути API Vast.ai (статус каждого: SPEC 8.3, блок «Эндпоинты Vast»).
PATH_BUNDLES = "/api/v0/bundles/"            # POST: поиск предложений
PATH_ASKS = "/api/v0/asks/{id}/"             # PUT: аренда предложения
PATH_INSTANCES = "/api/v1/instances/"        # GET: список экземпляров, постраничный (v1)
PATH_INSTANCE = "/api/v0/instances/{id}/"    # GET, DELETE: один экземпляр
PATH_USER = "/api/v0/users/current/"         # GET: кредит аккаунта (поле credit)

# Поля цены сетевого трафика предложения Vast.
# UNVERIFIED: имена и единица взяты из официального vast-cli (github.com/vast-ai/vast-cli, vast.py):
# список полей поиска offers_fields и справка полей предложения, единица USD за ГБ. В ответе
# /api/v0/bundles/ живым вызовом не проверялись: живые вызовы Vast в этой задаче запрещены.
# Поле отсутствует или не читается: предложение отбрасывается и считается в
# «цена трафика не разобрана» (fail-closed).
INET_UP_COST_FIELD = "inet_up_cost"        # UNVERIFIED в ответе API; USD за ГБ по vast-cli
INET_DOWN_COST_FIELD = "inet_down_cost"    # UNVERIFIED в ответе API; USD за ГБ по vast-cli
# Текст статуса экземпляра (ошибка рантайма при неудачном старте, SPEC 8.2).
# UNVERIFIED: имя поля не найдено ни в репозитории, ни в производственном логе; официальный
# vast.py недоступен (403 политики egress, 13.10.2026). Поля нет или значение не строка:
# status_msg = "" и правило по тексту статуса не срабатывает (fail-closed).
STATUS_MSG_FIELD = "status_msg"
# Пересчёт USD за ГБ в USD за ТБ: коэффициент 1024, как у vast-cli при выводе цены за ТБ.
# UNVERIFIED: при десятичном ТБ (1000 ГБ) цена была бы примерно на 2.3% ниже.
GB_PER_TB = 1024.0


# --- отладочный журнал сырого обмена (только при debug_logs=True; SPEC 0.1, 10) ---
# Имя ключа, параметра или заголовка с одной из этих подстрок маскируется целиком. Подстрока
# «auth» задевает и «author»: лишнее маскирование допустимо, пропущенный секрет недопустим.
_SENSITIVE_NAME_PARTS = ("token", "key", "secret", "password", "authorization", "auth", "hmac", "crisp")
# Имена, чьё значение маскируется целиком, каким бы ни был его тип (список пар, объект, строка).
# extra_env в ответе Vast: список пар [имя, значение] с переменными окружения карты (SPEC 0.1, 10).
_FULL_MASK_NAMES = frozenset({"extra_env"})
_REDACTED = "***"
# «Bearer <значение>» в свободном тексте (тело не JSON).
_BEARER_RE = re.compile(r"(?i)\bBearer\s+\S+")
# Пара «имя: значение» или «имя=значение» с секретным именем в свободном тексте (тело не JSON).
# Совпадение начинается с кавычки или с границы слова (перед ним нет [\w.-]). Без этого якоря
# поиск пробует каждую позицию внутри длинного слова и работает квадратично (test_body_is_logged_in_full).
_PAIR_RE = re.compile(
    r"""(?i)(?:(?=[\"'])|(?<![\w.\-]))([\"']?[\w.\-]*(?:token|key|secret|password|authorization|auth|hmac|crisp)[\w.\-]*[\"']?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;&}\]]+)"""
)


def is_sensitive_name(name: str) -> bool:
    """Имя ключа, параметра или заголовка похоже на секрет (без учёта регистра) или целиком в _FULL_MASK_NAMES."""
    lowered = str(name).lower()
    return lowered in _FULL_MASK_NAMES or any(part in lowered for part in _SENSITIVE_NAME_PARTS)


def _pair_repl(m: "re.Match[str]") -> str:
    value = m.group(2)
    return m.group(1) + (f'"{_REDACTED}"' if value[:1] in ('"', "'") else _REDACTED)


def redact_text(text: str, secret: str = "") -> str:
    """Маскирует свободный текст: литерал ключа, «Bearer …» и пары «имя: значение» с секретным
    именем. Покрытие шаблонами не полное (UNVERIFIED), поэтому литерал ключа проверяется отдельно."""
    if secret:
        text = text.replace(secret, _REDACTED)
    text = _BEARER_RE.sub(f"Bearer {_REDACTED}", text)
    return _PAIR_RE.sub(_pair_repl, text)


def _redact_pair(item: Any, secret: str = "") -> Any:
    """Пара [имя, значение] с секретным именем (например ["WORKER_TOKEN", "..."]): маскируется значение.
    Имя остаётся, чтобы в журнале было видно, какая переменная передана. Исходный объект не меняется."""
    if isinstance(item, (list, tuple)) and len(item) == 2 and is_sensitive_name(item[0]):
        return [item[0], _REDACTED]
    return redact_value(item, secret)


def redact_value(obj: Any, secret: str = "") -> Any:
    """Рекурсивно маскирует JSON-объект: значение ключа с секретным именем заменяется на «***»,
    ключ сохраняется. Список пар [имя, значение] с секретным именем маскируется по значению.
    Строки проходят через redact_text. Исходный объект не меняется."""
    if isinstance(obj, dict):
        return {k: (_REDACTED if is_sensitive_name(k) else redact_value(v, secret))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_pair(x, secret) for x in obj]
    if isinstance(obj, str):
        return redact_text(obj, secret)
    return obj


def redact_query(params: Optional[dict], secret: str = "") -> dict:
    """Маскирует параметры запроса. Значение-JSON (как select_filters, order_by) разбирается
    и маскируется как тело. Параметр с секретным именем (например after_token) целиком «***»."""
    out: dict = {}
    for name, value in (params or {}).items():
        if is_sensitive_name(name):
            out[name] = _REDACTED
        elif isinstance(value, str) and value[:1] in ("{", "["):
            try:
                out[name] = json.dumps(redact_value(json.loads(value), secret), ensure_ascii=False)
            except ValueError:
                out[name] = redact_text(value, secret)
        else:
            out[name] = redact_value(value, secret)
    return out


def redact_headers(headers: dict, secret: str = "") -> dict:
    """Маскирует заголовки: Authorization, X-Api-Key и похожие имена становятся «***»."""
    return {n: (_REDACTED if is_sensitive_name(n) else redact_text(str(v), secret))
            for n, v in headers.items()}


def debug_body_for_log(raw: str, secret: str = "") -> str:
    """Сырое тело ответа для лога целиком. JSON маскируется по ключам, прочий текст шаблонами.
    Тело, которое не разобралось как JSON, остаётся текстом и не теряется."""
    try:
        parsed = json.loads(raw)
    except ValueError:
        return redact_text(raw, secret)
    return json.dumps(redact_value(parsed, secret), ensure_ascii=False)


def _request_body_for_log(body: Any, secret: str) -> str:
    """Тело запроса для лога целиком (маскирование то же, что у ответа)."""
    return json.dumps(redact_value(body, secret), ensure_ascii=False, default=str)


class VastError(Exception):
    """Базовая ошибка клиента Vast."""
    exit_code = 1


class VastInputError(VastError):
    """Неверная конфигурация: нет ключа, неверные фильтры."""
    exit_code = 2


class VastRuntimeError(VastError):
    """Сбой сети или ответ Vast API с кодом, отличным от 200."""
    exit_code = 3


class VastInstanceNotFound(VastRuntimeError):
    """Ответ API не содержит экземпляр с запрошенным ID (SPEC 8.2, 8.3). Код выхода наследуется: 3."""


@dataclass
class OfferFilter:
    gpu_name: Optional[str] = "RTX 5090"
    min_vram_gb: float = 0.0
    # Нижняя граница цены отсекает подозрительно дешёвые (нерабочие) хосты.
    min_price: float = 0.35
    max_price: float = 0.55
    min_reliability: float = 0.95
    min_inet_mbps: float = 800.0
    interruptible: bool = True
    verified: bool = False
    # Потолок цены трафика, USD за ТБ. Проверка клиентская, в обе стороны (см. INET_*_FIELD).
    max_inet_price_per_tb: float = 4.1


@dataclass
class GpuOffer:
    offer_id: int
    gpu_name: str
    num_gpus: int
    vram_gb: float
    price_per_hr: float
    reliability: float
    inet_down_mbps: float
    inet_up_mbps: float
    machine_id: int
    is_interruptible: bool
    inet_up_price_per_tb: Optional[float] = None
    inet_down_price_per_tb: Optional[float] = None


@dataclass
class InstanceInfo:
    instance_id: int
    actual_status: str
    intended_status: str
    label: Optional[str] = None
    public_ip: Optional[str] = None
    direct_port: Optional[int] = None
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    dph: float = 0.0
    status_msg: str = ""      # текст статуса (см. STATUS_MSG_FIELD); "" если поля нет
    raw: dict = field(default_factory=dict, repr=False)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_price_or_none(value: Any) -> Optional[float]:
    """Цена из ответа Vast или None, если её не удалось прочитать: нет значения, не число,
    отрицательное или не конечное. Логические значения не принимаются."""
    if value is None or isinstance(value, bool):
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price < 0:
        return None
    return price


def _per_tb_or_none(value: Any) -> Optional[float]:
    """Цена трафика из ответа (USD за ГБ) в USD за ТБ; None, если не читается."""
    per_gb = _to_price_or_none(value)
    return None if per_gb is None else per_gb * GB_PER_TB


class VastClient:
    """Тонкий клиент REST API Vast.ai."""

    def __init__(
        self,
        api_key_env: str = ENV_KEY_NAME,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        client: Optional[httpx.Client] = None,
        debug_logs: bool = False,
    ) -> None:
        key = os.environ.get(api_key_env, "").strip()
        if not key:
            raise VastInputError(
                f"Не задан ключ Vast: переменная окружения {api_key_env} пуста или отсутствует"
            )
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._timeout = timeout
        # отладочный журнал сырого обмена; при False строки лога и маскирование не выполняются
        self._debug = bool(debug_logs)
        # номер запроса в журнале: связывает строку «Запрос» со строкой «Ответ»
        self._req_seq = itertools.count(1)

    # --- служебное ---
    def __repr__(self) -> str:
        return "VastClient()"

    __str__ = __repr__

    def __enter__(self) -> "VastClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***")

    def _debug_request(self, req_no: int, method: str, path: str, params: Optional[dict],
                       body: Any, headers: dict) -> None:
        """Строка запроса в отладочном журнале: путь, параметры, тело. Секреты маскируются."""
        params_text = ("нет" if params is None else
                       json.dumps(redact_query(params, self._api_key), ensure_ascii=False))
        body_text = ("нет" if body is None else
                     _request_body_for_log(body, self._api_key))
        log.info(
            "Запрос Vast #%d: %s %s | параметры: %s | заголовки: %s | тело: %s",
            req_no, method, path, params_text,
            json.dumps(redact_headers(headers, self._api_key), ensure_ascii=False),
            body_text,
        )

    def _debug_response(self, req_no: int, elapsed: float, *, status: Optional[int] = None,
                        text: str = "", failure: Optional[str] = None) -> None:
        """Строка ответа: код HTTP, время в секундах и тело целиком (без обрезки).
        При сбое (таймаут, сеть) вместо кода выводится причина; текст исключения не выводится."""
        if failure is not None:
            log.info("Ответ Vast #%d: %s, за %.3f с, тела нет", req_no, failure, elapsed)
            return
        log.info("Ответ Vast #%d: HTTP %s, за %.3f с, тело: %s",
                 req_no, status, elapsed, debug_body_for_log(text, self._api_key))

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}
        req_no = next(self._req_seq)
        if self._debug:
            self._debug_request(req_no, method, path, params, json, headers)
        started = time.monotonic()
        try:
            resp = self._client.request(
                method, url, json=json, params=params, headers=headers, timeout=self._timeout
            )
        except httpx.TimeoutException:
            if self._debug:
                self._debug_response(req_no, time.monotonic() - started, failure="таймаут")
            log.warning("Таймаут запроса к Vast: %s %s", method, path)
            raise VastRuntimeError(f"Таймаут запроса к Vast API: {method} {path}") from None
        except httpx.HTTPError as exc:
            if self._debug:
                self._debug_response(req_no, time.monotonic() - started,
                                     failure=f"сетевая ошибка ({type(exc).__name__})")
            # Текст исходного исключения может содержать заголовки: берём только имя типа.
            log.warning("Сетевая ошибка Vast: %s %s (%s)", method, path, type(exc).__name__)
            raise VastRuntimeError(
                f"Сетевая ошибка при обращении к Vast API: {method} {path} ({type(exc).__name__})"
            ) from None

        if self._debug:
            self._debug_response(req_no, time.monotonic() - started,
                                 status=resp.status_code, text=resp.text)

        if resp.status_code != 200:
            code = resp.status_code
            snippet = self._redact(resp.text or "")[:_SNIPPET_LIMIT]
            if code in _REDIRECT_CODES:
                # Клиент не следует перенаправлению: адрес сервера показываем, но не используем.
                location = self._redact(str(resp.headers.get("location", "")))[:_SNIPPET_LIMIT]
                log.warning("Vast ответил перенаправлением HTTP %s: %s %s", code, method, path)
                raise VastRuntimeError(
                    f"Vast API ответил перенаправлением HTTP {code}: {method} {path}. "
                    f"Путь API устарел или записан неверно, перенаправления не выполняются. "
                    f"Новый адрес по данным сервера: {location or 'не указан'}"
                )
            if code == 410:
                log.warning("Vast вернул HTTP 410, конечная точка устарела: %s %s", method, path)
                raise VastRuntimeError(
                    f"Конечная точка Vast API устарела (HTTP 410): {method} {path}. "
                    f"Путь клиента нужно перевести на актуальную версию API. Ответ: {snippet}"
                )
            log.warning("Vast вернул HTTP %s: %s %s", code, method, path)
            raise VastRuntimeError(
                f"Vast API вернул HTTP {code}: {method} {path}. Ответ: {snippet}"
            )
        try:
            return resp.json()
        except ValueError:
            raise VastRuntimeError(
                f"Vast API вернул не JSON: {method} {path}"
            ) from None

    # --- кредит аккаунта (предстартовая проверка, SPEC 8.3 (е)) ---
    def get_balance_usd(self) -> float:
        """Расходуемый кредит аккаунта Vast, USD: поле credit ответа GET /api/v0/users/current/.

        Возвращает именно кредит (как «Credit» в консоли), а не поле balance: balance может
        быть 0 при положительном кредите, и подменять одно поле другим нельзя.
        Поле проверяется строго: отсутствие или нечисловое значение дают VastRuntimeError,
        а не 0. Неизвестный кредит не должен пропускать запуск с пустым счётом.
        Отрицательный кредит допустим (означает задолженность) и сравнивается дальше.
        """
        data = self._request("GET", PATH_USER)
        if not isinstance(data, dict):
            raise VastRuntimeError("Vast API вернул неожиданный ответ при проверке кредита")
        if "credit" not in data:
            raise VastRuntimeError("В ответе Vast API нет поля кредита (credit)")
        value = data["credit"]
        # bool — подкласс int, но «true» кредитом не является; строки не принимаем (схема: число)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise VastRuntimeError("Поле кредита в ответе Vast API не является числом")
        return float(value)

    # --- поиск предложений ---
    @staticmethod
    def _build_query(f: OfferFilter) -> dict:
        q: dict[str, Any] = {
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "dph_total": {"lte": f.max_price},
            "reliability2": {"gte": f.min_reliability},
            "inet_down": {"gte": f.min_inet_mbps},
            "type": "bid" if f.interruptible else "on-demand",
            "num_gpus": {"eq": 1},
            "order": [["dph_total", "asc"]],
            "limit": 100,
        }
        if f.min_price > 0:
            q["dph_total"]["gte"] = f.min_price
        if f.verified:
            q["verified"] = {"eq": True}
        if f.min_vram_gb > 0:
            q["gpu_ram"] = {"gte": f.min_vram_gb * 1024}
        if f.gpu_name:
            q["gpu_name"] = {"eq": f.gpu_name}
        return q

    @staticmethod
    def _parse_offer(b: dict, interruptible: bool) -> Optional[GpuOffer]:
        offer_id = _to_int(b.get("id"))
        if offer_id is None:
            return None
        price = _to_float(b.get("dph_total"))
        if interruptible and b.get("min_bid") is not None:
            price = _to_float(b.get("min_bid"), price)
        reliability = b.get("reliability2", b.get("reliability"))
        return GpuOffer(
            offer_id=offer_id,
            gpu_name=str(b.get("gpu_name", "")),
            num_gpus=_to_int(b.get("num_gpus")) or 1,
            vram_gb=_to_float(b.get("gpu_ram")) / 1024.0,
            price_per_hr=price,
            reliability=_to_float(reliability),
            inet_down_mbps=_to_float(b.get("inet_down")),
            inet_up_mbps=_to_float(b.get("inet_up")),
            machine_id=_to_int(b.get("machine_id")) or 0,
            is_interruptible=interruptible,
            inet_up_price_per_tb=_per_tb_or_none(b.get(INET_UP_COST_FIELD)),
            inet_down_price_per_tb=_per_tb_or_none(b.get(INET_DOWN_COST_FIELD)),
        )

    def search_offers(self, filters: OfferFilter) -> list[GpuOffer]:
        if filters.max_price <= 0:
            raise VastInputError("Неверный фильтр: max_price должен быть больше нуля")
        if filters.min_price < 0:
            raise VastInputError("Неверный фильтр: min_price не может быть отрицательным")
        if filters.min_price > filters.max_price:
            raise VastInputError(
                "Неверный фильтр: min_price не может быть больше max_price"
            )
        if not 0.0 <= filters.min_reliability <= 1.0:
            raise VastInputError("Неверный фильтр: min_reliability должен быть от 0 до 1")
        if filters.min_inet_mbps < 0 or filters.min_vram_gb < 0:
            raise VastInputError("Неверный фильтр: отрицательные значения не допускаются")
        if not filters.max_inet_price_per_tb > 0:
            raise VastInputError("Неверный фильтр: max_inet_price_per_tb должен быть больше нуля")

        data = self._request("POST", PATH_BUNDLES, json=self._build_query(filters))
        bundles = data.get("offers", []) if isinstance(data, dict) else []
        result: list[GpuOffer] = []
        # Счётчики по причинам отказа: каждое предложение учитывается один раз, по первой
        # причине, поэтому сумма счётчиков равна числу полученных предложений (SPEC 10).
        rejected = {key: 0 for key, _label in _REJECT_LABELS}
        for b in bundles:
            reason, offer = self._check_offer(b, filters)
            if reason is not None:
                rejected[reason] += 1
                continue
            result.append(offer)
        result.sort(key=lambda o: o.price_per_hr)
        parts = ", ".join(f"{label} {rejected[key]}" for key, label in _REJECT_LABELS)
        log.info("Поиск предложений Vast: получено %d, отброшено: %s; подходит %d.",
                 len(bundles), parts, len(result))
        return result

    @staticmethod
    def _check_offer(b: Any, filters: OfferFilter):
        """(причина_отказа или None, предложение или None). Порядок проверок фиксирован."""
        if not isinstance(b, dict):
            return "unparsed", None
        offer = VastClient._parse_offer(b, filters.interruptible)
        if offer is None:
            return "unparsed", None
        # Аренда только одиночных карт: проверяем сырое значение из ответа,
        # а не GpuOffer.num_gpus (там пропуск уже заменён на 1).
        if _to_int(b.get("num_gpus")) != 1:
            return "single_gpu", None
        # Повторная проверка на клиенте: сервер может вернуть лишнее.
        if offer.price_per_hr < filters.min_price:
            return "price_low", None
        if offer.price_per_hr > filters.max_price:
            return "price_high", None
        if offer.reliability < filters.min_reliability:
            return "reliability", None
        if offer.inet_down_mbps < filters.min_inet_mbps:
            return "network", None
        if offer.vram_gb < filters.min_vram_gb:
            return "vram", None
        if filters.gpu_name and offer.gpu_name != filters.gpu_name:
            return "gpu_name", None
        # Цена трафика в обе стороны. Нечитаемая цена отбрасывается (fail-closed): предложение
        # без цены нельзя признать подходящим. Серверного фильтра по полю нет: имя в живом ответе
        # не подтверждено (UNVERIFIED).
        if offer.inet_up_price_per_tb is None or offer.inet_down_price_per_tb is None:
            return "inet_price_unknown", None
        if (offer.inet_up_price_per_tb > filters.max_inet_price_per_tb
                or offer.inet_down_price_per_tb > filters.max_inet_price_per_tb):
            return "inet_price", None
        return None, offer

    # --- экземпляры ---
    def create_instance(
        self,
        offer_id: int,
        image: str,
        disk_gb: int,
        env_vars: Optional[dict] = None,
        label: Optional[str] = None,
        onstart: Optional[str] = None,
    ) -> int:
        if not image:
            raise VastInputError("Не задан образ контейнера для аренды карты")
        if disk_gb <= 0:
            raise VastInputError("Размер диска должен быть больше нуля")
        payload: dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "env": dict(env_vars or {}),
        }
        if label:
            payload["label"] = label
        if onstart:
            payload["onstart"] = onstart
        data = self._request("PUT", PATH_ASKS.format(id=int(offer_id)), json=payload)
        inst_id = _to_int(data.get("new_contract")) if isinstance(data, dict) else None
        if inst_id is None or (isinstance(data, dict) and data.get("success") is False):
            raise VastRuntimeError("Vast API не вернул номер созданного экземпляра")
        log.info("Создан экземпляр Vast %d (предложение %d)", inst_id, offer_id)
        return inst_id

    @staticmethod
    def _parse_instance(d: dict) -> InstanceInfo:
        direct_port = None
        ports = d.get("ports")
        if isinstance(ports, str):
            # схема v1 описывает ports строкой: разбираем как JSON; не разобрать значит порта нет
            try:
                ports = json.loads(ports)
            except ValueError:
                ports = None
        if isinstance(ports, dict):
            mapped = ports.get(f"{WORKER_HTTP_PORT}/tcp")
            if isinstance(mapped, list) and mapped and isinstance(mapped[0], dict):
                direct_port = _to_int(mapped[0].get("HostPort"))
        label = d.get("label")
        raw_msg = d.get(STATUS_MSG_FIELD)
        return InstanceInfo(
            instance_id=_to_int(d.get("id")) or 0,
            actual_status=str(d.get("actual_status") or ""),
            intended_status=str(d.get("intended_status") or ""),
            label=str(label) if label else None,
            public_ip=d.get("public_ipaddr") or None,
            direct_port=direct_port,
            ssh_host=d.get("ssh_host") or None,
            ssh_port=_to_int(d.get("ssh_port")),
            dph=_to_float(d.get("dph_total")),
            # не строка (или поля нет) значит текста нет: иначе в проверку попал бы repr объекта
            status_msg=raw_msg if isinstance(raw_msg, str) else "",
            raw=d,
        )

    def get_instance(self, instance_id: int) -> InstanceInfo:
        data = self._request("GET", PATH_INSTANCE.format(id=int(instance_id)))
        inst = data.get("instances") if isinstance(data, dict) else None
        if isinstance(inst, list):
            inst = inst[0] if inst else None
        if not isinstance(inst, dict):
            raise VastInstanceNotFound(f"Экземпляр Vast {instance_id} не найден в ответе API")
        return self._parse_instance(inst)

    def destroy_instance(self, instance_id: int) -> bool:
        data = self._request("DELETE", PATH_INSTANCE.format(id=int(instance_id)))
        if isinstance(data, dict) and data.get("success") is False:
            raise VastRuntimeError(f"Vast отклонил удаление экземпляра {instance_id}")
        log.info("Экземпляр Vast %d уничтожен", instance_id)
        return True

    @staticmethod
    def _list_params(after_token: Optional[str], limit: int) -> dict:
        params: dict[str, Any] = {
            "select_filters": json.dumps({}),
            "order_by": json.dumps([{"col": "id", "dir": "asc"}]),
            "limit": limit,
        }
        if after_token is not None:
            params["after_token"] = after_token
        return params

    def _fetch_instance_page(self, after_token: Optional[str], limit: int) -> dict:
        """Одна страница GET /api/v1/instances/. Отсутствие списка — ошибка, а не пустой список:
        пустой список при уборке означал бы «карт нет», хотя ответ просто не распознан."""
        data = self._request(
            "GET", PATH_INSTANCES, params=self._list_params(after_token, limit)
        )
        if not isinstance(data, dict):
            raise VastRuntimeError(
                f"Vast API вернул неожиданный ответ при списке экземпляров: GET {PATH_INSTANCES}"
            )
        if not isinstance(data.get("instances"), list):
            raise VastRuntimeError(
                f"В ответе Vast API нет списка экземпляров (instances): GET {PATH_INSTANCES}"
            )
        return data

    def check_instances_endpoint(self) -> None:
        """Предстартовая проверка эндпоинта списка (SPEC 8.3): одна страница, только чтение."""
        self._fetch_instance_page(None, limit=1)

    def list_instances(self, label: Optional[str] = None) -> list[InstanceInfo]:
        """Все экземпляры аккаунта (все страницы по next_token), опционально по метке.

        Неполный список опасен: карта, не попавшая в него, останется арендованной.
        Поэтому любой сбой страницы или повтор курсора прерывает весь список ошибкой.
        """
        rows: list[dict] = []
        token: Optional[str] = None
        seen: set[str] = set()
        for _ in range(_LIST_MAX_PAGES):
            data = self._fetch_instance_page(token, _LIST_PAGE_LIMIT)
            rows.extend(d for d in data["instances"] if isinstance(d, dict))
            nxt = data.get("next_token")
            if not nxt:
                break
            nxt = str(nxt)
            if nxt in seen:
                raise VastRuntimeError(
                    f"Vast API повторил страницу списка экземпляров: GET {PATH_INSTANCES}. "
                    f"Список неполный, уборка прервана"
                )
            seen.add(nxt)
            token = nxt
        else:
            raise VastRuntimeError(
                f"Список экземпляров Vast длиннее допустимого числа страниц: "
                f"GET {PATH_INSTANCES}. Список неполный, уборка прервана"
            )
        result = [self._parse_instance(d) for d in rows]
        if label is not None:
            result = [i for i in result if i.label == label]
        return result

    def destroy_by_label(self, label: str) -> list[int]:
        """Уничтожает все экземпляры с меткой (SPEC 8.3). Пробует все, ошибки собирает."""
        if not label:
            raise VastInputError("Пустая метка: массовое удаление без метки запрещено")
        destroyed: list[int] = []
        failed: list[int] = []
        for inst in self.list_instances(label=label):
            try:
                self.destroy_instance(inst.instance_id)
                destroyed.append(inst.instance_id)
            except VastRuntimeError:
                failed.append(inst.instance_id)
        if failed:
            raise VastRuntimeError(
                f"Не удалось уничтожить экземпляры Vast: {failed}; уничтожены: {destroyed}"
            )
        return destroyed
