"""Диагностика Vast.ai и Hugging Face: только чтение (SPEC 0.1, 0.5, 8.3).

Ничего не арендует, не создаёт, не меняет и не удаляет. Из ответа аккаунта берутся только баланс
(balance) и кредит (credit); печатаются и пишутся в probe_result.json только эти два значения.
Вызовы:
  • POST /api/v0/bundles/                    — поиск предложений, сырой ответ (тот же запрос, что в production);
  • POST /api/v0/bundles/                    — VastClient.search_offers по текущим правилам;
  • GET  /api/v0/users/current/              — баланс и кредит аккаунта (через _request, только чтение);
  • GET  /api/v0/instances/999999999/        — проверка пути одного экземпляра (id заведомо не существует);
  • GET  /api/v1/instances/999999999/        — то же для пути v1;
  • HEAD <базовый адрес>/<подкаталог>/<имя>  — по одному на каждую строку download_model в
    Generate/worker/setup.sh (Hugging Face, заголовок Authorization с HF_TOKEN, перенаправления выполняются).

Ключ Vast читается только из окружения (имя — ENV_KEY_NAME), токен Hugging Face — из переменной HF_TOKEN.
Оба значения уходят только в заголовке Authorization. Всё, что печатается или пишется, проходит
замену значений секретов на «***».

Приватные члены VastClient: _request (тот же путь запроса и обработка ошибок, что у клиента),
_build_query (та же сборка запроса), _base_url и _timeout (тот же адрес и таймаут). Они нужны,
чтобы сырой ответ совпадал с production. Generate/ при этом не меняется.
Проверка путей Vast не использует _request: он поднимает исключение на любом коде, кроме 200,
и скрыл бы код ответа. Перенаправления для путей Vast не выполняются (поведение httpx по умолчанию).

Коды выхода: 0 — поиск выполнен; 2 — нет ключа или неверные входы (сеть не трогается);
3 — сбой вызова Vast; 1 — прочая ошибка (например, запись probe_result.json).
Результаты проверок счёта, путей и файлов модели на код выхода не влияют: неуспешная проверка
печатается и попадает в поле account или checks файла probe_result.json.

Входы — переменные окружения, все необязательные (значения по умолчанию совпадают с
Generate/config.default.yaml, кроме max_price и min_inet_mbps, которые заданы по запросу):
PROBE_GPU_NAME, PROBE_MIN_PRICE, PROBE_MAX_PRICE, PROBE_MIN_INET_MBPS,
PROBE_MIN_RELIABILITY, PROBE_SHOW_COUNT, PROBE_MIN_BALANCE (минимум для сравнения с балансом и
кредитом, USD; по умолчанию 2.0). HF_TOKEN необязателен: без него проверка файлов модели
пропускается.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]  # файл в Generate/tools/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Generate.core.vast_client import (  # noqa: E402
    ENV_KEY_NAME,
    PATH_BUNDLES,
    PATH_USER,
    OfferFilter,
    VastClient,
    VastError,
)

RESULT_FILE = "probe_result.json"
RESULT_OFFERS_LIMIT = 200
SENSITIVE_MARKERS = ("key", "token", "secret", "authorization")

DEFAULT_GPU = "RTX 5090"
DEFAULT_MIN_PRICE = 0.35
DEFAULT_MAX_PRICE = 0.55
DEFAULT_MIN_INET_MBPS = 800.0
DEFAULT_MIN_RELIABILITY = 0.95
DEFAULT_SHOW_COUNT = 5
DEFAULT_MIN_BALANCE = 2.0

# Поля ответа аккаунта, которые печатаются и попадают в probe_result.json. Других полей нет.
ACCOUNT_FIELDS = ("balance", "credit")

# Поля, по которым печатается статистика. Значения сырые, без пересчёта единиц.
PRICE_FIELDS = (
    "dph_total",
    "min_bid",
    "inet_down",
    "inet_up",
    "inet_down_cost",
    "inet_up_cost",
    "internet_down_cost_per_tb",
    "internet_up_cost_per_tb",
)
# Поля первых предложений. Поля *_per_tb показываются только если присутствуют в предложении.
SHOW_FIELDS = (
    "id",
    "gpu_name",
    "num_gpus",
    "dph_total",
    "min_bid",
    "inet_down",
    "inet_up",
    "inet_down_cost",
    "inet_up_cost",
)
SHOW_OPTIONAL_FIELDS = ("internet_down_cost_per_tb", "internet_up_cost_per_tb")

LOG_NAME = "Generate.vast_client"
SUMMARY_PREFIX = "Поиск предложений Vast"

# Проверка путей Vast: id заведомо не существует; пути одиночного экземпляра в v0 и v1.
VAST_CHECK_INSTANCE_ID = 999999999
VAST_CHECK_PATHS = ("/api/v0/instances/{id}/", "/api/v1/instances/{id}/")

# Проверка файлов модели: адрес по умолчанию совпадает с HF_BASE_URL в setup.sh.
SETUP_SCRIPT = REPO_ROOT / "Generate" / "worker" / "setup.sh"
HF_DEFAULT_BASE_URL = "https://huggingface.co/Lightricks/LTX-2.5/resolve/main"
HF_TIMEOUT_SEC = 30.0
HF_BASE_RE = re.compile(r'^HF_BASE_URL="([^"]+)"\s*$', re.MULTILINE)
HF_CALL_RE = re.compile(r'^\s*download_model\s+"([^"]+)"\s+"([^"]+)"\s*$')
HF_CALL_PREFIX_RE = re.compile(r"^\s*download_model\s")  # определение функции «download_model() {» не совпадает

SNIPPET_LIMIT = 200

# Значения секретов (ключ Vast, HF_TOKEN): заменяются на «***» во всём выводе.
# Задаются в main() до первого вывода; самое длинное значение первым.
_secrets: list[str] = []


class ProbeInputError(Exception):
    """Неверные переменные окружения пробы или неразбираемый setup.sh."""


class _Capture(logging.Handler):
    """Запоминает сообщения журнала клиента, чтобы напечатать строку итога поиска."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:  # журнал не должен ронять пробу
            self.messages.append("(запись журнала не прочитана)")


def _clean(text: str) -> str:
    for secret in _secrets:
        text = text.replace(secret, "***")
    return text


def _snippet(text: str) -> str:
    """Первые SNIPPET_LIMIT символов тела с замёнными секретами. Текст берётся с запасом
    на длину самого длинного секрета, чтобы секрет на границе заменился целиком, а не остался частью."""
    margin = max((len(s) for s in _secrets), default=0)
    return _clean(text[: SNIPPET_LIMIT + margin])[:SNIPPET_LIMIT].replace("\n", " ")


def say(text: str = "") -> None:
    print(_clean(text), flush=True)


def err(text: str) -> None:
    print(_clean(text), file=sys.stderr, flush=True)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ProbeInputError(f"{name}: «{raw}» — не число") from None
    if not math.isfinite(value):
        raise ProbeInputError(f"{name}: значение должно быть конечным числом")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ProbeInputError(f"{name}: «{raw}» — не целое число") from None


def _read_inputs() -> tuple[OfferFilter, int]:
    """Фильтр и число показываемых предложений. Границы те же, что у search_offers,
    но проверяются до сетевых вызовов."""
    gpu_name = os.environ.get("PROBE_GPU_NAME", DEFAULT_GPU).strip()
    min_price = _env_float("PROBE_MIN_PRICE", DEFAULT_MIN_PRICE)
    max_price = _env_float("PROBE_MAX_PRICE", DEFAULT_MAX_PRICE)
    min_inet = _env_float("PROBE_MIN_INET_MBPS", DEFAULT_MIN_INET_MBPS)
    min_rel = _env_float("PROBE_MIN_RELIABILITY", DEFAULT_MIN_RELIABILITY)
    show_count = _env_int("PROBE_SHOW_COUNT", DEFAULT_SHOW_COUNT)

    if max_price <= 0:
        raise ProbeInputError("max_price должен быть больше нуля")
    if min_price < 0 or min_price > max_price:
        raise ProbeInputError("нужно 0 ≤ min_price ≤ max_price")
    if not 0.0 <= min_rel <= 1.0:
        raise ProbeInputError("min_reliability должен быть от 0 до 1")
    if min_inet < 0:
        raise ProbeInputError("min_inet_mbps не может быть отрицательным")
    if show_count < 0:
        raise ProbeInputError("show_count не может быть отрицательным")

    offer_filter = OfferFilter(
        gpu_name=gpu_name or None,
        min_price=min_price,
        max_price=max_price,
        min_reliability=min_rel,
        min_inet_mbps=min_inet,
    )
    return offer_filter, show_count


def _read_min_balance() -> float:
    """Минимум для сравнения с балансом и кредитом, USD: конечное число, не отрицательное."""
    min_balance = _env_float("PROBE_MIN_BALANCE", DEFAULT_MIN_BALANCE)
    if min_balance < 0:
        raise ProbeInputError("PROBE_MIN_BALANCE не может быть отрицательным")
    return min_balance


def _filter_text(f: OfferFilter) -> str:
    return (
        f"GPU «{f.gpu_name or 'любая'}», цена {f.min_price}–{f.max_price} USD/ч, "
        f"inet_down ≥ {f.min_inet_mbps} Мбит/с, надёжность ≥ {f.min_reliability}, "
        f"потолок трафика {f.max_inet_price_per_tb} USD/ТБ"
    )


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _field_line(offers: list[dict], field: str) -> str:
    """Сколько предложений содержат поле и статистика по его числовым значениям."""
    present = [o[field] for o in offers if field in o]
    numbers = [v for v in present if _is_number(v)]
    if numbers:
        stats = f"min {min(numbers)}, median {statistics.median(numbers)}, max {max(numbers)}"
    else:
        stats = "числовых значений нет"
    return (
        f"  {field}: есть у {len(present)} из {len(offers)}; "
        f"числовых {len(numbers)}; {stats}"
    )


def _value(offer: dict, field: str) -> str:
    value = offer.get(field)
    return "—" if value is None else str(value)


def _is_sensitive(name: Any) -> bool:
    low = str(name).lower()
    return any(marker in low for marker in SENSITIVE_MARKERS)


def _sanitize(value: Any) -> Any:
    """Рекурсивно убирает поля, в имени которых есть key, token, secret или authorization."""
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items() if not _is_sensitive(k)}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def _check_account(client: VastClient, min_balance: float) -> dict[str, Any]:
    """Баланс и кредит аккаунта: один GET на PATH_USER через _request, только чтение.
    VastClient.get_balance_usd не подходит: он требует баланс и не читает кредит.
    Возвращает блок для probe_result.json: balance и credit (число или null) и использованный минимум."""
    account: dict[str, Any] = {"balance": None, "credit": None, "min_balance_usd": min_balance}
    say("=== Проверка счёта Vast ===")
    try:
        data = client._request("GET", PATH_USER)
    except VastError as exc:
        err(f"Сбой запроса счёта Vast: {exc}")
        say("Баланс и кредит не получены; остальные проверки выполняются.")
        return account
    if not isinstance(data, dict):
        say(f"Ответ счёта не является объектом JSON: тип {type(data).__name__}")
        say("Баланс и кредит не получены; остальные проверки выполняются.")
        return account
    for field in ACCOUNT_FIELDS:
        if field not in data:
            say(f"{field}: поля нет в ответе")
            continue
        value = data[field]
        if not _is_number(value):
            say(f"{field}: значение не является конечным числом")
            continue
        number = float(value)
        account[field] = number
        say(f"{field}: {number} USD")
        verdict = "да" if number >= min_balance else "нет"
        say(f"{field}: достаточно для минимума {min_balance} USD: {verdict}")
    return account


def _vast_path_note(code: int) -> str | None:
    if code == 404:
        return "путь жив, инстанса нет (ожидаемо)"
    if code == 410:
        return "путь устарел, нужен v1 или другой путь"
    if code in (401, 403):
        return "проблема с ключом или правами"
    return None


def _hf_file_note(code: int) -> str | None:
    if code == 200:
        return "файл доступен"
    if code in (401, 403):
        return "нет доступа: проверьте HF_TOKEN и принятие условий модели"
    if code == 404:
        return "имени файла нет в репозитории"
    return None


def _check_vast_paths(base_url: str, timeout: float, key: str) -> list[dict]:
    """Проверка путей Vast: GET на несуществующий экземпляр (v0 и v1). Запрос строится напрямую,
    с тем же базовым адресом, таймаутом и заголовком Bearer, что у клиента. Других методов нет."""
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    results: list[dict] = []
    say("=== Проверка путей Vast ===")
    with httpx.Client(timeout=timeout) as http:
        for template in VAST_CHECK_PATHS:
            path = template.format(id=VAST_CHECK_INSTANCE_ID)
            entry: dict[str, Any] = {"method": "GET", "path": path}
            try:
                resp = http.get(f"{base_url}{path}", headers=headers)
            except Exception as exc:
                # Сбой запроса — строка отчёта, а не падение пробы. Текст исключения может
                # содержать адрес и заголовки: берём только имя типа.
                entry.update(status=None, error=type(exc).__name__)
                results.append(entry)
                say(f"GET {path}: сетевая ошибка ({type(exc).__name__})")
                continue
            note = _vast_path_note(resp.status_code)
            snippet = _snippet(resp.text or "")
            entry.update(status=resp.status_code, body_snippet=snippet, interpretation=note)
            results.append(entry)
            line = f"GET {path}: HTTP {resp.status_code}"
            if note:
                line += f" — {note}"
            say(line)
            say(f"  тело (первые {SNIPPET_LIMIT} символов): {snippet}")
    return results


def _read_setup_models() -> tuple[str, str, list[tuple[str, str]], int]:
    """Разбор setup.sh: (базовый адрес, источник адреса, пары (подкаталог, имя), число непонятых строк).
    ProbeInputError — если файл не читается, адрес содержит переменную оболочки или строк нет совсем."""
    try:
        text = SETUP_SCRIPT.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProbeInputError(
            f"Generate/worker/setup.sh не прочитан ({type(exc).__name__})"
        ) from None

    match = HF_BASE_RE.search(text)
    if match:
        base, source = match.group(1), "setup.sh"
    else:
        base = HF_DEFAULT_BASE_URL
        source = "значение по умолчанию: HF_BASE_URL в setup.sh не найден"
    if "$" in base:
        raise ProbeInputError("HF_BASE_URL в setup.sh содержит переменную оболочки, адрес не вычисляется")

    files: list[tuple[str, str]] = []
    unparsed = 0
    for line in text.splitlines():
        if not HF_CALL_PREFIX_RE.match(line):
            continue
        call = HF_CALL_RE.match(line)
        if call and "$" not in call.group(1) + call.group(2):
            files.append((call.group(1), call.group(2)))
        else:
            unparsed += 1
    if not files:
        raise ProbeInputError("в Generate/worker/setup.sh нет разбираемых строк download_model")
    return base, source, files, unparsed


def _check_hf_files(token: str) -> dict[str, Any]:
    """Проверка файлов модели: HEAD на каждый файл из setup.sh с заголовком Authorization.
    Перенаправления выполняются; заголовок с токеном не должен уйти на другой хост."""
    state: dict[str, Any] = {
        "status": None,
        "reason": None,
        "base_url": None,
        "base_source": None,
        "unparsed_lines": 0,
        "files": [],
    }
    say("=== Проверка файлов модели Hugging Face ===")

    if not token:
        state.update(status="пропущено", reason="HF_TOKEN не задан или пуст")
        err(
            "Ошибка: не задан токен Hugging Face — переменная окружения HF_TOKEN пуста или отсутствует. "
            "Проверка файлов модели пропущена; код выхода не меняется."
        )
        say("Статус: пропущено (HF_TOKEN не задан)")
        return state

    try:
        base, source, files, unparsed = _read_setup_models()
    except ProbeInputError as exc:
        state.update(status="ошибка разбора", reason=str(exc))
        err(
            f"Ошибка разбора списка моделей: {exc}. "
            "Проверка файлов модели не выполнена; код выхода не меняется."
        )
        say(f"Статус: ошибка разбора ({exc})")
        return state

    state.update(base_url=base, base_source=source, unparsed_lines=unparsed)
    say(f"Базовый адрес: {base} (источник: {source})")
    say(f"Файлов в списке: {len(files)}; строк download_model, которые не разобраны: {unparsed}")

    auth = {"Authorization": f"Bearer {token}"}
    original_host = urlsplit(base).hostname
    stopped = False
    with httpx.Client(timeout=HF_TIMEOUT_SEC, follow_redirects=True) as http:
        for index, (subdir, name) in enumerate(files, start=1):
            url = f"{base}/{subdir}/{name}"
            label = f"[{index}/{len(files)}] {subdir}/{name}"
            entry: dict[str, Any] = {"index": index, "subdir": subdir, "name": name, "url": url}
            try:
                resp = http.head(url, headers=auth)
            except Exception as exc:
                # Сбой одного файла не останавливает остальные; имя типа, без текста исключения.
                entry.update(status=None, error=type(exc).__name__)
                state["files"].append(entry)
                say(f"{label}: сетевая ошибка ({type(exc).__name__})")
                continue

            # Защита: адрес последнего запроса (после перенаправлений) не на другом хосте
            # с заголовком Authorization. httpx такой заголовок снимает (см. отчёт).
            if resp.request.url.host != original_host and "Authorization" in resp.request.headers:
                entry.update(status=resp.status_code, error="токен ушёл на другой хост")
                state["files"].append(entry)
                state.update(
                    status="остановлена",
                    reason="заголовок с токеном Bearer ушёл на другой хост при перенаправлении",
                )
                err(
                    "ОШИБКА БЕЗОПАСНОСТИ: заголовок с токеном Bearer (HF_TOKEN) ушёл на другой хост "
                    "при перенаправлении. Проверка файлов модели остановлена."
                )
                say("Статус: остановлена — угроза безопасности (токен ушёл на другой хост)")
                stopped = True
                break

            code = resp.status_code
            entry["status"] = code
            line = f"{label}: HTTP {code}"
            if code == 200:
                length = resp.headers.get("content-length")
                entry["content_length"] = length
                line += f"; Content-Length: {length or 'нет'}"
            note = _hf_file_note(code)
            if note:
                entry["interpretation"] = note
                line += f" — {note}"
            state["files"].append(entry)
            say(line)

    if not stopped:
        state["status"] = "выполнена"
        ok = sum(1 for f in state["files"] if f.get("status") == 200)
        say(f"Итог: проверено файлов {len(state['files'])}, доступно (200): {ok}")
    return state


def main() -> int:
    global _secrets

    key = os.environ.get(ENV_KEY_NAME, "").strip()
    if not key:
        err(
            f"Ошибка: не задан ключ Vast — переменная окружения {ENV_KEY_NAME} "
            "пуста или отсутствует. Проверка не запущена."
        )
        return 2
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    _secrets = sorted({s for s in (key, hf_token) if s}, key=len, reverse=True)

    try:
        offer_filter, show_count = _read_inputs()
        min_balance = _read_min_balance()
    except ProbeInputError as exc:
        err(f"Ошибка входных параметров: {exc}. Проверка не запущена.")
        return 2

    capture = _Capture()
    log = logging.getLogger(LOG_NAME)
    log.setLevel(logging.INFO)
    log.addHandler(capture)

    failed = False
    checks: dict[str, Any] = {}
    say("=== Проверка Vast.ai (только чтение) ===")
    say(f"Фильтр: {_filter_text(offer_filter)}")

    with VastClient() as client:
        # 1. Сырой ответ поиска: тот же запрос и тот же путь, что в production.
        try:
            raw = client._request(
                "POST", PATH_BUNDLES, json=VastClient._build_query(offer_filter)
            )
        except VastError as exc:
            err(f"Сбой поиска предложений Vast: {exc}")
            return 3

        if isinstance(raw, dict):
            say("Ключи верхнего уровня ответа: " + ", ".join(sorted(map(str, raw.keys()))))
            offers_raw = raw.get("offers")
        else:
            say(f"Ответ не является объектом JSON: тип {type(raw).__name__}")
            offers_raw = None
        if not isinstance(offers_raw, list):
            say("В ответе нет списка offers: считаем, что предложений 0.")
            offers_raw = []

        total = len(offers_raw)
        dict_offers = [o for o in offers_raw if isinstance(o, dict)]
        say(f"Предложений в ответе: {total}")
        if len(dict_offers) != total:
            say(f"Элементов, не являющихся объектами: {total - len(dict_offers)}")

        if offers_raw:
            first = offers_raw[0]
            if isinstance(first, dict):
                say("Ключи первого предложения: " + ", ".join(sorted(map(str, first.keys()))))
            else:
                say(f"Первый элемент не объект, тип {type(first).__name__}")
        else:
            say("Ключи первого предложения: предложений нет")

        say(
            f"Поля цены и трафика по {len(dict_offers)} объектам "
            "(сырые значения, единицы не пересчитаны):"
        )
        for field in PRICE_FIELDS:
            say(_field_line(dict_offers, field))

        say(f"Первые {show_count} предложений (сырые значения):")
        for index, offer in enumerate(offers_raw[:show_count], start=1):
            if not isinstance(offer, dict):
                say(f"  #{index}: не объект")
                continue
            parts = [f"{f}={_value(offer, f)}" for f in SHOW_FIELDS]
            parts += [f"{f}={_value(offer, f)}" for f in SHOW_OPTIONAL_FIELDS if f in offer]
            say(f"  #{index}: " + "; ".join(parts))

        # 2. Текущие production-правила: второй запрос тем же клиентом и тем же фильтром.
        try:
            passed = client.search_offers(offer_filter)
            say(f"Проходят текущие production-правила: {len(passed)}")
        except VastError as exc:
            failed = True
            err(f"Сбой поиска по текущим правилам: {exc}")
        summary = next(
            (m for m in reversed(capture.messages) if m.startswith(SUMMARY_PREFIX)), None
        )
        say(f"Строка итога поиска: {summary or 'не получена'}")

        # 3. Баланс и кредит аккаунта (GET на PATH_USER). Код выхода не меняет.
        say()
        account = _check_account(client, min_balance)

        # 4. Проверка путей Vast (GET на несуществующий экземпляр). Код выхода не меняет.
        say()
        checks["vast_instance_endpoints"] = _check_vast_paths(
            client._base_url, client._timeout, key
        )

    # 5. Проверка файлов модели Hugging Face (HEAD). Код выхода не меняет.
    say()
    checks["hf_model_files"] = _check_hf_files(hf_token)

    # Файл: сырой ответ, первые 200 предложений, проверки, два поля счёта; без полей с ключами, токенами и секретами.
    payload: Any = dict(raw) if isinstance(raw, dict) else {"response": raw}
    if isinstance(payload.get("offers"), list):
        payload["offers"] = payload["offers"][:RESULT_OFFERS_LIMIT]
    payload["checks"] = checks
    payload["account"] = account
    text = _clean(json.dumps(_sanitize(payload), ensure_ascii=False, indent=2, default=str))
    try:
        Path(RESULT_FILE).write_text(text, encoding="utf-8")
    except OSError as exc:
        err(f"Не удалось записать {RESULT_FILE}: {exc}")
        return 1
    say(
        f"Записан {RESULT_FILE}: предложений {min(total, RESULT_OFFERS_LIMIT)}; "
        "поля с признаками секретов удалены."
    )
    return 3 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        err("Неожиданная ошибка пробы:\n" + traceback.format_exc())
        sys.exit(1)
