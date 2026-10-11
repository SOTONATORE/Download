"""Диагностика Vast.ai: только чтение (SPEC 0.1, 0.5, 8.3).

Ничего не арендует, не создаёт, не меняет и не удаляет. Баланс аккаунта не запрашивается и не выводится.
Вызовы:
  • POST /api/v0/bundles/       — поиск предложений, сырой ответ (тот же запрос, что в production);
  • POST /api/v0/bundles/       — VastClient.search_offers по текущим правилам.

Ключ читается только из окружения (имя — ENV_KEY_NAME) и уходит только в заголовке
Authorization через клиент. Всё, что печатается или пишется, проходит замену значения
ключа на «***».

Приватные члены VastClient: _request (тот же путь запроса и обработка ошибок, что у клиента)
и _build_query (та же сборка запроса). Они нужны, чтобы сырой ответ совпадал с production.
Generate/ при этом не меняется.

Коды выхода: 0 — поиск выполнен; 2 — нет ключа или неверные входы (сеть не трогается);
3 — сбой вызова Vast; 1 — прочая ошибка (например, запись probe_result.json).

Входы — переменные окружения, все необязательные (значения по умолчанию совпадают с
Generate/config.default.yaml, кроме max_price и min_inet_mbps, которые заданы по запросу):
PROBE_GPU_NAME, PROBE_MIN_PRICE, PROBE_MAX_PRICE, PROBE_MIN_INET_MBPS,
PROBE_MIN_RELIABILITY, PROBE_SHOW_COUNT.
"""
from __future__ import annotations

import json
import logging
import math
import os
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]  # файл в Generate/tools/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Generate.core.vast_client import (  # noqa: E402
    ENV_KEY_NAME,
    PATH_BUNDLES,
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

# Значение ключа: заменяется на «***» во всём выводе. Задаётся в main() до первого вывода.
_secret = ""


class ProbeInputError(Exception):
    """Неверные переменные окружения пробы."""


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
    return text.replace(_secret, "***") if _secret else text


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


def main() -> int:
    global _secret

    key = os.environ.get(ENV_KEY_NAME, "").strip()
    if not key:
        err(
            f"Ошибка: не задан ключ Vast — переменная окружения {ENV_KEY_NAME} "
            "пуста или отсутствует. Проверка не запущена."
        )
        return 2
    _secret = key

    try:
        offer_filter, show_count = _read_inputs()
    except ProbeInputError as exc:
        err(f"Ошибка входных параметров: {exc}. Проверка не запущена.")
        return 2

    capture = _Capture()
    log = logging.getLogger(LOG_NAME)
    log.setLevel(logging.INFO)
    log.addHandler(capture)

    failed = False
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

    # Файл: сырой ответ, первые 200 предложений, без полей с ключами, токенами и секретами.
    payload: Any = dict(raw) if isinstance(raw, dict) else {"response": raw}
    if isinstance(payload.get("offers"), list):
        payload["offers"] = payload["offers"][:RESULT_OFFERS_LIMIT]
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
