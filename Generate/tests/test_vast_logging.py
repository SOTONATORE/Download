"""Офлайн-тесты отладочного журнала сырого обмена с Vast (SPEC 0.1, 10).

Сеть не используется: httpx.MockTransport. Проверяется маскирование заголовка Authorization,
ключей тела и параметров запроса, парность запроса и ответа по номеру, отсутствие лишних строк
при выключенном журнале, текст ответа, который не является JSON, полный вывод тела без обрезки,
сбои (таймаут, сеть).
"""
from __future__ import annotations

import json
import logging
import re
import socket

import httpx
import pytest

try:
    from Generate.core import vast_client as vc
    from Generate.core.vast_client import OfferFilter, VastClient, VastRuntimeError
except ImportError:  # плоский импорт
    from core import vast_client as vc  # type: ignore
    from core.vast_client import OfferFilter, VastClient, VastRuntimeError  # type: ignore

# Имя переменной собрано из частей: литерал GEN_..._API_KEY в коде Generate/ запрещён (test_structure).
ENV_KEY_NAME = "GEN_" + "VAST_API_KEY"
SECRET = "TESTKEY-12345"
WORKER_TOKEN = "WORKER-TOKEN-VALUE-987"
HF_VALUE = "HF-VALUE-654"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("Сетевое соединение в офлайн-тесте запрещено")
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture(autouse=True)
def vast_key(monkeypatch):
    monkeypatch.setenv(ENV_KEY_NAME, SECRET)


@pytest.fixture(autouse=True)
def capture_debug(caplog):
    caplog.set_level(logging.DEBUG)   # корневой уровень: журнал виден при любом уровне логгера
    return caplog


def make(handler, debug: bool = True) -> VastClient:
    return VastClient(client=httpx.Client(transport=httpx.MockTransport(handler)),
                      debug_logs=debug)


def all_text(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records)


def debug_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith(("Запрос Vast", "Ответ Vast"))]


def _bundle(i: int = 1) -> dict:
    return {"id": i, "gpu_name": "RTX 5090", "num_gpus": 1, "gpu_ram": 24576,
            "dph_total": 0.5, "reliability2": 0.99, "inet_down": 3000.0, "inet_up": 1000.0,
            "inet_up_cost": 0.001, "inet_down_cost": 0.001, "machine_id": 100 + i}


# ---------------------------------------------------------------- чистые функции маскирования

def test_redact_value_is_recursive_and_pure():
    body = {"a": {"api_key": "x", "items": [{"password": "p", "n": 1}]}, "Token": "t"}
    out = vc.redact_value(body)
    assert out == {"a": {"api_key": "***", "items": [{"password": "***", "n": 1}]},
                   "Token": "***"}
    assert body["Token"] == "t"   # исходный объект не изменён
    assert vc.is_sensitive_name("HF_TOKEN") and vc.is_sensitive_name("x-api-key")
    assert not vc.is_sensitive_name("gpu_name")


def test_redact_headers_and_query_by_name():
    headers = vc.redact_headers({"Authorization": f"Bearer {SECRET}", "Accept": "application/json",
                                 "X-Api-Key": SECRET}, SECRET)
    assert headers == {"Authorization": "***", "Accept": "application/json", "X-Api-Key": "***"}
    query = vc.redact_query({"after_token": "CUR-1", "limit": 25,
                             "select_filters": json.dumps({"auth_secret": "s"})})
    assert query["after_token"] == "***"
    assert query["limit"] == 25
    assert json.loads(query["select_filters"]) == {"auth_secret": "***"}


# ---------------------------------------------------------------- маскирование запроса

def test_authorization_header_and_nested_body_keys_are_redacted(caplog):
    sent: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        # в самом запросе ключ уходит в заголовок, как и раньше; маскируется только лог
        assert SECRET in request.headers["authorization"]
        if request.content:
            body = json.loads(request.content)
            sent.append(body)
        return httpx.Response(200, json={"new_contract": 42, "success": True})

    client = make(handler)
    inst_id = client.create_instance(
        7, "img:latest", 100,
        env_vars={"WORKER_TOKEN": WORKER_TOKEN, "HF_TOKEN": HF_VALUE, "GEN_LOG_LEVEL": "DEBUG"},
        label="gen-1")
    assert inst_id == 42
    # отправляемый запрос не меняется: env с переменными карты уходит как прежде (этап 9а)
    assert sent and sent[0]["env"] == {"WORKER_TOKEN": WORKER_TOKEN, "HF_TOKEN": HF_VALUE, "GEN_LOG_LEVEL": "DEBUG"}
    text = "\n".join(debug_lines(caplog))
    assert SECRET not in text and WORKER_TOKEN not in text and HF_VALUE not in text
    assert '"Authorization": "***"' in text
    # поле env из лога удалено целиком (этап 9а, решение владельца 18.10.2026); остальные поля тела остаются
    assert '"env"' not in text and "WORKER_TOKEN" not in text and "HF_TOKEN" not in text
    assert "GEN_LOG_LEVEL" not in text
    assert '"label": "gen-1"' in text and '"img:latest"' in text


def test_token_like_query_parameter_is_redacted(caplog):
    cursor = "CURSOR-VALUE-42"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("after_token") is None:
            return httpx.Response(200, json={"instances": [], "next_token": cursor})
        return httpx.Response(200, json={"instances": []})

    make(handler).list_instances()
    second = [m for m in debug_lines(caplog) if m.startswith("Запрос Vast #2")]
    assert len(second) == 1
    assert '"after_token": "***"' in second[0]
    assert '"limit": 25' in second[0]          # несекретный параметр остаётся
    assert cursor not in all_text(caplog)


# ---------------------------------------------------------------- парность и номера

def test_request_and_response_are_logged_with_matching_numbers(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"offers": [_bundle(1)]})
        return httpx.Response(200, json={"credit": 3.5})

    client = make(handler)
    offers = client.search_offers(OfferFilter())
    assert client.get_balance_usd() == 3.5
    assert len(offers) == 1

    lines = debug_lines(caplog)
    req_nums = [int(re.search(r"Запрос Vast #(\d+)", m).group(1))
                for m in lines if m.startswith("Запрос Vast")]
    resp_nums = [int(re.search(r"Ответ Vast #(\d+)", m).group(1))
                 for m in lines if m.startswith("Ответ Vast")]
    assert req_nums == [1, 2] and resp_nums == [1, 2]
    assert "POST /api/v0/bundles/" in lines[0]
    assert "GET /api/v0/users/current/" in lines[2]
    assert "HTTP 200" in lines[1] and re.search(r"за \d+\.\d{3} с", lines[1])


def test_debug_off_adds_no_log_records_and_keeps_return_value(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"offers": [_bundle(1)]})

    caplog.clear()
    off_offers = make(handler, debug=False).search_offers(OfferFilter())
    off_records = list(caplog.records)
    caplog.clear()
    on_offers = make(handler, debug=True).search_offers(OfferFilter())
    on_records = list(caplog.records)

    assert off_offers == on_offers
    assert not any(r.getMessage().startswith(("Запрос Vast", "Ответ Vast")) for r in off_records)
    assert len(on_records) - len(off_records) == 2   # ровно строка запроса и строка ответа


# ---------------------------------------------------------------- тела ответа

def test_non_json_body_logs_only_code_time_and_length(caplog):
    body = "<html>технические работы у провайдера</html>"
    client = make(lambda r: httpx.Response(200, text=body))
    with pytest.raises(VastRuntimeError):
        client.get_balance_usd()          # поведение прежнее: не JSON остаётся ошибкой
    text = "\n".join(debug_lines(caplog))
    assert "HTTP 200" in text
    assert f"тело {len(body.encode('utf-8'))} байт" in text
    assert "технические работы у провайдера" not in text and "<html>" not in text


def test_non_json_body_secrets_are_masked(caplog):
    body = f"ошибка шлюза: auth_token=GW-VALUE-55, эхо ключа {SECRET}"
    client = make(lambda r: httpx.Response(200, text=body))
    with pytest.raises(VastRuntimeError):
        client.get_balance_usd()
    text = "\n".join(debug_lines(caplog))
    assert "HTTP 200" in text
    assert "GW-VALUE-55" not in text and SECRET not in text
    assert "ошибка шлюза" not in text                  # текст тела не логируется вообще


def test_only_whitelisted_fields_are_logged_for_large_body(caplog):
    # длинное тело (около 200 КБ): в лог попадают только поля белого списка, остальное отбрасывается
    payload = {"credit": 5.0, "note": "полный текст ответа, который не должен попасть в лог",
               "blob": "b" * 200_000}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    assert make(lambda r: httpx.Response(200, content=raw,
                                         headers={"content-type": "application/json"})
                ).get_balance_usd() == 5.0
    lines = debug_lines(caplog)
    text = "\n".join(lines)
    assert '"credit": 5.0' in text                       # поле из белого списка есть
    assert "note" not in text and "полный текст ответа" not in text   # лишний ключ и его текст отброшены
    assert "bbbbbbbbbb" not in text and len(text) < 2_000             # 200 КБ blob в лог не попадает
    assert f"тело {len(raw)} байт" in text               # длина тела пишется


# ---------------------------------------------------------------- сбои и ответы с ключом

@pytest.mark.parametrize(
    "exc_factory, marker",
    [
        (lambda req: httpx.ReadTimeout("таймаут", request=req), "таймаут, за"),
        (lambda req: httpx.ConnectError("сбой соединения", request=req),
         "сетевая ошибка (ConnectError), за"),
    ],
)
def test_timeout_and_network_error_are_logged_without_exception_text(caplog, exc_factory, marker):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc_factory(request)

    client = make(handler)
    with pytest.raises(VastRuntimeError) as ei:
        client.get_balance_usd()
    lines = debug_lines(caplog)
    assert len(lines) == 2   # строка запроса и строка сбоя
    assert re.search(r"Ответ Vast #1: " + re.escape(marker) + r" \d+\.\d{3} с, тела нет",
                     lines[1])
    assert SECRET not in all_text(caplog) and SECRET not in str(ei.value)


def test_secret_echoed_in_error_response_is_masked(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad header {request.headers['Authorization']}")

    client = make(handler)
    with pytest.raises(VastRuntimeError) as ei:
        client.get_balance_usd()
    text = "\n".join(debug_lines(caplog))
    assert "HTTP 401" in text
    assert SECRET not in text and SECRET not in str(ei.value)


# ---------------------------------------------------------------- пары [имя, значение], extra_env, crisp_hmac

EXTRA_TOKEN = "EXTRA-TOKEN-VALUE-321"
CRISP_VALUE = "CRISP-HMAC-VALUE-456"


def test_pure_redaction_of_pairs_extra_env_and_crisp_hmac():
    body = {"extra_env": [["WORKER_TOKEN", EXTRA_TOKEN], ["GEN_LOG_LEVEL", "DEBUG"]],
            "crisp_hmac": CRISP_VALUE, "label": "gen-1"}
    out = vc.redact_value(body)
    assert out == {"extra_env": "***", "crisp_hmac": "***", "label": "gen-1"}
    assert body["extra_env"][0][1] == EXTRA_TOKEN          # исходный объект не изменён
    # пара с секретным именем: маскируется значение, имя остаётся; соседние пары не трогаются
    assert vc.redact_value([["WORKER_TOKEN", EXTRA_TOKEN], ["GEN_LOG_LEVEL", "DEBUG"]]) == \
        [["WORKER_TOKEN", "***"], ["GEN_LOG_LEVEL", "DEBUG"]]
    assert vc.redact_value({"env_pairs": [["HF_TOKEN", EXTRA_TOKEN]]}) == {"env_pairs": [["HF_TOKEN", "***"]]}
    assert vc.is_sensitive_name("extra_env") and vc.is_sensitive_name("crisp_hmac")
    assert vc.is_sensitive_name("x_hmac") and vc.is_sensitive_name("crisp")
    assert not vc.is_sensitive_name("gpu_name") and not vc.is_sensitive_name("label")


def test_crisp_hmac_in_free_text_is_masked():
    text = vc.redact_text(f"crisp_hmac={CRISP_VALUE} gpu=RTX")
    assert CRISP_VALUE not in text and "gpu=RTX" in text


def test_extra_env_pairs_in_response_are_not_logged(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"instances": [{
            "id": 7, "actual_status": "running", "intended_status": "running", "label": "gen-1",
            "extra_env": [["WORKER_TOKEN", EXTRA_TOKEN], ["GEN_LOG_LEVEL", "DEBUG"]],
            "crisp_hmac": CRISP_VALUE}]})

    info = make(handler).get_instance(7)
    assert info.instance_id == 7
    lines = debug_lines(caplog)
    text = "\n".join(lines)
    assert any(m.startswith("Ответ Vast #1") for m in lines)
    assert EXTRA_TOKEN not in all_text(caplog)             # ни в одной строке журнала
    assert CRISP_VALUE not in all_text(caplog)
    # ключи вне белого списка не попадают в лог: ни имена, ни значения
    assert "extra_env" not in text and "label" not in text and "crisp_hmac" not in text
    assert '"env"' not in text
    # поля из белого списка остаются, контейнер «instances» отброшен, его поля видны на верхнем уровне
    assert '"actual_status": "running"' in text and '"id": 7' in text


def test_crisp_hmac_value_is_not_logged_outside_debug_too(caplog):
    """Журнал выключен: строк нет и значение не появляется ни в каком сообщении."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"instances": [{"id": 8, "crisp_hmac": CRISP_VALUE}]})

    make(handler, debug=False).get_instance(8)
    assert CRISP_VALUE not in all_text(caplog)
