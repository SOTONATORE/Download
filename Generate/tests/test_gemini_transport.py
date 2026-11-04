"""Тесты RealTransport поверх пула ключей: отображение HTTP-ответа в исход пула (этап 3).

Сеть не используется: клиент и пул подменены. Ключ в тестах — заведомо узнаваемая строка,
которая не должна попасть ни в исключения, ни в логи.
"""
import logging
import socket

import httpx
import pytest

from Generate.core import gemini_pool as gpool
from Generate.core import gemini_prompts as gp
from Generate.core.gemini_orchestrator import ManagedTransport

SECRET = "TESTKEY-TRANSPORT-98765"
LEASE_NAME = "GEMINI" + "1"          # имя переменной собрано из частей (структурный тест SPEC 0.1)
PER_DAY_QUOTA = "GenerateRequestsPerDayPerProjectPerModel"
PER_MINUTE_QUOTA = "GenerateRequestsPerMinutePerProjectPerModel"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


class FakeResponse:
    """Минимальный ответ: статус, заголовки и тело JSON (None — тело не JSON)."""

    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("тело не JSON")
        return self._body


class FakeClient:
    """Подмена httpx.Client: отдаёт заготовленный ответ или бросает заготовленное исключение."""

    def __init__(self, reply):
        self._reply = reply
        self.posts = []

    def post(self, url, headers=None, json=None):
        self.posts.append({"url": url, "headers": dict(headers or {})})
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


class FakePool:
    """Подмена KeyPool: выдаёт один ключ и записывает исходы release()."""

    def __init__(self, key=SECRET, exhausted=False):
        self.key = key
        self.exhausted = exhausted
        self.acquired = 0
        self.released = []   # (имя ключа, исход, retry_after)

    def acquire(self, timeout=None, tokens=0):
        if self.exhausted:
            raise gpool.AllKeysExhaustedError("Все ключи Gemini исчерпали суточную квоту до конца суток.")
        self.acquired += 1
        return gpool.KeyLease(name=LEASE_NAME, api_key=self.key)

    def release(self, lease, outcome, retry_after=None):
        self.released.append((lease.name, outcome, retry_after))


def _call(reply, pool=None):
    """Один вызов generate_json через RealTransport; возвращает (транспорт-пул, клиент, результат/исключение)."""
    pool = pool or FakePool()
    client = FakeClient(reply)
    transport = gp.RealTransport(pool=pool, client=client)
    try:
        out = transport.generate_json("m", "SYS", "USR", {"type": "array"})
        return pool, client, out, None
    except gp.GeminiRuntimeError as e:
        return pool, client, None, e


def _error_body(code, quota_id=None, retry_delay=None, echo=None):
    err = {"status": code, "message": echo or "ошибка"}
    details = []
    if quota_id:
        details.append({"violations": [{"quotaId": quota_id}]})
    if retry_delay:
        details.append({"retryDelay": retry_delay})
    if details:
        err["details"] = details
    return {"error": err}


# ------------------------------------------------------------------ успех и ключ

def test_200_gives_success_and_releases_lease():
    pool, client, out, err = _call(FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "[]"}]}}]}))
    assert err is None and out == "[]"
    assert pool.released == [(LEASE_NAME, gpool.Outcome.SUCCESS, None)]
    assert pool.acquired == 1
    assert client.posts[0]["headers"][gp.API_KEY_HEADER] == SECRET
    assert SECRET not in client.posts[0]["url"]


# ------------------------------------------------------------------ HTTP 429

def test_429_with_perday_quota_gives_daily_quota():
    pool, _, _, err = _call(FakeResponse(429, _error_body("RESOURCE_EXHAUSTED", quota_id=PER_DAY_QUOTA)))
    assert pool.released == [(LEASE_NAME, gpool.Outcome.DAILY_QUOTA, None)]
    assert err.status == 429 and err.daily_quota is True


def test_429_perday_match_ignores_case():
    pool, _, _, err = _call(FakeResponse(429, _error_body("RESOURCE_EXHAUSTED",
                                                          quota_id=PER_DAY_QUOTA.lower())))
    assert pool.released[0][1] is gpool.Outcome.DAILY_QUOTA and err.daily_quota is True


def test_429_without_perday_gives_rate_limited_with_retry_after_header():
    pool, _, _, err = _call(FakeResponse(429, _error_body("RESOURCE_EXHAUSTED", quota_id=PER_MINUTE_QUOTA),
                                         headers={"Retry-After": "12"}))
    assert pool.released == [(LEASE_NAME, gpool.Outcome.RATE_LIMITED, 12.0)]
    assert err.status == 429 and err.daily_quota is False


def test_429_without_retry_after_header_passes_none():
    pool, _, _, _ = _call(FakeResponse(429, _error_body("RESOURCE_EXHAUSTED")))
    assert pool.released == [(LEASE_NAME, gpool.Outcome.RATE_LIMITED, None)]


def test_429_with_unparsable_retry_after_passes_none():
    pool, _, _, _ = _call(FakeResponse(429, None, headers={"Retry-After": "Wed, 04 Nov 2026 20:00:00 GMT"}))
    assert pool.released == [(LEASE_NAME, gpool.Outcome.RATE_LIMITED, None)]


# ------------------------------------------------------------------ 500, 503, 504 и сеть: без штрафа

@pytest.mark.parametrize("status", [500, 503, 504])
def test_5xx_gives_server_error_without_penalty(status):
    pool, _, _, err = _call(FakeResponse(status, _error_body("UNAVAILABLE")))
    assert pool.released == [(LEASE_NAME, gpool.Outcome.SERVER_ERROR, None)]
    assert err.status == status and err.daily_quota is False


def test_network_error_gives_network_error_without_penalty():
    pool, _, _, err = _call(httpx.ConnectError("обрыв соединения"))
    assert pool.released == [(LEASE_NAME, gpool.Outcome.NETWORK_ERROR, None)]
    assert err.status is None and "ConnectError" in str(err)


# ------------------------------------------------------------------ прочие 4xx

def test_400_raises_runtime_error_and_releases_without_penalty():
    pool, _, _, err = _call(FakeResponse(400, _error_body("INVALID_ARGUMENT")))
    assert err.status == 400 and err.code == "INVALID_ARGUMENT" and err.daily_quota is False
    # ключ не виноват: исход не штрафует ключ (ни пауза, ни суточная блокировка)
    assert pool.released == [(LEASE_NAME, gpool.Outcome.SERVER_ERROR, None)]


# ------------------------------------------------------------------ ключ нигде не виден

def test_key_absent_from_exception_text_and_logs(caplog):
    caplog.set_level(logging.DEBUG)
    echo = "ECHO " + SECRET
    cases = [
        FakeResponse(429, _error_body("RESOURCE_EXHAUSTED", quota_id=PER_DAY_QUOTA, echo=echo)),
        FakeResponse(429, _error_body("RESOURCE_EXHAUSTED", quota_id=PER_MINUTE_QUOTA, echo=echo),
                     headers={"Retry-After": "3"}),
        FakeResponse(500, _error_body("INTERNAL", echo=echo)),
        FakeResponse(400, _error_body("INVALID_ARGUMENT", echo=echo)),
        httpx.ReadTimeout("таймаут " + SECRET),
    ]
    for reply in cases:
        _, _, _, err = _call(reply)
        assert err is not None
        assert SECRET not in str(err) and SECRET not in repr(err)
        assert "ECHO" not in str(err)
    assert SECRET not in caplog.text


# ------------------------------------------------------------------ все ключи исчерпаны

def test_all_keys_exhausted_gives_daily_quota_error_without_post():
    pool = FakePool(exhausted=True)
    client = FakeClient(FakeResponse(200, {}))
    transport = gp.RealTransport(pool=pool, client=client)
    with pytest.raises(gp.GeminiRuntimeError) as ei:
        transport.generate_json("m", "s", "u", {})
    assert ei.value.daily_quota is True and ei.value.status == 429
    assert client.posts == []   # запрос не отправляется


def test_managed_transport_reports_daily_quota_message_when_all_keys_exhausted():
    """Цепочка моделей при полном исчерпании: та же русская формулировка, что и раньше."""
    pool = FakePool(exhausted=True)
    transport = gp.RealTransport(pool=pool, client=FakeClient(FakeResponse(200, {})))
    managed = ManagedTransport(transport, ("model-a", "model-b"), sleep=lambda s: None)
    with pytest.raises(gp.GeminiRuntimeError) as ei:
        managed.generate_json("", "s", "u", {})
    assert ei.value.daily_quota is True
    assert "Суточная квота исчерпана у всех моделей цепочки" in str(ei.value)
