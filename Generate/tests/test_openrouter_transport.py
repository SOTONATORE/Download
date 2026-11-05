"""Тесты OpenRouterTransport на заглушке httpx-клиента и фейковых часах. Сеть не используется, реальных пауз нет."""
import json
import logging

import httpx
import pytest

from Generate.core.gemini_pool import KeyPool, Outcome
from Generate.core.gemini_prompts import GeminiInputError
from Generate.core.openrouter_transport import OpenRouterTransport, load_openrouter_keys

# Середина суток по UTC: все тесты идут внутри одних суток.
START = 10 * 86400 + 3600
KEY_A = "test-key-1"
KEY_B = "test-key-2"
MODEL = "model-a"
SCHEMA = {"type": "object"}


class FakeClock:
    def __init__(self, now: float = START) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None, text=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        if text is not None:
            self.text = text
        elif body is not None:
            self.text = json.dumps(body)
        else:
            self.text = ""

    def json(self):
        if self._body is None:
            raise ValueError("тела JSON нет")
        return self._body


class FakeClient:
    """Возвращает готовые ответы по очереди или бросает заданное исключение."""

    def __init__(self, responses=(), error=None):
        self._responses = list(responses)
        self._error = error
        self.calls = []
        self.closed = False

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if self._error is not None:
            raise self._error
        return self._responses.pop(0)

    def close(self):
        self.closed = True


def ok_body(content):
    return {"choices": [{"message": {"content": content}}]}


def make_transport(responses=(), error=None, **kwargs):
    clock = FakeClock()
    keys = [("OPENROUTER1", KEY_A), ("OPENROUTER2", KEY_B)]
    pool = KeyPool(keys, rpm_limit=20, tpm_limit=1_000_000, rpd_limit=1000,
                   clock=clock, poll_interval=0.001)
    released = []
    real_release = pool.release

    def spy_release(lease, outcome, model, retry_after=None):
        released.append((outcome, retry_after))
        return real_release(lease, outcome, model, retry_after=retry_after)

    pool.release = spy_release  # запись исходов для проверок
    client = FakeClient(responses, error)
    transport = OpenRouterTransport(pool=pool, client=client, **kwargs)
    return transport, client, released


def test_200_returns_content_and_releases_success():
    t, client, released = make_transport([FakeResponse(200, ok_body('{"segments": []}'))])
    assert t.generate_json(MODEL, "sys", "usr", SCHEMA) == '{"segments": []}'
    assert released == [(Outcome.SUCCESS, None)]
    call = client.calls[0]
    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {KEY_A}"
    assert call["json"]["model"] == MODEL
    assert call["json"]["messages"] == [{"role": "system", "content": "sys"},
                                        {"role": "user", "content": "usr"}]
    assert call["json"]["max_tokens"] == 4000
    assert call["json"]["response_format"] == {
        "type": "json_schema", "json_schema": {"name": "segments", "schema": SCHEMA}}


def test_json_object_mode_when_schema_disabled():
    t, client, _ = make_transport([FakeResponse(200, ok_body("{}"))], use_json_schema=False)
    t.generate_json(MODEL, "s", "u", SCHEMA)
    assert client.calls[0]["json"]["response_format"] == {"type": "json_object"}


def test_429_plain_body_is_rate_limited_with_retry_after():
    t, _, released = make_transport(
        [FakeResponse(429, text="Too many requests", headers={"Retry-After": "7"})])
    with pytest.raises(Exception) as info:
        t.generate_json(MODEL, "s", "u", SCHEMA)
    err = info.value
    assert err.status == 429
    assert err.daily_quota is False
    assert err.retry_after == 7.0
    assert not getattr(err, "key_exhausted_only", False)
    assert released == [(Outcome.RATE_LIMITED, 7.0)]


def test_429_with_daily_in_body_is_daily_quota_key_exhausted_only():
    t, _, released = make_transport(
        [FakeResponse(429, text="Free-models-per-day limit exceeded")])
    with pytest.raises(Exception) as info:
        t.generate_json(MODEL, "s", "u", SCHEMA)
    err = info.value
    assert err.status == 429
    assert err.daily_quota is True
    assert err.key_exhausted_only is True
    assert released == [(Outcome.DAILY_QUOTA, None)]


def test_429_log_redacts_key_like_text(caplog):
    caplog.set_level(logging.WARNING, logger="Generate.openrouter_transport")
    body = f"daily limit exceeded, key {KEY_A}, Bearer sk-or-abcdef123456, AIzaSySECRETVALUE99"
    t, _, _ = make_transport([FakeResponse(429, text=body)])
    with pytest.raises(Exception):
        t.generate_json(MODEL, "s", "u", SCHEMA)
    assert KEY_A not in caplog.text
    assert "sk-or-abcdef123456" not in caplog.text
    assert "AIzaSySECRETVALUE99" not in caplog.text
    assert "***" in caplog.text


def test_402_raises_payment_required_not_transient():
    t, _, released = make_transport([FakeResponse(402, text="insufficient credits")])
    with pytest.raises(Exception) as info:
        t.generate_json(MODEL, "s", "u", SCHEMA)
    err = info.value
    assert err.status == 402
    assert err.code == "PAYMENT_REQUIRED"
    assert err.daily_quota is False
    assert err.status not in (500, 503, 504)
    assert released == [(Outcome.SERVER_ERROR, None)]


def test_401_raises_unauthorized_and_does_not_leak_key(caplog):
    caplog.set_level(logging.ERROR, logger="Generate.openrouter_transport")
    t, _, released = make_transport([FakeResponse(401, text="invalid key")])
    with pytest.raises(Exception) as info:
        t.generate_json(MODEL, "s", "u", SCHEMA)
    err = info.value
    assert err.status == 401
    assert err.code == "UNAUTHORIZED"
    assert released == [(Outcome.SERVER_ERROR, None)]
    assert KEY_A not in str(err)
    assert KEY_A not in caplog.text


def test_503_raises_unavailable():
    t, _, released = make_transport([FakeResponse(503, text="busy")])
    with pytest.raises(Exception) as info:
        t.generate_json(MODEL, "s", "u", SCHEMA)
    assert info.value.status == 503
    assert info.value.code == "UNAVAILABLE"
    assert released == [(Outcome.SERVER_ERROR, None)]


def test_network_error_raises_code_network():
    t, _, released = make_transport(error=httpx.ConnectError("boom"))
    with pytest.raises(Exception) as info:
        t.generate_json(MODEL, "s", "u", SCHEMA)
    assert info.value.status is None
    assert info.value.code == "NETWORK"
    assert released == [(Outcome.NETWORK_ERROR, None)]


def test_empty_choices_raises_empty_response():
    t, _, released = make_transport([FakeResponse(200, {"choices": []})])
    with pytest.raises(Exception) as info:
        t.generate_json(MODEL, "s", "u", SCHEMA)
    assert info.value.code == "EMPTY_RESPONSE"
    assert info.value.status == 500
    assert released == [(Outcome.SERVER_ERROR, None)]


def test_load_keys_none_set_raises_and_names_variables():
    with pytest.raises(GeminiInputError) as info:
        load_openrouter_keys({})
    assert "OPENROUTER1" in str(info.value)


def test_load_keys_skips_empty_values():
    keys = load_openrouter_keys({"OPENROUTER1": "", "OPENROUTER2": "   ", "OPENROUTER3": "test-key-3"})
    assert keys == [("OPENROUTER3", "test-key-3")]


def test_repr_and_exceptions_do_not_contain_key_value():
    t, _, _ = make_transport([FakeResponse(401, text="invalid key")])
    with pytest.raises(Exception) as info:
        t.generate_json(MODEL, "s", "u", SCHEMA)
    assert KEY_A not in repr(t)
    assert KEY_B not in repr(t)
    assert KEY_A not in str(info.value)
    assert KEY_A not in repr(info.value)


def test_close_does_not_close_injected_client():
    t, client, _ = make_transport()
    t.close()
    assert client.closed is False
