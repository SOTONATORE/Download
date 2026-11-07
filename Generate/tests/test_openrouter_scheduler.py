"""Тесты планировщика отправок OpenRouter на фейковых часах. Сеть не используется, реальных пауз нет."""
import json

import pytest

from Generate.core.gemini_prompts import GeminiRuntimeError
from Generate.core.openrouter_transport import (
    DEFAULT_BLOCK_SEC, OpenRouterSendScheduler, OpenRouterTransport, _wait_for_key)

START = 10 * 86400 + 3600   # середина суток по UTC: все тесты внутри одних суток
GAP, PER_WINDOW, WINDOW = 3.0, 10, 60.0
MODEL = "model-a"
SCHEMA = {"type": "object"}
KEY_A = "test-key-1"
KEY_B = "test-key-2"


class FakeClock:
    """Часы, которые двигаются только вперёд: sleep() сдвигает их на запрошенное время."""

    def __init__(self, now=START):
        self.now = float(now)
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None, text=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")

    def json(self):
        if self._body is None:
            raise ValueError("тела JSON нет")
        return self._body


class FakeClient:
    def __init__(self, responses=()):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"headers": headers, "json": json})
        return self._responses.pop(0)

    def close(self):
        pass


def ok_body(content):
    return {"choices": [{"message": {"content": content}}]}


def make_scheduler(names, clock):
    return OpenRouterSendScheduler(names, send_gap_sec=GAP, sends_per_window=PER_WINDOW,
                                   window_sec=WINDOW, clock=clock, sleep=clock.sleep)


def make_transport(responses, clock, keys=((("OPENROUTER1", KEY_A)), ("OPENROUTER2", KEY_B))):
    client = FakeClient(responses)
    t = OpenRouterTransport(keys=list(keys), client=client, clock=clock, sleep=clock.sleep)
    return t, client


# --- чистое правило: три рабочих примера из плана ---

def test_example_1_gap_blocks_then_allows_at_6():
    # отправки в t=0 и t=3, сейчас t=3.5: нельзя; следующая в t=6
    sends = [0.0, 3.0]
    assert _wait_for_key(3.5, sends, 3.0, 0.0, GAP, PER_WINDOW, WINDOW) == pytest.approx(2.5)
    assert _wait_for_key(6.0, sends, 3.0, 0.0, GAP, PER_WINDOW, WINDOW) == 0.0


def test_example_2_full_window_allows_at_60():
    # десять отправок в t=0..27 с шагом 3, сейчас t=28: нельзя; следующая в t=60
    sends = [float(3 * i) for i in range(10)]
    assert _wait_for_key(28.0, sends, 27.0, 0.0, GAP, PER_WINDOW, WINDOW) == pytest.approx(32.0)
    assert _wait_for_key(60.0, sends, 27.0, 0.0, GAP, PER_WINDOW, WINDOW) == 0.0


def test_example_3_empty_window_allowed_now():
    # последняя отправка в t=0, сейчас t=100, окно пусто: можно сейчас
    assert _wait_for_key(100.0, [0.0], 0.0, 0.0, GAP, PER_WINDOW, WINDOW) == 0.0


# --- планировщик: пауза, окно, блокировка, суточная квота ---

def test_send_gap_on_single_key_waits_three_seconds():
    clock = FakeClock()
    s = make_scheduler(["K1"], clock)
    assert s.acquire_send(MODEL) == "K1"
    assert s.acquire_send(MODEL) == "K1"
    assert clock.sleeps == [pytest.approx(3.0)]


def test_sliding_window_of_ten_waits_until_oldest_is_sixty_seconds_old():
    clock = FakeClock()
    s = make_scheduler(["K1"], clock)
    for i in range(10):
        clock.now = START + 3 * i        # отправки в t=0..27 относительно старта
        assert s.acquire_send(MODEL) == "K1"
    clock.now = START + 28
    s.acquire_send(MODEL)                # одиннадцатая: ждём до t=60
    assert clock.sleeps == [pytest.approx(32.0)]
    assert clock.now == pytest.approx(START + 60)


def test_429_with_retry_after_blocks_only_that_key():
    clock = FakeClock()
    t, client = make_transport(
        [FakeResponse(429, text="Too many requests", headers={"Retry-After": "7"})]
        + [FakeResponse(200, ok_body('{"segments": []}')) for _ in range(6)], clock)
    with pytest.raises(GeminiRuntimeError) as info:
        t.generate_json(MODEL, "s", "u", SCHEMA)
    assert info.value.status == 429 and info.value.daily_quota is False
    assert info.value.retry_after == 7.0
    used = []   # (момент вызова, заголовок Authorization) для каждого последующего запроса
    for _ in range(6):
        t.generate_json(MODEL, "s", "u", SCHEMA)
        # планировщик ждёт внутри вызова, поэтому момент отправки — время ПОСЛЕ вызова (post его не сдвигает)
        used.append((clock.now, client.calls[-1]["headers"]["Authorization"]))
    # пока блокировка первого ключа не истекла (до START+7), запросы уходят только на второй ключ
    before_block = [k for at, k in used if at < START + 7]
    assert before_block and all(k == f"Bearer {KEY_B}" for k in before_block)
    # после истечения блокировки первый ключ снова участвует
    assert any(k == f"Bearer {KEY_A}" for at, k in used if at >= START + 7)


def test_429_without_retry_after_uses_default_block():
    clock = FakeClock()
    s = make_scheduler(["K1"], clock)
    s.block_key("K1", DEFAULT_BLOCK_SEC)
    s.acquire_send(MODEL)
    assert clock.sleeps == [pytest.approx(DEFAULT_BLOCK_SEC)]


def test_daily_quota_exhausts_key_for_model_then_all_keys_raise():
    clock = FakeClock()
    t, client = make_transport([
        FakeResponse(429, text="Free-models-per-day limit exceeded"),
        FakeResponse(429, text="Free-models-per-day limit exceeded"),
    ], clock)
    with pytest.raises(GeminiRuntimeError) as first:
        t.generate_json(MODEL, "s", "u", SCHEMA)
    assert first.value.key_exhausted_only is True
    with pytest.raises(GeminiRuntimeError) as second:
        t.generate_json(MODEL, "s", "u", SCHEMA)
    assert second.value.key_exhausted_only is True
    with pytest.raises(GeminiRuntimeError) as third:
        t.generate_json(MODEL, "s", "u", SCHEMA)
    assert third.value.daily_quota is True
    assert third.value.code == "RESOURCE_EXHAUSTED"
    assert len(client.calls) == 2                # третий вызов до сети не дошёл
