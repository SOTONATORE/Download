"""Тесты core/gemini_pool на фейковых часах и заглушках. Сеть не используется, реальных пауз нет."""
import logging
import threading

import pytest

from Generate.core.config_loader import load_config
from Generate.core.gemini_pool import (
    KEY_ENV_PREFIX, AllKeysExhaustedError, KeyPool, KeyState, Outcome, load_gemini_keys,
)
from Generate.core.gemini_prompts import GeminiInputError

# Середина суток по UTC: все тесты идут внутри одних суток, если не сказано иное.
START = 10 * 86400 + 3600
WINDOW = 60
SECRET = "SECRET-TEST-VALUE-XYZ"
MODEL = "model-a"            # модель, для которой берётся ключ (квота по паре ключ-модель)


class FakeClock:
    """Фейковые часы: время меняется только через advance() или присваивание now."""

    def __init__(self, now: float = START) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_pool(names=("K1", "K2"), clock=None, rpm=15, tpm=250000, rpd=500):
    clock = clock or FakeClock()
    keys = [(n, f"value-of-{n}") for n in names]
    pool = KeyPool(keys, rpm_limit=rpm, tpm_limit=tpm, rpd_limit=rpd,
                   clock=clock, poll_interval=0.001)
    return pool, clock


def test_one_key_is_not_handed_to_two_requests_at_once():
    pool, _ = make_pool(names=("K1",))
    first = pool.acquire(MODEL, timeout=0)
    with pytest.raises(TimeoutError):
        pool.acquire(MODEL, timeout=0)  # единственный ключ занят: второй запрос его не получает
    pool.release(first, Outcome.SUCCESS, MODEL)
    second = pool.acquire(MODEL, timeout=0)
    assert second.name == "K1"
    pool.release(second, Outcome.SUCCESS, MODEL)


def test_two_keys_are_handed_out_separately():
    pool, _ = make_pool(names=("K1", "K2"))
    a = pool.acquire(MODEL, timeout=0)
    b = pool.acquire(MODEL, timeout=0)
    assert {a.name, b.name} == {"K1", "K2"}


def test_rpm_is_counted_per_key():
    pool, clock = make_pool(rpm=2)
    a = pool.acquire(MODEL, timeout=0)
    pool.release(a, Outcome.SUCCESS, MODEL)
    b = pool.acquire(MODEL, timeout=0)
    pool.release(b, Outcome.SUCCESS, MODEL)
    assert a.name == b.name == "K1"       # K1 получил два запроса за окно
    c = pool.acquire(MODEL, timeout=0)
    assert c.name == "K2"                 # K1 исчерпал RPM, работает K2
    pool.release(c, Outcome.SUCCESS, MODEL)
    d = pool.acquire(MODEL, timeout=0)
    assert d.name == "K2"
    pool.release(d, Outcome.SUCCESS, MODEL)
    with pytest.raises(TimeoutError):
        pool.acquire(MODEL, timeout=0)           # оба ключа исчерпали RPM в этом окне
    clock.advance(WINDOW)
    e = pool.acquire(MODEL, timeout=0)
    assert e.name == "K1"                 # окно сдвинулось, K1 снова доступен


def test_tpm_is_counted_per_key():
    pool, _ = make_pool(tpm=1000)
    a = pool.acquire(MODEL, timeout=0, tokens=600)
    pool.release(a, Outcome.SUCCESS, MODEL)
    b = pool.acquire(MODEL, timeout=0, tokens=600)
    assert b.name == "K2"                 # K1 уже израсходовал 600 из 1000
    pool.release(b, Outcome.SUCCESS, MODEL)


def test_retry_after_pauses_only_that_key():
    pool, clock = make_pool()
    a = pool.acquire(MODEL, timeout=0)
    assert a.name == "K1"
    pool.release(a, Outcome.RATE_LIMITED, MODEL, retry_after=30)
    b = pool.acquire(MODEL, timeout=0)
    assert b.name == "K2"                 # K2 продолжает работать
    with pytest.raises(TimeoutError):
        pool.acquire(MODEL, timeout=0)           # K1 на паузе, K2 занят
    clock.advance(29)
    with pytest.raises(TimeoutError):
        pool.acquire(MODEL, timeout=0)           # пауза K1 ещё не закончилась
    clock.advance(1)
    c = pool.acquire(MODEL, timeout=0)
    assert c.name == "K1"                 # пауза K1 закончилась по Retry-After
    assert pool.states()["K2"] is KeyState.BUSY


def test_daily_quota_takes_key_out_until_day_ends():
    pool, clock = make_pool(names=("K1",))
    a = pool.acquire(MODEL, timeout=0)
    pool.release(a, Outcome.DAILY_QUOTA, MODEL)
    clock.advance(3600)                   # та же сутки по UTC
    with pytest.raises(AllKeysExhaustedError):
        pool.acquire(MODEL, timeout=0)
    clock.now = 11 * 86400 + 1            # начались следующие сутки
    b = pool.acquire(MODEL, timeout=0)
    assert b.name == "K1"


def test_all_keys_paused_acquire_waits_and_does_not_fail():
    pool, clock = make_pool()
    a = pool.acquire(MODEL, timeout=0)
    b = pool.acquire(MODEL, timeout=0)
    pool.release(a, Outcome.RATE_LIMITED, MODEL, retry_after=30)
    pool.release(b, Outcome.RATE_LIMITED, MODEL, retry_after=30)

    result, errors = [], []

    def worker():
        try:
            result.append(pool.acquire(MODEL))  # без timeout: ждёт
        except Exception as exc:  # pragma: no cover - фиксируем для диагностики
            errors.append(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    # Пока оба ключа на паузе, поток не может завершиться ни с ключом, ни с ошибкой.
    assert t.is_alive()
    clock.advance(30)
    t.join(timeout=5)
    assert not t.is_alive()
    assert not errors
    assert result[0].name in {"K1", "K2"}


def test_all_keys_exhausted_raises():
    pool, _ = make_pool()
    a = pool.acquire(MODEL, timeout=0)
    b = pool.acquire(MODEL, timeout=0)
    pool.release(a, Outcome.DAILY_QUOTA, MODEL)
    pool.release(b, Outcome.DAILY_QUOTA, MODEL)
    with pytest.raises(AllKeysExhaustedError):
        pool.acquire(MODEL, timeout=0)


def test_no_keys_refuses_start_without_printing_values(caplog):
    caplog.set_level(logging.INFO)
    environ = {KEY_ENV_PREFIX + str(i): "   " for i in range(1, 11)}  # пустые значения
    with pytest.raises(GeminiInputError) as info:
        load_gemini_keys(environ)
    assert "Не задан" in str(info.value)
    assert caplog.text == ""
    with pytest.raises(GeminiInputError):
        load_gemini_keys({})
    with pytest.raises(GeminiInputError):
        KeyPool([], rpm_limit=15, tpm_limit=250000, rpd_limit=500, clock=FakeClock())


def test_load_logs_only_the_count(caplog):
    caplog.set_level(logging.INFO, logger="Generate.gemini_pool")
    environ = {KEY_ENV_PREFIX + "1": SECRET, KEY_ENV_PREFIX + "3": SECRET}
    keys = load_gemini_keys(environ)
    assert len(keys) == 2
    assert SECRET not in caplog.text
    assert "2" in caplog.text


def test_repr_and_errors_do_not_show_key_values():
    pool = KeyPool([("K1", SECRET)], rpm_limit=15, tpm_limit=250000, rpd_limit=500,
                   clock=FakeClock(), poll_interval=0.001)
    lease = pool.acquire(MODEL, timeout=0)
    assert lease.api_key == SECRET
    assert SECRET not in repr(pool)
    assert SECRET not in repr(lease)
    pool.release(lease, Outcome.DAILY_QUOTA, MODEL)
    with pytest.raises(AllKeysExhaustedError) as info:
        pool.acquire(MODEL, timeout=0)
    assert SECRET not in str(info.value)


@pytest.mark.parametrize("outcome", [Outcome.SERVER_ERROR, Outcome.NETWORK_ERROR])
def test_server_and_network_errors_do_not_change_key_state(outcome):
    # Статусы 500, 503 и 504 сводятся к Outcome.SERVER_ERROR.
    pool, _ = make_pool(names=("K1",))
    a = pool.acquire(MODEL, timeout=0)
    pool.release(a, outcome, MODEL)
    assert pool.states()["K1"] is KeyState.FREE
    b = pool.acquire(MODEL, timeout=0)
    assert b.name == "K1"
    pool.release(b, Outcome.SUCCESS, MODEL)


def test_limits_come_from_config_values():
    cfg = load_config()
    pool = KeyPool([("K1", "value-of-K1")], rpm_limit=cfg.gemini_rpm_limit,
                   tpm_limit=cfg.gemini_tpm_limit, rpd_limit=cfg.gemini_rpd_limit,
                   clock=FakeClock(), poll_interval=0.001)
    # RPM: ровно cfg.gemini_rpm_limit запросов за окно, следующий не выдаётся.
    for _ in range(cfg.gemini_rpm_limit):
        lease = pool.acquire(MODEL, timeout=0)
        pool.release(lease, Outcome.SUCCESS, MODEL)
    with pytest.raises(TimeoutError):
        pool.acquire(MODEL, timeout=0)

    # TPM: один запрос на весь лимит токенов, следующий токен уже не помещается.
    tpm_pool = KeyPool([("K1", "value-of-K1")], rpm_limit=cfg.gemini_rpm_limit,
                       tpm_limit=cfg.gemini_tpm_limit, rpd_limit=cfg.gemini_rpd_limit,
                       clock=FakeClock(), poll_interval=0.001)
    tpm_pool.acquire(MODEL, timeout=0, tokens=cfg.gemini_tpm_limit)
    with pytest.raises(TimeoutError):
        tpm_pool.acquire(MODEL, timeout=0, tokens=1)

    # RPD: шаг 61 с держит запросы вне окна RPM, суточный лимит срабатывает точно.
    clock = FakeClock(START)
    day_pool = KeyPool([("K1", "value-of-K1")], rpm_limit=cfg.gemini_rpm_limit,
                       tpm_limit=cfg.gemini_tpm_limit, rpd_limit=cfg.gemini_rpd_limit,
                       clock=clock, poll_interval=0.001)
    for _ in range(cfg.gemini_rpd_limit):
        lease = day_pool.acquire(MODEL, timeout=0)
        day_pool.release(lease, Outcome.SUCCESS, MODEL)
        clock.advance(61)
    with pytest.raises(AllKeysExhaustedError):
        day_pool.acquire(MODEL, timeout=0)
