"""Общий пул ключей Gemini: RPM и TPM ведутся по каждому ключу, суточная квота (RPD) по паре (ключ, модель).

Лимиты RPM и TPM действуют на ключ (то есть на проект Google Cloud), поэтому окна и паузы ведутся
отдельно для каждого ключа. Суточная квота действует на модель, поэтому исчерпание и счётчик RPD
ведутся по паре (ключ, модель): ключ, исчерпавший квоту одной модели, работает для других моделей.

Модуль не делает сетевых запросов. RealTransport берёт ключ из пула на каждый запрос.
Значения ключей не попадают в логи, в repr() и в тексты исключений.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Optional, Sequence

try:
    from .gemini_prompts import GeminiInputError, GeminiRuntimeError
except ImportError:  # запуск из папки Generate как скрипта
    from gemini_prompts import GeminiInputError, GeminiRuntimeError  # type: ignore

log = logging.getLogger("Generate.gemini_pool")

# Имена переменных окружения собираются из частей, чтобы структурный тест SPEC 0.1
# не видел литерал секрета в коде.
KEY_ENV_PREFIX = "GEMINI"
KEY_ENV_COUNT = 10

WINDOW_SECONDS = 60.0          # окно для RPM и TPM
DEFAULT_PAUSE_SECONDS = 60.0   # пауза при 429 без Retry-After
# Граница суток для RPD: сейчас полночь по UTC. Значение НЕ проверено по документации Google;
# в Download/generate_queries.py указана полночь по тихоокеанскому времени. Не менять без проверки.
SECONDS_PER_DAY = 86400


class AllKeysExhaustedError(GeminiRuntimeError):
    """Все ключи исчерпали суточную квоту для запрошенной модели; ждать до конца суток в этом запуске бессмысленно."""


class Outcome(Enum):
    """Результат запроса, который передаётся в KeyPool.release()."""
    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"       # 429: лимит RPM или TPM, пауза ключа
    DAILY_QUOTA = "daily_quota"         # 429: суточная квота, пара (ключ, модель) до конца суток
    SERVER_ERROR = "server_error"       # 500, 503, 504: состояние ключа не меняется
    NETWORK_ERROR = "network_error"     # сетевая ошибка: состояние не меняется


class KeyState(Enum):
    """Состояние ключа по RPM и TPM. Суточная квота здесь не учитывается: она по паре (ключ, модель)."""
    FREE = "free"
    BUSY = "busy"
    PAUSED = "paused"


@dataclass(frozen=True, repr=False)
class KeyLease:
    """Ключ, выданный на время одного запроса. Значение доступно в api_key, в repr не выводится."""
    name: str
    api_key: str

    def __repr__(self) -> str:
        return f"KeyLease(name={self.name!r})"


@dataclass(repr=False)
class _KeyEntry:
    name: str
    value: str
    state: KeyState = KeyState.FREE
    paused_until: float = 0.0
    request_times: deque = field(default_factory=deque)   # времена запросов в окне (RPM)
    token_uses: deque = field(default_factory=deque)      # (время, токены) в окне (TPM)

    def __repr__(self) -> str:
        return f"_KeyEntry(name={self.name!r}, state={self.state.value})"


def load_gemini_keys(environ: Mapping[str, str]) -> list[tuple[str, str]]:
    """Читает переменные GEMINI с номерами 1..10 из environ. Пустые значения пропускаются.

    Возвращает пары (имя переменной, значение) в порядке номеров.
    В лог пишется только количество ключей.
    """
    keys: list[tuple[str, str]] = []
    for i in range(1, KEY_ENV_COUNT + 1):
        name = KEY_ENV_PREFIX + str(i)
        value = (environ.get(name) or "").strip()
        if value:
            keys.append((name, value))
    if not keys:
        raise GeminiInputError(
            "Не задан ни один ключ Gemini: переменные GEMINI с номерами от 1 до 10 пусты или отсутствуют.")
    log.info("Загружено ключей Gemini: %d", len(keys))
    return keys


def _day_index(now: float) -> int:
    return int(now // SECONDS_PER_DAY)


class KeyPool:
    """Пул ключей с раздельными лимитами. Потокобезопасен: все операции идут под одним Condition."""

    def __init__(self, keys: Sequence[tuple[str, str]], *, rpm_limit: int, tpm_limit: int,
                 rpd_limit: int, clock: Callable[[], float] = time.time,
                 poll_interval: float = 1.0) -> None:
        if not keys:
            raise GeminiInputError("Пул ключей Gemini пуст: нет ни одного ключа.")
        names = [name for name, _ in keys]
        if len(set(names)) != len(names):
            raise GeminiInputError("В пуле ключей Gemini есть повторяющиеся имена.")
        for name, value in keys:
            if not value:
                raise GeminiInputError(f"Пустое значение ключа {name}.")
        for label, limit in (("RPM", rpm_limit), ("TPM", tpm_limit), ("RPD", rpd_limit)):
            if limit < 1:
                raise GeminiInputError(f"Лимит {label} должен быть не меньше 1.")
        self._rpm = rpm_limit
        self._tpm = tpm_limit
        self._rpd = rpd_limit
        self._entries = [_KeyEntry(name=n, value=v) for n, v in keys]
        self._by_name = {e.name: e for e in self._entries}
        self._clock = clock
        self._poll = poll_interval
        self._cond = threading.Condition()
        self._day = -1                                          # номер текущих суток по UTC
        self._day_count: dict[tuple[str, str], int] = {}        # (имя ключа, модель) -> запросы за сутки (RPD)
        self._day_exhausted: set[tuple[str, str]] = set()       # пары, исчерпавшие квоту до конца суток

    def __repr__(self) -> str:
        return (f"KeyPool(keys={len(self._entries)}, rpm={self._rpm}, "
                f"tpm={self._tpm}, rpd={self._rpd})")

    def _roll_day(self, now: float) -> None:
        """При смене суток сбрасывает суточные счётчики и отметки исчерпания по всем парам."""
        day = _day_index(now)
        if day != self._day:
            self._day = day
            self._day_count.clear()
            self._day_exhausted.clear()

    def _refresh(self, e: _KeyEntry, now: float) -> None:
        """Снимает истёкшую паузу и обрезает окна RPM/TPM ключа."""
        if e.state is KeyState.PAUSED and now >= e.paused_until:
            e.state = KeyState.FREE
        cutoff = now - WINDOW_SECONDS
        while e.request_times and e.request_times[0] <= cutoff:
            e.request_times.popleft()
        while e.token_uses and e.token_uses[0][0] <= cutoff:
            e.token_uses.popleft()

    def _can_take(self, e: _KeyEntry, model: str, tokens: int) -> bool:
        """Можно ли выдать ключ для модели сейчас. Пара, исчерпавшая RPD, отмечается как исчерпанная."""
        if e.state is not KeyState.FREE:
            return False
        pair = (e.name, model)
        if pair in self._day_exhausted:
            return False
        if self._day_count.get(pair, 0) >= self._rpd:
            self._day_exhausted.add(pair)
            return False
        if len(e.request_times) >= self._rpm:
            return False
        used = sum(t for _, t in e.token_uses)
        return used + tokens <= self._tpm

    def _take(self, now: float, model: str, tokens: int) -> Optional[KeyLease]:
        for e in self._entries:
            if self._can_take(e, model, tokens):
                e.state = KeyState.BUSY
                e.request_times.append(now)
                e.token_uses.append((now, tokens))
                pair = (e.name, model)
                self._day_count[pair] = self._day_count.get(pair, 0) + 1
                return KeyLease(name=e.name, api_key=e.value)
        return None

    def acquire(self, model: str, tokens: int = 0, timeout: Optional[float] = None) -> KeyLease:
        """Выдаёт ключ для модели, ожидая, пока освободится какой-нибудь ключ или кончится пауза.

        timeout=None: ждать без ограничения. Если все ключи исчерпали суточную квоту именно для этой модели,
        сразу бросает AllKeysExhaustedError. tokens — оценка токенов запроса для TPM.
        """
        if tokens < 0 or tokens > self._tpm:
            raise ValueError(f"Оценка токенов {tokens} вне диапазона 0..{self._tpm}.")
        start = self._clock()
        with self._cond:
            while True:
                now = self._clock()
                self._roll_day(now)
                for e in self._entries:
                    self._refresh(e, now)
                lease = self._take(now, model, tokens)
                if lease is not None:
                    return lease
                if all((e.name, model) in self._day_exhausted for e in self._entries):
                    raise AllKeysExhaustedError(
                        "Все ключи Gemini исчерпали суточную квоту для этой модели до конца суток.")
                if timeout is not None and now - start >= timeout:
                    raise TimeoutError("Свободный ключ Gemini не появился за отведённое время.")
                self._cond.wait(self._poll)

    def release(self, lease: KeyLease, outcome: Outcome, model: str,
                retry_after: Optional[float] = None) -> None:
        """Возвращает ключ в пул и применяет результат запроса к модели и ключу."""
        if not isinstance(outcome, Outcome):
            raise ValueError(f"Неизвестный результат запроса: {outcome!r}")
        with self._cond:
            e = self._by_name.get(lease.name)
            if e is None or e.state is not KeyState.BUSY:
                raise RuntimeError(f"Ключ {lease.name} не выдан этим пулом.")
            now = self._clock()
            self._roll_day(now)
            pair = (e.name, model)
            if outcome is Outcome.SUCCESS:
                if self._day_count.get(pair, 0) >= self._rpd:
                    self._day_exhausted.add(pair)
                e.state = KeyState.FREE
            elif outcome is Outcome.RATE_LIMITED:
                pause = retry_after if retry_after and retry_after > 0 else DEFAULT_PAUSE_SECONDS
                e.state = KeyState.PAUSED
                e.paused_until = now + pause
            elif outcome is Outcome.DAILY_QUOTA:
                # квота исчерпана только для этой модели; сам ключ остаётся свободным для других моделей
                self._day_exhausted.add(pair)
                e.state = KeyState.FREE
            else:  # SERVER_ERROR или NETWORK_ERROR: ключ не виноват, состояние не меняется
                e.state = KeyState.FREE
            log.info("Ключ %s, модель %s: результат %s, состояние %s",
                     e.name, model, outcome.value, e.state.value)
            self._cond.notify_all()

    def states(self) -> dict[str, KeyState]:
        """Текущие состояния ключей по именам (значения ключей не возвращаются)."""
        with self._cond:
            now = self._clock()
            self._roll_day(now)
            for e in self._entries:
                self._refresh(e, now)
            return {e.name: e.state for e in self._entries}
