"""Этап 4б: оркестратор генерации промптов (лимитер, цепочка моделей, prompts.json).

Только стандартная библиотека + модуль gemini_prompts (httpx живёт там).
Ключ API здесь не читается и не упоминается: им владеет RealTransport (SPEC 0.1).
Тексты промптов в логи не пишутся. Все сообщения на русском (SPEC 0.5).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:  # пакетный и «плоский» импорт (как в остальных модулях core)
    from . import gemini_prompts as gp
    from .srt_parser import Segment, srt_hash
    from .timing import compute_timings
except ImportError:  # pragma: no cover
    import gemini_prompts as gp  # type: ignore
    from srt_parser import Segment, srt_hash  # type: ignore
    from timing import compute_timings  # type: ignore

log = logging.getLogger("Generate.gemini_orchestrator")

DEFAULT_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
WINDOW_SEC = 60.0
CHARS_PER_TOKEN = 3          # консервативная оценка входа
MAX_WAIT_SEC = 120.0         # потолок одной паузы после 429


# ---------------------------------------------------------------------------
# Лимитер RPM/TPM (скользящее окно 60 с)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Не пускает больше rpm запросов и tpm токенов за любые 60 с. clock/sleep подменяются в тестах."""

    def __init__(self, rpm: int, tpm: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if rpm <= 0 or tpm <= 0:
            raise gp.GeminiInputError("Лимиты RPM и TPM должны быть больше нуля.")
        self.rpm, self.tpm, self._clock, self._sleep = rpm, tpm, clock, sleep
        self._events: deque = deque()   # (время, токены)

    def acquire(self, tokens: int) -> None:
        tokens = max(0, min(int(tokens), self.tpm))  # запрос крупнее TPM иначе ждал бы вечно
        while True:
            now = self._clock()
            while self._events and now - self._events[0][0] >= WINDOW_SEC:
                self._events.popleft()
            wait = 0.0
            if len(self._events) >= self.rpm:
                wait = self._events[0][0] + WINDOW_SEC - now
            over = sum(k for _, k in self._events) + tokens - self.tpm
            if over > 0:
                freed = 0
                for t, k in self._events:
                    freed += k
                    if freed >= over:
                        wait = max(wait, t + WINDOW_SEC - now)
                        break
            if wait <= 0:
                self._events.append((now, tokens))
                return
            log.info("Лимитер: пауза %.1f с (RPM/TPM).", wait)
            self._sleep(wait)


# ---------------------------------------------------------------------------
# Транспорт с лимитером и цепочкой моделей
# ---------------------------------------------------------------------------

class ManagedTransport:
    """Обёртка над transport.generate_json: лимитер, счётчик RPD, смена модели, повтор при RPM.

    Аргумент model, который передаёт generate_prompts, игнорируется: модель выбирает цепочка.
    """

    def __init__(self, inner, models=DEFAULT_MODELS, *, rpm_limit: int = 15,
                 tpm_limit: int = 250_000, rpd_limit: int = 500,
                 output_reserve_tokens: int = 2000, max_rate_retries: int = 6,
                 rpd_used: Optional[dict] = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self._models = list(models)
        if not self._models:
            raise gp.GeminiInputError("Цепочка моделей Gemini пуста.")
        if rpd_limit <= 0:
            raise gp.GeminiInputError("Лимит RPD должен быть больше нуля.")
        self._inner = inner
        self._rpd_limit = rpd_limit
        self._reserve = max(0, output_reserve_tokens)
        self._max_retries = max_rate_retries
        self._sleep = sleep
        self._limiters = {m: RateLimiter(rpm_limit, tpm_limit, clock, sleep) for m in self._models}
        self._used = {m: int((rpd_used or {}).get(m, 0)) for m in self._models}
        self._exhausted: set = set()
        self.active_model: str = self._models[0]

    @property
    def requests_used(self) -> dict:
        return dict(self._used)

    def _pick_model(self) -> str:
        for m in self._models:
            if m in self._exhausted:
                continue
            if self._used[m] >= self._rpd_limit:
                self._exhausted.add(m)
                log.warning("Модель %s: достигнут суточный лимит запросов (%d).", m, self._rpd_limit)
                continue
            if m != self.active_model:
                log.warning("Переключение на запасную модель: %s.", m)
            self.active_model = m
            return m
        raise gp.GeminiRuntimeError(
            "Суточная квота исчерпана у всех моделей цепочки; прогресс сохранён, "
            "повторите запуск после сброса квоты.",
            status=429, code="RESOURCE_EXHAUSTED", daily_quota=True)

    def generate_json(self, model: str, system: str, user: str, schema: dict) -> str:
        tokens = (len(system) + len(user)) // CHARS_PER_TOKEN + 1 + self._reserve
        rate_retries = 0
        while True:
            m = self._pick_model()
            self._limiters[m].acquire(tokens)
            self._used[m] += 1
            try:
                return self._inner.generate_json(m, system, user, schema)
            except gp.GeminiRuntimeError as e:
                if e.daily_quota:
                    self._exhausted.add(m)
                    log.warning("Модель %s: суточная квота исчерпана, модель отключена.", m)
                    rate_retries = 0
                    continue
                if e.status == 429:
                    rate_retries += 1
                    if rate_retries > self._max_retries:
                        log.error("Модель %s: лимит запросов не снимается после %d повторов.",
                                  m, self._max_retries)
                        raise
                    wait = e.retry_after if e.retry_after is not None else min(5.0 * 2 ** (rate_retries - 1), 60.0)
                    wait = min(max(float(wait), 1.0), MAX_WAIT_SEC)
                    log.warning("Модель %s: превышен лимит в минуту, пауза %.1f с (повтор %d из %d).",
                                m, wait, rate_retries, self._max_retries)
                    self._sleep(wait)
                    continue
                raise


# ---------------------------------------------------------------------------
# prompts.json
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimingParams:
    fps: int
    frame_rule: dict
    clip_min_sec: float
    clip_max_sec: float
    gpu_max_sec: float
    tail_pad_sec: float


def atomic_write_json(path, data) -> None:
    """Временный файл в той же папке -> flush+fsync -> os.replace. При сбое старый файл цел."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_state(path, segments: list, profile: str) -> dict:
    """Читает prompts.json (если есть) и сверяет srt_hash. Несовпадение: GeminiInputError."""
    path = Path(path)
    current = srt_hash(segments)
    if not path.exists():
        return {"srt_hash": current, "model": "", "profile": profile, "prompts": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get("prompts"), dict):
            raise ValueError
    except (OSError, ValueError):
        raise gp.GeminiInputError(
            f"Файл {path.name} повреждён или имеет неверный формат; удалите его или исправьте.") from None
    if state.get("srt_hash") != current:
        raise gp.GeminiInputError(
            f"Файл {path.name} создан для другого SRT (srt_hash не совпадает). "
            "Запрос к Gemini не выполнен, чтобы не тратить квоту: удалите prompts.json "
            "или используйте исходный SRT.")
    state["profile"] = profile
    return state


def _has_prompt(entry) -> bool:
    return (isinstance(entry, dict) and entry.get("status") == "ready"
            and bool(str(entry.get("prompt", "")).strip()))


def generate_and_store(segments: list, wanted_nums: list, cfg: gp.PromptConfig, inner_transport,
                       prompts_path, *, profile: str, timing: TimingParams,
                       models=DEFAULT_MODELS, rpm_limit: int = 15, tpm_limit: int = 250_000,
                       rpd_limit: int = 500, rpd_used: Optional[dict] = None,
                       output_reserve_tokens: int = 2000, max_rate_retries: int = 6,
                       clock: Callable[[], float] = time.monotonic,
                       sleep: Callable[[float], None] = time.sleep) -> dict:
    """Генерирует недостающие промпты, дописывая prompts.json после каждого батча. Возвращает state."""
    wanted = sorted(set(wanted_nums))
    if not wanted:
        raise gp.GeminiInputError("Список нужных номеров пуст: нечего генерировать.")
    by_num = {s.num: s for s in segments}
    absent = [n for n in wanted if n not in by_num]
    if absent:
        raise gp.GeminiInputError("Номера отсутствуют в SRT: " + ", ".join(map(str, absent[:20])) + ".")

    state = load_state(prompts_path, segments, profile)
    try:
        timings = {t.num: t for t in compute_timings(
            segments, timing.fps, timing.frame_rule, timing.clip_min_sec,
            timing.clip_max_sec, timing.gpu_max_sec, timing.tail_pad_sec)}
    except ValueError as e:
        raise gp.GeminiInputError(f"Ошибка параметров тайминга: {e}") from None

    prompts = state["prompts"]
    pending = [n for n in wanted if not _has_prompt(prompts.get(str(n)))]
    if not pending:
        log.info("Все %d запрошенных промптов уже готовы, Gemini не вызывается.", len(wanted))
        return state
    log.info("К генерации: %d из %d запрошенных номеров.", len(pending), len(wanted))

    managed = ManagedTransport(
        inner_transport, models, rpm_limit=rpm_limit, tpm_limit=tpm_limit, rpd_limit=rpd_limit,
        rpd_used=rpd_used, output_reserve_tokens=output_reserve_tokens,
        max_rate_retries=max_rate_retries, clock=clock, sleep=sleep)

    def on_batch(results: list) -> None:
        for r in results:
            t = timings[r.num]
            old = prompts.get(str(r.num)) or {}
            prompts[str(r.num)] = {
                "num": r.num, "start_ms": t.start_ms, "end_ms": t.end_ms,
                "clip_ms": t.clip_ms, "num_frames": t.num_frames,
                "prompt": r.prompt,
                "status": "ready" if r.prompt.strip() else "failed",
                "attempts": int(old.get("attempts", 0)) + 1,
                "error": r.review_reason, "needs_review": r.needs_review,
                "extra": r.extra,
            }
        state["model"] = managed.active_model
        state["prompts"] = {k: prompts[k] for k in sorted(prompts, key=int)}
        atomic_write_json(prompts_path, state)
        log.info("Сохранён батч: %d промптов, всего в файле: %d.", len(results), len(prompts))

    gp.generate_prompts(segments, pending, cfg, managed, managed.active_model, on_batch=on_batch)
    return state
