"""Этап 4б: оркестратор генерации промптов (лимитер, цепочка моделей, prompts.json).

Только стандартная библиотека + модуль gemini_prompts (httpx живёт там).
Ключ API здесь не читается и не упоминается: им владеет RealTransport (SPEC 0.1).
Тексты промптов в логи не пишутся. Все сообщения на русском (SPEC 0.5).
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import tempfile
import threading
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

DEFAULT_MODELS = ("gemini-3.5-flash-lite",)
WINDOW_SEC = 60.0
CHARS_PER_TOKEN = 3          # консервативная оценка входа
MAX_WAIT_SEC = 120.0         # потолок одной паузы после 429
TRANSIENT_STATUSES = frozenset({500, 503, 504})  # сеть: HTTP-статуса нет (status None)
TRANSIENT_ATTEMPTS = 60      # неудачных попыток подряд на одной модели; после последней прогон останавливается
TRANSIENT_PAUSE_SEC = 10.0   # пауза перед каждой повторной попыткой (если в ответе нет retry_after)
TRANSIENT_SWITCH_AFTER = 15  # временных сбоев подряд на модели, после которых переходим к следующей (config: gemini_transient_switch_after)


def is_transient(err: gp.GeminiRuntimeError) -> bool:
    """Временный сбой: любой HTTP 5xx (500–599, в том числе 500/502/503/504) или сетевой сбой
    (RealTransport отдаёт его без статуса). Суточная квота, 429 и коды 4xx сюда не входят.
    TRANSIENT_STATUSES оставлен для читаемости: он входит в диапазон 5xx."""
    if err.daily_quota or err.status == 429:
        return False
    # сеть: статуса нет; 5xx: сбой сервера, ключ не виноват, запрос повторяется
    return err.status is None or err.status in TRANSIENT_STATUSES or 500 <= err.status <= 599


def _pct(part: int, whole: int) -> str:
    """Доля в процентах с одним знаком после запятой, половина округляется вверх (целая арифметика)."""
    if whole <= 0:
        return "0.0"
    tenths = (2 * part * 1000 + whole) // (2 * whole)  # десятые доли процента, округление вверх от половины
    return f"{tenths // 10}.{tenths % 10}"


def progress_line(prompts: dict, video_nums: list, photo_nums: list, started: float) -> str:
    """Строка прогресса обеих волн пула: сколько клипов каждой волны уже имеют промпт со статусом «ready».

    prompts: общий словарь промптов прогона; читать его можно только под замком состояния.
    started: время начала прогона (time.monotonic()), от него считается «за T с».
    """
    def ready_count(nums: list) -> int:
        return sum(1 for n in set(nums) if (prompts.get(str(n)) or {}).get("status") == "ready")

    video_total, photo_total = len(set(video_nums)), len(set(photo_nums))
    video_ready, photo_ready = ready_count(video_nums), ready_count(photo_nums)
    all_ready, all_total = video_ready + photo_ready, video_total + photo_total
    elapsed = time.monotonic() - started
    return (f"=== Прогресс: видео готово {video_ready} из {video_total} ({_pct(video_ready, video_total)}%), "
            f"фото готово {photo_ready} из {photo_total} ({_pct(photo_ready, photo_total)}%), "
            f"всего {all_ready} из {all_total} ({_pct(all_ready, all_total)}%), за {elapsed:.1f} с ===")


def _describe(status: Optional[int], code: str) -> str:
    """Безопасное описание ошибки для логов и сообщений: статус и код из ответа, без тела."""
    if status is None:
        return "сбой сети"
    return f"HTTP {status}" + (f", код {code}" if code else "")


def _transient_wait(err: gp.GeminiRuntimeError) -> float:
    """Пауза перед следующей попыткой: TRANSIENT_PAUSE_SEC; если в ответе есть retry_after, берётся он,
    но не меньше 1 с и не больше TRANSIENT_PAUSE_SEC."""
    if err.retry_after is None:
        return TRANSIENT_PAUSE_SEC
    return min(max(float(err.retry_after), 1.0), TRANSIENT_PAUSE_SEC)


# ---------------------------------------------------------------------------
# Лимитер RPM/TPM (скользящее окно 60 с)
# ---------------------------------------------------------------------------

# ВНИМАНИЕ: ManagedTransport этот класс больше не использует, лимиты ведёт пул ключей (gemini_pool).
# Класс оставлен без изменений, пока на него опираются его собственные тесты.
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

def _accepts_reasoning(fn) -> bool:
    """Есть ли у метода generate_json транспорта именованный параметр reasoning_effort."""
    try:
        return "reasoning_effort" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class ManagedTransport:
    """Обёртка над transport.generate_json: смена модели только при суточной квоте,
    повтор при RPM, повтор временных ошибок (500/503/504, сеть) на той же модели без смены.

    Лимиты RPM, TPM и RPD здесь больше не ведутся: их держит пул ключей (gemini_pool) внутри транспорта,
    по каждому ключу. Параметры rpm_limit, tpm_limit, rpd_limit и rpd_used сохранены для совместимости
    вызовов, но в этот класс не передаются и не действуют.
    Суточная квота исключает модель из цепочки только тогда, когда транспорт сообщает, что ключей для модели
    не осталось (ошибка без признака key_exhausted_only). Если исчерпан один ключ, а другие ещё есть,
    модель повторяется: пул выдаст следующий ключ.
    Временные сбои (503, сеть) подряд на модели: после switch_after сбоев переходим к следующей модели цепочки.
    Переход только вперёд: модель, с которой ушли, в этом прогоне не возвращается.
    Потолок TRANSIENT_ATTEMPTS общий на один запрос, на смену модели он не сбрасывается.

    Аргумент model, который передаёт generate_prompts, игнорируется: модель выбирает цепочка.
    """

    def __init__(self, inner, models=DEFAULT_MODELS, *, rpm_limit: int = 15,
                 tpm_limit: int = 250_000, rpd_limit: int = 500,
                 output_reserve_tokens: int = 2000, max_rate_retries: int = 6,
                 rpd_used: Optional[dict] = None,
                 transient_switch_after: int = TRANSIENT_SWITCH_AFTER,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self._models = list(models)
        if not self._models:
            raise gp.GeminiInputError("Цепочка моделей Gemini пуста.")
        if rpd_limit <= 0:
            raise gp.GeminiInputError("Лимит RPD должен быть больше нуля.")
        if transient_switch_after < 1:
            raise gp.GeminiInputError("Порог переключения после временных ошибок должен быть не меньше 1.")
        if rpd_used:
            log.warning("Счётчики RPD из прошлых запусков не учитываются: их ведёт пул ключей.")
        self._inner = inner
        # reasoning_effort передаём внутреннему транспорту только если он его принимает
        self._reasoning_ok = _accepts_reasoning(getattr(inner, "generate_json", None))
        self._reasoning_warned = False
        self._reserve = max(0, output_reserve_tokens)
        self._max_retries = max_rate_retries
        self._switch_after = transient_switch_after
        self._sleep = sleep
        self._clock = clock
        # замок защищает состояние цепочки моделей; на время сетевого вызова он не берётся
        self._lock = threading.Lock()
        self._exhausted: set = set()               # модели, отключённые на этот прогон (суточная квота)
        self._pos = 0                              # позиция текущей модели в цепочке; назад не возвращаемся
        self._streak: dict = {}                    # модель -> временных сбоев подряд (сброс: успех или смена модели)
        self._last_transient: tuple = (None, "")   # (status, code) последней временной ошибки
        self.active_model: str = self._models[0]

    def _usable(self, m: str) -> bool:
        return m not in self._exhausted

    def _transient_exhausted(self, model: str) -> gp.GeminiRuntimeError:
        """Ошибка после TRANSIENT_ATTEMPTS неудачных попыток подряд на модели; тело ответа в сообщение не входит."""
        with self._lock:
            status, code = self._last_transient
        return gp.GeminiRuntimeError(
            f"Модель {model} не ответила за {TRANSIENT_ATTEMPTS} попыток подряд (около 10 минут), "
            f"прогон остановлен; последняя ошибка: {_describe(status, code)}. "
            "Прогресс сохранён; повторите запуск позже",
            status=status, code=code)

    def _pick_model(self) -> str:
        with self._lock:   # выбор и смена active_model идут под замком: потоки не портят друг другу состояние
            for i in range(self._pos, len(self._models)):   # модели левее текущей уже не берём
                m = self._models[i]
                if not self._usable(m):
                    continue
                self._pos = i
                if m != self.active_model:
                    log.warning("Переключение на запасную модель: %s.", m)
                self.active_model = m
                return m
        raise gp.GeminiRuntimeError(
            "Суточная квота исчерпана у всех моделей цепочки; прогресс сохранён, "
            "повторите запуск после сброса квоты.",
            status=429, code="RESOURCE_EXHAUSTED", daily_quota=True)

    def _switch_after_transient(self, model: str) -> Optional[str]:
        """Переход после серии временных сбоев: следующая доступная модель цепочки или None, если её нет."""
        with self._lock:
            if self._models[self._pos] != model:   # другой поток уже перешёл дальше
                return None
            for i in range(self._pos + 1, len(self._models)):
                if self._usable(self._models[i]):
                    self._pos = i
                    self._streak[model] = 0
                    return self._models[i]
        return None

    def generate_json(self, model: str, system: str, user: str, schema: dict,
                      reasoning_effort: Optional[str] = None) -> str:
        rate_retries = 0
        tries = 0                  # неудачные попытки подряд в этом запросе: потолок TRANSIENT_ATTEMPTS на весь запрос
        # None: поведение как раньше; иначе параметр уходит только к транспорту, который его понимает
        extra = {}
        if reasoning_effort is not None:
            if self._reasoning_ok:
                extra = {"reasoning_effort": reasoning_effort}
            elif not self._reasoning_warned:
                self._reasoning_warned = True
                log.warning("Транспорт не принимает reasoning_effort: запрос идёт с настройками по умолчанию.")
        while True:
            m = self._pick_model()
            t0 = self._clock()
            try:
                text = self._inner.generate_json(m, system, user, schema, **extra)
                with self._lock:
                    self._streak[m] = 0            # успех сбрасывает серию временных сбоев модели
                log.info("Модель %s: ответ получен за %.1f с (сбоев подряд до этого: %d).",
                         m, self._clock() - t0, tries)
                return text
            except gp.GeminiRuntimeError as e:
                if e.daily_quota:
                    if getattr(e, "key_exhausted_only", False):
                        # квота исчерпана у одного ключа для этой модели: модель не меняем, пул выдаст другой ключ
                        log.warning("Модель %s: суточная квота исчерпана у одного ключа, пробуем другой ключ.", m)
                        continue
                    with self._lock:
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
                if not is_transient(e):
                    raise
                tries += 1
                with self._lock:
                    self._last_transient = (e.status, e.code)
                    self._streak[m] = self._streak.get(m, 0) + 1
                    streak = self._streak[m]
                if streak >= self._switch_after:
                    # серия временных сбоев на модели: переходим к следующей; если её нет, повторы идут до потолка
                    nxt = self._switch_after_transient(m)
                    if nxt is not None:
                        log.warning("Модель %s: %d подряд временных ошибок, переключаюсь на %s.",
                                    m, self._switch_after, nxt)
                if tries < TRANSIENT_ATTEMPTS:
                    wait = _transient_wait(e)
                    log.warning("Модель %s: временная ошибка (%s), пауза %.1f с (попытка %d из %d).",
                                m, _describe(e.status, e.code), wait, tries + 1, TRANSIENT_ATTEMPTS)
                    self._sleep(wait)
                    continue
                # последняя попытка неудачна: модель не меняется, прогон останавливается
                log.error("Модель %s: %d попыток подряд без ответа (%s), прогон остановлен.",
                          m, TRANSIENT_ATTEMPTS, _describe(e.status, e.code))
                raise self._transient_exhausted(m) from None


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
    if not isinstance(entry, dict):
        return False
    num_frames = entry.get("num_frames")
    return (entry.get("status") == "ready"
            and bool(str(entry.get("prompt", "")).strip())
            and isinstance(num_frames, int) and not isinstance(num_frames, bool)
            and num_frames > 0)


def missing_segments(prompts: dict, wanted: list) -> list:
    """Номера из wanted, у которых в prompts нет готового промпта (по возрастанию). Чистая функция."""
    return [n for n in sorted(set(wanted)) if not _has_prompt(prompts.get(str(n)))]


def _rework_seeds(prompts: dict) -> list:
    """Клипы видео из сохранённого состояния: нарушения ещё не исправлены, круги не кончились.

    Видео узнаётся по start_ms (его пишет только этот путь); фото сюда не попадает. Вызывать под замком
    состояния: словарь пишут обе волны.
    """
    seeds = []
    for key, e in prompts.items():
        pending_fix = e.get("rework_pending") or []
        rnd = int(e.get("rework_round", 0) or 0)
        if not pending_fix or rnd >= gp.REPAIR_ROUNDS or "start_ms" not in e:
            continue
        if not str(e.get("prompt", "")).strip():
            continue
        seeds.append(gp.PromptResult(
            num=int(key), prompt=e["prompt"], needs_review=bool(e.get("needs_review", False)),
            review_reason=str(e.get("error", "")), extra=dict(e.get("extra") or {}),
            rework_pending=list(pending_fix), rework_round=rnd))
    return seeds


def generate_and_store(segments: list, wanted_nums: list, cfg: gp.PromptConfig, inner_transport,
                       prompts_path, *, profile: str, timing: TimingParams,
                       models=DEFAULT_MODELS, rpm_limit: int = 15, tpm_limit: int = 250_000,
                       rpd_limit: int = 500, rpd_used: Optional[dict] = None,
                       output_reserve_tokens: int = 2000, max_rate_retries: int = 6,
                       transient_switch_after: int = TRANSIENT_SWITCH_AFTER,
                       clock: Callable[[], float] = time.monotonic,
                       sleep: Callable[[float], None] = time.sleep,
                       managed=None, state=None, lock=None, pool=None, wave=None,
                       progress: Optional[Callable[[dict], str]] = None) -> dict:
    """Генерирует недостающие промпты, дописывая prompts.json после каждого батча. Возвращает state.

    managed: общий ManagedTransport прогона (цепочка моделей одна на оба вида); None: создаётся здесь.
    state, lock: общий словарь состояния и замок; без них поведение прежнее (файл читается здесь).
    pool, wave: общий пул батчей обеих волн и имя этой волны; без пула батчи идут своей очередью, как раньше.
    progress: строка прогресса обеих волн после каждого сохранённого батча (только в пуле); None — не пишется.
    """
    wanted = sorted(set(wanted_nums))
    if not wanted:
        raise gp.GeminiInputError("Список нужных номеров пуст: нечего генерировать.")
    by_num = {s.num: s for s in segments}
    absent = [n for n in wanted if n not in by_num]
    if absent:
        raise gp.GeminiInputError("Номера отсутствуют в SRT: " + ", ".join(map(str, absent)) + ".")

    # общее состояние прогона передано извне: читаем файл только когда его не передали
    shared_state = state
    state = shared_state if shared_state is not None else load_state(prompts_path, segments, profile)
    # замок общего состояния: слияние и запись в файл идут под ним; без общего состояния замка не нужно
    guard = lock if lock is not None else threading.Lock()
    try:
        timings = {t.num: t for t in compute_timings(
            segments, timing.fps, timing.frame_rule, timing.clip_min_sec,
            timing.clip_max_sec, timing.gpu_max_sec, timing.tail_pad_sec)}
    except ValueError as e:
        raise gp.GeminiInputError(f"Ошибка параметров тайминга: {e}") from None

    prompts = state["prompts"]
    pending = missing_segments(prompts, wanted)  # B2: только номера без готового промпта
    # очередь исправлений из сохранённого состояния (под замком: словарь пишут обе волны)
    with guard:
        seeds = _rework_seeds(prompts)
    if not pending and not seeds:
        log.info("Все %d запрошенных промптов уже готовы, Gemini не вызывается.", len(wanted))
        return state
    log.info("К генерации: %d из %d запрошенных номеров.", len(pending), len(wanted))
    if seeds:
        log.info("Очередь исправлений из прошлого запуска: %d клипов.", len(seeds))

    # цепочка моделей: общая (передана managed) или своя на этот вызов, как раньше
    if managed is None:
        managed = ManagedTransport(
            inner_transport, models, rpm_limit=rpm_limit, tpm_limit=tpm_limit, rpd_limit=rpd_limit,
            rpd_used=rpd_used, output_reserve_tokens=output_reserve_tokens,
            max_rate_retries=max_rate_retries, transient_switch_after=transient_switch_after,
            clock=clock, sleep=sleep)

    # имя волны в строках батча и сохранения: только в пуле; без пула строки прежние, без префикса
    wave_tag = f"[{wave}] " if pool is not None and wave is not None else ""

    def on_batch(results: list) -> None:
        # под замком: второй вид может писать в тот же файл параллельно (на данный момент волны идут по очереди)
        with guard:
            for r in results:
                t = timings[r.num]
                old = prompts.get(str(r.num)) or {}
                entry = {
                    "num": r.num, "start_ms": t.start_ms, "end_ms": t.end_ms,
                    "clip_ms": t.clip_ms, "num_frames": t.num_frames,
                    "prompt": r.prompt,
                    "status": "ready" if r.prompt.strip() else "failed",
                    "attempts": int(old.get("attempts", 0)) + 1,
                    "error": r.review_reason, "needs_review": r.needs_review,
                    "extra": r.extra,
                }
                rework_round = getattr(r, "rework_round", None)
                if rework_round is not None:
                    # поля очереди исправлений есть только у клипов, которые её проходили; остальные записи не меняются
                    entry["rework_pending"] = list(getattr(r, "rework_pending", None) or [])
                    entry["rework_round"] = int(rework_round)
                prompts[str(r.num)] = entry
            state["model"] = managed.active_model
            state["prompts"] = {k: prompts[k] for k in sorted(prompts, key=int)}
            # журнал сохранения: начало и итог — ключевые строки для консоли; сбой — предупреждение с причиной
            log.info("%sСохранение промптов начато: файл %s, промптов к сохранению: %d.",
                     wave_tag, prompts_path, len(results), extra=gp.KEY_EVENT_EXTRA)
            try:
                atomic_write_json(prompts_path, state)
            except Exception as exc:  # причину пишем и пробрасываем дальше: поведение прогона не меняется
                log.warning("%sНе удалось сохранить промпты в файл %s: %s: %s.",
                            wave_tag, prompts_path, type(exc).__name__, exc)
                raise
            log.info("%sПромпты сохранены: файл %s, всего в файле: %d.",
                     wave_tag, prompts_path, len(prompts), extra=gp.KEY_EVENT_EXTRA)
            if progress is not None:
                # прогресс обеих волн: читается из того же словаря под тем же замком, уже после записи файла
                log.info(progress(prompts), extra=gp.KEY_EVENT_EXTRA)

    # провайдер берётся у транспорта (openrouter считает свои ключи для числа потоков);
    # для gemini аргумент не передаётся, чтобы вызов остался прежним (совместимость с gemini_prompts без provider)
    provider = getattr(inner_transport, "provider", "gemini")
    extra = {} if provider == "gemini" else {"provider": provider}
    # пул передаётся только при общем пуле: вызов для одной волны остаётся прежним
    pool_kw = {} if pool is None else {"pool": pool, "wave": wave}
    # очередь прошлого запуска передаётся только когда она не пуста: вызов без неё остаётся прежним
    rework_kw = {} if not seeds else {"rework": seeds}
    gp.generate_prompts(segments, pending, cfg, managed, managed.active_model, on_batch=on_batch,
                        **extra, **pool_kw, **rework_kw)
    return state
