"""Мониторинг загрузки весов модели по строкам лога воркера (SPEC 8.2, этап 6б).

Источник строк: `Generate/worker/setup.sh` (строки «[setup] ...») и вывод aria2c
(`--summary-interval=10`, строки «[#gid ...]»). Все регулярные выражения собраны в этом модуле.

Правила (каждое возвращает причину снятия карты, сама карта снимается оркестратором):
  а. скорость на срезе speed_check_at_pct ниже speed_min_mbps (проверка один раз на файл);
  б. прогресс не меняется дольше stall_timeout_min, пока файл загружается;
  в. этап (от первой строки файла до строки «все файлы на месте») дольше stage_max_min;
  е. строка «ОШИБКА: не удалось загрузить файл» (лимит повторов исчерпан).

Если формат прогресса aria2 не распознан, пишется одно предупреждение, проценты не выдумываются.
Модуль только разбирает строки и считает время; ничего не печатает сам, сообщения передаются
через callable `log`. Время внедряется (clock), поэтому тесты детерминированы.

Статус: строки setup.sh сверены с исходником. Формат строк aria2 **UNVERIFIED**: aria2 в репозитории
и в среде разработки не запускался, образца вывода нет. Регулярное выражение взято из документации
aria2 для строки сводки и должно быть подтверждено на первом прогоне (SPEC 13.1, п. 5).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

# --- строки setup.sh: префикс log() «[setup] », индекс «[i/N] » (MODEL_FILES_TOTAL) ---
_SETUP_PREFIX = r"\[setup\] \[(?P<idx>\d+)/(?P<total>\d+)\] "
START_RE = re.compile(_SETUP_PREFIX + r"Начинаю загрузку: (?P<name>\S+)\s*$")
SKIP_RE = re.compile(_SETUP_PREFIX + r"Файл уже существует, пропускаю: (?P<name>\S+)\s*$")
DONE_RE = re.compile(_SETUP_PREFIX + r"Файл успешно загружен: (?P<name>\S+) \((?P<size>\d+) байт\)\s*$")
ERROR_RE = re.compile(_SETUP_PREFIX + r"ОШИБКА: не удалось загрузить файл (?P<name>\S+) \(код (?P<rc>\d+)\)\.\s*$")
STAGE_DONE_RE = re.compile(r"\[setup\] Все файлы модели LTX-2\.5 на месте\.\s*$")

# --- строка сводки aria2 (UNVERIFIED на живом выводе, см. докстринг) ---
# пример по документации: [#2089b0 1.2GiB/21GiB(5%) CN:16 DL:450MiB ETA:40s]
_NUM = r"\d+(?:\.\d+)?[KMGT]?i?B"
ARIA_PROGRESS_RE = re.compile(
    r"\[#(?P<gid>[0-9a-f]+) (?P<done>" + _NUM + r")/(?P<total>" + _NUM + r")"
    r"\((?P<pct>\d{1,3})%\) CN:\d+ DL:(?P<speed>" + _NUM + r")(?: ETA:[0-9hms]+)?\]"
)
ARIA_MARK = "[#"

_UNIT = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def size_to_bytes(text: str) -> float:
    """«1.2GiB», «450MiB», «512B» -> байты (двоичные единицы, как у aria2)."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)i?B", text)
    if not m:
        raise ValueError("неизвестный формат размера")
    return float(m.group(1)) * _UNIT[m.group(2)]


def bytes_per_sec_to_mbps(bytes_per_sec: float) -> float:
    """Байты в секунду -> Мбит/с (десятичные мегабиты, как в min_inet_mbps)."""
    return bytes_per_sec * 8.0 / 1_000_000.0


# ---------------------------------------------------------------------------
# Разбор одной строки
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    kind: str                       # start | skip | done | error | stage_done | progress | aria_unknown | other
    name: str = ""                  # имя файла «подкаталог/имя» (для файловых строк)
    pct: int = -1                   # процент (progress)
    done_bytes: float = 0.0
    total_bytes: float = 0.0
    speed_bps: float = 0.0          # текущая скорость, байт/с (progress)


def parse_line(line: str) -> Event:
    """Классифицирует строку лога. Неизвестные строки получают kind «other» (их не трогаем)."""
    text = line.strip()
    m = START_RE.match(text)
    if m:
        return Event("start", name=m.group("name"))
    m = SKIP_RE.match(text)
    if m:
        return Event("skip", name=m.group("name"))
    m = DONE_RE.match(text)
    if m:
        return Event("done", name=m.group("name"))
    m = ERROR_RE.match(text)
    if m:
        return Event("error", name=m.group("name"))
    if STAGE_DONE_RE.match(text):
        return Event("stage_done")
    m = ARIA_PROGRESS_RE.search(text)
    if m:
        try:
            return Event("progress", pct=int(m.group("pct")),
                         done_bytes=size_to_bytes(m.group("done")),
                         total_bytes=size_to_bytes(m.group("total")),
                         speed_bps=size_to_bytes(m.group("speed")))
        except ValueError:
            return Event("aria_unknown")
    if ARIA_MARK in text:
        return Event("aria_unknown")  # строка aria2 есть, а формат не разобран
    return Event("other")


# ---------------------------------------------------------------------------
# Монитор одной карты
# ---------------------------------------------------------------------------

REASON_SPEED = "скорость загрузки весов ниже порога"
REASON_STALL = "загрузка весов зависла"
REASON_STAGE = "этап загрузки весов превысил лимит"
REASON_FILE_ERROR = "ошибка загрузки весов: "   # + имя файла
UNRECOGNISED_WARNING = "прогресс загрузки весов не распознан"


@dataclass
class _FileState:
    name: str
    started: float
    last_pct: int = -1
    next_step: int = 0
    speed_checked: bool = False
    recognised: int = 0          # распознанных строк прогресса по файлу


@dataclass
class WeightsMonitor:
    clock: Callable[[], float]
    log: Callable[..., None]          # log(текст, warn=False): warn=True для предупреждений и ошибок
    speed_check_pct: int
    speed_min_mbps: float
    progress_step_pct: int
    stall_sec: float
    stage_max_sec: float
    seen_start: bool = False          # первая строка файла уже пришла
    stage_done: bool = False          # «все файлы на месте» пришла
    stage_started: float = 0.0
    last_change: float = 0.0          # последнее изменение процента (или старт/финиш файла)
    active: dict = field(default_factory=dict)
    _unknown_warned: bool = False

    def feed(self, lines) -> Optional[str]:
        """Обрабатывает строки по порядку. Возвращает причину снятия (первую) или None."""
        for line in lines:
            reason = self._feed_one(line)
            if reason:
                return reason
        return None

    def _feed_one(self, line: str) -> Optional[str]:
        ev = parse_line(line)
        now = self.clock()
        if ev.kind in ("start", "skip", "done", "error") and not self.seen_start:
            self.seen_start = True
            self.stage_started = now
            self.last_change = now
        if ev.kind == "start":
            self.active[ev.name] = _FileState(name=ev.name, started=now, next_step=self.progress_step_pct)
            self.last_change = now
            self.log(f"Веса: начало загрузки {ev.name}.")
            return None
        if ev.kind == "skip":
            self.log(f"Веса: {ev.name} уже на диске, пропущен.")
            return None
        if ev.kind == "done":
            self.active.pop(ev.name, None)
            self.last_change = now
            self.log(f"Веса: {ev.name} загружен.")
            return None
        if ev.kind == "error":
            self.active.pop(ev.name, None)
            self.log(f"Веса: ошибка загрузки {ev.name}.", True)
            return REASON_FILE_ERROR + ev.name
        if ev.kind == "stage_done":
            self.stage_done = True
            self.log("Веса: этап загрузки весов завершён.")
            return None
        if ev.kind == "aria_unknown":
            if not self._unknown_warned:
                self._unknown_warned = True
                self.log(f"{UNRECOGNISED_WARNING[0].upper()}{UNRECOGNISED_WARNING[1:]}: действуют только лимит этапа "
                         f"(stage_max_min) и снятие по ошибке файла.", True)
            return None
        if ev.kind == "progress":
            return self._progress(ev, now)
        return None

    def _progress(self, ev: Event, now: float) -> Optional[str]:
        # строка прогресса относится к единственному активному файлу (setup.sh качает по одному)
        if len(self.active) != 1:
            return None
        state = next(iter(self.active.values()))
        state.recognised += 1
        if ev.pct == state.last_pct:
            return None                  # тот же процент: ни лога, ни сдвига таймера
        state.last_pct = ev.pct
        self.last_change = now
        if ev.pct >= state.next_step and ev.pct < 100:
            self._log_progress(state, ev, now)
            step = self.progress_step_pct
            state.next_step = (ev.pct // step + 1) * step
        if not state.speed_checked and ev.pct >= self.speed_check_pct:
            state.speed_checked = True   # правило (а): один раз на файл, в момент достижения среза
            mbps = bytes_per_sec_to_mbps(ev.speed_bps)
            if mbps < self.speed_min_mbps:
                self.log(f"Веса: {state.name}: на {ev.pct}% скорость {mbps:.0f} Мбит/с "
                         f"ниже порога {self.speed_min_mbps:g} Мбит/с.", True)
                return REASON_SPEED
        return None

    def _log_progress(self, state: _FileState, ev: Event, now: float) -> None:
        elapsed = max(now - state.started, 1e-6)
        avg_bps = ev.done_bytes / elapsed
        left = (ev.total_bytes - ev.done_bytes) / avg_bps if avg_bps > 0 else None
        left_text = f"~{left / 60:.1f} мин" if left is not None else "неизвестно"
        self.log(f"Веса: {state.name}: {ev.pct}%, скорость {bytes_per_sec_to_mbps(ev.speed_bps):.0f} Мбит/с, "
                 f"средняя {bytes_per_sec_to_mbps(avg_bps):.0f} Мбит/с, осталось {left_text}.")

    def check(self) -> Optional[str]:
        """Правила по времени (б) и (в). Вызывается на каждом шаге опроса карты."""
        if not self.seen_start or self.stage_done:
            return None
        now = self.clock()
        if now - self.stage_started >= self.stage_max_sec:
            return REASON_STAGE
        if self.active and now - self.last_change >= self.stall_sec:
            return REASON_STALL
        return None
