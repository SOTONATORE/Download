"""Мониторинг загрузки весов модели по строкам лога воркера (SPEC 8.2, этап 6б).

Источник строк: `Generate/worker/setup.sh` (строки «[setup] ...») и вывод aria2c / curl.
Каждая строка вывода aria2c и curl приходит с префиксом «[file=<имя файла>] » (setup.sh, функция
download_model: имя файла без подкаталога). Загрузки идут параллельно, поэтому прогресс, скорость и
зависание ведутся по каждому файлу отдельно (состояние _FileState, ключ — имя файла без подкаталога).
Все регулярные выражения собраны в этом модуле.

Правила (каждое возвращает причину снятия карты, сама карта снимается оркестратором):
  а. скорость на срезе speed_check_at_pct ниже speed_min_mbps (проверка один раз на файл);
  б. прогресс конкретного файла не меняется дольше stall_timeout_min, пока файл загружается;
  в. этап (от первой строки файла до строки «все файлы на месте») дольше stage_max_min;
  е. строка «ОШИБКА: не удалось загрузить файл» (лимит повторов исчерпан).

Строки без префикса [file=...] (прежний формат, другие источники) разбираются как раньше: прогресс
относится к единственному активному файлу, при нескольких активных файлах такие строки пропускаются.

Если формат прогресса aria2 не распознан, пишется одно предупреждение, проценты не выдумываются.
Модуль только разбирает строки и считает время; ничего не печатает сам, сообщения передаются
через callable `log`. Время внедряется (clock), поэтому тесты детерминированы. Адреса с query-частью
(подписанные ссылки) сокращаются до адреса без «?...» до разбора и до любого вывода.

Статус: строки setup.sh сверены с исходником. Формат строк aria2 сверен с реальным прогоном владельца
(aria2c 1.37, файл 522 МиБ): прогресс «[#gid done/total(pct%) CN:n DL:speed( ETA:t)?]», строки «FILE:»,
«Redirecting», «Download complete», таблица «Download Results» и сводка. Порог зависания (stall_timeout_min)
и наибольший файл LTX владелец проверяет по первому прогону (SPEC 13.1, п. 5).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

# --- строки setup.sh: префикс log() «[setup] », индекс «[i/N] » (MODEL_FILES_TOTAL) ---
_SETUP_PREFIX = r"\[setup\] \[(?P<idx>\d+)/(?P<total>\d+)\] "
START_RE = re.compile(_SETUP_PREFIX + r"Начинаю загрузку: (?P<name>\S+)\s*$")
SKIP_RE = re.compile(_SETUP_PREFIX + r"Файл уже существует, пропускаю: (?P<name>\S+)\s*$")
DONE_RE = re.compile(_SETUP_PREFIX + r"Файл успешно загружен: (?P<name>\S+) \((?P<size>\d+) байт\)\s*$")
# формы кода: «(код N)» (прежняя), «(код aria2 N)» и «(код curl N)» (так печатает setup.sh сейчас)
ERROR_RE = re.compile(_SETUP_PREFIX + r"ОШИБКА: не удалось загрузить файл (?P<name>\S+) "
                      r"\(код (?:(?:aria2|curl) )?(?P<rc>\d+)\)\.\s*$")
STAGE_DONE_RE = re.compile(r"\[setup\] Все файлы модели LTX-2\.5 на месте\.\s*$")

# --- префикс строк aria2c и curl из setup.sh: «[file=<имя файла>] » ---
FILE_PREFIX_RE = re.compile(r"\[file=(?P<file>[^\]\s]+)\] ")

# --- строки aria2 (формат проверен на прогоне владельца, aria2c 1.37) ---
# «[#6df3a8 488MiB/522MiB(93%) CN:3 DL:9.7MiB ETA:3s]»; ETA может отсутствовать
_NUM = r"\d+(?:\.\d+)?[KMGT]?i?B"
ARIA_PROGRESS_RE = re.compile(
    r"\[#(?P<gid>[0-9a-f]+) (?P<done>" + _NUM + r")/(?P<total>" + _NUM + r")"
    r"\((?P<pct>\d{1,3})%\) CN:\d+ DL:(?P<speed>" + _NUM + r")(?: ETA:[0-9hms]+)?\]"
)
# «10/24 14:42:07 [NOTICE] Download complete: <путь>» — файл завершён
ARIA_DONE_RE = re.compile(r"\[NOTICE\] Download complete: (?P<path>\S.*?)\s*$")
# строки, которые не относятся к прогрессу: перенаправления, сводка, таблица результатов, «FILE:»
ARIA_IGNORE_RE = re.compile(
    r"Redirecting to |Download Progress Summary|Download Results|^FILE: "
    r"|^gid\s*\|\s*stat\s*\||^[0-9a-f]+\|(?:OK|ERR|INPR|RM)\b|^=+"
)
ARIA_MARK = "[#"

# адрес с query-частью: «https://host/path?...» -> «https://host/path»
_QUERY_RE = re.compile(r"(https?://[^\s?]*)\?\S*")

_UNIT = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def size_to_bytes(text: str) -> float:
    """«1.2GiB», «450MiB», «512KiB», «512B» -> байты (двоичные единицы, как у aria2)."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)i?B", text)
    if not m:
        raise ValueError("неизвестный формат размера")
    return float(m.group(1)) * _UNIT[m.group(2)]


def bytes_per_sec_to_mbps(bytes_per_sec: float) -> float:
    """Байты в секунду -> Мбит/с (десятичные мегабиты, как в min_inet_mbps)."""
    return bytes_per_sec * 8.0 / 1_000_000.0


def _key(name: str) -> str:
    """Ключ состояния файла: имя без подкаталога («vae/x.safetensors» -> «x.safetensors»)."""
    return name.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Разбор одной строки
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    kind: str                       # start | skip | done | error | stage_done | progress | aria_done | ignore | aria_unknown | other
    name: str = ""                  # имя файла «подкаталог/имя» (для файловых строк)
    pct: int = -1                   # процент (progress)
    done_bytes: float = 0.0
    total_bytes: float = 0.0
    speed_bps: float = 0.0          # текущая скорость, байт/с (progress)
    file: str = ""                  # имя из префикса «[file=...]» (строки aria2 и curl), иначе пусто


def _drop_query(text: str) -> str:
    """Убирает query-часть адресов (подписанные ссылки) до любого хранения и вывода."""
    return _QUERY_RE.sub(r"\1", text)


def _classify(text: str) -> Event:
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
    m = ARIA_DONE_RE.search(text)
    if m:
        return Event("aria_done", name=_key(m.group("path")))
    if ARIA_IGNORE_RE.search(text):
        return Event("ignore")
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


def parse_line(line: str) -> Event:
    """Классифицирует строку лога. Неизвестные строки получают kind «other» (их не трогаем)."""
    text = _drop_query(line.strip())
    m = FILE_PREFIX_RE.match(text)
    if m:
        return replace(_classify(text[m.end():]), file=m.group("file"))
    return _classify(text)


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
    name: str                    # имя «подкаталог/имя» для логов
    started: float
    last_progress: float = 0.0   # время последнего изменения процента (или старта файла)
    last_pct: int = -1
    last_speed_bps: float = 0.0  # последняя скорость, байт/с
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
    last_change: float = 0.0          # общая отметка последнего события файла (для совместимости; зависание считается по файлам)
    active: dict = field(default_factory=dict)   # ключ: имя файла без подкаталога -> _FileState
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
            self.active[_key(ev.name)] = _FileState(name=ev.name, started=now, last_progress=now,
                                                    next_step=self.progress_step_pct)
            self.last_change = now
            self.log(f"Веса: начало загрузки {ev.name}.")
            return None
        if ev.kind == "skip":
            self.log(f"Веса: {ev.name} уже на диске, пропущен.")
            return None
        if ev.kind == "done":
            self.active.pop(_key(ev.name), None)
            self.last_change = now
            self.log(f"Веса: {ev.name} загружен.")
            return None
        if ev.kind == "aria_done":
            # aria2 сообщил о завершении файла: снимаем его с контроля зависания.
            # Запись в лог делает строка «Файл успешно загружен» из setup.sh, здесь без повтора.
            self.active.pop(ev.name, None)
            self.last_change = now
            return None
        if ev.kind == "error":
            self.active.pop(_key(ev.name), None)
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

    def _target(self, ev: Event, now: float) -> Optional[_FileState]:
        """Файл, к которому относится строка прогресса."""
        if ev.file:
            # строка с префиксом: только для файла, который уже начат и не завершён
            return self.active.get(_key(ev.file))
        # старый формат без префикса: только если активен ровно один файл
        if len(self.active) != 1:
            return None
        return next(iter(self.active.values()))

    def _progress(self, ev: Event, now: float) -> Optional[str]:
        state = self._target(ev, now)
        if state is None:
            return None
        state.recognised += 1
        state.last_speed_bps = ev.speed_bps
        if ev.pct == state.last_pct:
            return None                  # тот же процент: ни лога, ни сдвига таймера этого файла
        state.last_pct = ev.pct
        state.last_progress = now
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
        # правило (б) считается по каждому файлу: зависший файл снимает карту, даже если другой идёт
        for state in self.active.values():
            idle = now - state.last_progress
            if idle >= self.stall_sec:
                self.log(f"Веса: {state.name}: прогресс не менялся {idle:.0f} с.", True)
                return REASON_STALL
        return None
