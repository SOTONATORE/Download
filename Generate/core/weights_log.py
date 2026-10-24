"""Мониторинг загрузки весов модели по строкам лога воркера (SPEC 8.2, этап 6б).

Источник строк: `Generate/worker/setup.sh` (строки «[setup] ...») и вывод aria2c / curl.
Каждая строка вывода aria2c и curl приходит с префиксом «[file=<имя файла>] » (setup.sh, функция
download_model: имя файла без подкаталога). Загрузки идут параллельно. Скорость и прогресс считаются
по всему этапу (сумма байт всех файлов), зависание — по каждому файлу отдельно.
Все регулярные выражения собраны в этом модуле.

Правила (каждое возвращает причину снятия карты, сама карта снимается оркестратором):
  а. скорость всего этапа за окно измерения ниже порога speed_min_mbps (агрегат, не по файлу; порог — половина
     min_inet_mbps входа запуска, см. speed_threshold_mbps);
  б. прогресс конкретного файла не меняется дольше stall_timeout_min, пока файл загружается;
  в. этап (от первой строки файла до строки «все файлы на месте») дольше stage_max_min;
  е. строка «ОШИБКА: не удалось загрузить файл» (лимит повторов исчерпан).

Правила а и б решают только по окну измерения: окно считается набранным, когда с первого замера
этапа прошло не меньше MIN_WINDOW_SEC секунд и замеров не меньше MIN_SAMPLES. До этого решения нет.
Замер — выгрузка (один ответ GET /logs, одна отметка времени), а не строка: по каждому файлу берётся
последняя строка прогресса выгрузки; начало окна и замеры строятся по выгрузкам.
Средняя скорость для журнала печатается только когда с начала этапа прошло не меньше MIN_WINDOW_SEC,
раньше — «оценка недоступна». Оставшееся время — только при полном окне. Console получает только
проценты шагами 10% и только при известном общем размере (см. _update_percent).

Строки с «[#» и «%», которые не разобрал регэксп, не участвуют ни в скорости, ни в зависании:
считаются и предупреждаются один раз на этап, а полный текст (без адресов и query-частей) пишется в журнал
по одному разу на каждую форму строки, чтобы следующий прогон показал реальный формат. Строки без «[#»
не участвуют вовсе.

Строки без префикса [file=...] (прежний формат, другие источники) разбираются как раньше: прогресс
относится к единственному активному файлу, при нескольких активных файлах такие строки пропускаются.

Модуль только разбирает строки и считает время; ничего не печатает сам. Сообщения для журнала идут
через callable `log(текст, warn=False)` (подробные строки, только в файл), а ключевые события и шаги
процентов — через callable `console(вид, проценты)`. Время внедряется (clock), поэтому тесты
детерминированы. Адреса с query-частью (подписанные ссылки) сокращаются до адреса без «?...» до разбора
и до любого вывода.

Статус: строки setup.sh сверены с исходником. Формат строк aria2 сверен с реальным прогоном владельца
(aria2c 1.37, файл 522 МиБ): прогресс «[#gid done/total(pct%) CN:n DL:speed( ETA:t)?]», строки «FILE:»,
«Redirecting», «Download complete», таблица «Download Results» и сводка. Порог скорости (speed_min_mbps)
берётся как половина min_inet_mbps входа запуска (speed_threshold_mbps) и НЕ проверен на реальной линии по
агрегату; владелец проверяет его замером агрегатной скорости на арендованной карте (SPEC 13.1). Порог зависания (stall_timeout_min)
и наибольший файл LTX владелец проверяет по первому прогону (SPEC 13.1, п. 5).
"""
from __future__ import annotations

import math
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
# адрес целиком (полный текст нераспознанной строки для журнала не содержит адресов)
_URL_RE = re.compile(r"https?://\S+")
# события файла: строки прогресса, накопленные в выгрузке, применяются перед ними (см. feed)
_FLUSH_KINDS = frozenset({"start", "skip", "done", "error", "stage_done", "aria_done"})

_UNIT = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
MIB = 1024 ** 2
# 1 Мбит/с = 125000 байт/с (десятичные мегабиты, как в min_inet_mbps); единица скорости в журнале
BYTES_PER_MBIT = 125_000.0
# окно измерения скорости и зависания: набрано, когда прошло не меньше MIN_WINDOW_SEC с первого замера
# этапа и замеров не меньше MIN_SAMPLES (правило «решать только по достаточному окну»)
MIN_WINDOW_SEC = 90.0
MIN_SAMPLES = 3


def size_to_bytes(text: str) -> float:
    """«1.2GiB», «450MiB», «512KiB», «512B» -> байты (двоичные единицы, как у aria2)."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)i?B", text)
    if not m:
        raise ValueError("неизвестный формат размера")
    return float(m.group(1)) * _UNIT[m.group(2)]


def bytes_per_sec_to_mbps(bytes_per_sec: float) -> float:
    """Байты в секунду -> Мбит/с (десятичные мегабиты: 1 Мбит/с = 125000 байт/с)."""
    return bytes_per_sec / BYTES_PER_MBIT


def speed_threshold_mbps(min_inet_mbps: float) -> int:
    """Порог правила скорости, Мбит/с: половина min_inet_mbps входа запуска, округлённая вниз до целого.

    Вход — workflow MIN_INET или --min-inet-mbps (cfg.min_inet_mbps), а не значение из конфига.
    Меньше 1 Мбит/с — ошибка входа: проверяется в config_loader (GeminiInputError).
    """
    return math.floor(float(min_inet_mbps) / 2.0)


def _form_key(text: str) -> str:
    """Форма строки для однократной записи в журнал: адрес, идентификатор «#gid» и числа не учитываются."""
    shape = re.sub(r"#[0-9a-f]+", "#gid", text.replace("<адрес>", "").strip())
    return re.sub(r"\d+(?:\.\d+)?", "N", shape)


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
    index: int = 0                  # номер файла «[i/N]» из строк setup.sh (0 — строки без номера)
    count: int = 0                  # N из «[i/N]»
    text: str = ""                  # только для aria_unknown: строка без query-частей и адресов (для журнала формы)


def _drop_query(text: str) -> str:
    """Убирает query-часть адресов (подписанные ссылки) до любого хранения и вывода."""
    return _QUERY_RE.sub(r"\1", text)


def _num(m) -> tuple:
    """Номер файла и N из строк setup.sh: («[i/N]»)."""
    return int(m.group("idx")), int(m.group("total"))


def _classify(text: str) -> Event:
    m = START_RE.match(text)
    if m:
        i, n = _num(m)
        return Event("start", name=m.group("name"), index=i, count=n)
    m = SKIP_RE.match(text)
    if m:
        i, n = _num(m)
        return Event("skip", name=m.group("name"), index=i, count=n)
    m = DONE_RE.match(text)
    if m:
        i, n = _num(m)
        size = float(m.group("size"))
        return Event("done", name=m.group("name"), index=i, count=n, done_bytes=size, total_bytes=size)
    m = ERROR_RE.match(text)
    if m:
        i, n = _num(m)
        return Event("error", name=m.group("name"), index=i, count=n)
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
    # строка с «[#» и «%», которую регэксп не разобрал: считается и предупреждается, не используется
    if ARIA_MARK in text and "%" in text:
        return Event("aria_unknown")
    return Event("other")  # без «[#» (или без «%»): к скорости и зависанию отношения не имеет


def parse_line(line: str) -> Event:
    """Классифицирует строку лога. Неизвестные строки получают kind «other» (их не трогаем)."""
    text = _drop_query(line.strip())
    m = FILE_PREFIX_RE.match(text)
    ev = replace(_classify(text[m.end():]), file=m.group("file")) if m else _classify(text)
    if ev.kind == "aria_unknown":
        # полный текст для журнала формы: query-части уже убраны, адреса целиком заменяются
        ev = replace(ev, text=_URL_RE.sub("<адрес>", text))
    return ev


# ---------------------------------------------------------------------------
# Монитор одной карты
# ---------------------------------------------------------------------------

REASON_SPEED = "скорость загрузки весов ниже порога"
REASON_STALL = "загрузка весов зависла"
REASON_STAGE = "этап загрузки весов превысил лимит"
REASON_FILE_ERROR = "ошибка загрузки весов: "   # + имя файла
UNRECOGNISED_WARNING = "прогресс загрузки весов не распознан"
WINDOW_WAIT_NOTE = "Веса: окно измерения ещё не набрано."
ESTIMATE_UNAVAILABLE = "оценка недоступна"


@dataclass
class _FileState:
    name: str                    # имя «подкаталог/имя» для логов
    started: float
    last_progress: float = 0.0   # время последнего изменения процента (или старта файла)
    last_pct: int = -1
    last_speed_bps: float = 0.0  # последняя скорость файла, байт/с (только для журнала)
    done_bytes: float = 0.0      # байт получено (из строки прогресса или размер после завершения)
    total_bytes: float = 0.0     # размер файла, байт; 0 — ещё не известен
    recognised: int = 0          # распознанных строк прогресса по файлу


@dataclass
class WeightsMonitor:
    clock: Callable[[], float]
    log: Callable[..., None]          # log(текст, warn=False): подробные строки в файл; warn=True — предупреждение
    # Порог правила скорости берётся из speed_min_mbps (агрегат всех файлов за окно).
    # speed_check_pct принимается для совместимости конфигурации и правилом скорости больше не используется:
    # срез по проценту заменён окном измерения.
    speed_check_pct: int
    speed_min_mbps: float             # порог: половина min_inet_mbps входа (speed_threshold_mbps), агрегат всех файлов;
                                      # проверить на реальной линии (замер на арендованной карте, SPEC 13.1)
    progress_step_pct: int
    stall_sec: float
    stage_max_sec: float
    # console(вид, проценты): вид «start» (начало этапа), «progress» (шаг процентов), «loaded» (все файлы на месте)
    console: Optional[Callable[..., None]] = None
    profile_key: str = "ltx25"        # ключ профиля модели для строк консоли (не отображаемое имя)
    min_window_sec: float = MIN_WINDOW_SEC
    min_samples: int = MIN_SAMPLES
    seen_start: bool = False          # первая строка файла уже пришла
    stage_done: bool = False          # «все файлы на месте» пришла
    stage_started: float = 0.0
    last_change: float = 0.0          # общая отметка последнего события файла (для совместимости)
    active: dict = field(default_factory=dict)   # активные файлы: ключ (имя без подкаталога) -> _FileState
    files: dict = field(default_factory=dict)    # все файлы этапа (активные и завершённые) -> _FileState
    resolved: set = field(default_factory=set)   # номера «[i/N]», по которым пришла строка setup.sh
    stage_count: int = 0                         # N из «[i/N]»
    window_first: Optional[float] = None         # время первого замера этапа
    window_first_bytes: float = 0.0              # сумма байт всех файлов в момент первого замера
    window_samples: int = 0                      # замеров (выгрузок с прогрессом, не строк) с первого
    window_note_logged: bool = False
    reason_sent: bool = False                    # причину снятия уже отдали: повторов нет
    console_step: int = 0                        # последний напечатанный шаг процентов
    start_noted: bool = False
    unrecognised: int = 0                        # строк «[#» с «%», которые не разобрались
    _unknown_warned: bool = False
    _pending: dict = field(default_factory=dict)      # строки прогресса выгрузки: файл -> (состояние, событие), последняя
    _pull_sampled: bool = False                       # в текущей выгрузке замер уже учтён (одна выгрузка — один замер)
    _unknown_forms: set = field(default_factory=set)  # формы нераспознанных строк, уже записанных в журнал

    def feed(self, lines) -> Optional[str]:
        """Обрабатывает одну выгрузку (ответ GET /logs) по порядку. Возвращает причину снятия (первую) или None.

        Вся выгрузка имеет одну отметку времени. Строки прогресса копятся по файлам, и из каждого файла
        остаётся последняя; они применяются перед событиями файла (начало, завершение, ошибка) и в конце выгрузки.
        """
        now = self.clock()
        self._pull_sampled = False
        for line in lines:
            ev = parse_line(line)
            if ev.kind == "progress":
                state = self._target(ev)
                if state is not None:
                    self._pending[state.name] = (state, ev)
                continue
            if ev.kind in _FLUSH_KINDS:
                reason = self._flush(now)
                if reason:
                    return reason
            reason = self._feed_one(ev, now)
            if reason:
                return reason
        return self._flush(now)

    def _console(self, kind: str, pct: int = 0) -> None:
        if self.console is not None:
            self.console(kind, pct)

    def _reason(self, text: str) -> Optional[str]:
        """Причина снятия отдаётся один раз: повторные вызовы после неё ничего не возвращают."""
        if self.reason_sent:
            return None
        self.reason_sent = True
        return text

    def _feed_one(self, ev: Event, now: float) -> Optional[str]:
        if ev.kind in ("start", "skip", "done", "error") and not self.seen_start:
            self.seen_start = True
            self.stage_started = now
            self.last_change = now
        if ev.index:
            self.stage_count = ev.count
            self.resolved.add(ev.index)
        if ev.kind == "start":
            key = _key(ev.name)
            state = _FileState(name=ev.name, started=now, last_progress=now)
            self.active[key] = state
            self.files[key] = state
            self.last_change = now
            if not self.start_noted:
                self.start_noted = True
                self._console("start")
            self.log(f"Веса: начало загрузки {ev.name}.")
            return None
        if ev.kind == "skip":
            self.log(f"Веса: {ev.name} уже на диске, пропущен.")
            return None
        if ev.kind == "done":
            key = _key(ev.name)
            state = self.files.get(key)
            if state is None:
                state = _FileState(name=ev.name, started=now, last_progress=now)
                self.files[key] = state
            state.done_bytes = ev.done_bytes
            state.total_bytes = ev.total_bytes
            self.active.pop(key, None)
            self.last_change = now
            self.log(f"Веса: {ev.name} загружен ({ev.total_bytes / MIB:.1f} МиБ).")
            self._update_percent()
            return None
        if ev.kind == "aria_done":
            # aria2 сообщил о завершении файла: снимаем его с контроля зависания и считаем полным.
            # Запись в лог делает строка «Файл успешно загружен» из setup.sh, здесь без повтора.
            state = self.files.get(ev.name)
            if state is not None and state.total_bytes > 0:
                state.done_bytes = state.total_bytes
            self.active.pop(ev.name, None)
            self.last_change = now
            self._update_percent()
            return None
        if ev.kind == "error":
            self.active.pop(_key(ev.name), None)
            self.last_change = now
            self.log(f"Веса: ошибка загрузки {ev.name}.", True)
            return self._reason(REASON_FILE_ERROR + ev.name)
        if ev.kind == "stage_done":
            self.stage_done = True
            self.log(f"Веса: этап загрузки весов завершён; строк прогресса не распознано: {self.unrecognised}.")
            self._console("loaded")
            return None
        if ev.kind == "aria_unknown":
            self.unrecognised += 1
            form = _form_key(ev.text)
            if form not in self._unknown_forms:
                self._unknown_forms.add(form)
                self.log(f"Веса: строка с прогрессом не разобрана, полный текст для разбора формата: {ev.text}")
            if not self._unknown_warned:
                self._unknown_warned = True
                self.log(f"{UNRECOGNISED_WARNING[0].upper()}{UNRECOGNISED_WARNING[1:]}: эти строки не "
                         f"используются для скорости и зависания; действуют лимит этапа (stage_max_min) "
                         f"и снятие по ошибке файла.", True)
            return None
        return None  # строки прогресса применяет _flush (см. feed)

    def _target(self, ev: Event) -> Optional[_FileState]:
        """Файл, к которому относится строка прогресса."""
        if ev.file:
            # строка с префиксом: только для файла, который уже начат и не завершён
            return self.active.get(_key(ev.file))
        # старый формат без префикса: только если активен ровно один файл
        if len(self.active) != 1:
            return None
        return next(iter(self.active.values()))

    def _flush(self, now: float) -> Optional[str]:
        """Применяет накопленные строки прогресса: по одному значению на файл и один замер на выгрузку."""
        if not self._pending:
            return None
        items = list(self._pending.values())
        self._pending = {}
        changed = False
        for state, ev in items:
            state.recognised += 1
            state.done_bytes = ev.done_bytes
            state.total_bytes = ev.total_bytes
            state.last_speed_bps = ev.speed_bps
            if ev.pct != state.last_pct:       # тот же процент: ни записи, ни сдвига таймера файла
                changed = True
                state.last_pct = ev.pct
                state.last_progress = now
                self.last_change = now
                self.log(f"Веса: файл {state.name}: {ev.pct}%, скорость {bytes_per_sec_to_mbps(ev.speed_bps):.0f} "
                         f"Мбит/с (только для журнала).")
        # Замер учитывается ДО строки статуса: иначе на выгрузке, которая набирает окно, статус говорит «окно не набрано».
        if not self._pull_sampled:             # одна выгрузка — один замер окна, сколько бы строк в ней ни было
            self._pull_sampled = True
            if self.window_first is None:      # первый замер этапа открывает окно
                self.window_first = now
                self.window_first_bytes = self._agg_done()
            self.window_samples += 1
        if changed:
            self.log(self._status_text(now))
        self._update_percent()
        return self._check_speed(now)

    # --- агрегат этапа ---------------------------------------------------------

    def _agg_done(self) -> float:
        """Байт получено всеми файлами этапа (завершённые — полным размером)."""
        return sum(state.done_bytes for state in self.files.values())

    def _agg_total(self) -> Optional[float]:
        """Общий размер файлов этапа, байт; None, пока он не известен полностью.

        Известен, когда по всем номерам «[i/N]» пришла строка setup.sh (пропущенные файлы не входят
        в загрузку) и у каждого начатого файла есть размер.
        """
        if self.stage_count <= 0 or len(self.resolved) < self.stage_count or not self.files:
            return None
        total = 0.0
        for state in self.files.values():
            if state.total_bytes <= 0:
                return None
            total += state.total_bytes
        return total

    def _window_full(self, now: float) -> bool:
        if self.window_first is None:
            return False
        return (now - self.window_first) >= self.min_window_sec and self.window_samples >= self.min_samples

    def _window_speed(self, now: float) -> Optional[float]:
        """Скорость всех файлов за окно, байт/с: прирост суммы байт / длительность окна."""
        if self.window_first is None:
            return None
        duration = now - self.window_first
        if duration <= 0:
            return None
        return max(0.0, self._agg_done() - self.window_first_bytes) / duration

    def _status_text(self, now: float) -> str:
        """Строка журнала о всём этапе: процент, объём, скорость, средняя, оставшееся время (только файл)."""
        done = self._agg_done()
        total = self._agg_total()
        elapsed = max(0.0, now - self.stage_started)
        if elapsed >= self.min_window_sec:
            avg_text = f"{bytes_per_sec_to_mbps(done / elapsed):.0f} Мбит/с"
        else:                                  # раньше окна средняя не печатается: иначе стартовый всплеск
            avg_text = ESTIMATE_UNAVAILABLE
        full = self._window_full(now)
        speed = self._window_speed(now) if full else None
        if total:
            pct_text = f"{min(100, int(done * 100 // total))}%"
            size_text = f"загружено {done / MIB:.1f} из {total / MIB:.1f} МиБ"
        else:
            pct_text = "итог неизвестен"
            size_text = f"загружено {done / MIB:.1f} МиБ"
        if speed is not None:
            speed_text = f"скорость {bytes_per_sec_to_mbps(speed):.0f} Мбит/с"
        else:
            speed_text = "скорость: окно не набрано"
        if total and speed and speed > 0:
            left_text = f"осталось ~{(total - done) / speed / 60:.1f} мин"
        else:
            left_text = f"осталось: {ESTIMATE_UNAVAILABLE}"
        return (f"Веса: этап: {pct_text}, {size_text}, {speed_text}, "
                f"средняя {avg_text}, {left_text}.")

    def _update_percent(self) -> None:
        """Шаги процентов для консоли: по одной строке на каждый 10%-шаг, только при известном итоге."""
        total = self._agg_total()
        if not total:
            return
        pct = min(100, int(self._agg_done() * 100 // total))
        step_pct = self.progress_step_pct
        step = (pct // step_pct) * step_pct
        if step > 0 and step > self.console_step:
            self.console_step = step
            self._console("progress", step)

    # --- правила ---------------------------------------------------------------

    def _note_window(self) -> None:
        if not self.window_note_logged:
            self.window_note_logged = True
            self.log(WINDOW_WAIT_NOTE)

    def _check_speed(self, now: float) -> Optional[str]:
        """Правило (а): скорость всего этапа за окно ниже speed_min_mbps. Решает только полное окно."""
        if self.reason_sent:
            return None
        if not self._window_full(now):
            self._note_window()
            return None
        speed = self._window_speed(now)
        if speed is None:
            return None
        mbps = bytes_per_sec_to_mbps(speed)
        if mbps >= self.speed_min_mbps:
            return None
        self.log(f"Веса: скорость всех файлов за окно {mbps:.0f} Мбит/с ниже порога {self.speed_min_mbps:g} "
                 f"Мбит/с (окно {now - (self.window_first or now):.0f} с, замеров {self.window_samples}, "
                 f"активных файлов {len(self.active)}).")
        return self._reason(REASON_SPEED)

    def check(self) -> Optional[str]:
        """Правила по времени: (а) по окну, (б) зависание файла по окну, (в) лимит этапа.

        Вызывается на каждом шаге опроса карты. После причины снятия возвращает None.
        """
        if not self.seen_start or self.stage_done or self.reason_sent:
            return None
        now = self.clock()
        if now - self.stage_started >= self.stage_max_sec:
            self.log(f"Веса: этап загрузки идёт {now - self.stage_started:.0f} с, лимит {self.stage_max_sec:.0f} с.")
            return self._reason(REASON_STAGE)
        if not self._window_full(now):
            self._note_window()
            return None
        reason = self._check_speed(now)
        if reason:
            return reason
        # правило (б) считается по каждому файлу: зависший файл снимает карту, даже если другой идёт
        for state in self.active.values():
            idle = now - state.last_progress
            if idle >= self.stall_sec:
                self.log(f"Веса: {state.name}: прогресс не менялся {idle:.0f} с.")
                return self._reason(REASON_STALL)
        return None
