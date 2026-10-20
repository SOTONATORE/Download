"""HTTP-сервис воркера на карте Vast.ai (SPEC разделы 8.1, 8.3(а), 8.4).

Только стандартная библиотека и httpx (через comfy_client) (SPEC 0.2).
Все сообщения на русском (SPEC 0.5). Токен WORKER_TOKEN берётся из окружения
и никогда не попадает в логи и тела ответов (SPEC 0.1).
"""
from __future__ import annotations

import collections
import hashlib
import hmac
import json
import logging
import os
import queue
import re
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Tuple
from urllib.parse import parse_qs

try:
    from Generate.worker.comfy_client import (
        ComfyClient, ComfyError, inject_workflow)
    from Generate.worker.comfy_output import (
        COMFY_LOG_LOGGER, COMFY_MARK, ComfyOutputReader)
    from Generate.worker.telemetry import SystemSampler, TelemetrySampler
except ImportError:  # запуск прямо на карте, без пакета Generate
    from worker.comfy_client import ComfyClient, ComfyError, inject_workflow
    from worker.comfy_output import COMFY_LOG_LOGGER, COMFY_MARK, ComfyOutputReader
    from worker.telemetry import SystemSampler, TelemetrySampler

log = logging.getLogger("Generate.worker.server")

DEFAULT_PORT = 8000
DEFAULT_IDLE_TIMEOUT_MIN = 2
DEFAULT_MAX_UNACKED = 3
DEFAULT_OUTPUT_DIR = "/workspace/ComfyUI/output"
MAX_BODY_BYTES = 1024 * 1024
CHUNK_SIZE = 1024 * 1024
TOKEN_HEADER = "X-Worker-Token"

# Логи (SPEC 8.4, 10): уровень из GEN_LOG_LEVEL, буфер для GET /logs (без вытеснения строк, этап 9в)
LOG_LEVEL_ENV = "GEN_LOG_LEVEL"
LOG_LEVELS = ("INFO", "DEBUG")
LOGS_PATH = "/logs"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
STDOUT_HANDLER_NAME = "worker-stdout"
# Файл вывода ComfyUI, который пишет setup.sh через tee (пусто = чтение не запускается)
COMFY_LOG_ENV = "COMFY_LOG_PATH"
# Сборка образа (WORKER_BUILD из Dockerfile, аргумент docker_build.yml): первая строка вывода и GET /logs
BUILD_ENV = "WORKER_BUILD"
BUILD_UNKNOWN = "unknown"
# Заголовок ответа GET /logs: номер последней строки; передаётся обратно как since (SPEC 8.4)
LOG_NEXT_HEADER = "X-Log-Next"
# Файл готовности (SPEC 8.1): setup.sh создаёт его, когда ComfyUI отвечает; до этого /health даёт ok=false
READY_FLAG_ENV = "WORKER_READY_FLAG"
DEFAULT_READY_FLAG = "/workspace/worker_ready"
# Лог setup.sh и aria2 (onstart перенаправляет туда вывод bootstrap.sh): хвост идёт в буфер GET /logs (SPEC 8.1)
BOOTSTRAP_LOG_ENV = "WORKER_BOOTSTRAP_LOG"
DEFAULT_BOOTSTRAP_LOG = "/workspace/bootstrap.log"
# Лог ComfyUI шаблона образа (SPEC 8.5, B1, B2): путь пока не известен, первый прогон его подтвердит.
# Пустое значение = не настроено: в лог пишется одна строка, файл не читается. Путь в коде не угадывается.
TEMPLATE_COMFY_LOG_ENV = "COMFY_TEMPLATE_LOG_PATH"
TEMPLATE_COMFY_PREFIX = "[comfyui] "
TEMPLATE_COMFY_NOT_SET = ("Лог ComfyUI шаблона не задан (COMFY_TEMPLATE_LOG_PATH): "
                          "строки ComfyUI не читаются.")

# Буфер логов на диске (этап 9в2): строки в файлах-сегментах по SEGMENT_LINES строк, каталог из WORKER_LOG_DIR
SEGMENT_LINES = 10000
SEGMENT_NAME = "seg-{:06d}.log"
LOG_DIR_ENV = "WORKER_LOG_DIR"
DEFAULT_LOG_DIR = "/workspace/worker_logs"

_ID_RE = r"([A-Za-z0-9_-]+)"
_RE_TASK = re.compile(rf"^/task/{_ID_RE}$")
_RE_FILE = re.compile(rf"^/file/{_ID_RE}$")
_RE_ACK = re.compile(rf"^/ack/{_ID_RE}$")
_RE_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


# ---------------------------------------------------------------------------
# Маскирование секретов в логах (SPEC 0.1). До этого воркер не маскировал ничего.
# ---------------------------------------------------------------------------

MASK = "***"
# Значения короче этого не регистрируются: маска испортила бы обычный текст логов.
MIN_SECRET_LEN = 8
_SECRET_VALUES: set = set()
_SECRETS_LOCK = threading.Lock()
# (шаблон, нужна ли группа-префикс): префикс («Authorization:») остаётся, значение заменяется
_SECRET_PATTERNS = (
    (re.compile(r"(?i)(authorization\s*:)[^\r\n]*"), True),
    (re.compile(r"(?i)(x-worker-token\s*[:=]\s*)\S+"), True),
    (re.compile(r"(?i)(bearer\s+)\S+"), True),
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), False),
    (re.compile(r"AIza[0-9A-Za-z_-]{20,}"), False),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), False),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), False),
)


def register_secret(value: Optional[str]) -> None:
    """Регистрирует значение секрета (токен воркера, HF_TOKEN) для маскирования."""
    if value and len(value) >= MIN_SECRET_LEN:
        with _SECRETS_LOCK:
            _SECRET_VALUES.add(value)


def mask_secrets(text: str) -> str:
    """Заменяет известные секреты и выглядящие как ключи строки на «***»."""
    with _SECRETS_LOCK:
        values = sorted(_SECRET_VALUES, key=len, reverse=True)
    for value in values:
        text = text.replace(value, MASK)
    for pattern, keep_prefix in _SECRET_PATTERNS:
        if keep_prefix:
            text = pattern.sub(lambda m: m.group(1) + MASK, text)
        else:
            text = pattern.sub(MASK, text)
    return text


# Адреса перенаправлений aria2 (Hugging Face, CDN) могут нести подписанную query-часть: она не нужна в логе (SPEC 0.1)
_URL_QUERY_RE = re.compile(r"(https?://[^\s?#]+)\?\S*")


def strip_url_queries(text: str) -> str:
    """Заменяет query-часть http(s)-адресов на «?***». Путь адреса остаётся для диагностики."""
    return _URL_QUERY_RE.sub(lambda m: m.group(1) + "?" + MASK, text)


class MaskingFormatter(logging.Formatter):
    """Форматтер, который маскирует всю строку (включая трассировку) перед выводом."""

    def format(self, record: logging.LogRecord) -> str:
        return mask_secrets(super().format(record))


# ---------------------------------------------------------------------------
# Буфер логов и настройка логирования (SPEC 8.4)
# ---------------------------------------------------------------------------

def normalize_build(value: Optional[str]) -> str:
    """Значение сборки для строки «Сборка воркера: …»: без переводов строк и пробелов по краям."""
    text = " ".join(str(value or "").split())
    return text or BUILD_UNKNOWN


def build_line(build: Optional[str]) -> str:
    return f"Сборка воркера: {normalize_build(build)}"


def _escape_entry(text: str) -> str:
    # Запись занимает одну физическую строку файла: обратная косая черта удваивается,
    # перевод строки записывается как два символа «\n» (обратно разбирается в _unescape_entry).
    return text.replace("\\", "\\\\").replace("\n", "\\n")


_UNESCAPE_RE = re.compile(r"\\(.)", re.DOTALL)


def _unescape_entry(text: str) -> str:
    return _UNESCAPE_RE.sub(lambda m: "\n" if m.group(1) == "n" else m.group(1), text)


def worker_log_dir() -> Optional[str]:
    """Каталог сегментов: WORKER_LOG_DIR не задана — DEFAULT_LOG_DIR; задана пустой — временный каталог (None)."""
    raw = os.environ.get(LOG_DIR_ENV)
    if raw is None:
        return DEFAULT_LOG_DIR
    return raw.strip() or None


class LogSegmentStore:
    """Строки лога в файлах seg-NNNNNN.log по SEGMENT_LINES строк. Номер строки сквозной, с 1.

    Строка N лежит в сегменте (N-1)//SEGMENT_LINES + 1 на месте (N-1)%SEGMENT_LINES.
    В памяти держатся только счётчики и позиция незавершённой строки в текущем файле; строки
    читаются с диска. Каждая часть маскируется до записи. Файл сбрасывается после каждой записи.
    Незавершённая строка (часть из ComfyUI) уже в файле, но в GET /logs не видна, пока не закрыта.
    """

    def __init__(self, directory: Optional[str]) -> None:
        self.dir = self._prepare_dir(directory)
        self._remove_previous()
        self._lock = threading.RLock()
        self._seq = 0                          # закрытых строк
        self._seg = 0                          # номер текущего файла
        self._seg_count = 0                    # закрытых строк в текущем файле
        self._fh: Any = None
        self._open_pos: Optional[int] = None   # байт в текущем файле, с которого начата незакрытая строка
        self._queued: collections.deque = collections.deque()  # готовые строки, ждущие закрытия незакрытой
        self._closed = False

    @staticmethod
    def _prepare_dir(directory: Optional[str]) -> Path:
        if directory:
            path = Path(directory)
            try:
                path.mkdir(parents=True, exist_ok=True)
                return path
            except OSError as exc:
                print(f"Каталог логов недоступен ({type(exc).__name__}): логи пишутся во временный каталог.",
                      file=sys.stderr, flush=True)
        return Path(tempfile.mkdtemp(prefix="worker_logs_"))

    def _remove_previous(self) -> None:
        """Файлы сегментов прошлого запуска удаляются при создании хранилища: номера строк начинаются с 1."""
        for path in self.dir.glob("seg-*.log"):
            path.unlink(missing_ok=True)

    def _begin_entry(self) -> None:
        if self._fh is None or self._seg_count >= SEGMENT_LINES:
            if self._fh is not None:
                self._fh.close()
            self._seg += 1
            self._seg_count = 0
            self._fh = open(self.dir / SEGMENT_NAME.format(self._seg), "ab")
        self._fh.seek(0, os.SEEK_END)
        self._open_pos = self._fh.tell()

    def write(self, text: str) -> None:
        """Часть строки (без перевода строки). Маска применяется до записи в файл."""
        data = _escape_entry(mask_secrets(text)).encode("utf-8")
        with self._lock:
            if self._closed:
                return
            if self._open_pos is None:
                self._begin_entry()
            self._fh.write(data)
            self._fh.flush()

    def commit(self) -> None:
        """Закрывает строку: перевод строки в файл, строка получает номер. Затем пишутся строки, ждавшие её."""
        with self._lock:
            if self._closed:
                return
            if self._open_pos is None:
                self._begin_entry()
            self._fh.write(b"\n")
            self._fh.flush()
            self._open_pos = None
            self._seq += 1
            self._seg_count += 1
            self._drain_locked()

    def cancel(self) -> None:
        """Отменяет незавершённую строку: её части удаляются из файла (строка с «\\r» заменяется последним состоянием).
        Трогает только свою незакрытую строку: строки других источников не удаляются."""
        with self._lock:
            if self._closed or self._open_pos is None:
                return
            self._fh.flush()
            os.ftruncate(self._fh.fileno(), self._open_pos)
            self._fh.seek(0, os.SEEK_END)
            self._open_pos = None
            self._drain_locked()

    def write_line(self, parts: list) -> None:
        """Строка целиком (части без перевода строки). Маска, запись и перевод строки выполняются под одной
        блокировкой, поэтому строки других источников в неё не попадают. Если открыта незакрытая строка
        (части длиннее PIECE_LIMIT), эта строка ждёт её закрытия и пишется сразу после неё."""
        data = b"".join(_escape_entry(mask_secrets(part)).encode("utf-8") for part in parts)
        with self._lock:
            if self._closed:
                return
            self._queued.append(data)
            self._drain_locked()

    def _drain_locked(self) -> None:
        """Пишет ожидающие строки, пока нет незакрытой строки. Вызывать под блокировкой."""
        while self._queued and self._open_pos is None and not self._closed:
            data = self._queued.popleft()
            self._begin_entry()
            self._fh.write(data + b"\n")
            self._fh.flush()
            self._open_pos = None
            self._seq += 1
            self._seg_count += 1

    def add_entry(self, text: str) -> None:
        self.write_line([text])

    def read(self, since: Optional[int] = None, tail: Optional[int] = None) -> Tuple[list, int]:
        """(строки, номер последней строки). since: строки с номером больше since; tail без since."""
        with self._lock:
            last = self._seq
            if since is not None:
                first = since + 1
            elif tail is not None:
                first = max(1, last - tail + 1)
            else:
                first = 1
            if first > last:
                return [], last
            out: list = []
            for seg in range((first - 1) // SEGMENT_LINES + 1, (last - 1) // SEGMENT_LINES + 2):
                seg_first = (seg - 1) * SEGMENT_LINES + 1
                seg_last = min(seg * SEGMENT_LINES, last)
                parts = self._segment_bytes(seg).split(b"\n")
                chosen = parts[max(first, seg_first) - seg_first: seg_last - seg_first + 1]
                out.extend(_unescape_entry(p.decode("utf-8", errors="replace")) for p in chosen)
            return out, last

    def _segment_bytes(self, seg: int) -> bytes:
        with open(self.dir / SEGMENT_NAME.format(seg), "rb") as fh:
            return fh.read()

    @property
    def count(self) -> int:
        return self._seq

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._queued.clear()
            if self._fh is not None:
                self._fh.close()
                self._fh = None


class LogBuffer:
    """Буфер логов за весь запуск воркера: строки лежат на диске в сегментах (LogSegmentStore).
    Строки не вытесняются. max_lines принимается для совместимости и не используется (этап 9в2).
    Каждая строка имеет номер с 1 в порядке поступления.
    """

    def __init__(self, log_dir: Optional[str] = None, max_lines: Optional[int] = None) -> None:
        self._store = LogSegmentStore(log_dir)

    def add(self, line: str) -> None:
        self._store.add_entry(line)

    def write_piece(self, text: str) -> None:
        self._store.write(text)

    def end_piece(self) -> None:
        self._store.commit()

    def cancel_piece(self) -> None:
        self._store.cancel()

    def write_line(self, parts: list) -> None:
        self._store.write_line(parts)

    def read(self, since: Optional[int] = None, tail: Optional[int] = None) -> Tuple[list, int]:
        """(строки, номер последней строки). since: только строки с номером больше since; tail без since."""
        return self._store.read(since=since, tail=tail)

    def tail(self, n: Optional[int] = None) -> list:
        return self._store.read(tail=n)[0]

    def __len__(self) -> int:
        return self._store.count

    def close(self) -> None:
        self._store.close()


_COMFY_RECORD_FORMAT = logging.Formatter(LOG_FORMAT)


def _comfy_header() -> str:
    """Начало строки ComfyUI в буфере: как у записи логгера COMFY_LOG_LOGGER (время, уровень, имя, пометка)."""
    record = logging.LogRecord(COMFY_LOG_LOGGER, logging.INFO, __file__, 0, COMFY_MARK, None, None)
    return _COMFY_RECORD_FORMAT.format(record)


class _LineSink:
    """Приёмник частей строки для ComfyOutputReader. Части копятся в памяти и пишутся в буфер одной записью
    по окончании строки (LogSegmentStore.write_line): строка ComfyUI не перемешивается с записями других
    источников (телеметрия, журнал, bootstrap.log). Только строка длиннее PIECE_LIMIT без перевода строки
    начинает писаться в буфер как незакрытая запись по частям (write_piece), как и раньше.
    header — начало строки (для ComfyUI: как у записи логгера), prefix — пометка перед текстом."""

    def __init__(self, logbuf: LogBuffer, *, header: Optional[Callable[[], str]] = None,
                 prefix: str = "") -> None:
        self._logbuf = logbuf
        self._header = header
        self._prefix = prefix
        self._parts: list = []
        self._size = 0
        self._spilled = False          # строка уже открыта в буфере по частям

    def write_piece(self, text: str) -> None:
        if self._spilled:
            self._logbuf.write_piece(text)
            return
        if not self._parts:            # первая часть строки: начало и пометка
            text = (self._header() if self._header else "") + self._prefix + text
        self._parts.append(text)
        self._size += len(text)
        if self._size > PIECE_LIMIT:
            self._spilled = True
            self._logbuf.write_piece("".join(self._parts))
            self._parts = []

    def cancel_line(self) -> None:
        if self._spilled:
            self._logbuf.cancel_piece()
        self._reset()

    def end_line(self) -> None:
        if self._spilled:
            self._logbuf.end_piece()
        elif self._parts:
            self._logbuf.write_line(self._parts)
        self._reset()

    def _reset(self) -> None:
        self._parts = []
        self._size = 0
        self._spilled = False


class BufferHandler(logging.Handler):
    """Пишет отформатированные записи в LogBuffer (то же, что уходит в stdout)."""

    def __init__(self, buffer: LogBuffer) -> None:
        super().__init__()
        self.buffer = buffer
        self.setFormatter(MaskingFormatter(LOG_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.add(self.format(record))
        except Exception:  # журнал не должен ронять обработку запроса
            self.handleError(record)


def read_log_level(raw: Optional[str]) -> Tuple[str, bool]:
    """(уровень, корректно ли задан). Пусто = INFO. Неизвестное значение = INFO с предупреждением.

    Значение в сообщении не печатается: оно может оказаться чем угодно (SPEC 0.1).
    """
    value = (raw or "").strip()
    if not value:
        return "INFO", True
    if value in LOG_LEVELS:
        return value, True
    return "INFO", False


def configure_logging(level: str) -> None:
    """Вывод в stdout с маскированием; уровень корневого логгера = level."""
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == STDOUT_HANDLER_NAME:
            root.removeHandler(existing)
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(STDOUT_HANDLER_NAME)
    handler.setFormatter(MaskingFormatter(LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level))


class BacklogFullError(Exception):
    """Слишком много готовых, но не подтверждённых клипов."""


class RangeNotSatisfiableError(Exception):
    """Запрошенный диапазон байт не пересекается с файлом."""


# ---------------------------------------------------------------------------
# Диапазоны (RFC 7233)
# ---------------------------------------------------------------------------

def parse_range(header: Optional[str], size: int) -> Optional[Tuple[int, int]]:
    """Разбирает заголовок Range. None = отдать файл целиком.

    Некорректный по синтаксису или многодиапазонный заголовок игнорируется (200).
    Неудовлетворимый диапазон вызывает RangeNotSatisfiableError (416).
    """
    if not header:
        return None
    match = _RE_RANGE.match(header.strip())
    if not match:
        return None
    first, last = match.groups()
    if not first and not last:
        return None
    if not first:  # суффикс: последние N байт
        length = int(last)
        if length == 0 or size == 0:
            raise RangeNotSatisfiableError()
        return max(size - length, 0), size - 1
    start = int(first)
    if start >= size:
        raise RangeNotSatisfiableError()
    end = int(last) if last else size - 1
    if end < start:
        return None
    return start, min(end, size - 1)


# ---------------------------------------------------------------------------
# Dead-Man Switch (SPEC 8.3(а))
# ---------------------------------------------------------------------------

class DeadManSwitch:
    """Вызывает callback, если не было аутентифицированных запросов дольше timeout."""

    def __init__(self, timeout_seconds: float, callback: Callable[[], None],
                 check_interval: float = 5.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.timeout = float(timeout_seconds)
        self._callback = callback
        self._interval = check_interval
        self._clock = clock
        self._last = clock()
        self._fired = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def fired(self) -> bool:
        return self._fired

    def touch(self) -> None:
        with self._lock:
            self._last = self._clock()

    def check(self) -> bool:
        """Одна проверка. True, если выключатель сработал (сейчас или ранее)."""
        with self._lock:
            if self._fired:
                return True
            if self._clock() - self._last < self.timeout:
                return False
            self._fired = True
        # воркер только останавливает себя; уничтожение карты делает наблюдатель (SPEC 8.3, а)
        log.error("Нет запросов от раннера дольше %g с: остановка воркера.", self.timeout)
        try:
            self._callback()
        except Exception as exc:  # callback не должен ронять монитор
            log.error("Ошибка в обработчике остановки воркера: %s.", type(exc).__name__)
        return True

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            if self.check():
                break

    def start(self) -> None:
        if self._thread is not None:
            return
        self.touch()
        self._thread = threading.Thread(target=self._loop, name="dead-man-switch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Менеджер задач
# ---------------------------------------------------------------------------

@dataclass
class _Task:
    task_id: str
    num: int
    prompt: str
    num_frames: int
    seed: int
    received: str = ""
    state: str = "queued"
    stage: str = "в очереди"
    error: Optional[str] = None
    path: Optional[Path] = None
    size_bytes: Optional[int] = None
    acked: bool = False
    lines: Deque[str] = field(default_factory=collections.deque)  # журнал клипа целиком, без хвоста (этап 9в)


def _clock_text() -> str:
    return time.strftime("%H:%M:%S")


def prompt_fingerprint(prompt: str) -> str:
    """Короткий SHA-256 префикс промпта: по нему можно сверить клип с prompts.json без текста."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def _scrub(task: "_Task", text: str) -> str:
    """Текст без текста промпта и без секретов: в логи и ответы попадает только это (SPEC 0.1)."""
    if task.prompt:
        text = text.replace(task.prompt, "[текст промпта скрыт]")
    return mask_secrets(text)


class TaskManager:
    """Последовательно выполняет задачи через ComfyUI (одна карта = один клип за раз)."""

    def __init__(self, comfy: Any, workflow: Optional[dict] = None,
                 bindings: Optional[dict] = None, output_dir: Any = DEFAULT_OUTPUT_DIR,
                 extra_inputs: Optional[dict] = None, max_unacked: int = DEFAULT_MAX_UNACKED,
                 task_timeout: float = 600.0,
                 on_event: Optional[Callable[[str], None]] = None) -> None:
        self.comfy = comfy
        self.workflow = workflow if workflow is not None else {}
        self.bindings = bindings if bindings is not None else {}
        self.output_dir = Path(output_dir)
        self.extra_inputs = extra_inputs or {}
        self.max_unacked = max_unacked
        self.task_timeout = task_timeout
        self._on_event = on_event  # снимок телеметрии при старте и конце клипа; не должен мешать задаче
        self._tasks: Dict[str, _Task] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    # --- жизненный цикл ---
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="task-runner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)

    # --- публичный API ---
    def _unacked_locked(self) -> int:
        return sum(1 for t in self._tasks.values() if t.state == "done" and not t.acked)

    def unacked_count(self) -> int:
        with self._lock:
            return self._unacked_locked()

    def busy(self) -> bool:
        with self._lock:
            return any(t.state in ("queued", "running") for t in self._tasks.values())

    def _open_task_locked(self, num: int) -> Optional[_Task]:
        """Задача этого клипа, ещё не подтверждённая и не завершившаяся ошибкой (SPEC 8.1, идемпотентность)."""
        for task in self._tasks.values():
            if task.num == num and not task.acked and task.state != "error":
                return task
        return None

    def submit(self, num: int, prompt: str, num_frames: int, seed: int) -> str:
        """Ставит клип в очередь. Идемпотентность: если для того же номера есть открытая задача
        (не подтверждена и не в ошибке), возвращается её id и новая задача не создаётся.
        Проверка идёт до лимита очереди: повтор не должен получить 429 из-за собственной задачи."""
        received = _clock_text()
        with self._lock:
            existing = self._open_task_locked(num)
            if existing is not None:
                task_id = existing.task_id
            else:
                if self._unacked_locked() >= self.max_unacked:
                    raise BacklogFullError()
                task_id = uuid.uuid4().hex
                task = _Task(task_id, num, prompt, num_frames, seed, received=received)
                self._log(task, "задача принята, в очереди")
                self._tasks[task_id] = task
        if existing is not None:
            log.info("Клип %d уже принят (id %s): повтор не создаёт новую задачу.", num, task_id[:8])
            return task_id
        self._queue.put(task_id)
        # текст промпта в лог не пишем: только номер клипа, время, длина и хеш (SPEC 0.1)
        log.info("Клип %d принят в %s (id %s, промпт %d симв., sha256 %s).",
                 num, received, task_id[:8], len(prompt), prompt_fingerprint(prompt))
        return task_id

    def snapshot(self, task_id: str) -> Optional[dict]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return {"state": task.state, "stage": task.stage,
                    "log_tail": "\n".join(task.lines),
                    "size_bytes": task.size_bytes, "error": task.error}

    def file_info(self, task_id: str) -> Optional[Tuple[Path, int]]:
        """(путь, размер) готового и не подтверждённого файла, иначе None."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.state != "done" or task.acked or task.path is None:
                return None
            path = task.path
        try:
            return path, path.stat().st_size
        except OSError:
            return None

    def ack(self, task_id: str) -> bool:
        """Подтверждение скачивания: удаляет файл. False, если задачи нет."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            path = task.path
            task.acked = True
            task.size_bytes = None
            task.path = None
            self._log(task, "скачивание подтверждено, файл удалён")
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.error("Не удалось удалить файл клипа: %s.", type(exc).__name__)
        log.info("Клип подтверждён и удалён (id %s).", task_id[:8])
        return True

    # --- внутреннее ---
    @staticmethod
    def _log(task: _Task, line: str) -> None:
        # журнал клипа отдаётся целиком (log_tail): строка маскируется при записи (этап 9в)
        task.lines.append(mask_secrets(f"[{time.strftime('%H:%M:%S')}] {line}"))

    def _set(self, task: _Task, state: str, stage: str) -> None:
        with self._lock:
            task.state = state
            task.stage = stage
            self._log(task, stage)

    def _loop(self) -> None:
        while True:
            task_id = self._queue.get()
            if task_id is None:
                return
            with self._lock:
                task = self._tasks.get(task_id)
            if task is not None:
                self._run(task)

    def _resolve_output(self, rel: str) -> Path:
        root = self.output_dir.resolve()
        full = (root / rel).resolve()
        if root != full and root not in full.parents:
            raise ComfyError("Путь результата выходит за пределы папки вывода ComfyUI.")
        if not full.is_file():
            raise ComfyError("Файл результата не найден на диске.")
        return full

    def _notify(self, text: str) -> None:
        """Событие для телеметрии: только постановка в очередь снимка, ошибка здесь клип не задевает."""
        if self._on_event is None:
            return
        try:
            self._on_event(text)
        except Exception:
            pass

    def _run(self, task: _Task) -> None:
        self._notify(f"начало клипа {task.num}")
        try:
            self._run_body(task)
        finally:
            self._notify(f"конец клипа {task.num}")

    def _run_body(self, task: _Task) -> None:
        started: Optional[float] = None
        try:
            self._set(task, "running", "подстановка параметров в workflow")
            prefix = f"clip_{task.num}_{task.task_id[:8]}"
            graph = inject_workflow(self.workflow, self.bindings, task.prompt,
                                    task.num_frames, task.seed, prefix, self.extra_inputs)
            self._set(task, "running", "постановка в очередь ComfyUI")
            started = time.monotonic()
            log.info("Клип %d: старт рендера в %s (id %s).",
                     task.num, _clock_text(), task.task_id[:8])
            prompt_id = self.comfy.queue_prompt(graph)
            self._set(task, "running", "генерация в ComfyUI")
            rel = self.comfy.wait_for_completion(prompt_id, timeout=self.task_timeout)
            path = self._resolve_output(rel)
            size = path.stat().st_size
            duration = time.monotonic() - started
            with self._lock:
                task.path = path
                task.size_bytes = size
            self._set(task, "done", "готово")
            log.info("Клип %d: конец рендера в %s, длительность %.1f с, размер %d байт (id %s).",
                     task.num, _clock_text(), duration, size, task.task_id[:8])
        except ComfyError as exc:
            self._fail(task, str(exc), started)
        except Exception as exc:  # без подробностей, чтобы не утекли лишние данные
            self._fail(task, f"Внутренняя ошибка воркера: {type(exc).__name__}.", started)
            # трассировка только на уровне DEBUG: в ней могут оказаться фрагменты данных запроса
            log.debug("Трассировка клипа %d:\n%s", task.num, _scrub(task, traceback.format_exc()))

    def _fail(self, task: _Task, message: str, started: Optional[float] = None) -> None:
        # полный текст ошибки, но без промпта и секретов (SPEC 0.1)
        message = _scrub(task, message)
        took = f", через {time.monotonic() - started:.1f} с" if started is not None else ""
        with self._lock:
            task.state = "error"
            task.stage = "ошибка"
            task.error = message
            self._log(task, f"ошибка: {message}")
        log.error("Задача клипа %d завершилась ошибкой%s: %s", task.num, took, message)


# ---------------------------------------------------------------------------
# HTTP-слой
# ---------------------------------------------------------------------------

def _validate_payload(data: Any) -> Tuple[int, str, int, int]:
    if not isinstance(data, dict):
        raise ValueError("Тело запроса должно быть JSON-объектом.")

    def _int(name: str, minimum: Optional[int] = None) -> int:
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Поле «{name}» должно быть целым числом.")
        if minimum is not None and value < minimum:
            raise ValueError(f"Поле «{name}» должно быть не меньше {minimum}.")
        return value

    num = _int("num")
    num_frames = _int("num_frames", 1)
    seed = _int("seed")
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Поле «prompt» должно быть непустой строкой.")
    return num, prompt, num_frames, seed


def _make_handler(app: "WorkerApp") -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "VideogenWorker"
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            log.debug("HTTP: " + format, *args)

        # --- ответы ---
        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        # --- аутентификация (SPEC 8.1) ---
        def _authorized(self) -> bool:
            given = self.headers.get(TOKEN_HEADER)
            if not given:
                self._error(401, "Требуется заголовок аутентификации воркера.")
                return False
            if not hmac.compare_digest(given.encode("utf-8"), app.token.encode("utf-8")):
                self._error(401, "Неверный токен воркера.")
                return False
            app.deadman.touch()
            return True

        # --- маршрутизация ---
        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def _dispatch(self, method: str) -> None:
            path = self.path.split("?", 1)[0]
            if not self._authorized():
                self._drain_body()
                return
            try:
                if path == "/health" and method == "GET":
                    return self._health()
                if path == LOGS_PATH and method == "GET":
                    return self._get_logs()
                if path == "/task" and method == "POST":
                    return self._post_task()
                m = _RE_TASK.match(path)
                if m and method == "GET":
                    return self._get_task(m.group(1))
                m = _RE_FILE.match(path)
                if m and method == "GET":
                    return self._get_file(m.group(1))
                m = _RE_ACK.match(path)
                if m and method == "POST":
                    return self._post_ack(m.group(1))
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                log.error("Ошибка обработки запроса: %s.", type(exc).__name__)
                try:
                    self._error(500, "Внутренняя ошибка воркера.")
                except OSError:
                    pass
                return
            known = path in ("/health", "/task", LOGS_PATH) or _RE_TASK.match(path) \
                or _RE_FILE.match(path) or _RE_ACK.match(path)
            self._drain_body()
            if known:
                self._error(405, "Метод не поддерживается для этого пути.")
            else:
                self._error(404, "Маршрут не найден.")

        def _drain_body(self) -> None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if 0 < length <= MAX_BODY_BYTES:
                self.rfile.read(length)

        # --- обработчики ---
        def _health(self) -> None:
            ok, gpu, vram_free = False, "", 0
            try:
                stats = app.comfy.check_health()
                devices = stats.get("devices") or []
                dev = devices[0] if devices and isinstance(devices[0], dict) else {}
                gpu = str(dev.get("name") or "")
                vram_free = int(dev.get("vram_free") or 0)
                ok = True
            except (ComfyError, AttributeError, TypeError, ValueError) as exc:
                log.error("Проверка состояния ComfyUI не пройдена: %s.", type(exc).__name__)
            # до файла готовности ok всегда false: веса и ComfyUI ещё не проверены setup.sh (SPEC 8.1)
            self._json(200, {"ok": bool(ok and app.is_ready()), "gpu": gpu, "vram_free": vram_free,
                             "busy": app.manager.busy()})

        def _get_logs(self) -> None:
            """GET /logs[?since=N][&tail=N]: буфер логов текстом, одна запись на строку (только с токеном).

            since=N: только строки с номером больше N (номер строки = её место с 1 в буфере).
            tail действует только без since. Заголовок X-Log-Next: номер последней строки;
            его передают в следующем запросе как since (SPEC 8.4).
            """
            params = parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            since: Optional[int] = None
            tail: Optional[int] = None
            raw_since = params.get("since", [None])[0]
            if raw_since is not None:
                try:
                    since = int(raw_since)
                except ValueError:
                    since = -1
                if since < 0:
                    return self._error(400, "Параметр «since» должен быть целым неотрицательным числом.")
            raw_tail = params.get("tail", [None])[0]
            if raw_tail is not None and since is None:
                try:
                    tail = int(raw_tail)
                except ValueError:
                    tail = -1
                if tail < 0:
                    return self._error(400, "Параметр «tail» должен быть целым неотрицательным числом.")
            lines, last_index = app.logbuf.read(since=since, tail=tail)
            body = "".join(line + "\n" for line in lines).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header(LOG_NEXT_HEADER, str(last_index))
            self.end_headers()
            self.wfile.write(body)

        def _post_task(self) -> None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._error(400, "Некорректный заголовок Content-Length.")
            if length <= 0:
                return self._error(400, "Пустое тело запроса.")
            if length > MAX_BODY_BYTES:
                return self._error(413, "Тело запроса слишком большое.")
            raw = self.rfile.read(length)
            try:
                num, prompt, num_frames, seed = _validate_payload(json.loads(raw.decode("utf-8")))
            except (ValueError, UnicodeDecodeError) as exc:
                msg = str(exc) if isinstance(exc, ValueError) and not isinstance(
                    exc, (json.JSONDecodeError, UnicodeDecodeError)) else "Тело запроса — некорректный JSON."
                return self._error(400, msg)
            try:
                task_id = app.manager.submit(num, prompt, num_frames, seed)
            except BacklogFullError:
                return self._error(429, "Очередь заполнена, ожидается скачивание готовых клипов")
            self._json(200, {"task_id": task_id})

        def _get_task(self, task_id: str) -> None:
            snap = app.manager.snapshot(task_id)
            if snap is None:
                return self._error(404, "Задача не найдена.")
            self._json(200, snap)

        def _post_ack(self, task_id: str) -> None:
            self._drain_body()
            if not app.manager.ack(task_id):
                return self._error(404, "Задача не найдена.")
            self._json(200, {"ok": True})

        def _get_file(self, task_id: str) -> None:
            info = app.manager.file_info(task_id)
            if info is None:
                return self._error(404, "Файл не готов или не найден.")
            path, size = info
            try:
                rng = parse_range(self.headers.get("Range"), size)
            except RangeNotSatisfiableError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Type", "application/json; charset=utf-8")
                body = json.dumps({"error": "Диапазон не может быть удовлетворён."},
                                  ensure_ascii=False).encode("utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            try:
                fh = open(path, "rb")
            except OSError:
                return self._error(404, "Файл не готов или не найден.")
            with fh:
                if rng is None:
                    start, end, status = 0, size - 1, 200
                else:
                    (start, end), status = rng, 206
                length = max(end - start + 1, 0)
                self.send_response(status)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------

class WorkerApp:
    def __init__(self, token: str, comfy: Any, *, host: str = "0.0.0.0",
                 port: int = DEFAULT_PORT,
                 idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_MIN * 60.0,
                 on_idle_timeout: Optional[Callable[[], None]] = None,
                 check_interval: float = 5.0, workflow: Optional[dict] = None,
                 bindings: Optional[dict] = None, output_dir: Any = DEFAULT_OUTPUT_DIR,
                 extra_inputs: Optional[dict] = None,
                 max_unacked: int = DEFAULT_MAX_UNACKED,
                 task_timeout: float = 600.0,
                 log_level: str = "INFO",
                 build_id: str = BUILD_UNKNOWN,
                 comfy_log_path: Optional[str] = None,
                 telemetry_gpu_stats: Optional[Callable[[], dict]] = None,
                 telemetry_interval_check: float = 1.0,
                 ready_flag: Optional[str] = None,
                 bootstrap_log_path: Optional[str] = None,
                 template_log_path: Optional[str] = None,
                 log_dir: Optional[str] = None) -> None:
        if not token:
            raise ValueError("Не задан токен воркера: переменная окружения WORKER_TOKEN пуста.")
        if log_level not in LOG_LEVELS:
            raise ValueError("Уровень логов воркера должен быть INFO или DEBUG.")
        self.token = token
        self.log_level = log_level
        register_secret(token)
        register_secret(os.environ.get("HF_TOKEN", ""))
        self.logbuf = LogBuffer(log_dir)  # строки на диске, сегменты по SEGMENT_LINES (этап 9в2)
        self.build_id = normalize_build(build_id)
        self.logbuf.add(build_line(self.build_id))  # первая строка буфера (SPEC 8.4)
        self._log_handler = BufferHandler(self.logbuf)
        self._log_attached = False
        self.comfy = comfy
        # Телеметрия: видеопамять берётся из /system_stats ComfyUI (check_health), см. telemetry.py
        self.sysampler = SystemSampler(gpu_stats=telemetry_gpu_stats or comfy.check_health,
                                       disk_path=output_dir)
        self.telemetry = TelemetrySampler(self.sysampler.snapshot, log_level,
                                          check_interval=telemetry_interval_check)
        self.manager = TaskManager(comfy, workflow, bindings, output_dir, extra_inputs,
                                   max_unacked, task_timeout, on_event=self.telemetry.notify)
        self.deadman = DeadManSwitch(idle_timeout_seconds,
                                     on_idle_timeout or self._default_shutdown,
                                     check_interval)
        # Вывод ComfyUI: файл из setup.sh; без пути строки ComfyUI в буфер не попадают
        self.comfy_reader: Optional[ComfyOutputReader] = None
        if comfy_log_path:
            self.comfy_reader = ComfyOutputReader(
                comfy_log_path, _LineSink(self.logbuf, header=_comfy_header), mask_secrets)
        # Файл готовности: без пути /health не ждёт файла (тесты, прямой запуск); в main() путь всегда задан
        self.ready_flag: Optional[Path] = Path(ready_flag) if ready_flag else None
        # Лог setup.sh и aria2: хвост с начала файла, маска и срез query-частей адресов (SPEC 8.1, 0.1)
        self.bootstrap_reader: Optional[ComfyOutputReader] = None
        if bootstrap_log_path:
            self.bootstrap_reader = ComfyOutputReader(
                bootstrap_log_path, _LineSink(self.logbuf),
                lambda text: mask_secrets(strip_url_queries(text)))
        # Лог ComfyUI шаблона: тот же читатель, что у bootstrap.log (без второго), строки с пометкой [comfyui]
        self.template_reader: Optional[ComfyOutputReader] = None
        self.template_log_path = template_log_path or None
        if self.template_log_path:
            self.template_reader = ComfyOutputReader(
                self.template_log_path, _LineSink(self.logbuf, prefix=TEMPLATE_COMFY_PREFIX),
                lambda text: mask_secrets(strip_url_queries(text)))
        self.httpd = ThreadingHTTPServer((host, port), _make_handler(self))
        self.httpd.daemon_threads = True
        self._serve_thread: Optional[threading.Thread] = None

    def is_ready(self) -> bool:
        """Файл готовности есть (или путь не задан). Его создаёт setup.sh после ответа ComfyUI (SPEC 8.1)."""
        return self.ready_flag is None or self.ready_flag.exists()

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def _default_shutdown(self) -> None:
        """По умолчанию: останавливает HTTP-сервер, процесс воркера завершается."""
        threading.Thread(target=self.httpd.shutdown, name="idle-shutdown", daemon=True).start()

    def _attach_logs(self) -> None:
        """Буфер получает записи логгера «Generate.worker» (и его потомков); уровень из GEN_LOG_LEVEL.
        Строки ComfyUI идут в буфер напрямую (propagate=False): их уже выводит tee в контейнерный лог."""
        worker_log = logging.getLogger("Generate.worker")
        worker_log.setLevel(getattr(logging, self.log_level))
        if not self._log_attached:
            worker_log.addHandler(self._log_handler)
            comfy_log = logging.getLogger(COMFY_LOG_LOGGER)
            comfy_log.addHandler(self._log_handler)
            comfy_log.propagate = False
            self._log_attached = True

    def _detach_logs(self) -> None:
        if self._log_attached:
            logging.getLogger("Generate.worker").removeHandler(self._log_handler)
            comfy_log = logging.getLogger(COMFY_LOG_LOGGER)
            comfy_log.removeHandler(self._log_handler)
            comfy_log.propagate = True
            self._log_attached = False

    def _start_workers(self) -> None:
        self._attach_logs()
        log.info("Воркер запущен: уровень логов %s, буфер логов без ограничения числа строк.",
                 self.log_level)
        self.manager.start()
        self.deadman.start()
        self.telemetry.start()  # первый снимок «запуск воркера» ставится в очередь здесь
        if self.comfy_reader is not None:
            self.comfy_reader.start()
        if self.bootstrap_reader is not None:
            self.bootstrap_reader.start()
        if self.template_reader is not None:
            self.template_reader.start()
        else:
            self.logbuf.add(TEMPLATE_COMFY_NOT_SET)  # B1: одна строка при старте, если путь не задан

    def serve_forever(self) -> None:
        self._start_workers()
        log.info("Воркер слушает порт %d.", self.port)
        try:
            self.httpd.serve_forever()
        finally:
            self.close()

    def start_background(self) -> threading.Thread:
        self._start_workers()
        self._serve_thread = threading.Thread(target=self.httpd.serve_forever,
                                              name="worker-http", daemon=True)
        self._serve_thread.start()
        return self._serve_thread

    def close(self) -> None:
        log.info("Воркер останавливается.")
        self.deadman.stop()
        self.manager.stop()
        self.telemetry.stop()
        if self.comfy_reader is not None:
            self.comfy_reader.stop()
        if self.bootstrap_reader is not None:
            self.bootstrap_reader.stop()
        if self.template_reader is not None:
            self.template_reader.stop()
        if self._serve_thread is not None:
            self.httpd.shutdown()
            self._serve_thread.join(timeout=5)
            self._serve_thread = None
        self.httpd.server_close()
        self._detach_logs()
        self.logbuf.close()


def _load_json_file(path: Optional[str], what: str) -> dict:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Не удалось прочитать {what} из файла «{path}»: {type(exc).__name__}.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{what.capitalize()} в файле «{path}» должен быть JSON-объектом.")
    return data


def main() -> int:
    build = os.environ.get(BUILD_ENV, "")
    # первая строка вывода контейнера: по ней видно, какой коммит собран в образе
    print(build_line(build), flush=True)
    level, level_ok = read_log_level(os.environ.get(LOG_LEVEL_ENV))
    configure_logging(level)
    if not level_ok:
        # значение не печатаем; на карте лучше продолжить с INFO, чем потерять оплаченный запуск
        log.warning("Значение %s не распознано (допустимо INFO или DEBUG): используется INFO.",
                    LOG_LEVEL_ENV)
    token = os.environ.get("WORKER_TOKEN", "")
    if not token:
        log.error("Не задана переменная окружения WORKER_TOKEN, запуск невозможен.")
        return 2
    try:
        port = int(os.environ.get("WORKER_HTTP_PORT", DEFAULT_PORT))
        idle_min = float(os.environ.get("WORKER_IDLE_TIMEOUT_MIN", DEFAULT_IDLE_TIMEOUT_MIN))
        workflow = _load_json_file(os.environ.get("WORKER_WORKFLOW_PATH"), "workflow")
        bindings = _load_json_file(os.environ.get("WORKER_BINDINGS_PATH"), "привязки")
    except ValueError as exc:
        log.error("Некорректная настройка воркера: %s", exc)
        return 2
    # ComfyUI образа слушает 18188 (COMFYUI_ARGS в env карты, SPEC 8.5)
    comfy = ComfyClient(os.environ.get("COMFY_URL", "http://127.0.0.1:18188"))
    app = WorkerApp(token, comfy, port=port, idle_timeout_seconds=idle_min * 60.0,
                    workflow=workflow, bindings=bindings,
                    output_dir=os.environ.get("COMFY_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
                    log_level=level, build_id=build,
                    # COMFY_LOG_PATH больше не читается: путь лога ComfyUI шаблона задаёт COMFY_TEMPLATE_LOG_PATH
                    template_log_path=os.environ.get(TEMPLATE_COMFY_LOG_ENV, "") or None,
                    # пустое значение не отключает проверку: иначе /health отдал бы ok=true до готовности
                    ready_flag=os.environ.get(READY_FLAG_ENV) or DEFAULT_READY_FLAG,
                    bootstrap_log_path=os.environ.get(BOOTSTRAP_LOG_ENV) or DEFAULT_BOOTSTRAP_LOG,
                    log_dir=worker_log_dir())
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        log.info("Воркер остановлен вручную.")
    finally:
        comfy.close()
    log.info("Воркер завершил работу.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
