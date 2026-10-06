"""HTTP-сервис воркера на карте Vast.ai (SPEC разделы 8.1, 8.3(а), 8.4).

Только стандартная библиотека и httpx (через comfy_client) (SPEC 0.2).
Все сообщения на русском (SPEC 0.5). Токен WORKER_TOKEN берётся из окружения
и никогда не попадает в логи и тела ответов (SPEC 0.1).
"""
from __future__ import annotations

import collections
import hmac
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Tuple

try:
    from Generate.worker.comfy_client import (
        ComfyClient, ComfyError, inject_workflow)
except ImportError:  # запуск прямо на карте, без пакета Generate
    from worker.comfy_client import ComfyClient, ComfyError, inject_workflow

log = logging.getLogger("Generate.worker.server")

DEFAULT_PORT = 8000
DEFAULT_IDLE_TIMEOUT_MIN = 5
DEFAULT_MAX_UNACKED = 3
DEFAULT_OUTPUT_DIR = "/workspace/ComfyUI/output"
MAX_BODY_BYTES = 1024 * 1024
CHUNK_SIZE = 1024 * 1024
TOKEN_HEADER = "X-Worker-Token"

_ID_RE = r"([A-Za-z0-9_-]+)"
_RE_TASK = re.compile(rf"^/task/{_ID_RE}$")
_RE_FILE = re.compile(rf"^/file/{_ID_RE}$")
_RE_ACK = re.compile(rf"^/ack/{_ID_RE}$")
_RE_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


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
        log.error("Нет запросов от раннера дольше %g с: самоуничтожение воркера.", self.timeout)
        try:
            self._callback()
        except Exception as exc:  # callback не должен ронять монитор
            log.error("Ошибка в обработчике самоуничтожения: %s.", type(exc).__name__)
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
    state: str = "queued"
    stage: str = "в очереди"
    error: Optional[str] = None
    path: Optional[Path] = None
    size_bytes: Optional[int] = None
    acked: bool = False
    lines: Deque[str] = field(default_factory=lambda: collections.deque(maxlen=20))


class TaskManager:
    """Последовательно выполняет задачи через ComfyUI (одна карта = один клип за раз)."""

    def __init__(self, comfy: Any, workflow: Optional[dict] = None,
                 bindings: Optional[dict] = None, output_dir: Any = DEFAULT_OUTPUT_DIR,
                 extra_inputs: Optional[dict] = None, max_unacked: int = DEFAULT_MAX_UNACKED,
                 task_timeout: float = 600.0) -> None:
        self.comfy = comfy
        self.workflow = workflow if workflow is not None else {}
        self.bindings = bindings if bindings is not None else {}
        self.output_dir = Path(output_dir)
        self.extra_inputs = extra_inputs or {}
        self.max_unacked = max_unacked
        self.task_timeout = task_timeout
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

    def submit(self, num: int, prompt: str, num_frames: int, seed: int) -> str:
        with self._lock:
            if self._unacked_locked() >= self.max_unacked:
                raise BacklogFullError()
            task_id = uuid.uuid4().hex
            task = _Task(task_id, num, prompt, num_frames, seed)
            self._log(task, "задача принята, в очереди")
            self._tasks[task_id] = task
        self._queue.put(task_id)
        log.info("Принята задача для клипа %d (id %s).", num, task_id[:8])
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
        task.lines.append(f"[{time.strftime('%H:%M:%S')}] {line}")

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

    def _run(self, task: _Task) -> None:
        try:
            self._set(task, "running", "подстановка параметров в workflow")
            prefix = f"clip_{task.num}_{task.task_id[:8]}"
            graph = inject_workflow(self.workflow, self.bindings, task.prompt,
                                    task.num_frames, task.seed, prefix, self.extra_inputs)
            self._set(task, "running", "постановка в очередь ComfyUI")
            prompt_id = self.comfy.queue_prompt(graph)
            self._set(task, "running", "генерация в ComfyUI")
            rel = self.comfy.wait_for_completion(prompt_id, timeout=self.task_timeout)
            path = self._resolve_output(rel)
            size = path.stat().st_size
            with self._lock:
                task.path = path
                task.size_bytes = size
            self._set(task, "done", "готово")
            log.info("Клип %d готов, %d байт (id %s).", task.num, size, task.task_id[:8])
        except ComfyError as exc:
            self._fail(task, str(exc))
        except Exception as exc:  # без подробностей, чтобы не утекли лишние данные
            self._fail(task, f"Внутренняя ошибка воркера: {type(exc).__name__}.")

    def _fail(self, task: _Task, message: str) -> None:
        with self._lock:
            task.state = "error"
            task.stage = "ошибка"
            task.error = message
            self._log(task, f"ошибка: {message}")
        log.error("Задача клипа %d завершилась ошибкой: %s", task.num, message)


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
                self._error(403, "Неверный токен воркера.")
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
            known = path in ("/health", "/task") or _RE_TASK.match(path) \
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
            self._json(200, {"ok": ok, "gpu": gpu, "vram_free": vram_free,
                             "busy": app.manager.busy()})

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
                 task_timeout: float = 600.0) -> None:
        if not token:
            raise ValueError("Не задан токен воркера: переменная окружения WORKER_TOKEN пуста.")
        self.token = token
        self.comfy = comfy
        self.manager = TaskManager(comfy, workflow, bindings, output_dir, extra_inputs,
                                   max_unacked, task_timeout)
        self.deadman = DeadManSwitch(idle_timeout_seconds,
                                     on_idle_timeout or self._default_shutdown,
                                     check_interval)
        self.httpd = ThreadingHTTPServer((host, port), _make_handler(self))
        self.httpd.daemon_threads = True
        self._serve_thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def _default_shutdown(self) -> None:
        """По умолчанию: останавливает HTTP-сервер, процесс воркера завершается."""
        threading.Thread(target=self.httpd.shutdown, name="idle-shutdown", daemon=True).start()

    def _start_workers(self) -> None:
        self.manager.start()
        self.deadman.start()

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
        self.deadman.stop()
        self.manager.stop()
        if self._serve_thread is not None:
            self.httpd.shutdown()
            self._serve_thread.join(timeout=5)
            self._serve_thread = None
        self.httpd.server_close()


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
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
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
    comfy = ComfyClient(os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    app = WorkerApp(token, comfy, port=port, idle_timeout_seconds=idle_min * 60.0,
                    workflow=workflow, bindings=bindings,
                    output_dir=os.environ.get("COMFY_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
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
