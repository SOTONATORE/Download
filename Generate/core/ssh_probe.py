"""Проверка арендованной карты Vast по SSH: одна фиксированная команда (SPEC 0.1).

Модуль запускает только команды из словаря COMMANDS. Любое другое имя отвергается до подключения.
Закрытый ключ приходит текстом (значение переменной VAST_SSH_KEY, которое передаёт вызывающий код),
пишется во временный файл с правами 0600 и удаляется в finally даже при ошибке.
Адрес и порт передаёт вызывающий код; модуль их сам не ищет.

Вывод и текст ошибок ssh проходят через mask_fetched_text из log_setup, плюс литеральная замена
адреса, порта, пути к ключу и текста ключа. Адрес, порт, ключ и строка команды ssh не попадают
в ProbeResult, в исключения и в журнал. Сам модуль ничего не пишет в журнал: решение за вызывающим кодом.
Сообщения об ошибках на русском (SPEC 0.5) и не содержат значений.
"""
from __future__ import annotations

import contextlib
import os
import re
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

try:  # пакетный и «плоский» импорт, как в cli.py
    from Generate.core.log_setup import mask_fetched_text
except ImportError:  # pragma: no cover
    from log_setup import mask_fetched_text  # type: ignore

# Фиксированные команды. Имя -> точная строка команды на машине. Добавлять только явным решением владельца.
COMMANDS: dict[str, str] = {
    "ping": "echo ok",
    "bootstrap_tail": "tail -n 100 /workspace/bootstrap.log",
    "workspace_list": "ls -la /workspace",
    "processes": "ps -eo pid,comm --no-headers | head -n 50",
    "listening": "(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | head -n 30",
    "worker_health": "curl -sS -m 5 http://127.0.0.1:8000/health",
    "gpu": "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader",
    "comfy_health": "curl -sS -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:18188/system_stats",
    "onstart_log": "cat /opt/pilot/onstart.log",  # копия onstart пилота в артефакт (только пилот)
}

FAILURE_CLASSES = (
    "ok",
    "key_missing",
    "key_rejected",
    "connect_refused",
    "connect_timeout",
    "command_missing",
    "command_failed",
    "unknown",
)

_MASK = "***"
_SSH_EXIT_FAILED = 255  # код, который ssh отдаёт при собственной ошибке (связь, ключ), а не команды


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    returncode: Optional[int]
    output: str  # замаскированный вывод целиком, без обрезки
    failure_class: str  # одно из FAILURE_CLASSES
    output_length: int  # длина output в символах (сообщается вызывающему коду)


def _scrub(text: str, host: str, port: int, key_text: str, key_path: str) -> str:
    """Литеральная замена адреса, порта, пути к ключу и текста ключа, затем общая маска журнала.

    Порт заменяется только как отдельное число (не внутри большего числа), чтобы не портить
    обычный вывод. Общая маска (mask_fetched_text) добавляет IP и пары host:port по форме значения.
    """
    for literal in (key_text.strip(), key_path, host):
        if literal:
            text = text.replace(literal, _MASK)
    text = re.sub(r"(?<!\d)" + re.escape(str(port)) + r"(?!\d)", _MASK, text)
    return mask_fetched_text(text, keep_internal_ports=True)  # 8000 и 18188 видимы (SPEC 0.1, правило портов)


def _classify(returncode: int, stderr_text: str) -> str:
    """Класс отказа по коду выхода ssh и тексту stderr. Вызывается только при ненулевом коде."""
    low = stderr_text.lower()
    if returncode == _SSH_EXIT_FAILED:
        if "permission denied" in low:
            return "key_rejected"
        if "connection refused" in low:
            return "connect_refused"
        if "timed out" in low:
            return "connect_timeout"
        return "unknown"
    if returncode == 127 or "not found" in low:
        return "command_missing"
    return "command_failed"


def _write_key(key_text: str) -> str:
    """Пишет ключ во временный файл с правами 0600 до записи содержимого. Возвращает путь."""
    fd, path = tempfile.mkstemp(prefix="vast_probe_", suffix=".key")
    os.chmod(path, 0o600)
    text = key_text.strip() + "\n"  # OpenSSH требует завершающий перевод строки
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _check_target(host: str, port: int) -> None:
    """Адрес и порт SSH: ValueError до любого подключения. Общая проверка для команд и туннеля."""
    if not isinstance(host, str) or not host.strip() or any(c.isspace() for c in host):
        raise ValueError("Некорректный адрес для проверки.")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("Некорректный порт для проверки.")


def _ssh_options(port: int, key_path: str) -> list[str]:
    """Общие параметры ssh (порт, ключ, опции) для команд и для туннеля. Одно место для опций подключения."""
    # Проверка ключа хостов отключена: адреса Vast меняются при каждой аренде. Риск принят владельцем.
    return [
        "-p", str(port),
        "-i", key_path,
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]


def run_probe(host: str, port: int, key_text: Optional[str], command_name: str,
              timeout_sec: int = 20) -> ProbeResult:
    """Выполняет одну команду из COMMANDS на карте по SSH и возвращает замаскированный результат.

    Порядок проверок: имя команды (ValueError до любого подключения), адрес и порт, наличие ключа
    (key_missing без вызова ssh), затем запуск ssh со списком аргументов и без shell.
    """
    if command_name not in COMMANDS:
        raise ValueError("Неизвестное имя команды проверки.")
    _check_target(host, port)
    if key_text is None or not key_text.strip():
        return ProbeResult(ok=False, returncode=None, output="", failure_class="key_missing", output_length=0)

    key_path: Optional[str] = None
    try:
        key_path = _write_key(key_text)
        args = [
            "ssh",
            *_ssh_options(port, key_path),
            f"root@{host}",
            COMMANDS[command_name],
        ]
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ProbeResult(ok=False, returncode=None, output="", failure_class="connect_timeout", output_length=0)
        except OSError:
            return ProbeResult(ok=False, returncode=None, output="", failure_class="unknown", output_length=0)

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        rc = completed.returncode
        raw = stdout
        if stderr.strip():
            raw = stdout + "\n[stderr]\n" + stderr
        output = _scrub(raw, host, port, key_text, key_path)
        if rc == 0:
            failure_class = "ok"
        else:
            failure_class = _classify(rc, stderr)
        return ProbeResult(
            ok=(rc == 0),
            returncode=rc,
            output=output,
            failure_class=failure_class,
            output_length=len(output),
        )
    finally:
        if key_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(key_path)


# --- туннель к loopback карты (проба ComfyUI пилота, SPEC 8.5) ---------------------------------------
TUNNEL_POLL_SEC = 0.5  # шаг ожидания локального порта туннеля


class TunnelError(Exception):
    """Туннель не открылся. Несёт только класс отказа из FAILURE_CLASSES: без адреса, порта и текста ssh."""

    def __init__(self, failure_class: str) -> None:
        super().__init__(failure_class)
        self.failure_class = failure_class


def _free_local_port() -> int:
    """Свободный порт на 127.0.0.1 (узкое окно гонки допустимо: ssh с ExitOnForwardFailure сам откажет)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _exit_class(proc: subprocess.Popen) -> str:
    """Класс отказа ssh, который завершился до готовности туннеля. Текст stderr не выводится."""
    rc = proc.returncode
    err = proc.stderr.read() if proc.stderr is not None else ""
    if rc in (None, 0):
        return "unknown"
    return _classify(rc, err or "")


def _wait_ready(proc: subprocess.Popen, local_port: int, timeout_sec: float) -> None:
    """Ждёт, пока локальный порт туннеля примет соединение. Ошибка: ssh завершился или истёк срок."""
    deadline = time.monotonic() + timeout_sec
    while True:
        if proc.poll() is not None:
            raise TunnelError(_exit_class(proc))
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=1.0):
                return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise TunnelError("connect_timeout")
        time.sleep(TUNNEL_POLL_SEC)


def _stop_tunnel(proc: subprocess.Popen) -> None:
    """Останавливает ssh: сначала terminate, потом kill. Ошибки остановки не пробрасываются."""
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    if proc.stderr is not None:
        with contextlib.suppress(OSError):
            proc.stderr.close()


@contextlib.contextmanager
def open_tunnel(host: str, port: int, key_text: Optional[str], remote_port: int,
                ready_timeout_sec: float = 30.0):
    """Туннель с локального 127.0.0.1 на 127.0.0.1:<remote_port> карты; отдаёт локальный порт.

    Ключ, адрес и порт SSH — те же, что у run_probe (одни и те же параметры ssh, _ssh_options). Дополнительно:
    -N, -L, ExitOnForwardFailure, ServerAliveInterval 15 и ServerAliveCountMax 3. Ключ пишется во временный файл
    и удаляется, ssh останавливается при выходе из блока, в том числе при ошибке. Порт ожидается не дольше
    ready_timeout_sec. При сбое поднимается TunnelError с классом отказа; пустой или неверный адрес дают
    TunnelError("unknown") до запуска ssh (карта может ещё не сообщить адрес SSH).
    """
    try:
        _check_target(host, port)
    except ValueError:
        raise TunnelError("unknown") from None
    if key_text is None or not key_text.strip():
        raise TunnelError("key_missing")
    if isinstance(remote_port, bool) or not isinstance(remote_port, int) or not 1 <= remote_port <= 65535:
        raise ValueError("Некорректный порт для туннеля.")
    key_path: Optional[str] = None
    proc: Optional[subprocess.Popen] = None
    try:
        try:
            key_path = _write_key(key_text)
            local_port = _free_local_port()
        except OSError:
            raise TunnelError("unknown") from None
        args = [
            "ssh",
            "-N",
            *_ssh_options(port, key_path),
            "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            f"root@{host}",
        ]
        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            raise TunnelError("unknown") from None
        _wait_ready(proc, local_port, ready_timeout_sec)
        yield local_port
    finally:
        if proc is not None:
            _stop_tunnel(proc)
        if key_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(key_path)
