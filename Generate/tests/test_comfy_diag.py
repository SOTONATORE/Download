"""Офлайн-тесты диагностики шага 4б bootstrap.sh: проверка жизни ComfyUI и хвост его лога.

Без сети и без ComfyUI: COMFY_PYTHON подменяется заглушкой, как в test_worker_bootstrap.py.
Значения COMFYUI_ARGS, HF_TOKEN и WORKER_TOKEN не должны попасть в вывод bootstrap.
"""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from test_worker_bootstrap import CANARY, _run_bootstrap, _write_exec

HF_CANARY = "HFKEY-67890"
ARGS_MARK = "--enable-cors-header"  # часть COMFYUI_ARGS из _run_bootstrap, в выводе быть не должна


def _stub_python(tmp_path: Path, body: str) -> str:
    """Заглушка COMFY_PYTHON с произвольным телом shell-скрипта."""
    stub = tmp_path / "comfy_diag_stub"
    _write_exec(stub, "#!/bin/sh\n" + body)
    return str(stub)


def _comfy_lines(text: str) -> list:
    return [line for line in text.splitlines() if "[comfy] " in line]


def test_exit_at_once_reports_not_alive_and_log_tail(tmp_path):
    """Заглушка сразу падает: в лог бутстрапа попадают сообщение о сбое и строка из comfyui.log."""
    stub = _stub_python(tmp_path, "echo COMFY_DIAG_MARKER_LINE\nexit 3\n")
    proc = _run_bootstrap(tmp_path, extra={"COMFY_PYTHON": stub})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    bootstrap_log = (tmp_path / "bootstrap.log").read_text(encoding="utf-8")
    assert "не запущен через" in bootstrap_log
    assert any("COMFY_DIAG_MARKER_LINE" in line and "[comfy] " in line
               for line in bootstrap_log.splitlines())


def test_alive_process_reports_alive_and_no_failure(tmp_path):
    """Заглушка живёт (sleep 60): пишется «жив через», и «не запущен через» нет."""
    stub = _stub_python(tmp_path, "exec sleep 60\n")
    try:
        proc = _run_bootstrap(tmp_path, extra={"COMFY_PYTHON": stub})
        assert proc.returncode == 0, proc.stdout + proc.stderr
        bootstrap_log = (tmp_path / "bootstrap.log").read_text(encoding="utf-8")
        assert "жив через" in bootstrap_log
        assert "не запущен через" not in bootstrap_log
    finally:
        pid_file = tmp_path / "comfyui.pid"
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text().strip()), signal.SIGKILL)
            except (ValueError, ProcessLookupError, PermissionError):
                pass


def test_check_sec_one_keeps_the_test_fast(tmp_path):
    """COMFY_ALIVE_CHECK_SEC=1 не даёт ждать 20 секунд по умолчанию."""
    stub = _stub_python(tmp_path, "exit 0\n")
    started = time.monotonic()
    proc = _run_bootstrap(tmp_path, extra={"COMFY_PYTHON": stub, "COMFY_ALIVE_CHECK_SEC": "1"})
    elapsed = time.monotonic() - started
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert elapsed < 15, f"проверка заняла {elapsed:.1f} с"


def test_tail_of_sixty_lines_gives_exactly_forty_comfy_lines(tmp_path):
    """В comfyui.log 60 строк: в лог бутстрапа попадают ровно 40 строк с «[comfy] »."""
    lines = "; ".join(f"echo line-{n}" for n in range(1, 61))
    stub = _stub_python(tmp_path, f"{lines}\nexit 1\n")
    proc = _run_bootstrap(tmp_path, extra={"COMFY_PYTHON": stub})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    bootstrap_log = (tmp_path / "bootstrap.log").read_text(encoding="utf-8")
    comfy = _comfy_lines(bootstrap_log)
    assert len(comfy) == 40, len(comfy)
    assert any(line.endswith("line-60") for line in comfy)
    assert any(line.endswith("line-21") for line in comfy)
    assert not any(line.endswith("line-20") for line in comfy)


def test_no_args_or_token_values_in_output(tmp_path):
    """Значения COMFYUI_ARGS, WORKER_TOKEN и HF_TOKEN не печатаются ни в stdout, ни в bootstrap.log."""
    stub = _stub_python(tmp_path, "echo ordinary comfy line\nexit 2\n")
    proc = _run_bootstrap(tmp_path, extra={"COMFY_PYTHON": stub, "HF_TOKEN": HF_CANARY})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    bootstrap_log = (tmp_path / "bootstrap.log").read_text(encoding="utf-8")
    for text in (proc.stdout + proc.stderr, bootstrap_log):
        assert CANARY not in text
        assert HF_CANARY not in text
        assert ARGS_MARK not in text
