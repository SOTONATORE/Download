"""Общие помощники офлайн-тестов bootstrap.sh: используются test_worker_bootstrap.py и test_comfy_diag.py.

Тестовые модули не импортируют друг друга (tests — пакет, но тесты его не используют как модуль),
поэтому общий код вынесен сюда. Тела функций перенесены без изменений.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parents[1] / "worker"
BOOTSTRAP = WORKER_DIR / "bootstrap.sh"
CANARY = "TESTKEY-12345"
BUILD = "0123456789abcdef0123456789abcdef01234567"

def _write_exec(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)

def _make_bin(tmp_path: Path) -> Path:
    """Папка с заглушками: aria2c, ffmpeg и python3 (= текущий интерпретатор с httpx)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_exec(bin_dir / "aria2c", "#!/bin/sh\nexit 0\n")
    _write_exec(bin_dir / "ffmpeg", "#!/bin/sh\nexit 0\n")
    py3 = bin_dir / "python3"
    if not py3.exists():
        py3.symlink_to(sys.executable)
    return bin_dir

def _comfy_stub(tmp_path: Path) -> Path:
    """Заглушка COMFY_PYTHON (шаг 4б): пишет свои аргументы в файл и выходит с кодом 0. ComfyUI не нужен."""
    stub = tmp_path / "comfy_python_stub"
    args_file = tmp_path / "comfy_args.txt"
    if not stub.exists():
        _write_exec(stub, "#!/bin/sh\n"
                    f"printf '%s\\n' \"$@\" > '{args_file}'\n"
                    "exit 0\n")
    return stub

def _run_bootstrap(tmp_path: Path, *, comfy_exists: bool = True, repo_has_setup: bool = True,
                   extra: dict | None = None, drop: tuple = ()) -> subprocess.CompletedProcess:
    try:
        import httpx  # noqa: F401
    except ImportError:
        pytest.skip("нужен httpx (requirements-dev.txt)")
    bin_dir = _make_bin(tmp_path)
    repo = tmp_path / "Generate"
    (repo / "worker").mkdir(parents=True, exist_ok=True)
    marker = tmp_path / "setup_ran.txt"
    if repo_has_setup:
        _write_exec(repo / "worker" / "setup.sh",
                    "#!/bin/bash\n"
                    "echo 'FAKE_SETUP_STARTED'\n"
                    f"{{ echo ran; echo pythonpath=$PYTHONPATH; echo comfy=$COMFY_DIR; "
                    "echo token=${WORKER_TOKEN:+yes}; echo build=${WORKER_BUILD:+yes}; "
                    f"}} > '{marker}'\n")
    comfy = tmp_path / "ComfyUI"
    if comfy_exists:
        comfy.mkdir(exist_ok=True)
        (comfy / "main.py").write_text("# заглушка\n", encoding="utf-8")
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "REPO_DIR": str(repo),
        "COMFY_DIR": str(comfy),
        "WORKER_TOKEN": CANARY,
        "WORKER_BUILD": BUILD,
        "COMFY_PYTHON": str(_comfy_stub(tmp_path)),  # шаг 4б: вместо /venv/main/bin/python
        "COMFYUI_ARGS": "--disable-auto-launch --port 18188 --enable-cors-header",
        "WORKER_BOOTSTRAP_LOG": str(tmp_path / "bootstrap.log"),
        "COMFY_LOG_FILE": str(tmp_path / "comfyui.log"),
        "COMFY_PID_FILE": str(tmp_path / "comfyui.pid"),
        "COMFY_ALIVE_CHECK_SEC": "1",  # короткая проверка жизни ComfyUI (шаг 4б), иначе каждый тест ждёт 20 с
    }
    env.update(extra or {})
    for key in drop:
        env.pop(key, None)
    return subprocess.run(["/bin/bash", str(BOOTSTRAP)], env=env, capture_output=True,
                          text=True, timeout=60, cwd=str(tmp_path))
