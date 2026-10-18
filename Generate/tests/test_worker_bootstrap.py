"""Офлайн-тесты bootstrap.sh (onstart, вариант A, SPEC 8.1) и правок setup.sh (SPEC 0.1, п. 3).

Без сети и без ключей: окружение процесса собирается с нуля, инструменты подменяются
заглушками в отдельной папке PATH. Канареечный ключ TESTKEY-12345 не должен попасть
ни в stdout, ни в stderr, ни в аргументы процессов.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parents[1] / "worker"
BOOTSTRAP = WORKER_DIR / "bootstrap.sh"
SETUP = WORKER_DIR / "setup.sh"
CONFIG = Path(__file__).resolve().parents[1] / "config.default.yaml"
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
    }
    env.update(extra or {})
    for key in drop:
        env.pop(key, None)
    return subprocess.run(["/bin/bash", str(BOOTSTRAP)], env=env, capture_output=True,
                          text=True, timeout=60, cwd=str(tmp_path))


def _output(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout + proc.stderr


# --- структура файла ---

def test_bootstrap_is_valid_bash():
    proc = subprocess.run(["bash", "-n", str(BOOTSTRAP)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_bootstrap_has_no_env_dump_or_command_trace():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    forbidden = re.compile(r"\benv\b|printenv|export\s+-p|declare\s+-p|set\s+-[a-zA-Z]*x|xtrace|/proc/self/environ")
    assert forbidden.search(text) is None


def test_bootstrap_contains_no_canary_value():
    """В файле нет значений ключей: только имена переменных."""
    assert CANARY not in BOOTSTRAP.read_text(encoding="utf-8")


# --- шаги и коды ---

def test_success_logs_each_step_and_hands_over(tmp_path):
    proc = _run_bootstrap(tmp_path)
    assert proc.returncode == 0, _output(proc)
    out = proc.stdout
    positions = [out.find(f"[bootstrap] шаг {n}:") for n in range(1, 6)]
    assert all(p >= 0 for p in positions), out
    assert positions == sorted(positions)
    assert "(код 0)" in out
    assert "FAKE_SETUP_STARTED" in out  # управление передано setup.sh


def test_success_passes_env_to_setup(tmp_path):
    proc = _run_bootstrap(tmp_path)
    assert proc.returncode == 0, _output(proc)
    lines = (tmp_path / "setup_ran.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "ran"
    assert f"pythonpath={tmp_path}" in lines  # родитель папки Generate: импорт Generate.worker
    assert f"comfy={tmp_path / 'ComfyUI'}" in lines
    assert "token=yes" in lines and "build=yes" in lines


def test_missing_comfy_dir_fails_with_path_and_no_setup(tmp_path):
    proc = _run_bootstrap(tmp_path, comfy_exists=False)
    assert proc.returncode != 0
    assert proc.returncode == 5
    text = _output(proc)
    assert str(tmp_path / "ComfyUI") in text
    assert "ComfyUI" in text and "COMFY_DIR" in text
    assert not (tmp_path / "setup_ran.txt").exists()


def test_existing_repo_without_setup_is_refused_not_touched(tmp_path):
    repo = tmp_path / "Generate"
    (repo / "junk").mkdir(parents=True)
    (repo / "junk" / "keep.txt").write_text("x", encoding="utf-8")
    proc = _run_bootstrap(tmp_path, repo_has_setup=False)
    assert proc.returncode == 3
    assert (repo / "junk" / "keep.txt").exists()


# --- секреты и проверка WORKER_BUILD ---

@pytest.mark.parametrize("build", ["", "abc", "UNKNOWN", "0123456789ABCDEF", "g123456", "a" * 65])
def test_invalid_build_refused_without_printing_it(tmp_path, build):
    proc = _run_bootstrap(tmp_path, extra={"WORKER_BUILD": build})
    assert proc.returncode == 2
    assert "WORKER_BUILD" in _output(proc)
    if build:  # значение не печатается (даже неверное)
        assert build not in _output(proc)


def test_missing_token_refused(tmp_path):
    proc = _run_bootstrap(tmp_path, drop=("WORKER_TOKEN",))
    assert proc.returncode == 2
    assert "WORKER_TOKEN" in _output(proc)


@pytest.mark.parametrize("kind", ["success", "no_comfy", "bad_build", "no_setup_repo"])
def test_token_never_in_bootstrap_output(tmp_path, kind):
    kwargs = {
        "success": {},
        "no_comfy": {"comfy_exists": False},
        "bad_build": {"extra": {"WORKER_BUILD": "XYZ"}},
        "no_setup_repo": {"repo_has_setup": False},
    }[kind]
    proc = _run_bootstrap(tmp_path, **kwargs)
    assert CANARY not in proc.stdout
    assert CANARY not in proc.stderr


# --- setup.sh: токен HF не попадает в аргументы процессов (SPEC 0.1, п. 3) ---

def _setup_env(tmp_path: Path, bin_dir: Path, comfy: Path, *, extra: dict | None = None) -> dict:
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "HF_TOKEN": CANARY,
        "COMFY_DIR": str(comfy),
        "COMFY_LOG_PATH": str(tmp_path / "comfy.log"),
        "WORKER_READY_FLAG": str(tmp_path / "worker_ready"),
    }
    env.update(extra or {})
    return env


def _stub_tools(tmp_path: Path, *, curl_rc: int = 0) -> tuple:
    """Заглушки aria2c (создаёт файлы моделей), curl (ответ ComfyUI), python (сервер воркера сразу завершается)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    argv_log = tmp_path / "argv.txt"
    curl_log = tmp_path / "curl_args.txt"
    _write_exec(bin_dir / "aria2c", f"""#!{sys.executable}
import os, sys
with open({str(argv_log)!r}, "a", encoding="utf-8") as fh:
    fh.write(" ".join(sys.argv) + "\\n")
args = sys.argv[1:]
src = args[args.index("-i") + 1]
lines = [l for l in open(src, encoding="utf-8").read().splitlines() if l.strip()]
opts = dict(l.strip().split("=", 1) for l in lines[1:])
with open({str(argv_log)!r}, "a", encoding="utf-8") as fh:
    fh.write("header_seen=" + ("yes" if opts.get("header") == "Authorization: Bearer {CANARY}" else "no") + "\\n")
os.makedirs(opts["dir"], exist_ok=True)
with open(os.path.join(opts["dir"], opts["out"]), "wb") as fh:
    fh.write(b"0123456789abcdef")
""")
    _write_exec(bin_dir / "ffmpeg", "#!/bin/sh\nexit 0\n")
    _write_exec(bin_dir / "curl", f"#!/bin/sh\necho \"$@\" >> '{curl_log}'\nexit {curl_rc}\n")
    _write_exec(bin_dir / "python", "#!/bin/sh\nexit 0\n")  # сервер воркера-заглушка: завершается сразу
    return bin_dir, argv_log, curl_log


def test_setup_keeps_hf_token_out_of_argv_and_marks_ready(tmp_path):
    try:
        import httpx  # noqa: F401
    except ImportError:
        pytest.skip("нужен httpx")
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    bin_dir, argv_log, curl_log = _stub_tools(tmp_path, curl_rc=0)
    env = _setup_env(tmp_path, bin_dir, comfy)
    proc = subprocess.run(["/bin/bash", str(SETUP)], env=env, capture_output=True, text=True,
                          timeout=120, cwd=str(tmp_path))
    argv = argv_log.read_text(encoding="utf-8")
    assert CANARY not in argv
    assert "header_seen=yes" in argv
    assert (comfy / "models" / "diffusion_models").is_dir()  # COMFY_DIR учтён в setup.sh
    assert CANARY not in _output(proc)
    # готовность проверяется по 18188, и только после неё создаётся файл готовности (SPEC 8.1, 8.5)
    curl_args = curl_log.read_text(encoding="utf-8")
    assert "http://127.0.0.1:18188/system_stats" in curl_args and ":8188" not in curl_args
    assert (tmp_path / "worker_ready").exists()
    assert proc.returncode == 0, _output(proc)


def test_setup_readiness_timeout_exits_nonzero_in_russian_and_no_flag(tmp_path):
    """Таймаут ожидания ComfyUI: код не 0, русская строка ошибки, файла готовности нет.
    Копия setup.sh с нулевым таймаутом: рабочий файл не меняется, пауза не нужна."""
    try:
        import httpx  # noqa: F401
    except ImportError:
        pytest.skip("нужен httpx")
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    bin_dir, _argv, _curl = _stub_tools(tmp_path, curl_rc=22)  # curl: ComfyUI не отвечает
    quick = tmp_path / "setup_quick.sh"
    quick.write_text(SETUP.read_text(encoding="utf-8").replace(
        "COMFY_TIMEOUT_SEC=180", "COMFY_TIMEOUT_SEC=0", 1), encoding="utf-8")
    env = _setup_env(tmp_path, bin_dir, comfy)
    proc = subprocess.run(["/bin/bash", str(quick)], env=env, capture_output=True, text=True,
                          timeout=120, cwd=str(tmp_path))
    assert proc.returncode != 0
    assert "ОШИБКА: ComfyUI на порту 18188 не стал готов за 0 сек." in _output(proc)
    assert not (tmp_path / "worker_ready").exists()


# --- setup.sh: структура (статические проверки, SPEC 8.1, 8.5) ---

def _setup_code() -> str:
    """Текст setup.sh без комментариев (всё после «#» в строке)."""
    return "\n".join(line.split("#", 1)[0] for line in SETUP.read_text(encoding="utf-8").splitlines())


def test_setup_does_not_start_comfyui_on_8188():
    code = _setup_code()
    assert "8188" not in code
    assert "main.py" not in code
    assert "--listen" not in code
    assert "18188/system_stats" in code


def test_setup_starts_server_before_weights_download():
    text = SETUP.read_text(encoding="utf-8")
    server = text.index("worker/server.py")
    first_download = text.index('download_model "diffusion_models"')
    assert server < first_download


def test_setup_logs_weights_presence_before_readiness_wait():
    text = SETUP.read_text(encoding="utf-8")
    presence = text.index("Веса перед ожиданием ComfyUI")
    wait = text.index("Ожидаю готовности ComfyUI на порту 18188")
    assert presence < wait


def test_setup_ready_flag_is_cleared_first_and_created_after_readiness():
    text = SETUP.read_text(encoding="utf-8")
    cleared = text.index('rm -f "${WORKER_READY_FLAG}"')
    touched = text.index('touch "${WORKER_READY_FLAG}"')
    readiness_ok = text.index("Проверка готовности ComfyUI (${COMFY_READY_URL}): пройдена")
    assert cleared < touched and readiness_ok < touched


def test_setup_has_no_leftover_comfy_pid():
    code = _setup_code()
    assert "COMFY_PID" not in code

# --- конфиг: образ стокового Vast (вариант A) ---

def test_config_docker_image_is_stock_vast_tag():
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert data["docker_image"] == "vastai/comfy:v0.41.0-cuda-13.2-py312"
