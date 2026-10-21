"""Тесты вывода bootstrap.sh: каждая строка шага идёт и в stdout, и в файл лога; секреты не печатаются.

Запуск (на GitHub, локально не требуется):
    pytest Generate/tests/test_worker_bootstrap_log.py
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

BOOTSTRAP = Path(__file__).resolve().parents[1] / "worker" / "bootstrap.sh"
CANARY = "TESTKEY-12345"


def _log_function_source() -> str:
    """Вырезает из bootstrap.sh определение функции log(), чтобы проверить её отдельно."""
    text = BOOTSTRAP.read_text(encoding="utf-8")
    m = re.search(r"^log\(\) \{\n.*?^\}\n", text, flags=re.S | re.M)
    assert m, "в bootstrap.sh нет функции log()"
    return m.group(0)


def _run_bash(script: str, env: dict, stdout_target=subprocess.PIPE):
    return subprocess.run(
        ["bash", "-c", script],
        env=env,
        stdout=stdout_target,
        stderr=subprocess.PIPE,
        check=False,
    )


def _base_env(tmp_path: Path) -> dict:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "WORKER_BOOTSTRAP_LOG": str(tmp_path / "bootstrap.log"),
    }


def test_log_writes_line_to_file_and_stdout(tmp_path):
    log_file = tmp_path / "bootstrap.log"
    log_file.write_text("", encoding="utf-8")
    env = _base_env(tmp_path)
    script = (
        "set -u\n"
        f"BOOTSTRAP_LOG={log_file}\n"
        + _log_function_source()
        + "log 'шаг 1: тест'\n"
    )
    res = _run_bash(script, env)

    assert res.returncode == 0, res.stderr
    assert res.stdout.decode("utf-8") == "[bootstrap] шаг 1: тест\n"
    assert log_file.read_text(encoding="utf-8") == "[bootstrap] шаг 1: тест\n"


def test_log_does_not_duplicate_when_stdout_is_the_log_file(tmp_path):
    # Так запускает onstart: nohup bash bootstrap.sh >> лог 2>&1
    log_file = tmp_path / "bootstrap.log"
    log_file.write_text("", encoding="utf-8")
    env = _base_env(tmp_path)
    script = (
        "set -u\n"
        f"BOOTSTRAP_LOG={log_file}\n"
        + _log_function_source()
        + "log 'шаг 2: один раз'\n"
    )
    with open(log_file, "a", encoding="utf-8") as out:
        res = _run_bash(script, env, stdout_target=out)

    assert res.returncode == 0, res.stderr
    assert log_file.read_text(encoding="utf-8") == "[bootstrap] шаг 2: один раз\n"


def test_bootstrap_error_path_prints_steps_and_no_secrets(tmp_path):
    # WORKER_BUILD неверный: скрипт выходит с кодом 2 на шаге 1, сеть не нужна.
    # Значения WORKER_TOKEN и HF_TOKEN заданы канарейкой и не должны появиться ни в stdout, ни в stderr, ни в логе.
    log_file = tmp_path / "bootstrap.log"
    env = _base_env(tmp_path)
    env.update(
        {
            "WORKER_TOKEN": CANARY,
            "HF_TOKEN": CANARY,
            "WORKER_BUILD": "not-a-commit",
            "REPO_DIR": str(tmp_path / "Generate"),
        }
    )
    res = subprocess.run(
        ["bash", str(BOOTSTRAP)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    out = res.stdout.decode("utf-8")
    err = res.stderr.decode("utf-8")
    file_text = log_file.read_text(encoding="utf-8")

    assert res.returncode == 2
    # шаги и диагностика видны в stdout
    assert "[bootstrap] шаг 1: проверка переменных окружения" in out
    assert "[bootstrap] шаг 1: ОШИБКА: WORKER_BUILD" in out
    assert "переменная HF_TOKEN: задана" in out
    assert "переменная WORKER_TOKEN: задана" in out
    # то же самое попадает в файл лога, построчно совпадая со stdout
    assert file_text == out

    # канарейка не печатается нигде
    for name, text in (("stdout", out), ("stderr", err), ("log", file_text)):
        assert CANARY not in text, f"секрет найден в {name}"


def test_bootstrap_has_no_trace_or_token_echo():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert not re.search(r"^\s*set -x", text, flags=re.M), "трассировка set -x печатает секреты"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.search(r"\b(log|say|say_err|printf|echo)\b", stripped):
            assert not re.search(r"\$\{?(WORKER_TOKEN|HF_TOKEN)\b", stripped), (
                f"строка печатает токен: {stripped}"
            )
