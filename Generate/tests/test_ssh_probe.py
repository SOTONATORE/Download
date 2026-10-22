"""Офлайн-тесты SSH-проверки карты (core/ssh_probe.py).

Никакой сети и никакого реального ssh: subprocess.run заменён заглушкой.
Канареечные значения (адрес, порт, ключ) не должны попасть ни в результат, ни в исключение, ни в журнал.
Сообщения на русском (SPEC 0.5).
"""
from __future__ import annotations

import logging
import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from Generate.core import ssh_probe as sp
except ImportError:  # pragma: no cover
    from core import ssh_probe as sp  # type: ignore

CANARY_HOST = "203.0.113.7"
CANARY_PORT = 2222
CANARY_KEY = "CANARY-KEY-12345"
KEY_TEXT = "-----BEGIN OPENSSH PRIVATE KEY-----\n" + CANARY_KEY + "\n-----END OPENSSH PRIVATE KEY-----\n"


def _completed(rc: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["ssh"], returncode=rc, stdout=stdout, stderr=stderr)


def _assert_no_canary(*texts: str) -> None:
    for text in texts:
        assert CANARY_HOST not in text
        assert CANARY_KEY not in text
        assert "2222" not in text


# ---------------------------------------------------------------------------
# Имя команды и аргументы ssh
# ---------------------------------------------------------------------------

def test_unknown_command_raises_before_any_subprocess():
    with patch.object(sp.subprocess, "run") as run:
        with pytest.raises(ValueError) as exc:
            sp.run_probe(CANARY_HOST, CANARY_PORT, KEY_TEXT, "rm_everything")
    run.assert_not_called()
    _assert_no_canary(str(exc.value))


def test_argument_list_has_no_shell_and_correct_target():
    with patch.object(sp.subprocess, "run", return_value=_completed(0, "ok\n")) as run:
        sp.run_probe(CANARY_HOST, CANARY_PORT, KEY_TEXT, "ping")
    run.assert_called_once()
    args, kwargs = run.call_args
    argv = args[0]
    assert isinstance(argv, list)
    assert kwargs.get("shell", False) is False
    assert argv[0] == "ssh"
    assert argv[argv.index("-p") + 1] == str(CANARY_PORT)
    assert "root@" + CANARY_HOST in argv
    assert argv[-2] == "root@" + CANARY_HOST
    assert argv[-1] == sp.COMMANDS["ping"]
    assert kwargs.get("timeout") == 20
    for opt in ("BatchMode=yes", "ConnectTimeout=10", "IdentitiesOnly=yes",
                "StrictHostKeyChecking=no", "UserKnownHostsFile=/dev/null", "LogLevel=ERROR"):
        assert opt in argv


def test_timeout_argument_is_passed_through():
    with patch.object(sp.subprocess, "run", return_value=_completed(0)) as run:
        sp.run_probe(CANARY_HOST, CANARY_PORT, KEY_TEXT, "ping", timeout_sec=7)
    assert run.call_args.kwargs["timeout"] == 7


# ---------------------------------------------------------------------------
# Файл ключа
# ---------------------------------------------------------------------------

def test_key_file_is_0600_during_call_and_removed_after_success():
    seen = {}

    def fake_run(args, **kwargs):
        path = args[args.index("-i") + 1]
        seen["path"] = path
        seen["exists"] = os.path.exists(path)
        seen["mode"] = stat.S_IMODE(os.stat(path).st_mode)
        with open(path, encoding="utf-8") as fh:
            seen["content"] = fh.read()
        return _completed(0, "ok\n")

    with patch.object(sp.subprocess, "run", side_effect=fake_run):
        result = sp.run_probe(CANARY_HOST, CANARY_PORT, KEY_TEXT, "ping")

    assert result.ok is True
    assert seen["exists"] is True
    assert seen["mode"] == 0o600
    assert CANARY_KEY in seen["content"]
    assert not os.path.exists(seen["path"])


def test_key_file_removed_after_command_failure():
    seen = {}

    def fake_run(args, **kwargs):
        seen["path"] = args[args.index("-i") + 1]
        return _completed(1, "", "boom")

    with patch.object(sp.subprocess, "run", side_effect=fake_run):
        result = sp.run_probe(CANARY_HOST, CANARY_PORT, KEY_TEXT, "ping")

    assert result.failure_class == "command_failed"
    assert not os.path.exists(seen["path"])


def test_key_file_removed_after_timeout():
    seen = {}

    def fake_run(args, **kwargs):
        seen["path"] = args[args.index("-i") + 1]
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])

    with patch.object(sp.subprocess, "run", side_effect=fake_run):
        result = sp.run_probe(CANARY_HOST, CANARY_PORT, KEY_TEXT, "ping")

    assert result.failure_class == "connect_timeout"
    assert result.returncode is None
    assert not os.path.exists(seen["path"])


# ---------------------------------------------------------------------------
# Ключ отсутствует
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [None, "", "   \n"])
def test_missing_key_returns_key_missing_without_subprocess(key):
    with patch.object(sp.subprocess, "run") as run:
        result = sp.run_probe(CANARY_HOST, CANARY_PORT, key, "ping")
    run.assert_not_called()
    assert result.ok is False
    assert result.failure_class == "key_missing"


# ---------------------------------------------------------------------------
# Классы отказов
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rc, stderr, expected", [
    (0, "", "ok"),
    (255, "root@host: Permission denied (publickey).", "key_rejected"),
    (255, "ssh: connect to host 203.0.113.7 port 2222: Connection refused", "connect_refused"),
    (255, "ssh: connect to host 203.0.113.7 port 2222: Connection timed out", "connect_timeout"),
    (127, "", "command_missing"),
    (1, "bash: nvidia-smi: command not found", "command_missing"),
    (2, "", "command_failed"),
    (255, "Could not resolve something odd", "unknown"),
])
def test_failure_class_mapping(rc, stderr, expected):
    with patch.object(sp.subprocess, "run", return_value=_completed(rc, "", stderr)):
        result = sp.run_probe(CANARY_HOST, CANARY_PORT, KEY_TEXT, "ping")
    assert result.failure_class == expected
    assert result.failure_class in sp.FAILURE_CLASSES
    assert result.ok is (rc == 0)
    assert result.returncode == rc


def test_timeout_expired_maps_to_connect_timeout_and_no_output():
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])

    with patch.object(sp.subprocess, "run", side_effect=fake_run):
        result = sp.run_probe(CANARY_HOST, CANARY_PORT, KEY_TEXT, "ping")
    assert result.failure_class == "connect_timeout"
    assert result.output == ""
    assert result.output_length == 0


def test_long_output_is_kept_whole_and_length_reported():
    body = "line\n" * 2000
    with patch.object(sp.subprocess, "run", return_value=_completed(0, body)):
        result = sp.run_probe(CANARY_HOST, CANARY_PORT, KEY_TEXT, "bootstrap_tail")
    assert result.output == body
    assert result.output_length == len(body)


# ---------------------------------------------------------------------------
# Канарейки не попадают никуда
# ---------------------------------------------------------------------------

def test_canaries_absent_from_result_when_ssh_stderr_echoes_them(caplog):
    echoed = (
        f"ssh: connect to host {CANARY_HOST} port {CANARY_PORT}: Connection refused\n"
        f"Warning: Permanently added '{CANARY_HOST}' (ED25519) to the list of known hosts.\n"
        f"Load key {CANARY_KEY}: invalid format\n"
    )
    caplog.set_level(logging.DEBUG)
    with patch.object(sp.subprocess, "run", return_value=_completed(255, "", echoed)):
        result = sp.run_probe(CANARY_HOST, CANARY_PORT, CANARY_KEY, "ping")

    assert result.failure_class == "connect_refused"
    _assert_no_canary(result.output, repr(result))
    _assert_no_canary(*(r.getMessage() for r in caplog.records))
    assert len(caplog.records) == 0  # модуль ничего не пишет в журнал


def test_canaries_absent_from_output_of_successful_command():
    stdout = f"ok {CANARY_HOST} {CANARY_PORT} {CANARY_KEY}\n"
    with patch.object(sp.subprocess, "run", return_value=_completed(0, stdout)):
        result = sp.run_probe(CANARY_HOST, CANARY_PORT, CANARY_KEY, "ping")
    _assert_no_canary(result.output, repr(result))


def test_canaries_absent_from_exception_on_bad_input(caplog):
    caplog.set_level(logging.DEBUG)
    with patch.object(sp.subprocess, "run") as run:
        with pytest.raises(ValueError) as exc:
            sp.run_probe(CANARY_HOST, CANARY_PORT, KEY_TEXT, "nope")
    run.assert_not_called()
    _assert_no_canary(str(exc.value))
    _assert_no_canary(*(r.getMessage() for r in caplog.records))


def test_key_path_absent_from_result_when_echoed():
    captured = {}

    def fake_run(args, **kwargs):
        path = args[args.index("-i") + 1]
        captured["path"] = path
        return _completed(255, "", f"Load key \"{path}\": bad permissions")

    with patch.object(sp.subprocess, "run", side_effect=fake_run):
        result = sp.run_probe(CANARY_HOST, CANARY_PORT, KEY_TEXT, "ping")
    assert captured["path"] not in result.output
