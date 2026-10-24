"""Тесты: логи карт через Vast при снятии без логов воркера, маска текста с карты, онстарт.

Сеть не используется: клиент Vast работает поверх httpx.MockTransport, команды онстарта запускаются
в bash с подставными curl и tar. Канареечные значения — заведомо фальшивые.
"""
import json
import logging
import os
import shutil
import subprocess
import sys

import httpx
import pytest

try:
    from Generate.core import log_setup as ls
    from Generate.core import orchestrator as orch
    from Generate.core import vast_client as vc
except ImportError:
    from core import log_setup as ls  # type: ignore
    from core import orchestrator as orch  # type: ignore
    from core import vast_client as vc  # type: ignore

ENV_KEY_NAME = "GEN_" + "VAST_API_KEY"
CANARY = "TESTKEY-12345"
RESULT_URL = "https://result.example.test/logs/f.txt?X-Amz-Signature=URLCANARY-987654"
URL_CANARY = "URLCANARY-987654"
VALID_BUILD = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv(ENV_KEY_NAME, CANARY)


class FakeLogsVast:
    """Подставка клиента Vast: request_logs и download_logs. codes — коды ответов загрузки по порядку."""

    def __init__(self, codes, body=b"line one\n"):
        self.codes = list(codes)
        self.body = body
        self.requests = []
        self.downloads = []

    def request_logs(self, instance_id, *, tail=2000, daemon_logs=False):
        self.requests.append((instance_id, tail, daemon_logs))
        return RESULT_URL

    def download_logs(self, url, dest):
        self.downloads.append(url)
        code = self.codes.pop(0) if self.codes else 200
        if code == 200:
            with open(dest, "wb") as fh:
                fh.write(self.body)
        return code


def _bare_orc(tmp_path, vast, *, attempts=5, pause=30.0, sleeps=None, token=""):
    """Оркестратор без конструктора: только то, что нужно путям снятия и маске."""
    orc = orch.Orchestrator.__new__(orch.Orchestrator)
    orc._token = token
    orc._card_addresses = set()
    orc.vast = vast
    orc.pilot = False  # генерация: __init__ не вызывается, поэтому флаг задаётся явно
    orc._artifact = orch.RunArtifact(tmp_path / "artifact", orc._mask_remote)
    orc._vast_logs_attempts = attempts
    orc._vast_logs_pause = pause
    orc._sleep = (sleeps.append if sleeps is not None else (lambda sec: None))
    return orc


def _slot(instance_id=77):
    return orch._CardSlot(index=1, generation=1, instance_id=instance_id, machine_id=5,
                          price=0.1, created=0.0)


def _card_file(tmp_path, name="vast_logs.txt"):
    return tmp_path / "artifact" / "cards" / "slot1_generation1" / name


# ---------------------------------------------------------------- клиент: request_logs и download_logs

def _client_with(handler):
    # ключ задаёт фикстура key (через monkeypatch), здесь окружение не меняется
    return vc.VastClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_request_logs_sends_put_with_tail_and_returns_result_url(key, caplog):
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"success": True, "result_url": RESULT_URL})

    caplog.set_level(logging.DEBUG)
    client = _client_with(handler)
    url = client.request_logs(77, tail=2000)
    assert url == RESULT_URL
    assert seen["method"] == "PUT" and seen["path"] == "/api/v0/instances/request_logs/77"
    assert seen["body"] == {"tail": 2000, "daemon_logs": "false"}
    assert seen["auth"] == f"Bearer {CANARY}"
    # адрес result_url не попадает в журнал клиента, даже в отладочном режиме
    assert URL_CANARY not in caplog.text and RESULT_URL not in caplog.text


def test_request_logs_daemon_flag_is_string_true(key):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"result_url": RESULT_URL})

    _client_with(handler).request_logs(9, tail=5, daemon_logs=True)
    assert seen["body"] == {"tail": 5, "daemon_logs": "true"}


def test_request_logs_error_carries_status_without_body_in_attr(key):
    def handler(request):
        return httpx.Response(503, text="busy")

    with pytest.raises(vc.VastStatusError) as info:
        _client_with(handler).request_logs(9)
    assert info.value.status == 503


def test_download_logs_access_denied_then_ok_and_no_auth_header(key, tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(403, text="<Error><Code>AccessDenied</Code></Error>")
        return httpx.Response(200, content=b"container output\n")

    client = _client_with(handler)
    dest = tmp_path / "raw.txt"
    assert client.download_logs(RESULT_URL, str(dest)) == 403
    assert not dest.exists()                       # при отказе файл не пишется
    assert client.download_logs(RESULT_URL, str(dest)) == 200
    assert dest.read_bytes() == b"container output\n"
    assert all("authorization" not in {k.lower() for k in r.headers} for r in calls)


# ---------------------------------------------------------------- снятие без адреса: повторы и файл

def test_retire_without_address_fetches_once_and_writes_artifact(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    caplog.set_level(logging.INFO)
    vast = FakeLogsVast(codes=[200], body="boot ok\n[onstart] итог: код 0\n".encode("utf-8"))
    orc = _bare_orc(tmp_path, vast)
    orc._dump_before_destroy(_slot())
    assert len(vast.requests) == 1 and len(vast.downloads) == 1
    assert vast.requests[0] == (77, orch.VAST_LOGS_TAIL_LINES, False)
    text = _card_file(tmp_path).read_text(encoding="utf-8")
    assert "[onstart] итог: код 0" in text
    assert RESULT_URL not in caplog.text and URL_CANARY not in caplog.text
    assert "логи Vast получены" in caplog.text and "код 200" in caplog.text


def test_access_denied_is_retried_with_pause_and_logs_each_code(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    caplog.set_level(logging.INFO)
    sleeps = []
    vast = FakeLogsVast(codes=[403, 403, 200])
    orc = _bare_orc(tmp_path, vast, attempts=5, pause=30.0, sleeps=sleeps)
    assert orc._fetch_vast_logs(_slot()) is True
    assert len(vast.downloads) == 3                 # две ошибки доступа, потом успех
    # пауза между попытками: 30 с на каждую (две паузы), кусками по 10 с (F2); после успеха пауз нет
    assert sleeps == [10.0] * 6 and sum(sleeps) == 60.0
    assert "попытка 1 из 5, код 403" in caplog.text and "попытка 2 из 5, код 403" in caplog.text
    assert _card_file(tmp_path).exists()
    assert RESULT_URL not in caplog.text and URL_CANARY not in caplog.text


def test_gives_up_after_configured_attempts(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    caplog.set_level(logging.WARNING)
    sleeps = []
    vast = FakeLogsVast(codes=[500, 500, 500])
    orc = _bare_orc(tmp_path, vast, attempts=3, pause=7.5, sleeps=sleeps)
    assert orc._fetch_vast_logs(_slot()) is False
    assert len(vast.requests) == 3 and len(vast.downloads) == 3
    assert sleeps == [7.5, 7.5]
    assert "не получены после 3 попыток" in caplog.text
    assert not _card_file(tmp_path).exists()


def test_unexpected_error_is_swallowed_without_retry(tmp_path):
    class Broken:
        def request_logs(self, *a, **k):
            raise AttributeError("no such method")

    sleeps = []
    orc = _bare_orc(tmp_path, Broken(), sleeps=sleeps)
    assert orc._fetch_vast_logs(_slot()) is False
    assert sleeps == []                             # не повтор: сбой кода, а не Vast


# ---------------------------------------------------------------- маска: каждое правило с канарейкой

MASK_CASES = [
    ("worker token", f"token {CANARY} in line", {"token": CANARY}),
    ("HF_TOKEN", f"hf value {CANARY} here", {"hf": CANARY}),
    ("api key pattern", "key sk-TESTKEY12345abc in line", {}),
    ("bearer", f"Authorization: Bearer {CANARY}", {}),
    ("ipv4", "addr 203.0.113.77 ok", {}),
    ("host:port", "relay relay.example.test:47123 ok", {}),
    ("ip:port", "addr 203.0.113.77:47123 ok", {}),
    ("signed url", f"GET https://s3.example.test/f.bin?X-Amz-Signature={CANARY} done", {}),
]


@pytest.mark.parametrize(("label", "line", "extra"), MASK_CASES, ids=[c[0] for c in MASK_CASES])
def test_each_masking_rule_removes_canary_from_artifact_file(tmp_path, monkeypatch, label, line, extra):
    monkeypatch.setenv("HF_TOKEN", CANARY)          # HF_TOKEN задан: его значение маскирует оркестратор
    token = CANARY if "token" in extra else ""
    body = (line + "\n").encode("utf-8")
    vast = FakeLogsVast(codes=[200], body=body)
    orc = _bare_orc(tmp_path, vast, token=token)
    assert orc._fetch_vast_logs(_slot()) is True
    out = _card_file(tmp_path).read_text(encoding="utf-8")
    for needle in (CANARY, "203.0.113.77", "relay.example.test", "sk-TESTKEY12345abc", "X-Amz-Signature"):
        if needle in line:
            assert needle not in out, f"{label}: осталось {needle!r}"
    assert "***" in out


def test_mask_fetched_text_keeps_plain_status_text():
    sample = "status=running max_tokens=512 time 12:30:45"
    assert ls.mask_fetched_text(sample) == sample


# ---------------------------------------------------------------- ограничение строк

def test_long_log_keeps_first_200_and_last_2000_with_marker():
    lines = [f"row {i}" for i in range(2500)]
    out = orch.limit_log_lines("\n".join(lines) + "\n").splitlines()
    assert out[:200] == lines[:200]
    assert out[200] == "[пропущено строк в середине: 300]"
    assert out[201:] == lines[-2000:]


def test_short_log_is_unchanged():
    text = "a\nb\nc\n"
    assert orch.limit_log_lines(text) == text


# ---------------------------------------------------------------- онстарт: файл и stdout

def _run_onstart(tmp_path, curl_exit=0):
    ws = tmp_path / "ws"
    src = tmp_path / "src"
    bin_dir = tmp_path / "bin"
    for d in (ws, src / "Generate" / "worker", bin_dir):
        d.mkdir(parents=True, exist_ok=True)
    (bin_dir / "curl").write_text(
        "#!/bin/bash\nout=''\nwhile [ $# -gt 0 ]; do if [ \"$1\" = '-o' ]; then out=\"$2\"; shift; fi; shift; done\n"
        f"echo fake > \"$out\"\nexit {curl_exit}\n", encoding="utf-8")
    (bin_dir / "tar").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (src / "Generate" / "worker" / "bootstrap.sh").write_text("#!/bin/bash\necho stub\n", encoding="utf-8")
    for name in ("curl", "tar"):
        os.chmod(bin_dir / name, 0o755)
    cmd = orch._worker_onstart(VALID_BUILD)
    cmd = (cmd.replace(orch.WORKER_BOOTSTRAP_LOG, str(ws / "bootstrap.log"))
              .replace(orch.WORKER_SRC_DIR, str(src))
              .replace("/tmp/worker-src.tar.gz", str(ws / "src.tgz")))
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
               WORKER_TOKEN=CANARY, HF_TOKEN=CANARY)
    proc = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True, timeout=60)
    log_text = (ws / "bootstrap.log").read_text(encoding="utf-8")
    return proc, log_text


@pytest.mark.skipif(shutil.which("bash") is None, reason="нужен bash")
def test_onstart_step_lines_reach_both_file_and_stdout(tmp_path):
    proc, log_text = _run_onstart(tmp_path)
    assert proc.returncode == 0
    step_lines = [line for line in log_text.splitlines() if line.startswith("[onstart]")]
    assert len(step_lines) == 6
    for line in step_lines:
        assert line in proc.stdout.splitlines(), line
    # ни архив, ни секреты не выводятся
    combined = proc.stdout + log_text + proc.stderr
    assert "codeload.github.com" not in combined and CANARY not in combined


@pytest.mark.skipif(shutil.which("bash") is None, reason="нужен bash")
def test_onstart_failure_keeps_exit_code_and_reports_step(tmp_path):
    proc, log_text = _run_onstart(tmp_path, curl_exit=22)
    assert proc.returncode != 0
    assert "[onstart] шаг curl: код 22" in proc.stdout
    assert "[onstart] итог: код 22" in log_text
    assert "bootstrap" not in log_text.split("[onstart] шаг curl")[1]   # после сбоя шаги пропущены
