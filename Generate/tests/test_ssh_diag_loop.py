"""Тесты: SSH-диагностика карт, не готовых к приёму задач (Orchestrator._pull_ssh_diag).

Оффлайн: часы подставные (числа), run_probe подменён, артефакт во временной папке. Настоящего SSH нет.
Канареечные значения заведомо фальшивые: адрес из диапазона документации и ключ-заглушка.
time.sleep в этом файле запрещён фикстурой: путь диагностики не должен ждать.
"""
import logging
import subprocess
import time

import pytest

try:
    from Generate.core import orchestrator as orch
except ImportError:
    from core import orchestrator as orch  # type: ignore

CANARY_HOST = "203.0.113.7"
CANARY_PORT = 2222
CANARY_KEY = "CANARY-KEY-12345"
BASE = ("ping", "worker_health", "bootstrap_tail")
EXTRA = ("workspace_list", "processes", "listening", "gpu")  # каждый пятый тик
COMFY = "comfy_health"  # каждый тик, последним в блоке
EVERY_TICK = BASE + (COMFY,)
ALL = BASE + EXTRA + (COMFY,)  # тик 5


class FakeProbe:
    """Подставка run_probe: запоминает аргументы, отдаёт чистый вывод; raise_on — имя команды -> исключение."""

    def __init__(self, raise_on=None):
        self.calls = []  # (host, port, key_text, command_name, timeout_sec)
        self.raise_on = raise_on or {}

    def __call__(self, host, port, key_text, command_name, timeout_sec=20):
        self.calls.append((host, port, key_text, command_name, timeout_sec))
        if command_name in self.raise_on:
            raise self.raise_on[command_name]
        return orch.ssh_probe.ProbeResult(ok=True, returncode=0, output=f"out {command_name}\n",
                                          failure_class="ok", output_length=len(command_name) + 5)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VAST_SSH_KEY", CANARY_KEY)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def no_sleep(*args, **kwargs):
        raise AssertionError("time.sleep вызван в пути SSH-диагностики")

    monkeypatch.setattr(time, "sleep", no_sleep)


def _bare_orc(tmp_path):
    """Оркестратор без конструктора: только то, что нужно маске, артефакту и разбору слотов."""
    orc = orch.Orchestrator.__new__(orch.Orchestrator)
    orc._token = ""
    orc._card_addresses = set()
    orc.vast = object()
    orc._artifact = orch.RunArtifact(tmp_path / "artifact", orc._mask_remote)
    orc._slots = {}
    return orc


def _slot(state="booting", host=CANARY_HOST, port=CANARY_PORT):
    slot = orch._CardSlot(index=1, generation=1, instance_id=77, machine_id=5,
                          price=0.1, created=0.0)
    slot.state = state
    slot.ssh_host = host
    slot.ssh_port = port
    return slot


def _diag_file(tmp_path):
    return tmp_path / "artifact" / "cards" / "slot1_generation1" / orch.SSH_DIAG_FILE


def _ssh_messages(caplog):
    return [r.getMessage() for r in caplog.records if "SSH" in r.getMessage()]


def _assert_no_canary(*texts):
    for text in texts:
        assert CANARY_HOST not in text
        assert str(CANARY_PORT) not in text
        assert CANARY_KEY not in text


def _run(orc, monkeypatch, fake, times):
    monkeypatch.setattr(orch.ssh_probe, "run_probe", fake)
    for t in times:
        orc._pull_ssh_diag(t)


def test_ready_card_gets_no_probe_calls(tmp_path, monkeypatch):
    fake = FakeProbe()
    orc = _bare_orc(tmp_path)
    slot = _slot(state="ready")
    orc._slots[1] = slot
    _run(orc, monkeypatch, fake, [0.0, 30.0, 60.0])
    assert fake.calls == []
    assert not _diag_file(tmp_path).exists()


def test_card_without_ssh_host_is_skipped(tmp_path, monkeypatch):
    fake = FakeProbe()
    orc = _bare_orc(tmp_path)
    orc._slots[1] = _slot(host=None, port=None)
    _run(orc, monkeypatch, fake, [0.0, 30.0])
    assert fake.calls == []


def test_cadence_runs_at_zero_and_thirty_only(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeProbe()
    orc = _bare_orc(tmp_path)
    slot = _slot()
    orc._slots[1] = slot
    _run(orc, monkeypatch, fake, [0.0, 10.0, 20.0, 30.0])
    assert slot.ssh_diag_tick == 2
    assert len(fake.calls) == 8  # два тика по четыре проверки
    assert tuple(c[3] for c in fake.calls) == EVERY_TICK * 2


def test_tick_one_runs_base_and_tick_five_runs_all(tmp_path, monkeypatch):
    fake = FakeProbe()
    orc = _bare_orc(tmp_path)
    slot = _slot()
    orc._slots[1] = slot
    _run(orc, monkeypatch, fake, [0.0, 30.0, 60.0, 90.0, 120.0])
    assert slot.ssh_diag_tick == 5
    names = [c[3] for c in fake.calls]
    assert tuple(names[:4]) == EVERY_TICK  # тик 1
    assert tuple(names[16:]) == ALL  # тик 5: базовые, дополнительные и comfy_health
    text = _diag_file(tmp_path).read_text(encoding="utf-8")
    assert "tick 5 ===" in text
    assert "--- gpu (ok) ---" in text


def test_ssh_diag_log_is_appended_not_overwritten(tmp_path, monkeypatch):
    fake = FakeProbe()
    orc = _bare_orc(tmp_path)
    orc._slots[1] = _slot()
    _run(orc, monkeypatch, fake, [0.0, 30.0])
    text = _diag_file(tmp_path).read_text(encoding="utf-8")
    headers = [line for line in text.splitlines() if line.startswith("=== ")]
    assert len(headers) == 2
    assert "tick 1 ===" in headers[0]
    assert "tick 2 ===" in headers[1]
    assert text.count("--- ping (ok) ---") == 2
    assert "out ping" in text


def test_one_main_log_line_per_tick(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeProbe()
    orc = _bare_orc(tmp_path)
    orc._slots[1] = _slot()
    _run(orc, monkeypatch, fake, [0.0, 10.0, 30.0])
    lines = _ssh_messages(caplog)
    assert len(lines) == 2
    assert lines[0] == "карта 1 поколение 1 (экземпляр 77): SSH-диагностика, тик 1, ping=ok"
    assert lines[1] == "карта 1 поколение 1 (экземпляр 77): SSH-диагностика, тик 2, ping=ok"


def test_runtime_error_does_not_stop_loop(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeProbe(raise_on={"ping": RuntimeError(f"boom {CANARY_HOST}:{CANARY_PORT} {CANARY_KEY}")})
    orc = _bare_orc(tmp_path)
    orc._slots[1] = _slot()
    _run(orc, monkeypatch, fake, [0.0])
    assert tuple(c[3] for c in fake.calls) == EVERY_TICK  # ping упал, остальные проверки шли
    messages = _ssh_messages(caplog)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("RuntimeError" in w for w in warnings)
    _assert_no_canary(*messages, *warnings)
    assert "--- ping (exception) ---" in _diag_file(tmp_path).read_text(encoding="utf-8")
    assert "ping=exception" in messages[-1]


def test_empty_key_skips_probes_with_one_line(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("VAST_SSH_KEY", "")
    fake = FakeProbe()
    orc = _bare_orc(tmp_path)
    orc._slots[1] = _slot()
    _run(orc, monkeypatch, fake, [0.0, 10.0])  # второй вызов раньше интервала: строки нет
    assert fake.calls == []
    assert _ssh_messages(caplog) == ["Карта 1 поколение 1 (экземпляр 77): SSH-диагностика пропущена: ключ не задан"]
    assert not _diag_file(tmp_path).exists()


def test_canaries_absent_from_artifact_and_logs(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeProbe()
    orc = _bare_orc(tmp_path)
    orc._slots[1] = _slot()
    _run(orc, monkeypatch, fake, [0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
    diag = _diag_file(tmp_path).read_text(encoding="utf-8")
    messages = [r.getMessage() for r in caplog.records]
    _assert_no_canary(diag, *messages)
    # адрес, порт и ключ доходят до run_probe только аргументами
    assert fake.calls[0][:3] == (CANARY_HOST, CANARY_PORT, CANARY_KEY)


def test_no_sleep_across_full_path_with_errors(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeProbe(raise_on={"worker_health": ValueError("x")})
    orc = _bare_orc(tmp_path)
    orc._slots[1] = _slot()
    _run(orc, monkeypatch, fake, [0.0, 30.0, 60.0, 90.0, 120.0])  # autouse-фикстура поднимет на любой time.sleep
    assert len(fake.calls) == 24  # четыре тика по четыре и пятый по восемь


# ---------------------------------------------------------------------------
# Правило портов (SPEC 0.1): внутренние порты видны, внешние адреса маскируются
# ---------------------------------------------------------------------------

EXT_IP = "198.51.100.44"  # внешний адрес карты (документационный диапазон)
EXT_PORT = "47123"        # внешний HostPort


def _fake_ssh(stdout):
    """Заглушка subprocess.run для настоящего run_probe: ssh не запускается, вывод задан."""
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")
    return fake_run


def test_diag_log_keeps_internal_ports_and_masks_other_ports(tmp_path, monkeypatch):
    stdout = ("LISTEN 0 4096 0.0.0.0:8000 0.0.0.0:*\n"
              "LISTEN 0 4096 0.0.0.0:18188 0.0.0.0:*\n"
              "LISTEN 0 4096 0.0.0.0:41234 0.0.0.0:*\n")
    monkeypatch.setattr(orch.ssh_probe.subprocess, "run", _fake_ssh(stdout))
    orc = _bare_orc(tmp_path)
    orc._slots[1] = _slot()
    orc._pull_ssh_diag(0.0)  # реальный run_probe: маска на пути ssh_probe, без подмены проверок
    text = _diag_file(tmp_path).read_text(encoding="utf-8")
    assert "0.0.0.0:8000" in text
    assert "0.0.0.0:18188" in text
    assert "41234" not in text
    _assert_no_canary(text)


def test_diag_log_masks_external_literals_in_output_and_logs(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    stdout = f"bound to {EXT_IP} on port {EXT_PORT}\n"
    monkeypatch.setattr(orch.ssh_probe.subprocess, "run", _fake_ssh(stdout))
    orc = _bare_orc(tmp_path)
    slot = _slot()
    slot.ext_values = (EXT_IP, EXT_PORT)  # публичный IP и HostPort, как их передаёт _step_slot
    orc._slots[1] = slot
    orc._pull_ssh_diag(0.0)
    text = _diag_file(tmp_path).read_text(encoding="utf-8")
    messages = [r.getMessage() for r in caplog.records]
    for marker in (EXT_IP, EXT_PORT):
        assert marker not in text
        assert all(marker not in m for m in messages)


def test_empty_key_message_once_per_interval(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("VAST_SSH_KEY", "")
    fake = FakeProbe()
    orc = _bare_orc(tmp_path)
    orc._slots[1] = _slot()
    # 0 — строка; 10, 29, 45, 59 — в интервале, строк нет; 30 и 60 — новый интервал, строка
    _run(orc, monkeypatch, fake, [0.0, 10.0, 29.0, 30.0, 45.0, 59.0, 60.0])
    assert fake.calls == []
    assert _ssh_messages(caplog) == ["Карта 1 поколение 1 (экземпляр 77): SSH-диагностика пропущена: ключ не задан"] * 3


def test_empty_key_message_is_per_card(tmp_path, monkeypatch, caplog):
    """P4: ключ не задан — строка на каждую карту (отдельный интервал у каждой), не одна на запуск."""
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("VAST_SSH_KEY", "")
    fake = FakeProbe()
    orc = _bare_orc(tmp_path)
    orc._slots[1] = _slot()
    second = _slot()
    second.index, second.instance_id = 2, 78
    orc._slots[2] = second
    _run(orc, monkeypatch, fake, [0.0, 10.0, 30.0])  # 0 и 30 — по строке на карту; 10 — интервал не прошёл
    assert _ssh_messages(caplog) == [
        "Карта 1 поколение 1 (экземпляр 77): SSH-диагностика пропущена: ключ не задан",
        "Карта 2 поколение 1 (экземпляр 78): SSH-диагностика пропущена: ключ не задан",
    ] * 2


def test_comfy_health_runs_on_tick_one_and_two(tmp_path, monkeypatch):
    fake = FakeProbe()
    orc = _bare_orc(tmp_path)
    slot = _slot()
    orc._slots[1] = slot
    _run(orc, monkeypatch, fake, [0.0, 30.0])
    names = [c[3] for c in fake.calls]
    assert slot.ssh_diag_tick == 2
    assert names == list(EVERY_TICK) * 2
    assert names.count(COMFY) == 2


def test_heavy_commands_run_only_on_tick_five(tmp_path, monkeypatch):
    fake = FakeProbe()
    orc = _bare_orc(tmp_path)
    orc._slots[1] = _slot()
    _run(orc, monkeypatch, fake, [0.0, 30.0, 60.0, 90.0, 120.0])
    names = [c[3] for c in fake.calls]
    assert names[:16] == list(EVERY_TICK) * 4  # тики 1–4: без тяжёлых проверок
    for heavy in EXTRA:
        assert names.count(heavy) == 1  # только на тике 5
    assert names.count(COMFY) == 5  # comfy_health на каждом из пяти тиков
