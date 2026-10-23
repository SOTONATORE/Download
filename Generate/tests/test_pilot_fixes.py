"""Тесты исправлений пилота: выход при пропаже карты, гейт SSH-диагностики по running, instance.json пилота.

Офлайн: Vast, HTTP и часы — заглушки из test_orchestrator.py (PilotVast, _pilot_orc, make_offer, PILOT_PORTS_OK).
Настоящей SSH-пробы нет: run_probe подменён. Канареечные значения (203.0.113.7, 2222) — только входные данные
фальшивой карты; в ассертах не печатаются, проверяется их отсутствие в артефакте и журнале.
"""
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

try:
    from Generate.core import log_setup as ls
    from Generate.core import orchestrator as orch
    from Generate.core import vast_client as vc
    from Generate.tests.test_orchestrator import PILOT_PORTS_OK, PilotVast, _pilot_orc, make_offer
except ImportError:  # pragma: no cover
    _HERE = Path(__file__).resolve()
    for _p in (_HERE.parents[2], _HERE.parents[1]):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    from core import log_setup as ls  # type: ignore
    from core import orchestrator as orch  # type: ignore
    from core import vast_client as vc  # type: ignore
    from tests.test_orchestrator import PILOT_PORTS_OK, PilotVast, _pilot_orc, make_offer  # type: ignore

CANARY_HOST = "203.0.113.7"
CANARY_PORT = 2222
CANARY_KEY = "CANARY-KEY-TEST-ONLY-NOT-A-KEY"
# четыре опроса «loading», затем «running»
FAKE_STEPS = ("loading", "loading", "loading", "loading", "running")
SILENT_SEC = 60  # silent_host_timeout_min=1 в _pilot_orc


def _offline_probe(host, port, key_text, command_name, timeout_sec=None):
    return orch.ssh_probe.ProbeResult(ok=True, returncode=0, output="ok", failure_class="ok", output_length=2)


class _GoneVast(PilotVast):
    """Первые missing обращений к статусу бросают VastInstanceNotFound; дальше карта отвечает как в PilotVast."""

    def __init__(self, offers, events, *, missing: int, **kw) -> None:
        super().__init__(offers, events, **kw)
        self.missing = missing
        self.calls = 0

    def get_instance(self, instance_id):
        self.calls += 1
        if self.calls <= self.missing:
            raise vc.VastInstanceNotFound("экземпляр не найден")
        return super().get_instance(instance_id)


class _SshVast(PilotVast):
    """Как PilotVast, но у карты есть ssh_host и ssh_port (нужны для выбора карты в диагностике)."""

    def get_instance(self, instance_id):
        info = super().get_instance(instance_id)
        info.ssh_host = CANARY_HOST
        info.ssh_port = CANARY_PORT
        return info


def _vast_raw(instance_id: int, status: str) -> dict:
    """Ответ Vast со всеми полями, которых не должно быть в instance.json пилота, плюс поля статуса."""
    return {
        "id": instance_id, "label": "pilot-test", "is_bid": True, "dph_total": 0.3,
        "actual_status": status, "intended_status": "running", "status_msg": "",
        "public_ipaddr": CANARY_HOST, "ssh_host": CANARY_HOST, "ssh_port": CANARY_PORT,
        "ports": {
            "8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "31800"}],
            "18188/tcp": [{"HostIp": "0.0.0.0", "HostPort": "31888"}],
            "22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "31822"}],
        },
        "direct_port_start": 31888, "direct_port_end": 31888, "direct_port_count": 1,
        "env": {"PILOT_ENV_TEST": "value"}, "extra_env": [["PILOT_ENV_TEST", "value"]],
        "instance_api_key": "INSTANCE-KEY-TEST-ONLY",
    }


class _RawVast(PilotVast):
    """Как PilotVast, но ответ карты полный: адреса, порты, окружение и ключ экземпляра (фальшивые значения)."""

    def get_instance(self, instance_id):
        info = super().get_instance(instance_id)
        info.ssh_host = CANARY_HOST
        info.ssh_port = CANARY_PORT
        info.raw = _vast_raw(instance_id, info.actual_status)
        return info


def test_three_not_found_in_a_row_ends_pilot_with_instance_gone(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    vast = _GoneVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK, missing=10 ** 6)
    orc = _pilot_orc(tmp_path, monkeypatch, vast=vast)
    start = orc.clock()
    summary = orc.run()
    # ровно три опроса с ошибкой: после третьего пилот выходит, а не ждёт таймаут
    assert caplog.text.count("PILOT: не удалось получить статус карты") == 3
    assert "PILOT: карта не найдена в Vast (3 раза подряд), пилот завершается." in caplog.text
    assert "PILOT-ИТОГ: результат=not_ready; причина=instance_gone" in caplog.text
    assert summary.exit_code == orch.EXIT_PARTIAL
    assert summary.failed == 1 and summary.completed == 0
    # часы показывают остановку раньше таймаута молчания (60 с): три опроса по 10 с
    assert orc.clock() - start < SILENT_SEC
    # уборка по метке выполнена и на этом пути
    assert vast.label_destroyed


def test_single_not_found_then_success_does_not_stop_pilot(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    vast = _GoneVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK, missing=1)
    summary = _pilot_orc(tmp_path, monkeypatch, vast=vast).run()
    assert "instance_gone" not in caplog.text
    assert "PILOT-ИТОГ: результат=ok" in caplog.text
    assert summary.exit_code == orch.EXIT_OK


def test_ssh_diag_called_only_when_card_is_running(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    seen = []  # статус карты в момент каждого вызова _pull_ssh_diag из цикла пилота
    original = orch.Orchestrator._pull_ssh_diag

    def spy(self, now):
        slot = next(iter(self._slots.values()), None)
        seen.append(slot.vast_status if slot is not None else None)
        return original(self, now)

    monkeypatch.setattr(orch.Orchestrator, "_pull_ssh_diag", spy)
    monkeypatch.setattr(orch.ssh_probe, "run_probe", _offline_probe)
    monkeypatch.setenv("VAST_SSH_KEY", CANARY_KEY)
    vast = _SshVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK, steps=FAKE_STEPS)
    summary = _pilot_orc(tmp_path, monkeypatch, vast=vast).run()
    assert "loading" not in seen, "диагностика вызвана во время loading"
    assert "running" in seen, "диагностика не вызвана при running"
    assert all(s == "running" for s in seen)
    assert summary.completed == 1


def test_pilot_instance_json_is_masked_and_ssh_presence_is_logged(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(orch.ssh_probe, "run_probe", _offline_probe)
    monkeypatch.setenv("VAST_SSH_KEY", CANARY_KEY)
    art = tmp_path / "art"
    handlers, _note = ls.setup_logging("DEBUG", str(art))  # как cli: run/debug.log в каталоге артефакта
    try:
        vast = _RawVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK, steps=FAKE_STEPS)
        _pilot_orc(tmp_path, monkeypatch, vast=vast, artifact_dir=art).run()
    finally:
        ls.teardown_logging(handlers)

    files = list(art.rglob("instance.json"))
    assert len(files) == 1, "instance.json карты пилота не записан"
    raw_text = files[0].read_text(encoding="utf-8")
    data = json.loads(raw_text)

    # поля статуса на месте (последняя запись — при running)
    assert data["actual_status"] == "running"
    assert data["intended_status"] == "running"
    assert "is_bid" in data and "label" in data
    # порты: только внутренние 8000/tcp и 18188/tcp, значения замаскированы
    assert set(data["ports"]) == {"8000/tcp", "18188/tcp"}
    assert data["ports"]["18188/tcp"] == "***"
    # удалённые поля отсутствуют на верхнем уровне
    for key in ("ssh_host", "ssh_port", "public_ipaddr", "env", "extra_env", "instance_api_key"):
        assert key not in data
    assert not any(k.startswith("direct_port") for k in data)
    # ни адрес, ни порт SSH не попали в файл
    assert CANARY_HOST not in raw_text
    assert str(CANARY_PORT) not in raw_text

    debug_text = (art / "run" / "debug.log").read_text(encoding="utf-8")
    assert "PILOT: поля SSH в ответе Vast: ssh_host=есть, ssh_port=есть" in debug_text
    assert CANARY_HOST not in debug_text
    assert str(CANARY_PORT) not in debug_text
    assert CANARY_KEY not in debug_text


# ---------------------------------------------------------------- pilot_start.sh (этап 1.4)

PILOT_START = Path(__file__).resolve().parents[1] / "worker" / "pilot_start.sh"


def test_pilot_start_has_no_workspace_slash_path():
    text = PILOT_START.read_text(encoding="utf-8")
    # /opt/workspace-internal/ComfyUI содержит «/workspace», но не «/workspace/»: проверка по форме со слешем
    assert "/workspace/" not in text


def test_pilot_start_reports_every_failure_step():
    text = PILOT_START.read_text(encoding="utf-8")
    for marker in ("FAILED: venv", "FAILED: clone", "FAILED: args", "FAILED: exited", "FAILED: timeout"):
        assert marker in text, marker


@pytest.mark.skipif(shutil.which("bash") is None, reason="нужен bash")
def test_pilot_start_passes_bash_syntax_check():
    res = subprocess.run([shutil.which("bash"), "-n", str(PILOT_START)],
                         capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
