"""Тест: SSH-диагностика вызывается из цикла пилота (Orchestrator._run_pilot).

Офлайн: Vast, HTTP и часы — заглушки из test_orchestrator.py (PilotVast, _pilot_orc, make_offer).
Настоящей SSH-пробы нет: run_probe подменён на функцию, которая возвращает успешный результат и запоминает
аргументы адреса. Канареечные значения заведомо фальшивые: адрес из диапазона документации; в ассертах не печатаются.
"""
import logging
import sys
from pathlib import Path

try:
    from Generate.core import orchestrator as orch
    from Generate.tests.test_orchestrator import PILOT_PORTS_OK, PilotVast, _pilot_orc, make_offer
except ImportError:  # pragma: no cover
    _HERE = Path(__file__).resolve()
    for _p in (_HERE.parents[2], _HERE.parents[1]):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    from core import orchestrator as orch  # type: ignore
    from tests.test_orchestrator import PILOT_PORTS_OK, PilotVast, _pilot_orc, make_offer  # type: ignore

CANARY_HOST = "203.0.113.7"
CANARY_PORT = 2222
CANARY_KEY = "CANARY-KEY-TEST-ONLY-NOT-A-KEY"
# четыре опроса «loading», затем «running». Первый тик диагностики идёт на первом опросе (адрес уже скопирован),
# следующий тик due не раньше SSH_DIAG_INTERVAL_SEC = 30 с; при шаге опроса 10 с это как раз пятый опрос (running)
FAKE_STEPS = ("loading", "loading", "loading", "loading", "running")


class _SshPilotVast(PilotVast):
    """Как PilotVast, но у карты есть ssh_host и ssh_port (нужны для выбора карты в диагностике)."""

    def get_instance(self, instance_id):
        info = super().get_instance(instance_id)
        info.ssh_host = CANARY_HOST
        info.ssh_port = CANARY_PORT
        return info


def test_pilot_loop_calls_ssh_diag_while_card_runs(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    seen = []  # (статус карты в момент вызова) по каждой итерации цикла пилота
    original = orch.Orchestrator._pull_ssh_diag
    probe_calls = []  # (host, port, имя проверки): аргументы подменённой пробы, в ассертах не печатаются

    def spy(self, now):
        slot = next(iter(self._slots.values()), None)
        seen.append(slot.vast_status if slot is not None else None)
        return original(self, now)

    def fake_probe(host, port, key_text, command_name, timeout_sec=None):
        probe_calls.append((host, port, command_name))
        return orch.ssh_probe.ProbeResult(ok=True, returncode=0, output="ok",
                                          failure_class="ok", output_length=2)

    monkeypatch.setattr(orch.Orchestrator, "_pull_ssh_diag", spy)
    monkeypatch.setattr(orch.ssh_probe, "run_probe", fake_probe)
    monkeypatch.setenv("VAST_SSH_KEY", CANARY_KEY)

    art = tmp_path / "art"  # то же, что --artifact-dir в рабочем процессе: сюда пишется ssh_diag.log
    vast = _SshPilotVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK, steps=FAKE_STEPS)
    summary = _pilot_orc(tmp_path, monkeypatch, vast=vast, artifact_dir=art).run()

    assert seen, "_pull_ssh_diag не вызывался в цикле пилота"
    assert "running" in seen, "карта не дошла до running в цикле пилота"

    matched = any(host == CANARY_HOST and port == CANARY_PORT for host, port, _ in probe_calls)
    assert probe_calls, "run_probe не вызывался в цикле пилота"
    assert matched, "run_probe получил не адрес карты"

    logs = list(art.rglob(orch.SSH_DIAG_FILE))
    assert logs, "ssh_diag.log не записан в каталог артефакта"
    assert "--- ping (ok) ---" in logs[0].read_text(encoding="utf-8")

    # результат пилота прежний: ответ 200 даёт ok, код выхода EXIT_OK
    assert "PILOT-ИТОГ: результат=ok" in caplog.text
    assert summary.completed == 1
    assert summary.exit_code == orch.EXIT_OK
