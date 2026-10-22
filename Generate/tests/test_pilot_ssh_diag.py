"""Тест: SSH-диагностика вызывается из цикла пилота (Orchestrator._run_pilot).

Офлайн: Vast, HTTP и часы — заглушки из test_orchestrator.py (PilotVast, _pilot_orc, make_offer).
Настоящей SSH-пробы нет: run_probe подменён на функцию, которая падает при вызове.
Канареечные значения заведомо фальшивые: адрес из диапазона документации; в ассертах не печатаются.
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

    def spy(self, now):
        slot = next(iter(self._slots.values()), None)
        seen.append(slot.vast_status if slot is not None else None)
        return original(self, now)

    def no_probe(*args, **kwargs):
        raise AssertionError("в тесте не должна выполняться настоящая SSH-проба")

    monkeypatch.setattr(orch.Orchestrator, "_pull_ssh_diag", spy)
    monkeypatch.setattr(orch.ssh_probe, "run_probe", no_probe)

    vast = _SshPilotVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    summary = _pilot_orc(tmp_path, monkeypatch, vast=vast).run()

    assert seen, "_pull_ssh_diag не вызывался в цикле пилота"
    assert "running" in seen, "карта не дошла до running в цикле пилота"
    # результат пилота прежний: ответ 200 даёт ok, код выхода EXIT_OK
    assert "PILOT-ИТОГ: результат=ok" in caplog.text
    assert summary.completed == 1
    assert summary.exit_code == orch.EXIT_OK
