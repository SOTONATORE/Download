"""Офлайн-тесты консольных строк загрузки весов для нескольких карт (SPEC 8.2).

Монитор каждой карты печатает своё «начало загрузки» (один раз на монитор) и «загружены», оба с номером карты.
Тесты строят монитор через _make_monitor у заготовки Orchestrator без __init__: без Vast, сети и HTTP.
Журнал собирается собственным обработчиком, поэтому тесты не требуют pytest-фикстур.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from Generate.core import orchestrator as orch
    from Generate.core import weights_log as wl
except ImportError:  # pragma: no cover
    from core import orchestrator as orch  # type: ignore
    from core import weights_log as wl  # type: ignore

NAME = "diffusion_models/zimage_turbo-test.safetensors"
MODEL = "zimage_turbo"
CARD_A = 500   # номер экземпляра Vast первой карты
CARD_B = 501   # номер экземпляра Vast второй карты


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _Capture(logging.Handler):
    """Копит тексты записей журнала оркестратора."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _orchestrator(clock: _Clock):
    """Заготовка Orchestrator: только поля, которые читает _make_monitor."""
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    o.clock = clock
    o.speed_check_at_pct = 10
    o.speed_min_mbps = float(wl.speed_threshold_mbps(800.0))
    o.progress_step_pct = 10
    o._stall_sec = 180.0
    o._stage_max_sec = 1200.0
    return o


def _slot(index: int, instance_id: int):
    return orch._CardSlot(index=index, generation=1, instance_id=instance_id,
                          machine_id=1, price=0.0, created=0.0)


def _start_line() -> str:
    return f"[setup] [1/1] Начинаю загрузку: {NAME}"


def _finish_lines() -> list[str]:
    # тот же порядок строк setup.sh, что в test_console_start_once_and_loaded_at_stage_end
    return [f"[setup] [1/1] Файл успешно загружен: {NAME} (1 байт)",
            "[setup] Все файлы модели LTX-2.5 на месте."]


def _run_capture(fn) -> list[str]:
    handler = _Capture()
    prev_level = orch.log.level
    orch.log.addHandler(handler)
    orch.log.setLevel(logging.INFO)
    try:
        fn()
    finally:
        orch.log.removeHandler(handler)
        orch.log.setLevel(prev_level)
    return handler.messages


def _two_cards_run() -> None:
    """Две карты, у каждой свой монитор: сначала оба начала, затем загрузка каждой."""
    o = _orchestrator(_Clock())
    m_a = o._make_monitor(_slot(1, CARD_A), MODEL)
    m_b = o._make_monitor(_slot(2, CARD_B), MODEL)
    m_a.feed([_start_line()])
    m_b.feed([_start_line()])
    m_a.feed(_finish_lines())
    m_b.feed(_finish_lines())


def _console_for(msgs: list[str], card: int) -> list[str]:
    return [m for m in msgs if m.startswith(f"карта {card}:") and f"Веса {MODEL} " in m]


def test_start_line_names_each_card_and_second_card_is_not_suppressed():
    """Старт печатается для каждой карты с её номером: вторая карта не теряет строку из-за первой."""
    msgs = _run_capture(_two_cards_run)
    starts = [m for m in msgs if m.startswith("карта ") and m.endswith("начало загрузки")]
    assert starts == [f"карта {CARD_A}: Веса {MODEL} начало загрузки",
                      f"карта {CARD_B}: Веса {MODEL} начало загрузки"], msgs


def test_start_printed_once_per_monitor():
    """Внутри одного монитора строка старта не повторяется, даже если пришло второе начало файла."""

    def run() -> None:
        mon = _orchestrator(_Clock())._make_monitor(_slot(1, CARD_A), MODEL)
        mon.feed([_start_line()])
        mon.feed([f"[setup] [2/2] Начинаю загрузку: {NAME}"])

    msgs = _run_capture(run)
    starts = [m for m in msgs if m.startswith("карта ") and m.endswith("начало загрузки")]
    assert starts == [f"карта {CARD_A}: Веса {MODEL} начало загрузки"], starts


def test_loaded_line_carries_card_number_and_follows_its_100_percent():
    """«загружены» несёт номер карты и идёт после шага 100% этой же карты."""
    msgs = _run_capture(_two_cards_run)
    for card in (CARD_A, CARD_B):
        loaded = f"карта {card}: Веса {MODEL} загружены"
        assert loaded in msgs, msgs
        hundred = next(i for i, m in enumerate(msgs) if m.startswith(f"карта {card}: Веса {MODEL} 100% ("))
        assert hundred < msgs.index(loaded), msgs


def test_each_card_has_start_progress_and_loaded_and_no_other_card_number():
    """У каждой карты: старт первым, десять шагов процентов, «загружены» последним; чужих номеров нет."""
    msgs = _run_capture(_two_cards_run)
    for card in (CARD_A, CARD_B):
        mine = _console_for(msgs, card)
        assert mine[0] == f"карта {card}: Веса {MODEL} начало загрузки", mine
        assert mine[-1] == f"карта {card}: Веса {MODEL} загружены", mine
        progress = [m for m in mine if "%" in m]
        assert len(progress) == 10, mine
    other_card_lines = [m for m in msgs if m.startswith("карта ") and f"Веса {MODEL} " in m
                        and not (m.startswith(f"карта {CARD_A}:") or m.startswith(f"карта {CARD_B}:"))]
    assert other_card_lines == [], other_card_lines
