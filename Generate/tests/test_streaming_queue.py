"""Стриминговая очередь задач карты (Part 1A): буфер на карту, уровни назначения, 429, откат буфера.

Офлайн: Rig и FakeWorker из test_orchestrator, время — фальшивые часы. Сообщения на русском (SPEC 0.5).
Тесты проверяют назначение и откат по одной карте и по нескольким картам; задержки опроса и Vast не проверяются.
"""
from __future__ import annotations

import dataclasses
import sys
from collections import deque
from pathlib import Path

import httpx
import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from Generate.core import orchestrator as orch
    from Generate.tests.test_orchestrator import CARD_IP, CARD_PORT, Rig
except ImportError:  # pragma: no cover
    from core import orchestrator as orch  # type: ignore
    from tests.test_orchestrator import CARD_IP, CARD_PORT, Rig  # type: ignore


def _use_prompts(rig, nums=range(1, 11)) -> None:
    """Промпты клипов для _dispatch: текст и число кадров, как в prompts.json."""
    rig.orc._prompts = {str(n): {"prompt": f"промпт {n}", "num_frames": 9} for n in nums}


def _card(rig, index: int = 1, *, instance_id: int = 101, machine_id: int = 11) -> orch._CardSlot:
    """Готовая карта, зарегистрированная в оркестраторе; адрес воркера — фейковый."""
    slot = orch._CardSlot(index=index, generation=1, instance_id=instance_id, machine_id=machine_id,
                          price=0.3, created=rig.clock(), state="ready")
    slot.url = f"http://{CARD_IP}:{CARD_PORT}"
    slot.accepted = True
    rig.orc._slots[index] = slot
    return slot


# ---------------------------------------------------------------------------
# Совместимость _CardSlot и _Task
# ---------------------------------------------------------------------------

def test_card_slot_task_argument_seeds_buffer_and_is_not_a_field():
    t = orch._Task(1, "t1", 0.0)
    slot = orch._CardSlot(index=1, generation=1, instance_id=1, machine_id=1, price=0.1, created=0.0, task=t)

    assert slot.tasks == [t]
    assert slot.task is t
    assert "task" not in {f.name for f in dataclasses.fields(orch._CardSlot)}


def test_card_slot_task_setter_assigns_and_clears_buffer():
    slot = orch._CardSlot(index=1, generation=1, instance_id=1, machine_id=1, price=0.1, created=0.0)
    assert slot.task is None and slot.tasks == []

    t = orch._Task(2, "t2", 0.0)
    slot.task = t
    assert slot.tasks == [t] and slot.task is t

    slot.task = None
    assert slot.tasks == [] and slot.task is None


def test_task_getter_prefers_running_task_in_buffer():
    slot = orch._CardSlot(index=1, generation=1, instance_id=1, machine_id=1, price=0.1, created=0.0)
    queued = orch._Task(1, "a", 0.0)
    running = orch._Task(2, "b", 0.0, state="running")
    slot.tasks = [queued, running]

    assert slot.task is running


def test_task_dataclass_defaults_keep_positional_signature():
    t = orch._Task(3, "t3", 5.0)

    assert (t.num, t.id, t.started) == (3, "t3", 5.0)
    assert t.running_since is None and t.state == "queued"


# ---------------------------------------------------------------------------
# Уровни назначения: каждой готовой карте по одной задаче, потом по второй
# ---------------------------------------------------------------------------

def test_level_dispatch_gives_each_card_one_task_before_second(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2, 3, 4, 5, 6))
    _use_prompts(rig)
    a = _card(rig, 1)
    b = _card(rig, 2, instance_id=102, machine_id=12)
    rig.orc._queue = deque([1, 2, 3, 4, 5, 6])
    now = rig.clock()

    # уровень 1: по одной задаче каждой карте (как в _step_slot)
    rig.orc._dispatch(a, now)
    rig.orc._dispatch(b, now)
    # уровни 2..MAX: сначала вторые задачи обеим картам, затем третьи
    rig.orc._dispatch_levels(now, False)

    assert [t.num for t in a.tasks] == [1, 3, 5]
    assert [t.num for t in b.tasks] == [2, 4, 6]
    assert len(rig.orc._queue) == 0


def test_single_card_buffers_active_plus_three_queued(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2, 3, 4, 5))
    _use_prompts(rig)
    a = _card(rig, 1)
    rig.orc._queue = deque([1, 2, 3, 4, 5])
    now = rig.clock()

    rig.orc._dispatch(a, now)
    rig.orc._dispatch_levels(now, False)

    assert [t.num for t in a.tasks] == [1, 2, 3, 4]      # 1 активная + 3 в очереди воркера
    assert list(rig.orc._queue) == [5]
    assert rig.orc._inflight() == 4


# ---------------------------------------------------------------------------
# 429: клип возвращается в начало очереди без попытки и без сбоя карты
# ---------------------------------------------------------------------------

def test_429_returns_clip_to_front_without_attempt_or_net_fail(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2, 3))
    _use_prompts(rig)
    slot = _card(rig, 1)
    rig.orc._queue = deque([1, 2, 3])
    now = rig.clock()
    rig.orc._dispatch(slot, now)                 # уровень 1: клип 1 принят воркером

    calls = []

    def busy(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(429, text="busy")

    rig.orc._http = httpx.Client(transport=httpx.MockTransport(busy))
    rig.orc._dispatch_levels(now, False)         # уровень 2: воркер отвечает 429

    assert list(rig.orc._queue) == [2, 3]         # клип 2 вернулся в начало очереди
    assert rig.orc._attempts.get(2, 0) == 0       # попытка не засчитана
    assert slot.net_fail == 0                     # сбоем карты не считается
    assert len(slot.tasks) == 1                   # карта по-прежнему держит только клип 1
    assert slot.index in rig.orc._tick_429        # до следующего обхода карте задачи не назначаем
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Откат буфера при снятии карты: все задачи возвращаются без попытки
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [True, False])
def test_retire_rolls_back_all_buffered_tasks_without_attempts(tmp_path, monkeypatch, bad):
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2, 3, 4, 5))
    _use_prompts(rig)
    slot = _card(rig, 1)
    rig.orc._queue = deque([1, 2, 3, 4, 5])
    now = rig.clock()
    for _ in range(4):
        rig.orc._dispatch(slot, now)
    assert [t.num for t in slot.tasks] == [1, 2, 3, 4]
    assert list(rig.orc._queue) == [5]

    rig.orc._retire(slot, "тест снятия карты", bad=bad)

    assert slot.tasks == []
    assert list(rig.orc._queue) == [1, 2, 3, 4, 5]   # все четыре клипа снова в очереди, по порядку
    assert all(rig.orc._attempts.get(n, 0) == 0 for n in (1, 2, 3, 4))
    assert not rig.orc._failed and not rig.orc._done
    assert rig.orc._slots.get(1) is None
    assert (11 in rig.orc._bad_machines) == bad      # при аварии машина исключается на запуск, как раньше


# ---------------------------------------------------------------------------
# Сбой одной задачи на здоровой карте: попытка тратится только на неё
# ---------------------------------------------------------------------------

def test_single_task_failure_keeps_other_buffered_tasks(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2, 3, 4))
    _use_prompts(rig)
    rig.worker.plan = {1: ["error"]}             # первая попытка клипа 1 на воркере завершится ошибкой
    slot = _card(rig, 1)
    rig.orc._queue = deque([1, 2, 3, 4])
    now = rig.clock()
    for _ in range(4):
        rig.orc._dispatch(slot, now)

    rig.orc._poll_task(slot, now)

    assert [t.num for t in slot.tasks] == [2, 3, 4]          # снята только задача с ошибкой
    assert all(t.state in ("queued", "running") for t in slot.tasks)  # остальные остаются в очереди или в работе
    assert list(rig.orc._queue) == [1]                       # клип 1 — на повтор
    assert rig.orc._attempts[1] == 1                         # одна попытка, как при обычной ошибке задачи
    assert all(rig.orc._attempts[n] == 1 for n in (2, 3, 4))
    assert 1 not in rig.orc._failed
    assert slot.net_fail == 0
