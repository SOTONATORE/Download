"""Офлайн-тесты core/budget.py: фиктивные часы, без time.sleep."""
import pytest

try:
    from Generate.core.budget import (
        BudgetError,
        BudgetExceededError,
        BudgetTracker,
        InstanceCost,
    )
except ImportError:
    from core.budget import (
        BudgetError,
        BudgetExceededError,
        BudgetTracker,
        InstanceCost,
    )

H = 3600.0


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def tracker(clock):
    return BudgetTracker(budget_limit_usd=5.0, clock=clock)


# --- инициализация ---
def test_default_limit():
    assert BudgetTracker().budget_limit_usd == 5.0


@pytest.mark.parametrize("bad", [0, 0.0, -1, -5.5])
def test_invalid_budget_rejected(bad):
    with pytest.raises(ValueError) as ei:
        BudgetTracker(budget_limit_usd=bad)
    assert "Лимит бюджета" in str(ei.value)


def test_exception_hierarchy():
    assert issubclass(BudgetExceededError, BudgetError)
    assert issubclass(BudgetError, Exception)


# --- InstanceCost ---
def test_accrued_cost_active_and_stopped():
    ic = InstanceCost(1, 0.9, 0.0)
    assert ic.accrued_cost(H) == pytest.approx(0.9)
    assert ic.accrued_cost(H / 2) == pytest.approx(0.45)
    ic.stop_time = H
    assert ic.accrued_cost(10 * H) == pytest.approx(0.9)


def test_accrued_cost_zero_stop_time_is_stop():
    ic = InstanceCost(1, 1.0, 0.0, stop_time=0.0)
    assert ic.accrued_cost(H) == 0.0


def test_accrued_cost_never_negative():
    assert InstanceCost(1, 1.0, 100.0).accrued_cost(50.0) == 0.0


# --- одна карта ---
def test_single_instance_over_time(tracker, clock):
    tracker.track_instance_start(1, 0.9)
    assert tracker.current_spent() == 0.0
    clock.t = H
    assert tracker.current_spent() == pytest.approx(0.9)
    clock.t = 2 * H
    assert tracker.current_spent() == pytest.approx(1.8)


def test_stop_returns_final_cost_and_freezes(tracker, clock):
    tracker.track_instance_start(1, 0.9)
    clock.t = H
    assert tracker.track_instance_stop(1) == pytest.approx(0.9)
    assert tracker.current_spent(now=5 * H) == pytest.approx(0.9)


def test_explicit_timestamps(tracker):
    tracker.track_instance_start(1, 1.0, start_time=100.0)
    cost = tracker.track_instance_stop(1, stop_time=100.0 + H)
    assert cost == pytest.approx(1.0)


# --- несколько карт ---
def test_multiple_overlapping_instances(tracker):
    tracker.track_instance_start(1, 1.0, start_time=0.0)
    tracker.track_instance_start(2, 0.5, start_time=H)
    # в момент 2ч: карта 1 = 2.0, карта 2 = 0.5
    assert tracker.current_spent(now=2 * H) == pytest.approx(2.5)
    tracker.track_instance_stop(1, stop_time=2 * H)
    # в момент 4ч: карта 1 заморожена на 2.0, карта 2 = 1.5
    assert tracker.current_spent(now=4 * H) == pytest.approx(3.5)
    s = tracker.summary(now=4 * H)
    assert s["active_cards"] == 1
    assert s["total_cards"] == 2


def test_restart_after_stop_allowed(tracker):
    tracker.track_instance_start(1, 1.0, start_time=0.0)
    tracker.track_instance_stop(1, stop_time=H)
    tracker.track_instance_start(1, 1.0, start_time=2 * H)
    assert tracker.current_spent(now=3 * H) == pytest.approx(2.0)


# --- превышение и остаток ---
def test_exceeded_and_remaining_transitions(tracker):
    tracker.track_instance_start(1, 1.0, start_time=0.0)
    assert not tracker.is_exceeded(now=4 * H)
    assert tracker.remaining_budget(now=4 * H) == pytest.approx(1.0)
    assert tracker.is_exceeded(now=5 * H)  # ровно лимит = превышение
    assert tracker.remaining_budget(now=5 * H) == 0.0
    assert tracker.remaining_budget(now=9 * H) == 0.0


def test_check_raises_russian(tracker):
    tracker.track_instance_start(1, 1.0, start_time=0.0)
    tracker.check(now=H)
    with pytest.raises(BudgetExceededError) as ei:
        tracker.check(now=6 * H)
    assert "Бюджет исчерпан" in str(ei.value)


def test_stopping_instance_halts_growth(tracker):
    tracker.track_instance_start(1, 1.0, start_time=0.0)
    tracker.track_instance_stop(1, stop_time=2 * H)
    assert not tracker.is_exceeded(now=100 * H)


# --- can_afford ---
def test_can_afford_default_duration(tracker):
    # 0.9 $/ч * 0.5 ч = 0.45 $ <= 5.0
    assert tracker.can_afford(0.9, now=0.0)


def test_can_afford_boundaries(tracker):
    assert tracker.can_afford(5.0, 3600.0, now=0.0)  # ровно остаток
    assert not tracker.can_afford(5.01, 3600.0, now=0.0)
    assert tracker.can_afford(10.0, 1800.0, now=0.0)
    assert not tracker.can_afford(10.0, 1801.0, now=0.0)


def test_can_afford_accounts_for_running_cards(tracker):
    tracker.track_instance_start(1, 1.0, start_time=0.0)
    # к 4ч потрачено 4.0, остаток 1.0
    assert tracker.can_afford(1.0, 3600.0, now=4 * H)
    assert not tracker.can_afford(1.0, 3601.0, now=4 * H)
    assert not tracker.can_afford(0.9, 3600.0 * 1.2, now=4 * H)


def test_can_afford_false_when_exceeded_even_free(tracker):
    tracker.track_instance_start(1, 1.0, start_time=0.0)
    assert not tracker.can_afford(0.0, now=5 * H)


def test_can_afford_rejects_negative(tracker):
    with pytest.raises(ValueError) as ei:
        tracker.can_afford(-1.0)
    assert "отрицательными" in str(ei.value)
    with pytest.raises(ValueError):
        tracker.can_afford(1.0, -5.0)


# --- summary ---
def test_summary_contents(tracker):
    tracker.track_instance_start(1, 1.0, start_time=0.0)
    tracker.track_instance_start(2, 2.0, start_time=0.0)
    tracker.track_instance_stop(1, stop_time=H)
    s = tracker.summary(now=H)
    assert s["budget_limit_usd"] == 5.0
    assert s["total_spent"] == pytest.approx(3.0)
    assert s["remaining_budget"] == pytest.approx(2.0)
    assert s["active_cards"] == 1
    assert s["total_cards"] == 2
    by_id = {b["instance_id"]: b for b in s["breakdown"]}
    assert by_id[1]["active"] is False
    assert by_id[2]["active"] is True
    assert by_id[1]["cost_usd"] == pytest.approx(1.0)
    assert by_id[2]["cost_str"] == "$2.0000"


def test_summary_empty(tracker):
    s = tracker.summary()
    assert s["total_spent"] == 0.0
    assert s["remaining_budget"] == 5.0
    assert s["breakdown"] == []


def test_default_now_uses_clock(tracker, clock):
    tracker.track_instance_start(1, 1.0)
    clock.t = H
    assert tracker.summary()["total_spent"] == pytest.approx(1.0)


# --- русские сообщения ---
def test_duplicate_start_message(tracker):
    tracker.track_instance_start(7, 1.0, start_time=0.0)
    with pytest.raises(BudgetError) as ei:
        tracker.track_instance_start(7, 1.0, start_time=1.0)
    assert "уже учитывается" in str(ei.value)


def test_stop_unknown_message(tracker):
    with pytest.raises(BudgetError) as ei:
        tracker.track_instance_stop(99)
    assert "не найдена" in str(ei.value)


def test_stop_twice_message(tracker):
    tracker.track_instance_start(1, 1.0, start_time=0.0)
    tracker.track_instance_stop(1, stop_time=10.0)
    with pytest.raises(BudgetError):
        tracker.track_instance_stop(1, stop_time=20.0)


def test_negative_dph_message(tracker):
    with pytest.raises(BudgetError) as ei:
        tracker.track_instance_start(1, -0.1)
    assert "отрицательной" in str(ei.value)


def test_stop_before_start_message(tracker):
    tracker.track_instance_start(1, 1.0, start_time=100.0)
    with pytest.raises(BudgetError) as ei:
        tracker.track_instance_stop(1, stop_time=50.0)
    assert "раньше" in str(ei.value)
