"""Учёт и контроль бюджета аренды карт Vast.ai (этап 8, часть 1; SPEC 3, 8.2, 8.3 (г), 0.2, 0.5).

Только стандартная библиотека. Время берётся из внедряемых часов (по умолчанию
time.monotonic), поэтому модуль полностью детерминирован в тестах.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger("Generate.budget")

_SEC_PER_HOUR = 3600.0


class BudgetError(Exception):
    """Базовая ошибка учёта бюджета."""
    exit_code = 1


class BudgetExceededError(BudgetError):
    """Расходы достигли или превысили budget_limit_usd."""
    exit_code = 4


@dataclass
class InstanceCost:
    """Стоимость одной арендованной карты."""
    instance_id: int
    dph: float
    start_time: float
    stop_time: Optional[float] = None

    def accrued_cost(self, now: float) -> float:
        """Начисленная стоимость: (конец - начало) / 3600 * dph; конец = stop_time или now."""
        end = self.stop_time if self.stop_time is not None else now
        elapsed = max(0.0, end - self.start_time)
        return elapsed / _SEC_PER_HOUR * self.dph

    @property
    def is_active(self) -> bool:
        return self.stop_time is None


class BudgetTracker:
    """Отслеживает расходы на арендованные карты и сверяет их с лимитом."""

    def __init__(
        self,
        budget_limit_usd: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not budget_limit_usd > 0:
            raise ValueError(
                f"Лимит бюджета должен быть больше нуля, получено: {budget_limit_usd}"
            )
        self.budget_limit_usd = float(budget_limit_usd)
        self._clock = clock
        self._records: list[InstanceCost] = []

    def __repr__(self) -> str:
        return f"BudgetTracker(лимит={self.budget_limit_usd:.2f} USD, карт={len(self._records)})"

    # --- служебное ---
    def _now(self, now: Optional[float]) -> float:
        return self._clock() if now is None else now

    def _find_active(self, instance_id: int) -> Optional[InstanceCost]:
        for rec in self._records:
            if rec.instance_id == instance_id and rec.is_active:
                return rec
        return None

    # --- учёт карт ---
    def track_instance_start(
        self, instance_id: int, dph: float, start_time: Optional[float] = None
    ) -> None:
        if dph < 0:
            raise BudgetError(
                f"Цена карты {instance_id} не может быть отрицательной: {dph} USD/ч"
            )
        if self._find_active(instance_id) is not None:
            raise BudgetError(f"Карта {instance_id} уже учитывается как активная")
        start = self._now(start_time)
        self._records.append(InstanceCost(instance_id, float(dph), start))
        log.info("Учёт карты %d начат: %.4f USD/ч", instance_id, dph)

    def track_instance_stop(
        self, instance_id: int, stop_time: Optional[float] = None
    ) -> float:
        rec = self._find_active(instance_id)
        if rec is None:
            raise BudgetError(f"Активная карта {instance_id} не найдена в учёте бюджета")
        stop = self._now(stop_time)
        if stop < rec.start_time:
            raise BudgetError(
                f"Время остановки карты {instance_id} раньше времени её запуска"
            )
        rec.stop_time = stop
        cost = rec.accrued_cost(stop)
        log.info("Учёт карты %d завершён: итого %.4f USD", instance_id, cost)
        return cost

    # --- расчёты ---
    def current_spent(self, now: Optional[float] = None) -> float:
        t = self._now(now)
        return sum(rec.accrued_cost(t) for rec in self._records)

    def remaining_budget(self, now: Optional[float] = None) -> float:
        return max(0.0, self.budget_limit_usd - self.current_spent(now))

    def is_exceeded(self, now: Optional[float] = None) -> bool:
        return self.current_spent(now) >= self.budget_limit_usd

    def check(self, now: Optional[float] = None) -> None:
        """Бросает BudgetExceededError, если лимит достигнут или превышен."""
        spent = self.current_spent(now)
        if spent >= self.budget_limit_usd:
            raise BudgetExceededError(
                f"Бюджет исчерпан: потрачено {spent:.4f} USD при лимите "
                f"{self.budget_limit_usd:.2f} USD"
            )

    def can_afford(
        self,
        dph: float,
        estimated_duration_sec: float = 1800.0,
        now: Optional[float] = None,
    ) -> bool:
        """Уложится ли ещё одна карта по цене dph на заданное время в остаток бюджета."""
        if dph < 0 or estimated_duration_sec < 0:
            raise ValueError(
                "Цена и ожидаемая длительность аренды не могут быть отрицательными"
            )
        t = self._now(now)
        if self.is_exceeded(t):
            return False
        estimated = dph * estimated_duration_sec / _SEC_PER_HOUR
        return estimated <= self.remaining_budget(t)

    def summary(self, now: Optional[float] = None) -> dict:
        t = self._now(now)
        breakdown = []
        for rec in self._records:
            cost = rec.accrued_cost(t)
            breakdown.append(
                {
                    "instance_id": rec.instance_id,
                    "dph": rec.dph,
                    "active": rec.is_active,
                    "cost_usd": round(cost, 4),
                    "cost_str": f"${cost:.4f}",
                }
            )
        spent = self.current_spent(t)
        return {
            "budget_limit_usd": self.budget_limit_usd,
            "total_spent": spent,
            "remaining_budget": self.remaining_budget(t),
            "active_cards": sum(1 for r in self._records if r.is_active),
            "total_cards": len(self._records),
            "breakdown": breakdown,
        }
