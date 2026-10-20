"""Цикл аренды (этап 9е-3): перебитие ставки, исключение предложения на запуск, лимит запросов аренды,
ошибка API при аренде и пауза между поисками.

Офлайн: Rig и FakeVast из test_orchestrator, время — фальшивые часы. Сообщения на русском (SPEC 0.5).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from Generate.core import config_loader as cl
    from Generate.core import orchestrator as orch
    from Generate.core import vast_client as vc
    from Generate.core.gemini_prompts import GeminiInputError
    from Generate.tests.test_orchestrator import FakeVast, Rig, make_offer
except ImportError:  # pragma: no cover
    from core import config_loader as cl  # type: ignore
    from core import orchestrator as orch  # type: ignore
    from core import vast_client as vc  # type: ignore
    from core.gemini_prompts import GeminiInputError  # type: ignore
    from tests.test_orchestrator import FakeVast, Rig, make_offer  # type: ignore


class _SpyVast(FakeVast):
    """Фиксирует время и предложение каждого запроса аренды и каждого поиска."""

    def __init__(self, offers, events, clock) -> None:
        super().__init__(offers, events)
        self.clock = clock
        self.attempts: list = []   # (время, offer_id) для каждого запроса аренды, успешного или нет
        self.searches: list = []   # время каждого поиска

    def create_instance(self, offer_id, *args, **kwargs):
        self.attempts.append((self.clock(), offer_id))
        return super().create_instance(offer_id, *args, **kwargs)

    def search_offers(self, filters):
        self.searches.append(self.clock())
        return super().search_offers(filters)

    def offer_ids(self) -> list:
        return [offer for _, offer in self.attempts]


class _OutbidVast(_SpyVast):
    """Карты предложений из outbid_offers сразу остановлены Vast: is_bid true, cur_state stopped."""

    def __init__(self, offers, events, clock, outbid_offers) -> None:
        super().__init__(offers, events, clock)
        self.outbid_offers = set(outbid_offers)

    def get_instance(self, instance_id):
        if self._offer_of.get(instance_id) in self.outbid_offers:
            raw = {"id": instance_id, "is_bid": True, "cur_state": "stopped", "intended_status": "stopped"}
            return vc.InstanceInfo(instance_id=instance_id, actual_status="exited", intended_status="stopped",
                                   is_bid=True, dph=0.2, raw=raw)
        return super().get_instance(instance_id)


def _use_vast(rig: Rig, vast: _SpyVast) -> _SpyVast:
    """Подменяет клиент Vast у уже созданного Rig (Rig передаёт его оркестратору при создании)."""
    rig.vast = vast
    rig.orc.vast = vast
    return vast


# ---------------------------------------------------------------------------
# a. Перебитие ставки на первой карте
# ---------------------------------------------------------------------------

def test_outbid_on_first_card_excludes_offer_and_rents_next(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    offers = [make_offer(1, 11, price=0.2), make_offer(2, 22, price=0.3)]
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=offers)
    vast = _use_vast(rig, _OutbidVast(offers, rig.events, rig.clock, outbid_offers={1}))

    summary = rig.run()

    assert vast.offer_ids() == [1, 2]                      # второй поиск дал другое предложение, оно арендовано
    assert 1 in rig.orc._bad_offers                        # предложение исключено на этот запуск
    assert 11 not in rig.orc._bad_machines                 # машина не исключается
    assert list(tmp_path.rglob("bad_hosts.txt")) == []     # постоянный список не пишется
    assert vast.created[0]["id"] in vast.destroyed         # перебитая карта снята обычным путём
    assert summary.completed == 1 and summary.exit_code == 0
    assert "ставка bid перебита" in caplog.text


def test_outbid_check_uses_bid_flag_and_stopped_state():
    """Таблица правила перебития ставки: is_bid true и карта остановлена (cur_state или intended_status)."""
    def info(is_bid, cur_state, intended):
        return vc.InstanceInfo(instance_id=1, actual_status="exited", intended_status=intended,
                               is_bid=is_bid, raw={"cur_state": cur_state})

    assert orch.Orchestrator._is_outbid(info(True, "stopped", "running")) is True
    assert orch.Orchestrator._is_outbid(info(True, "running", "stopped")) is True
    assert orch.Orchestrator._is_outbid(info(True, "running", "running")) is False
    assert orch.Orchestrator._is_outbid(info(False, "stopped", "stopped")) is False
    assert orch.Orchestrator._is_outbid(info(None, "stopped", "stopped")) is False


# ---------------------------------------------------------------------------
# b. Предложение, по которому был запрос аренды, в этом запуске не выбирается снова
# ---------------------------------------------------------------------------

def test_offer_that_failed_rental_request_is_not_chosen_again(tmp_path, monkeypatch):
    offers = [make_offer(1, 11, price=0.2), make_offer(2, 22, price=0.3)]
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=offers)
    vast = _use_vast(rig, _SpyVast(offers, rig.events, rig.clock))
    vast.fail_offers = {1}                                 # запрос по предложению 1 не принят API

    summary = rig.run()
    assert vast.offer_ids() == [1, 2]                      # предложение 1 пробовано один раз

    # позже в том же запуске: ни предложение 1, ни успешное предложение 2 снова не выбираются
    assert rig.orc._rent_one(rig.clock.now) == "retry"
    assert vast.offer_ids() == [1, 2]
    assert summary.completed == 1


def test_offer_that_succeeded_is_not_rented_again_in_same_run(tmp_path, monkeypatch):
    offers = [make_offer(2, 22, price=0.3)]
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=offers)
    vast = _use_vast(rig, _SpyVast(offers, rig.events, rig.clock))

    assert rig.orc._rent_one(rig.clock.now) == "ok"
    assert rig.orc._rent_one(rig.clock.now) == "retry"     # то же предложение, карта уже арендована
    assert vast.offer_ids() == [2]


# ---------------------------------------------------------------------------
# c. Лимит запросов аренды
# ---------------------------------------------------------------------------

def test_max_rent_tries_three_failed_requests_end_the_run(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    offers = [make_offer(i, 10 + i, price=0.2 + i / 100) for i in range(1, 5)]
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=offers, max_rent_tries=3)
    vast = _use_vast(rig, _SpyVast(offers, rig.events, rig.clock))
    vast.fail_offers = {1, 2, 3, 4}                        # все запросы аренды отклонены API

    summary = rig.run()

    assert vast.offer_ids() == [1, 2, 3]                   # ровно три запроса, четвёртое предложение не пробовано
    assert rig.orc._rent_requests == 3
    assert rig.orc._rent_exhausted is True
    assert summary.cards_rented == 0 and summary.completed == 0
    lines = [r for r in caplog.records if "max_rent_tries" in r.getMessage()]
    assert len(lines) == 1                                 # одна строка в журнал
    assert "rent_error" in lines[0].getMessage()


def test_max_rent_tries_defaults_to_five_and_rejects_zero(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    assert rig.orc.max_rent_tries == 5
    with pytest.raises(GeminiInputError):
        Rig(tmp_path / "zero", monkeypatch, missing=(1,), max_rent_tries=0)


def test_config_key_max_rent_tries_default_and_minimum():
    raw = yaml.safe_load(cl.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert cl.parse_config(raw).max_rent_tries == 5
    raw["max_rent_tries"] = 0
    with pytest.raises(GeminiInputError):
        cl.parse_config(raw)


# ---------------------------------------------------------------------------
# d. Ошибка API при аренде: повтор как и раньше (rent_retry_sec, со следующим предложением)
# ---------------------------------------------------------------------------

def test_rental_api_error_is_retried_after_rent_retry_sec_as_before(tmp_path, monkeypatch):
    offers = [make_offer(1, 11, price=0.2), make_offer(2, 22, price=0.3)]
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=offers, rent_retry_sec=30.0)
    vast = _use_vast(rig, _SpyVast(offers, rig.events, rig.clock))
    vast.fail_offers = {1}

    summary = rig.run()

    assert vast.offer_ids() == [1, 2]                      # два запроса: отказ, затем успешная аренда
    t_fail, t_retry = (t for t, _ in vast.attempts)
    assert t_retry - t_fail == 30.0                        # пауза равна rent_retry_sec, как сейчас
    assert summary.cards_rented == 1 and summary.completed == 1


# ---------------------------------------------------------------------------
# e. Пауза, когда нет подходящего непробованного предложения: scan_step_sec
# ---------------------------------------------------------------------------

def test_wait_without_eligible_offer_uses_scan_step_sec(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), poll_interval_sec=1.0, scan_step_sec=7.0,
              rent_retry_sec=30.0, job_deadline_min=1)
    vast = _use_vast(rig, _SpyVast([], rig.events, rig.clock))   # подходящих предложений нет

    summary = rig.run()

    assert len(vast.searches) >= 3
    gaps = [b - a for a, b in zip(vast.searches, vast.searches[1:])]
    assert gaps and all(gap == 7.0 for gap in gaps)        # шаг scan_step_sec, не rent_retry_sec (30 с)
    waits = [r for r in caplog.records if "Подходящих предложений Vast нет" in r.getMessage()]
    assert len(waits) == len(vast.searches)                # одна строка журнала на каждое ожидание
    assert vast.attempts == [] and summary.completed == 0
