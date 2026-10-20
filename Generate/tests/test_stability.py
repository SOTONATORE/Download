"""Фильтр стабильности предложений (этап 9е-2, как PARSE_RANK в vast_stable_run.sh) и проверка окна сбора цен.

Офлайн: без сети и без Vast API. Пороги по умолчанию: min_price 0.35, max_spread 0.05, 5 проб, окно 15 проб.
"""
import pytest
import yaml

try:
    from Generate.core import config_loader as cl
    from Generate.core import orchestrator as orc
    from Generate.core.gemini_prompts import GeminiInputError
    from Generate.core.vast_client import GpuOffer
except ImportError:  # плоский импорт
    from core import config_loader as cl  # type: ignore
    from core import orchestrator as orc  # type: ignore
    from core.gemini_prompts import GeminiInputError  # type: ignore
    from core.vast_client import GpuOffer  # type: ignore

MIN_PRICE = 0.35
MAX_SPREAD = 0.05


def _offer(offer_id: int, machine_id: int, bid: float) -> GpuOffer:
    return GpuOffer(
        offer_id=offer_id, gpu_name="RTX 5090", num_gpus=1, vram_gb=32.0, price_per_hr=bid,
        reliability=0.99, inet_down_mbps=900.0, inet_up_mbps=900.0, machine_id=machine_id,
        is_interruptible=True,
    )


def _history(bids, offer_id: int = 1, machine_id: int = 11) -> dict:
    """История одного предложения: по одной пробе на каждое значение min_bid, в порядке поступления."""
    hist: dict = {}
    for bid in bids:
        orc.record_scan_samples(hist, [_offer(offer_id, machine_id, bid)])
    return hist


def _eligible(hist: dict, min_price: float = MIN_PRICE, max_spread: float = MAX_SPREAD) -> list:
    return orc.rank_stable_offers(hist, min_price, max_spread)


def test_four_samples_not_eligible():
    # меньше 5 проб: предложение не допускается, даже если история ровная
    assert _eligible(_history([0.40] * 4)) == []


def test_five_samples_spread_six_percent_not_eligible():
    # (0.424 - 0.40) / 0.40 = 6 %, больше 5 %
    assert _eligible(_history([0.40, 0.40, 0.40, 0.40, 0.424])) == []


def test_five_samples_spread_four_percent_eligible():
    # (0.416 - 0.40) / 0.40 = 4 %, не больше 5 %
    res = _eligible(_history([0.40, 0.40, 0.40, 0.40, 0.416]))
    assert [o.price_per_hr for o in res] == [0.416]


def test_last_sample_below_min_price_not_eligible_even_with_flat_history():
    # все пробы 0.34 — история ровная, но последняя проба ниже min_price_per_hour 0.35
    assert _eligible(_history([0.34] * 5)) == []


def test_only_last_fifteen_samples_count():
    # Числа из постановки: первые 10 проб 0.20, последние 15 проб 0.30. Окно хранит только последние 15,
    # поэтому разброс считается по одним 0.30 и предложение допускается.
    # Порог передан явно (0.25): при пороге 0.35 последняя проба 0.30 отсеивается правилом «последняя проба ≥ min_price».
    hist = _history([0.20] * 10 + [0.30] * 15)
    assert len(hist[(1, 11)]) == 15  # память ограничена окном на ключ
    res = _eligible(hist, min_price=0.25)
    assert [o.price_per_hr for o in res] == [0.30]


def test_eligible_offers_sorted_by_min_bid_ascending():
    hist: dict = {}
    for offer_id, bid in [(1, 0.40), (2, 0.36), (3, 0.38)]:
        for _ in range(5):
            orc.record_scan_samples(hist, [_offer(offer_id, 100 + offer_id, bid)])
    res = _eligible(hist)
    assert [o.offer_id for o in res] == [2, 3, 1]
    assert [o.price_per_hr for o in res] == [0.36, 0.38, 0.40]


def test_scan_sec_shorter_than_five_steps_is_refused_at_load():
    raw = yaml.safe_load(cl.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["scan_sec"] = 60
    raw["scan_step_sec"] = 20  # 60 < 5 × 20 = 100
    with pytest.raises(GeminiInputError) as ei:
        cl.parse_config(raw)
    assert ei.value.exit_code == 2
    assert "«scan_sec»" in str(ei.value), str(ei.value)


def test_scan_sec_exactly_five_steps_is_accepted():
    raw = yaml.safe_load(cl.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["scan_sec"] = 100
    raw["scan_step_sec"] = 20  # 100 = 5 × 20: граница допустима
    cfg = cl.parse_config(raw)
    assert cfg.scan_sec == 100
