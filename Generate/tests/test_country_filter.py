"""Фильтр страны предложения (exclude_countries). Офлайн: сеть не используется, предложения — словари."""
try:
    from Generate.core import vast_client as vc
except ImportError:  # pragma: no cover
    from core import vast_client as vc  # type: ignore


def _bundle(**over):
    """Предложение, которое проходит все остальные проверки; страна задаётся тестом."""
    b = {
        "id": 1, "dph_total": 0.5, "reliability2": 0.99, "num_gpus": 1, "machine_id": 5,
        "inet_down": 1000, "inet_up_cost": 0.001, "inet_down_cost": 0.001,
        "gpu_ram": 32 * 1024, "gpu_name": "RTX 5090", "disk_space": 200,
        "cuda_max_good": 12.9, "disk_bw": 2000,
        "geolocation": "California, US",
    }
    b.update(over)
    return b


def _filter(countries=("Vietnam",)):
    return vc.OfferFilter(exclude_countries=list(countries))


def _reason(b, f=None):
    return vc.VastClient._check_offer(b, f or _filter())[0]


# ---------------------------------------------------------------- исключение

def test_vietnam_geolocation_is_excluded():
    assert _reason(_bundle(geolocation="Vietnam, VN")) == "country"


def test_hanoi_vn_is_excluded_by_code():
    # регион «Hanoi» не совпадает с названием, но код VN совпадает с Vietnam -> VN
    assert _reason(_bundle(geolocation="Hanoi, VN")) == "country"


def test_viet_nam_name_maps_to_vn_code():
    assert _reason(_bundle(geolocation="Ho Chi Minh, VN"), _filter(["Viet Nam"])) == "country"


def test_code_entry_excludes_matching_code():
    assert _reason(_bundle(geolocation="Hanoi, VN"), _filter(["VN"])) == "country"


def test_name_match_is_case_insensitive():
    assert _reason(_bundle(geolocation="VIETNAM, vn"), _filter(["vietnam"])) == "country"


# ---------------------------------------------------------------- сохранение

def test_sweden_is_kept():
    assert _reason(_bundle(geolocation="Sweden, SE")) is None


def test_empty_region_cn_is_kept():
    assert _reason(_bundle(geolocation=", CN")) is None


def test_croatia_is_kept():
    assert _reason(_bundle(geolocation="Croatia, HR")) is None


def test_empty_exclude_list_excludes_nothing():
    assert _reason(_bundle(geolocation="Vietnam, VN"), _filter([])) is None


# ---------------------------------------------------------------- отсутствие поля

def test_missing_geolocation_gets_country_unknown_reason_and_is_kept():
    b = _bundle()
    b.pop("geolocation")
    reason, offer = vc.VastClient._check_offer(b, _filter())
    assert reason == "country_unknown"
    assert offer is not None
    assert dict(vc._REJECT_LABELS)["country_unknown"] == "страна не разобрана"


def test_unreadable_geolocation_is_kept_as_unknown():
    assert _reason(_bundle(geolocation="Vietnam")) == "country_unknown"
    assert _reason(_bundle(geolocation=None)) == "country_unknown"


def test_country_check_skipped_when_list_empty_even_if_field_missing():
    b = _bundle()
    b.pop("geolocation")
    assert _reason(b, _filter([])) is None


# ---------------------------------------------------------------- прочее

def test_region_part_is_not_used_for_code_match():
    # «VN» в регионе не должен совпасть с кодом страны «US»
    assert _reason(_bundle(geolocation="VN Province, US")) is None


def test_default_config_lists_vietnam():
    import yaml
    from pathlib import Path
    raw = yaml.safe_load((Path(vc.__file__).resolve().parent.parent / "config.default.yaml").read_text(encoding="utf-8"))
    assert raw["exclude_countries"] == ["Vietnam", "China"]
