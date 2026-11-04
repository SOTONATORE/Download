"""Фильтры предложений (CUDA ≥ 12.9, диск ≥ 1500 МБ/с), состояние «ComfyUI умер» и проверка файлов модели в пробе.

Оффлайн: HTTP воркера и Vast подставлены, файлы модели HEAD-запросами не проверяются (клиент httpx подменён).
"""
import logging
import types

try:
    from Generate.core import vast_client as vc
    from Generate.tests import test_removal_rules as rules
    from Generate.tools import vast_probe as vp
except ImportError:
    from core import vast_client as vc  # type: ignore
    from tests import test_removal_rules as rules  # type: ignore
    from tools import vast_probe as vp  # type: ignore


def _bundle(**over):
    """Предложение, которое проходит все проверки, кроме тех, что задаёт тест."""
    b = {
        "id": 1, "dph_total": 0.5, "reliability2": 0.99, "num_gpus": 1, "machine_id": 5,
        "inet_down": 1000, "inet_up_cost": 0.001, "inet_down_cost": 0.001,
        "gpu_ram": 32 * 1024, "gpu_name": "RTX 5090", "disk_space": 200,
        "cuda_max_good": 12.9, "disk_bw": 2000,
    }
    b.update(over)
    return b


def _bundle_without(*keys, **over):
    b = _bundle(**over)
    for key in keys:
        b.pop(key)
    return b


FILTER = vc.OfferFilter(gpu_name="RTX 5090", min_cuda_max=12.9, min_disk_bw_mbps=1500.0)


def _reason(b, f=FILTER):
    return vc.VastClient._check_offer(b, f)[0]


# ---------------------------------------------------------------- фильтр CUDA

def test_cuda_12_8_is_dropped():
    assert _reason(_bundle(cuda_max_good=12.8)) == "cuda"


def test_cuda_12_9_is_kept():
    assert _reason(_bundle(cuda_max_good=12.9)) is None


def test_missing_cuda_field_is_dropped_with_own_reason():
    reason = _reason(_bundle_without("cuda_max_good"))
    assert reason == "cuda_unknown"
    assert "CUDA" in dict(vc._REJECT_LABELS)[reason]


def test_unreadable_cuda_value_is_dropped_as_unknown():
    assert _reason(_bundle(cuda_max_good="n/a")) == "cuda_unknown"


# ---------------------------------------------------------------- фильтр диска

def test_disk_bw_1499_is_dropped():
    assert _reason(_bundle(disk_bw=1499)) == "disk_bw"


def test_disk_bw_1500_is_kept():
    assert _reason(_bundle(disk_bw=1500)) is None


def test_missing_disk_bw_field_is_dropped_with_own_reason():
    reason = _reason(_bundle_without("disk_bw"))
    assert reason == "disk_bw_unknown"
    assert "скорость диска" in dict(vc._REJECT_LABELS)[reason]


def test_default_filter_keeps_offers_without_new_fields():
    """Пороги по умолчанию в OfferFilter выключены (0): старые вызовы без новых полей не меняются."""
    f = vc.OfferFilter(gpu_name="RTX 5090")
    assert _reason(_bundle_without("cuda_max_good", "disk_bw"), f) is None


def test_search_summary_counts_new_reasons(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    bundles = [
        _bundle(id=1),
        _bundle(id=2, cuda_max_good=12.8),
        _bundle(id=3, disk_bw=1499),
        _bundle_without("cuda_max_good", id=4),
    ]
    monkeypatch.setattr(vc.VastClient, "_request",
                        lambda self, method, path, json=None: {"offers": bundles})
    client = object.__new__(vc.VastClient)  # без сети: поиск читает только _request
    result = client.search_offers(FILTER)
    assert [o.offer_id for o in result] == [1]
    line = next(r.getMessage() for r in caplog.records if "Поиск предложений Vast" in r.getMessage())
    assert "получено 4" in line
    assert "CUDA ниже минимума 1" in line
    assert "версия CUDA не разобрана 1" in line
    assert "скорость диска ниже минимума 1" in line
    assert "подходит 1" in line


# ---------------------------------------------------------------- ComfyUI умер (Part 4)

def test_dead_state_removes_card_at_once_before_300_seconds(tmp_path):
    orc = rules._orc(tmp_path, http=rules._HttpStub(
        {"ok": False, "comfy_alive": False, "comfy_state": "dead"}))
    slot = rules._slot()
    orc._probe_health(slot)
    assert slot.comfy_state == "dead"
    # 10 с после аренды, far before 300 s: снятие сразу, с причиной про умерший процесс, bad=True
    assert rules._step_calls(orc, slot, now=10.0) == [("ComfyUI процесс умер", True)]


def test_dead_state_logs_once_per_card_with_card_id(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    orc = rules._orc(tmp_path, http=rules._HttpStub(
        {"ok": False, "comfy_alive": False, "comfy_state": "dead"}))
    slot = rules._slot()
    orc._probe_health(slot)
    orc._probe_health(slot)
    assert caplog.text.count("comfy_state dead") == 1
    assert f"карте {rules.INSTANCE}" in caplog.text          # номер карты (экземпляр Vast) в строке


def test_starting_state_keeps_rule_a_300_seconds(tmp_path):
    orc = rules._orc(tmp_path, http=rules._HttpStub(
        {"ok": False, "comfy_alive": False, "comfy_state": "starting"}))
    slot = rules._slot()
    orc._probe_health(slot)
    assert slot.comfy_state == "starting"
    assert rules._step_calls(orc, slot, now=299.0) == []
    assert rules._step_calls(orc, slot, now=300.0) == [(rules.RULE_A_REASON, True)]


# ---------------------------------------------------------------- проба: проверка файлов модели

class _FakeHfClient:
    """Подмена httpx.Client: HEAD отвечает 200 на адрес huggingface.co, сети нет."""
    urls: list = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def head(self, url, headers=None):
        _FakeHfClient.urls.append(url)
        return types.SimpleNamespace(
            status_code=200,
            headers={"content-length": "1"},
            request=types.SimpleNamespace(url=types.SimpleNamespace(host="huggingface.co"), headers={}),
        )


def test_probe_hf_check_reads_weights_json_and_reports_real_status(monkeypatch):
    _FakeHfClient.urls = []
    monkeypatch.setattr(vp.httpx, "Client", _FakeHfClient)
    state = vp._check_hf_files("token-for-test")
    assert state["status"] == "выполнена"                 # не «ошибка разбора»
    assert state["base_source"] == "weights.json профиля ltx25"
    assert len(_FakeHfClient.urls) == 5                   # по одному HEAD на файл из weights.json
    assert all(u.startswith("https://huggingface.co/Lightricks/LTX-2.5/resolve/main/") for u in _FakeHfClient.urls)
    assert "latent_upscale_models/" in _FakeHfClient.urls[-1]
