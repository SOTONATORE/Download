"""Офлайн-тесты клиента Vast (SPEC 0.1, 0.2, 0.5). Сеть не используется."""
import json
import pathlib
import socket
import traceback

import httpx
import pytest

try:
    from Generate.core import vast_client as vc
    from Generate.core.vast_client import (
        GpuOffer, InstanceInfo, OfferFilter, VastClient, VastError,
        VastInputError, VastRuntimeError,
    )
except ImportError:
    from core import vast_client as vc
    from core.vast_client import (
        GpuOffer, InstanceInfo, OfferFilter, VastClient, VastError,
        VastInputError, VastRuntimeError,
    )

ENV_KEY_NAME = "GEN_" + "VAST_API_KEY"
SECRET = "TESTKEY-12345"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def _blocked(*a, **k):
        raise AssertionError("Сетевое соединение в офлайн-тесте запрещено")
    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv(ENV_KEY_NAME, SECRET)


def make(handler):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return VastClient(client=http)


def bundle(i, price=0.5, rel=0.99, down=3000.0, ram=24576, name="RTX 4090", bid=None):
    d = {"id": i, "gpu_name": name, "num_gpus": 1, "gpu_ram": ram, "dph_total": price,
         "reliability2": rel, "inet_down": down, "inet_up": 1000.0, "machine_id": 100 + i}
    if bid is not None:
        d["min_bid"] = bid
    return d


def test_exit_codes():
    assert VastError.exit_code == 1
    assert VastInputError.exit_code == 2
    assert VastRuntimeError.exit_code == 3


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv(ENV_KEY_NAME, raising=False)
    with pytest.raises(VastInputError) as e:
        VastClient()
    assert e.value.exit_code == 2


def test_repr_and_context(key):
    c = make(lambda r: httpx.Response(200, json={}))
    assert repr(c) == "VastClient()"
    assert SECRET not in repr(c) and SECRET not in str(c)
    with c:
        pass


def test_search_serialization_filter_sort(key):
    seen = {}

    def handler(req):
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["query"] = str(req.url.query)
        seen["auth"] = req.headers["Authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"offers": [
            bundle(1, price=0.8),
            bundle(2, price=0.3),
            bundle(3, price=1.5),            # дорого
            bundle(4, rel=0.5),              # ненадёжный
            bundle(5, down=100.0),           # медленная сеть
            bundle(6, price=0.6, ram=8192),  # мало VRAM
        ]})

    with make(handler) as c:
        offers = c.search_offers(OfferFilter(min_vram_gb=16))
    assert [o.offer_id for o in offers] == [2, 1]
    assert isinstance(offers[0], GpuOffer)
    assert offers[0].vram_gb == 24.0 and offers[0].is_interruptible
    assert seen["method"] == "POST" and seen["path"].endswith("/bundles/")
    assert seen["auth"] == f"Bearer {SECRET}"
    assert SECRET not in seen["query"]
    q = seen["body"]
    assert q["dph_total"] == {"lte": 0.9}
    assert q["reliability2"] == {"gte": 0.95}
    assert q["inet_down"] == {"gte": 2000.0}
    assert q["type"] == "bid"
    assert q["gpu_ram"] == {"gte": 16 * 1024}


def test_search_uses_min_bid_for_interruptible(key):
    c = make(lambda r: httpx.Response(200, json={"offers": [bundle(1, price=2.0, bid=0.4)]}))
    assert c.search_offers(OfferFilter())[0].price_per_hr == 0.4


def test_search_invalid_filter(key):
    c = make(lambda r: httpx.Response(200, json={"offers": []}))
    with pytest.raises(VastInputError):
        c.search_offers(OfferFilter(max_price=0))
    with pytest.raises(VastInputError):
        c.search_offers(OfferFilter(min_reliability=1.5))


def test_create_instance_payload(key):
    seen = {}

    def handler(req):
        seen["method"], seen["path"] = req.method, req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"success": True, "new_contract": 777})

    c = make(handler)
    iid = c.create_instance(42, "img:latest", 50, env_vars={"WORKER_TOKEN": "t"},
                            label="gen-run1", onstart="echo hi")
    assert iid == 777
    assert seen["method"] == "PUT" and seen["path"].endswith("/asks/42/")
    b = seen["body"]
    assert b["client_id"] == "me" and b["image"] == "img:latest" and b["disk"] == 50
    assert b["env"] == {"WORKER_TOKEN": "t"}
    assert b["label"] == "gen-run1" and b["onstart"] == "echo hi"


def test_create_instance_bad_response(key):
    c = make(lambda r: httpx.Response(200, json={"success": False}))
    with pytest.raises(VastRuntimeError):
        c.create_instance(1, "img", 10)


def inst(i, status="running", label="gen-a", ports=True):
    d = {"id": i, "actual_status": status, "intended_status": "running", "label": label,
         "public_ipaddr": "1.2.3.4", "ssh_host": "ssh.vast", "ssh_port": 2222, "dph_total": 0.4}
    if ports:
        d["ports"] = {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "34567"}]}
    return d


def test_get_instance_running_and_loading(key):
    c = make(lambda r: httpx.Response(200, json={"instances": inst(5)}))
    info = c.get_instance(5)
    assert isinstance(info, InstanceInfo)
    assert info.actual_status == "running" and info.public_ip == "1.2.3.4"
    assert info.direct_port == 34567 and info.ssh_port == 2222 and info.dph == 0.4

    c2 = make(lambda r: httpx.Response(200, json={"instances": inst(6, "loading", ports=False)}))
    info2 = c2.get_instance(6)
    assert info2.actual_status == "loading" and info2.direct_port is None


def test_destroy_and_destroy_by_label(key):
    deleted = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"instances": [
                inst(1, label="gen-a"), inst(2, label="gen-b"), inst(3, label="gen-a")]})
        deleted.append((req.method, req.url.path))
        return httpx.Response(200, json={"success": True})

    c = make(handler)
    assert c.destroy_instance(9) is True
    assert [i.instance_id for i in c.list_instances("gen-a")] == [1, 3]
    assert c.destroy_by_label("gen-a") == [1, 3]
    assert ("DELETE", "/api/v0/instances/3/") in deleted or any(
        p.endswith("/instances/3/") for _, p in deleted)
    assert not any(p.endswith("/instances/2/") for _, p in deleted)
    with pytest.raises(VastInputError):
        c.destroy_by_label("")


def test_destroy_by_label_partial_failure(key):
    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"instances": [inst(1), inst(2)]})
        if req.url.path.endswith("/1/"):
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"success": True})

    with pytest.raises(VastRuntimeError) as e:
        make(handler).destroy_by_label("gen-a")
    assert "[1]" in str(e.value)


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors(key, status):
    c = make(lambda r: httpx.Response(status, text="error"))
    with pytest.raises(VastRuntimeError) as e:
        c.list_instances()
    assert e.value.exit_code == 3 and str(status) in str(e.value)


def test_network_errors(key):
    def timeout(req):
        raise httpx.ReadTimeout("slow", request=req)

    def conn(req):
        raise httpx.ConnectError("down", request=req)

    for h in (timeout, conn):
        with pytest.raises(VastRuntimeError) as e:
            make(h).list_instances()
        assert e.value.exit_code == 3


def test_key_never_leaks(key, caplog, capsys):
    caplog.set_level("DEBUG")
    texts = []

    def ok(req):
        return httpx.Response(200, json={"instances": [inst(1)], "offers": []})

    def echo(req):
        return httpx.Response(401, text=f"bad header {req.headers['Authorization']}")

    def neterr(req):
        raise httpx.ConnectError(f"failed headers={dict(req.headers)}", request=req)

    c = make(ok)
    c.list_instances()
    c.search_offers(OfferFilter())
    texts += [repr(c), str(c)]

    for h in (echo, neterr):
        cc = make(h)
        with pytest.raises(VastRuntimeError) as e:
            cc.list_instances()
        texts.append(str(e.value))
        texts.append(repr(e.value))
        texts.append("".join(traceback.format_exception(e.type, e.value, e.tb)))

    out = capsys.readouterr()
    texts += [caplog.text, out.out, out.err]
    for t in texts:
        assert SECRET not in t


def test_source_has_no_secret_names():
    names = ("GEN_" + "GEMINI_API_KEY", "GEN_" + "VAST_API_KEY")
    here = pathlib.Path(__file__).resolve()
    for p in (here, pathlib.Path(vc.__file__)):
        code = "\n".join(line.split("#", 1)[0] for line in p.read_text(encoding="utf-8").splitlines())
        for n in names:
            assert n not in code, f"{p.name}: литерал {n}"
