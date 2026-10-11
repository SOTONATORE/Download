"""Офлайн-тесты клиента Vast (SPEC 0.1, 0.2, 0.5). Сеть не используется."""
import json
import pathlib
import re
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


def bundle(i, price=0.5, rel=0.99, down=3000.0, ram=24576, name="RTX 5090", bid=None):
    d = {"id": i, "gpu_name": name, "num_gpus": 1, "gpu_ram": ram, "dph_total": price,
         "reliability2": rel, "inet_down": down, "inet_up": 1000.0, "machine_id": 100 + i}
    if bid is not None:
        d["min_bid"] = bid
    return d


def test_exit_codes():
    assert VastError.exit_code == 1
    assert VastInputError.exit_code == 2
    assert VastRuntimeError.exit_code == 3


def test_offer_filter_defaults():
    f = OfferFilter()
    assert f.gpu_name == "RTX 5090"
    assert f.min_price == 0.35
    assert f.max_price == 0.9


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
            bundle(2, price=0.4),
            bundle(3, price=1.5),            # дорого
            bundle(4, rel=0.5),              # ненадёжный
            bundle(5, down=100.0),           # медленная сеть
            bundle(6, price=0.6, ram=8192),  # мало VRAM
            bundle(7, price=0.1),            # подозрительно дёшево
            bundle(8, price=0.5, name="RTX 4090"),  # другая карта
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
    assert q["dph_total"] == {"lte": 0.9, "gte": 0.35}
    assert q["reliability2"] == {"gte": 0.95}
    assert q["inet_down"] == {"gte": 2000.0}
    assert q["type"] == "bid"
    assert q["gpu_ram"] == {"gte": 16 * 1024}
    assert q["gpu_name"] == {"eq": "RTX 5090"}


def test_search_min_price_serialization_custom(key):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"offers": []})

    make(handler).search_offers(OfferFilter(min_price=0.5, max_price=1.0))
    assert seen["body"]["dph_total"] == {"lte": 1.0, "gte": 0.5}


def test_search_min_price_zero_omits_gte(key):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"offers": []})

    make(handler).search_offers(OfferFilter(min_price=0.0))
    assert seen["body"]["dph_total"] == {"lte": 0.9}


def test_search_filters_cheaper_than_min_price(key):
    c = make(lambda r: httpx.Response(200, json={"offers": [
        bundle(1, price=0.10),
        bundle(2, price=0.34),
        bundle(3, price=0.35),   # граница включается
        bundle(4, price=0.60),
    ]}))
    offers = c.search_offers(OfferFilter())
    assert [o.offer_id for o in offers] == [3, 4]
    assert all(o.price_per_hr >= 0.35 for o in offers)


def test_search_min_price_applies_to_min_bid(key):
    c = make(lambda r: httpx.Response(200, json={"offers": [
        bundle(1, price=0.8, bid=0.2),   # ставка ниже минимума
        bundle(2, price=0.8, bid=0.5),
    ]}))
    assert [o.offer_id for o in c.search_offers(OfferFilter())] == [2]


def test_search_sorted_ascending(key):
    c = make(lambda r: httpx.Response(200, json={"offers": [
        bundle(1, price=0.8), bundle(2, price=0.4), bundle(3, price=0.6), bundle(4, price=0.35),
    ]}))
    prices = [o.price_per_hr for o in c.search_offers(OfferFilter())]
    assert prices == sorted(prices) == [0.35, 0.4, 0.6, 0.8]


def test_search_gpu_name_none_allows_any_gpu(key):
    c = make(lambda r: httpx.Response(200, json={"offers": [
        bundle(1, price=0.5, name="RTX 4090"), bundle(2, price=0.6),
    ]}))
    offers = c.search_offers(OfferFilter(gpu_name=None))
    assert [o.offer_id for o in offers] == [1, 2]


def test_search_excludes_multi_gpu_offer(key):
    c = make(lambda r: httpx.Response(200, json={"offers": [
        {**bundle(1, price=0.5), "num_gpus": 2},
        bundle(2, price=0.6),
    ]}))
    assert [o.offer_id for o in c.search_offers(OfferFilter())] == [2]


def test_search_includes_single_gpu_offer(key):
    c = make(lambda r: httpx.Response(200, json={"offers": [
        {**bundle(1, price=0.5), "num_gpus": 1},
    ]}))
    offers = c.search_offers(OfferFilter())
    assert [o.offer_id for o in offers] == [1]
    assert offers[0].num_gpus == 1 and offers[0].gpu_name == "RTX 5090"


def test_search_excludes_offer_without_num_gpus(key, caplog):
    caplog.set_level("INFO")
    no_field = bundle(1, price=0.5)
    del no_field["num_gpus"]
    c = make(lambda r: httpx.Response(200, json={"offers": [no_field, bundle(2, price=0.6)]}))
    assert [o.offer_id for o in c.search_offers(OfferFilter())] == [2]
    # по одной строке на поиск, только счётчики: номера предложений не выводятся (SPEC 10)
    assert "Пропущено предложение" not in caplog.text
    assert "получено 2" in caplog.text and "не одиночная карта 1" in caplog.text
    assert "подходит 1" in caplog.text
    assert SECRET not in caplog.text


def test_search_query_requires_single_gpu(key):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"offers": []})

    make(handler).search_offers(OfferFilter())
    assert seen["body"]["num_gpus"] == {"eq": 1}


def test_search_uses_min_bid_for_interruptible(key):
    c = make(lambda r: httpx.Response(200, json={"offers": [bundle(1, price=2.0, bid=0.4)]}))
    assert c.search_offers(OfferFilter())[0].price_per_hr == 0.4


def test_search_invalid_filter(key):
    c = make(lambda r: httpx.Response(200, json={"offers": []}))
    with pytest.raises(VastInputError):
        c.search_offers(OfferFilter(max_price=0))
    with pytest.raises(VastInputError):
        c.search_offers(OfferFilter(min_reliability=1.5))


def test_search_invalid_min_price(key):
    c = make(lambda r: httpx.Response(200, json={"offers": []}))
    with pytest.raises(VastInputError) as e1:
        c.search_offers(OfferFilter(min_price=-0.1))
    assert e1.value.exit_code == 2
    with pytest.raises(VastInputError):
        c.search_offers(OfferFilter(min_price=1.0, max_price=0.9))
    # min_price == max_price допустимо
    assert c.search_offers(OfferFilter(min_price=0.9, max_price=0.9)) == []


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


# --- баланс аккаунта (предстартовая проверка, SPEC 8.3 (е)) ---

def test_balance_normal_value(key):
    seen = {}

    def handler(req):
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["auth"] = req.headers["Authorization"]
        seen["query"] = str(req.url.query)
        return httpx.Response(200, json={"id": 1, "email": "x@example.com", "balance": 12.5})

    c = make(handler)
    assert c.get_balance_usd() == 12.5
    assert seen["method"] == "GET" and seen["path"] == "/api/v0/users/current/"
    assert seen["auth"] == f"Bearer {SECRET}"
    assert SECRET not in seen["query"]


def test_balance_integer_and_negative_values(key):
    # целое число допустимо и возвращается как float; отрицательный баланс (задолженность) не ошибка разбора
    v = make(lambda r: httpx.Response(200, json={"balance": 5})).get_balance_usd()
    assert v == 5.0 and isinstance(v, float)
    assert make(lambda r: httpx.Response(200, json={"balance": -2.5})).get_balance_usd() == -2.5


@pytest.mark.parametrize("payload", [
    {"id": 1},                              # поля нет
    {"id": 1, "balance": None},
    {"id": 1, "balance": "12.5"},           # строка: схема Vast описывает число
    {"id": 1, "balance": "много"},
    {"id": 1, "balance": True},             # bool не считается числом
])
def test_balance_non_numeric_or_missing_raises(key, payload):
    c = make(lambda r: httpx.Response(200, json=payload))
    with pytest.raises(VastRuntimeError) as e:
        c.get_balance_usd()
    # ошибка, а не тихий ноль: неизвестный баланс не должен пропускать запуск
    assert e.value.exit_code == 3
    assert "баланс" in str(e.value).lower()


def test_balance_nan_is_rejected(key):
    # Python разбирает NaN из JSON, но баланс NaN не принимаем: сравнение с порогом было бы ложным
    c = make(lambda r: httpx.Response(200, content=b'{"id": 1, "balance": NaN}'))
    with pytest.raises(VastRuntimeError) as e:
        c.get_balance_usd()
    assert e.value.exit_code == 3


def test_balance_unexpected_shape_raises(key):
    c = make(lambda r: httpx.Response(200, json=[12.5]))
    with pytest.raises(VastRuntimeError) as e:
        c.get_balance_usd()
    assert e.value.exit_code == 3


@pytest.mark.parametrize("status", [401, 500])
def test_balance_http_error_raises(key, status):
    c = make(lambda r: httpx.Response(status, text="error"))
    with pytest.raises(VastRuntimeError) as e:
        c.get_balance_usd()
    assert e.value.exit_code == 3 and str(status) in str(e.value)


def test_balance_error_does_not_leak_key(key):
    c = make(lambda r: httpx.Response(401, text=f"bad {r.headers['Authorization']}"))
    with pytest.raises(VastRuntimeError) as e:
        c.get_balance_usd()
    assert SECRET not in str(e.value)


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


# --- пути API и перенаправления ---
# Регрессия: прод упал на 301 (баланс) и 410 (список экземпляров). Старые тесты проверяли
# только тело ответа, а не точный путь, поэтому сбой не был виден. Здесь путь проверяется точно.

SRT = pathlib.Path(__file__).resolve().parent / "data" / "final.srt"


def _ru(text: str) -> bool:
    return re.search(r"[А-Яа-яЁё]", text) is not None


def test_endpoint_paths_are_exact(key):
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path))
        p = req.url.path
        if p == "/api/v0/bundles/":
            return httpx.Response(200, json={"offers": []})
        if p == "/api/v0/users/current/":
            return httpx.Response(200, json={"balance": 1.0})
        if p == "/api/v1/instances/":
            return httpx.Response(200, json={"instances": []})
        if p == "/api/v0/asks/42/":
            return httpx.Response(200, json={"success": True, "new_contract": 7})
        if p == "/api/v0/instances/7/":
            if req.method == "GET":
                return httpx.Response(200, json={"instances": inst(7)})
            return httpx.Response(200, json={"success": True})
        return httpx.Response(404, text="нет такого пути")

    c = make(handler)
    c.search_offers(OfferFilter())
    c.get_balance_usd()
    c.list_instances()
    c.create_instance(42, "img", 10)
    c.get_instance(7)
    c.destroy_instance(7)
    assert calls == [
        ("POST", "/api/v0/bundles/"),
        ("GET", "/api/v0/users/current/"),
        ("GET", "/api/v1/instances/"),
        ("PUT", "/api/v0/asks/42/"),
        ("GET", "/api/v0/instances/7/"),
        ("DELETE", "/api/v0/instances/7/"),
    ]


def test_list_uses_v1_params(key):
    seen = []

    def handler(req):
        seen.append((req.url.path, dict(req.url.params)))
        return httpx.Response(200, json={"instances": []})

    make(handler).list_instances()
    assert len(seen) == 1
    path, params = seen[0]
    assert path == "/api/v1/instances/"
    assert json.loads(params["select_filters"]) == {}
    assert json.loads(params["order_by"]) == [{"col": "id", "dir": "asc"}]
    assert params["limit"] == "25" and "after_token" not in params


@pytest.mark.parametrize("code", [301, 302, 307, 308])
def test_redirect_is_russian_error_without_key(key, code):
    def handler(req):
        # сервер назвал новый адрес, в котором случайно оказался ключ: он должен замаскироваться
        return httpx.Response(
            code,
            headers={"location": f"https://console.vast.ai/api/v0/users/current/?x={SECRET}"},
            text="moved",
        )

    with pytest.raises(VastRuntimeError) as e:
        make(handler).get_balance_usd()
    text = str(e.value)
    assert e.value.exit_code == 3
    assert f"перенаправлением HTTP {code}" in text
    assert "GET /api/v0/users/current/" in text
    assert "https://console.vast.ai/api/v0/users/current/" in text
    assert _ru(text)
    assert SECRET not in text


def test_410_is_russian_error_naming_endpoint_without_key(key):
    def handler(req):
        return httpx.Response(410, json={
            "success": False, "error": "deprecated_endpoint",
            "msg": "/api/v0/instances/ is deprecated. Use /api/v1/instances/ instead.",
            "echo": SECRET,
        })

    with pytest.raises(VastRuntimeError) as e:
        make(handler).check_instances_endpoint()
    text = str(e.value)
    assert e.value.exit_code == 3
    assert "устарела (HTTP 410)" in text and "GET /api/v1/instances/" in text
    assert _ru(text) and SECRET not in text


def test_check_instances_endpoint_is_one_read_only_request(key):
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path))
        return httpx.Response(200, json={"instances": [], "next_token": "more"})

    make(handler).check_instances_endpoint()
    # одна страница, без следования next_token, без записи
    assert seen == [("GET", "/api/v1/instances/")]


@pytest.mark.parametrize("status", [301, 410, 500])
def test_check_instances_endpoint_raises_on_bad_status(key, status):
    c = make(lambda r: httpx.Response(status, text="x"))
    with pytest.raises(VastRuntimeError) as e:
        c.check_instances_endpoint()
    assert e.value.exit_code == 3 and "GET /api/v1/instances/" in str(e.value)


def test_list_follows_next_token_across_pages(key):
    seen = []

    def handler(req):
        seen.append(dict(req.url.params))
        if "after_token" not in req.url.params:
            return httpx.Response(200, json={"instances": [inst(1, label="gen-a")],
                                             "next_token": "t1"})
        return httpx.Response(200, json={"instances": [inst(2, label="gen-b")],
                                         "next_token": None})

    rows = make(handler).list_instances()
    assert [r.instance_id for r in rows] == [1, 2]
    assert seen[1]["after_token"] == "t1"


def test_destroy_by_label_covers_all_pages(key):
    deleted = []

    def handler(req):
        if req.method == "DELETE":
            deleted.append(req.url.path)
            return httpx.Response(200, json={"success": True})
        if "after_token" not in req.url.params:
            return httpx.Response(200, json={"instances": [inst(1, label="gen-a")],
                                             "next_token": "t1"})
        return httpx.Response(200, json={"instances": [inst(3, label="gen-a")],
                                         "next_token": None})

    assert make(handler).destroy_by_label("gen-a") == [1, 3]
    assert deleted == ["/api/v0/instances/1/", "/api/v0/instances/3/"]


def test_list_page_failure_aborts_without_partial_kill(key):
    """Вторая страница упала: ни одна карта не уничтожается по неполному списку."""
    deleted = []

    def handler(req):
        if req.method == "DELETE":
            deleted.append(req.url.path)
            return httpx.Response(200, json={"success": True})
        if "after_token" not in req.url.params:
            return httpx.Response(200, json={"instances": [inst(1, label="gen-a")],
                                             "next_token": "t1"})
        return httpx.Response(410, text="gone")

    with pytest.raises(VastRuntimeError) as e:
        make(handler).destroy_by_label("gen-a")
    assert e.value.exit_code == 3
    assert deleted == []


def test_repeated_next_token_is_error(key):
    c = make(lambda r: httpx.Response(200, json={"instances": [inst(1)], "next_token": "same"}))
    with pytest.raises(VastRuntimeError) as e:
        c.list_instances()
    assert e.value.exit_code == 3


def test_missing_instances_key_is_error_not_empty_list(key):
    """Без списка нельзя считать, что карт нет: уборка должна упасть, а не молча пройти."""
    c = make(lambda r: httpx.Response(200, json={"next_token": None}))
    with pytest.raises(VastRuntimeError) as e:
        c.destroy_by_label("gen-a")
    assert e.value.exit_code == 3


def test_v1_ports_as_json_string_is_parsed(key):
    d = inst(9)
    d["ports"] = json.dumps(d["ports"])
    assert VastClient._parse_instance(d).direct_port == 34567
    d["ports"] = "не json"
    assert VastClient._parse_instance(d).direct_port is None


# --- предстартовая проверка в run: остановка до поиска и аренды (cli.py) ---

@pytest.fixture
def cli_run(monkeypatch):
    """Внешнее для run подменено: оркестратор считает вызовы, Gemini и Release не трогаются."""
    from Generate import cli
    orchestrator_calls = []
    monkeypatch.setattr(cli, "_build_timing", lambda cfg, profile: "T")
    monkeypatch.setattr(cli, "_build_prompt_config", lambda *a: "P")
    monkeypatch.setattr(cli, "_make_transport", lambda: "X")
    monkeypatch.setattr(cli, "_make_release_store", lambda repo: "S")

    def fake_run(*a, **k):
        orchestrator_calls.append(1)
        import types
        return types.SimpleNamespace(exit_code=0)

    monkeypatch.setattr(cli, "run_generation", fake_run)
    return cli, orchestrator_calls


def _cli_run(cli, monkeypatch, tmp_path, handler):
    monkeypatch.setattr(cli, "VastClient", lambda *a, **k: make(handler))
    m = tmp_path / "missing.txt"
    m.write_text("1\n", encoding="utf-8")
    return cli.main(["run", "--srt", str(SRT), "--missing", str(m)])


def test_preflight_list_410_stops_before_search_and_rental(key, monkeypatch, tmp_path,
                                                           capsys, cli_run):
    cli, orchestrator_calls = cli_run
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path))
        return httpx.Response(410, json={"success": False, "error": "deprecated_endpoint"})

    code = _cli_run(cli, monkeypatch, tmp_path, handler)
    cap = capsys.readouterr()
    assert code == 3
    assert "Предстартовая проверка Vast API не пройдена" in cap.err
    assert "GET /api/v1/instances/" in cap.err and "карты не арендованы" in cap.err
    assert seen == [("GET", "/api/v1/instances/")]       # ни баланса, ни поиска, ни аренды
    assert orchestrator_calls == []
    assert SECRET not in cap.err + cap.out


def test_preflight_balance_301_stops_before_search(key, monkeypatch, tmp_path, capsys, cli_run):
    cli, orchestrator_calls = cli_run
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path))
        if req.url.path == "/api/v1/instances/":
            return httpx.Response(200, json={"instances": []})
        if req.url.path == "/api/v0/users/current/":
            # прод: без слэша сервер отвечает 301 на адрес со слэшем
            return httpx.Response(301, headers={"location": "https://console.vast.ai/api/v0/users/current/"})
        return httpx.Response(200, json={"offers": []})

    code = _cli_run(cli, monkeypatch, tmp_path, handler)
    cap = capsys.readouterr()
    assert code == 3
    assert "GET /api/v0/users/current/" in cap.err
    # порядок: список, баланс, и больше ничего (ни /bundles/, ни /asks/)
    assert seen == [("GET", "/api/v1/instances/"), ("GET", "/api/v0/users/current/")]
    assert orchestrator_calls == []


def test_preflight_low_balance_stops_before_search(key, monkeypatch, tmp_path, capsys, cli_run):
    cli, orchestrator_calls = cli_run
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path))
        if req.url.path == "/api/v1/instances/":
            return httpx.Response(200, json={"instances": []})
        return httpx.Response(200, json={"balance": 1.0})

    assert _cli_run(cli, monkeypatch, tmp_path, handler) == 2
    assert [p for _, p in seen] == ["/api/v1/instances/", "/api/v0/users/current/"]
    assert orchestrator_calls == []


def test_preflight_all_ok_continues_to_orchestrator(key, monkeypatch, tmp_path, cli_run):
    cli, orchestrator_calls = cli_run

    def handler(req):
        if req.url.path == "/api/v1/instances/":
            return httpx.Response(200, json={"instances": []})
        return httpx.Response(200, json={"balance": 12.5})

    assert _cli_run(cli, monkeypatch, tmp_path, handler) == 0
    assert orchestrator_calls == [1]
