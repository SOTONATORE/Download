"""Офлайн-тесты A/B-теста сценариев workflow (cartoon_photos).

Проверяют сборку графов четырёх вариантов, каталог из 16 сцен, неизменность шаблона, разбор аргументов
команды test-workflows и соответствие кодов завершения. Прогон run_ab_test идёт офлайн: клиент Vast,
SSH-туннель и HTTP (httpx.MockTransport) подменяются заглушками; сети нет. Проверяются переменные
аренды, терпимость к сбоям опроса, замена плохой карты с учётом bad_hosts.txt и маскирование артефактов.
"""

from __future__ import annotations

import contextlib
import copy
import json
import types
import uuid

import httpx
import pytest

from Generate import cli
from Generate.core import ssh_probe
from Generate.core import workflow_ab_test as wab
from Generate.core.bad_hosts import BadHostsError
from Generate.core.vast_client import GpuOffer, InstanceInfo, VastInstanceNotFound, VastRuntimeError


@pytest.fixture(scope="module")
def template_and_refs():
    return wab.load_template()


def _scene(scene_id: str) -> wab.Scene:
    return next(s for s in wab.SCENE_CATALOG if s.scene_id == scene_id)


def _graph(template_and_refs, scene_id: str) -> dict:
    template, refs = template_and_refs
    return wab.build_graph(template, refs, _scene(scene_id))


def _loads(graph: dict) -> set[str]:
    return {k for k, v in graph.items() if v.get("class_type") == "LoadImage"}


# --- каталог сцен ---------------------------------------------------------------------------------

def test_catalog_has_16_unique_scenes_and_prompts():
    ids = [s.scene_id for s in wab.SCENE_CATALOG]
    prompts = [s.prompt for s in wab.SCENE_CATALOG]
    assert len(ids) == 16
    assert len(set(ids)) == 16
    assert len(set(prompts)) == 16


def test_catalog_has_four_scenes_per_variant():
    counts: dict[str, int] = {}
    for s in wab.SCENE_CATALOG:
        counts[s.variant] = counts.get(s.variant, 0) + 1
    assert counts == {"A": 4, "B": 4, "C": 4, "D": 4}


def test_hero_flags_only_for_b_and_c_first_two():
    heroes = {s.scene_id for s in wab.SCENE_CATALOG if s.hero}
    assert heroes == {"B1_snow_cliff", "B2_lab_desk", "C1_rain_meadow", "C2_ship_spyglass"}


def test_template_ref_bindings_point_to_matching_load_nodes(template_and_refs):
    _, refs = template_and_refs
    assert refs == {
        "ref_1": "470", "ref_2": "475", "ref_3": "503", "ref_4": "504", "ref_5": "505",
        "ref_6": "506", "ref_7": "507", "ref_8": "508", "ref_9": "509", "ref_10": "510",
    }


def test_template_is_not_mutated_by_builders(template_and_refs):
    template, refs = template_and_refs
    before = copy.deepcopy(template)
    for scene in wab.SCENE_CATALOG:
        wab.build_graph(template, refs, scene)
    assert template == before


# --- общие настройки графа -----------------------------------------------------------------------

@pytest.mark.parametrize("scene_id", [s.scene_id for s in wab.SCENE_CATALOG])
def test_shared_settings_are_kept_for_every_scene(template_and_refs, scene_id):
    g = _graph(template_and_refs, scene_id)
    sampler = g["488"]["inputs"]
    assert sampler["steps"] == 8
    assert sampler["cfg"] == 1
    assert sampler["sampler_name"] == "er_sde"
    assert sampler["scheduler"] == "linear_quadratic"
    assert sampler["seed"] == wab.TESTWF_SEED
    assert g["493"]["inputs"]["unet_name"] == "qwen_image_2.1_turbo_int8_convrot.safetensors"
    assert g["496"]["inputs"]["clip_name"] == "qwen3vl_8b_int8_convrot.safetensors"
    assert g["497"]["inputs"]["vae_name"] == "qwen_image_2.1_vae_bf16.safetensors"
    assert g["490"]["class_type"] == "QwenImage21Cache"
    scale = g["511"]["inputs"]
    assert (scale["width"], scale["height"], scale["upscale_method"]) == (1920, 1080, "lanczos")
    assert g["461"]["class_type"] == "SaveImageAdvanced"
    assert g["487"]["class_type"] == "VAEDecode"


@pytest.mark.parametrize("scene_id", [s.scene_id for s in wab.SCENE_CATALOG])
def test_every_graph_passes_link_validation(template_and_refs, scene_id):
    wab.validate_graph(_graph(template_and_refs, scene_id))


def test_validate_graph_rejects_dangling_link(template_and_refs):
    g = _graph(template_and_refs, "A1_forest_torch")
    g["488"]["inputs"]["latent_image"] = ["999", 0]
    with pytest.raises(wab.GraphError):
        wab.validate_graph(g)


# --- вариант A ----------------------------------------------------------------------------------

@pytest.mark.parametrize("scene_id", ["A1_forest_torch", "A2_cliff_new_world", "A3_coral_reef", "A4_offer_card"])
def test_variant_a_is_pure_t2i(template_and_refs, scene_id):
    g = _graph(template_and_refs, scene_id)
    assert _loads(g) == set()
    inputs = g["491"]["inputs"]
    assert "vae" not in inputs
    assert not any(k.startswith("images.image_") for k in inputs)
    assert g["512"]["class_type"] == "EmptyLatentImage"
    assert g["512"]["inputs"]["width"] == 1920
    assert g["512"]["inputs"]["height"] == 1088
    assert g["488"]["inputs"]["latent_image"] == ["512", 0]


# --- вариант B ----------------------------------------------------------------------------------

@pytest.mark.parametrize("scene_id", ["B1_snow_cliff", "B2_lab_desk"])
def test_variant_b_hero_uses_ref6_with_vae_and_node_latent(template_and_refs, scene_id):
    g = _graph(template_and_refs, scene_id)
    assert _loads(g) == {"506"}
    inputs = g["491"]["inputs"]
    assert inputs["images.image_1"] == ["506", 0]
    assert inputs["vae"] == ["497", 0]
    assert inputs["resolution"] == 0
    assert g["488"]["inputs"]["latent_image"] == ["491", 2]
    assert "512" not in g


@pytest.mark.parametrize("scene_id", ["B3_mars_canyon", "B4_timeline_card"])
def test_variant_b_nonhero_falls_back_to_pure_t2i(template_and_refs, scene_id):
    g = _graph(template_and_refs, scene_id)
    assert _loads(g) == set()
    assert "vae" not in g["491"]["inputs"]
    assert g["488"]["inputs"]["latent_image"] == ["512", 0]
    assert g["512"]["inputs"]["width"] == 1920 and g["512"]["inputs"]["height"] == 1088


# --- вариант C ----------------------------------------------------------------------------------

@pytest.mark.parametrize("scene_id", ["C1_rain_meadow", "C2_ship_spyglass"])
def test_variant_c_hero_is_vlm_only_without_vae(template_and_refs, scene_id):
    g = _graph(template_and_refs, scene_id)
    assert _loads(g) == {"506"}
    inputs = g["491"]["inputs"]
    assert inputs["images.image_1"] == ["506", 0]
    assert "vae" not in inputs
    assert g["488"]["inputs"]["latent_image"] == ["512", 0]
    assert g["512"]["inputs"]["width"] == 1920 and g["512"]["inputs"]["height"] == 1088


@pytest.mark.parametrize("scene_id", ["C3_volcano_island", "C4_surfshark_card"])
def test_variant_c_nonhero_falls_back_to_pure_t2i(template_and_refs, scene_id):
    g = _graph(template_and_refs, scene_id)
    assert _loads(g) == set()
    assert "vae" not in g["491"]["inputs"]
    assert g["488"]["inputs"]["latent_image"] == ["512", 0]


# --- вариант D ----------------------------------------------------------------------------------

def test_variant_d1_d2_have_ref1_and_ref6_without_vae(template_and_refs):
    for scene_id in ("D1_jungle_flashlight", "D2_campfire_night"):
        g = _graph(template_and_refs, scene_id)
        assert _loads(g) == {"470", "506"}
        inputs = g["491"]["inputs"]
        assert inputs["images.image_1"] == ["470", 0]
        assert inputs["images.image_2"] == ["506", 0]
        assert "vae" not in inputs
        assert g["488"]["inputs"]["latent_image"] == ["512", 0]


def test_variant_d3_has_only_ref1(template_and_refs):
    g = _graph(template_and_refs, "D3_drought_flood")
    assert _loads(g) == {"470"}
    assert g["491"]["inputs"]["images.image_1"] == ["470", 0]
    assert "images.image_2" not in g["491"]["inputs"]
    assert "vae" not in g["491"]["inputs"]


def test_variant_d4_has_only_ref2_in_slot_one(template_and_refs):
    g = _graph(template_and_refs, "D4_ice_age_card")
    assert _loads(g) == {"475"}
    assert g["491"]["inputs"]["images.image_1"] == ["475", 0]
    assert "vae" not in g["491"]["inputs"]
    assert g["488"]["inputs"]["latent_image"] == ["512", 0]


# --- аргументы CLI ------------------------------------------------------------------------------

def test_parser_test_workflows_defaults():
    args = cli.build_parser().parse_args(["test-workflows"])
    assert args.command == "test-workflows"
    assert args.config is None
    assert args.budget_limit is None
    assert args.min_balance_usd is None
    assert args.rental_type is None
    assert args.verified is None
    assert args.scan_sec is None
    assert args.docker_image is None
    assert args.worker_build == ""
    assert args.run_id == ""
    assert args.artifact_dir == ""
    assert args.debug_logs is False


def test_parser_test_workflows_types():
    args = cli.build_parser().parse_args([
        "test-workflows", "--verified", "1", "--rental-type", "on-demand", "--scan-sec", "300",
        "--disk-gb", "100", "--budget-limit", "2.0", "--silent-host-timeout-min", "15", "--debug-logs",
    ])
    assert args.verified is True
    assert args.rental_type == "on-demand"
    assert args.scan_sec == 300
    assert args.disk_gb == 100
    assert args.budget_limit == 2.0
    assert args.silent_host_timeout_min == 15
    assert args.debug_logs is True


def test_parser_rejects_bad_verified():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["test-workflows", "--verified", "2"])


# --- соответствие кодов завершения (без сети) ----------------------------------------------------

class _FakeVast:
    def __init__(self, debug_logs: bool = False) -> None:
        self.debug_logs = debug_logs


class _FakeSummary:
    def __init__(self, ok: bool) -> None:
        self.ok = ok
        self.ok_count = 16 if ok else 3
        self.results = [None] * 16
        self.label = "testwf-123"


def _patch_cli(monkeypatch, run_impl):
    captured: dict = {}

    def fake_run(vast, cfg, **kwargs):
        captured.update(kwargs)
        return run_impl()

    fake_module = types.SimpleNamespace(
        MODEL_PROFILE="cartoon_photos", ABInputError=wab.ABInputError, ABRuntimeError=wab.ABRuntimeError,
        run_ab_test=fake_run)
    monkeypatch.setattr(cli, "VastClient", _FakeVast)
    monkeypatch.setattr(cli, "_vast_preflight", lambda vast, min_balance_usd: None)
    monkeypatch.setattr(cli, "_install_stop_handlers", lambda stop: {})
    monkeypatch.setattr(cli, "_restore_handlers", lambda previous: None)
    monkeypatch.setattr(cli, "_wab", lambda: fake_module)
    return captured


def test_cmd_test_workflows_returns_ok_when_all_scenes_ready(monkeypatch):
    captured = _patch_cli(monkeypatch, lambda: _FakeSummary(ok=True))
    args = cli.build_parser().parse_args(["test-workflows", "--run-id", "123", "--worker-build", "a" * 40])
    assert cli.cmd_test_workflows(args) == cli.EXIT_OK
    assert captured["run_id"] == "123"
    assert captured["worker_build"] == "a" * 40


def test_cmd_test_workflows_maps_input_error_to_exit_2(monkeypatch):
    def boom():
        raise wab.ABInputError("плохой вход")

    _patch_cli(monkeypatch, boom)
    args = cli.build_parser().parse_args(["test-workflows"])
    assert cli.cmd_test_workflows(args) == cli.EXIT_INPUT


def test_cmd_test_workflows_maps_runtime_error_to_exit_3(monkeypatch):
    def boom():
        raise wab.ABRuntimeError("карта не ответила")

    _patch_cli(monkeypatch, boom)
    args = cli.build_parser().parse_args(["test-workflows"])
    assert cli.cmd_test_workflows(args) == cli.EXIT_RUNTIME


def test_cmd_test_workflows_partial_result_is_runtime_exit(monkeypatch):
    _patch_cli(monkeypatch, lambda: _FakeSummary(ok=False))
    args = cli.build_parser().parse_args(["test-workflows"])
    assert cli.cmd_test_workflows(args) == cli.EXIT_RUNTIME


# --- прогон run_ab_test без сети: заглушки Vast, SSH и HTTP (httpx.MockTransport) -----------------

SSH_KEY = "-----BEGIN OPENSSH PRIVATE KEY-----\nAAAAFAKEKEYMATERIAL0123456789\n-----END OPENSSH PRIVATE KEY-----"
BUILD = "a" * 40


def _cfg(**over):
    values = dict(
        rental_type="on-demand", interruptible=False, gpu_name="RTX 5090", min_price_per_hour=0.1,
        max_price_per_hour=5.0, min_reliability=0.9, min_inet_mbps=100.0, max_inet_price_per_tb=4.1,
        verified=False, disk_gb=100, min_cuda_max=12.0, min_disk_bw_mbps=0.0, exclude_countries=[],
        worker_idle_timeout_min=60, budget_limit_usd=100.0, silent_host_timeout_min=15, bid_margin=0.03,
        max_rent_tries=5, min_balance_usd=0.0)
    values.update(over)
    return types.SimpleNamespace(**values)


def _offer(offer_id: int, machine_id: int, price: float) -> GpuOffer:
    return GpuOffer(offer_id=offer_id, gpu_name="RTX 5090", num_gpus=1, vram_gb=32.0, price_per_hr=price,
                    reliability=0.99, inet_down_mbps=1000.0, inet_up_mbps=1000.0, machine_id=machine_id,
                    is_interruptible=False, dph_total=price, disk_gb=100.0)


def _ready_info(ip: str, *, port: int = 40123, dph: float = 0.5) -> InstanceInfo:
    raw = {
        "id": 0, "actual_status": "running", "intended_status": "running", "public_ipaddr": ip,
        "ssh_host": "ssh.example.net", "ssh_port": 2222, "direct_port_start": 40100,
        "extra_env": [["WORKER_TOKEN", "t"]], "instance_api_key": "k-123456789",
        "ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": str(port)}],
                  "18188/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40124"}],
                  "22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "2222"}]},
    }
    return InstanceInfo(instance_id=0, actual_status="running", intended_status="running", public_ip=ip,
                        direct_port=port, ssh_host="ssh.example.net", ssh_port=2222, dph=dph,
                        disk_gb=100.0, raw=raw)


class _RunVast:
    """Заглушка VastClient: предложения, аренда, сценарии опроса по предложению. Сети нет."""

    def __init__(self, offers, *, create_fails=(), polls=None):
        self.offers = list(offers)
        self.create_fails = set(create_fails)
        self.polls = dict(polls or {})   # offer_id -> очередь ответов get_instance (InstanceInfo или исключение)
        self.created: list = []          # все попытки create_instance по порядку
        self.destroyed: list = []        # instance_id, снятые через destroy_instance
        self.instances: dict = {}        # instance_id -> offer_id
        self._seq = 1000

    def search_offers(self, flt):
        return list(self.offers)

    def create_instance(self, offer_id, image, disk_gb, env_vars=None, label=None, onstart=None, bid_price=None):
        self.created.append({"offer_id": offer_id, "env": dict(env_vars or {}), "bid": bid_price, "label": label})
        if offer_id in self.create_fails:
            raise VastRuntimeError("отказ аренды (тест)")
        self._seq += 1
        self.instances[self._seq] = offer_id
        return self._seq

    def get_instance(self, instance_id):
        queue = self.polls[self.instances[instance_id]]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    def destroy_instance(self, instance_id):
        self.destroyed.append(instance_id)
        return True

    def destroy_by_label(self, label):
        return []  # страховка по метке; проверки идут через destroy_instance

    def request_logs(self, instance_id, *, tail=2000, daemon_logs=False):
        raise VastRuntimeError("логи недоступны (тест)")

    def download_logs(self, url, dest):
        return 404


def _scene_handler(request):
    """Заглушка ComfyUI: очередь, history с картинкой, /view с байтами."""
    path = request.url.path
    if request.method == "POST" and path == "/prompt":
        return httpx.Response(200, json={"prompt_id": uuid.uuid4().hex}, request=request)
    if path.startswith("/history/"):
        pid = path.rsplit("/", 1)[1]
        outputs = {"461": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}
        return httpx.Response(200, json={pid: {"status": {"status_str": "success"}, "outputs": outputs}},
                              request=request)
    if path == "/view":
        return httpx.Response(200, content=b"\x89PNG-fake", request=request)
    return httpx.Response(404, request=request)


def _install_http(monkeypatch, health_by_ip: dict) -> None:
    real_client = httpx.Client

    def fake_client(**kwargs):
        return real_client(transport=httpx.MockTransport(_scene_handler), **kwargs)

    def fake_get(url, headers=None, timeout=None, params=None):
        req = httpx.Request("GET", url)
        ip = url.split("//", 1)[1].rsplit(":", 1)[0]
        if url.endswith("/health"):
            data = health_by_ip.get(ip, {"ok": True, "comfy_alive": True, "comfy_state": "alive"})
            return httpx.Response(200, json=data, request=req)
        return httpx.Response(404, request=req)

    monkeypatch.setattr(httpx, "Client", fake_client)
    monkeypatch.setattr(httpx, "get", fake_get)


def _install_ssh(monkeypatch, output: str) -> None:
    @contextlib.contextmanager
    def fake_tunnel(host, port, key_text, remote_port, ready_timeout_sec=None):
        yield 45678

    def fake_probe(host, port, key_text, command_name, *args, **kwargs):
        return types.SimpleNamespace(ok=True, output=output, failure_class="ok")

    monkeypatch.setattr(ssh_probe, "open_tunnel", fake_tunnel)
    monkeypatch.setattr(ssh_probe, "run_probe", fake_probe)


def _setup(monkeypatch, *, health_by_ip=None, output: str = "") -> None:
    monkeypatch.setenv("VAST_SSH_KEY", SSH_KEY)
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setattr(wab, "POLL_SEC", 0.0)
    monkeypatch.setattr(wab, "load_bad_hosts", lambda path=None: frozenset())
    _install_http(monkeypatch, health_by_ip or {})
    _install_ssh(monkeypatch, output)


def _run(vast, cfg, artifact_dir, **kwargs):
    return wab.run_ab_test(vast, cfg, run_id="42", worker_build=BUILD, repo="owner/repo",
                           artifact_dir=str(artifact_dir), **kwargs)


def test_run_create_env_has_port_mapping_build_and_idle_timeout(monkeypatch, tmp_path):
    _setup(monkeypatch)
    vast = _RunVast([_offer(1, 11, 0.5)], polls={1: [_ready_info("203.0.113.1")]})
    cfg = _cfg()
    summary = _run(vast, cfg, tmp_path)
    assert summary.ok
    env = vast.created[0]["env"]
    assert env["-p 8000:8000"] == "1"
    assert env["WORKER_BUILD"] == BUILD
    assert env["WORKER_IDLE_TIMEOUT_MIN"] == str(cfg.worker_idle_timeout_min)


def test_run_tolerates_polling_errors_below_not_found_limit(monkeypatch, tmp_path):
    _setup(monkeypatch)
    polls = {1: [VastInstanceNotFound("нет в ответе (тест)"), VastInstanceNotFound("нет в ответе (тест)"),
                 VastRuntimeError("сбой API (тест)"), _ready_info("203.0.113.1")]}
    vast = _RunVast([_offer(1, 11, 0.5)], polls=polls)
    summary = _run(vast, _cfg(), tmp_path)
    assert summary.ok
    assert len(vast.created) == 1          # одна аренда: короткие сбои опроса не сняли карту
    assert vast.destroyed == []            # карта не снималась при сбоях опроса


def test_run_skips_bad_hosts_and_replaces_failing_cards(monkeypatch, tmp_path):
    _setup(monkeypatch, health_by_ip={"203.0.113.3": {"ok": False, "comfy_alive": False, "comfy_state": "dead"}})
    monkeypatch.setattr(wab, "load_bad_hosts", lambda path=None: frozenset({101}))
    offers = [_offer(1, 101, 0.5), _offer(2, 202, 0.6), _offer(3, 303, 0.7), _offer(4, 404, 0.8)]
    polls = {3: [_ready_info("203.0.113.3")], 4: [_ready_info("203.0.113.4")]}
    vast = _RunVast(offers, create_fails={2}, polls=polls)
    summary = _run(vast, _cfg(), tmp_path)
    assert summary.ok
    # машина 101 из bad_hosts.txt не арендуется; предложение 2 не создалось и взято следующее (3);
    # карта 3 снята из-за comfy_state dead, взята 4
    assert [c["offer_id"] for c in vast.created] == [2, 3, 4]
    dead_card = [iid for iid, oid in vast.instances.items() if oid == 3]
    assert vast.destroyed == dead_card
    retired = (tmp_path / "cards" / "slot1_generation1" / "status.txt").read_text(encoding="utf-8")
    assert retired.startswith("снята:")   # поколение 1 (карта 3) помечено как снятое


def test_run_artifacts_mask_addresses_and_keep_internal_ports(monkeypatch, tmp_path):
    tail = ("boot step ok\n"
            "connect 203.0.113.7:40123 via ssh.example.net:2222\n"
            "key " + SSH_KEY + "\n"
            "listen 0.0.0.0:8000\n"
            "host ssh.example.net ready, external port 40123 open\n")  # голые значения без пары host:port
    _setup(monkeypatch, output=tail)
    vast = _RunVast([_offer(1, 11, 0.5)], polls={1: [_ready_info("203.0.113.7")]})
    summary = _run(vast, _cfg(), tmp_path)
    assert summary.ok

    inst_text = (tmp_path / "run" / "instance.json").read_text(encoding="utf-8")
    inst = json.loads(inst_text)
    assert "ports" in inst
    assert set(inst["ports"]) <= {"8000/tcp", "18188/tcp"}
    assert set(inst["ports"].values()) == {"***"}
    for key in ("public_ipaddr", "ssh_host", "ssh_port", "direct_port_start", "extra_env", "instance_api_key"):
        assert key not in inst
    for secret in ("203.0.113.7", "ssh.example.net", "k-123456789"):
        assert secret not in inst_text

    tail_text = (tmp_path / "run" / "worker_bootstrap_tail.txt").read_text(encoding="utf-8")
    for secret in ("AAAAFAKEKEYMATERIAL0123456789", "203.0.113.7", "ssh.example.net", "40123", "2222"):
        assert secret not in tail_text
    assert "0.0.0.0:8000" in tail_text     # внутренний порт воркера сохраняется


def test_run_bad_hosts_format_error_is_input_error_before_rent(monkeypatch, tmp_path):
    _setup(monkeypatch)

    def broken(path=None):
        raise BadHostsError("неверный формат файла bad_hosts.txt (тест)")

    monkeypatch.setattr(wab, "load_bad_hosts", broken)
    vast = _RunVast([_offer(1, 11, 0.5)], polls={1: [_ready_info("203.0.113.1")]})
    with pytest.raises(wab.ABInputError):
        _run(vast, _cfg(), tmp_path)
    assert vast.created == []              # до аренды ничего не потрачено
