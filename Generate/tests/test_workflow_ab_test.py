"""Офлайн-тесты A/B-теста сценариев workflow (cartoon_photos).

Проверяют сборку графов четырёх вариантов, каталог из 16 сцен, неизменность шаблона, разбор аргументов
команды test-workflows и соответствие кодов завершения. Сети нет, Vast не вызывается: клиент и модуль
теста подменяются заглушками. Тесты написаны, но в песочнице не запускались (по указанию задачи).
"""

from __future__ import annotations

import copy
import types

import pytest

from Generate import cli
from Generate.core import workflow_ab_test as wab


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
