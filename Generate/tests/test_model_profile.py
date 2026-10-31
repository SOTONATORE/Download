"""Фаза D: профиль модели ltx25 как единый источник значений, сверка с графом и setup.sh,
случайный сид второго прохода. Пути берутся от папки models/ltx25 внутри model_profiles."""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

import pytest
import yaml

GEN = pathlib.Path(__file__).resolve().parents[1]
ROOT = GEN.parent
MODEL = GEN / "model_profiles" / "models" / "ltx25"
PROFILE = MODEL / "profile.yaml"
WORKFLOW = MODEL / "workflow.json"
BINDINGS = MODEL / "bindings.json"
STYLE = MODEL / "prompts" / "style.md"
SETUP = GEN / "worker" / "setup.sh"
CONFIG_DEFAULT = GEN / "config.default.yaml"

try:
    from Generate.core import config_loader as cl
    from Generate.worker import comfy_client as cc
except ImportError:  # pragma: no cover
    from core import config_loader as cl  # type: ignore
    from worker import comfy_client as cc  # type: ignore

try:
    from Generate import cli
except ImportError:  # pragma: no cover
    cli = None


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- пути профиля

def test_profile_paths_point_to_the_moved_files():
    p = _yaml(PROFILE)
    assert ROOT / p["workflow"]["file"] == WORKFLOW
    assert ROOT / p["workflow"]["bindings_file"] == BINDINGS
    assert ROOT / p["prompt_style"] == STYLE
    assert all(f.is_file() for f in (PROFILE, WORKFLOW, BINDINGS, STYLE))
    assert cl.profile_dir("ltx25") == MODEL
    assert cl.profile_exists("ltx25") and not cl.profile_exists("no_such_profile")


# ---------------------------------------------- значения профиля = код = граф

def test_profile_values_match_graph_and_config():
    p = _yaml(PROFILE)
    cfg = _yaml(CONFIG_DEFAULT)
    g = _json(WORKFLOW)
    inp = lambda nid: g[nid]["inputs"]  # noqa: E731

    # fps: config.default.yaml, профиль и два PrimitiveNode графа (узлы 30 и 34)
    assert p["fps"] == cfg["fps"] == inp("30")["value"] == inp("34")["value"]
    # тайминги клипа и хвост: профиль = config.default.yaml (код читает config)
    assert p["clip_min_sec"] == cfg["clip_min_sec"]
    assert p["clip_max_sec"] == cfg["clip_max_sec"]
    assert p["tail_pad_sec"] == cfg["tail_pad_sec"]
    # итоговый размер: профиль = config.default.yaml = ImageScale (узел 36)
    assert p["target_resolution"] == cfg["target_resolution"] == f"{inp('36')['width']}x{inp('36')['height']}"
    # первый проход 576x320, апскейл x2
    assert p["params"]["resolution"] == f"{2 * inp('32')['value']}x{2 * inp('29')['value']}"
    # cfg: LTXVDualCFGGuider, узлы 7 и 15, video_cfg и audio_cfg
    for nid in ("7", "15"):
        assert p["params"]["cfg"] == inp(nid)["video_cfg"] == inp(nid)["audio_cfg"]
    # sampler: KSamplerSelect, узлы 6 и 16
    assert {inp("6")["sampler_name"], inp("16")["sampler_name"]} == {p["params"]["sampler"]}
    # quantization: входит в имена файлов UNET и CLIP
    assert p["params"]["quantization"] in p["model_files"]["unet"]
    assert p["params"]["quantization"] in p["model_files"]["clip"]
    # число кадров (узел 31) удовлетворяет правилу 8k+1
    assert (inp("31")["value"] - 1) % 8 == 0
    # gpu_max_sec: в конфиге null, значит берётся из профиля
    assert cfg["gpu_max_sec"] is None
    assert cl.load_profile_number("ltx25", "gpu_max_sec") == float(p["gpu_max_sec"]) > 0


def test_frame_rule_is_read_from_profile_and_matches_old_behaviour():
    assert _yaml(PROFILE)["frame_rule"] == {"kind": "8k+1"}
    assert cl.load_profile_frame_rule("ltx25") == {"kind": "8k+1"}
    if cli is None:
        pytest.skip("пакет cli не импортируется в этом окружении")
    assert cli._frame_rule_for("ltx25") == {"kind": "8k+1"}
    # профиля нет: правило запасного профиля (прежний 8k+1)
    assert cli._frame_rule_for("no_such_profile") == {"kind": "8k+1"}
    assert not hasattr(cli, "TIMING_FRAME_RULE")


def test_single_model_constant_used_in_place_of_literal():
    assert cl.DEFAULT_MODEL_PROFILE == "ltx25"
    # литерал профиля как значение в коде: допустим только в объявлении константы
    literal = re.compile(r'(?:=|\bor)\s*"ltx25"')
    files = ["cli.py", "core/orchestrator.py", "core/weights_log.py", "core/config_loader.py",
             "core/model_constants.py"]
    hits = {}
    for rel in files:
        lines = (GEN / rel).read_text(encoding="utf-8").splitlines()
        hits[rel] = [n + 1 for n, line in enumerate(lines)
                     if literal.search(line) and not line.lstrip().startswith("#")]
    decl = "core/model_constants.py"
    assert hits[decl] == [
        next(n + 1 for n, line in enumerate((GEN / decl).read_text(encoding="utf-8").splitlines())
             if line.startswith("DEFAULT_MODEL_PROFILE"))]
    assert all(hits[rel] == [] for rel in files if rel != decl)


# ------------------------------------------------ setup.sh = граф = профиль

_DIR_BY_NODE = {"UNETLoader": "diffusion_models", "CLIPLoader": "text_encoders",
                "VAELoader": "vae", "LatentUpscaleModelLoader": "latent_upscale_models"}
_KEY_BY_NODE = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name",
                "VAELoader": "vae_name", "LatentUpscaleModelLoader": "model_name"}


def test_setup_download_names_equal_workflow_and_profile_names():
    # Список загрузки LTX берётся из weights.json (его читает setup.sh через weights_plan.py files)
    weights = json.loads((MODEL / "weights.json").read_text(encoding="utf-8"))
    profile_files = _yaml(PROFILE)["model_files"]
    ltx_names = set(profile_files.values())
    setup_names = [entry["name"] for entry in weights["files"]]
    setup_pairs = {(entry["subdir"], entry["name"]) for entry in weights["files"]}
    assert len(setup_names) == len(profile_files) == 5  # 5 файлов LTX в weights.json
    assert set(setup_names) == ltx_names
    graph_pairs = set()
    for node in _json(WORKFLOW).values():
        cls = node.get("class_type")
        if cls in _DIR_BY_NODE:
            graph_pairs.add((_DIR_BY_NODE[cls], node["inputs"][_KEY_BY_NODE[cls]]))
    assert setup_pairs == graph_pairs
    assert set(setup_names) == set(_yaml(PROFILE)["model_files"].values())


# ------------------------------------------------ сид второго прохода

def test_second_pass_seed_random_per_run_first_pass_bound_and_file_unchanged():
    before = _sha(WORKFLOW)
    graph = _json(WORKFLOW)
    bind = _json(BINDINGS)
    bound = str(bind["seed"]["node_id"])
    randoms = sorted(nid for nid, n in graph.items() if n.get("class_type") == "RandomNoise")
    assert randoms == ["17", "9"] and bound == "17"
    unbound = next(n for n in randoms if n != bound)

    g1 = cc.inject_workflow(graph, bind, "промпт", 97, 7, "clip_1")
    g2 = cc.inject_workflow(graph, bind, "промпт", 97, 7, "clip_2")

    # первый проход (привязанный сид) не меняется и равен заданному
    assert g1[bound]["inputs"]["noise_seed"] == 7 == g2[bound]["inputs"]["noise_seed"]
    # второй проход: новый случайный 32-битный сид на каждый запуск
    s1 = g1[unbound]["inputs"]["noise_seed"]
    s2 = g2[unbound]["inputs"]["noise_seed"]
    assert all(0 <= s < 2 ** 32 for s in (s1, s2))
    assert s1 != s2
    # исходный граф в памяти и файл на диске не изменены
    assert graph[unbound]["inputs"]["noise_seed"] == 42
    assert _sha(WORKFLOW) == before

    # явный сид второго прохода принимается и попадает в граф
    g3 = cc.inject_workflow(graph, bind, "промпт", 97, 7, "clip_3", second_seed=123)
    assert g3[unbound]["inputs"]["noise_seed"] == 123 and g3[bound]["inputs"]["noise_seed"] == 7
    with pytest.raises(cc.ComfyError):
        cc.inject_workflow(graph, bind, "промпт", 97, 7, "clip_4", second_seed=2 ** 32)
