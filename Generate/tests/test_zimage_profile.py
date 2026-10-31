"""Профиль Z-Image-Turbo: граф, привязки и profile.yaml согласованы.

Проверяются только файлы профиля. Модель, сеть и Gemini не вызываются.
Если PyYAML не установлен, проверки, которым нужен profile.yaml, пропускаются.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

GEN = pathlib.Path(__file__).resolve().parents[1]
MODEL = GEN / "model_profiles" / "models" / "zimage_turbo"
PROFILE = MODEL / "profile.yaml"
SETUP = GEN / "worker" / "setup.sh"
WORKFLOW = MODEL / "workflow.json"
BINDINGS = MODEL / "bindings.json"

# Ключи, которые есть только у видео-профиля ltx25 и не должны попасть в профиль изображения
LTX_ONLY_KEYS = ("frame_rule", "clip_min_sec", "clip_max_sec", "gpu_max_sec", "tail_pad_sec")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _profile():
    # Без PyYAML проверки, которым нужен profile.yaml, пропускаются с понятным сообщением
    if yaml is None:
        pytest.skip("PyYAML не установлен в этом окружении: проверка profile.yaml пропущена")
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def test_a_bindings_reference_existing_nodes_and_fields():
    graph = _json(WORKFLOW)
    bindings = _json(BINDINGS)
    for name, b in bindings.items():
        node_id = str(b["node_id"])
        assert node_id in graph, f"привязка «{name}»: в графе нет узла {node_id}"
        assert b["field"] in graph[node_id]["inputs"], (
            f"привязка «{name}»: в узле {node_id} нет поля {b['field']}")


def test_b_output_node_is_saveimage():
    profile = _profile()
    graph = _json(WORKFLOW)
    node_id = str(profile["workflow"]["output"]["node_id"])
    assert graph[node_id]["class_type"] == "SaveImage"


def test_c_model_file_names_match_workflow():
    mf = _profile()["model_files"]
    graph = _json(WORKFLOW)
    assert graph["57:28"]["inputs"]["unet_name"] == mf["unet"]
    assert graph["57:30"]["inputs"]["clip_name"] == mf["clip"]
    assert graph["57:29"]["inputs"]["vae_name"] == mf["vae"]


def test_d_prompt_is_empty_placeholder():
    graph = _json(WORKFLOW)
    assert graph["57:27"]["inputs"]["text"] == ""
    for node_id, node in graph.items():
        text = node.get("inputs", {}).get("text")
        if isinstance(text, str):
            assert text == "", f"в узле {node_id} остался текст промпта"


def test_e_resolution_is_1920x1080():
    inputs = _json(WORKFLOW)["57:13"]["inputs"]
    assert inputs["width"] == 1920
    assert inputs["height"] == 1080


def test_f_no_ltx_only_keys_in_profile():
    profile = _profile()
    for key in LTX_ONLY_KEYS:
        assert key not in profile, f"в профиле изображения не должно быть ключа {key}"


# Пути в репозитории Comfy-Org/z_image_turbo, по которым setup.sh скачивает веса Z-Image
ZIMAGE_REPO_PATHS = (
    "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
    "split_files/text_encoders/qwen_3_4b.safetensors",
    "split_files/vae/ae.safetensors",
)


def test_g_setup_zimage_files_match_profile():
    # Список загрузки Z-Image берётся из weights.json (его читает setup.sh через weights_plan.py files)
    weights = json.loads((MODEL / "weights.json").read_text(encoding="utf-8"))
    files = weights["files"]
    names_in_json = [entry["name"] for entry in files]
    paths_in_json = [entry["repo_path"] for entry in files]
    mf = _profile()["model_files"]
    assert names_in_json == [mf["unet"], mf["clip"], mf["vae"]]
    assert paths_in_json == list(ZIMAGE_REPO_PATHS)
    for entry in files:
        assert entry["repo_path"].startswith("split_files/")
