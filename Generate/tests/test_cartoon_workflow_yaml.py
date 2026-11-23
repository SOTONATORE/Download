"""Офлайн-проверка YAML-конфигураций workflow и профиля cartoon/i2v (без сети)."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from Generate.core import config_loader as cl

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
GENERATE_CARTOON = WORKFLOWS / "generate-cartoon.yml"
VERIFY_PROMPTS = WORKFLOWS / "verify_prompts.yml"
I2V_DIR = ROOT / "Generate" / "Nishes" / "cartoon" / "i2v"
MAX_DISPATCH_INPUTS = 25


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dispatch_inputs(data: dict) -> dict:
    on = data.get("on", data.get(True))
    assert isinstance(on, dict)
    return (on.get("workflow_dispatch") or {}).get("inputs") or {}


def test_generate_cartoon_workflow_yaml_and_inputs():
    assert GENERATE_CARTOON.is_file()
    data = _load_yaml(GENERATE_CARTOON)
    inputs = _dispatch_inputs(data)
    assert 1 <= len(inputs) <= MAX_DISPATCH_INPUTS
    assert inputs["mode"]["options"] == ["partial_i2v", "photo_only"]
    assert inputs["mode"]["default"] == "partial_i2v"
    assert float(inputs["intro_minutes"]["default"]) == 3.0
    assert float(inputs["intro_video_percent"]["default"]) == 70.0
    assert float(inputs["after_intro_video_percent"]["default"]) == 10.0
    assert float(inputs["min_anim_sec"]["default"]) == 2.0

    steps_text = GENERATE_CARTOON.read_text(encoding="utf-8")
    assert "--i2v-mode" in steps_text
    assert "watch-heartbeat" in steps_text
    assert "kill-cards" in steps_text
    assert "finalize-release" in steps_text


def test_verify_prompts_workflow_yaml_and_cartoon_i2v_inputs():
    assert VERIFY_PROMPTS.is_file()
    data = _load_yaml(VERIFY_PROMPTS)
    inputs = _dispatch_inputs(data)
    assert 1 <= len(inputs) <= MAX_DISPATCH_INPUTS
    assert "cartoon" in inputs["niche"]["options"]
    assert "cartoon_photos" not in inputs["niche"]["options"]
    assert inputs["cartoon_i2v"]["type"] == "boolean"
    assert inputs["cartoon_i2v"]["default"] is False
    assert float(inputs["intro_minutes"]["default"]) == 3.0
    assert float(inputs["intro_video_percent"]["default"]) == 70.0
    assert float(inputs["after_intro_video_percent"]["default"]) == 10.0
    assert float(inputs["min_anim_sec"]["default"]) == 2.0


def test_cartoon_i2v_profile_weights_bindings_and_workflow():
    assert cl.profile_dir("cartoon_i2v") == I2V_DIR
    assert cl.load_profile_type("cartoon_i2v") == "video"
    assert cl.load_profile_requires_input_image("cartoon_i2v") is True
    assert cl.load_profile_frame_rule("cartoon_i2v") == {"kind": "8k+1"}

    profile = _load_yaml(I2V_DIR / "profile.yaml")
    assert profile["name"] == "cartoon_i2v"
    assert profile["type"] == "video"
    assert profile["requires_input_image"] is True
    assert profile["frame_rule"] == {"kind": "8k+1"}

    weights = json.loads((I2V_DIR / "weights.json").read_text(encoding="utf-8"))
    assert len(weights["files"]) == 5
    assert {f["name"] for f in weights["files"]} == set(profile["model_files"].values())

    bindings_path = I2V_DIR / "bindings.json"
    assert bindings_path.is_file()
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    for key in ("prompt", "num_frames", "seed", "filename_prefix", "input_image"):
        assert key in bindings
        assert "node_id" in bindings[key] and "field" in bindings[key]

    workflow_path = I2V_DIR / "workflow.json"
    if workflow_path.is_file():
        graph = json.loads(workflow_path.read_text(encoding="utf-8"))
        assert isinstance(graph, dict) and graph
        out_id = str(profile["workflow"]["output"]["node_id"])
        assert out_id in graph
        for key, spec in bindings.items():
            nid = str(spec["node_id"])
            assert nid in graph, f"узел {nid} для {key} отсутствует в workflow.json"
            assert spec["field"] in graph[nid]["inputs"], (
                f"поле {spec['field']} для {key} отсутствует в узле {nid}"
            )
