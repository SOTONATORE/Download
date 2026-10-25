"""Варианты выбора GitHub (generate.yml) совпадают с папками профилей и файлами пресетов.

Добавили папку профиля или файл пресета без строки в YAML (или строку без папки) — тест падает.
README в папке пресетов — описание, не пресет: в список выбора он не входит.
"""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "generate.yml"
MODELS_DIR = ROOT / "Generate" / "model_profiles" / "models"
PRESETS_DIR = ROOT / "Generate" / "model_profiles" / "presets" / "prompts"
NOT_PRESETS = {"README"}


def _dispatch_inputs():
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    on = data.get("on", data.get(True))  # PyYAML читает ключ on как True
    return on["workflow_dispatch"]["inputs"]


def _choice(name):
    inp = _dispatch_inputs()[name]
    assert inp["type"] == "choice", name
    return set(inp["options"]), inp["default"]


def test_model_profile_options_match_model_folders():
    options, default = _choice("model_profile")
    folders = {p.name for p in MODELS_DIR.iterdir() if p.is_dir()}
    assert options == folders
    assert options == {"ltx25"}
    assert default in options


def test_preset_options_match_preset_files():
    options, default = _choice("preset")
    files = {p.stem for p in PRESETS_DIR.glob("*.md")} - NOT_PRESETS
    assert options == {"none"} | files
    assert options == {"none"}
    assert default in options
