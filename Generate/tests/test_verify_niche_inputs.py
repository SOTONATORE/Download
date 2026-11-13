"""Входы ниши и выбора моделей в verify_prompts.yml совпадают с папками профилей и ниш.

Варианты niche = папки Generate/Nishes с profile.yaml type: image; варианты photo_profile = none и
папки models с type: image (ниши туда не входят); text_model по умолчанию auto.
Проверка env-выражений слабая: только подстрока inputs.niche в строке переменной.
"""
import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "verify_prompts.yml"
MODELS_DIR = ROOT / "Generate" / "model_profiles" / "models"
NISHES_DIR = ROOT / "Generate" / "Nishes"
CARTOON_DIR = NISHES_DIR / "cartoon_photos"


def _dispatch_inputs():
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    on = data.get("on", data.get(True))  # PyYAML читает ключ on как True
    return on["workflow_dispatch"]["inputs"]


def _choice(name):
    inp = _dispatch_inputs()[name]
    assert inp["type"] == "choice", name
    return set(inp["options"]), inp["default"]


def _declared_type(folder):
    # Тип из profile.yaml только если поле явно задано; без profile.yaml или поля — None
    path = folder / "profile.yaml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("type") if isinstance(data, dict) else None


def _folders_with_type(root, wanted_type):
    return {p.name for p in root.iterdir()
            if p.is_dir() and _declared_type(p) == wanted_type}


def test_niche_options_match_niche_folders():
    # Варианты niche без «none» совпадают с нишами на диске: папки Nishes с profile.yaml type: image
    options, _ = _choice("niche")
    assert "none" in options
    assert options - {"none"} == _folders_with_type(NISHES_DIR, "image")


def test_photo_profile_options_have_no_niches():
    # Фото-профили: «none» плюс папки models с type: image; имена ниш в списке не должны быть
    options, _ = _choice("photo_profile")
    assert options == {"none"} | _folders_with_type(MODELS_DIR, "image")
    assert not (options & _folders_with_type(NISHES_DIR, "image"))


def test_text_model_default_is_auto():
    options, default = _choice("text_model")
    assert default == "auto"
    assert "auto" in options


def test_niche_profile_has_required_keys():
    data = yaml.safe_load((CARTOON_DIR / "profile.yaml").read_text(encoding="utf-8"))
    assert data["type"] == "image"
    assert data["allow_text"] is True
    assert data["refs_enabled"] is True
    assert data["on_unfixed"] == "fail"
    assert data["text_model"] == "nemotron"
    assert str(data["workflow"]["output"]["node_id"]) == "461"


def test_niche_has_prompt_files():
    assert (CARTOON_DIR / "profile.yaml").is_file()
    style = CARTOON_DIR / "prompts" / "style.md"
    assert style.is_file()
    first_line = style.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("TEMPORARY VERIFY STUB")


def test_niche_env_expressions_mention_niche_input():
    # Слабая проверка: только подстрока inputs.niche в строке переменной env (не значения выражения).
    # TEXT_MODEL сюда не входит: его выражение зависит от inputs.text_model (auto), не от ниши.
    text = WORKFLOW.read_text(encoding="utf-8")
    for name in ("VIDEO_MODEL", "VIDEO_PERCENT", "PHOTO_MODEL", "PHOTO_PROFILE"):
        line = re.search(rf"^\s*{name}: .*$", text, re.MULTILINE)
        assert line is not None, name
        assert "inputs.niche" in line.group(0), name
