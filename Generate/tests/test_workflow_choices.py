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


def _profile_type(folder):
    # Тип профиля из profile.yaml; если поля нет, профиль считается видео (как в коде)
    path = folder / "profile.yaml"
    if not path.is_file():
        return "video"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("type", "video")


def test_model_profile_options_match_model_folders():
    options, default = _choice("model_profile")
    # Сверяем только видео-профили; профили изображений проверяет test_zimage_profile.py
    folders = {p.name for p in MODELS_DIR.iterdir()
               if p.is_dir() and _profile_type(p) == "video"}
    assert options == folders
    assert options == {"ltx25"}
    assert default in options


def test_preset_options_match_preset_files():
    options, default = _choice("preset")
    files = {p.stem for p in PRESETS_DIR.glob("*.md")} - NOT_PRESETS
    assert options == {"none"} | files
    assert default in options


# S2a–S2d: выбор видео- и фото-модели в generate.yml

FAIL_FAST_MESSAGE = (
    "Обе модели выключены (video_model=none, photo_model=none). "
    "Выберите хотя бы одну модель."
)


def _profile_folders(wanted_type):
    # Папки профилей с нужным типом; без поля type профиль считается видео (как в коде)
    return {p.name for p in MODELS_DIR.iterdir()
            if p.is_dir() and _profile_type(p) == wanted_type}


def test_model_inputs_have_expected_options():
    video_options, _ = _choice("video_model")
    photo_options, _ = _choice("photo_model")
    assert video_options == {"ltx25", "none"}
    assert photo_options == {"zimage_turbo", "none"}


def test_model_inputs_have_expected_defaults():
    _, video_default = _choice("video_model")
    _, photo_default = _choice("photo_model")
    assert video_default == "ltx25"
    assert photo_default == "zimage_turbo"


def test_model_input_options_match_profile_types():
    # Без «none» варианты должны совпадать с профилями на диске своего типа
    video_options, _ = _choice("video_model")
    photo_options, _ = _choice("photo_model")
    assert video_options - {"none"} == _profile_folders("video")
    assert photo_options - {"none"} == _profile_folders("image")


def test_fail_fast_step_contains_russian_message():
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["generate"]["steps"]
    # Проверка стоит первым шагом job, до Checkout и до аренды карт
    first_run = steps[0].get("run", "")
    assert FAIL_FAST_MESSAGE in first_run
    assert "exit 1" in first_run
