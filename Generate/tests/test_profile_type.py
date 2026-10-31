"""Каждый профиль модели в models/ обязан явно задавать top-level type: video или image.

Тест не использует PyYAML и сеть: читает profile.yaml как текст.
"""
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parents[1] / "model_profiles" / "models"


def _top_level_type(text: str):
    for line in text.splitlines():
        if line.startswith("type:"):
            return line.split(":", 1)[1].split("#", 1)[0].strip().strip("\"'")
    return None


def _profile_folders():
    return sorted(p for p in MODELS_DIR.iterdir() if p.is_dir())


@pytest.mark.parametrize("folder", _profile_folders(), ids=lambda p: p.name)
def test_profile_declares_type(folder):
    profile = folder / "profile.yaml"
    assert profile.is_file(), f"профиль «{folder.name}»: нет файла profile.yaml"
    value = _top_level_type(profile.read_text(encoding="utf-8"))
    assert value in ("video", "image"), (
        f"профиль «{folder.name}»: в profile.yaml нет top-level type: video или image "
        f"(найдено: {value!r})")
