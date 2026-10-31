"""Тесты стиля промптов для фото-модели Z-Image-Turbo и выбора файла стиля в cli.

Тесты написаны для pytest, но в этом шаге не запускаются. Реальные файлы
профилей не трогаются: тип профиля подменяется, каталог стилей указывает во временную папку.
"""
import re
from pathlib import Path

import pytest

from Generate import cli
from Generate.core import config_loader

GENERATE_DIR = Path(__file__).resolve().parents[1]
ZIMAGE_STYLE = GENERATE_DIR / "model_profiles" / "models" / "zimage_turbo" / "prompts" / "style.md"

# русские основы, которых не должно быть в тексте стиля (проверка без учёта регистра)
FORBIDDEN_RU_STEMS = (
    "карт", "календар", "документ", "график", "диаграмм",
    "инфограф", "флаг", "герб", "газет", "удостоверен", "паспорт",
)


def _style_body(text: str) -> str:
    """Текст без HTML-комментария в начале файла (там русская справка для владельца)."""
    return re.sub(r"<!--.*?-->", "", text, count=1, flags=re.DOTALL)


class _FakePromptConfig:
    """Заглушка класса PromptConfig: запоминает аргументы конструктора."""

    def __init__(self, style_file, batch_size, preset, media=None):
        # media принимается, как в настоящем PromptConfig (cli передаёт его при сборке)
        self.style_file = style_file
        self.batch_size = batch_size
        self.preset = preset
        self.media = media


@pytest.fixture
def fake_cls(monkeypatch):
    monkeypatch.setattr(cli, "_find_symbol", lambda name, modules: _FakePromptConfig)
    return _FakePromptConfig


def write_profile(root, name, type_value):
    """Создаёт root/<name>/profile.yaml; type_value None означает файл без строки type:."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f"name: {name}"]
    if type_value is not None:
        # строка type: на верхнем уровне, без отступа
        lines.append(f"type: {type_value}")
    (folder / "profile.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_zimage_style_file_checks():
    assert ZIMAGE_STYLE.is_file()
    text = ZIMAGE_STYLE.read_text(encoding="utf-8")
    assert len(text.encode("utf-8")) < 4096
    assert "1920" in text or "16:9" in text
    assert "no text" in text.lower()
    # запрещённые в промпте вещи должны быть перечислены в разделе Avoid
    assert "metaphor" in text.lower()
    assert "8K" in text


def test_zimage_style_body_has_no_forbidden_ru_stems():
    body = _style_body(ZIMAGE_STYLE.read_text(encoding="utf-8")).lower()
    for stem in FORBIDDEN_RU_STEMS:
        assert stem not in body, f"запрещённая основа в тексте стиля: {stem}"
    # английское слово documents разрешено только во фразе «no maps, documents or flags»
    assert "document" not in body.replace("no maps, documents or flags", "")


def test_image_profile_uses_own_style_path(monkeypatch, tmp_path, fake_cls):
    monkeypatch.setattr(cli, "STYLE_DIR", str(tmp_path))
    # profile_dir строит путь как MODEL_PROFILES_DIR / "models" / name, поэтому профили кладём в подпапку models
    profiles_root = tmp_path / "models"
    monkeypatch.setattr(config_loader, "MODEL_PROFILES_DIR", profiles_root)
    write_profile(profiles_root / "models", "fake_image", "image")
    own = tmp_path / "fake_image" / "prompts"
    own.mkdir(parents=True)
    (own / "style.md").write_text("style", encoding="utf-8")

    cfg = cli._build_prompt_config("fake_image", 5, "none")

    assert cfg.style_file == f"{tmp_path}/fake_image/prompts/style.md"
    assert cfg.batch_size == 5 and cfg.preset == "none"


def test_video_profile_builds_config(monkeypatch, tmp_path, fake_cls):
    monkeypatch.setattr(cli, "STYLE_DIR", str(tmp_path))
    profiles_root = tmp_path / "models"
    monkeypatch.setattr(config_loader, "MODEL_PROFILES_DIR", profiles_root)
    write_profile(profiles_root / "models", "fake_video", "video")

    cfg = cli._build_prompt_config("fake_video", 3, "none")

    assert cfg.media == "video", "media must be read from type: video in profile.yaml"


def test_missing_type_raises(monkeypatch, tmp_path, fake_cls):
    monkeypatch.setattr(cli, "STYLE_DIR", str(tmp_path))
    profiles_root = tmp_path / "models"
    monkeypatch.setattr(config_loader, "MODEL_PROFILES_DIR", profiles_root)
    write_profile(profiles_root / "models", "fake_missing_type", None)

    with pytest.raises(cli.GeminiInputError) as info:
        cli._build_prompt_config("fake_missing_type", 3, "none")

    message = str(info.value)
    assert "fake_missing_type" in message
    assert "type" in message


def test_image_profile_missing_style_raises_with_path_and_profile(monkeypatch, tmp_path, fake_cls):
    monkeypatch.setattr(cli, "STYLE_DIR", str(tmp_path))
    profiles_root = tmp_path / "models"
    monkeypatch.setattr(config_loader, "MODEL_PROFILES_DIR", profiles_root)
    write_profile(profiles_root / "models", "fake_image_missing", "image")
    expected = f"{tmp_path}/fake_image_missing/prompts/style.md"

    with pytest.raises(cli.GeminiInputError) as info:
        cli._build_prompt_config("fake_image_missing", 5, "none")

    message = str(info.value)
    assert expected in message
    assert "fake_image_missing" in message
    # без подмены на LTX-стиль файл не выбирается
    assert cli.FALLBACK_STYLE_PROFILE not in message
