"""Тесты стиля промптов для фото-модели Z-Image-Turbo и выбора файла стиля в cli.

Тесты написаны для pytest, но в этом шаге не запускаются. Реальные файлы
профилей не трогаются: тип профиля подменяется, каталог стилей указывает во временную папку.
"""
import re
from pathlib import Path

import pytest

from Generate import cli

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

    def __init__(self, style_file, batch_size, preset):
        self.style_file = style_file
        self.batch_size = batch_size
        self.preset = preset


@pytest.fixture
def fake_cls(monkeypatch):
    monkeypatch.setattr(cli, "_find_symbol", lambda name, modules: _FakePromptConfig)
    return _FakePromptConfig


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
    monkeypatch.setattr(cli, "load_profile_type", lambda name: "image")
    own = tmp_path / "fake_image" / "prompts"
    own.mkdir(parents=True)
    (own / "style.md").write_text("style", encoding="utf-8")

    cfg = cli._build_prompt_config("fake_image", 5, "none")

    assert cfg.style_file == f"{tmp_path}/fake_image/prompts/style.md"
    assert cfg.batch_size == 5 and cfg.preset == "none"


@pytest.mark.parametrize("profile_type", ["video", None])
def test_video_or_missing_type_keeps_ltx_fallback(monkeypatch, tmp_path, fake_cls, profile_type):
    monkeypatch.setattr(cli, "STYLE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "load_profile_type", lambda name: profile_type or "video")
    # у профиля нет своего style.md: прежняя подмена на профиль по умолчанию
    cfg = cli._build_prompt_config("fake_video", 3, "none")
    assert cfg.style_file == f"{tmp_path}/{cli.FALLBACK_STYLE_PROFILE}/prompts/style.md"


def test_image_profile_missing_style_raises_with_path_and_profile(monkeypatch, tmp_path, fake_cls):
    monkeypatch.setattr(cli, "STYLE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "load_profile_type", lambda name: "image")
    expected = f"{tmp_path}/fake_image_missing/prompts/style.md"

    with pytest.raises(cli.GeminiInputError) as info:
        cli._build_prompt_config("fake_image_missing", 5, "none")

    message = str(info.value)
    assert expected in message
    assert "fake_image_missing" in message
    # без подмены на LTX-стиль файл не выбирается
    assert cli.FALLBACK_STYLE_PROFILE not in message
