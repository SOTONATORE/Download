"""Слои системной инструкции (фаза E2): правила, стиль модели, затем пресет. Файлы пресетов — только tmp_path."""
import hashlib
import pathlib

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import GeminiInputError, PromptConfig, build_system_instruction

GEN = pathlib.Path(__file__).resolve().parents[1]
LTX_STYLE = GEN / "model_profiles" / "models" / "ltx25" / "prompts" / "style.md"

# Эталон: системная инструкция ltx25 ДО фазы E2 (код до слоя пресета), снято с исходного дерева.
BASELINE_BYTES = 16007
BASELINE_SHA256 = "df7351ffd49270bfed51b1b768340f43f48152fd30256928fa82621d44454f1b"


@pytest.fixture
def presets(tmp_path, monkeypatch):
    """Папка пресетов подменяется временной; в репозитории файлов пресетов не появляется."""
    d = tmp_path / "prompts"
    d.mkdir()
    monkeypatch.setattr(gp, "PRESETS_DIR", d)
    return d


def test_real_presets_dir_is_the_documented_location():
    assert gp.PRESETS_DIR == GEN / "model_profiles" / "presets" / "prompts"


def test_preset_none_is_byte_identical_to_baseline():
    absent = build_system_instruction(PromptConfig(style_file=str(LTX_STYLE)))   # поле preset не задано
    explicit = build_system_instruction(PromptConfig(style_file=str(LTX_STYLE), preset="none"))
    assert absent == explicit
    raw = absent.encode("utf-8")
    assert len(raw) == BASELINE_BYTES
    assert hashlib.sha256(raw).hexdigest() == BASELINE_SHA256
    assert "# Preset" not in absent


def test_preset_overlay_sits_after_model_style(presets):
    (presets / "pz.md").write_text(
        "PRESET LINE one\n<!-- USER_ONLY_START -->HIDDEN NOTE<!-- USER_ONLY_END -->\nPRESET LINE two\n",
        encoding="utf-8")
    base = build_system_instruction(PromptConfig(style_file=str(LTX_STYLE)))
    system = build_system_instruction(PromptConfig(style_file=str(LTX_STYLE), preset="pz"))
    # только добавлен блок в конец: правила и стиль модели не меняются
    assert system == base + "\n\n# Preset\nPRESET LINE one\n\nPRESET LINE two"
    assert system.index("# Style guide") < system.index("# Preset")
    assert "HIDDEN NOTE" not in system


def test_preset_missing_file_is_input_error(presets):
    with pytest.raises(GeminiInputError, match="не найден"):
        build_system_instruction(PromptConfig(style_file=str(LTX_STYLE), preset="nope"))


def test_preset_empty_file_is_input_error(presets):
    (presets / "empty.md").write_text("  \n", encoding="utf-8")
    with pytest.raises(GeminiInputError, match="пуст"):
        build_system_instruction(PromptConfig(style_file=str(LTX_STYLE), preset="empty"))


def test_preset_only_user_only_block_counts_as_empty(presets):
    (presets / "hidden.md").write_text("<!-- USER_ONLY_START -->x<!-- USER_ONLY_END -->", encoding="utf-8")
    with pytest.raises(GeminiInputError, match="пуст"):
        build_system_instruction(PromptConfig(style_file=str(LTX_STYLE), preset="hidden"))


@pytest.mark.parametrize("bad", ["../x", "a b", "a/b", "README", ""])
def test_preset_bad_name_rejected(presets, bad):
    with pytest.raises(GeminiInputError, match="Недопустимое имя пресета"):
        build_system_instruction(PromptConfig(style_file=str(LTX_STYLE), preset=bad))
