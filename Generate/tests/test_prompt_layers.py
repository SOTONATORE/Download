"""Слои системной инструкции (фаза E2): правила, стиль модели, затем пресет. Файлы пресетов — только tmp_path."""
import hashlib
import json
import pathlib

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import GeminiInputError, PromptConfig, build_system_instruction
from Generate.core.srt_parser import Segment

GEN = pathlib.Path(__file__).resolve().parents[1]
LTX_STYLE = GEN / "model_profiles" / "models" / "ltx25" / "prompts" / "style.md"

# длина базы изменена правилом «not X, but Y» в системной инструкции видео (шаг: противопоставление); прежнее значение 16168.
BASELINE_BYTES = 16340
BASELINE_SHA256 = "bece7c1d709ffee13cb921a43b216a68a778754fbc1cdaea38323a94c9b40107"


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
    absent = build_system_instruction(PromptConfig(media="video", style_file=str(LTX_STYLE)))   # поле preset не задано
    explicit = build_system_instruction(PromptConfig(media="video", style_file=str(LTX_STYLE), preset="none"))
    assert absent == explicit
    raw = absent.encode("utf-8")
    assert len(raw) == BASELINE_BYTES
    assert hashlib.sha256(raw).hexdigest() == BASELINE_SHA256
    assert "# Preset" not in absent


def test_preset_not_in_system_instruction_and_attached_to_prompts(presets):
    (presets / "test.md").write_text("test\n", encoding="utf-8")
    base = build_system_instruction(PromptConfig(media="video", style_file=str(LTX_STYLE)))
    system = build_system_instruction(PromptConfig(media="video", style_file=str(LTX_STYLE), preset="test"))
    # системная инструкция с пресетом совпадает с базовой: ни заголовка, ни текста пресета
    assert system == base
    assert "# Preset" not in system
    assert system.count("test") == base.count("test")
    # готовый промпт заканчивается текстом пресета
    segs = [Segment(1, 0, 1000, "The old city woke slowly.")]
    cfg = PromptConfig(media="video", style_file=str(LTX_STYLE), preset="test", max_prompt_chars=300)

    class _Stub:
        def generate_json(self, model, system, user, schema):
            return json.dumps([{"segment_index": 1, "prompt": "A baker lifts a tray at dawn."}])

    res = gp.generate_prompts(segs, [1], cfg, _Stub(), "m")
    assert res[0].prompt == "A baker lifts a tray at dawn. test"


def test_preset_missing_file_is_input_error(presets):
    with pytest.raises(GeminiInputError, match="не найден"):
        build_system_instruction(PromptConfig(media="video", style_file=str(LTX_STYLE), preset="nope"))


def test_preset_empty_file_is_input_error(presets):
    (presets / "empty.md").write_text("  \n", encoding="utf-8")
    with pytest.raises(GeminiInputError, match="пуст"):
        build_system_instruction(PromptConfig(media="video", style_file=str(LTX_STYLE), preset="empty"))


def test_preset_only_user_only_block_counts_as_empty(presets):
    (presets / "hidden.md").write_text("<!-- USER_ONLY_START -->x<!-- USER_ONLY_END -->", encoding="utf-8")
    with pytest.raises(GeminiInputError, match="пуст"):
        build_system_instruction(PromptConfig(media="video", style_file=str(LTX_STYLE), preset="hidden"))


@pytest.mark.parametrize("bad", ["../x", "a b", "a/b", "README", ""])
def test_preset_bad_name_rejected(presets, bad):
    with pytest.raises(GeminiInputError, match="Недопустимое имя пресета"):
        build_system_instruction(PromptConfig(media="video", style_file=str(LTX_STYLE), preset=bad))
