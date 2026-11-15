"""Контракт refs в системной инструкции: ниша с allowed_refs (cartoon_photos) и профили без него.

Проверяют: сочетания в инструкции берутся из allowed_refs; пустой список и «сегмент без refs» не пишутся;
профили без allowed_refs — байт в байт как раньше; файлы отладки при fail_fast пишутся только при проблеме
и только при raw_dir; сбой записи не скрывает GeminiError и даёт предупреждение только с путём.

Тесты написаны, не запускались (прогоняет пользователь на GitHub). Сеть не используется.
"""
import hashlib
import json
import logging
import pathlib
import socket

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import GeminiError, PromptConfig, generate_prompts
from Generate.core.srt_parser import Segment

GEN = pathlib.Path(__file__).resolve().parents[1]
NICHE = GEN / "Nishes" / "cartoon_photos"
MODELS = GEN / "model_profiles" / "models"
MARK = "SECRET-RAW-MARKER"
GOOD = "A wide shot of a quiet street at dawn, a baker lifts a tray."
SEGMENTS = [
    Segment(1, 0, 1000, "The old city woke slowly."),
    Segment(2, 1000, 2000, "Merchants opened their stalls"),
    Segment(3, 2000, 3000, "and carts rolled in from the hills."),
]

# sha256 и длина системной инструкции до правки (старый код, профиль без refs; значения из test_rules_values.py)
LTX_SHA256 = "50391333ebd2262acbd02c9a50cbb8d6f8e078d8371dc24eb9559c364edeaeb1"
LTX_BYTES = 19940
ZIM_SHA256 = "9a894f59b81b712a6f4aa87a8f4cf8f03ce95937525b4bd84527fc43ed2fe21e"
ZIM_BYTES = 6735
# то же до правки, но с включёнными refs без allowed_refs (старый текст блока «# Reference images»)
LTX_REFS_OLD_SHA256 = "35f2e52254710dce665c56cbfca4b8b5024390fe20e2ce16f785c21252d525ca"
LTX_REFS_OLD_BYTES = 20409
ZIM_REFS_OLD_SHA256 = "a208d1c2b8f9fbfafb6cdcc197c9e4e4e73df0b86722583fb4aa7fbd4d7afac0"
ZIM_REFS_OLD_BYTES = 7204


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


def _cartoon_instruction() -> str:
    lines = (NICHE / "profile.yaml").read_text(encoding="utf-8").splitlines()
    allowed = gp.parse_allowed_refs(lines, "cartoon_photos")
    cfg = PromptConfig(style_file=str(NICHE / "prompts" / "style.md"), media="image", preset="none",
                       refs_enabled=True, ref_max=6, allowed_refs=allowed)
    return gp.build_system_instruction(cfg)


def _model_cfg(folder: str, media: str, **kw) -> PromptConfig:
    prof = MODELS / folder
    rules = gp.load_rules(prof / "prompts" / "rules.yaml")
    return PromptConfig(style_file=str(prof / "prompts" / "style.md"), media=media, preset="none",
                        rules=rules, **kw)


# --- (a) ниша с allowed_refs ------------------------------------------------------------------------

def test_cartoon_instruction_lists_each_allowed_combination_as_a_line():
    lines = _cartoon_instruction().split("\n")
    lines_expected = ["1", "1, 2", "1, 2, 6", "1, 6", "2", "3", "4", "5"]  # совпадает с profile.yaml
    for line in lines_expected:
        assert line in lines


def test_cartoon_instruction_has_no_old_empty_list_wording():
    text = _cartoon_instruction()
    assert "an empty list if none" not in text
    assert "never an empty list" in text


def test_cartoon_instruction_has_no_no_refs_tag_line():
    text = _cartoon_instruction()
    assert "no refs has" not in text
    assert ("Every segment has at least one reference, so every segment has at least one <imageN> tag."
            in text)


def test_cartoon_instruction_and_style_guide_use_reference_number_rule():
    # Комментарий: инструкция и style.md называют правило «номер референса», а не позицию в refs (K-th entry).
    text = _cartoon_instruction()
    style = (NICHE / "prompts" / "style.md").read_text(encoding="utf-8")
    assert "written with the position of that image" in text
    assert "the tags follow its positions" in style
    assert "K-th entry" not in text
    assert "K-th entry" not in style


def test_normaliser_maps_six_to_two_and_keeps_positions_for_refs_one_and_six():
    # Комментарий: refs [1, 6]; <image6> — номер 6, то есть позиция 2. <image1> и <image2> — позиции, не меняются.
    assert gp._normalise_labels("A boy waves at <image6>.", [1, 6]) == "A boy waves at <image2>."
    assert gp._normalise_labels("A dog <image1> and a boy <image2>.", [1, 6]) == "A dog <image1> and a boy <image2>."
    assert gp._normalise_labels("A dog <image1> sits.", [1, 6]) == "A dog <image1> sits."


# --- (b) профили без allowed_refs: байт в байт -------------------------------------------------------

@pytest.mark.parametrize("folder, media, sha, size", [
    ("ltx25", "video", LTX_SHA256, LTX_BYTES),
    ("zimage_turbo", "image", ZIM_SHA256, ZIM_BYTES),
])
def test_model_profiles_without_refs_have_no_refs_block_and_same_sha(folder, media, sha, size):
    data = gp.build_system_instruction(_model_cfg(folder, media)).encode("utf-8")
    assert b"# Reference images" not in data
    assert len(data) == size
    assert hashlib.sha256(data).hexdigest() == sha


@pytest.mark.parametrize("folder, media, sha, size", [
    ("ltx25", "video", LTX_REFS_OLD_SHA256, LTX_REFS_OLD_BYTES),
    ("zimage_turbo", "image", ZIM_REFS_OLD_SHA256, ZIM_REFS_OLD_BYTES),
])
def test_refs_without_allowed_refs_keep_old_block_byte_for_byte(folder, media, sha, size):
    cfg = _model_cfg(folder, media, refs_enabled=True, ref_max=6)
    text = gp.build_system_instruction(cfg)
    assert "(an empty list if none)" in text
    assert "A segment with no refs has no <imageN> tag." in text
    data = text.encode("utf-8")
    assert len(data) == size
    assert hashlib.sha256(data).hexdigest() == sha


# --- (c) fail_fast: файлы отладки -------------------------------------------------------------------

class CountingTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate_json(self, model, system, user, schema):
        self.calls += 1
        return self.replies.pop(0)


def _cfg(tmp_path, **kw):
    style = tmp_path / "style.md"
    style.write_text("# Style guide\nWrite one paragraph.\n", encoding="utf-8")
    kw.setdefault("style_file", str(style))
    kw.setdefault("max_prompt_chars", 300)
    kw.setdefault("batch_size", 10)
    kw.setdefault("media", "video")
    return PromptConfig(**kw)


def _ok(nums):
    return json.dumps([{"segment_index": n, "prompt": GOOD} for n in nums])


def test_fail_fast_failure_writes_raw_and_debug_files_with_input_content(tmp_path):
    raw_dir = tmp_path / "raw"
    cfg = _cfg(tmp_path, fail_fast=True, raw_dir=str(raw_dir))
    raw_answer = "это не JSON " + MARK
    with pytest.raises(GeminiError):
        generate_prompts(SEGMENTS, [1, 3], cfg, CountingTransport([raw_answer]), "m", max_workers=1)
    debug = tmp_path / "debug"
    assert (raw_dir / "batch_1-3_not_json.txt").read_text(encoding="utf-8") == raw_answer
    assert (debug / "system_instruction.txt").read_text(encoding="utf-8") == gp.build_system_instruction(cfg)
    expected_user = gp._Builder(SEGMENTS, [1, 3], cfg).batches([1, 3])[0].user
    assert (debug / "user_request.txt").read_text(encoding="utf-8") == expected_user


def test_fail_fast_failure_without_raw_dir_writes_no_debug_files(tmp_path):
    cfg = _cfg(tmp_path, fail_fast=True)  # raw_dir не задан
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(GeminiError):
        generate_prompts(SEGMENTS, [1, 3], cfg, CountingTransport(["это не JSON " + MARK]), "m", max_workers=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_fail_fast_success_writes_no_debug_files(tmp_path):
    cfg = _cfg(tmp_path, fail_fast=True, raw_dir=str(tmp_path / "raw"))
    res = generate_prompts(SEGMENTS, [1, 3], cfg, CountingTransport([_ok([1, 3])]), "m", max_workers=1)
    assert [r.num for r in res] == [1, 3]
    assert not (tmp_path / "debug").exists()


# --- (d) сбой записи файлов отладки -----------------------------------------------------------------

def test_debug_write_failure_keeps_error_and_warns_with_path_only(tmp_path, monkeypatch, caplog):
    real_mkdir = pathlib.Path.mkdir

    def failing_mkdir(self, *args, **kwargs):
        if self.name == "debug":
            raise PermissionError("read-only")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", failing_mkdir)
    raw_dir = tmp_path / "raw"
    cfg = _cfg(tmp_path, fail_fast=True, raw_dir=str(raw_dir))
    caplog.set_level(logging.WARNING)
    with pytest.raises(GeminiError) as info:
        generate_prompts(SEGMENTS, [1, 3], cfg, CountingTransport(["это не JSON " + MARK]), "m", max_workers=1)
    message = str(info.value)
    assert message.startswith("Ошибка")
    assert "fail_fast" in message
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "отладочные файлы" in r.getMessage()]
    assert warnings and str(tmp_path / "debug") in warnings[0]
    assert "read-only" not in caplog.text
    assert MARK not in caplog.text
    assert (raw_dir / "batch_1-3_not_json.txt").exists()
