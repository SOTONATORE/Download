"""Диагностика сырых ответов модели: debug.dump_raw_responses (профиль cartoon_photos).

Проверяют: без флага каталога raw/ нет; с флагом один файл на запрос (первый ответ батча, повтор формата,
исправление) с сырым текстом; сбой записи не поднимает исключение и даёт предупреждение только с путём;
флаг включён только у cartoon_photos, у ltx25 и zimage_turbo ключи не передаются; разбор блока debug.

Тесты написаны, не запускались (прогоняет пользователь). Сеть не используется.
"""
import json
import logging
import pathlib
import socket

import pytest

from Generate import cli
from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import PromptConfig, generate_prompts
from Generate.core.prompt_check import PROBLEM_CAMERA_MOTION
from Generate.core.srt_parser import Segment

MARK = "SECRET-RAW-MARKER"
# причина в очереди исправлений: настоящий код нарушения (problem_en читает код до «: »)
REASON = f"{PROBLEM_CAMERA_MOTION}: pan left"
GOOD = "A wide shot of a quiet street at dawn, a baker lifts a tray."
SEGMENTS = [
    Segment(1, 0, 1000, "The old city woke slowly."),
    Segment(2, 1000, 2000, "Merchants opened their stalls"),
    Segment(3, 2000, 3000, "and carts rolled in from the hills."),
]
# имя волны в пути: для media video «видео» (как в generate_prompts без пула)
WAVE = "видео"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


class CountingTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate_json(self, model, system, user, schema):
        self.calls += 1
        return self.replies.pop(0)


def _cfg(tmp_path, dump, raw_dir=None):
    style = tmp_path / "style.md"
    style.write_text("# Style guide\nWrite one paragraph.\n", encoding="utf-8")
    return PromptConfig(style_file=str(style), max_prompt_chars=300, batch_size=10, media="video",
                        raw_dir=str(raw_dir if raw_dir is not None else tmp_path / "raw"),
                        niche="cartoon_photos", dump_raw_responses=dump)


def _ok(nums):
    return json.dumps([{"segment_index": n, "prompt": GOOD} for n in nums])


def _names(folder):
    return sorted(p.name for p in folder.iterdir())


# --- флаг выключен: каталога raw/ нет ---------------------------------------------------------------

def test_flag_off_writes_no_raw_folder(tmp_path):
    cfg = _cfg(tmp_path, dump=False)
    res = generate_prompts(SEGMENTS, [1, 3], cfg, CountingTransport([_ok([1, 3])]), "m", max_workers=1)
    assert [r.num for r in res] == [1, 3]
    assert not (tmp_path / "raw").exists()


def test_flag_off_format_retry_writes_no_raw_folder(tmp_path):
    cfg = _cfg(tmp_path, dump=False)
    transport = CountingTransport(["это не JSON " + MARK, _ok([1, 3])])
    res = generate_prompts(SEGMENTS, [1, 3], cfg, transport, "m", max_workers=1)
    assert transport.calls == 2
    assert [r.num for r in res] == [1, 3]
    assert not (tmp_path / "raw").exists()


# --- флаг включён: один файл на запрос ----------------------------------------------------------------

def test_flag_on_clean_answer_writes_one_file_with_raw_text(tmp_path):
    answer = _ok([1, 3])
    cfg = _cfg(tmp_path, dump=True)
    generate_prompts(SEGMENTS, [1, 3], cfg, CountingTransport([answer]), "m", max_workers=1)
    raw = tmp_path / "raw"
    assert _names(raw) == [f"cartoon_photos_{WAVE}_0_1.txt"]
    assert (raw / f"cartoon_photos_{WAVE}_0_1.txt").read_text(encoding="utf-8") == answer


def test_flag_on_format_retry_writes_first_answer_and_retry_files(tmp_path):
    bad = "это не JSON " + MARK
    good = _ok([1, 3])
    cfg = _cfg(tmp_path, dump=True)
    transport = CountingTransport([bad, good])
    generate_prompts(SEGMENTS, [1, 3], cfg, transport, "m", max_workers=1)
    assert transport.calls == 2
    raw = tmp_path / "raw"
    assert _names(raw) == sorted([f"cartoon_photos_{WAVE}_0_1.txt", f"cartoon_photos_{WAVE}_0_1r1.txt"])
    assert (raw / f"cartoon_photos_{WAVE}_0_1.txt").read_text(encoding="utf-8") == bad
    assert (raw / f"cartoon_photos_{WAVE}_0_1r1.txt").read_text(encoding="utf-8") == good


def test_flag_on_fix_request_writes_round_and_package_file(tmp_path):
    # Русский: прямой вызов исправления (process_fix): клип в очереди, раунд 1, пакет 1 — один файл с сырым ответом
    fix_answer = _ok([1])
    cfg = _cfg(tmp_path, dump=True)
    builder = gp._Builder(SEGMENTS, [1], cfg)
    transport = CountingTransport([fix_answer])
    job = gp._WaveJob(WAVE, builder, builder.batches([1]), transport, "m", "", None)
    clip = gp.PromptResult(1, "A street with a baker.", True, REASON, {},
                           [REASON], None, "A street with a baker.")
    job.rework_queue.append(clip)
    job.start_round()   # раунд 1, пакет 1
    job.process_fix(gp._LoggedTransport(transport), [clip], 1)
    raw = tmp_path / "raw"
    assert _names(raw) == [f"cartoon_photos_{WAVE}_1_1.txt"]
    assert (raw / f"cartoon_photos_{WAVE}_1_1.txt").read_text(encoding="utf-8") == fix_answer


def test_flag_on_fix_request_without_dump_flag_writes_nothing(tmp_path):
    cfg = _cfg(tmp_path, dump=False)
    builder = gp._Builder(SEGMENTS, [1], cfg)
    transport = CountingTransport([_ok([1])])
    job = gp._WaveJob(WAVE, builder, builder.batches([1]), transport, "m", "", None)
    clip = gp.PromptResult(1, "A street with a baker.", True, REASON, {},
                           [REASON], None, "A street with a baker.")
    job.rework_queue.append(clip)
    job.start_round()
    job.process_fix(gp._LoggedTransport(transport), [clip], 1)
    assert not (tmp_path / "raw").exists()


# --- сбой записи: не поднимается, предупреждение только с путём --------------------------------------

def test_write_failure_does_not_raise_and_warns_with_path_only(tmp_path, monkeypatch, caplog):
    real_mkdir = pathlib.Path.mkdir

    def failing_mkdir(self, *args, **kwargs):
        if self.name == "raw":
            raise PermissionError("read-only")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", failing_mkdir)
    cfg = _cfg(tmp_path, dump=True)
    caplog.set_level(logging.WARNING)
    res = generate_prompts(SEGMENTS, [1, 3], cfg, CountingTransport([_ok([1, 3])]), "m", max_workers=1)
    assert [r.num for r in res] == [1, 3]   # прогон не остановлен
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "сырой ответ модели" in r.getMessage()]
    assert warnings and str(tmp_path / "raw") in warnings[0]
    assert "read-only" not in caplog.text
    assert MARK not in caplog.text


# --- профиль: флаг только у cartoon_photos -----------------------------------------------------------

@pytest.mark.parametrize("profile, expected", [("cartoon_photos", True), ("ltx25", False), ("zimage_turbo", False)])
def test_flag_is_enabled_only_for_cartoon_profile(profile, expected):
    cfg = cli._build_prompt_config(profile, 25, "none", fail_fast=False, raw_dir=None, srt_path=None)
    assert cfg.dump_raw_responses is expected
    assert (cfg.niche == profile) is expected


def test_nested_flag_parser_reads_only_the_debug_block():
    lines = ["name: x", "debug:", "  dump_raw_responses: true  # комментарий", "other: 1"]
    assert cli._profile_nested_flag(lines, "debug", "dump_raw_responses", "x", default="false") == "true"
    assert cli._profile_nested_flag(["name: x"], "debug", "dump_raw_responses", "x", default="false") == "false"
    # ключ вне блока (без отступа) блока не открывает
    assert cli._profile_nested_flag(["debug:", "  other: 1", "dump_raw_responses: true"],
                                    "debug", "dump_raw_responses", "x", default="false") == "false"
    assert cli._profile_nested_flag(["debug:", "# комментарий", "  dump_raw_responses: TRUE"],
                                    "debug", "dump_raw_responses", "x", default="false") == "true"
