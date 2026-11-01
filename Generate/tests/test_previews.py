"""Превью из prompts.txt: разбор файла, переключатель, номера задач, индекс, проверка до аренды."""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import pytest

_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

try:
    from Generate import cli
    from Generate.core import orchestrator as orch
    from Generate.core import previews as pv
except ImportError:  # pragma: no cover
    import cli  # type: ignore
    from core import orchestrator as orch  # type: ignore
    from core import previews as pv  # type: ignore


# --- разбор файла -------------------------------------------------------------

def test_parse_splits_blocks_on_blank_lines_crlf_and_bom():
    text = "﻿первый промпт\r\n\r\nвторой промпт\r\n\r\n\r\nтретий промпт\r\n"
    assert pv.parse_prompts(text) == ["первый промпт", "второй промпт", "третий промпт"]


def test_parse_joins_lines_inside_one_block_and_collapses_spaces():
    text = "строка  один\nстрока два\n\nсосед"
    assert pv.parse_prompts(text) == ["строка один строка два", "сосед"]


def test_parse_ignores_leading_and_trailing_blank_lines():
    assert pv.parse_prompts("\n\n  \nпромпт\n\n\n") == ["промпт"]


def test_parse_empty_text_gives_no_prompts():
    assert pv.parse_prompts("") == []
    assert pv.parse_prompts("\n\n\n") == []


def test_load_prompts_missing_file_is_input_error(tmp_path):
    with pytest.raises(pv.PreviewInputError, match="не найден"):
        pv.load_prompts(str(tmp_path / "нет.txt"))


def test_load_prompts_empty_path_is_input_error():
    with pytest.raises(pv.PreviewInputError, match="не найден"):
        pv.load_prompts("")


def test_load_prompts_file_without_prompts_is_input_error(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("\n\n   \n", encoding="utf-8")
    with pytest.raises(pv.PreviewInputError, match="нет ни одного промпта"):
        pv.load_prompts(str(f))


def test_load_prompts_non_utf8_is_input_error(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_bytes("промпт".encode("cp1251"))
    with pytest.raises(pv.PreviewInputError, match="UTF-8"):
        pv.load_prompts(str(f))


def test_load_prompts_reads_user_format(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_bytes("Типо промпт один.\r\n\r\nТипо промпт два.\r\n".encode("utf-8"))
    assert pv.load_prompts(str(f)) == ["Типо промпт один.", "Типо промпт два."]


# --- переключатель ------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [(None, 0), ("", 0), ("выкл", 0),
                                             ("1", 1), ("2", 2), ("3", 3), ("4", 4)])
def test_count_value_accepts_off_and_1_to_4(value, expected):
    assert pv.count_value(value) == expected


@pytest.mark.parametrize("value", ["5", "0", "off", "да"])
def test_count_value_rejects_other_values(value):
    with pytest.raises(pv.PreviewInputError):
        pv.count_value(value)


# --- номера, имена и сиды -------------------------------------------------------

def test_build_previews_names_order_and_seeds():
    seeds = iter(range(100, 200))
    items = pv.build_previews(["а", "б"], 2, seed_fn=lambda: next(seeds))
    assert [(i.x, i.y, i.filename) for i in items] == [
        (1, 1, "preview_1_1.png"), (1, 2, "preview_1_2.png"),
        (2, 1, "preview_2_1.png"), (2, 2, "preview_2_2.png")]
    assert [i.seed for i in items] == [100, 101, 102, 103]
    assert len({i.num for i in items}) == len(items)
    assert all(pv.is_preview_num(i.num) for i in items)


def test_build_previews_off_gives_nothing():
    assert pv.build_previews(["а"], 0) == []


def test_default_seed_is_random_and_in_range():
    seeds = {pv.random_seed() for _ in range(50)}
    assert len(seeds) > 1
    assert all(0 <= s < pv.SEED_LIMIT for s in seeds)


def test_preview_numbers_never_collide_with_clip_numbers():
    items = pv.build_previews(["а"] * 3, 4, seed_fn=lambda: 1)
    assert min(i.num for i in items) > 10 ** 6
    assert not pv.is_preview_num(12345)


# --- индекс ---------------------------------------------------------------------

def test_index_text_has_all_details_and_variant_not_seed():
    items = pv.build_previews(["кот у окна"], 2, seed_fn=iter([777, 888]).__next__)
    items[0].card = "карта 1"
    items[0].status = pv.STATUS_DONE
    text = pv.index_text(items)
    assert "preview_1_1.png" in text and "промпт 1" in text
    assert "y 1" in text and "seed 777" in text and "карта: карта 1" in text
    assert "статус: готово" in text and "кот у окна" in text
    assert "y 2" in text and "seed 888" in text


def test_index_text_without_items_says_so():
    assert "не запускались" in pv.index_text([])


# --- проверка до аренды (cli) -------------------------------------------------

def test_cli_check_off_needs_no_file_and_no_photo(tmp_path):
    assert cli._check_previews("выкл", str(tmp_path / "нет.txt"), None) is None


def test_cli_check_count_without_photo_model_is_input_error(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("промпт\n", encoding="utf-8")
    assert cli._check_previews("2", str(f), None) == cli.EXIT_INPUT


def test_cli_check_count_with_missing_file_is_input_error(tmp_path):
    assert cli._check_previews("1", str(tmp_path / "нет.txt"), "zimage_turbo") == cli.EXIT_INPUT


def test_cli_check_count_with_good_file_passes(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("промпт\n\nещё промпт\n", encoding="utf-8")
    assert cli._check_previews("3", str(f), "zimage_turbo") is None


def test_cli_run_parser_accepts_prompts_options():
    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    if parser is None:
        pytest.skip("парсер не вынесен в build_parser")
    args = parser.parse_args(["run", "--prompts-file", "prompts.txt", "--prompts-count", "выкл"])
    assert args.prompts_count == "выкл"


# --- оркестратор: очередь, отправка, ошибки -------------------------------------

def _bare_orchestrator(items):
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    o._video_queue = deque()
    o._photo_queue = deque()
    o._photo_nums = set()
    o._attempts = {}
    o._previews = {i.num: i for i in items}
    o._preview_order = [i.num for i in items]
    o.photo_profile = "zimage_turbo"
    return o


def test_preview_tasks_go_to_photo_queue_not_video():
    items = pv.build_previews(["а"], 2, seed_fn=lambda: 5)
    o = _bare_orchestrator(items)
    for item in items:
        assert o._queue_of(item.num) is o._photo_queue


def test_failed_preview_is_retried_then_marked_failed_without_clip_counters():
    items = pv.build_previews(["а"], 1, seed_fn=lambda: 5)
    o = _bare_orchestrator(items)
    item = items[0]
    o._attempts[item.num] = 1
    o._fail_clip(item.num, "тест")  # маршрут через _fail_clip для превью
    assert list(o._photo_queue) == [item.num]
    assert item.status == pv.STATUS_QUEUED
    assert not hasattr(o, "_failed") or item.num not in getattr(o, "_failed", set())

    o._photo_queue.clear()
    o._attempts[item.num] = orch.MAX_ATTEMPTS
    o._fail_clip(item.num, "тест")
    assert item.status == pv.STATUS_FAILED
    assert not o._photo_queue


def test_prepare_previews_off_does_nothing(tmp_path):
    o = _bare_orchestrator([])
    o.prompts_count = "выкл"
    o.prompts_file = str(tmp_path / "нет.txt")  # не читается при «выкл»
    o._prepare_previews()
    assert o._previews == {}


def test_prepare_previews_builds_items_for_count(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("один\n\nдва\n", encoding="utf-8")
    o = _bare_orchestrator([])
    o.prompts_count = "2"
    o.prompts_file = str(f)
    o._prepare_previews()
    assert len(o._previews) == 4
    assert len(o._preview_order) == 4


def test_prepare_previews_without_photo_profile_is_error(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("один\n", encoding="utf-8")
    o = _bare_orchestrator([])
    o.photo_profile = None
    o.prompts_count = "1"
    o.prompts_file = str(f)
    with pytest.raises(pv.PreviewInputError):
        o._prepare_previews()


def test_class_defaults_keep_orchestrator_without_init_working():
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    assert o._previews == {}
    assert list(o._preview_order) == []
    o._prepare_previews()  # «выкл» по умолчанию: ничего не делает
