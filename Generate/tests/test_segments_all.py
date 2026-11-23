"""Тесты segments: all в profile.yaml ниши (cartoon): список клипов из SRT, missing.txt не читается.

Сеть не используется: профиль и missing.txt создаются во временной папке.
"""
import pathlib

import pytest

from Generate import cli
from Generate.core import orchestrator as orch
from Generate.core.gemini_prompts import GeminiInputError
from Generate.core.missing import parse_missing
from Generate.core.srt_parser import Segment

ROOT = pathlib.Path(__file__).resolve().parents[2]
CARTOON_STYLE = ROOT / "Generate" / "Nishes" / "cartoon" / "prompts" / "style.md"


def _segs(*nums):
    return [Segment(n, n * 1000, n * 1000 + 500, f"текст {n}") for n in nums]


def _profile(tmp_path, monkeypatch, yaml_text):
    folder = tmp_path / "niche"
    folder.mkdir()
    (folder / "profile.yaml").write_text(yaml_text, encoding="utf-8")
    monkeypatch.setattr(cli, "profile_dir", lambda profile: folder)
    return "niche"


def _missing(tmp_path, text):
    path = tmp_path / "missing.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _fail_if_called(*args, **kwargs):
    raise AssertionError("parse_missing не должен вызываться при segments: all")


def test_segments_all_gives_srt_numbers_in_ascending_order(tmp_path, monkeypatch):
    profile = _profile(tmp_path, monkeypatch, "name: cartoon\nsegments: all\n")
    monkeypatch.setattr(orch, "parse_missing", _fail_if_called)
    segs = _segs(9, 2, 5, 1)
    assert cli._segments_mode(profile) == "all"
    assert orch.wanted_segment_numbers(segs, tmp_path / "missing.txt", True) == [1, 2, 5, 9]


def test_profile_without_key_keeps_parse_missing_result(tmp_path, monkeypatch):
    profile = _profile(tmp_path, monkeypatch, "name: cartoon\ntype: image\n")
    segs = _segs(1, 2, 3, 4)
    path = _missing(tmp_path, "3\n1\n")
    assert cli._segments_mode(profile) == "missing"
    assert orch.wanted_segment_numbers(segs, path, False) == parse_missing(path, segs) == [1, 3]


def test_unknown_segments_value_is_russian_error_naming_value(tmp_path, monkeypatch):
    profile = _profile(tmp_path, monkeypatch, "name: cartoon\nsegments: some\n")
    with pytest.raises(GeminiInputError) as info:
        cli._segments_mode(profile)
    assert str(info.value).startswith("Ошибка входных данных")
    assert "«some»" in str(info.value)


def test_generation_list_with_all_ignores_two_number_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "parse_missing", _fail_if_called)
    path = _missing(tmp_path, "2\n4\n")
    segs = _segs(1, 2, 3, 4, 5, 6)
    assert orch.wanted_segment_numbers(segs, path, True) == [1, 2, 3, 4, 5, 6]


def _default_hero_section():
    # Блок 7A: в style.md нет заголовка «Default hero» и нет абзаца «Hero placement:».
    # Ищем пункт «a) Default hero» внутри раздела «# Character Canon». Текст читается из style.md во время теста.
    text = CARTOON_STYLE.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "# Character Canon" in text, "в style.md нет заголовка «# Character Canon»"
    canon = text.split("# Character Canon", 1)[1].split("\n# ", 1)[0]
    paras = [p for p in canon.split("\n\n") if p.startswith("a) Default hero")]
    assert len(paras) == 1, "в разделе «# Character Canon» style.md нет пункта «a) Default hero»"
    return paras[0]


def test_cartoon_default_hero_has_no_ref_tags():
    # Блок 7A: в описании героя нет меток референсов (ref_ и <image): ниша без референсов.
    sec = _default_hero_section()
    assert "ref_" not in sec, "в «Default hero» (style.md) найдена метка ref_"
    assert "<image" not in sec, "в «Default hero» (style.md) найдена метка <image"


def test_cartoon_default_hero_names_main_character_with_thin_stick_arms():
    # Блок 7A: герой назван «the main character», руки описаны как тонкие палочки (thin / stick) — как в style.md.
    sec = _default_hero_section()
    assert "the main character" in sec, "в «Default hero» (style.md) нет имени «the main character»"
    assert "thin" in sec, "в «Default hero» (style.md) нет слова «thin» про руки"
    assert "stick" in sec, "в «Default hero» (style.md) нет слова «stick» про руки"
