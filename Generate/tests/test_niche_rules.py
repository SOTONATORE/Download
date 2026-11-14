"""Ниша cartoon_photos: отключаемые правила (relaxed_rules), язык текста в кадре (text_language), число референсов.

Тесты написаны, не запускались (прогоняет пользователь на GitHub). cli не импортируется: он тянет orchestrator,
поэтому профиль читается через функции gemini_prompts на строках реального profile.yaml.
"""
from pathlib import Path

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core import prompt_check as pc
from Generate.core.gemini_prompts import GeminiInputError, PromptConfig

NICHE_DIR = Path(__file__).resolve().parents[1] / "Nishes" / "cartoon_photos"
RELAX_ALL = frozenset({"art_medium", "small_objects", "frame_text"})
STICKMAN = "A cartoon stickman in a flat style."
PEN = "a pen on the desk"
NEUTRAL = "two red diagonal strokes drawn like a brush over the scene"

RUSSIAN_SRT = (
    "1\n00:00:00,000 --> 00:00:03,000\nВ центральном архиве есть картонная карточка.\n\n"
    "2\n00:00:03,100 --> 00:00:06,000\nНа этой карточке записано немного.\n"
)
ENGLISH_SRT = "1\n00:00:00,000 --> 00:00:02,000\nThe cat is on the table and it is black.\n"


def _cfg(tmp_path, **kw):
    style = tmp_path / "style.md"
    style.write_text("# Style\nPlain scene.\n", encoding="utf-8")
    kw.setdefault("style_file", str(style))
    kw.setdefault("media", "image")
    return PromptConfig(**kw)


def _make_refs(folder, count):
    folder.mkdir(parents=True, exist_ok=True)
    for n in range(1, count + 1):
        (folder / f"ref_{n}.png").write_bytes(b"")


def _profile_lines():
    return (NICHE_DIR / "profile.yaml").read_text(encoding="utf-8").splitlines()


# --- а) набор отключаемых правил ---------------------------------------------------------------

def test_relaxable_rules_is_exactly_three_ids_and_no_safety_rule():
    assert pc.RELAXABLE_RULES == frozenset({"art_medium", "small_objects", "frame_text"})
    for name in ("PROBLEM_CHILD", "PROBLEM_NUDITY", "PROBLEM_PEOPLE_COUNT", "PROBLEM_MANY_OF_ONE",
                 "PROBLEM_SPEECH", "PROBLEM_FIRE_HANDS", "PROBLEM_CHAOS", "PROBLEM_WEAPON",
                 "PROBLEM_CLOSE_FACE", "PROBLEM_ETHNICITY"):
        assert getattr(pc, name) not in pc.RELAXABLE_RULES


# --- б) неизвестное имя в relaxed_rules: ошибка при загрузке профиля -----------------------------

def test_unknown_relaxed_rule_fails_at_load_with_russian_message():
    with pytest.raises(GeminiInputError) as exc:
        gp.profile_relaxation(["type: image", "relaxed_rules: [child_in_frame]"], "cartoon_photos")
    msg = str(exc.value)
    assert "child_in_frame" in msg
    assert "Ошибка входных данных" in msg
    assert "art_medium" in msg


def test_flow_list_parser_reads_items_and_empty_list():
    assert gp.parse_flow_list("[art_medium, small_objects]", "relaxed_rules", "x") == ["art_medium", "small_objects"]
    assert gp.parse_flow_list("[]", "relaxed_rules", "x") == []


def test_flow_list_parser_rejects_non_list():
    with pytest.raises(GeminiInputError, match="списком"):
        gp.parse_flow_list("art_medium", "relaxed_rules", "x")


# --- в) art_medium: стилевое правило снимается только при отключении -------------------------------

def test_art_medium_relaxation_passes_stickman_sentence_and_default_rejects_it():
    assert gp._style_reasons(STICKMAN, "image", frozenset({"art_medium"})) == []
    assert gp._ART_MEDIUM_REASON in gp._style_reasons(STICKMAN, "image")


# --- г) small_objects: мелкий предмет снимается только при отключении ------------------------------

def test_small_objects_relaxation_passes_pen_and_default_rejects_it():
    assert pc.content_hits(PEN, "image", frozenset({"small_objects"})) == []
    codes = [code for code, _ in pc.content_hits(PEN, "image")]
    assert pc.PROBLEM_SMALL_OBJECT in codes


# --- д) нейтральная фраза без нарушений, с отключением и без ---------------------------------------

def test_neutral_sentence_has_no_problem_with_or_without_relaxation():
    for relaxed in (frozenset(), RELAX_ALL):
        assert gp._style_reasons(NEUTRAL, "image", relaxed) == []


# --- е) системная инструкция: строки medium и paper снимаются ---------------------------------------

def test_system_instruction_drops_medium_and_paper_lines_when_relaxed(tmp_path):
    text = gp.build_system_instruction(_cfg(tmp_path, relaxed_rules=frozenset({"art_medium", "frame_text"})))
    assert "Never name a medium" not in text
    assert "without readable text" not in text
    assert "Never describe camera movement" in text


def test_system_instruction_unchanged_without_relaxation(tmp_path):
    text = gp.build_system_instruction(_cfg(tmp_path))
    assert "Never name a medium" in text
    assert "without readable text" in text
    assert "Always write the output in English." in text


# --- ж) text_language: язык текста в кадре из SRT --------------------------------------------------

def test_system_instruction_names_on_frame_language(tmp_path):
    cfg = _cfg(tmp_path, relaxed_rules=RELAX_ALL, text_language="srt", frame_language="Russian")
    text = gp.build_system_instruction(cfg)
    assert "Write all prompt text in English, except text that appears inside the frame, which must be written in Russian." in text
    assert "Always write the output in English" not in text
    assert ("Text that appears inside the frame is written in the on-frame language named in the next rule."
            in text)


def test_text_language_without_detected_language_is_error(tmp_path):
    with pytest.raises(GeminiInputError, match="не определён"):
        gp.build_system_instruction(_cfg(tmp_path, text_language="srt", frame_language=None))


# --- з) detect_srt_language ----------------------------------------------------------------------

def test_detect_srt_language_russian(tmp_path):
    path = tmp_path / "ru.srt"
    path.write_text(RUSSIAN_SRT, encoding="utf-8")
    assert gp.detect_srt_language(path) == "Russian"


def test_detect_srt_language_english(tmp_path):
    path = tmp_path / "en.srt"
    path.write_text(ENGLISH_SRT, encoding="utf-8")
    assert gp.detect_srt_language(path) == "English"


def test_detect_srt_language_empty_is_russian_error(tmp_path):
    path = tmp_path / "empty.srt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(GeminiInputError, match="не определён"):
        gp.detect_srt_language(path)


def test_detect_srt_language_mixed_is_russian_error(tmp_path):
    path = tmp_path / "mixed.srt"
    path.write_text("Привет друзья. hello friends.\n", encoding="utf-8")
    with pytest.raises(GeminiInputError, match="смешанный"):
        gp.detect_srt_language(path)


# --- и) число референсов -------------------------------------------------------------------------

def test_ref_count_six_files_allows_1_to_6_and_rejects_7(tmp_path):
    refs = tmp_path / "refs"
    _make_refs(refs, 6)
    limit = gp.count_ref_files(refs, "niche")
    assert limit == 6
    assert gp._check_refs([1, 6], limit) == ([1, 6], [])
    value, reasons = gp._check_refs([7], limit)
    assert value is None
    assert reasons == ["refs: номер вне 1…6 (в нише 6 референсов)"]


def test_ref_count_three_files_rejects_4(tmp_path):
    refs = tmp_path / "refs"
    _make_refs(refs, 3)
    limit = gp.count_ref_files(refs, "niche")
    value, reasons = gp._check_refs([4], limit)
    assert value is None
    assert reasons == ["refs: номер вне 1…3 (в нише 3 референсов)"]


def test_ref_count_is_capped_at_graph_limit(tmp_path):
    refs = tmp_path / "refs"
    _make_refs(refs, 12)
    assert gp.count_ref_files(refs, "niche") == gp.REF_MAX == 10


def test_ref_count_rejects_gaps_and_missing_files(tmp_path):
    refs = tmp_path / "refs"
    _make_refs(refs, 3)
    (refs / "ref_2.png").unlink()
    with pytest.raises(GeminiInputError, match="подряд"):
        gp.count_ref_files(refs, "niche")
    with pytest.raises(GeminiInputError, match="refs"):
        gp.count_ref_files(tmp_path / "no_such_dir", "niche")


def test_default_refs_range_keeps_old_reason():
    value, reasons = gp._check_refs([11])
    assert value is None
    assert reasons == [gp.REF_REASON_RANGE] == ["refs: номер вне 1…10"]


# --- й) профиль cartoon_photos: ключи читаются, без ключей поведение прежнее -------------------------

def test_cartoon_profile_reports_relaxed_rules_and_text_language():
    relaxed, text_language = gp.profile_relaxation(_profile_lines(), "cartoon_photos")
    assert relaxed == RELAX_ALL
    assert text_language == "srt"


def test_cartoon_refs_folder_counts_six():
    assert gp.count_ref_files(NICHE_DIR / "refs", "cartoon_photos") == 6


def test_profile_without_new_keys_behaves_as_before():
    relaxed, text_language = gp.profile_relaxation(["type: video", "refs_enabled: false"], "ltx25")
    assert relaxed == frozenset()
    assert text_language is None


def test_frame_text_without_text_language_is_error():
    with pytest.raises(GeminiInputError, match="text_language"):
        gp.profile_relaxation(["relaxed_rules: [frame_text]"], "x")
