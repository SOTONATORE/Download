"""Разнообразие батча cartoon_photos (batch_passage): вид кадра в setting, канон героя, «free of X» и
монотонность соседних клипов в очереди исправлений. У прочих профилей поведение прежнее. Сеть не используется."""
import re
import threading
from types import SimpleNamespace

import pytest

from Generate.core import config_loader
from Generate.core import gemini_prompts as gp
from Generate.core import prompt_check as pc
from Generate.core import prompt_pipeline as pp
from Generate.core.gemini_prompts import PromptConfig
from Generate.core.srt_parser import Segment

STYLE = "Soft light."
SETTING_FORMAT = [{"segment_index": int, "setting": str, "prompt": str}]
HERO_OK = ("A wide scene with the main character crouching by a fossil. The main character has a smooth "
           "pure-white round head, small black dot eyes and a plain solid-black stick body.")
HERO_NO_STICK = "A wide scene with the main character crouching by a fossil, a smooth pure-white round head."


def _cfg(tmp_path, passage=True):
    style = tmp_path / "style.md"
    style.write_text("Style guide for tests.", encoding="utf-8")
    kwargs = {}
    if passage:
        kwargs["batch_passage"] = True
        kwargs["response_format"] = SETTING_FORMAT
    return PromptConfig(style_file=str(style), media="image", batch_size=3, **kwargs)


def _entry(text, setting=None, rework_round=0, pending=None):
    """Запись состояния клипа: промпт со стилем и setting в extra, как пишет конвейер."""
    entry = {"num": 0, "prompt": f"{text}. {STYLE}", "status": "ready", "num_frames": 1,
             "rework_round": rework_round}
    if setting is not None:
        entry["extra"] = {"setting": setting}
    if pending:
        entry["rework_pending"] = pending
    return entry


# --- системная инструкция: разнообразие видов кадра и формат setting ----------

def test_system_instruction_has_visual_mode_variety_and_setting_contract(tmp_path):
    system = gp.build_system_instruction(_cfg(tmp_path))
    assert "visual mode" in system
    assert "<visual mode>: <place or subject>" in system
    assert "shot framing" in system and "focal subject" in system and "composition" in system
    assert "centred stickman pose" in system
    assert "cutaways" in system and "balance scales" in system


def test_system_instruction_without_batch_passage_has_no_visual_mode_contract(tmp_path):
    system = gp.build_system_instruction(_cfg(tmp_path, passage=False))
    assert "<visual mode>" not in system
    assert "visual mode" not in system


# --- крупные планы: профиль с max_characters разрешает предметы и руки, лица запрещены -------

def test_close_faces_line_depends_on_max_characters(tmp_path):
    """max_characters=4 (cartoon_photos): строка разрешает крупные планы рук и предметов;
    без max_characters прежняя строка «Never show close-ups» остаётся на месте.
    У каждого конфига свой RulesValues: PromptConfig синхронизирует max_characters с rules."""
    style = _write_style(tmp_path)
    niche = gp.build_system_instruction(PromptConfig(
        style_file=str(style), media="image",
        rules=gp.RulesValues(max_characters=4, close_faces_allowed=False)))
    plain = gp.build_system_instruction(PromptConfig(
        style_file=str(style), media="image",
        rules=gp.RulesValues(close_faces_allowed=False)))
    assert gp._CLOSE_FACES_OBJECTS_OK_LINE in niche
    assert gp._CLOSE_FACES_LINE not in niche
    assert "close-ups of hands, feet, objects, food, plants and cutaways are allowed" in niche
    assert gp._CLOSE_FACES_LINE in plain
    assert gp._CLOSE_FACES_OBJECTS_OK_LINE not in plain


def _write_style(tmp_path):
    style = tmp_path / "style.md"
    style.write_text("Style guide for tests.", encoding="utf-8")
    return style


def test_same_sentence_note_only_when_consecutive_segments_share_sentence():
    # одно предложение на сегменты 1 и 2; сегмент 3 — другое предложение
    seg_a = gp.Sentence(number=1, seg_nums=[1, 2], text="Merchants opened their stalls and carts rolled in.", spans={})
    seg_b = gp.Sentence(number=2, seg_nums=[3], text="Rain fell.", spans={})
    by_segment = {1: seg_a, 2: seg_a, 3: seg_b}
    assert gp._shares_sentence([1, 2], by_segment) is True
    assert gp._shares_sentence([2, 3], by_segment) is False


def test_fragment_without_terminal_punctuation_needs_full_sentence():
    assert gp._needs_full_sentence("and carts rolled in from the hills") is True   # нет точки
    assert gp._needs_full_sentence("Rain fell.") is True                          # 2 слова
    assert gp._needs_full_sentence("Farmers dig the fields at dawn every single year.") is False


# --- канон главного героя --------------------------------------------------

def test_hero_missing_round_head_or_stick_is_flagged_when_batch_passage(tmp_path):
    cfg = _cfg(tmp_path)
    found = gp._find_all_problems(HERO_NO_STICK, cfg)
    assert gp._PARSE_PREFIX + gp._HERO_CANON_REASON in found


def test_hero_with_full_canon_is_not_flagged_when_batch_passage(tmp_path):
    found = gp._find_all_problems(HERO_OK, _cfg(tmp_path))
    assert not any(gp._HERO_CANON_REASON in p for p in found)


def test_hero_canon_is_ignored_without_batch_passage(tmp_path):
    found = gp._find_all_problems(HERO_NO_STICK, _cfg(tmp_path, passage=False))
    assert not any(gp._HERO_CANON_REASON in p for p in found)


def test_hero_hint_is_english_and_describes_canon():
    hint = gp._parse_hint_en(gp._PARSE_PREFIX + gp._HERO_CANON_REASON)
    assert "round head" in hint and "stick body" in hint and "dot eyes" in hint
    assert "Fix the problem found" not in hint


# --- «free of X» -------------------------------------------------------------

def test_free_of_words_and_smoke_are_flagged_with_positive_hint(tmp_path):
    cfg = _cfg(tmp_path)
    for phrase in ("free of words", "free of smoke", "free of smiling face"):
        found = gp._find_all_problems(f"A bare wooden table in a hut, {phrase}.", cfg)
        assert gp._PARSE_PREFIX + gp._FREE_OF_REASON in found, phrase
    hint = gp._parse_hint_en(gp._PARSE_PREFIX + gp._FREE_OF_REASON)
    assert "positive description" in hint and "blank dark screen" in hint
    assert "Fix the problem found" not in hint


def test_free_of_visual_nouns_ignored_without_batch_passage(tmp_path):
    found = gp._find_all_problems("A bare wooden table in a hut, free of words.", _cfg(tmp_path, passage=False))
    assert not any(gp._FREE_OF_REASON in p for p in found)


def test_free_of_non_visual_phrase_is_not_flagged(tmp_path):
    found = gp._find_all_problems("A bare wooden table in a hut, free of quotation marks.", _cfg(tmp_path))
    assert not any(gp._FREE_OF_REASON in p for p in found)


# --- монотонность соседних клипов ------------------------------------------

def test_same_setting_on_adjacent_clips_is_queued_and_first_is_kept():
    prompts = {
        "1": _entry("Potato rows in a field under a bright sky", "wide: potato rows"),
        "2": _entry("Farmers lift tubers from the soil one by one", "wide: potato rows"),
    }
    queued = pp.mark_monotony_for_rework(prompts, [1, 2], STYLE)
    assert queued == [2]
    assert prompts["1"].get("rework_pending") is None
    assert prompts["2"]["rework_pending"]


def test_distinct_compositions_are_left_untouched():
    prompts = {
        "1": _entry("Potato rows in a field under a bright sky", "wide: potato rows"),
        "2": _entry("Macro view of hands peeling a warm tuber", "macro: hands peeling tuber"),
        "3": _entry("Cutaway of soil with roots and a fossil stone", "cutaway: soil roots"),
    }
    assert pp.mark_monotony_for_rework(prompts, [1, 2, 3], STYLE) == []
    assert all(not v.get("rework_pending") for v in prompts.values())


def test_near_identical_scene_is_queued_even_with_renamed_setting():
    body = "A wide field of potato rows under a bright sky with two farmers carrying baskets near a red barn"
    prompts = {
        "1": _entry(body, "wide: potato rows"),
        "2": _entry(body.replace("two", "three"), "wide: tuber field"),   # одно слово другое, сцена та же
    }
    assert pp.mark_monotony_for_rework(prompts, [1, 2], STYLE) == [2]


def test_repeated_hero_sentence_does_not_make_scenes_similar():
    prompts = {
        "1": _entry("The main character has a round head. Potatoes lie in a wicker basket on a table", "a"),
        "2": _entry("The main character has a bigger round head. Sun rises over distant hills at dawn", "b"),
    }
    assert pp.mark_monotony_for_rework(prompts, [1, 2], STYLE) == []


def test_window_is_three_clips_so_fourth_clip_is_not_compared_with_first():
    prompts = {
        "1": _entry("Potato rows in a field", "wide: potato rows"),
        "2": _entry("Cutaway of soil with roots", "cutaway: soil roots"),
        "3": _entry("Balance scale weighing mass", "scale: mass vs calories"),
        "4": _entry("Macro view of hands peeling a tuber", "wide: potato rows"),
    }
    assert pp.mark_monotony_for_rework(prompts, [1, 2, 3, 4], STYLE) == []


def test_clip_at_repair_limit_or_already_queued_is_skipped():
    prompts = {
        "1": _entry("Potato rows in a field", "wide: potato rows"),
        "2": _entry("Farmers lift tubers", "wide: potato rows", rework_round=pp.gp.REPAIR_ROUNDS),
        "3": _entry("Farmers pour water", "wide: potato rows", pending=["x"]),
    }
    assert pp.mark_monotony_for_rework(prompts, [1, 2, 3], STYLE) == []


def _stage_args(state, photo_cfg):
    return dict(photo_nums={1, 2, 3}, segments=[], cfg=photo_cfg, photo_cfg=photo_cfg,
                profile="fake_video", photo_profile="fake_photo", timing=None, watch=object(),
                prompts_path="unused.json", gemini_options={}, errors=[], aborts=[],
                state=state, lock=threading.Lock())


def _profile(monkeypatch):
    monkeypatch.setattr(config_loader, "profile_exists", lambda name: True)
    monkeypatch.setattr(config_loader, "_read_profile", lambda name: {"fail_on_duplicate_prompts": True})


def test_stage_queues_monotony_in_batch_and_reworks_once(monkeypatch, tmp_path):
    _profile(monkeypatch)
    state = {"prompts": {
        "1": _entry("Potato rows in a field", "wide: potato rows"),
        "2": _entry("Farmers pick tubers by hand", "wide: potato rows"),
        "3": _entry("Scale balancing mass and calories", "scale: mass vs calories"),
    }}
    calls = []

    def fake_photo_step(needed, **kw):
        calls.append(list(needed))
        for n in needed:  # модель вернула новую композицию и новый setting
            e = kw["state"]["prompts"][str(n)]
            e["prompt"] = f"Cutaway of soil roots number {n}. {STYLE}"
            e["extra"] = {"setting": "cutaway: soil roots"}
            e["rework_pending"] = []
            e["rework_round"] = 1

    monkeypatch.setattr(pp, "_photo_step", fake_photo_step)
    monkeypatch.setattr(pp.go, "ManagedTransport", lambda *a, **k: object())
    monkeypatch.setattr(pp.go, "atomic_write_json", lambda path, data: None)
    monkeypatch.setattr(pp, "read_state", lambda path: state)
    out = pp._duplicate_rework([1, 2, 3], **_stage_args(state, _batch_cfg(tmp_path)))
    assert calls == [[2]]
    assert out["prompts"]["2"]["extra"]["setting"] == "cutaway: soil roots"


def test_stage_does_not_queue_monotony_without_batch_passage(monkeypatch, tmp_path):
    _profile(monkeypatch)
    state = {"prompts": {
        "1": _entry("Potato rows in a field", "wide: potato rows"),
        "2": _entry("Farmers pick tubers by hand", "wide: potato rows"),
    }}

    def forbidden(*a, **k):
        raise AssertionError("без batch_passage монотонность не ставит клипы в очередь")

    monkeypatch.setattr(pp, "_photo_step", forbidden)
    monkeypatch.setattr(pp, "read_state", lambda path: state)
    pp._duplicate_rework([1, 2], **_stage_args(state, _plain_cfg(tmp_path)))
    assert all(not v.get("rework_pending") for v in state["prompts"].values())


def _batch_cfg(tmp_path):
    return _cfg(tmp_path, passage=True)


def _plain_cfg(tmp_path):
    return _cfg(tmp_path, passage=False)


# --- batch_size профиля = 15 для обычных батчей и пакетов исправлений -------------------

def _profile_yaml(tmp_path, monkeypatch, name, body):
    """Профиль в tmp_path: profile_dir(name) указывает на tmp_path/name; body — текст profile.yaml (или None — без файла)."""
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    if body is not None:
        (folder / "profile.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setattr(config_loader, "profile_dir", lambda p: tmp_path / p)


def test_load_profile_batch_size_missing_file_or_key_is_none(tmp_path, monkeypatch):
    _profile_yaml(tmp_path, monkeypatch, "nofile", None)
    assert config_loader.load_profile_batch_size("nofile") is None
    _profile_yaml(tmp_path, monkeypatch, "nokey", "type: image\n")
    assert config_loader.load_profile_batch_size("nokey") is None


def test_load_profile_batch_size_reads_fifteen(tmp_path, monkeypatch):
    _profile_yaml(tmp_path, monkeypatch, "p15", "type: image\nbatch_size: 15\n")
    assert config_loader.load_profile_batch_size("p15") == 15


@pytest.mark.parametrize("value", ["0", "-3", "true", "'15'", "2.5"])
def test_load_profile_batch_size_rejects_bad_values_in_russian(tmp_path, monkeypatch, value):
    _profile_yaml(tmp_path, monkeypatch, "bad", f"type: image\nbatch_size: {value}\n")
    with pytest.raises(gp.GeminiInputError, match="batch_size"):
        config_loader.load_profile_batch_size("bad")


def test_prompt_config_fix_batch_size_default_none_and_validated(tmp_path):
    assert _cfg(tmp_path).fix_batch_size is None
    style = str(tmp_path / "style.md")
    with pytest.raises(gp.GeminiInputError):
        PromptConfig(style_file=style, media="image", fix_batch_size=0)
    with pytest.raises(gp.GeminiInputError):
        PromptConfig(style_file=style, media="image", fix_batch_size=True)
    assert PromptConfig(style_file=style, media="image", fix_batch_size=15).fix_batch_size == 15


def _wave_job(cfg, clips):
    builder = SimpleNamespace(cfg=cfg, texts={}, by_segment={})
    return gp._WaveJob("видео", builder, [], None, "m", "style", None, rework=clips)


def test_wave_job_default_fix_size_is_25_and_fifteen_when_profile_sets_it(tmp_path):
    clips = [gp.PromptResult(n, f"p{n}") for n in range(1, 41)]
    job = _wave_job(_cfg(tmp_path), list(clips))
    assert job.fix_batch_size == gp.FIX_BATCH_SIZE == 25
    job.start_round()
    assert [len(c) for _no, c in job.round_pkgs] == [25, 15]
    job15 = _wave_job(replace_cfg(tmp_path, fix_batch_size=15), list(clips))
    job15.start_round()
    assert [len(c) for _no, c in job15.round_pkgs] == [15, 15, 10]


def replace_cfg(tmp_path, **kw):
    from dataclasses import replace
    return replace(_cfg(tmp_path), **kw)


def test_pool_picks_full_fix_package_of_fifteen_not_twenty_five(tmp_path):
    clips = [gp.PromptResult(n, f"p{n}") for n in range(1, 16)]
    job = _wave_job(replace_cfg(tmp_path, fix_batch_size=15), list(clips))
    pool = gp.SharedBatchPool({"видео": 1})
    pool._jobs = {"видео": job}
    job.round_pkgs = []
    fix = pool._pick_fix("видео", remainder=False)
    assert fix is not None and len(fix[1]) == 15
    # при размере 25 пакет из 15 клипов — ещё не полный
    job25 = _wave_job(_cfg(tmp_path), list(clips))
    pool._jobs = {"видео": job25}
    assert pool._pick_fix("видео", remainder=False) is None


def test_fix_batch_size_default_keeps_existing_25_pool_behaviour(tmp_path):
    clips = [gp.PromptResult(n, f"p{n}") for n in range(1, 26)]
    job = _wave_job(_cfg(tmp_path), list(clips))
    pool = gp.SharedBatchPool({"видео": 1})
    pool._jobs = {"видео": job}
    fix = pool._pick_fix("видео", remainder=False)
    assert fix is not None and len(fix[1]) == 25


# --- Type A / Type B: строка одежды ----------------------------------------------------------

def test_clothing_line_is_type_ab_only_with_max_characters(tmp_path):
    rv = gp.RulesValues()
    text_cc = gp._content_rules("image", 4, rv)
    assert "Type B" in text_cc and "Type A" in text_cc and "plain solid-black stick torso" in text_cc
    assert "Never show anyone naked, shirtless or undressed." in text_cc
    assert "Every person wears named clothing" not in text_cc
    text_none = gp._content_rules("image", None, gp.RulesValues())
    assert gp._NUDITY_LINE in text_none
    assert "Type A" not in text_none


# --- язык вывода: антифлуд и русский пример только для русского текста -----------------------

def _lang_cfg(tmp_path, *, dangling, passage, frame_language, text_language="srt"):
    rules = gp.RulesValues(dangling_text_check=dangling)
    return PromptConfig(style_file=str(tmp_path / "s.md"), media="image", batch_size=3, rules=rules,
                        batch_passage=passage, text_language=text_language, frame_language=frame_language,
                        response_format=SETTING_FORMAT)


def test_output_language_unchanged_without_dangling_check(tmp_path):
    cfg = _lang_cfg(tmp_path, dangling=False, passage=True, frame_language=None, text_language=None)
    assert gp._output_language_sentence(cfg) == "Always write the output in English."


def test_batch_framing_is_before_formatting_rules_and_language_neutral_for_english(tmp_path):
    out = gp._output_language_sentence(_lang_cfg(tmp_path, dangling=True, passage=True, frame_language="English"))
    assert "Most frames are text-free visual illustrations" in out
    assert "label-card pattern" in out and "header bands" in out and "balance scales" in out
    assert "lone adjective without its noun" in out
    assert out.index("Most frames are text-free") < out.index("On-frame text must be a complete phrase")
    assert "ЗЕМЛЯНАЯ" not in out and "земляной" not in out
    assert not re.search("[\u0400-\u04FF]", out)


def test_non_russian_language_has_no_cyrillic_example_for_spanish(tmp_path):
    out = gp._output_language_sentence(_lang_cfg(tmp_path, dangling=True, passage=True, frame_language="Spanish"))
    assert "for example, write" not in out
    assert not re.search("[\u0400-\u04FF]", out)
    assert "nominative singular or dictionary form" in out


def test_russian_frame_language_keeps_russian_example(tmp_path):
    out = gp._output_language_sentence(_lang_cfg(tmp_path, dangling=True, passage=True, frame_language="Russian"))
    assert "ЗЕМЛЯНАЯ ГРУША" in out and "земляной грушей" in out


def test_framing_only_for_batch_passage(tmp_path):
    out = gp._output_language_sentence(_lang_cfg(tmp_path, dangling=True, passage=False, frame_language="Russian"))
    assert "Most frames are text-free" not in out
    assert "ЗЕМЛЯНАЯ ГРУША" in out     # русский пример остаётся для русского текста


# --- CJK: полноширинные завершители делят предложения ----------------------------------------

def test_cjk_full_width_enders_split_sentences():
    segs = [Segment(1, 0, 1000, "今日は晴れです。"), Segment(2, 1000, 2000, "明日は雨ですか？"),
            Segment(3, 2000, 3000, "本当！")]
    sentences, by_segment = gp.merge_segments_into_sentences(segs)
    assert [s.text for s in sentences] == ["今日は晴れです。", "明日は雨ですか？", "本当！"]
    assert by_segment[2].number == 2


# --- TEXTLESS_FROM_ROUND = 2 -----------------------------------------------------------------

def test_textless_rework_starts_on_second_round(tmp_path):
    assert gp.TEXTLESS_FROM_ROUND == 2
    cfg = _lang_cfg(tmp_path, dangling=True, passage=True, frame_language="English")
    pending = [pc.make_problem(pc.PROBLEM_DANGLING_TEXT, "x")]
    assert gp._textless_round(cfg, pending, 1) is False
    assert gp._textless_round(cfg, pending, 2) is True


# --- короткий фрагмент при дубликате получает смысл сцены -----------------------------------

def test_duplicate_fragment_gets_sentence_context_only_when_short():
    frag = gp.PromptResult(5, "p", rework_pending=[pc.make_problem(pc.PROBLEM_DUPLICATE_PROMPT, "3")])
    sen = SimpleNamespace(text="Going home was hard, быть удобно.")
    texts = {5: "быть удобно."}
    note = gp._rework_subtitle_note(frag, texts, {5: sen})
    assert "Subtitle text for segment_index 5: быть удобно." in note
    assert "Sentence (for visual meaning only, never copy words from it into on-frame text):" in note
    # без контекста — прежняя строка
    assert "Sentence (for visual" not in gp._rework_subtitle_note(frag, texts)
    # длинная законченная реплика — контекст не добавляется
    long_text = {5: "This line is a complete sentence about the harvest of the year."}
    long_sen = SimpleNamespace(text=long_text[5])
    assert "Sentence (for visual" not in gp._rework_subtitle_note(frag, long_text, {5: long_sen})


# --- per-clip проверки _batch_style_reasons (batch_passage) ----------------------------------

def _reasons(text, setting=""):
    return gp._batch_style_reasons(text, setting)


def test_blank_card_without_reading_is_flagged_and_card_with_text_is_not():
    assert gp._BLANK_CARD_REASON in _reasons("A medium shot of a farmer beside a label card on a fence.", "medium: farmer")
    assert gp._BLANK_CARD_REASON in _reasons("A header band across the top, nothing written.", "")
    assert gp._BLANK_CARD_REASON not in _reasons(
        "A wide field with a label card reading EARTH PEAR in bold lettering.", "wide: field")


def test_text_on_ordinary_scene_is_flagged_but_analytical_and_numbers_are_not():
    plaque = "A wide field with a plaque reading EARTH PEAR in bold lettering."
    assert gp._UNAUTH_TEXT_REASON in _reasons(plaque, "wide: field")
    assert gp._UNAUTH_TEXT_REASON in _reasons(plaque, "")                  # вид из начала промпта
    assert gp._UNAUTH_TEXT_REASON not in _reasons(
        "A map of the valley with labels reading EARTH PEAR in bold lettering.", "map: valley")
    assert gp._UNAUTH_TEXT_REASON not in _reasons(
        "A wide field with a plaque reading 1842 in bold lettering.", "wide: field")
    assert gp._UNAUTH_TEXT_REASON not in _reasons(
        "A wide field with a plaque reading XVIII CENTURY in bold lettering.", "wide: field")


def test_zero_phrase_is_flagged_positive_wording_is_not():
    assert gp._ZERO_PHRASE_REASON in _reasons("A macro shot with zero people in the frame.", "macro: soil")
    assert gp._ZERO_PHRASE_REASON in _reasons("A table with zero text on it.", "")
    assert gp._ZERO_PHRASE_REASON not in _reasons("A table with an empty surface and one bowl.", "")


def test_type_b_clothing_on_type_a_hero_is_flagged_only_in_hero_sentence():
    bad = "A wide scene with the main character wearing a red shirt and trousers by a fossil."
    assert gp._TYPE_B_ON_HERO_REASON in _reasons(bad, "wide: fossil")
    ok = ("A wide scene with the main character crouching by a fossil. The main character has a smooth pure-white "
          "round head and a plain solid-black stick body. A farmer in a shirt and trousers stands nearby.")
    assert gp._TYPE_B_ON_HERO_REASON not in _reasons(ok, "wide: fossil")


def test_contradictory_or_floating_mouth_is_flagged():
    assert gp._MOUTH_REASON in _reasons("The main character stands, a mouth drawn above the head.", "wide: road")
    both = ("A medium shot of the main character with a closed-line smile and a wavy worried mouth.")
    assert gp._MOUTH_REASON in _reasons(both, "medium: door")
    group = ("A medium shot of two villagers, one with a closed-line smile and one with a wavy worried mouth.")
    assert gp._MOUTH_REASON not in _reasons(group, "medium: door")
    assert gp._MOUTH_REASON not in _reasons("A medium shot of the main character with a closed-line smile.", "medium: door")


def test_long_label_is_flagged_and_short_or_numeric_label_is_not():
    long_one = "A plaque reading THE QUIET OLD FARM ON THE HILL in bold lettering."
    assert gp._LONG_LABEL_REASON in _reasons(long_one, "cutaway: farm")
    assert gp._LONG_LABEL_REASON not in _reasons("A sign reading GRUSHA in bold lettering.", "cutaway: farm")
    assert gp._LONG_LABEL_REASON not in _reasons(
        "A sign reading THE QUIET OLD FARM ON THE HILL 1842 in bold lettering.", "cutaway: farm")


def test_clean_scene_has_no_reasons():
    clean = ("A macro shot of hands peeling a tuber on a wooden board. The main character has a smooth pure-white "
             "round head and a plain solid-black stick body.")
    assert _reasons(clean, "macro: hands peeling tuber") == []


def test_guards_run_only_when_passage_and_english_hints_exist():
    assert "card" in gp._parse_hint_en(gp._PARSE_PREFIX + gp._BLANK_CARD_REASON).lower()
    assert "26 characters" in gp._parse_hint_en(gp._PARSE_PREFIX + gp._LONG_LABEL_REASON)
    assert gp._parse_hint_en(gp._PARSE_PREFIX + gp._ZERO_PHRASE_REASON).startswith("Do not write zero")


# --- монотонность: серия одного вида кадра и очистка шаблонного текста -----------------------

def test_boilerplate_character_text_does_not_count_as_scene_words(tmp_path):
    entry = _entry("A macro shot of tuber in a sack. The stick figure has warm natural skin tones and "
                   "small black dot eyes. Smooth closed outlines and flat matte fills.", setting="macro: tuber")
    words = pp._scene_words(entry, STYLE)
    assert "tuber" in words and "warm" not in words and "outlines" not in words and "matte" not in words


# общие слова + два уникальных на клип: попарное сходство 15/19 ≈ 0.79 — выше 0.65 и ниже окна 0.85
_RUN_COMMON = ("farmer sorts green pears in orchard under tree beside old wooden fence near stone wall")
_RUN_UNIQUE = ["alpha beta", "gamma delta", "epsilon zeta"]


def _run_prompts(settings):
    prompts = {}
    for n, (uniq, setting) in enumerate(zip(_RUN_UNIQUE, settings), start=1):
        prompts[str(n)] = _entry(f"{_RUN_COMMON} {uniq}", setting=setting)
        prompts[str(n)]["num"] = n
    return prompts


def test_three_consecutive_same_mode_clips_with_overlap_are_queued():
    prompts = _run_prompts(["macro: pears one", "macro: pears two", "macro: pears three"])
    queued = pp.mark_monotony_for_rework(prompts, [1, 2, 3], STYLE)
    assert queued == [3]
    assert pc.problem_code(prompts["3"]["rework_pending"][0]) == pc.PROBLEM_DUPLICATE_PROMPT


def test_two_same_mode_clips_or_different_mode_are_not_run_queued():
    prompts = _run_prompts(["macro: pears one", "macro: pears two", "wide: pears three"])
    assert pp.mark_monotony_for_rework(prompts, [1, 2, 3], STYLE) == []
