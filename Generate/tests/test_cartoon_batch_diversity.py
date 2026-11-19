"""Разнообразие батча cartoon_photos (batch_passage): вид кадра в setting, канон героя, «free of X» и
монотонность соседних клипов в очереди исправлений. У прочих профилей поведение прежнее. Сеть не используется."""
import threading

from Generate.core import config_loader
from Generate.core import gemini_prompts as gp
from Generate.core import prompt_pipeline as pp
from Generate.core.gemini_prompts import PromptConfig

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
