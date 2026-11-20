"""Тесты _scene_words и mark_monotony_for_rework: типовой текст персонажей не должен считаться сценой.

Русский: проверяем, что канон героя и взрослых персонажей (style.md повторяет его в каждом промпте) вырезается
независимо от порядка клауз, что реальные предметы и действия остаются, что различные сцены не помечаются как клоны
и что настоящие клоны по-прежнему ставятся в очередь.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

GENERATE_DIR = Path(__file__).resolve().parents[1]
if str(GENERATE_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATE_DIR))

# Русский: gemini_orchestrator импортирует core/timing.py; если файла нет в архиве, подставляем заглушку.
# Для _scene_words и mark_monotony_for_rework она не нужна; реальный модуль, если он есть, не подменяется.
if not (GENERATE_DIR / "core" / "timing.py").exists() and "core.timing" not in sys.modules:
    _timing_stub = types.ModuleType("core.timing")

    def _compute_timings(*args, **kwargs):
        raise RuntimeError("core/timing.py отсутствует в архиве: заглушка для тестов")

    _timing_stub.compute_timings = _compute_timings
    sys.modules["core.timing"] = _timing_stub

from core import prompt_pipeline as pp  # noqa: E402

# Русский: стиль, дописанный к каждому клипу через «. » (как attach_style); _raw_prompt снимает его обратно
STYLE = ("Hand-drawn 2D cartoon, warm cream backdrop, flat matte earthy fills, "
         "smooth closed black ink outlines.")

# Русский: слова канона, которые не должны попасть в _scene_words
MAIN_CANON_WORDS = {"pure", "round", "head", "ink", "dot", "eyebrows", "torso", "stick", "arms", "legs",
                    "percent", "twenty", "smile", "solid", "outline", "fine"}
ADULT_CANON_WORDS = {"proportional", "cartoon", "limbs", "tones", "skin", "fingers", "rounded", "hair", "nose",
                     "curved", "eyebrows", "closed", "smile", "percent", "fifth", "body", "height", "muted",
                     "rust", "plaid", "flannel", "overalls", "denim", "boots", "brown", "medium", "blue"}


def _entry(prompt: str, setting: str) -> dict:
    """Клип в формате prompts: текст с дописанным стилем и setting в extra."""
    return {"prompt": f"{prompt}. {STYLE}", "extra": {"setting": setting}}


def _words(prompt: str, setting: str = "wide: test") -> frozenset:
    return pp._scene_words(_entry(prompt, setting), STYLE)


# Русский: клипы 612 и 615 — канон героя написан через «, with a large smooth pure-white round head...»
CLIP_612 = ("Wide view of a calm garden patch with rows of lettuce under soft morning light. "
            "The main character stands at about twenty percent of image height at the right edge, "
            "with a large smooth pure-white round head with a crisp medium-fine black ink outline, "
            "small solid black dot eyes, expressive thin curved eyebrows, a closed-line smile, "
            "and a plain solid-black stick torso with thin black stick arms, simple black stick hands "
            "and thin black stick legs.")
CLIP_615 = ("Wide view of a marked garden bed with taut twine between corner stakes and fresh soil ridges. "
            "The main character stands at about twenty percent of image height at the right edge, "
            "with a large smooth pure-white round head with a crisp medium-fine black ink outline, "
            "small solid black dot eyes, expressive thin curved eyebrows, a closed-line smile, "
            "and a plain solid-black stick torso with thin black stick arms, simple black stick hands "
            "and thin black stick legs.")

# Русский: фермер и макро-руки — типовой текст взрослых персонажей в другом порядке клауз
FARMER_SHOP = ("A farmer stands in a farm shop beside a wooden crate of carrots. "
               "The farmer has proportional cartoon limbs, warm natural skin tones, simple brown hair, "
               "small black dot eyes, a small curved-line nose, raised eyebrows and a closed-line smile. "
               "The farmer wears a muted rust plaid flannel shirt, medium-blue denim overalls and brown boots.")
FARMER_CELLAR = ("A farmer kneels in a cellar holding a golden potato up to the light. "
                 "The farmer has proportional cartoon limbs, warm natural skin tones, simple brown hair, "
                 "small black dot eyes, a small curved-line nose, raised eyebrows and a closed-line smile. "
                 "The farmer wears a muted rust plaid flannel shirt, medium-blue denim overalls and brown boots.")
HANDS = ("Macro view of two hands holding a golden potato over a warm wooden board, the right hand guiding "
         "a vegetable peeler along the skin so one thin curled peel drops onto the board. "
         "The hands have warm natural skin tones, simple rounded fingers and smooth closed outlines, "
         "with a few pale peel curls scattered beside the tuber.")


def test_main_character_canon_without_has_clause_is_stripped_and_scene_kept():
    words_612 = _words(CLIP_612, "wide: calm garden patch")
    words_615 = _words(CLIP_615, "wide: marked garden bed")
    # Русский: реальная сцена остаётся
    assert {"calm", "garden", "patch", "lettuce"} <= words_612
    assert {"marked", "garden", "bed", "twine", "stakes", "corner"} <= words_615
    # Русский: канон героя не попадает в слова сцены
    assert not (MAIN_CANON_WORDS & words_612)
    assert not (MAIN_CANON_WORDS & words_615)


def test_adult_and_macro_hand_canon_is_stripped_and_props_kept():
    cellar = _words(FARMER_CELLAR, "medium: farmer in cellar holding golden potato")
    assert {"cellar", "golden", "potato", "kneels", "light"} <= cellar
    assert not (ADULT_CANON_WORDS & cellar)

    shop = _words(FARMER_SHOP, "medium: farmer in farm shop with crate")
    assert {"farm", "shop", "crate", "carrots"} <= shop
    assert not (ADULT_CANON_WORDS & shop)

    hands = _words(HANDS, "macro: hands peeling jerusalem artichoke tuber")
    assert {"peeler", "peel", "board", "tuber", "potato"} <= hands
    # Русский: «skin» здесь — часть сцены («along the skin»), поэтому проверяем только слова канона рук
    assert not ({"fingers", "rounded", "tones", "natural", "closed", "outlines"} & hands)


# Русский: три разных клипа между первым и последним — анти-клон сравнивает только клипы вне локального окна
# MONOTONY_WINDOW (два предыдущих), поэтому без них первый клип не попадает в проверку последнего
FILLERS = [
    ("Wide view of a red barn yard with a wooden cart and chickens pecking grain.", "wide: barn yard"),
    ("Medium scene of a chef whisking eggs over a copper pan.", "medium: chef in kitchen corner"),
    ("Close view of burlap sacks stacked beside a stone wall.", "close: potato sacks in cellar"),
]


def _monotony_sequence(first: tuple, last: tuple):
    """Клипы 1..5: первый и последний задаются (текст, setting), между ними три разных клипа."""
    prompts = {"1": _entry(first[0], first[1])}
    for num, (text, setting) in enumerate(FILLERS, start=2):
        prompts[str(num)] = _entry(text, setting)
    prompts["5"] = _entry(last[0], last[1])
    queued = pp.mark_monotony_for_rework(prompts, [1, 2, 3, 4, 5], STYLE)
    return queued, prompts


def test_monotony_does_not_flag_distinct_main_character_scenes():
    queued, prompts = _monotony_sequence((CLIP_612, "wide: calm garden patch"),
                                         (CLIP_615, "wide: marked garden bed"))
    assert queued == []
    assert "rework_pending" not in prompts["5"]


def test_monotony_does_not_flag_distinct_farmer_scenes():
    queued, prompts = _monotony_sequence((FARMER_SHOP, "medium: farmer in farm shop with crate"),
                                         (FARMER_CELLAR, "medium: farmer in cellar holding golden potato"))
    assert queued == []
    assert "rework_pending" not in prompts["5"]


def test_monotony_still_flags_genuine_clone_beyond_local_window():
    setting = "macro: hands peeling jerusalem artichoke tuber"
    # Русский: почти та же сцена — отличаются только два слова
    clone_b = HANDS.replace("golden potato", "knobby potato")
    queued, prompts = _monotony_sequence((HANDS, setting), (clone_b, setting))
    assert queued == [5]
    assert prompts["5"]["rework_pending"]
    assert "rework_pending" not in prompts["1"]


def test_monotony_thresholds_are_unchanged():
    assert pp.CLONE_WINDOW == 100
    assert pp.CLONE_JACCARD == 0.65
    assert pp.MONOTONY_WINDOW == 3
    assert pp.MONOTONY_JACCARD == 0.85
    assert pp.MONOTONY_RUN == 3
    assert pp.MONOTONY_RUN_JACCARD == 0.65


@pytest.mark.parametrize("prompt", [CLIP_612, FARMER_SHOP, HANDS])
def test_scene_words_never_contains_style_filler(prompt):
    # Русский: стиль (cream, flat, matte, outlines) отрезается вместе с канон-фразами
    words = _words(prompt)
    assert not ({"cream", "matte", "flat", "outlines", "earthy"} & words)
