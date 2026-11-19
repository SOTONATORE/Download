"""Проверка готовых промптов для 2D-комиксов (профили с max_characters, например cartoon_photos).

Покрывает: подсчёт людей по числу и групп предметов, крупные планы (только лица запрещены), машины по числу,
надписи в кадре (типографские слова и запятая перед «reading»), падежные русские надписи в реплике сегмента.
Контракт ltx25 и zimage_turbo (max_characters=None) проверяется отдельными тестами.
Сеть не используется; модуль prompt_check stdlib-only.
"""
import pytest

from Generate.core import prompt_check as pc
from Generate.core.prompt_check import RulesValues

# Профиль комикса: max_characters 4, предметы не запрещены, крупные планы лиц запрещены по умолчанию
COMIC = RulesValues(max_characters=4, animals_count=True, object_sets_forbidden=False, close_faces_allowed=False)


def _codes(prompt, rules=COMIC, media="image", max_characters=4):
    return [pc.problem_code(p) for p in pc.find_problems(prompt, media, max_characters=max_characters, rules=rules)]


# ---------------------------------------------------------------------------
# Люди по числу и группы предметов при max_characters=4
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", [
    "neat rows of potato plants",
    "a group of tubers on a wooden table",
    "two farmers",
    "three workers",
    "four shoppers",
    "two adult women",
])
def test_bounded_people_and_plant_rows_pass_with_max_characters(prompt):
    assert _codes(prompt) == [], prompt


@pytest.mark.parametrize("prompt", [
    "five workers",
    "a crowd",
    "a crowd of villagers in the square",
    "several people gather by the field",
    "dozens of farmers walk the road",
])
def test_too_many_people_and_crowds_still_fail(prompt):
    assert pc.PROBLEM_PEOPLE_COUNT in _codes(prompt), prompt


def test_numeral_counts_exactly():
    assert pc.count_people("two farmers", human_only=True) == 2
    assert pc.count_people("four shoppers", human_only=True) == 4
    assert pc.count_people("two adult women", human_only=True) == 2
    assert pc.count_people("three workers", human_only=True) == 3
    # число не относится к существительному, если между ними предлог или союз
    assert pc.count_people("two of the men", human_only=True) == 2


def test_group_of_objects_is_not_people_count_even_with_limit_one():
    assert pc.count_people("a group of tubers", human_only=True) == 0
    one = RulesValues(max_characters=1)
    assert pc.PROBLEM_PEOPLE_COUNT not in _codes("a group of tubers", one, max_characters=1)


def test_count_people_default_mode_is_unchanged_for_new_nouns():
    # shoppers, cooks и chefs входят только в human_only: подсчёт без max_characters не меняется
    assert pc.count_people("four shoppers") == 0
    assert pc.count_people("a cook and a chef") == 0


# ---------------------------------------------------------------------------
# Крупные планы: предметы и руки разрешены, лица — нет
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", [
    "close-up of two hands peeling a tuber",
    "close-up of a tuber on a wooden board",
    "macro close-up of soil and roots",
])
def test_object_and_hand_close_ups_pass(prompt):
    assert pc.PROBLEM_CLOSE_FACE not in _codes(prompt), prompt


@pytest.mark.parametrize("prompt", [
    "close-up of a woman's face",
    "extreme close-up of an old man's face",
    "a close-up portrait of a farmer",
    "the face fills the frame with sharp eyes",
    "close-up of the head of a boy",
])
def test_face_close_ups_still_fail(prompt):
    assert pc.PROBLEM_CLOSE_FACE in _codes(prompt), prompt


def test_face_close_up_only_for_images_as_before():
    assert pc.PROBLEM_CLOSE_FACE not in _codes("close-up of a woman's face", media="video")


def test_close_faces_allowed_skips_close_face_check():
    rules = RulesValues(max_characters=4, close_faces_allowed=True)
    assert pc.PROBLEM_CLOSE_FACE not in _codes("close-up of a woman's face", rules=rules)


# ---------------------------------------------------------------------------
# Машины и механизмы по числу (без object_sets: forbidden)
# ---------------------------------------------------------------------------

def test_small_bounded_machines_pass():
    assert pc.PROBLEM_MANY_OF_ONE not in _codes("two sorting machines")


def test_bounded_trucks_pass_up_to_limit_and_fail_above_it():
    rules3 = RulesValues(max_characters=4, horses_vehicles_animals_max=3)
    assert pc.PROBLEM_MANY_OF_ONE not in _codes("three delivery trucks", rules=rules3)
    assert pc.PROBLEM_MANY_OF_ONE in _codes("five delivery trucks", rules=rules3)


@pytest.mark.parametrize("prompt", ["a fleet of trucks on the road", "rows of parked cars along the street"])
def test_unbounded_fleets_and_rows_of_vehicles_fail(prompt):
    assert pc.PROBLEM_MANY_OF_ONE in _codes(prompt), prompt


def test_rows_of_plants_are_not_vehicles():
    assert pc.PROBLEM_MANY_OF_ONE not in _codes("neat rows of potato plants")


def test_machines_stay_flagged_when_object_sets_forbidden():
    rules = RulesValues(max_characters=4, object_sets_forbidden=True)
    assert pc.PROBLEM_MANY_OF_ONE in _codes("two sorting machines", rules=rules)


# ---------------------------------------------------------------------------
# max_characters=None: ltx25 и zimage_turbo остаются строгими
# ---------------------------------------------------------------------------

STRICT = RulesValues()


def test_none_keeps_rows_of_as_people_count():
    assert pc.PROBLEM_PEOPLE_COUNT in _codes("neat rows of potato plants", STRICT, max_characters=None)


def test_none_two_farmers_fail_image_limit():
    assert pc.PROBLEM_PEOPLE_COUNT in _codes("two farmers", STRICT, max_characters=None)


def test_none_close_up_of_hands_still_flagged():
    assert pc.PROBLEM_CLOSE_FACE in _codes("close-up of two hands peeling a tuber", STRICT, max_characters=None)


def test_none_two_machines_still_flagged():
    assert pc.PROBLEM_MANY_OF_ONE in _codes("two sorting machines", STRICT, max_characters=None)


def test_none_content_hits_match_strict_defaults():
    hits = pc.content_hits("close-up of two hands peeling a tuber", "image", frozenset(), None, STRICT)
    assert ("close_face", "close-up") in hits


# ---------------------------------------------------------------------------
# Надписи в кадре: регулярное выражение
# ---------------------------------------------------------------------------

def test_lettering_rx_captures_header_and_banner_phrasing():
    m = pc._LETTERING_RX.search("a small banner title reading HARVEST in centered sans-serif lettering")
    assert m is not None and m.group(2) == "HARVEST"


def test_lettering_rx_captures_header_lettering_with_comma_before_reading():
    m = pc._LETTERING_RX.search("white header lettering, reading SUNFLOWER")
    assert m is not None
    assert pc._lettering_phrase(m) == "SUNFLOWER"


def test_lettering_rx_comma_phrase_is_cut_at_comma():
    m = pc._LETTERING_RX.search("lettering, reading SUNFLOWER, a yellow flower")
    assert pc._lettering_phrase(m) == "SUNFLOWER"


def test_frame_has_lettering_with_new_font_words():
    assert pc.frame_has_lettering("off-white label lettering reading TOP HARVEST in bold black lettering")


# ---------------------------------------------------------------------------
# Надписи в кадре: русские падежные формы против реплики сегмента
# ---------------------------------------------------------------------------

def _mismatch(title, segment):
    prompt = f"Hand-lettered title reading {title} in bold black uppercase lettering."
    return pc.segment_lettering_mismatch(prompt, segment)


@pytest.mark.parametrize("title, segment", [
    ("ЗЕМЛЯНАЯ ГРУША", "земляной грушей"),
    ("ХРАНЕНИЕ", "хранения"),
    ("СЕВООБОРОТ", "севооборотом"),
    ("СУХИЕ ВЕЩЕСТВА", "сухого вещества"),
])
def test_inflected_russian_title_matches_segment(title, segment):
    assert _mismatch(title, segment) == []


def test_exact_phrase_still_matches():
    assert _mismatch("ЗЕМЛЯНАЯ ГРУША", "Это земляная груша.") == []


def test_unrelated_russian_word_is_mismatch():
    assert _mismatch("МОРКОВЬ", "земляной грушей") == ["МОРКОВЬ"]


def test_english_word_not_in_segment_is_mismatch():
    assert _mismatch("HELLO", "земляной грушей") == ["HELLO"]


def test_negated_russian_phrase_is_mismatch():
    assert _mismatch("АРТИШОК", "Это не является артишоком.") == ["АРТИШОК"]


def test_negated_inflected_phrase_is_mismatch():
    assert _mismatch("ЗЕМЛЯНАЯ ГРУША", "Это не является земляной грушей.") == ["ЗЕМЛЯНАЯ ГРУША"]


def test_english_negation_window_unchanged():
    # точная подстрока с отрицанием в трёх словах перед ней: надпись ошибочна (как раньше)
    prompt = "Hand-lettered sign reading FIRE in bold black uppercase lettering."
    assert pc.segment_lettering_mismatch(prompt, "can't start a fire") == ["FIRE"]
