"""Проверка готовых промптов для 2D-комиксов (профили с max_characters, например cartoon).

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


# ---------------------------------------------------------------------------
# Оси диаграммы не оружие: «axes» проверяется как оружие только без слов диаграммы рядом
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", [
    "A warm cream chart card with X and Y axes and short labels.",
    "Graph paper with labeled axes on a warm cream backdrop.",
    "Plot axes on a cream card marked with dates.",
    "Diagram of two coordinate axes in clean black ink.",
])
def test_chart_axes_are_not_weapons(prompt):
    assert pc.PROBLEM_WEAPON not in _codes(prompt), prompt


@pytest.mark.parametrize("prompt", [
    "A man holding axes near the fence.",
    "Two battle axes leaning against a wall.",
    "A pile of axes on the ground beside a wooden block.",
    "A warrior with a pair of axes.",
])
def test_standalone_axes_still_flagged_as_weapon(prompt):
    assert pc.PROBLEM_WEAPON in _codes(prompt), prompt


def test_weapons_allowed_skips_axes_too():
    # weapons: allowed снимает проверку оружия целиком, и «axes» тоже
    rules = RulesValues(weapons_allowed=True)
    assert pc.PROBLEM_WEAPON not in _codes("A man holding axes near the fence.", rules=rules)


# ---------------------------------------------------------------------------
# «cooks» как глагол после единственного подлежащего не считается множественным числом людей
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "the woman cooks soup",
    "he cooks at home",
    "she cooks eggs",
    "the cook cooks rice",
    "a farmer cooks beans",
])
def test_singular_subject_with_cooks_verb_is_not_plural_people(text):
    # одна женщина или один человек, глагол «cooks»: не больше одного персонажа с учётом героя
    assert pc.count_people(text, human_only=True) <= 1, text


def test_plural_cooks_still_counted_as_people():
    assert pc.count_people("two cooks at the stove", human_only=True) == 2
    assert pc.count_people("the cooks prepare soup", human_only=True) == 2


def test_woman_cooks_verb_does_not_fail_person_limit_with_limit_one():
    one = RulesValues(max_characters=1, animals_count=True)
    assert pc.PROBLEM_PEOPLE_COUNT not in _codes("the woman cooks soup", one, max_characters=1)


# ---------------------------------------------------------------------------
# Надпись: английское описание сцены после заглавной или не латинской надписи отрезается
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("КАРТОФЕЛЬ above the crate", "КАРТОФЕЛЬ"),
    ("МАГАЗИНЫ beside the arrow", "МАГАЗИНЫ"),
    ("ТОПИНАМБУР and points to the field", "ТОПИНАМБУР"),
    ("ОГРОМНОЕ КОЛИЧЕСТВО points from the sky", "ОГРОМНОЕ КОЛИЧЕСТВО"),
    ("ТОПИНАМБУР on a warm cream backdrop", "ТОПИНАМБУР"),
    ("POTATO STARCH on a warm cream backdrop", "POTATO STARCH"),
    ("YEAR 2020 with a dashed border", "YEAR 2020"),
    ("земляная груша on a hill", "земляная груша"),
    ("A | B", "A | B"),
])
def test_lettering_phrase_trims_english_scene_clause(raw, expected):
    m = pc._LETTERING_RX.search(f"white label lettering, reading {raw}.")
    assert pc._lettering_phrase(m) == expected


@pytest.mark.parametrize("raw", ["can't start a fire", "on a hill", "many farmers above the field"])
def test_lowercase_latin_phrase_is_not_trimmed(raw):
    m = pc._LETTERING_RX.search(f"white label lettering, reading {raw}.")
    assert pc._lettering_phrase(m) == raw


# ---------------------------------------------------------------------------
# «shoots at»: ростки растения не стрельба; стрельба по цели остаётся нарушением
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "new shoots at the base",
    "young shoots at the soil of the bed",
    "green shoots at the top of the row",
])
def test_botanical_shoots_are_not_battle(text):
    assert pc.PROBLEM_CHAOS not in [c for c, _ in pc.content_hits(text, "image")], text


@pytest.mark.parametrize("text", [
    "a rider shoots at the enemy",
    "soldiers shooting at the rider",
    "the herder shoot back at the raiders",
    "the rider is shot at",
])
def test_shooting_at_target_is_still_battle(text):
    assert pc.PROBLEM_CHAOS in [c for c, _ in pc.content_hits(text, "image")], text


# ---------------------------------------------------------------------------
# «forming» / «formed» — не бланк; «lettering» — не запрещённое слово «letter»
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["a red X forming on the wall", "a circle formed by pebbles on the ground"])
def test_forming_and_formed_are_not_forbidden_shot_type(text):
    assert pc.PROBLEM_FORBIDDEN_SHOT_TYPE not in _codes(text), text


def test_plain_form_is_still_forbidden():
    assert pc.PROBLEM_FORBIDDEN_SHOT_TYPE in _codes("a form lies on the table")


# Глагол «forms» (форма, а не бланк) не запрещён; существительное во множественном числе и «fill out a form» — запрещены.
@pytest.mark.parametrize("text", [
    "a chain forms a curved bridge over the ground",
    "the stem forms into a ring above the soil",
])
def test_forms_as_verb_is_not_forbidden_shot_type(text):
    assert pc.PROBLEM_FORBIDDEN_SHOT_TYPE not in _codes(text), text


@pytest.mark.parametrize("text", [
    "a stack of forms on the desk",
    "fill out a form at the counter",
])
def test_forms_as_paper_is_still_forbidden(text):
    assert pc.PROBLEM_FORBIDDEN_SHOT_TYPE in _codes(text), text


def test_uniform_is_not_forbidden_shot_type():
    # «uniform» не входит в список запрещённых типов кадра prompt_check (это слово сохраняется как есть)
    assert pc.PROBLEM_FORBIDDEN_SHOT_TYPE not in _codes("a farmer in a clean uniform stands by the field")


def test_lettering_word_is_not_forbidden_letter():
    assert pc.PROBLEM_FORBIDDEN_SHOT_TYPE not in _codes("a hand lettering on a wall")


def test_letter_is_still_forbidden_without_allowance():
    assert pc.PROBLEM_FORBIDDEN_SHOT_TYPE in _codes("A letter lies open on the table.")


def test_lettering_hints_tell_model_to_remove_unmatched_label():
    for hint in (pc.LETTERING_SEGMENT_HINT_EN, pc.DANGLING_TEXT_HINT_EN):
        assert "remove the label card, plaque or lettering completely" in hint
        assert "text-free" in hint


# ---------------------------------------------------------------------------
# Токенизация: комбинирующие знаки и апострофы
# ---------------------------------------------------------------------------

def test_word_tokens_keep_combining_marks():
    assert pc._word_tokens("मैं आलू खाता") == ["मैं", "आलू", "खाता"]


def test_word_tokens_keep_inner_apostrophe_and_strip_quotes():
    assert pc._word_tokens("can't 'fire'") == ["can't", "fire"]


# ---------------------------------------------------------------------------
# Надписи в реплике сегмента: все языки SRT (кириллица, украинский, латиница, немецкий, испанский,
# польский, турецкий, деванагари, CJK)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title, segment", [
    # русская кириллица и украинская (і, ї, є, ґ)
    ("КЛУБНИ", "клубни"),
    ("ТОПИНАМБУР", "топинамбура"),
    ("ЧОЛОВІК", "чоловіка"),
    ("КАРТОПЛЯ", "картоплі"),
    ("ЇЖАК", "їжака"),
    ("ЄВРОПА", "Європа."),
    # английский: множественное число и флексии
    ("POTATO", "potatoes"),
    ("LEAF", "leaves"),
    ("FARMERS", "farmer"),
    # немецкий и испанский
    ("KARTOFFEL", "Kartoffeln"),
    ("STRASSE", "Straße"),
    ("PATATA", "patatas"),
    # польский: ó без диакритики
    ("ZIEMNIAK", "ziemniaków"),
    # турецкий: İ и ı, падежный суффикс после апострофа
    ("İSTANBUL", "istanbul'da yaşıyorum"),
    ("ILIK", "ılık"),
    # деванагари: комбинирующие знаки входят в слово
    ("आलू", "मैं आलू खाता हूँ।"),
    # CJK: слова без пробелов
    ("马铃薯 | 菊芋", "这种马铃薯和菊芋很有营养。"),
    ("ジャガイモ", "ジャガイモを育てる"),
])
def test_multilang_label_matches_segment(title, segment):
    assert _mismatch(title, segment) == [], (title, segment)


@pytest.mark.parametrize("title, segment", [
    ("РАСТУТ", "растение"),                       # разные основы: общее начало «раст» короче порога
    ("POTATO", "potassium"),
    ("FORM", "format"),                           # «format» — не флексия «form»
    ("LAND", "landscape"),
    ("आलू", "आल"),                                # комбинирующий знак входит в слово: «आल» — другое слово
    ("马铃薯 | 菊芋", "这种马铃薯很有营养。"),          # второго слова в реплике нет
])
def test_multilang_label_not_in_segment_is_mismatch(title, segment):
    assert _mismatch(title, segment) == [title], (title, segment)


# ---------------------------------------------------------------------------
# ALL-CAPS надпись в кадре при frame_text: текст этикетки не проверяется как описание сцены
# ---------------------------------------------------------------------------

FRAME_TEXT = frozenset({"frame_text"})


def _plain_codes(prompt, relaxed=frozenset()):
    return [pc.problem_code(p) for p in pc.find_problems(prompt, "image", relaxed=relaxed)]


@pytest.mark.parametrize("label", ["FARMERS", "PEOPLE", "MANY", "FORM"])
def test_caps_label_passes_when_frame_text_relaxed(label):
    prompt = f"A small banner title reading {label} in bold black uppercase lettering on a cream card."
    assert _plain_codes(prompt, FRAME_TEXT) == [], label


def test_caps_label_is_checked_without_frame_text():
    prompt = "A small banner title reading FARMERS in bold black uppercase lettering on a cream card."
    assert pc.PROBLEM_PEOPLE_COUNT in _plain_codes(prompt)


def test_lowercase_many_farmers_still_fail_when_frame_text_relaxed():
    assert pc.PROBLEM_PEOPLE_COUNT in _plain_codes("A quiet field with many farmers walking along the road.",
                                                   FRAME_TEXT)


@pytest.mark.parametrize("label", ["ID CARD", "FLAG"])
def test_strict_symbols_stay_forbidden_inside_caps_label(label):
    prompt = f"A small card reading {label} in bold black uppercase lettering."
    assert pc.PROBLEM_FORBIDDEN_SHOT_TYPE in _plain_codes(prompt, FRAME_TEXT), label


# ---------------------------------------------------------------------------
# Этнический признак: нормализация диакритики и европейские написания корней
# ---------------------------------------------------------------------------

def test_ethnicity_root_accent_and_german_spelling_match_segment():
    prompt = "An African man walks by the river."
    assert pc.ethnicity_reasons(prompt, "Ein Mann geht in Áfrika am Fluss.") == []
    assert pc.ethnicity_reasons(prompt, "Der Mann geht in Afrika am Fluss.") == []
    assert pc.ethnicity_reasons(prompt, "Der Mann geht am Fluss.") == [pc.CONTENT_REASON_RU[pc.PROBLEM_ETHNICITY]]
