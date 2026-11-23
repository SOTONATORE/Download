"""Ниша cartoon: лимит персонажей (герой и до трёх других), max_prompt_chars из профиля, подпись fail_fast.

Тесты написаны, не запускались (прогоняет пользователь на GitHub). Сеть и ключи не используются.
Промпты в тестах — английские, как в настоящих промптах; комментарии — по-русски.
"""
import pytest

from Generate import cli
from Generate.core import gemini_prompts as gp
from Generate.core import prompt_check as pc
from Generate.core.gemini_prompts import GeminiError, PromptConfig

# Главный герой (main character as in ref_6) плюс три других персонажа: фермер, женщина, собака
FOUR = ("The main character as in ref_6 stands in a green field. "
        "A farmer in a brown coat stands beside a dog. "
        "A woman in a red shawl holds a basket.")
# Пятый персонаж (коза) сверх лимита четырёх
FIVE = FOUR + " A goat grazes near a grey rock."


def _codes(text, media="image", max_characters=None):
    return [code for code, _ in pc.content_hits(text, media, frozenset(), max_characters)]


# (a) четыре персонажа при max_characters 4 проходят, пять — нет
def test_four_characters_pass_and_five_fail_with_max_characters():
    assert pc.count_characters(FOUR) == 4
    assert pc.count_characters(FIVE) == 5
    assert pc.PROBLEM_PEOPLE_COUNT not in _codes(FOUR, max_characters=4)
    assert pc.PROBLEM_PEOPLE_COUNT in _codes(FIVE, max_characters=4)


# (b) собака и человек считаются оба
def test_dog_and_person_are_both_counted():
    text = "A man walks a dog past the gate."
    assert pc.count_characters(text) == 2
    assert pc.PROBLEM_PEOPLE_COUNT in _codes(text, max_characters=1)
    assert pc.PROBLEM_PEOPLE_COUNT not in _codes(text, max_characters=2)


# (c) без max_characters действует прежний PEOPLE_LIMIT (для фото — один человек)
def test_profile_without_max_characters_keeps_people_limit():
    assert pc.PEOPLE_LIMIT["image"] == 1
    assert pc.PROBLEM_PEOPLE_COUNT in _codes(FOUR, max_characters=None)
    assert pc.PROBLEM_PEOPLE_COUNT not in _codes("A man walks a dog past the gate.", max_characters=None)
    assert PromptConfig(style_file="x", media="image").max_characters is None
    # поля нет в profile.yaml — значение None, как раньше
    assert cli._profile_int(["type: video"], "max_characters", "x") is None


# (d) подпись fail_fast называет правило и не содержит текста модели
def test_fail_fast_label_names_people_rule_without_model_text():
    cfg = PromptConfig(style_file="x", media="image", fail_fast=True)   # raw_dir не задан: файл не пишется
    label = pc.problem_label(pc.PROBLEM_PEOPLE_COUNT)
    assert label == pc.CONTENT_REASON_RU[pc.PROBLEM_PEOPLE_COUNT]
    # путь «content»: код берётся из записи нарушения
    assert gp._content_rule_label([pc.make_problem(pc.PROBLEM_PEOPLE_COUNT, "a man")], 4) == label
    # без max_characters подписи нет: сообщение других ниш прежнее
    assert gp._content_rule_label([pc.make_problem(pc.PROBLEM_PEOPLE_COUNT, "a man")]) is None
    raw = "A man, a woman and a dog"
    with pytest.raises(GeminiError) as exc:
        gp._fail_fast_stop(cfg, [1, 2], [2], "content", raw, label)
    message = str(exc.value)
    assert label in message
    assert "нарушение в тексте промпта" not in message
    assert "man" not in message and "dog" not in message and raw not in message


# (e) max_prompt_chars 6000 и max_characters 4 из профиля попадают в PromptConfig
def test_cartoon_profile_keys_reach_prompt_config():
    cfg = cli._build_prompt_config("cartoon", 20, "none")
    assert cfg.max_prompt_chars == 6000
    assert cfg.max_characters == 4
