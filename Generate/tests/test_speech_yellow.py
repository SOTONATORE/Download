"""Регрессия: цвет «yellow» не должен срабатывать как речь/крик (PROBLEM_SPEECH).

Раньше шаблон содержал «yell\\w*», и слово «yellow» попадало в нарушение речи (сегмент 557 cartoon_photos).
Теперь формы «yell», «yells», «yelling», «yelled» перечислены явно. Сеть не используется.
"""
import pytest

from Generate.core.prompt_check import PROBLEM_SPEECH, content_hits


def speech_codes(prompt, media="video"):
    return {code for code, _ in content_hits(prompt, media)}


@pytest.mark.parametrize("prompt", [
    "three round fruits in muted red and yellow",
    "a yellow taxi on a street",
])
def test_yellow_colour_is_not_speech(prompt):
    assert PROBLEM_SPEECH not in speech_codes(prompt)


@pytest.mark.parametrize("prompt", [
    "the man yells at the crowd",
    "she yelled for help",
    "he is yelling across the yard",
    "a guard yell out a warning",
])
def test_yell_forms_are_speech(prompt):
    assert PROBLEM_SPEECH in speech_codes(prompt)


@pytest.mark.parametrize("word", ["yell", "yells", "yelling", "yelled"])
def test_yell_word_forms_match(word):
    assert PROBLEM_SPEECH in speech_codes(f"a man will {word} here")


def test_yellow_word_does_not_match():
    assert PROBLEM_SPEECH not in speech_codes("a yellow flower")
