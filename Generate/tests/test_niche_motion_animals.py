"""Ниша cartoon: слова про движение в строке медиа, животные во множественном числе и числа перед животными,
устаревший ключ max_characters в profile.yaml.

Тесты написаны, не запускались (прогоняет пользователь на GitHub). Сеть и ключи не используются.
Промпты в тестах — английские, как в настоящих промптах; комментарии — по-русски.
"""
import hashlib
from pathlib import Path

import pytest

from Generate import cli
from Generate.core import gemini_prompts as gp
from Generate.core import prompt_check as pc
from Generate.core.gemini_prompts import PromptConfig, RulesValues

GEN = Path(__file__).resolve().parents[1]
NICHE_DIR = GEN / "Nishes" / "cartoon"
LTX_DIR = GEN / "model_profiles" / "models" / "ltx25"
ZIM_DIR = GEN / "model_profiles" / "models" / "zimage_turbo"

# sha256 системной инструкции до правки — те же значения, что в test_rules_values.py
LTX_SHA256 = "50391333ebd2262acbd02c9a50cbb8d6f8e078d8371dc24eb9559c364edeaeb1"
ZIM_SHA256 = "9a894f59b81b712a6f4aa87a8f4cf8f03ce95937525b4bd84527fc43ed2fe21e"


def _codes(text, max_characters=None, rules=None, media="image"):
    return [code for code, _ in pc.content_hits(text, media, frozenset(), max_characters, rules)]


# (a) cartoon: scene_motion_wording drop убирает слова «and its motion» из строки медиа для изображений
def test_cartoon_image_instruction_drops_and_its_motion():
    rules = gp.load_rules(NICHE_DIR / "prompts" / "rules.yaml")
    assert rules.scene_motion_wording == "drop"
    cfg = PromptConfig(style_file=str(NICHE_DIR / "prompts" / "style.md"), media="image", preset="none", rules=rules)
    text = gp.build_system_instruction(cfg)
    assert "and its motion" not in text
    # остальная часть строки медиа на месте
    assert "- Describe only the scene. Never name a medium, rendering method, film look" in text


# (a) zimage_turbo и ltx25: слова «and its motion» есть, байты инструкции как до правки.
# ltx25 — профиль video, поэтому его строка медиа берётся для видео (в image-ветке он не используется).
@pytest.mark.parametrize("profile, media, sha", [(ZIM_DIR, "image", ZIM_SHA256), (LTX_DIR, "video", LTX_SHA256)])
def test_zimage_and_ltx25_keep_and_its_motion_byte_identical(profile, media, sha):
    rules = gp.load_rules(profile / "prompts" / "rules.yaml")
    assert rules.scene_motion_wording == "keep"
    cfg = PromptConfig(style_file=str(profile / "prompts" / "style.md"), media=media, preset="none", rules=rules)
    text = gp.build_system_instruction(cfg)
    assert "and its motion" in text
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == sha


# (b) «three dogs» при лимите 4 и животных в подсчёте проходит по правилу «много одинаковых» и по лимиту
def test_three_dogs_pass_with_limit_four_and_animals_counted():
    rules = RulesValues(max_characters=4, animals_count=True)
    assert pc.count_characters("three dogs") == 3
    assert pc.PROBLEM_MANY_OF_ONE not in _codes("three dogs", 4, rules)
    assert pc.PROBLEM_PEOPLE_COUNT not in _codes("three dogs", 4, rules)


# (b) «three dogs and a goat and a cat»: три собаки, коза и кошка — 5 персонажей при лимите 4, нарушение лимита
def test_three_dogs_goat_and_cat_fail_character_count_with_limit_four():
    text = "three dogs and a goat and a cat"
    rules = RulesValues(max_characters=4, animals_count=True)
    assert pc.count_characters(text) == 5
    assert pc.PROBLEM_PEOPLE_COUNT in _codes(text, 4, rules)


# (c) «two horses» без max_characters по-прежнему нарушает правило «много одинаковых» (другие профили)
def test_two_horses_without_max_characters_still_fail_many_of_one():
    assert pc.PROBLEM_MANY_OF_ONE in _codes("two horses", max_characters=None, rules=RulesValues())
    # машины проверяются и при гейте животных; при object_sets: forbidden «two cars» нарушает запрет наборов
    assert pc.PROBLEM_MANY_OF_ONE in _codes(
        "two cars", 4, RulesValues(max_characters=4, animals_count=True, object_sets_forbidden=True))


def test_bounded_cars_pass_at_two_and_fail_above_horses_vehicles_animals_max():
    # object_sets: allowed (object_sets_forbidden False): машины ограничены числом horses_vehicles_animals_max
    rules = RulesValues(max_characters=4, animals_count=True, horses_vehicles_animals_max=3)
    assert pc.PROBLEM_MANY_OF_ONE not in _codes("two cars", 4, rules)
    assert pc.PROBLEM_MANY_OF_ONE in _codes("four cars", 4, rules)


class _Log:
    """Запись предупреждений cli.log: текст после подстановки аргументов."""

    def __init__(self):
        self.warnings = []

    def warning(self, message, *args):
        self.warnings.append(message % args)


def _legacy_profile(tmp_path, monkeypatch, rules_text, profile_max_line):
    """Временный профиль image: profile.yaml с полем max_characters (или без него), rules.yaml по тексту."""
    profile = "legacy_niche"
    base = tmp_path / profile
    (base / "prompts").mkdir(parents=True)
    (base / "prompts" / "style.md").write_text("# Style\nPlain scene.\n", encoding="utf-8")
    if rules_text is not None:
        (base / "prompts" / "rules.yaml").write_text(rules_text, encoding="utf-8")
    (base / "profile.yaml").write_text("type: image\n" + profile_max_line, encoding="utf-8")
    monkeypatch.setattr(cli, "profile_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(cli, "_style_dir", lambda name: str(tmp_path))
    monkeypatch.setattr(cli, "load_profile_type", lambda name: "image")
    log = _Log()
    monkeypatch.setattr(cli, "log", log)
    return profile, log


# (d) устаревший max_characters в profile.yaml без поля в rules.yaml: предупреждение один раз, значение берётся
def test_legacy_max_characters_warns_once_and_value_is_used(tmp_path, monkeypatch):
    profile, log = _legacy_profile(tmp_path, monkeypatch, "animals_count: true\n", "max_characters: 4\n")
    cfg = cli._build_prompt_config(profile, 20, "none")
    assert len(log.warnings) == 1
    assert "устарел" in log.warnings[0] and "prompts/rules.yaml" in log.warnings[0]
    assert cfg.max_characters == 4
    assert cfg.rules.max_characters == 4


# (d) max_characters задан и в profile.yaml, и в rules.yaml с разными значениями: ошибка по-русски
def test_legacy_and_rules_max_characters_different_raises_russian_error(tmp_path, monkeypatch):
    profile, _ = _legacy_profile(tmp_path, monkeypatch, "max_characters: 3\n", "max_characters: 4\n")
    with pytest.raises(cli.GeminiInputError) as exc:
        cli._build_prompt_config(profile, 20, "none")
    message = str(exc.value)
    assert "не совпадает" in message
    assert "max_characters" in message
