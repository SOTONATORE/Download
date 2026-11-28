"""Правила содержания из prompts/rules.yaml: загрузчик, системная инструкция, проверка и байтовая идентичность.

Тесты написаны, не запускались (прогоняет пользователь на GitHub).
"""
import hashlib
from pathlib import Path

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core import prompt_check as pc
from Generate.core.gemini_prompts import GeminiInputError, PromptConfig, RulesValues

GEN = Path(__file__).resolve().parents[1]
NICHE_DIR = GEN / "Nishes" / "cartoon"
LTX_DIR = GEN / "model_profiles" / "models" / "ltx25"
ZIM_DIR = GEN / "model_profiles" / "models" / "zimage_turbo"

# sha256 и длина системной инструкции до правки (старый код, конфиг по умолчанию, preset none)
LTX_SHA256 = "c2c22f67e35f4ebba736cfe3b93555d33b65d93d1edd3fc9f50316c434439dbb"
LTX_BYTES = 21778
ZIM_SHA256 = "8d73370aa6d2a4cfc3f2a3a75775dd795f8b9942bc779ddab150eec649f687e0"
ZIM_BYTES = 7249


# (a) нет файла rules.yaml — значения по умолчанию
def test_missing_rules_file_gives_defaults(tmp_path):
    assert gp.load_rules(tmp_path / "rules.yaml") == RulesValues()


# (b) неизвестный ключ — ошибка по-русски с именем ключа
def test_unknown_key_fails_with_russian_message_naming_it(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("max_characters: 3\nhorse_count: 2\n", encoding="utf-8")
    with pytest.raises(GeminiInputError) as exc:
        gp.load_rules(rules_file)
    message = str(exc.value)
    assert "horse_count" in message
    assert "Ошибка входных данных" in message


# (c) значения cartoon: нет удалённых фраз, лимит животных 3
def test_cartoon_system_instruction_drops_removed_sentences_and_uses_animal_limit_three():
    rules = gp.load_rules(NICHE_DIR / "prompts" / "rules.yaml")
    cfg = PromptConfig(style_file=str(NICHE_DIR / "prompts" / "style.md"), media="image", preset="none",
                       relaxed_rules=frozenset({"art_medium", "small_objects"}), rules=rules)
    text = gp.build_system_instruction(cfg)
    assert "Show at most three horses, vehicles or animals" in text
    assert "Show at most 4 characters in total, people and animals together" in text
    # small_objects allowed: строки о мелких предметах нет
    assert "Show only large, clearly visible objects" not in text
    # weapons, close_faces forbidden: строки запрета остаются
    # speech: allowed (намеренно): строка запрета речи не выводится, немые выражения лиц остаются
    assert "No speech of any kind" not in text
    assert "No weapons in the frame" in text
    assert "Never show close-ups" in text
    assert "Show emotion only through posture and visible actions." in text
    # static_frame true: запрет движения камеры остаётся
    assert "Never describe camera movement" in text


# (d) animals_count true: животные входят в лимит персонажей; false — не входят
def test_animals_count_true_counts_animals_in_limit():
    text = "A man, a dog, a cat and a goat stand by the gate."
    on = pc.content_hits(text, "image", frozenset(), 3, RulesValues(animals_count=True))
    off = pc.content_hits(text, "image", frozenset(), 3, RulesValues(animals_count=False))
    assert pc.PROBLEM_PEOPLE_COUNT in [code for code, _ in on]
    assert pc.PROBLEM_PEOPLE_COUNT not in [code for code, _ in off]


# (e) subject_source: text_only добавляет строку, text_and_context — нет
def test_subject_source_text_only_adds_line_and_text_and_context_does_not(tmp_path):
    style = tmp_path / "style.md"
    style.write_text("# Style\nPlain scene.\n", encoding="utf-8")
    line = "Objects and on-frame text come from the Text line only"
    only = gp.build_system_instruction(
        PromptConfig(style_file=str(style), media="image", rules=RulesValues(subject_source="text_only")))
    both = gp.build_system_instruction(
        PromptConfig(style_file=str(style), media="image", rules=RulesValues(subject_source="text_and_context")))
    assert line in only
    assert line not in both


# (f) ltx25 и zimage_turbo: системная инструкция байт в байт как до правки (с rules.yaml и без него)
@pytest.mark.parametrize("profile, media, sha, size", [
    (LTX_DIR, "video", LTX_SHA256, LTX_BYTES),
    (ZIM_DIR, "image", ZIM_SHA256, ZIM_BYTES),
])
def test_model_profile_system_instruction_is_byte_identical_to_before(profile, media, sha, size):
    style = profile / "prompts" / "style.md"
    rules = gp.load_rules(profile / "prompts" / "rules.yaml")
    for cfg in (PromptConfig(style_file=str(style), media=media, preset="none"),
                PromptConfig(style_file=str(style), media=media, preset="none", rules=rules)):
        data = gp.build_system_instruction(cfg).encode("utf-8")
        assert len(data) == size
        assert hashlib.sha256(data).hexdigest() == sha
