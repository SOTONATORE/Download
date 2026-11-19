"""Офлайн-проверки конфигурации ниши cartoon_photos после правок rules.yaml и style.md.

Что проверяется:
- profile.yaml читается загрузчиками конфигурации без ошибок (тип, режим seed, референсы, relaxed_rules);
- rules.yaml разбирается через load_rules: object_sets и speech разрешены, текст в кадре и лимит персонажей
  соответствуют ожиданию;
- style.md без блоков USER_ONLY собирается в системную инструкцию через build_system_instruction, не содержит
  формулировок «free of», а каждый пример из раздела Example Prompts проходит find_problems и _style_reasons
  с правилами rules.yaml и relaxed_rules для media=image.

Сети и установки пакетов не требуется: тест только читает файлы ниши.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Generate.core import config_loader as cl  # noqa: E402
from Generate.core import gemini_prompts as gp  # noqa: E402
from Generate.core import prompt_check as pc  # noqa: E402

NICHE = "cartoon_photos"
NICHE_DIR = _HERE.parents[1] / "Nishes" / NICHE
PROFILE = NICHE_DIR / "profile.yaml"
RULES = NICHE_DIR / "prompts" / "rules.yaml"
STYLE = NICHE_DIR / "prompts" / "style.md"
PRESET = NICHE_DIR / "prompts" / "preset.md"

# Правила, отключённые в профиле (art_medium, small_objects, emotion) и текстом кадра (frame_text)
RELAXED = frozenset({"art_medium", "small_objects", "emotion", "frame_text"})
MAX_CHARACTERS = 4


def _style_text() -> str:
    return gp.strip_user_only(STYLE.read_text(encoding="utf-8"))


def _examples() -> list:
    """Тексты примеров из раздела # Example Prompts: строка «Example N — ...:» и следующий абзац."""
    section = _style_text().split("# Example Prompts", 1)[1].split("# Final Check", 1)[0]
    return [m.strip() for m in re.findall(r"^Example \d+ [^\n]*:\n(.+?)(?:\n\s*\n|\Z)", section,
                                           re.M | re.S)]


# ------------------------------------------------------------------ profile.yaml

def test_profile_loads_through_config_loader():
    """profile.yaml читается загрузчиками: тип image, режим seed run, без референсов."""
    assert cl.load_profile_type(NICHE) == "image"
    assert cl.load_profile_seed_mode(NICHE) == "run"
    assert cl.profile_refs_enabled(NICHE) is False


def test_profile_relaxed_rules_parse_cleanly():
    """relaxed_rules и text_language разбираются из строк profile.yaml без ошибок."""
    lines = PROFILE.read_text(encoding="utf-8").splitlines()
    relaxed, text_language = gp.profile_relaxation(lines, NICHE)
    # allow_text: true добавляет frame_text (правило текста в кадре отключено профилем с текстом)
    assert relaxed == frozenset({"art_medium", "small_objects", "emotion", "frame_text"})
    assert text_language == "srt"


# ------------------------------------------------------------------ rules.yaml

def test_rules_yaml_parses_with_new_switches():
    """rules.yaml: object_sets и speech разрешены, текст в кадре проверяется, лимит персонажей 4."""
    rules = gp.load_rules(RULES)
    assert rules.object_sets_forbidden is False
    assert rules.speech_allowed is True
    assert rules.dangling_text_check is True
    assert rules.max_characters == MAX_CHARACTERS
    assert rules.close_faces_allowed is False
    assert rules.fire_hands_allowed is True


def test_rules_yaml_keeps_allowed_shot_items():
    """Разрешённые типы кадра из прежнего файла остаются, включая lettering."""
    rules = gp.load_rules(RULES)
    assert {"map", "diagram", "chart", "lettering", "paper"} <= set(rules.allowed_shot_items)


# ------------------------------------------------------------------ style.md

def test_style_strips_user_only_and_has_no_free_of_advice():
    """style.md без блоков USER_ONLY и без формулировки «free of» (отрицания вредят генерации)."""
    raw = STYLE.read_text(encoding="utf-8")
    assert "USER_ONLY" not in gp.strip_user_only(raw)
    assert "free of" not in raw.lower()


def test_system_instruction_builds_from_style():
    """build_system_instruction собирает системную инструкцию из style.md и правил ниши."""
    rules = gp.load_rules(RULES)
    cfg = gp.PromptConfig(style_file=str(STYLE), media="image", preset="none", rules=rules,
                          relaxed_rules=RELAXED, batch_passage=True)
    assert cfg.max_characters == MAX_CHARACTERS
    text = gp.build_system_instruction(cfg)
    assert "# Style guide" in text
    assert "# Subject and Context" in text
    assert "reading <TEXT> in bold black uppercase lettering" in text
    assert "free of" not in text.lower()


def test_examples_count_is_within_range():
    """В разделе Example Prompts 7–8 примеров."""
    assert 7 <= len(_examples()) <= 8


def test_examples_have_no_violations_or_closing_sentence():
    """Каждый пример проходит find_problems и _style_reasons без нарушений; без закрывающей фразы стиля и кавычек."""
    rules = gp.load_rules(RULES)
    for i, example in enumerate(_examples(), 1):
        found = pc.find_problems(example, "image", RELAXED, MAX_CHARACTERS, rules)
        style = gp._style_reasons(example, "image", RELAXED, MAX_CHARACTERS, rules)
        assert found == [], f"пример {i}: {found}"
        assert style == [], f"пример {i}: {style}"
        assert "closing style" not in example.lower(), f"пример {i}"
        assert "\"" not in example, f"пример {i}: кавычки"


def test_preset_passes_checks_with_profile_limits():
    """Текст preset.md, который дописывается к промптам, проходит проверки при лимите персонажей ниши."""
    rules = gp.load_rules(RULES)
    text = gp.read_niche_preset(str(PRESET)).strip()
    assert pc.find_problems(text, "image", RELAXED, MAX_CHARACTERS, rules) == []
    assert gp._style_reasons(text, "image", RELAXED, MAX_CHARACTERS, rules) == []
