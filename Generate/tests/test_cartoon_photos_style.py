"""Офлайн-проверки style.md ниши cartoon_photos.

Что проверяется:
- инструкция модели не просит «закрывающее предложение стиля» и не содержит слов without и rendered
  (строка с перечнем запрещённых слов исключается: в ней эти слова перечислены намеренно);
- список запрещённых слов содержит все слова, которые ловят _STYLE_RULES и prompt_check.py;
- каждый пример из style.md и текст preset.md проходят prompt_check.find_problems и
  gemini_prompts._style_reasons с правилами rules.yaml и relaxed_rules из profile.yaml.

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

from Generate.core import gemini_prompts as gp  # noqa: E402
from Generate.core import prompt_check as pc  # noqa: E402

NICHE = _HERE.parents[1] / "Nishes" / "cartoon_photos"
STYLE = NICHE / "prompts" / "style.md"
PRESET = NICHE / "prompts" / "preset.md"
RULES = NICHE / "prompts" / "rules.yaml"
PROFILE = NICHE / "profile.yaml"

# Слова, которые обязаны быть в перечне запрещённых (синхронно с _STYLE_RULES и prompt_check.py).
REQUIRED_FORBIDDEN = [
    "render", "rendered", "rendering", "4k", "8k", "uhd", "photorealistic", "hyperrealistic",
    "masterpiece", "trending", "octane", "unreal engine",
    "without", "nothing", "nobody", "cannot",
    "group", "groups", "crowd", "crowds", "row of", "rows of",
    "lens", "35mm", "anamorphic", "aperture", "focal length", "film grain", "film still",
]


def _style_text() -> str:
    return STYLE.read_text(encoding="utf-8")


def _style_without_forbidden_line() -> str:
    """Текст style.md без строки «Forbidden prompt words»: только там слова перечислены намеренно."""
    return "\n".join(ln for ln in _style_text().splitlines()
                     if not ln.startswith("Forbidden prompt words"))


def _forbidden_line() -> str:
    lines = [ln for ln in _style_text().splitlines() if ln.startswith("Forbidden prompt words")]
    assert len(lines) == 1, "строка с перечнем запрещённых слов должна быть одна"
    return lines[0]


def _examples() -> list:
    """Тексты примеров из раздела # Example Prompts: блок «Example N — ...:» и следующий абзац."""
    section = _style_text().split("# Example Prompts", 1)[1].split("# Final Check", 1)[0]
    return [m.strip() for m in re.findall(r"^Example \d+ [^\n]*:\n(.+?)(?:\n\s*\n|\Z)", section,
                                           re.M | re.S)]


def _rules_and_relaxed():
    """Правила из rules.yaml и relaxed_rules из profile.yaml, как их читает конвейер."""
    lines = PROFILE.read_text(encoding="utf-8").splitlines()
    assert gp._profile_value(lines, "type") == "image", "профиль ниши должен быть типа image"
    rules = gp.load_rules(RULES)
    names = gp.parse_flow_list(gp._profile_value(lines, "relaxed_rules"), "relaxed_rules", "cartoon_photos")
    return rules, gp.parse_relaxed_rules(names, "cartoon_photos")


def _problems(text: str) -> tuple:
    """Нарушения prompt_check и стилевые причины gemini_prompts для одного текста (media=image)."""
    rules, relaxed = _rules_and_relaxed()
    found = pc.find_problems(text, "image", relaxed, None, rules)
    style = gp._style_reasons(text, "image", relaxed, None, rules)
    return found, style


# ------------------------------------------------------------------ инструкция модели

def test_style_does_not_ask_for_closing_sentence():
    """Модели не должна требоваться фраза «closing style sentence is present»."""
    assert "closing style sentence is present" not in _style_text()


def test_style_has_explicit_no_closing_sentence_rule():
    """Гайд явно запрещает писать закрывающее предложение стиля в разделах Global Style, Examples и Final Check."""
    text = _style_text()
    for section in ("# Global Style", "# Example Prompts", "# Final Check"):
        body = text.split(section, 1)[1].split("\n# ", 1)[0]
        assert "closing style summary sentence" in body, section


def test_style_has_no_without_outside_forbidden_list():
    """Слово without не употребляется в тексте гайда (кроме перечня запрещённых слов)."""
    assert re.search(r"\bwithout\b", _style_without_forbidden_line(), re.I) is None


def test_style_has_no_rendered_outside_forbidden_list():
    """Слово rendered не употребляется в тексте гайда (кроме перечня запрещённых слов)."""
    assert re.search(r"\brendered\b", _style_without_forbidden_line(), re.I) is None


def test_forbidden_list_covers_validator_words():
    """Перечень запрещённых слов в style.md содержит все слова, которые ловят валидаторы."""
    forbidden = _forbidden_line().lower()
    missing = [w for w in REQUIRED_FORBIDDEN if not re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])",
                                                              forbidden)]
    assert missing == []


def test_plaque_lettering_phrase_is_specified():
    """Инструкция требует точную формулировку «reading <TEXT> in bold black uppercase lettering»."""
    assert "reading <TEXT> in bold black uppercase lettering" in _style_text()


# ------------------------------------------------------------------ примеры и preset

def test_examples_are_present():
    """В разделе примеров найдены все восемь примеров (style.md: Example 1–8)."""
    assert len(_examples()) == 8


def test_examples_pass_prompt_check_and_style_reasons():
    """Каждый пример проходит find_problems и _style_reasons без нарушений."""
    for i, example in enumerate(_examples(), 1):
        found, style = _problems(example)
        assert found == [], f"пример {i}: {found}"
        assert style == [], f"пример {i}: {style}"


def test_preset_passes_prompt_check_and_style_reasons():
    """Текст preset.md, который дописывается к промптам, проходит те же проверки без нарушений."""
    found, style = _problems(PRESET.read_text(encoding="utf-8").strip())
    assert found == []
    assert style == []
