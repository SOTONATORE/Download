"""Запрещённые типы кадров для пайплайна Generate (шаг 4 плана PLAN_generate_photo_video.md).

Список запрещённых типов английский и общий с проверкой готового промпта (prompt_check.py):
он импортируется оттуда, а не копируется. Модуль использует только стандартную библиотеку (re)
и prompt_check.py, который тоже stdlib-only и не импортирует этот модуль, поэтому цикла нет.
"""

import re

try:  # пакетный и «плоский» импорт (как в gemini_prompts.py)
    from .prompt_check import FORBIDDEN_IMAGE_PHRASES_EN, FORBIDDEN_IMAGE_WORDS_EN, _STEM_EXCLUDE, _stem_alternative
except ImportError:  # pragma: no cover
    from prompt_check import FORBIDDEN_IMAGE_PHRASES_EN, FORBIDDEN_IMAGE_WORDS_EN, _STEM_EXCLUDE, _stem_alternative  # type: ignore

# Совпадение только с начала слова (\b перед основой), без учёта регистра, с любым окончанием.
# prompt_check.py не содержит готового построителя регулярного выражения для всего списка,
# поэтому выражение собрано здесь из импортированного списка и _stem_alternative (те же правила).
FORBIDDEN_FRAME_RE = re.compile(
    r"\b(?:" + "|".join(_stem_alternative(s) for s in FORBIDDEN_IMAGE_WORDS_EN) + r")",
    re.IGNORECASE,
)

# Правило, которое добавляется в запрос модели для сегмента с запрещённым кадром.
# Само правило не называет запрещённые предметы как задание нарисовать их в кадре.
REPHRASE_RULE_EN = (
    "The text of this segment names an item that must not appear in the image. "
    "Do not show that item, its labels or its writing. "
    "Describe an allowed image of the same subject instead: for a map, a landscape or a city view "
    "without labels; for documents, letters, papers, passports, forms, tickets or archive material, "
    "a hand, a table or a storage room without readable text; for a flag or an emblem, the building "
    "or square where it hangs, without a close-up of the symbol. "
    "Never write the name of the forbidden item in the prompt. "
    "Base the image only on what the segment text directly supports; add no details of your own."
)


# Фразы бумажных листов (blank sheet, sheet of paper, sheet of parchment): только точные сочетания, без учёта регистра.
_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in FORBIDDEN_IMAGE_PHRASES_EN) + r")\b",
    re.IGNORECASE,
)


def needs_rephrase(segment_text: str) -> bool:
    """True, если в тексте сегмента есть слово или фраза запрещённого типа кадра."""
    text = segment_text or ""
    return bool(FORBIDDEN_FRAME_RE.search(text)) or bool(_PHRASE_RE.search(text))


def rephrase_note(segment_text: str) -> str:
    """Правило перефразирования для сегмента, либо пустая строка, если оно не нужно."""
    return REPHRASE_RULE_EN if needs_rephrase(segment_text) else ""
