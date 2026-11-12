"""Запрещённые типы кадров для пайплайна Generate (шаг 4 плана PLAN_generate_photo_video.md).

Список запрещённых типов английский и общий с проверкой готового промпта (prompt_check.py):
он импортируется оттуда, а не копируется. Модуль использует только стандартную библиотеку (re)
и prompt_check.py, который тоже stdlib-only и не импортирует этот модуль, поэтому цикла нет.

Текст сегментов SRT русский, поэтому здесь же есть русские основы тех же типов кадров
(карта, документ, письмо, паспорт и т. д.). Без них правило не срабатывало на русском тексте.
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

# Русские основы тех же типов (текст сегмента): карта, календарь, документ, график, диаграмма,
# инфографика, флаг, герб, эмблема, газета, удостоверение, паспорт, бумага, письмо, пергамент,
# бланк, анкета, формуляр, таблица. Окончания перечислены явно: «документальный», «картина»,
# «письменный», «форма» (мундир) сюда не попадают.
FORBIDDEN_FRAME_RU_RE = re.compile(
    r"\b(?:карт(?:а|ы|у|е|ой|ою|ам|ами|ах)?"
    r"|календар\w*"
    r"|документ(?:ы|ов|ам|ами|ах|а|у|е|ом)?"
    r"|график(?:и|ов|ам|ами|ах|а|у|е|ом)?"
    r"|диаграмм\w*"
    r"|инфографик\w*"
    r"|флаг(?:и|ов|ам|ами|ах|а|у|е|ом)?"
    r"|герб(?:ы|а|ов|у|е|ом)?"
    r"|эмблем\w*"
    r"|газет\w*"
    r"|удостоверени\w*"
    r"|паспорт\w*"
    r"|бумаг\w*|бумажн\w*"
    r"|письм(?:о|а|ам|ами|ах|ом)"
    r"|пергамент\w*"
    r"|бланк\w*|анкет\w*|формуляр\w*"
    r"|таблиц\w*)\b",
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
# Русские фразы: чистый лист, лист бумаги, лист пергамента.
_PHRASE_RU_RE = re.compile(
    r"\bчист(?:ый|ого|ому|ым|ом|ая|ую|ой|ие|их) лист\w*|\bлист\w* бумаги\b|\bлист\w* пергамента\b",
    re.IGNORECASE,
)


def needs_rephrase(segment_text: str) -> bool:
    """True, если в тексте сегмента есть слово или фраза запрещённого типа кадра (английская или русская)."""
    text = segment_text or ""
    return (
        bool(FORBIDDEN_FRAME_RE.search(text))
        or bool(FORBIDDEN_FRAME_RU_RE.search(text))
        or bool(_PHRASE_RE.search(text))
        or bool(_PHRASE_RU_RE.search(text))
    )


def rephrase_note(segment_text: str) -> str:
    """Правило перефразирования для сегмента, либо пустая строка, если оно не нужно."""
    return REPHRASE_RULE_EN if needs_rephrase(segment_text) else ""
