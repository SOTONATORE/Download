"""Запрещённые типы кадров для пайплайна Generate (шаг 4 плана PLAN_generate_photo_video.md).

Список запрещённых типов английский и общий с проверкой готового промпта (prompt_check.py):
он импортируется оттуда, а не копируется. Модуль использует только стандартную библиотеку (re)
и prompt_check.py, который тоже stdlib-only и не импортирует этот модуль, поэтому цикла нет.

Текст сегментов SRT русский, поэтому здесь же есть русские основы тех же типов кадров
(карта, документ, письмо, паспорт и т. д.). Без них правило не срабатывало на русском тексте.
"""

import re

try:  # пакетный и «плоский» импорт (как в gemini_prompts.py)
    from .prompt_check import (FORBIDDEN_IMAGE_PHRASES_EN, FORBIDDEN_IMAGE_WORDS_EN, _STEM_EXCLUDE, _stem_alternative,
                               released_terms)
except ImportError:  # pragma: no cover
    from prompt_check import (FORBIDDEN_IMAGE_PHRASES_EN, FORBIDDEN_IMAGE_WORDS_EN, _STEM_EXCLUDE, _stem_alternative,  # type: ignore
                              released_terms)

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

# Французские и европейские (DE/ES) основы запрещённых типов кадра для мультиязычных SRT.
FORBIDDEN_FRAME_EU_RE = re.compile(
    r"\b(?:cartes?"
    r"|calendriers?"
    r"|graphiques?"
    r"|diagrammes?"
    r"|infographies?"
    r"|drapeaux?"
    r"|blasons?"
    r"|armoiries"
    r"|embl[eè]mes?"
    r"|journa(?:l|ux)"
    r"|pi[eè]ces?\s+d['\u2019]identit[eé]"
    r"|passeports?"
    r"|papiers?"
    r"|lettres?"
    r"|parchemins?"
    r"|tableaux?"
    r"|formulaires?"
    r"|karten?"
    r"|kalendern?"
    r"|flaggen?"
    r"|reisep(?:ass|ässe|asse)"
    r"|zeitung(?:en)?"
    r"|brief(?:e|en)?"
    r"|mapas?"
    r"|banderas?"
    r"|pasaportes?"
    r"|peri[oó]dicos?)\b",
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
# Русские фразы с именем разрешённого типа (rules.yaml allowed_shot_items): чистый лист → blank sheet,
# лист бумаги → sheet of paper, лист пергамента → sheet of parchment (никогда не разрешён).
_RU_PHRASE_ITEMS = (
    (re.compile(r"\bчист(?:ый|ого|ому|ым|ом|ая|ую|ой|ие|их) лист\w*", re.IGNORECASE), "blank sheet"),
    (re.compile(r"\bлист\w* бумаги\b", re.IGNORECASE), "sheet of paper"),
    (re.compile(r"\bлист\w* пергамента\b", re.IGNORECASE), "sheet of parchment"),
)
# Русские основы слов с именем разрешённого типа: карта → map, бумага → paper и т. д. Пергамент, паспорт, флаг,
# герб, эмблема, удостоверение, бланк, анкета, формуляр сюда не входят: они всегда запрещены.
_RU_WORD_ITEMS = (
    ("карт", "map"), ("календар", "calendar"), ("документ", "document"), ("график", "chart"),
    ("диаграмм", "diagram"), ("инфографик", "infographic"), ("газет", "newspaper"), ("бумаг", "paper"),
    ("бумажн", "paper"), ("письм", "letter"), ("таблиц", "spreadsheet"),
)
# Французские фразы бумажных листов (как _RU_PHRASE_ITEMS).
_EU_PHRASE_ITEMS = (
    (re.compile(r"\bfeuilles?\s+blanches?\b", re.IGNORECASE), "blank sheet"),
    (re.compile(r"\bfeuilles?\s+de\s+papier\b", re.IGNORECASE), "sheet of paper"),
    (re.compile(r"\bfeuilles?\s+de\s+parchemin\b", re.IGNORECASE), "sheet of parchment"),
)
# Французские/DE/ES основы с именем разрешённого типа (rules.yaml allowed_shot_items).
# Всегда запрещённые (drapeau, blason, armoiries, emblème, pièce d'identité, passeport, parchemin,
# formulaire, flagge, reisepass, bandera, pasaporte) сюда не входят.
_EU_WORD_ITEMS = (
    ("carte", "map"), ("calendrier", "calendar"), ("graphique", "chart"),
    ("diagramme", "diagram"), ("infographie", "infographic"),
    ("journal", "newspaper"), ("journaux", "newspaper"), ("papier", "paper"),
    ("lettre", "letter"), ("tableau", "spreadsheet"),
    ("karte", "map"), ("kalender", "calendar"), ("zeitung", "newspaper"), ("brief", "letter"),
    ("mapa", "map"), ("periódico", "newspaper"), ("periodico", "newspaper"),
)


def _in_spans(span: tuple, spans: list) -> bool:
    """True, если отрезок целиком лежит внутри одного из разрешённых отрезков (фраз)."""
    return any(start <= span[0] and span[1] <= end for start, end in spans)


def uncovered_hits(segment_text: str, allowed_items=frozenset()) -> list:
    """Совпадения запрещённых типов кадра в тексте сегмента, которые разрешённые типы не снимают.

    allowed_items: имена из rules.yaml allowed_shot_items. Пусто — каждое совпадение запрещено (прежнее поведение).
    Разрешённое слово или фраза не считается совпадением; запрещённое слово рядом с разрешённым — считается.
    """
    text = segment_text or ""
    released = released_terms(allowed_items or ())
    hits = []
    # разрешённые фразы: отрезки, внутри которых слова уже сняты («paper» в «sheet of paper»)
    spans = []
    for m in _PHRASE_RE.finditer(text):
        if m.group(0).lower() in released:
            spans.append(m.span())
        else:
            hits.append(m.group(0))
    for rx, item in _RU_PHRASE_ITEMS + _EU_PHRASE_ITEMS:
        for m in rx.finditer(text):
            if item in released:
                spans.append(m.span())
            else:
                hits.append(m.group(0))
    for m in FORBIDDEN_FRAME_RE.finditer(text):
        word = m.group(0).lower()
        if _in_spans(m.span(), spans) or any(word.startswith(t) for t in released if t in FORBIDDEN_IMAGE_WORDS_EN):
            continue
        hits.append(m.group(0))
    for m in FORBIDDEN_FRAME_RU_RE.finditer(text):
        word = m.group(0).lower()
        item = next((it for stem, it in _RU_WORD_ITEMS if word.startswith(stem)), None)
        if item in released and not _in_spans(m.span(), spans):
            continue
        hits.append(m.group(0))
    for m in FORBIDDEN_FRAME_EU_RE.finditer(text):
        if _in_spans(m.span(), spans):
            continue
        word = m.group(0).lower()
        item = next((it for stem, it in _EU_WORD_ITEMS if word.startswith(stem)), None)
        if item in released:
            continue
        hits.append(m.group(0))
    return hits


def needs_rephrase(segment_text: str, allowed_items=frozenset()) -> bool:
    """True, если в тексте сегмента есть запрещённый тип кадра (английский или русский), не снятый разрешёнными."""
    return bool(uncovered_hits(segment_text, allowed_items))


def rephrase_note(segment_text: str, allowed_items=frozenset()) -> str:
    """Правило перефразирования для сегмента, либо пустая строка, если оно не нужно.
    allowed_items: разрешённые типы кадра ниши (rules.yaml); при пустом значении текст прежний."""
    return REPHRASE_RULE_EN if needs_rephrase(segment_text, allowed_items) else ""
