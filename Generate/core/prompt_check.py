"""Проверка готовых промптов на запрещённые типы кадров (шаг 6 плана PLAN_generate_photo_video.md).

Модуль самодостаточен: только стандартная библиотека (re). Он не импортирует gemini_prompts,
чтобы не возникло циклического импорта; повтор запроса и ошибка остановки живут в gemini_prompts.

Список здесь английский, потому что промпты пишутся по-английски. Русский список запрещённых
типов в frame_rules.py (правило для текста сегмента) сюда не копируется по-русски и не заменяется.
"""

import re

# Английские основы. Соответствуют русскому списку frame_rules.py:
# map = карта, calendar = календарь, document = документ, chart = график, diagram = диаграмма,
# infographic = инфографика, flag = флаг, coat of arms = герб, emblem = эмблема (герб),
# newspaper = газета, id card = удостоверение, passport = паспорт, form = формы, spreadsheet = таблица; обычное sheet не считается.
FORBIDDEN_IMAGE_WORDS_EN = (
    "map",
    "calendar",
    "document",
    "chart",
    "diagram",
    "infographic",
    "flag",
    "coat of arms",
    "emblem",
    "newspaper",
    "id card",
    "passport",
    "paper",  # бумага: листы, записи, бумажные документы
    "letter",  # письмо: письма, письменные послания
    "parchment",  # пергамент
    "form",
    "spreadsheet",
)

# Исключения для основ, которые являются началом обычных слов. «form» входит в «format», «formal»,
# «formation», «formula»: эти слова допустимы, поэтому ставится отрицательный просмотр вперёд.
_STEM_EXCLUDE = {
    "form": ("format", "formal", "formation", "formula"),
}


def _stem_alternative(stem: str) -> str:
    """Регулярное выражение для одной основы с учётом исключений (как _stem_alternative в frame_rules.py)."""
    negatives = "".join(f"(?!{re.escape(e)})" for e in _STEM_EXCLUDE.get(stem, ()))
    return f"{negatives}{re.escape(stem)}\\w*"


# Совпадение только с начала слова (\b перед основой), без учёта регистра, с любым окончанием.
FORBIDDEN_IMAGE_RE = re.compile(
    r"\b(?:" + "|".join(_stem_alternative(s) for s in FORBIDDEN_IMAGE_WORDS_EN) + r")",
    re.IGNORECASE,
)

# бумажные листы: только точные сочетания, слово sheet само по себе разрешено.
FORBIDDEN_IMAGE_PHRASES_EN = (
    "blank sheet",
    "sheet of paper",
    "sheet of parchment",
)
# Точное сочетание целиком, без учёта регистра.
FORBIDDEN_IMAGE_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in FORBIDDEN_IMAGE_PHRASES_EN) + r")\b",
    re.IGNORECASE,
)

# форматы кадра и соотношения сторон.
FORBIDDEN_FORMAT_PATTERNS_EN = re.compile(
    r"\b\d{1,2}:\d{1,2}\b|\bwidescreen\b|\baspect ratio\b|\bletterbox\b|\bpillarbox\b|\bframing\b",
    re.IGNORECASE,
)

# Квадратные метки вида [CHAR_1], [OBJ_2]: в финальном промпте их быть не должно.
LEFTOVER_MARKERS_RE = re.compile(r"\[[A-Z_]+\d*\]")

# Сколько раз один промпт отправляется Gemini на переписывание после нарушения.
MAX_REWORK_ATTEMPTS = 2


# Машинные коды нарушений. cli и тесты сравнивают код, а не текст сообщения.
PROBLEM_FORBIDDEN_SHOT_TYPE = "forbidden_shot_type"
PROBLEM_FORBIDDEN_COMBINATION = "forbidden_combination"
PROBLEM_FORBIDDEN_FORMAT = "forbidden_format"
PROBLEM_LEFTOVER_MARKER = "leftover_marker"
PROBLEM_CAMERA_MOTION = "camera_motion"

# Подписи на английском для промпта Gemini и на русском для журнала, отчёта и текста ошибки.
_PROBLEM_EN = {
    PROBLEM_FORBIDDEN_SHOT_TYPE: "forbidden shot type",
    PROBLEM_FORBIDDEN_COMBINATION: "forbidden combination",
    PROBLEM_FORBIDDEN_FORMAT: "forbidden frame format",
    PROBLEM_LEFTOVER_MARKER: "unfilled marker",
    PROBLEM_CAMERA_MOTION: "camera motion",
}
_PROBLEM_RU = {
    PROBLEM_FORBIDDEN_SHOT_TYPE: "запрещённый тип кадра",
    PROBLEM_FORBIDDEN_COMBINATION: "запрещённое сочетание",
    PROBLEM_FORBIDDEN_FORMAT: "запрещённый формат кадра",
    PROBLEM_LEFTOVER_MARKER: "незаполненный маркер",
    PROBLEM_CAMERA_MOTION: "движение камеры",
}


def make_problem(code: str, fragment: str) -> str:
    """Сообщение о нарушении вида «код: «фрагмент»». Код идёт до первого двоеточия."""
    return f"{code}: «{fragment}»"


def problem_code(problem: str) -> str:
    """Машинный код нарушения."""
    return problem.partition(": ")[0]


def problem_en(problem: str) -> str:
    """Английская запись нарушения для промпта Gemini."""
    code, _, rest = problem.partition(": ")
    return f"{_PROBLEM_EN[code]}: {rest}"


def problem_ru(problem: str) -> str:
    """Русская запись нарушения для журнала, отчёта и текста ошибки."""
    code, _, rest = problem.partition(": ")
    return f"{_PROBLEM_RU[code]}: {rest}"


def find_problems(prompt: str) -> list:
    """Список сообщений вида «код: «фрагмент»», по одному на нарушение. Пусто, если нарушений нет."""
    text = prompt or ""
    problems = []
    for m in FORBIDDEN_IMAGE_RE.finditer(text):
        problems.append(make_problem(PROBLEM_FORBIDDEN_SHOT_TYPE, m.group(0)))
    for m in FORBIDDEN_IMAGE_PHRASE_RE.finditer(text):
        problems.append(make_problem(PROBLEM_FORBIDDEN_COMBINATION, m.group(0)))
    for m in FORBIDDEN_FORMAT_PATTERNS_EN.finditer(text):
        problems.append(make_problem(PROBLEM_FORBIDDEN_FORMAT, m.group(0)))
    for m in LEFTOVER_MARKERS_RE.finditer(text):
        problems.append(make_problem(PROBLEM_LEFTOVER_MARKER, m.group(0)))
    # одинаковые сообщения (например, «map» дважды) не повторяем
    return list(dict.fromkeys(problems))


def rework_remark(problems: list) -> str:
    """Английская помощь для Gemini: что исправить в промпте при повторном запросе."""
    joined = "; ".join(problem_en(p) for p in problems)
    return (
        "Rewrite this prompt. Remove these problems: "
        f"{joined}. "
        "Keep the subject, count, action and setting from the segment. "
        "Replace any forbidden frame type with a permitted image of the same subject "
        "(for example a map becomes a landscape or a city view without labels). "
        "Output only the rewritten prompt."
    )


# Слова движения камеры (английские, совпадение целым словом, без учёта регистра).
CAMERA_MOTION_WORDS = frozenset({
    "pan", "pans", "panning", "zoom", "zooms", "zooming", "track", "tracks", "tracking",
    "dolly", "dollies", "tilt", "tilts", "tilting", "push", "pushes", "pushing",
    "pull", "pulls", "pulling", "rotate", "rotates", "rotating", "move", "moves", "moving",
    "follow", "follows", "following", "slide", "slides", "sliding", "glide", "glides", "gliding",
})
# Слова-якоря: движение засчитывается, только если оно рядом с «camera» или «frame».
CAMERA_ANCHOR_WORDS = frozenset({"camera", "frame"})
CAMERA_MOTION_WINDOW = 3  # в пределах трёх слов от якоря, в любом порядке
_WORD_RE = re.compile(r"[A-Za-z]+")


def find_camera_motion(text: str) -> list:
    """Фрагменты текста, где слово движения стоит рядом с «camera» или «frame». Текст не меняется."""
    words = [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(text or "")]
    found = []
    for i, (word, _, _) in enumerate(words):
        if word not in CAMERA_ANCHOR_WORDS:
            continue
        lo = max(0, i - CAMERA_MOTION_WINDOW)
        hi = min(len(words), i + CAMERA_MOTION_WINDOW + 1)
        for j in range(lo, hi):
            if j != i and words[j][0] in CAMERA_MOTION_WORDS:
                start = min(words[i][1], words[j][1])
                end = max(words[i][2], words[j][2])
                found.append(text[start:end])
                break
    return found
