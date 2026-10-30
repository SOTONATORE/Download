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
# newspaper = газета, id card = удостоверение, passport = паспорт, form = формы, sheet = лист.
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
    "form",
    "sheet",
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

# Квадратные метки вида [CHAR_1], [OBJ_2]: в финальном промпте их быть не должно.
LEFTOVER_MARKERS_RE = re.compile(r"\[[A-Z_]+\d*\]")

# Сколько раз один промпт отправляется Gemini на переписывание после нарушения.
MAX_REWORK_ATTEMPTS = 2


def find_problems(prompt: str) -> list:
    """Список коротких сообщений на русском, по одному на найденное нарушение. Пусто, если нарушений нет."""
    text = prompt or ""
    problems = []
    for m in FORBIDDEN_IMAGE_RE.finditer(text):
        problems.append(f"запрещённый тип кадра: «{m.group(0)}»")
    for m in LEFTOVER_MARKERS_RE.finditer(text):
        problems.append(f"незаполненный маркер: «{m.group(0)}»")
    # одинаковые сообщения (например, «map» дважды) не повторяем
    return list(dict.fromkeys(problems))


def rework_remark(problems: list) -> str:
    """Английская помощь для Gemini: что исправить в промпте при повторном запросе."""
    joined = "; ".join(problems)
    return (
        "Rewrite this prompt. Remove these problems: "
        f"{joined}. "
        "Keep the subject, count, action and setting from the segment. "
        "Replace any forbidden frame type with a permitted image of the same subject "
        "(for example a map becomes a landscape or a city view without labels). "
        "Output only the rewritten prompt."
    )
