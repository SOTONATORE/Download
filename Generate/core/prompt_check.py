"""Проверка готовых промптов на запрещённые типы кадров и на нежелательное содержание (шаг 6 плана PLAN_generate_photo_video.md).

Модуль самодостаточен: только стандартная библиотека (re). Он не импортирует gemini_prompts,
чтобы не возникло циклического импорта; повтор запроса и ошибка остановки живут в gemini_prompts.

Список здесь английский, потому что промпты пишутся по-английски. Русский список запрещённых
типов в frame_rules.py (правило для текста сегмента) сюда не копируется по-русски и не заменяется.

Содержание кадра (дети, нагота, толпа, мелкие предметы, речь, руки у огня, битвы, оружие,
крупные планы лиц, внешность не из текста) проверяется здесь же. Это решения владельца проекта
после разбора прогона: правила не только в style.md, но и в коде, чтобы нарушения не уходили в GPU.
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
# содержание кадра
PROBLEM_CHILD = "child_in_frame"
PROBLEM_NUDITY = "nudity"
PROBLEM_PEOPLE_COUNT = "people_count"
PROBLEM_MANY_OF_ONE = "many_of_one_kind"
PROBLEM_SMALL_OBJECT = "small_object_action"
PROBLEM_SPEECH = "speech"
PROBLEM_FIRE_HANDS = "hands_near_fire"
PROBLEM_CHAOS = "battle_or_chaos"
PROBLEM_WEAPON = "weapon"
PROBLEM_CLOSE_FACE = "close_face"
PROBLEM_ETHNICITY = "ethnicity_not_in_text"

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

# Причины для журнала и для повторного запроса (русские), подписи для Gemini (английские), подсказки для Gemini.
CONTENT_REASON_RU = {
    PROBLEM_CHILD: "ребёнок или молодой человек в кадре",
    PROBLEM_NUDITY: "нагота или неодетый человек",
    PROBLEM_PEOPLE_COUNT: "слишком много людей или толпа",
    PROBLEM_MANY_OF_ONE: "много одинаковых машин или животных",
    PROBLEM_SMALL_OBJECT: "мелкий предмет или ручное действие с ним",
    PROBLEM_SPEECH: "речь, крик, смех или движение губ",
    PROBLEM_FIRE_HANDS: "руки рядом с огнём",
    PROBLEM_CHAOS: "битва, атака или хаотичное движение",
    PROBLEM_WEAPON: "оружие в кадре",
    PROBLEM_CLOSE_FACE: "крупный план лица",
    PROBLEM_ETHNICITY: "внешность или происхождение не из текста сегмента",
}
CONTENT_LABEL_EN = {
    PROBLEM_CHILD: "child in frame",
    PROBLEM_NUDITY: "nudity or undressed person",
    PROBLEM_PEOPLE_COUNT: "too many people or a crowd",
    PROBLEM_MANY_OF_ONE: "many identical vehicles or animals",
    PROBLEM_SMALL_OBJECT: "small object or hand action with it",
    PROBLEM_SPEECH: "speech, shouting, laughing or lip movement",
    PROBLEM_FIRE_HANDS: "hands near fire",
    PROBLEM_CHAOS: "battle or chaotic motion",
    PROBLEM_WEAPON: "weapon in frame",
    PROBLEM_CLOSE_FACE: "close-up of a face",
    PROBLEM_ETHNICITY: "origin or features not in the segment text",
}
CONTENT_HINT_EN = {
    PROBLEM_CHILD: "Remove every child, boy, girl, baby or young person. Show only adults, or no people, "
                   "and keep the segment's meaning through the setting and the objects.",
    PROBLEM_NUDITY: "Every person wears named clothing (for example a wool shirt, trousers, a greatcoat). "
                    "Show nobody naked, shirtless or undressed.",
    PROBLEM_PEOPLE_COUNT: "Show at most two people in a video and one person in a photo, each described "
                          "distinctly (age, clothing colour, hair). No crowds, groups, rows or plural people.",
    PROBLEM_MANY_OF_ONE: "Show at most one or two horses, vehicles or animals, each with a distinct "
                         "description. No herds, rows or fleets.",
    PROBLEM_SMALL_OBJECT: "Remove small objects and hand actions with them (needles, thread, abacus, beads, "
                          "cloth, bundles, cups, dung cakes, stacking, stitching, tucking). Show one large "
                          "object or the wide scene instead.",
    PROBLEM_SPEECH: "Remove all speaking, shouting, laughing, crying, singing and lip movement. Show emotion "
                    "only through posture.",
    PROBLEM_FIRE_HANDS: "Keep hands and fingers away from fire, flames, candles and lanterns. Show the fire "
                        "from a distance.",
    PROBLEM_CHAOS: "Remove battles, charges, gallops in formation, gunfire, explosions, smoke puffs and dust "
                   "clouds. Show a calm, static or slowly moving scene.",
    PROBLEM_WEAPON: "Remove all firearms, sabres, swords, spears and other weapons from the frame.",
    PROBLEM_CLOSE_FACE: "Remove close-ups and portraits. Show people from a medium or wide distance, with "
                        "faces small in the frame.",
    PROBLEM_ETHNICITY: "Describe people's origin, skin tone and features only as the segment text supports them. "
                       "Do not add an ethnicity, nationality or skin tone the text does not mention; use the "
                       "place and period named in the text.",
}
CONTENT_HINT_BY_REASON = {CONTENT_REASON_RU[c]: CONTENT_HINT_EN[c] for c in CONTENT_REASON_RU}
# Коды нарушений содержания (для счётчиков отчёта проверки).
CONTENT_CODES = frozenset(CONTENT_REASON_RU)

_PROBLEM_EN.update(CONTENT_LABEL_EN)
_PROBLEM_RU.update(CONTENT_REASON_RU)


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


# ---------------------------------------------------------------------------
# Содержание кадра
# ---------------------------------------------------------------------------

_CONTENT_RX = (
    (PROBLEM_CHILD, re.compile(
        r"\b(?:child|children|kids?|boys?|girls?|babies|baby|infants?|toddlers?|sons?|daughters?|"
        r"teens?|teenagers?|adolescents?|youths?|youngsters?|little ones?)\b", re.IGNORECASE)),
    (PROBLEM_NUDITY, re.compile(
        r"\b(?:naked|nude|nudity|nudes?|shirtless|topless|undressed|undress|unclothed|bare[- ]chested|"
        r"bare[- ]bodied)\b", re.IGNORECASE)),
    # plural и групповые слова: несколько людей или толпа
    (PROBLEM_PEOPLE_COUNT, re.compile(
        r"\b(?:people|men|women|soldiers|officers|riders|horsemen|villagers|farmers|herders|elders|workers|"
        r"guards|warriors|fighters|travell?ers|figures|persons|crowds?|groups?|troops?|several|many|dozens|"
        r"multiple|rows? of|lines? of)\b", re.IGNORECASE)),
    (PROBLEM_MANY_OF_ONE, re.compile(
        r"\b(?:horses|cars|carts|trucks|tanks|camels|sheep|cattle|cows|machines|vehicles|motorcycles|bikes|"
        r"wagons|carriages|dogs|goats|donkeys|oxen|tractors)\b", re.IGNORECASE)),
    (PROBLEM_SMALL_OBJECT, re.compile(
        r"\b(?:abacus|abacuses|needles?|threads?|threading|beads?|coins?|pens?|pencils?|cups?|bowls?|spoons?|"
        r"pillows?|bundles?|cloths?|rags?|ropes?|strings?|stitch\w*|sew\w*|sewing|stacking|stacks? of|"
        r"tuck\w*|knots?|knotting|folding|dung|cakes?|buckets?|shovels?|scythes?|sickles?|jugs?|bottles?|"
        r"spindles?|skeins?|twigs?|pebbles?|pick(?:s|ing)? up|picked up)\b", re.IGNORECASE)),
    (PROBLEM_SPEECH, re.compile(
        r"\b(?:says?|said|speaks?|speaking|spoke|talks?|talking|talked|shout\w*|yell\w*|cry|cries|crying|cried|"
        r"laugh\w*|whisper\w*|asks?|asking|asked|answers?|answering|answered|repl(?:y|ies|ied)|sings?|singing|"
        r"sang|lips?|mouth (?:moves|opens|is open|open)|dialogue|conversation|gestures?|gesturing|"
        r"gesticulat\w*|calls? out|hollers?|exclaims?|mutters?|murmurs?|chants?|chanting|screams?|screaming|"
        r"sobs?|sobbing|weeps?|weeping)\b", re.IGNORECASE)),
    # руки или пальцы рядом с огнём (в любом порядке, в пределах одного предложения)
    (PROBLEM_FIRE_HANDS, re.compile(
        r"\b(?:hands?|fingers?|palms?)\b[^.;]{0,60}\b(?:fire|fires|flames?|campfires?|bonfires?|candles?|"
        r"lanterns?|torch(?:es)?|embers?|burning|firelight)\b|"
        r"\b(?:fire|fires|flames?|campfires?|bonfires?|candles?|lanterns?|torch(?:es)?|embers?|firelight)\b"
        r"[^.;]{0,60}\b(?:hands?|fingers?|palms?)\b", re.IGNORECASE)),
    (PROBLEM_CHAOS, re.compile(
        r"\b(?:charg\w*|gallop\w*|battles?|battling|skirmish\w*|explo\w*|gunfire|firing|shoot\w*|bullets?|"
        r"smoke puffs?|smoke clouds?|clouds? of (?:dust|smoke)|in formation|assault\w*|attack\w*|ambush\w*|"
        r"rush\w*|stampede\w*|riot\w*|fight\w*|blood\w*|wounded)\b", re.IGNORECASE)),
    (PROBLEM_WEAPON, re.compile(
        r"\b(?:rifles?|guns?|machine ?guns?|automatics?|submachine guns?|pistols?|sabres?|sabers?|swords?|"
        r"bayonets?|muskets?|carbines?|knives|knife|daggers?|spears?|lances?|axes|whips?)\b", re.IGNORECASE)),
    # крупные планы лиц: только для фото (см. _content_hits)
    (PROBLEM_CLOSE_FACE, re.compile(
        r"\b(?:close-?ups?|extreme close-?ups?|portraits?|face fills? the frame|"
        r"fills? the frame with (?:a |the |his |her )?(?:face|head)|face in (?:sharp )?focus|sharp eyes|"
        r"faces? up close)\b", re.IGNORECASE)),
)
_IMAGE_ONLY = frozenset({PROBLEM_CLOSE_FACE})

# Люди по названию (для подсчёта): в видео не больше двух, в фото одного.
_PERSON_RX = re.compile(
    r"\b(?:man|woman|person|boy|girl|child|soldier|officer|rider|horseman|villager|farmer|herder|elder|"
    r"figure|worker|guard|warrior|fighter|traveller|traveler|lady|gentleman|mother|father|wife|husband|"
    r"son|daughter|kid|baby|infant|teenager)\b",
    re.IGNORECASE,
)
PEOPLE_LIMIT = {"video": 2, "image": 1}
# Вводящее упоминание человека («an older man», «beside her, a middle-aged man», «two riders»):
# каждый человек в промпте вводится один раз, а дальше называется «the older man», «the woman in brown».
# Считаем вводящие упоминания, а не все слова-названия: иначе два человека дают четыре «людей».
_INTRO_RX = re.compile(
    r"\b(an?|another|one|two|three|four|several|some|many)\s+(?:[\w-]+\s+){0,3}?"
    r"(?:man|woman|person|boy|girl|child|soldier|officer|rider|horseman|villager|farmer|herder|elder|"
    r"figure|worker|guard|warrior|fighter|traveller|traveler|lady|gentleman|mother|father|wife|husband|"
    r"son|daughter|kid|baby|infant|teenager|men|women|people|children|soldiers|officers|riders|villagers|"
    r"farmers|herders|elders|figures|workers|guards|warriors|fighters|travellers|travelers|ladies|gentlemen)\b",
    re.IGNORECASE,
)
_INTRO_COUNT = {"a": 1, "an": 1, "another": 1, "one": 1, "two": 2, "three": 3, "four": 4,
                "several": 3, "some": 3, "many": 4}


def count_people(text: str) -> int:
    """Сколько разных людей описано в промпте. Вводящие упоминания считаются по числу, остальные упоминания
    считаются по имени: «the older man» после «an older man» — тот же человек."""
    intros = 0
    for m in _INTRO_RX.finditer(text or ""):
        intros += _INTRO_COUNT.get(m.group(1).lower(), 1)
    if intros:
        return intros
    # вводящих нет (например, «the man» без «a man»): различаем по тексту названия без артикля
    return len({re.sub(r"^(the|a|an)\s+", "", m.group(0).lower()) for m in _PERSON_RX.finditer(text or "")})

# Этнические и национальные признаки: допустимы, только если сегмент сам о них говорит.
_ETHNIC_RX = re.compile(
    r"\b(?:dark[- ]skinned|dark skin|african|africans|arabs?|arabic (?:man|men|woman|women|people|boy|girl)|"
    r"indians?|south asians?|middle[- ]eastern|asians?|chinese|japanese|korean|"
    r"black (?:man|men|woman|women|people|person|boy|girl|villagers?|soldiers?|riders?|workers?|farmers?|"
    r"herders?))\b",
    re.IGNORECASE,
)
# Корни в тексте сегмента (русский и английский); любой из них разрешает этнический признак в промпте.
ETHNIC_ROOTS = (
    "араб", "индус", "индий", "африк", "негр", "темн", "тёмн", "черн", "чёрн", "азиат", "китай", "япон",
    "корей", "монгол", "arab", "indian", "india", "african", "africa", "black", "dark", "asian", "chinese",
    "japan", "korean", "mongol", "middle east", "south asia",
)


def content_hits(text: str, media: str = "video") -> list:
    """Нарушения содержания: список пар (код, фрагмент). Текст не меняется."""
    t = text or ""
    hits = []
    for code, rx in _CONTENT_RX:
        if code in _IMAGE_ONLY and media != "image":
            continue
        hits.extend((code, m.group(0)) for m in rx.finditer(t))
    people = count_people(t)
    limit = PEOPLE_LIMIT.get(media, PEOPLE_LIMIT["video"])
    if people > limit:
        hits.append((PROBLEM_PEOPLE_COUNT, f"людей в описании: {people}"))
    return hits


def content_reasons(prompt: str, media: str = "video") -> list:
    """Причины содержания по-русски для needs_review (без фрагментов), без повторов."""
    return list(dict.fromkeys(CONTENT_REASON_RU[code] for code, _ in content_hits(prompt, media)))


def ethnicity_reasons(prompt: str, segment_text: str) -> list:
    """Причина, если в промпте есть этнический признак, которого нет в тексте сегмента."""
    if not _ETHNIC_RX.search(prompt or ""):
        return []
    seg = (segment_text or "").lower()
    if any(root in seg for root in ETHNIC_ROOTS):
        return []
    return [CONTENT_REASON_RU[PROBLEM_ETHNICITY]]


def find_problems(prompt: str, media: str = "video") -> list:
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
    for code, fragment in content_hits(text, media):
        problems.append(make_problem(code, fragment))
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
