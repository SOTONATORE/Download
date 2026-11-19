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
from dataclasses import dataclass
from typing import Optional

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

# Разрешённые типы кадра (rules.yaml: allowed_shot_items). Ключ — имя из rules.yaml, значение — запрещённые слова
# или фразы выше, которые это имя снимает. Соответствие записано явно, не через проверку ниши.
#   letter → letter; lettering → ничего (надпись без отдельного запрещённого слова); map → map;
#   map pin → map (булавка на карте: слово map разрешено, как у карты); diagram → diagram; chart → chart;
#   infographic → infographic; calendar → calendar; spreadsheet → spreadsheet; newspaper → newspaper;
#   document → document; paper → paper; blank sheet → фраза «blank sheet»;
#   sheet of paper → фраза «sheet of paper» (слово paper внутри неё снимается вместе с фразой).
# Паспорт, флаг, герб, эмблема, пергамент, бланк, форма и «sheet of parchment» сюда не входят: остаются запрещёнными.
ALLOWED_SHOT_ITEM_TERMS = {
    "letter": ("letter",),
    "lettering": (),
    "map": ("map",),
    "map pin": ("map",),
    "diagram": ("diagram",),
    "chart": ("chart",),
    "infographic": ("infographic",),
    "calendar": ("calendar",),
    "spreadsheet": ("spreadsheet",),
    "newspaper": ("newspaper",),
    "document": ("document",),
    "paper": ("paper",),
    "blank sheet": ("blank sheet",),
    "sheet of paper": ("sheet of paper",),
}
# Проверка при загрузке модуля: каждое слово и фраза словаря должны быть в списках запрещённых типов кадра.
for _terms in ALLOWED_SHOT_ITEM_TERMS.values():
    for _term in _terms:
        if _term not in FORBIDDEN_IMAGE_WORDS_EN and _term not in FORBIDDEN_IMAGE_PHRASES_EN:
            raise RuntimeError(f"ALLOWED_SHOT_ITEM_TERMS: «{_term}» нет в списках запрещённых типов кадра")


def validate_allowed_shot_items(names) -> frozenset:
    """Имена allowed_shot_items из rules.yaml. Неизвестное имя — ValueError с перечнем допустимых."""
    items = frozenset(names)
    unknown = sorted(name for name in items if name not in ALLOWED_SHOT_ITEM_TERMS)
    if unknown:
        raise ValueError(f"неизвестный тип кадра «{', '.join(unknown)}»; "
                         f"допустимые: {', '.join(sorted(ALLOWED_SHOT_ITEM_TERMS))}")
    return items


def released_terms(allowed_items) -> frozenset:
    """Запрещённые слова и фразы, которые сняты разрешёнными типами кадра (пусто, если ничего не разрешено)."""
    out = set()
    for name in allowed_items:
        out.update(ALLOWED_SHOT_ITEM_TERMS[name])
    return frozenset(out)


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
# наборы предметов (стопки, кучи, ряды, линии): только при rules object_sets: forbidden
PROBLEM_OBJECT_SET = "object_set"
# толстые контуры и векторный стиль: только при rules forbid_thick_vector: true (задаёт ниша в rules.yaml)
PROBLEM_THICK_VECTOR = "thick_vector_style"
# незаконченный текст в кадре (надпись обрывается на служебном слове): только при rules dangling_text_check: true
PROBLEM_DANGLING_TEXT = "dangling_text_fragment"
# надпись в кадре не из реплики текущего сегмента: только при dangling_text_check: true и переданном segment_text
PROBLEM_LETTERING_NOT_IN_SEGMENT = "lettering_not_in_segment"
# текст в кадре у клипа, которому после 3 кругов нужен кадр без текста: только при textless=True и dangling_text_check: true
PROBLEM_TEXT_IN_TEXTLESS_FRAME = "text_in_textless_frame"
LETTERING_SEGMENT_HINT_EN = "copy the on-frame text exactly from the current subtitle line only"
# английская подсказка для повтора: дописать текст в кадре до полной фразы
DANGLING_TEXT_HINT_EN = "finish the in-frame text as a complete 1 to 4 word phrase"
# метки <imageN> и ref_N в промпте профиля без референсов (refs_enabled: false)
PROBLEM_REF_TAG = "ref_tag_without_refs"
# одинаковый текст промпта у разных клипов (нарушение очереди дубликатов, ниша с fail_on_duplicate_prompts)
PROBLEM_DUPLICATE_PROMPT = "duplicate_prompt"
# английская подсказка для повтора: что нарисовать вместо толстых контуров
THICK_VECTOR_HINT_EN = ("thick or bold outlines, or a vector look; use thin slightly sketchy black ink pen outlines "
                        "and thin single-line pure-black stick-figure arms instead")

# Подписи на английском для промпта Gemini и на русском для журнала, отчёта и текста ошибки.
_PROBLEM_EN = {
    PROBLEM_FORBIDDEN_SHOT_TYPE: "forbidden shot type",
    PROBLEM_FORBIDDEN_COMBINATION: "forbidden combination",
    PROBLEM_FORBIDDEN_FORMAT: "forbidden frame format",
    PROBLEM_LEFTOVER_MARKER: "unfilled marker",
    PROBLEM_CAMERA_MOTION: "camera motion",
    PROBLEM_THICK_VECTOR: THICK_VECTOR_HINT_EN,
    PROBLEM_DANGLING_TEXT: DANGLING_TEXT_HINT_EN,
    PROBLEM_LETTERING_NOT_IN_SEGMENT: LETTERING_SEGMENT_HINT_EN,
    PROBLEM_TEXT_IN_TEXTLESS_FRAME: "this frame must contain no on-frame text; remove all lettering",
    PROBLEM_REF_TAG: "image tag <imageN> or ref_N in a niche without reference images; write no tags",
    PROBLEM_DUPLICATE_PROMPT: "the same prompt text as another clip",
}
_PROBLEM_RU = {
    PROBLEM_FORBIDDEN_SHOT_TYPE: "запрещённый тип кадра",
    PROBLEM_FORBIDDEN_COMBINATION: "запрещённое сочетание",
    PROBLEM_FORBIDDEN_FORMAT: "запрещённый формат кадра",
    PROBLEM_LEFTOVER_MARKER: "незаполненный маркер",
    PROBLEM_CAMERA_MOTION: "движение камеры",
    PROBLEM_THICK_VECTOR: "толстые контуры или векторный стиль",
    PROBLEM_REF_TAG: "метка <imageN> или ref_N без референсов",
    PROBLEM_DUPLICATE_PROMPT: "дубликат промпта",
    PROBLEM_DANGLING_TEXT: "незаконченный текст в кадре",
    PROBLEM_LETTERING_NOT_IN_SEGMENT: "надпись не из реплики этого сегмента",
    PROBLEM_TEXT_IN_TEXTLESS_FRAME: "текст в кадре, который должен быть без текста",
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
    PROBLEM_OBJECT_SET: "стопки, кучи, ряды или линии предметов",
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
    PROBLEM_OBJECT_SET: "stacks, piles, rows or lines of objects",
}
CONTENT_HINT_EN = {
    PROBLEM_OBJECT_SET: "Describe one object, or two or three objects; never write stacks, piles, rows or lines of objects.",
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


def problem_label(code: str) -> Optional[str]:
    """Русская подпись правила по коду нарушения (фиксированный текст из таблицы); None, если код неизвестен."""
    return _PROBLEM_RU.get(code)


# ---------------------------------------------------------------------------
# Содержание кадра
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Значения правил содержания из prompts/rules.yaml (ниша или модель). Файл читает gemini_prompts.load_rules.
# Поля по умолчанию дают прежнее поведение ltx25 и zimage_turbo.
# ---------------------------------------------------------------------------

EMOTION_VALUES = ("posture_and_action", "posture", "action")
SUBJECT_SOURCE_VALUES = ("text_only", "text_and_context")


@dataclass
class RulesValues:
    max_characters: Optional[int] = None         # персонажи в кадре; None — лимит людей (max_people_per_image или по виду медиа)
    animals_count: bool = True                   # животные входят в max_characters
    max_people_per_image: Optional[int] = None   # лимит людей, если max_characters не задан; None — PEOPLE_LIMIT по виду медиа
    small_objects_allowed: bool = False          # мелкие предметы разрешены (rules: small_objects)
    speech_allowed: bool = False                 # речь и движение губ разрешены (rules: speech)
    weapons_allowed: bool = False                # оружие в кадре разрешено (rules: weapons)
    close_faces_allowed: bool = False            # крупные планы лиц разрешены (rules: close_faces; только фото)
    static_frame: bool = True                    # фото: один неподвижный кадр, без слов о движении камеры
    emotion: str = "posture"                     # posture_and_action | posture | action
    subject_source: str = "text_and_context"     # text_only | text_and_context
    horses_vehicles_animals_max: int = 2         # лимит лошадей, машин и животных в группе (текст для модели)
    scene_motion_wording: str = "keep"           # keep | drop: слова «and its motion» в строке медиа для изображений (rules: scene_motion_wording)
    object_sets_forbidden: bool = False          # стопки, кучи, ряды и линии предметов запрещены (rules: object_sets: forbidden)
    allowed_shot_items: frozenset = frozenset()  # разрешённые типы кадра, имена из ALLOWED_SHOT_ITEM_TERMS (rules: allowed_shot_items)
    forbid_thick_vector: bool = False            # толстые контуры и векторный стиль запрещены (rules: forbid_thick_vector)
    sketch_style: bool = False                   # стиль эскиза (rules: sketch_style); отдельной проверки не включает
    dangling_text_check: bool = False            # незаконченный текст в кадре запрещён (rules: dangling_text_check)
    fire_hands_allowed: bool = False             # руки рядом с огнём разрешены (rules: fire_hands; по умолчанию запрещены)


def rule_skipped_codes(rules: RulesValues) -> frozenset:
    """Коды нарушений, которые правила ниши отключают (значение allowed). Правила безопасности сюда не входят."""
    pairs = ((rules.small_objects_allowed, PROBLEM_SMALL_OBJECT), (rules.speech_allowed, PROBLEM_SPEECH),
             (rules.weapons_allowed, PROBLEM_WEAPON), (rules.close_faces_allowed, PROBLEM_CLOSE_FACE),
             (rules.fire_hands_allowed, PROBLEM_FIRE_HANDS))
    return frozenset(code for allowed, code in pairs if allowed)


# Подсказка «много одинаковых» для машин и механизмов: животные при гейте считаются лимитом персонажей
MANY_OF_ONE_VEHICLES_HINT_EN = ("Show at most one or two vehicles or machines, each with a distinct description. "
                                "No rows or fleets.")

# Много одинаковых: животные во множественном числе и машины или механизмы (два списка с теми же словами, что были)
_MANY_OF_ONE_ANIMAL_WORDS = ("horses", "camels", "sheep", "cows", "donkeys", "goats", "dogs", "oxen")
_MANY_OF_ONE_VEHICLE_WORDS = ("cars", "carts", "trucks", "tanks", "cattle", "machines", "vehicles", "motorcycles",
                              "bikes", "wagons", "carriages", "tractors")


def _word_list_rx(words) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


_MANY_OF_ONE_ALL_RX = _word_list_rx(_MANY_OF_ONE_ANIMAL_WORDS + _MANY_OF_ONE_VEHICLE_WORDS)
_MANY_OF_ONE_VEHICLES_RX = _word_list_rx(_MANY_OF_ONE_VEHICLE_WORDS)


def animal_plurals_by_count(max_characters: Optional[int], animals_count: bool = True) -> bool:
    """True, если животные во множественном числе проверяются лимитом персонажей, а не правилом «много одинаковых».
    Гейт: задан max_characters и животные входят в подсчёт. Без max_characters — прежняя проверка обоих списков."""
    return max_characters is not None and bool(animals_count)


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
    (PROBLEM_MANY_OF_ONE, _MANY_OF_ONE_ALL_RX),
    (PROBLEM_SMALL_OBJECT, re.compile(
        r"\b(?:abacus|abacuses|needles?|threads?|threading|beads?|coins?|pens?|pencils?|cups?|bowls?|spoons?|"
        r"pillows?|bundles?|cloths?|rags?|ropes?|strings?|stitch\w*|sew\w*|sewing|stacking|stacks? of|"
        r"tuck\w*|knots?|knotting|folding|dung|cakes?|buckets?|shovels?|scythes?|sickles?|jugs?|bottles?|"
        r"spindles?|skeins?|twigs?|pebbles?|pick(?:s|ing)? up|picked up)\b", re.IGNORECASE)),
    # «yell\w*» давал ложное срабатывание на цвет «yellow»: формы «yell», «yells», «yelling», «yelled» перечисляем явно
    (PROBLEM_SPEECH, re.compile(
        r"\b(?:says?|said|speaks?|speaking|spoke|talks?|talking|talked|shout\w*|yells?|yelling|yelled|cry|cries|crying|cried|"
        r"laugh\w*|whisper\w*|asks?|asking|asked|answers?|answering|answered|repl(?:y|ies|ied)|sings?|singing|"
        r"sang|lips? (?:move|moves|moving|part|parts|open|opens|tremble|trembles|quiver|quivers)|moving lips|"
        r"mouth (?:moves|opens|is open|open)|dialogue|conversation|gestures?|gesturing|"
        r"gesticulat\w*|calls? out|hollers?|exclaims?|mutters?|murmurs?|chants?|chanting|screams?|screaming|"
        r"sobs?|sobbing|weeps?|weeping)\b", re.IGNORECASE)),
    # руки или пальцы рядом с огнём (в любом порядке, в пределах одного предложения)
    (PROBLEM_FIRE_HANDS, re.compile(
        r"\b(?:hands?|fingers?|palms?)\b[^.;]{0,60}\b(?:fire|fires|flames?|campfires?|bonfires?|candles?|"
        r"lanterns?|torch(?:es)?|embers?|burning|firelight)\b|"
        r"\b(?:fire|fires|flames?|campfires?|bonfires?|candles?|lanterns?|torch(?:es)?|embers?|firelight)\b"
        r"[^.;]{0,60}\b(?:hands?|fingers?|palms?)\b", re.IGNORECASE)),
    (PROBLEM_CHAOS, re.compile(
        r"\b(?:charg\w*|gallop\w*|battles?|battling|skirmish\w*|explo\w*|gunfire|firing|shoot(?:s|ing)? (?:at|back)|shot at|bullets?|"
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
# При max_characters слова, которые описывают не людей (figures, several, many, multiple, lines of), из проверки
# людей убраны: «stick-figures» и «several circles» не считаются толпой. Толпа, группы, ряды людей и люди остаются.
_PEOPLE_COUNT_CHARS_RX = re.compile(
    r"\b(?:people|men|women|soldiers|officers|riders|horsemen|villagers|farmers|herders|elders|workers|"
    r"guards|warriors|fighters|travell?ers|persons|crowds?|groups?|troops?|dozens|rows? of)\b", re.IGNORECASE)
# наборы предметов: отдельный шаблон, включается только правилом object_sets: forbidden (см. content_hits)
_OBJECT_SET_RX = re.compile(r"\b(?:stacks?|piles?|rows?|lines?) of\b", re.IGNORECASE)
# шаблон людей берём из _CONTENT_RX: «rows of» и «lines of» уже отмечены как PROBLEM_PEOPLE_COUNT
_PEOPLE_COUNT_RX = dict(_CONTENT_RX)[PROBLEM_PEOPLE_COUNT]

# Правила, которые профиль может отключить (relaxed_rules). Только эти четыре id; правила безопасности сюда не входят.
RELAXABLE_RULES = frozenset({"art_medium", "small_objects", "frame_text", "emotion"})
# отключаемое правило → код нарушения содержания (art_medium — в gemini_prompts, frame_text — в системной инструкции)
_RELAXED_CONTENT_CODE = {"small_objects": PROBLEM_SMALL_OBJECT}

# Люди по названию (для подсчёта): в видео не больше двух, в фото одного.
_PERSON_RX = re.compile(
    r"\b(?:man|woman|person|boy|girl|child|soldier|officer|rider|horseman|villager|farmer|herder|elder|"
    r"worker|guard|warrior|fighter|traveller|traveler|lady|gentleman|mother|father|wife|husband|"
    r"son|daughter|kid|baby|infant|teenager)\b",
    re.IGNORECASE,
)
PEOPLE_LIMIT = {"video": 2, "image": 1}

# Этнические и национальные признаки: допустимы, только если сегмент сам о них говорит.
# Названия мест и животных (East African grassland, African elephant) не признак внешности: проверяются
# только формы про людей (African man, African woman, African people, African boy, African girl).
_ETHNIC_RX = re.compile(
    r"\b(?:dark[- ]skinned|dark skin|african (?:man|men|woman|women|people|boy|girl)|arabs?|arabic (?:man|men|woman|women|people|boy|girl)|"
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


_DEFINITE_BEFORE = frozenset({"the", "his", "her", "their", "its", "my", "your", "this", "that", "these", "those", "our"})


def _is_repeat_mention(text: str, start: int) -> bool:
    """True, если упоминание человека отсылает к уже введённому («the man», «his wife», «the standing man»).

    Считаются только введения: «One man», «a woman», «two officers». Повтор того же человека не считается вторым.
    """
    words = re.findall(r"[A-Za-z]+", text[max(0, start - 40):start].lower())[-3:]
    return any(w in _DEFINITE_BEFORE for w in words)


_PLURAL_PERSON_RX = re.compile(
    r"\b(?:men|women|people|soldiers|officers|riders|horsemen|villagers|farmers|herders|elders|workers|guards|"
    r"warriors|fighters|travell?ers|persons|children|boys|girls|kids|sons|daughters|figures)\b", re.IGNORECASE)
_CROWD_RX = re.compile(r"\b(?:crowds?|groups?|troops?|several|many|dozens|multiple)\b", re.IGNORECASE)
# варианты для подсчёта персонажей при max_characters: без figures, several, many, multiple (не люди)
_PLURAL_PERSON_HUMAN_RX = re.compile(
    r"\b(?:men|women|people|soldiers|officers|riders|horsemen|villagers|farmers|herders|elders|workers|guards|"
    r"warriors|fighters|travell?ers|persons|children|boys|girls|kids|sons|daughters)\b", re.IGNORECASE)
_CROWD_HUMAN_RX = re.compile(r"\b(?:crowds?|groups?|troops?|dozens)\b", re.IGNORECASE)


def count_people(text: str, human_only: bool = False) -> int:
    """Оценка числа людей в тексте: введения («One man», «a woman») по одному; множественное число — не меньше двух;
    толпа и группа — не меньше трёх. Повторные упоминания одного человека не считаются.
    human_only: только слова про людей (figures, several, many, multiple не считаются), для max_characters."""
    t = text or ""
    plural_rx, crowd_rx = (_PLURAL_PERSON_HUMAN_RX, _CROWD_HUMAN_RX) if human_only else (_PLURAL_PERSON_RX, _CROWD_RX)
    singular = sum(1 for m in _PERSON_RX.finditer(t) if not _is_repeat_mention(t, m.start()))
    return singular + 2 * len(plural_rx.findall(t)) + 3 * len(crowd_rx.findall(t))


# Животные для подсчёта персонажей (нужны только при max_characters): одиночное и множественное число
_ANIMAL_RX = re.compile(
    r"\b(?:dog|cat|horse|cow|goat|sheep|bird|wolf|fox|bear|rabbit|camel|donkey|goose|chicken|pig)(s|es)?\b",
    re.IGNORECASE)
# Герой кадра: по style.md он назван «the main character» или «main hero»; считается один раз
_HERO_RX = re.compile(r"\bmain (?:character|hero)\b", re.IGNORECASE)


def _animal_matches(text: str) -> list:
    """Упоминания животных, которые вводят новое животное (повтор «the dog» не считается)."""
    return [m for m in _ANIMAL_RX.finditer(text) if not _is_repeat_mention(text, m.start())]


# Число перед животным: слово two–five или цифра 2–9 сразу перед словом (через пробелы)
_NUMBER_BEFORE_ANIMAL_RX = re.compile(r"(?:\b(two|three|four|five)|\b([2-9]))\s+$", re.IGNORECASE)
_NUMBER_WORD_VALUE = {"two": 2, "three": 3, "four": 4, "five": 5}


def _animal_count(text: str, m) -> int:
    """Животных в одном упоминании: число перед словом; без числа — множественное 2, единственное 1."""
    before = _NUMBER_BEFORE_ANIMAL_RX.search(text[:m.start()])
    if before:
        return _NUMBER_WORD_VALUE[before.group(1).lower()] if before.group(1) else int(before.group(2))
    return 2 if m.group(1) else 1


def count_characters(text: str, animals_count: bool = True) -> int:
    """Персонажи для max_characters: люди (count_people), животные (число перед словом, иначе множественное — 2,
    единственное — 1) и герой (1, если назван). animals_count False (rules.yaml): животные не входят в подсчёт."""
    t = text or ""
    animals = sum(_animal_count(t, m) for m in _animal_matches(t)) if animals_count else 0
    return count_people(t, human_only=True) + animals + (1 if _HERO_RX.search(t) else 0)


def people_hint(max_characters: Optional[int], animals_count: bool = True) -> str:
    """Подсказка модели при повторе за число персонажей. Без max_characters — прежняя подсказка."""
    if max_characters is None:
        return CONTENT_HINT_EN[PROBLEM_PEOPLE_COUNT]
    scope = "people and animals together" if animals_count else "people only"
    return (f"Show at most {max_characters} characters in total, {scope}, the main hero "
            "included. Describe every character other than the main hero fully (age, clothing colour, hair, "
            "or the animal's colour and size). No crowds, groups, rows or plural people.")


def content_hits(text: str, media: str = "video", relaxed=frozenset(), max_characters: Optional[int] = None,
                 rules: Optional[RulesValues] = None) -> list:
    """Нарушения содержания: список пар (код, фрагмент). Текст не меняется.
    relaxed: отключённые профилем правила из RELAXABLE_RULES; правила безопасности отключить нельзя.
    max_characters: лимит персонажей профиля (герой и остальные); None — прежний PEOPLE_LIMIT.
    rules: значения rules.yaml; None — значения по умолчанию (прежнее поведение)."""
    t = text or ""
    rules = rules or RulesValues()
    hits = []
    skipped = frozenset(_RELAXED_CONTENT_CODE[r] for r in relaxed if r in _RELAXED_CONTENT_CODE) | rule_skipped_codes(rules)
    # гейт: животные во множественном числе считает лимит персонажей; список машин проверяется всегда
    many_rx = _MANY_OF_ONE_VEHICLES_RX if animal_plurals_by_count(max_characters, rules.animals_count) else _MANY_OF_ONE_ALL_RX
    for code, rx in _CONTENT_RX:
        if code in skipped:
            continue
        if code == PROBLEM_MANY_OF_ONE:
            rx = many_rx
        if code == PROBLEM_PEOPLE_COUNT and max_characters is not None:
            rx = _PEOPLE_COUNT_CHARS_RX
        if code in _IMAGE_ONLY and media != "image":
            continue
        hits.extend((code, m.group(0)) for m in rx.finditer(t))
    # наборы предметов: не зависят от relaxed и от rule_skipped_codes; совпадение с людьми (тот же фрагмент) не дублируем
    if rules.object_sets_forbidden:
        people_spans = {m.span() for m in _PEOPLE_COUNT_RX.finditer(t)}
        hits.extend((PROBLEM_OBJECT_SET, m.group(0)) for m in _OBJECT_SET_RX.finditer(t)
                    if m.span() not in people_spans)
    if max_characters is None:
        limit = rules.max_people_per_image if rules.max_people_per_image is not None else PEOPLE_LIMIT.get(media, PEOPLE_LIMIT["video"])
        if count_people(t) > limit:
            found = [m.group(0) for m in _PERSON_RX.finditer(t) if not _is_repeat_mention(t, m.start())]
            found += _PLURAL_PERSON_RX.findall(t) + _CROWD_RX.findall(t)
            hits.append((PROBLEM_PEOPLE_COUNT, ", ".join(found)))
    elif count_characters(t, rules.animals_count) > max_characters:
        found = [m.group(0) for m in _PERSON_RX.finditer(t) if not _is_repeat_mention(t, m.start())]
        animal_words = [m.group(0) for m in _animal_matches(t)] if rules.animals_count else []
        found += _PLURAL_PERSON_RX.findall(t) + _CROWD_RX.findall(t) + animal_words
        hits.append((PROBLEM_PEOPLE_COUNT, ", ".join(found)))
    return hits


def content_reasons(prompt: str, media: str = "video", relaxed=frozenset(), max_characters: Optional[int] = None,
                    rules: Optional[RulesValues] = None) -> list:
    """Причины содержания по-русски для needs_review (без фрагментов), без повторов."""
    return list(dict.fromkeys(CONTENT_REASON_RU[code]
                              for code, _ in content_hits(prompt, media, relaxed, max_characters, rules)))


def ethnicity_reasons(prompt: str, segment_text: str) -> list:
    """Причина, если в промпте есть этнический признак, которого нет в тексте сегмента."""
    if not _ETHNIC_RX.search(prompt or ""):
        return []
    seg = (segment_text or "").lower()
    if any(root in seg for root in ETHNIC_ROOTS):
        return []
    return [CONTENT_REASON_RU[PROBLEM_ETHNICITY]]


# Толстые контуры и векторный стиль (проверка при rules forbid_thick_vector: true)
_THICK_VECTOR_RX = re.compile(
    r"\b(?:bold (?:dark |black )?outlines?|thick (?:dark |black )?outlines?|heavy (?:dark |black )?outlines?|"
    r"vector|tube arms|thick white arms)\b",
    re.IGNORECASE,
)


# Разделитель фразы от пояснения после запятой (числа с запятой вроде 300,000 не режутся)
_LETTERING_COMMA_RE = re.compile(r"(?<!\d),(?!\d)")
# Слова описания шрифта между «in» и «lettering» в шаблонах style.md («in bold black uppercase lettering»)
_LETTERING_FONT_WORDS = (
    r"(?:bold|black|dark[- ]red|dark|red|gold|golden|muted|uppercase|lowercase|"
    r"capital|block|hand|handwritten|clean|crisp|white|thin|thick|heavy|medium|"
    r"simple|solid|ink|colou?red|style|print|printed|[a-z]+-[a-z]+)"
)
# Надпись в кадре: «lettering ... : <текст>», «reading <текст> in ... lettering» и «lettering ... reading <текст>»
_LETTERING_RX = re.compile(
    r"\blettering\b[^:.\n]{0,60}:\s*([^.\n;]+)"
    r"|\bread(?:s|ing)\s+([^.\n;]+?)\s+in\s+(?:" + _LETTERING_FONT_WORDS + r"\s+){0,10}lettering\b"
    r"|\blettering\b[^:.,;\n]{0,60}\bread(?:s|ing)\s+(?!:)([^.\n;]+)",
    re.IGNORECASE,
)
# служебные слова, на которых надпись не может закончиться
# Служебные слова, на которых надпись в кадре не может закончиться (ключи в нижнем регистре, апостроф прямой).
DANGLING_TEXT_WORDS = frozenset({
    "a", "an", "and", "by", "for", "it", "it's", "of", "so", "the", "to",
    "was", "were", "is", "are", "be", "am", "your", "my", "his", "her", "their", "its", "our", "like",
    "aren't", "isn't", "wasn't", "weren't", "don't", "doesn't", "didn't", "can't", "won't",
    "that", "this", "with", "from", "into", "about", "at", "on", "in", "as", "or", "if", "than", "then",
    "which", "who",
})
# Типографский апостроф и хвостовая пунктуация последнего слова надписи
_TYPO_APOSTROPHE = "\u2019"
_TRAILING_PUNCT_RX = re.compile(r"[.,!?;:\u2026\"\u201d\u00bb)]+$")


def _lettering_phrase(match) -> str:
    """Текст надписи из совпадения _LETTERING_RX (первая непустая группа альтернативы)."""
    raw = next((g for g in match.groups() if g is not None), "")
    return _LETTERING_COMMA_RE.split(raw, maxsplit=1)[0].strip()


def frame_has_lettering(prompt) -> bool:
    """Есть ли в промпте надпись в кадре (то же правило, что у проверки надписи): True при любом совпадении."""
    return _LETTERING_RX.search(prompt or "") is not None


def dangling_text_hits(text: str) -> list:
    """Надписи в кадре, оборванные на служебном слове (только при rules dangling_text_check: true)."""
    hits = []
    for m in _LETTERING_RX.finditer(text):
        phrase = _lettering_phrase(m).replace(_TYPO_APOSTROPHE, "'")
        words = phrase.split()
        if not words:
            continue
        last = _TRAILING_PUNCT_RX.sub("", words[-1]).lower()
        if last in DANGLING_TEXT_WORDS:
            hits.append(phrase)
    return hits


def _norm_lettering(text: str) -> str:
    """Нормализация для сравнения надписи с репликой: нижний регистр, прямой апостроф, одиночные пробелы."""
    return " ".join(text.replace(_TYPO_APOSTROPHE, "'").lower().split())


# Слова-отрицания перед фразой надписи (нижний регистр, прямой апостроф); токен слова — \w с апострофом
_NEGATION_WORDS = frozenset({
    "can't", "cannot", "don't", "not", "never", "no", "won't", "isn't", "aren't",
    "doesn't", "didn't", "without",
})
_WORD_TOKEN_RX = re.compile(r"[\w']+")


def segment_lettering_mismatch(prompt: str, segment_text: str) -> list:
    """Надписи в кадре, которых нет в реплике текущего сегмента (целая фраза как подстрока). Чистая функция."""
    segment = _norm_lettering(segment_text or "")
    out = []
    for m in _LETTERING_RX.finditer(prompt or ""):
        phrase = _lettering_phrase(m)
        if not phrase:
            continue
        norm = _norm_lettering(phrase)
        if norm not in segment:
            out.append(phrase)
            continue
        # Русский комментарий: фраза есть в реплике, но если перед её первым вхождением
        # в трёх словах стоит отрицание («can't start a fire»), надпись всё равно считается ошибочной
        idx = segment.find(norm)
        before = _WORD_TOKEN_RX.findall(segment[:idx])[-3:]
        if any(w in _NEGATION_WORDS for w in before):
            out.append(phrase)
    return out


def find_problems(prompt: str, media: str = "video", relaxed=frozenset(), max_characters: Optional[int] = None,
                  rules: Optional[RulesValues] = None, segment_text: Optional[str] = None,
                  textless: bool = False) -> list:
    """Список сообщений вида «код: «фрагмент»», по одному на нарушение. Пусто, если нарушений нет."""
    text = prompt or ""
    rules = rules or RulesValues()
    problems = []
    # разрешённые типы кадра снимают только свои слова и фразы; без allowed_shot_items — прежнее поведение
    released = released_terms(rules.allowed_shot_items)
    released_phrases = frozenset(t for t in released if t in FORBIDDEN_IMAGE_PHRASES_EN)
    released_words = released - released_phrases
    # слово внутри разрешённой фразы («paper» в «sheet of paper») снимается вместе с фразой
    released_spans = [m.span() for m in FORBIDDEN_IMAGE_PHRASE_RE.finditer(text)
                      if m.group(0).lower() in released_phrases]
    for m in FORBIDDEN_IMAGE_RE.finditer(text):
        word = m.group(0).lower()
        inside = any(start <= m.start() and m.end() <= end for start, end in released_spans)
        if inside or any(word.startswith(w) for w in released_words):
            continue
        problems.append(make_problem(PROBLEM_FORBIDDEN_SHOT_TYPE, m.group(0)))
    for m in FORBIDDEN_IMAGE_PHRASE_RE.finditer(text):
        if m.group(0).lower() not in released_phrases:
            problems.append(make_problem(PROBLEM_FORBIDDEN_COMBINATION, m.group(0)))
    for m in FORBIDDEN_FORMAT_PATTERNS_EN.finditer(text):
        problems.append(make_problem(PROBLEM_FORBIDDEN_FORMAT, m.group(0)))
    for m in LEFTOVER_MARKERS_RE.finditer(text):
        problems.append(make_problem(PROBLEM_LEFTOVER_MARKER, m.group(0)))
    # толстые контуры и векторный стиль: только при forbid_thick_vector (ltx25, zimage_turbo — не включено)
    if rules.forbid_thick_vector:
        problems += [make_problem(PROBLEM_THICK_VECTOR, m.group(0)) for m in _THICK_VECTOR_RX.finditer(text)]
    # незаконченный текст в кадре: только при dangling_text_check: true (ключ не задан — проверки нет)
    if rules.dangling_text_check:
        problems += [make_problem(PROBLEM_DANGLING_TEXT, hit) for hit in dangling_text_hits(text)]
    # надпись только из реплики текущего сегмента: без segment_text проверки нет
    if rules.dangling_text_check and segment_text is not None:
        problems += [make_problem(PROBLEM_LETTERING_NOT_IN_SEGMENT, hit)
                     for hit in segment_lettering_mismatch(text, segment_text)]
    # кадр без текста (повтор 4 и далее): надпись в кадре остаётся — нарушение; только при textless=True и dangling_text_check
    if textless and rules.dangling_text_check and frame_has_lettering(text):
        problems += [make_problem(PROBLEM_TEXT_IN_TEXTLESS_FRAME, m.group(0)) for m in _LETTERING_RX.finditer(text)]
    for code, fragment in content_hits(text, media, relaxed, max_characters, rules):
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
        # «frame height» / «frame width» задают размер объекта, а не движение рамки камеры
        if word == "frame" and i + 1 < len(words) and words[i + 1][0] in ("height", "width"):
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