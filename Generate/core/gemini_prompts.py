"""Gemini: SRT -> промпты для видеомодели (этап 4а: данные, запрос, ответ, проверка).

Порт идей из первой части (склейка сегментов в предложения, Before/After по целым
предложениям, батчи, structured output, проверка набора номеров), а не копия.
Здесь НЕТ: лимитера RPM/TPM, счётчика RPD, цепочки моделей, файла prompts.json (этап 4б).

Модель приходит параметром. Транспорт заменяемый (GeminiTransport): в тестах заглушка,
в бою RealTransport на httpx. Ключ только из окружения, только в заголовке (SPEC 0.1).
Ничего не печатается; логи на русском, тексты промптов в логи не попадают.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

try:  # пакетный и «плоский» импорт (как в остальных модулях core)
    from .srt_parser import Segment
    from .model_constants import PRESET_NONE, PROFILE_NAME_PATTERN
    from .frame_rules import rephrase_note  # правило для запрещённых типов кадра (шаг 4)
    from .prompt_check import (PROBLEM_CAMERA_MOTION, find_camera_motion, find_problems,  # проверка промптов (шаг 6)
                           make_problem, problem_ru, rework_remark,
                           CONTENT_HINT_BY_REASON, content_reasons, ethnicity_reasons)
except ImportError:  # pragma: no cover
    from srt_parser import Segment  # type: ignore
    from model_constants import PRESET_NONE, PROFILE_NAME_PATTERN  # type: ignore
    from frame_rules import rephrase_note  # type: ignore
    from prompt_check import (PROBLEM_CAMERA_MOTION, find_camera_motion, find_problems,  # type: ignore
                          make_problem, problem_ru, rework_remark,
                          CONTENT_HINT_BY_REASON, content_reasons, ethnicity_reasons)

try:  # отметка «ключевое событие» для консоли GitHub (см. log_setup.ConsoleFilter)
    from .log_setup import KEY_EVENT_EXTRA
except ImportError:
    try:
        from log_setup import KEY_EVENT_EXTRA  # type: ignore
    except ImportError:  # log_setup тянет vast_client, а тот — httpx; без httpx модуль грузится, значение то же
        KEY_EVENT_EXTRA = {"console_key": True}

log = logging.getLogger("Generate.gemini_prompts")

API_KEY_HEADER = "x-goog-api-key"
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Размеры окон контекста (слова), как в первой части.
CONTEXT_BEFORE_WORDS = 40
CONTEXT_AFTER_WORDS = 25
SHORT_SEGMENT_MAX_WORDS = 3

# ---------------------------------------------------------------------------
# Исключения
# ---------------------------------------------------------------------------

class GeminiError(Exception):
    exit_code = 1


class GeminiInputError(GeminiError):
    """Неверный вход (номер вне SRT, пустой список, плохая настройка, нет ключа). Код выхода 2."""
    exit_code = 2


class GeminiRuntimeError(GeminiError):
    """Сбой при обращении к Gemini (сеть, HTTP-статус, неразборчивый ответ). Код выхода 3.

    Поля безопасные: только числа/короткие коды, без заголовков, тела и запроса.
      status       HTTP-статус или None (сетевой сбой);
      code         код ошибки Google (например RESOURCE_EXHAUSTED) или "";
      daily_quota  True, если в ответе признак суточной квоты (для 4б: смена модели);
      retry_after  рекомендованная пауза в секундах или None.
    """
    exit_code = 3

    def __init__(self, message: str, *, status: Optional[int] = None, code: str = "",
                 daily_quota: bool = False, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.daily_quota = daily_quota
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Конфиг и результаты
# ---------------------------------------------------------------------------

@dataclass
class PromptConfig:
    style_file: str                  # путь к файлу стиля models/<профиль>/prompts/style.md (из профиля)
    media: str                       # тип промпта: "video" или "image"; без значения по умолчанию
    max_prompt_chars: int = 1500
    batch_size: int = 20
    response_format: Any = field(default_factory=lambda: [{"segment_index": int, "prompt": str}])
    # пресет профиля: Generate/model_profiles/presets/prompts/<preset>.md; «none» = без пресета
    preset: str = PRESET_NONE


@dataclass(frozen=True)
class BatchRequest:
    nums: tuple          # номера сегментов батча (по возрастанию)
    system: str
    user: str
    schema: dict


@dataclass
class PromptResult:
    num: int
    prompt: str
    needs_review: bool = False
    review_reason: str = ""
    extra: dict = field(default_factory=dict)   # дополнительные поля из response_format профиля
    # очередь исправлений (шаг 6): нарушения, которые ещё не исправлены (строки, как у find_problems)
    rework_pending: list = field(default_factory=list)
    # число кругов исправления; None: клип очередь не проходил (в prompts.json поля не пишутся)
    rework_round: Optional[int] = None
    # сырой текст клипа в очереди исправлений (без стиля): по нему проверка и исправление; пусто вне очереди
    prompt_raw: str = ""


# Очередь исправлений: кругов на клип и размер пакета исправления (одна волна, один запрос)
REPAIR_ROUNDS = 5
FIX_BATCH_SIZE = 25


@dataclass(frozen=True)
class Problem:
    num: Optional[int]       # None: проблема всего ответа
    reason: str


class GeminiTransport(Protocol):
    def generate_json(self, model: str, system: str, user: str, schema: dict) -> str:
        """Один запрос. Возвращает сырой JSON-текст ответа модели. Ошибки: GeminiRuntimeError."""
        ...


# ---------------------------------------------------------------------------
# Склейка сегментов в предложения (порт идей из первой части)
# ---------------------------------------------------------------------------

SENTENCE_END_CHARS = ".?!\u2026"
SENTENCE_CLOSERS = "\"'\u00bb\u201d\u2019)]"
SENTENCE_OPENERS = "\"'\u00ab\u201c\u2018\u201e(["
NO_BREAK_ABBREVIATIONS = frozenset({"Mr.", "Mrs.", "Ms.", "Dr.", "St.", "No.", "Jr.", "Sr.", "vs."})
LOWERCASE_CONTINUES_ABBREVIATIONS = frozenset({"etc."})


@dataclass
class Sentence:
    number: int                       # порядковый номер предложения, с 1
    seg_nums: list                    # Segment.num входящих сегментов
    text: str
    spans: dict                       # Segment.num -> (start, end) внутри text

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _strip_closers(text: str) -> str:
    return text.rstrip(SENTENCE_CLOSERS + " ")


def _ends_sentence(text: str, next_text: Optional[str]) -> bool:
    """Заканчивается ли предложение на этом сегменте (text непустой, next_text — ближайший
    следующий непустой сегмент или None)."""
    core = _strip_closers(text)
    if not core or core[-1] not in SENTENCE_END_CHARS:
        return False
    nxt = next_text.lstrip(SENTENCE_OPENERS + " ") if next_text else ""

    if core.endswith("\u2026") or core.endswith("..."):
        return next_text is None or nxt[:1].isupper()
    if core[-1] in "?!":
        return True

    token = core.split()[-1].lstrip(SENTENCE_OPENERS)
    if token == "No.":
        return not re.match(r"\d", nxt)
    if token in NO_BREAK_ABBREVIATIONS:
        return False
    if token in LOWERCASE_CONTINUES_ABBREVIATIONS:
        return not nxt[:1].islower()
    if re.search(r"\d\.$", token) and nxt[:1].isdigit():
        return False
    return True


def merge_segments_into_sentences(segments: list) -> tuple:
    """Склеивает ВЕСЬ список сегментов в предложения. Чистая функция.

    Возвращает (sentences, by_segment): список предложений и словарь Segment.num -> Sentence.
    Пустой сегмент не разрывает открытое предложение; вне предложения он образует отдельное
    предложение с пустым текстом."""
    texts = [" ".join((s.text or "").split()) for s in segments]
    next_nonempty: list = [None] * len(segments)
    upcoming: Optional[str] = None
    for i in range(len(segments) - 1, -1, -1):
        next_nonempty[i] = upcoming
        if texts[i]:
            upcoming = texts[i]

    sentences: list = []
    by_segment: dict = {}
    current: Optional[Sentence] = None

    for i, seg in enumerate(segments):
        t = texts[i]
        if not t:
            if current is None:
                lone = Sentence(len(sentences) + 1, [seg.num], "", {seg.num: (0, 0)})
                sentences.append(lone)
                by_segment[seg.num] = lone
            else:
                pos = len(current.text)
                current.seg_nums.append(seg.num)
                current.spans[seg.num] = (pos, pos)
                by_segment[seg.num] = current
            continue

        if current is None:
            current = Sentence(len(sentences) + 1, [], "", {})
            sentences.append(current)
        if current.text:
            current.text += " "
        start = len(current.text)
        current.text += t
        current.seg_nums.append(seg.num)
        current.spans[seg.num] = (start, len(current.text))
        by_segment[seg.num] = current

        if _ends_sentence(t, next_nonempty[i]):
            current = None

    return sentences, by_segment


def _collect_context_sentences(sentences: list, start: int, step: int, window_words: int) -> list:
    """Целые предложения от ближайшего наружу, пока сумма слов не достигнет window_words.
    Предложение, перешагнувшее порог, берётся целиком. Пустые пропускаются."""
    out: list = []
    total = 0
    i = start
    while 0 <= i < len(sentences):
        sen = sentences[i]
        i += step
        if not sen.text:
            continue
        out.append(sen)
        total += sen.word_count
        if total >= window_words:
            break
    return out


def _format_segment_block(seg: Segment, sentences: list, by_segment: dict) -> str:
    """Блок одного сегмента для пользовательского запроса. Номер идёт только как служебное
    поле segment_index (его модель вернёт в ответе), в смысловой текст он не входит."""
    sen = by_segment[seg.num]
    lines = [f"segment_index: {seg.num}"]
    text = " ".join((seg.text or "").split())

    if text:
        a, b = sen.spans[seg.num]
        marked = f"{sen.text[:a]}>>{sen.text[a:b]}<<{sen.text[b:]}"
        lines.append(f"Text (what this segment says): {text}")
        lines.append(f"Sentence: {marked}")
        if len(text.split()) <= SHORT_SEGMENT_MAX_WORDS:
            lines.append("Note: this is a very short fragment; base the visual on the whole "
                         "sentence, not on these words alone.")
    else:
        lines.append("Text (what this segment says): (silence / no text)")
        if sen.text:
            lines.append(f"Sentence: {sen.text}")

    pos = sen.number - 1
    before = _collect_context_sentences(sentences, pos - 1, -1, CONTEXT_BEFORE_WORDS)
    after = _collect_context_sentences(sentences, pos + 1, +1, CONTEXT_AFTER_WORDS)
    if before:
        lines.append("Before (context only): " + " ".join(x.text for x in reversed(before)))
    if after:
        lines.append("After (context only): " + " ".join(x.text for x in after))
    # Правило перефразирования добавляется к блоку сегмента, только если нужно (шаг 4).
    note = rephrase_note(text)
    if note:
        lines.append(note)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Файл стиля и системная инструкция
# ---------------------------------------------------------------------------

_USER_ONLY_RE = re.compile(r"<!--\s*USER_ONLY_START\s*-->.*?<!--\s*USER_ONLY_END\s*-->",
                           re.DOTALL)
_USER_ONLY_MARK_RE = re.compile(r"<!--\s*USER_ONLY_(START|END)\s*-->")


def strip_user_only(text: str) -> str:
    """Вырезает блоки USER_ONLY_START..USER_ONLY_END. Непарный маркер — ошибка настройки:
    иначе пользовательская сводка могла бы уйти в Gemini."""
    out = _USER_ONLY_RE.sub("", text)
    if _USER_ONLY_MARK_RE.search(out):
        raise GeminiInputError("В файле стиля непарные маркеры USER_ONLY_START / USER_ONLY_END.")
    return out.strip()


_TYPE_NAMES = {"int": "integer", "integer": "integer", "str": "string", "string": "string",
               "float": "number", "number": "number", "bool": "boolean", "boolean": "boolean"}
_PY_TYPES = {int: "integer", str: "string", float: "number", bool: "boolean"}


def _schema_type(t: Any) -> str:
    if isinstance(t, type) and t in _PY_TYPES:
        return _PY_TYPES[t]
    if isinstance(t, str) and t.strip().lower() in _TYPE_NAMES:
        return _TYPE_NAMES[t.strip().lower()]
    raise GeminiInputError(f"Неподдерживаемый тип в response_format: {t!r}.")


def response_schema(response_format: Any) -> dict:
    """JSON Schema массива объектов из примера формата: [{"segment_index": int, "prompt": str}]."""
    if (not isinstance(response_format, list) or len(response_format) != 1
            or not isinstance(response_format[0], dict)):
        raise GeminiInputError("response_format должен быть списком из одного объекта-примера.")
    item = response_format[0]
    if "segment_index" not in item or "prompt" not in item:
        raise GeminiInputError("В response_format обязательны поля segment_index и prompt.")
    props = {name: {"type": _schema_type(t)} for name, t in item.items()}
    return {"type": "array",
            "items": {"type": "object", "properties": props, "required": list(props)}}


# Пресеты: файлы presets/prompts/<имя>.md рядом с профилями моделей. README в этой папке — описание,
# не пресет (в списке выбора GitHub его нет), поэтому имя README запрещено.
PRESETS_DIR = Path(__file__).resolve().parent.parent / "model_profiles" / "presets" / "prompts"
_PRESET_RESERVED = {"README"}


def read_preset(preset: str) -> str:
    """Текст пресета (USER_ONLY вырезан). Общая проверка для cli (до аренды) и для системной инструкции.
    Ошибка — GeminiInputError с текстом на русском: имя недопустимо, файла нет или он пуст."""
    if not re.fullmatch(PROFILE_NAME_PATTERN, preset or "") or preset in _PRESET_RESERVED:
        raise GeminiInputError("Недопустимое имя пресета: допустимы латинские буквы, цифры, «_» и «-», "
                               "кроме README.")
    path = PRESETS_DIR / f"{preset}.md"
    if not path.is_file():
        raise GeminiInputError(f"Пресет «{preset}» не найден: нет файла presets/prompts/{preset}.md.")
    try:
        text = strip_user_only(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        raise GeminiInputError(f"Пресет «{preset}» не удалось прочитать: presets/prompts/{preset}.md.") from None
    if not text:
        raise GeminiInputError(f"Пресет «{preset}» пуст: в файле presets/prompts/{preset}.md нет текста.")
    return text


def attach_style(prompt: str, style_text: str) -> str:
    """Дописывает текст стиля в конец готового промпта через «. ». Пустой стиль — промпт без изменений."""
    if not style_text:
        return prompt
    return prompt.rstrip().rstrip(".") + ". " + style_text


def _content_rules(media: str) -> str:
    """Жёсткие правила содержания кадра (решения владельца после разбора прогона).
    Те же правила проверяет prompt_check.py: нарушение уходит на повтор, а не в GPU."""
    people_limit = "one person" if media == "image" else "at most two people"
    rules = (
        "- Show only the people, animals and objects that the segment text names or directly implies. "
        "Add no extra characters, animals, vehicles or props.\n"
        f"- Show {people_limit}, each named once and described distinctly (age, clothing colour, hair). "
        "Never show a crowd, a group, a row, a line or several people.\n"
        "- Never show children, babies, boys, girls, teenagers or young people. If the segment mentions "
        "them, show adults or no people, and keep the meaning through the setting.\n"
        "- Every person wears named clothing. Never show anyone naked, shirtless or undressed.\n"
        "- Show at most one or two horses, vehicles or animals, each described distinctly. No herds or rows.\n"
        "- Show only large, clearly visible objects. No small objects and no hand actions with small things "
        "(needles, thread, abacus, beads, cloth, bundles, cups, dung cakes; no stacking, stitching, tucking "
        "or threading).\n"
        "- No speech of any kind: no speaking, shouting, laughing, crying, singing, conversation, dialogue, "
        "gestures for talking or moving lips. Show emotion only through posture.\n"
        "- Keep hands and fingers away from fire, flames, candles and lanterns.\n"
        "- No battles, charges, gallops in formation, gunfire, explosions, smoke puffs or dust clouds. "
        "Show calm, static or slowly moving scenes.\n"
        "- No weapons in the frame: no firearms, sabres, swords, spears or knives.\n"
        "- When the text says something is absent (for example no saddle), describe the visible state "
        "instead (for example a bare horse back).\n"
        "- Derive the place, period and people's appearance from the segment text and its setting. Do not add "
        "a nationality, ethnicity or skin tone the text does not mention.\n"
    )
    if media == "image":
        rules += (
            "- Never show close-ups, extreme close-ups or portraits. Show people from a medium or wide "
            "distance, with faces small in the frame.\n"
        )
    return rules


def build_system_instruction(cfg: PromptConfig) -> str:
    style = strip_user_only(Path(cfg.style_file).read_text(encoding="utf-8"))
    if cfg.media == "video":
        kind = "text-to-video prompt"
        camera_rules = (
            "- Describe only the scene and its motion. Never name a medium, rendering method, film look, "
            "or camera or lens equipment; camera movement is the only camera wording allowed.\n"
            "- When the text contrasts two things (\"not X, but Y\", \"not X, rather Y\"), show only Y, as a visible action or scene. Never describe X and never use a negation to refer to it.\n"
            "- No paper, papers, letters, passports, forms or documents of any kind in the frame, including stacks, sheets and folded paper. Show the place or the activity instead, without readable text.\n"
            "- No aspect ratios, frame sizes, widescreen or framing terms, and no blank sheets, sheets of paper or sheets of parchment. Describe the visible scene only, without any format or sheet of writing material.\n"
            "- Exactly one camera move. Finish the camera sentence with how the frame looks when "
            "the move ends (for a static camera: what the frame holds on).\n"
        )
    elif cfg.media == "image":
        kind = "text-to-image prompt, one still frame"
        camera_rules = (
            "- Describe only the scene and its motion. Never name a medium, rendering method, film look, "
            "or camera or lens equipment; camera or lens equipment is forbidden.\n"
            "- When the text contrasts two things (\"not X, but Y\", \"not X, rather Y\"), show only Y, as a visible action or scene. Never describe X and never use a negation to refer to it.\n"
            "- No paper, papers, letters, passports, forms or documents of any kind in the frame, including stacks, sheets and folded paper. Show the place or the activity instead, without readable text.\n"
            "- No aspect ratios, frame sizes, widescreen or framing terms, and no blank sheets, sheets of paper or sheets of parchment. Describe the visible scene only, without any format or sheet of writing material.\n"
            "- Never describe camera movement: no pans, zooms, tracks, dolly, tilts, pushes, pulls, "
            "or any motion of the camera or the frame. Describe the single still frame as a finished state.\n"
        )
    else:
        raise GeminiInputError(f"media должен быть «video» или «image», получено: {cfg.media!r}.")
    parts = [
        "# Task\n"
        "You receive subtitle segments of a narrated video. For every segment write ONE "
        + kind + ", in English, that visually illustrates what the segment says, "
        "following the style guide below. Use the Sentence, Before and After lines only to "
        "understand the meaning; the >>marked<< part is the passage to illustrate. The source subtitle text may be in any language. Always write the output in English.\n\n"
        "# Output contract\n"
        "Return a JSON array with exactly one object per requested segment: "
        "{\"segment_index\": <the integer given for that segment>, \"prompt\": <string>}. "
        "No other segments, no duplicates, no markdown, no code fences. "
        "Never write the segment number, the word \"segment\" or any numbering into the prompt text. "
        f"Each prompt must be at most {cfg.max_prompt_chars} characters.\n\n"
        "# Hard rules for every prompt\n"
        "- Show moods and ideas only through visible physical cues. Never use mood or emotion "
        "words such as thoughtful, pensive, solemn, serene, tense, mysterious, majestic, epic, "
        "cinematic, sad, lonely, or phrases like \"quiet stillness\".\n"
        "- No negations (no, not, without, never, nothing), no quotation marks, "
        "no keyword lists (4k, masterpiece, trending).\n"
        + camera_rules +
        "- Give each person one consistent outfit; do not mix conflicting garments such as a "
        "suit and an overcoat unless the layering is stated clearly.\n"
        + _content_rules(cfg.media) +
        "- Before returning, check each prompt against the self-check checklist at the end of "
        "the style guide and rewrite any prompt that fails it.",
        "# Style guide\n" + style,
    ]
    # пресет в системную инструкцию не входит: здесь только проверка имени и файла, текст дописывает generate_prompts
    if cfg.preset != PRESET_NONE:
        read_preset(cfg.preset)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Построение запросов
# ---------------------------------------------------------------------------

def _check_config(cfg: PromptConfig) -> None:
    if cfg.media not in ("video", "image"):
        raise GeminiInputError(f"media должен быть «video» или «image», получено: {cfg.media!r}.")
    if cfg.batch_size < 1:
        raise GeminiInputError("batch_size должен быть не меньше 1.")
    if cfg.max_prompt_chars < 1:
        raise GeminiInputError("max_prompt_chars должен быть не меньше 1.")


class _Builder:
    """Контекст считается один раз по ПОЛНОМУ списку сегментов; затем выбираются нужные номера."""

    def __init__(self, segments: list, wanted_nums: list, cfg: PromptConfig):
        _check_config(cfg)
        if not segments:
            raise GeminiInputError("Список сегментов SRT пуст.")
        wanted = sorted(set(wanted_nums))
        if not wanted:
            raise GeminiInputError("Список нужных номеров пуст: нечего генерировать.")
        self.by_num = {s.num: s for s in segments}
        self.texts = {s.num: s.text for s in segments}  # для проверки внешности по тексту сегмента
        absent = [n for n in wanted if n not in self.by_num]
        if absent:
            shown = ", ".join(str(n) for n in absent)  # полный список номеров, без обрезки
            raise GeminiInputError(f"Номера отсутствуют в SRT: {shown}.")
        self.wanted = wanted
        self.cfg = cfg
        self.sentences, self.by_segment = merge_segments_into_sentences(list(segments))
        self.system = build_system_instruction(cfg)
        self.schema = response_schema(cfg.response_format)

    def batches(self, nums: list) -> list:
        nums = sorted(set(nums))
        size = self.cfg.batch_size
        out = []
        for i in range(0, len(nums), size):
            chunk = nums[i:i + size]
            blocks = [_format_segment_block(self.by_num[n], self.sentences, self.by_segment)
                      for n in chunk]
            user = ("Write prompts for the following segments. Return only the JSON array.\n\n"
                    + "\n\n".join(blocks))
            out.append(BatchRequest(tuple(chunk), self.system, user, self.schema))
        return out


def build_requests(segments: list, wanted_nums: list, cfg: PromptConfig) -> list:
    """Список батчей (BatchRequest) для нужных номеров. Все проверки входа здесь, до транспорта."""
    b = _Builder(segments, wanted_nums, cfg)
    return b.batches(b.wanted)


# ---------------------------------------------------------------------------
# Разбор и проверка ответа
# ---------------------------------------------------------------------------

_GARBAGE_PATTERNS = [
    (re.compile(r"```"), "markdown-ограждение"),
    # служебные формы: segment_index, «Segment 4», «segment #4», «Segment:» в начале.
    # Обычное слово («a segment of the wall») не мусор.
    (re.compile(r"\bsegment_index\b|\bsegment\s*#?\d+|^\s*segment\s*[:#\-]", re.I),
     "служебный маркер «segment»"),
    (re.compile(r"^\s*#+\s"), "markdown-заголовок"),
    # номер-метка в начале: «[3]», «12: », «3. », «4) »; «35-year-old», «2.5 m», «12:30» не мусор.
    (re.compile(r"^\s*(?:\[\d+\]|#?\d+\s*[:.)\-\u2013\u2014](?:\s|$))"), "номер в начале"),
    (re.compile(r"[\"']prompt[\"']\s*:"), "обломок JSON"),
]


def _garbage_reason(prompt: str) -> str:
    for rx, name in _GARBAGE_PATTERNS:
        if rx.search(prompt):
            return f"служебный мусор: {name}"
    return ""


# Проверка стиля по правилам models/<профиль>/prompts/style.md: отрицания, кавычки, keyword spam, эмоциональные метки.
# Цифры намеренно не запрещены ("35-year-old", "3 candles" допустимы).
# Формат: (регулярка, причина для needs_review, подсказка для Gemini при повторном запросе).
_STYLE_RULES = [
    (re.compile(r"\b(?:no|not|without|never|nothing|nobody|none|"
                r"(?:don|doesn|isn|aren|can|won)['\u2019]?t|cannot)\b", re.I),
     "стиль: отрицание",
     "Remove negations (no, not, without, never, nothing); describe only what is present."),
    (re.compile(r"[\"\u201c\u201d\u00ab\u00bb]"),
     "стиль: кавычки",
     "Remove quotation marks and any spoken or written words."),
    (re.compile(r"\b(?:4k|8k|uhd|masterpiece|trending|ultra[- ]detailed|highly detailed|"
                r"photorealistic|hyperrealistic|render\s+(?:engine|farm)|rendered|octane|unreal[ -]engine)\b", re.I),
     "стиль: keyword spam",
     "Remove photo and render terms such as 4k, 8k, UHD, photorealistic, masterpiece, trending, "
     "render, octane, unreal engine."),
    (re.compile(r"\b(?:thoughtful|pensive|contemplative|melancholy|melancholic|wistful|"
                r"nostalgic|solemn|somber|sombre|serene|peaceful|tense|anxious|mysterious|"
                r"eerie|haunting|majestic|epic|cinematic|sad|lonely|hopeless|hopeful|"
                r"confused|happy|angry|afraid|scared|joyful|stillness)\b", re.I),
     "стиль: эмоциональная метка",
     "Replace mood or emotion words (thoughtful, solemn, serene, tense, cinematic, stillness) "
     "with visible physical cues or concrete objects."),
    (re.compile(r"\b(?:film\s+(?:still|grain)|anamorphic|bokeh|depth\s+of\s+field|35\s?mm|"
                r"lens|shot\s+on\s+(?:a\s+|an\s+)?(?:camera|film)|aperture|focal\s+length)\b", re.I),
     "стиль: кино и камера",
     "Remove film and lens terms such as film grain, anamorphic, bokeh, depth of field, 35mm, lens, "
     "aperture, focal length, shot on. Describe only the scene and its one camera move."),
    (re.compile(r"\b(?:illustrations?|cartoons?|anime|watercolou?r|oil\s+painting|sketch(?:es)?|"
                r"stickmen?|pixel\s+art)\b", re.I),
     "стиль: художественная техника",
     "Remove art-medium words such as illustration, cartoon, anime, watercolor, oil painting, sketch, "
     "stickman, pixel art. Describe the real scene instead."),
]
_STYLE_HINTS = {reason: hint for _, reason, hint in _STYLE_RULES}


_STYLE_HINTS.update(CONTENT_HINT_BY_REASON)  # подсказки для повтора по содержанию кадра


def _style_reasons(prompt: str, media: str = "video") -> list:
    """Стилевые причины и причины содержания кадра (дети, толпа, мелкие предметы, речь и т. д.)."""
    style = [reason for rx, reason, _ in _STYLE_RULES if rx.search(prompt)]
    return style + content_reasons(prompt, media)


def _retry_note(nums, problems: list) -> str:
    """Текст для повторного запроса: что именно исправить. Пусто, если стилевых нарушений нет."""
    lines = []
    for n in nums:
        hints = [_STYLE_HINTS[p.reason] for p in problems
                 if p.num == n and p.reason in _STYLE_HINTS]
        if hints:
            lines.append(f"- segment_index: {n}: " + " ".join(dict.fromkeys(hints)))
    if not lines:
        return ""
    return ("\n\nThe previous attempt broke style rules. Fix these issues and keep the rest "
            "of the idea:\n" + "\n".join(lines))


def parse_and_validate(raw: str, expected_nums, cfg: PromptConfig, texts: Optional[dict] = None) -> tuple:
    """Разбирает ответ Gemini. Возвращает (results, problems).

    results: dict num -> PromptResult для каждого ожидаемого номера, у которого есть непустая
      строка prompt. Если к нему есть претензия, needs_review=True и указана причина.
    problems: list[Problem]: нарушения (номер или None для проблемы всего ответа).
    Номера без записи в results не получили пригодного текста. Лишние номера не попадают в results
    (фиксируются как Problem, но повторный запрос не вызывают: нужные данные целы).
    texts: номер -> текст сегмента; нужен для проверки внешности людей (не из текста ли она)."""
    expected = set(expected_nums)
    results: dict = {}
    problems: list = []

    def fail_all(reason: str):
        return {}, [Problem(None, reason)] + [Problem(n, reason) for n in sorted(expected)]

    if not isinstance(raw, str) or not raw.strip():
        return fail_all("пустой ответ")
    try:
        data = json.loads(raw)
    except ValueError:
        return fail_all("ответ не является JSON")
    if not isinstance(data, list):
        return fail_all("ожидался JSON-массив")

    seen: dict = {}
    for item in data:
        if not isinstance(item, dict):
            problems.append(Problem(None, "элемент ответа не объект"))
            continue
        idx = item.get("segment_index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            problems.append(Problem(None, "у элемента нет целого segment_index"))
            continue
        if idx not in expected:
            problems.append(Problem(idx, "лишний номер в ответе"))
            continue
        if idx in seen:
            problems.append(Problem(idx, "номер повторяется в ответе"))
            seen[idx]["dup"] = True
            continue
        seen[idx] = {"item": item, "dup": False}

    for n in sorted(expected):
        entry = seen.get(n)
        if entry is None:
            problems.append(Problem(n, "номер отсутствует в ответе"))
            continue
        item = entry["item"]
        raw_prompt = item.get("prompt")
        reasons = ["номер повторяется в ответе"] if entry["dup"] else []
        if not isinstance(raw_prompt, str) or not raw_prompt.strip():
            problems.append(Problem(n, "пустой prompt"))
            continue
        prompt = " ".join(raw_prompt.split())   # один абзац: переводы строк в пробелы
        if len(prompt) > cfg.max_prompt_chars:
            reasons.append(f"длина {len(prompt)} больше лимита {cfg.max_prompt_chars}")
        g = _garbage_reason(prompt)
        if g:
            reasons.append(g)
        reasons.extend(_style_reasons(prompt, cfg.media))
        if texts is not None and n in texts:
            reasons.extend(ethnicity_reasons(prompt, texts[n]))
        extra = {k: v for k, v in item.items() if k not in ("segment_index", "prompt")}
        res = PromptResult(n, prompt, bool(reasons), "; ".join(reasons), extra)
        results[n] = res
        for r in reasons:
            if r != "номер повторяется в ответе":
                problems.append(Problem(n, r))
    return results, problems


# ---------------------------------------------------------------------------
# Прогон батчей на заданном транспорте (без лимитера и цепочки: это 4б)
# ---------------------------------------------------------------------------

def _is_clean(results: dict, num: int, problems: list) -> bool:
    r = results.get(num)
    return r is not None and not r.needs_review and not any(p.num == num for p in problems)


def _find_all_problems(text: str, cfg: PromptConfig) -> list:
    """Нарушения промпта: запрещённые типы кадра, маркеры и содержание кадра; для фото ещё движение камеры."""
    problems = find_problems(text, cfg.media)
    if cfg.media == "image":
        problems += [make_problem(PROBLEM_CAMERA_MOTION, m) for m in find_camera_motion(text)]
    return list(dict.fromkeys(problems))


def _num_ranges(nums: list) -> str:
    """Номера списком диапазонов: «3-5, 9». Пусто, если номеров нет."""
    spans: list = []
    for n in sorted(set(nums)):
        if spans and n == spans[-1][1] + 1:
            spans[-1][1] = n
        else:
            spans.append([n, n])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def _wave_summary(job) -> None:
    """Итог волны перед выходом с ошибкой: клипы с нарушениями, не исправленные, и клипы, оставшиеся в очереди.

    Вызывается перед каждым raise в generate_prompts, после abandon_queue. Потоки волны уже завершены,
    поэтому out и очередь читаются без замков; модель не вызывается.
    """
    unfixed = [c.num for c in job.out.values() if c.needs_review and c.rework_pending]
    log.info("[%s] Итог волны до остановки: не исправлено клипов %d, номера: %s; в очереди исправлений клипов %d.",
             job.wave, len(unfixed), _num_ranges(unfixed) or "нет", len(job.rework_queue))


def _run_batch(transport: GeminiTransport, model: str, builder: _Builder, batch: BatchRequest,
               tag: str = "") -> dict:
    """Запрос батча + один retry для номеров с нарушениями. Возвращает num -> PromptResult.

    tag: префикс строк журнала батча (имя волны в пуле, например «[видео] »); без пула пустой."""
    expected = list(batch.nums)
    raw = transport.generate_json(model, batch.system, batch.user, batch.schema)
    results, problems = parse_and_validate(raw, expected, builder.cfg, builder.texts)
    final: dict = {n: results[n] for n in expected if _is_clean(results, n, problems)}
    bad = [n for n in expected if n not in final]
    extras = [p for p in problems if p.num is not None and p.num not in set(expected)]
    if extras:
        log.warning("%sGemini вернул лишние номера: %d шт., они проигнорированы.", tag, len(extras))

    if bad:
        log.warning("%sБатч %s..%s: нарушения в %d из %d записей, повторный запрос.",
                    tag, expected[0], expected[-1], len(bad), len(expected))
        retry = builder.batches(bad)  # bad <= batch_size: ровно один батч
        for rb in retry:
            note = _retry_note(rb.nums, problems)
            if note:
                rb = replace(rb, user=rb.user + note)
            raw2 = transport.generate_json(model, rb.system, rb.user, rb.schema)
            res2, prob2 = parse_and_validate(raw2, list(rb.nums), builder.cfg, builder.texts)
            for n in rb.nums:
                if _is_clean(res2, n, prob2):
                    final[n] = res2[n]
                    continue
                # остаток: лучшая из имеющихся версий, помечается needs_review
                best = res2.get(n) or results.get(n)
                reason = "; ".join(p.reason for p in prob2 if p.num == n) or \
                         "; ".join(p.reason for p in problems if p.num == n) or "нарушение формата"
                if best is None:
                    best = PromptResult(n, "")
                final[n] = PromptResult(n, best.prompt, True, reason, best.extra)
    # шаг 6 без запроса: запрещённые типы и маркеры помечаются очередью исправлений (rework_pending);
    # текст пока без стиля, исправление идёт пакетами после главных батчей (_WaveJob.process_fix)
    for n in list(final):
        r = final[n]
        if r.prompt.strip():
            found = _find_all_problems(r.prompt, builder.cfg)
            if found:
                r.rework_pending = found
                r.rework_round = 0
    return final


class _LoggedTransport:
    """Обёртка только для журнала: при неудачном запросе пишет номер батча, номер попытки в батче
    и класс ошибки. Ответ и исключение пробрасываются без изменений; повторов обёртка не делает.
    В журнал не попадают тело запроса, текст промпта и ключ API."""

    def __init__(self, inner: "GeminiTransport", tag: str = ""):
        self._inner = inner
        self._batch_no = 0
        self._attempt = 0
        self._tag = tag  # префикс строк журнала: имя волны в пуле; без пула пустой
        self._unit = "Батч"  # слово перед номером в строке журнала: «Батч» или «Фикс»

    def start_batch(self, batch_no: int, unit: str = "Батч") -> None:
        # номер попытки считается заново для каждого батча (или пакета исправлений)
        self._batch_no = batch_no
        self._unit = unit
        self._attempt = 0

    def generate_json(self, model: str, system: str, user: str, schema: dict) -> str:
        self._attempt += 1
        try:
            return self._inner.generate_json(model, system, user, schema)
        except Exception as exc:
            # безопасное сообщение GeminiRuntimeError не содержит тела ответа; прочим классам — только имя класса
            detail = f" ({exc})" if isinstance(exc, GeminiRuntimeError) else ""
            log.warning("%s%s %d: попытка %d не удалась, класс ошибки %s%s.",
                        self._tag, self._unit, self._batch_no, self._attempt, type(exc).__name__, detail)
            raise


# Максимум параллельных батчей в generate_prompts (не больше числа ключей Gemini)
MAX_PARALLEL_BATCHES = 10


def _provider_key_count(provider: str) -> int:
    """Число ключей провайдера в окружении: ключи провайдера openrouter из окружения; для остальных — ключи Gemini из окружения, как раньше.

    Ключи нигде не выводятся. Если ключей нет, load_* бросает исключение: вызывающий код считает это нулём."""
    if provider == "openrouter":
        try:
            from . import openrouter_transport as ort
        except ImportError:  # плоский импорт, как в остальных модулях core
            import openrouter_transport as ort  # type: ignore
        return len(ort.load_openrouter_keys(os.environ))
    return len(_pool_module().load_gemini_keys(os.environ))


def openrouter_thread_count(key_count: int, sends_per_window: Optional[int] = None) -> int:
    """Потоки для OpenRouter: ключи × настроенное openrouter_sends_per_window.

    Константа SENDS_PER_WINDOW здесь не используется. Лимит MAX_PARALLEL_BATCHES для OpenRouter не действует.
    Значение не передано: один поток на ключ.
    """
    # значение приходит из конфига через gemini_options; без него — по одному потоку на ключ
    per_key = 1 if sends_per_window is None else max(1, int(sends_per_window))
    return max(1, key_count) * per_key


class _WaveJob:
    """Батчи одной волны и всё, нужное для их выполнения: построитель, модель, стиль, колбэк, свои out и out_lock.

    Поля очереди (next_idx, stopped, first_error, inflight, main_inflight) меняются под замком очереди: своим у одиночной волны
    или общим у SharedBatchPool. Модель вызывается вне всех замков.
    """

    def __init__(self, wave: str, builder, batches: list, transport, model: str, style_text: str,
                 on_batch: Optional[Callable[[list], None]], tag: str = "", sched_lock=None,
                 rework: Optional[list] = None) -> None:
        self.wave = wave                 # имя волны: по нему пул учитывает вызовы модели
        self.tag = tag                   # префикс строк журнала батчей: «[имя волны] » в пуле, «» без пула
        self.builder = builder
        self.batches = batches
        self.transport = transport
        self.model = model
        self.style_text = style_text
        self.on_batch = on_batch
        self.out: dict = {}
        self.out_lock = threading.Lock()  # out и вызов on_batch под одним замком; не держится при вызове модели
        self.next_idx = 0                 # следующий непринятый батч
        self.stopped = False              # первая ошибка волны (или остановка пула): новые батчи не берём
        self.first_error: list = []       # первая ошибка волны; последующие не сохраняются
        self.inflight = 0                 # батчи, которые выполняются сейчас (в любом потоке)
        self.main_inflight = 0            # главные батчи, которые выполняются сейчас; остаток исправлений ждёт их конца
        # замок очереди: у пула свой (его _cond), у одиночной волны — замок её очереди батчей
        self.sched_lock = sched_lock if sched_lock is not None else threading.Lock()
        self.rework_queue: list = list(rework or [])  # клипы, ждущие исправления (PromptResult); меняются под замком очереди
        self.fix_no = 0                   # номер последнего пакета исправлений волны (с 1); растёт под замком очереди пула, без пула — в вызывающем потоке
        self.pool = None                  # общий пул, если волна в нём; ставится в arrive_and_wait

    def process(self, logged, idx: int) -> None:
        """Выполняет батч idx: запрос, стиль, журнал, затем out и on_batch под out_lock. Исключения не глотает."""
        batch_no = idx + 1
        batch = self.batches[idx]
        logged.start_batch(batch_no)
        log.info("%sБатч %d из %d: начало, клипов в батче: %d.", self.tag, batch_no, len(self.batches),
                 len(batch.nums), extra=KEY_EVENT_EXTRA)
        t_batch = time.monotonic()
        batch_res = _run_batch(logged, self.model, self.builder, batch, self.tag)
        for r in batch_res.values():
            if r.prompt.strip() and r.rework_pending:
                # клип в очереди исправлений: сырой текст в prompt_raw (по нему проверка и исправление), в prompt — со стилем
                r.prompt_raw = r.prompt
            if r.prompt.strip():
                # стиль пресета прикрепляется последним; пустой промпт не трогаем
                r.prompt = attach_style(r.prompt, self.style_text)
        # журнал по клипам: принятые — с длиной и коротким хешем (текст промпта в журнал не пишется)
        review_nums = []
        for n in batch.nums:
            r = batch_res.get(n)
            if r is None:
                continue
            if r.needs_review:
                review_nums.append(n)
            elif r.prompt.strip() and not r.rework_pending:
                digest = hashlib.sha256(r.prompt.encode("utf-8")).hexdigest()[:12]
                log.info("Промпт принят: клип %d, %d символов, sha256 %s.", n, len(r.prompt), digest)
        if review_nums:
            log.warning("%sБатч %d: промпты требуют проверки, клипы: %s.",
                        self.tag, batch_no, ", ".join(str(n) for n in review_nums))
        # клипы с нарушениями: в запись уходят со стилем, сырой текст в prompt_raw; в очередь исправлений — после записи
        queued = [batch_res[n] for n in batch.nums if n in batch_res and batch_res[n].rework_pending]
        if queued:
            log.info("%sБатч %d: клипов с нарушениями, в очереди исправлений: %d.", self.tag, batch_no, len(queued))
        with_text = sum(1 for r in batch_res.values() if r.prompt.strip())
        log.info("%sБатч %d: получено промптов с текстом: %d из %d, за %.1f с.",
                 self.tag, batch_no, with_text, len(batch_res), time.monotonic() - t_batch,
                 extra=KEY_EVENT_EXTRA)
        with self.out_lock:
            self.out.update(batch_res)
            if self.on_batch is not None:
                self.on_batch([batch_res[n] for n in batch.nums])
        self.enqueue(queued)

    def enqueue(self, clips: list) -> None:
        """Ставит клипы в очередь исправлений. Замок очереди не держится при вызове модели."""
        if not clips:
            return
        if self.pool is not None:
            self.pool.enqueue(self, clips)
            return
        with self.sched_lock:
            self.rework_queue.extend(clips)

    def abandon_queue(self) -> None:
        """Клипы, оставшиеся в очереди исправлений после остановки волны, не исправлены: помечаются на проверку.

        Вызывается только когда потоков волны уже нет (после join или pool.wait_job): остановленная волна
        новых пакетов не берёт, поэтому замок очереди не нужен. Клипы не возвращаются в очередь; rework_pending
        и prompt_raw сохраняются, повторный запуск продолжит исправление. Записываются один раз под out_lock.
        """
        if not self.rework_queue:
            return
        left = list(self.rework_queue)
        self.rework_queue.clear()
        for c in left:
            c.needs_review = True
            c.review_reason = "очередь исправлений не выполнена: прогон остановлен"
            if c.prompt_raw.strip():
                c.prompt = attach_style(c.prompt_raw, self.style_text)
        ordered = sorted(left, key=lambda c: c.num)
        with self.out_lock:
            self.out.update({c.num: c for c in ordered})
            if self.on_batch is not None:
                self.on_batch(ordered)
        log.warning("%sНе исправлено клипов из-за остановки: %d, номера: %s.",
                    self.tag, len(ordered), _num_ranges([c.num for c in ordered]))

    def process_fix(self, logged, clips: list, fix_no: int) -> None:
        """Исправляет пакет клипов очереди: один запрос на волну, разбор, проверка, стиль, запись.

        fix_no: номер пакета в волне (с 1); он попадает в строки журнала пакета с тем же префиксом волны, что и батчи.
        В запрос входят только текущий промпт и что убрать: без SRT, пресета и стиля. Системная инструкция и схема
        берутся у волны. Замков при вызове модели нет. Исключения транспорта не глотаются (как у батча).
        """
        builder = self.builder
        nums = [c.num for c in clips]
        logged.start_batch(fix_no, "Фикс")   # номер попытки и номер пакета в строках повторов
        log.info("%sФикс %d: начало, клипов в пакете: %d.", self.tag, fix_no, len(clips), extra=KEY_EVENT_EXTRA)
        lines = [f"Prompt for segment_index {c.num}: {c.prompt_raw} {rework_remark(c.rework_pending)}" for c in clips]
        user = "\n".join(lines) + "\n\nReturn only the JSON array with one item per segment_index."
        try:
            raw = logged.generate_json(self.model, builder.system, user, builder.schema)
            results, _ = parse_and_validate(raw, nums, builder.cfg, builder.texts)
        except BaseException as exc:
            # пакет не выполнен (запрос или разбор): каждый клип пакета помечается на проверку с причиной.
            # rework_pending, rework_round и prompt_raw сохраняются; в очередь клипы не возвращаются.
            # Запись та же, что у обычного пути: под out_lock, один раз. Ошибка пробрасывается, волна останавливается.
            reason = f"исправление прервано: {type(exc).__name__}"
            for c in clips:
                c.needs_review = True
                c.review_reason = reason
                if c.prompt_raw.strip():
                    c.prompt = attach_style(c.prompt_raw, self.style_text)
            failed = sorted(clips, key=lambda c: c.num)
            with self.out_lock:
                self.out.update({c.num: c for c in failed})
                if self.on_batch is not None:
                    self.on_batch(failed)
            raise
        fixed = 0
        requeue: list = []
        for c in clips:
            new_text = results[c.num].prompt if c.num in results else ""
            if new_text.strip():
                c.prompt_raw = new_text   # пустой ответ: прежний сырой текст остаётся
            c.rework_round = (c.rework_round or 0) + 1
            found = _find_all_problems(c.prompt_raw, builder.cfg)  # проверка по сырому тексту, до стиля
            c.prompt = attach_style(c.prompt_raw, self.style_text)  # в записи — текст со стилем, как у клипов вне очереди
            if not found:
                c.rework_pending = []
                c.prompt_raw = ""   # клип исправлен: сырого текста в записи нет
                # Флаг пересчитывается по новому тексту: исправленный промпт, прошедший все проверки
                # (стиль, содержание, внешность по тексту сегмента), снимает старую причину. Раньше флаг
                # снимался только у «прерванных» причин, и клипы с причиной содержания оставались на проверке.
                rr = results.get(c.num)
                if rr is not None and rr.prompt.strip():
                    c.needs_review = rr.needs_review
                    c.review_reason = rr.review_reason if rr.needs_review else ""
                else:
                    c.needs_review = False
                    c.review_reason = ""
                fixed += 1
            elif c.rework_round >= REPAIR_ROUNDS:
                # круги кончились: текст последней версии сохраняется, клип помечается на проверку
                c.rework_pending = found
                c.needs_review = True
                c.review_reason = "; ".join(problem_ru(p) for p in found)
                log.warning("%sКлип %d: не исправлено за %d кругов: %s.",
                            self.tag, c.num, REPAIR_ROUNDS, c.review_reason)
            else:
                c.rework_pending = found
                requeue.append(c)
        ordered = sorted(clips, key=lambda c: c.num)
        with self.out_lock:
            self.out.update({c.num: c for c in ordered})
            if self.on_batch is not None:
                self.on_batch(ordered)
        if self.pool is not None:
            log.info("%sФикс: круг %d, клипов %d, исправлено %d, осталось %d.",
                     self.tag, max(c.rework_round for c in clips), len(clips), fixed, len(clips) - fixed)
        self.enqueue(requeue)


class SharedBatchPool:
    """Общий пул батчей двух волн (видео и фото): поток берёт батч из своей волны, иначе из чужой.

    shares: число потоков на волну (ключи делятся пополам: видео floor(K/2), фото ceil(K/2)).
    Если чужая волна в пул не пришла (вышла, не дойдя до батчей), её потоки достаются этой волне.
    bind: необязательный колбэк bind(имя волны | None) вокруг каждого батча в потоке пула; нужен для учёта вызовов по волнам.

    Замки: self._cond — очередь пула; его не держат при вызове модели и колбэка. out_lock волны и замок
    состояния файла от него независимы; вложенного захвата в обратном порядке нет.
    Барьер: ни один батч не берётся, пока обе волны не пришли в пул (или не вышли через leave). Так все
    чтения общего состояния идут до первого on_batch, и обе волны пишут в один словарь prompts.
    """

    def __init__(self, shares: dict, bind: Optional[Callable] = None) -> None:
        self._shares = dict(shares)
        self._bind = bind
        self._cond = threading.Condition()   # очередь батчей пула
        self._arrived: set = set()           # волны, которые пришли в пул или вышли
        self._jobs: dict = {}                # волна -> _WaveJob
        self._fatal: Optional[BaseException] = None  # первая ошибка не-Gemini: останавливает все волны

    def leave(self, wave: str) -> None:
        """Волна вышла, не дойдя до пула (работы нет или ошибка до запросов). Повторный вызов безвреден."""
        with self._cond:
            self._arrived.add(wave)
            self._cond.notify_all()

    def arrive_and_wait(self, wave: str, job: "_WaveJob") -> int:
        """Регистрирует батчи волны, ждёт прихода обеих волн и возвращает число потоков этой волны."""
        if wave not in self._shares:
            raise ValueError(f"волна {wave!r} не объявлена в общем пуле")
        with self._cond:
            self._jobs[wave] = job
            job.pool = self   # очередь исправлений этой волны ведёт пул под своим замком
            self._arrived.add(wave)
            self._cond.notify_all()
            self._cond.wait_for(lambda: set(self._shares) <= self._arrived)
            other_has_work = any(j is not None and (j.batches or j.rework_queue)
                                 for w, j in self._jobs.items() if w != wave)
            if other_has_work:
                return int(self._shares[wave])
            return sum(int(v) for v in self._shares.values())

    def _set_fatal(self, exc: BaseException) -> None:
        """Под замком очереди: первая ошибка не-Gemini останавливает все волны."""
        if self._fatal is None:
            self._fatal = exc
        for job in self._jobs.values():
            if job is not None:
                job.stopped = True

    def fail(self, exc: BaseException) -> None:
        """Остановка всего пула из вызывающего кода (например, ошибка видео вне батчей)."""
        with self._cond:
            self._set_fatal(exc)
            self._cond.notify_all()

    def failure(self) -> Optional[BaseException]:
        """Первая ошибка пула (не-Gemini) или None."""
        with self._cond:
            return self._fatal

    def _take(self, own: str):
        """Следующая работа потока: ('main', волна, idx) или ('fix', волна, (номер пакета, клипы)); None, когда работы больше не будет.

        Порядок: главный батч (своя волна, затем чужая). Если главных не осталось: пакет исправлений, полный
        (FIX_BATCH_SIZE клипов одной волны) сразу; остаток — только когда нигде не выполняется главный батч (исправления других волн не мешают).
        Иначе поток ждёт в wait(); замок при ожидании отпускается. Ошибка пула останавливает выдачу.
        """
        with self._cond:
            while True:
                if self._fatal is not None:
                    return None
                main = self._pick_main(own)
                if main is not None:
                    job, idx = main
                    return "main", job, idx
                fix = self._pick_fix(own)
                if fix is not None:
                    job, clips, fix_no = fix
                    return "fix", job, (fix_no, clips)
                if self._idle():
                    return None
                self._cond.wait()

    def _pick_main(self, own: str):
        """Под замком очереди: следующий главный батч (своя волна, затем чужая) или None."""
        order = [own] + [w for w in self._shares if w != own]
        for w in order:
            job = self._jobs.get(w)
            if job is None or job.stopped or job.next_idx >= len(job.batches):
                continue
            idx = job.next_idx
            job.next_idx += 1
            job.inflight += 1
            job.main_inflight += 1   # парный декремент — в _finish при kind == "main"
            return job, idx
        return None

    def _pick_fix(self, own: str):
        """Под замком очереди: пакет исправлений одной волны (своя, затем чужая) или None."""
        order = [own] + [w for w in self._shares if w != own]
        for w in order:
            job = self._jobs.get(w)
            if job is not None and not job.stopped and len(job.rework_queue) >= FIX_BATCH_SIZE:
                return self._cut_fix(job)
        if all(j.main_inflight == 0 for j in self._jobs.values() if j is not None):
            for w in order:
                job = self._jobs.get(w)
                if job is not None and not job.stopped and job.rework_queue:
                    return self._cut_fix(job)
        return None

    @staticmethod
    def _cut_fix(job: "_WaveJob"):
        """Под замком очереди: снимает с очереди до FIX_BATCH_SIZE клипов, отмечает выполнение и выдаёт номер пакета волны."""
        clips = job.rework_queue[:FIX_BATCH_SIZE]
        del job.rework_queue[:FIX_BATCH_SIZE]
        job.inflight += 1
        job.fix_no += 1   # номер берётся здесь, под замком пула: два потока не получат один номер
        return job, clips, job.fix_no

    def _idle(self) -> bool:
        """Под замком очереди: в не остановленных волнах не осталось ни батчей, ни исправлений, ни выполнений."""
        for job in self._jobs.values():
            if job is None or job.stopped:
                continue
            if job.next_idx < len(job.batches) or job.rework_queue or job.inflight:
                return False
        return True

    def enqueue(self, job: "_WaveJob", clips: list) -> None:
        """Ставит клипы волны в очередь исправлений и будит потоки пула."""
        with self._cond:
            job.rework_queue.extend(clips)
            self._cond.notify_all()

    def _finish(self, job: "_WaveJob", exc: Optional[BaseException], *, kind: str) -> None:
        """Под замком очереди: снимает отметку выполнения; главный батч (kind == "main") снимает и main_inflight."""
        with self._cond:
            job.inflight -= 1
            if kind == "main":
                job.main_inflight -= 1
            if exc is not None:
                # ошибка не глотается: первая ошибка волны; новые батчи волны не берём, запущенные доходят до конца
                if not job.first_error:
                    job.first_error.append(exc)
                job.stopped = True
                if not isinstance(exc, GeminiRuntimeError):
                    self._set_fatal(exc)
            self._cond.notify_all()

    def wait_job(self, job: "_WaveJob") -> None:
        """Ждёт, пока все батчи волны выполнены (или волна остановлена) и никто не выполняет её батч."""
        with self._cond:
            self._cond.wait_for(lambda: job.inflight == 0 and (
                job.stopped or self._fatal is not None
                or (job.next_idx >= len(job.batches) and not job.rework_queue)))

    def run_worker(self, own: str) -> None:
        """Тело потока пула: берёт главные батчи и пакеты исправлений, пока есть работа. Ошибка батча не останавливает поток."""
        while True:
            got = self._take(own)
            if got is None:
                return
            kind, job, payload = got
            logged = _LoggedTransport(job.transport, tag=job.tag)   # журнал свой у каждого батча: номер попытки сбрасывается
            exc: Optional[BaseException] = None
            if self._bind is not None:
                self._bind(job.wave)
            try:
                if kind == "main":
                    job.process(logged, payload)
                else:
                    fix_no, clips = payload
                    job.process_fix(logged, clips, fix_no)
            except BaseException as err:
                exc = err
            finally:
                if self._bind is not None:
                    self._bind(None)
            self._finish(job, exc, kind=kind)


def generate_prompts(segments: list, wanted_nums: list, cfg: PromptConfig,
                     transport: GeminiTransport, model: str,
                     on_batch: Optional[Callable[[list], None]] = None,
                     max_workers: Optional[int] = None, provider: str = "gemini",
                     pool: Optional["SharedBatchPool"] = None, wave: Optional[str] = None,
                     rework: Optional[list] = None, sends_per_window: Optional[int] = None) -> list:
    """Промпты для нужных номеров: список PromptResult по возрастанию номера.

    Все проверки входа выполняются до первого обращения к transport. Ошибки транспорта
    (GeminiRuntimeError) не перехватываются. Номер без пригодного текста получает prompt \"\"
    и needs_review=True: решение, пускать ли такой в генерацию, за оркестратором.

    on_batch(results): необязательный колбэк. Вызывается после каждого готового батча (с учётом
    retry) под общим замком; при параллельных батчах порядок вызовов соответствует порядку завершения батчей, а не номеру. results: list[PromptResult] этого батча по возрастанию номера,
    включая needs_review. Если позже случится GeminiRuntimeError, уже готовые батчи колбэку
    отданы. Исключения колбэка не перехватываются. Без колбэка поведение прежнее.

    max_workers: число потоков. По умолчанию берётся число ключей провайдера provider из окружения
    (для gemini тот же источник, что у RealTransport), не больше MAX_PARALLEL_BATCHES и числа батчей. При одном потоке
    работа идёт в вызывающем потоке, как раньше. Батчи берутся из очереди по порядку; вызовы
    on_batch идут под одним замком в порядке завершения батчей. Первая ошибка останавливает взятие
    новых батчей; уже запущенные батчи доходят до конца, после чего ошибка пробрасывается.

    provider: провайдер текстовой модели (gemini по умолчанию или openrouter). От него зависит,
    чьи ключи считаются для числа потоков, если max_workers не задан.

    pool, wave: общий пул батчей двух волн (SharedBatchPool) и имя этой волны в нём. Без пула (по умолчанию)
    работа идёт как раньше: своя очередь и потоки по числу ключей. С пулом число потоков и очередь задаёт пул,
    max_workers не используется; вызов возвращается, когда выполнены все батчи его волны, даже если их
    выполняли потоки другой волны. Ошибка волны пробрасывается ей, ошибка не-Gemini останавливает весь пул.

    rework: клипы из сохранённого состояния, у которых нарушения ещё не исправлены (очередь прошлого запуска).
    Им текст уже есть, батчей для них нет: они только попадают в очередь исправлений этого вызова.
    Без них поведение прежнее. Исправления после главных батчей идут в обоих путях (с пулом и без него).
    """
    seeds = list(rework or [])
    # построитель знает нужные номера и номера очереди; батчи строятся только для нужных номеров
    builder = _Builder(segments, list(wanted_nums) + [r.num for r in seeds], cfg)
    # текст пресета читается до первого запроса: ошибка файла остаётся ошибкой входа
    style_text = read_preset(cfg.preset) if cfg.preset != PRESET_NONE else ""
    t_all = time.monotonic()
    batches = builder.batches(list(wanted_nums))
    # ключи для баннера: те же ключи провайдера, что и для числа потоков; ошибка чтения — ноль
    try:
        key_count = _provider_key_count(provider)
    except Exception:
        key_count = 0
    kind = "видео" if cfg.media == "video" else "фото"
    log.info("=== Генерация %s-промптов: клипов %d, батчей %d, ключей %d ===",
             kind, len(set(wanted_nums)), len(batches), key_count, extra=KEY_EVENT_EXTRA)

    if pool is None:
        # одиночная волна: число потоков по числу ключей провайдера; без ключей — один поток (как раньше)
        if max_workers is None:
            if provider == "openrouter":
                # значение openrouter_sends_per_window из конфига; без него — один поток на ключ
                max_workers = openrouter_thread_count(key_count, sends_per_window)
            else:
                max_workers = key_count
            if max_workers < 1:
                log.info("Ключей провайдера %s не найдено, батчи идут в один поток.", provider)
                max_workers = 1
        if provider == "openrouter":
            n_workers = max(1, min(max_workers, len(batches)))
        else:
            n_workers = max(1, min(max_workers, MAX_PARALLEL_BATCHES, len(batches)))
        log.info("Батчи выполняются в %d потоках.", n_workers)

        queue_lock = threading.Lock()    # очередь батчей, флаг остановки, первая ошибка и очередь исправлений
        job = _WaveJob(kind, builder, batches, transport, model, style_text, on_batch,
                       sched_lock=queue_lock, rework=seeds)
        stop = threading.Event()
        first_error: list = []           # первая ошибка; последующие не сохраняются (их видно в журнале попыток)

        def worker() -> None:
            # журнал свой у каждого потока: номер батча и попытка не делятся между потоками
            logged = _LoggedTransport(transport)
            while True:
                with queue_lock:
                    if stop.is_set() or job.next_idx >= len(job.batches):
                        return
                    idx = job.next_idx
                    job.next_idx += 1
                try:
                    job.process(logged, idx)
                except BaseException as exc:
                    # ошибка не глотается: запоминаем первую, новые батчи не берём, запущенные доходят до конца
                    with queue_lock:
                        if not first_error:
                            first_error.append(exc)
                        stop.set()
                    return

        if n_workers == 1:
            worker()   # один ключ: тот же поток, та же последовательность, что раньше
        else:
            threads = [threading.Thread(target=worker, name=f"gemini-batch-{i + 1}")
                       for i in range(n_workers)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
        # исправления очереди этой волны: потоков уже нет, пакеты идут по одному (без пула)
        logged_fix = _LoggedTransport(transport)
        while job.rework_queue and not first_error:
            clips = job.rework_queue[:FIX_BATCH_SIZE]
            del job.rework_queue[:FIX_BATCH_SIZE]
            job.fix_no += 1   # потоков уже нет, счётчик меняет только этот поток: замок не нужен
            try:
                job.process_fix(logged_fix, clips, job.fix_no)
            except BaseException as exc:
                first_error.append(exc)
        job.abandon_queue()   # клипы очереди остановленной волны не исправлены: на проверку, с причиной
        if first_error:
            _wave_summary(job)
            raise first_error[0]
        out = job.out
    else:
        # общий пул: батчи этой волны выполняют потоки пула (свои и чужие), число потоков задаёт пул
        wave_name = wave if wave is not None else kind
        # имя волны в строках батчей пула: «[видео] » или «[фото] »
        job = _WaveJob(wave_name, builder, batches, transport, model, style_text, on_batch,
                       tag=f"[{wave_name}] ", rework=seeds)
        n_workers = pool.arrive_and_wait(job.wave, job)
        log.info("%sБатчи выполняются в %d потоках (общий пул волн).", job.tag, n_workers)
        threads = [threading.Thread(target=pool.run_worker, args=(job.wave,), name=f"gemini-{job.wave}-{i + 1}")
                   for i in range(n_workers)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        # батчи этой волны могут выполнять потоки другой волны: ждём их конца, прежде чем отдать результат
        pool.wait_job(job)
        job.abandon_queue()   # клипы очереди остановленной волны не исправлены: на проверку, с причиной
        if job.first_error:
            _wave_summary(job)
            raise job.first_error[0]
        failure = pool.failure()
        if failure is not None:
            _wave_summary(job)
            raise failure
        out = job.out

    results = [out[n] for n in builder.wanted]
    flagged = sum(1 for r in results if r.needs_review)
    log.info("Промпты готовы: %d, из них требуют проверки: %d.", len(results), flagged)
    # итог по волне: клипы с нарушениями, не исправленными за REPAIR_ROUNDS кругов (текст последней версии сохранён)
    unfixed = [r.num for r in results if r.needs_review and r.rework_pending]
    log.info("[%s] Нарушения запрещённых типов и маркеров: не исправлено клипов %d, номера: %s.",
             wave if wave is not None else kind, len(unfixed), _num_ranges(unfixed) or "нет")
    # итог прогона: готово / без текста; клипы с отметкой проверки в эти числа не входят (их число в строке «Промпты готовы»)
    ready = sum(1 for r in results if r.prompt.strip() and not r.needs_review)
    no_text = sum(1 for r in results if not r.prompt.strip())
    log.info("=== Генерация %s-промптов готова: готово %d, без текста %d, за %.1f с ===",
             kind, ready, no_text, time.monotonic() - t_all, extra=KEY_EVENT_EXTRA)
    return results


# ---------------------------------------------------------------------------
# Реальный транспорт (httpx)
# ---------------------------------------------------------------------------

def _silence_http_logs() -> None:
    for name in ("httpx", "httpcore"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)


_SAFE_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,40}$")


def _safe_error_fields(response: Any) -> tuple:
    """(code, daily_quota, retry_after) из тела ошибки. Текст сообщения из тела не берётся."""
    code, daily, retry = "", False, None
    try:
        err = response.json().get("error", {})
        c = err.get("status") or err.get("code")
        if isinstance(c, str):
            c = c.upper()
            if _SAFE_CODE_RE.match(c):
                code = c
        for d in err.get("details", []) or []:
            if not isinstance(d, dict):
                continue
            for v in d.get("violations", []) or []:
                qid = str(v.get("quotaId", "")) if isinstance(v, dict) else ""
                # суточная квота: как _is_daily_quota_error в Download/generate_queries.py (без учёта регистра)
                if "perday" in qid.lower():
                    daily = True
            delay = d.get("retryDelay")
            if isinstance(delay, str):
                m = re.fullmatch(r"(\d+(?:\.\d+)?)s", delay)
                if m:
                    retry = float(m.group(1))
    except Exception:
        pass
    if code == "QUOTA_EXCEEDED":   # запасной признак суточной квоты
        daily = True
    return code, daily, retry


def _quota_log_fields(response: Any) -> list:
    """Безопасные поля тела 429 для диагностики: какая квота исчерпана. Только перечисленные ключи;
    error.message, содержимое запроса и ключ API не берутся. Для сбоя разбора возвращается неполный список."""
    def _text(value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""  # вложенные структуры в лог не пишем
        return str(value)[:200]

    out: list = []
    try:
        err = response.json().get("error", {})
        if not isinstance(err, dict):
            return out
        out.append({"status": _text(err.get("status")), "code": _text(err.get("code"))})
        for d in err.get("details", []) or []:
            if not isinstance(d, dict):
                continue
            item = {"@type": _text(d.get("@type"))}
            for key in ("quotaId", "quotaMetric", "quotaValue", "retryDelay"):
                if key in d:
                    item[key] = _text(d.get(key))
            violations = [
                {k: _text(v.get(k)) for k in ("quotaMetric", "quotaId", "quotaValue") if k in v}
                for v in (d.get("violations", []) or []) if isinstance(v, dict)
            ]
            if violations:
                item["violations"] = violations
            out.append(item)
    except Exception:
        pass
    return out

# Значения по умолчанию для лимитов пула: совпадают с config.default.yaml. Конфиг (cli) передаёт свои значения явно.
DEFAULT_POOL_RPM = 15         # запросов в минуту
DEFAULT_POOL_TPM = 250_000    # токенов в минуту
DEFAULT_POOL_RPD = 500        # запросов в сутки на одну модель
# Оценка токенов для TPM: те же константы, что в gemini_orchestrator (CHARS_PER_TOKEN, output_reserve_tokens).
POOL_CHARS_PER_TOKEN = 3
POOL_OUTPUT_RESERVE_TOKENS = 2000


def _pool_module():
    """Модуль пула подгружается при создании транспорта: gemini_pool сам импортирует gemini_prompts."""
    try:
        from . import gemini_pool as gpool
    except ImportError:  # плоский импорт, как в остальных модулях core
        import gemini_pool as gpool  # type: ignore
    return gpool


def _retry_after_header(response: Any) -> Optional[float]:
    """Заголовок Retry-After в секундах. Дата в HTTP-формате или мусор дают None."""
    try:
        value = float(response.headers.get("Retry-After", ""))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


class RealTransport:
    """Реальный транспорт Gemini на httpx поверх общего пула ключей (gemini_pool).

    Для каждого вызова generate_json берётся ключ (lease) из пула; ключ уходит только в заголовок
    x-goog-api-key; в URL, логи и исключения он не попадает. Исход запроса возвращается в пул."""

    def __init__(self, *, pool: Any = None, timeout: float = 120.0, client: Any = None,
                 base_url: str = API_BASE_URL, environ: Any = None,
                 rpm_limit: int = DEFAULT_POOL_RPM, tpm_limit: int = DEFAULT_POOL_TPM,
                 rpd_limit: int = DEFAULT_POOL_RPD, model_limits: Any = None):
        # лимиты передаются в пул; без явных значений берутся DEFAULT_POOL_* (совпадают с конфигом по умолчанию)
        # model_limits: модель -> (rpm, tpm, rpd); для остальных моделей действуют общие лимиты
        try:
            import httpx
        except ImportError:
            raise GeminiInputError("Не установлен пакет httpx.") from None
        self._gpool = _pool_module()
        if pool is None:
            # ключи читаются здесь, при создании пула; в самом запросе окружение не трогаем
            keys = self._gpool.load_gemini_keys(os.environ if environ is None else environ)
            pool = self._gpool.KeyPool(keys, rpm_limit=rpm_limit, tpm_limit=tpm_limit, rpd_limit=rpd_limit,
                                       model_limits=model_limits)
        self._pool = pool
        self._tpm = tpm_limit    # оценка токенов не должна превышать TPM пула
        _silence_http_logs()
        self._httpx = httpx
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._base_url = base_url.rstrip("/")

    def __repr__(self) -> str:
        return "RealTransport()"

    def close(self) -> None:
        """Закрывает внутренний httpx.Client. Клиент, переданный снаружи, не закрывается.
        Повторный вызов безопасен."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "RealTransport":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def generate_json(self, model: str, system: str, user: str, schema: dict) -> str:
        gpool = self._gpool
        url = f"{self._base_url}/{model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseJsonSchema": schema},
        }
        tokens = min((len(system) + len(user)) // POOL_CHARS_PER_TOKEN + 1 + POOL_OUTPUT_RESERVE_TOKENS,
                     self._tpm)
        try:
            lease = self._pool.acquire(model, tokens=tokens)
        except gpool.AllKeysExhaustedError:
            # все ключи исчерпали суточную квоту: признак daily_quota, чтобы ManagedTransport сменил модель
            # или остановил прогон (как раньше, без временного сбоя)
            raise GeminiRuntimeError(
                "Все ключи Gemini исчерпали суточную квоту.",
                status=429, code="RESOURCE_EXHAUSTED", daily_quota=True) from None
        # заголовок с ключом живёт только внутри этого вызова и никуда не передаётся
        headers = {API_KEY_HEADER: lease.api_key, "Content-Type": "application/json"}
        try:
            resp = self._client.post(url, headers=headers, json=body)
        except self._httpx.HTTPError as e:
            # ключ не виноват: состояние не меняется; в сообщение идёт только имя класса ошибки
            self._pool.release(lease, gpool.Outcome.NETWORK_ERROR, model)
            raise GeminiRuntimeError(
                f"Сбой сети при обращении к Gemini ({type(e).__name__}).") from None
        except Exception as e:
            # любой сбой до ответа тоже без штрафа ключу
            self._pool.release(lease, gpool.Outcome.NETWORK_ERROR, model)
            raise GeminiRuntimeError(
                f"Непредвиденный сбой при обращении к Gemini ({type(e).__name__}).") from None

        status = resp.status_code
        if status == 200:
            self._pool.release(lease, gpool.Outcome.SUCCESS, model)
        else:
            code, daily, retry = _safe_error_fields(resp)
            if status == 429:
                if daily:
                    self._pool.release(lease, gpool.Outcome.DAILY_QUOTA, model)
                else:
                    self._pool.release(lease, gpool.Outcome.RATE_LIMITED, model,
                                       retry_after=_retry_after_header(resp))
            else:
                # 500, 503, 504 и прочие ошибки (в том числе 4xx): ключ не виноват, состояние не меняется
                self._pool.release(lease, gpool.Outcome.SERVER_ERROR, model)
            extra = f", код {code}" if code else ""
            err = GeminiRuntimeError(
                f"Gemini вернул ошибку: HTTP {status}{extra}.",
                status=status, code=code, daily_quota=daily, retry_after=retry)
            if status == 429:
                # диагностика: какая квота исчерпана; ключ, промпт и текст ответа в лог не пишутся
                log.warning("HTTP 429 от Gemini, модель %s: признаки квоты %s; оценка входных токенов: %d.",
                            model, json.dumps(_quota_log_fields(resp), ensure_ascii=False),
                            round((len(system) + len(user)) / 4))
            if status == 429 and daily:
                # квота исчерпана только у этого ключа для этой модели: пул может выдать другой ключ.
                # ManagedTransport по этому признаку повторит модель, а не переключит её.
                err.key_exhausted_only = True
            raise err
        try:
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except Exception:
            # пустой/заблокированный ответ: пусть валидатор пометит батч и сделает retry
            return ""
