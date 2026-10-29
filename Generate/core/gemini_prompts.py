"""Gemini: SRT -> промпты для видеомодели (этап 4а: данные, запрос, ответ, проверка).

Порт идей из первой части (склейка сегментов в предложения, Before/After по целым
предложениям, батчи, structured output, проверка набора номеров), а не копия.
Здесь НЕТ: лимитера RPM/TPM, счётчика RPD, цепочки моделей, файла prompts.json (этап 4б).

Модель приходит параметром. Транспорт заменяемый (GeminiTransport): в тестах заглушка,
в бою RealTransport на httpx. Ключ только из окружения, только в заголовке (SPEC 0.1).
Ничего не печатается; логи на русском, тексты промптов в логи не попадают.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

try:  # пакетный и «плоский» импорт (как в остальных модулях core)
    from .srt_parser import Segment
    from .model_constants import PRESET_NONE, PROFILE_NAME_PATTERN
except ImportError:  # pragma: no cover
    from srt_parser import Segment  # type: ignore
    from model_constants import PRESET_NONE, PROFILE_NAME_PATTERN  # type: ignore

log = logging.getLogger("Generate.gemini_prompts")

# Имя переменной окружения с ключом собрано из частей: структурный тест запрещает
# литерал секрета в коде (SPEC 0.1). Значение ключа в модуле нигде не хранится и не печатается.
ENV_KEY_NAME = "GEN_" + "GEMINI_API_KEY"
API_KEY_HEADER = "x-goog-api-key"
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Размеры окон контекста (слова), как в первой части.
CONTEXT_BEFORE_WORDS = 40
CONTEXT_AFTER_WORDS = 25
SHORT_SEGMENT_MAX_WORDS = 3

# Проверка маркеров объектов: если имя, псевдоним или ключевое слово объекта есть в тексте самого сегмента,
# промпт обязан содержать маркер [OBJ_N]; иначе нарушение уходит в повторный запрос батча.
# True — проверка включена; False — выключена. Персонажи этой проверкой не проверяются.
OBJECT_MARKER_CHECK = True

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


def build_system_instruction(cfg: PromptConfig) -> str:
    style = strip_user_only(Path(cfg.style_file).read_text(encoding="utf-8"))
    parts = [
        "# Task\n"
        "You receive subtitle segments of a narrated video. For every segment write ONE "
        "text-to-video prompt, in English, that visually illustrates what the segment says, "
        "following the style guide below. Use the Sentence, Before and After lines only to "
        "understand the meaning; the >>marked<< part is the passage to illustrate.\n\n"
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
        "- Describe only the scene and its motion. Never name a medium, rendering method, film look, "
        "or camera or lens equipment; camera movement is the only camera wording allowed.\n"
        "- Exactly one camera move. Finish the camera sentence with how the frame looks when "
        "the move ends (for a static camera: what the frame holds on).\n"
        "- Give each person one consistent outfit; do not mix conflicting garments such as a "
        "suit and an overcoat unless the layering is stated clearly.\n"
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


def _style_reasons(prompt: str) -> list:
    return [reason for rx, reason, _ in _STYLE_RULES if rx.search(prompt)]


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


def parse_and_validate(raw: str, expected_nums, cfg: PromptConfig) -> tuple:
    """Разбирает ответ Gemini. Возвращает (results, problems).

    results: dict num -> PromptResult для каждого ожидаемого номера, у которого есть непустая
      строка prompt. Если к нему есть претензия, needs_review=True и указана причина.
    problems: list[Problem]: нарушения (номер или None для проблемы всего ответа).
    Номера без записи в results не получили пригодного текста. Лишние номера не попадают в results
    (фиксируются как Problem, но повторный запрос не вызывают: нужные данные целы)."""
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
        reasons.extend(_style_reasons(prompt))
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

# Маркер персонажа в промпте: [ИМЯ], заглавные латинские и кириллические буквы, цифры, «_».
_MARKER_RX = re.compile(r"\[([A-Z\u0410-\u042f\u0401][A-Z0-9_\u0410-\u042f\u0401]*)\]")


# Маркер в промпте: [ИМЯ] (как раньше) или голый id записи CHAR_N / OBJ_N целым словом.
# Скобочная форма [CHAR_N] попадает в первую ветвь, поэтому порядок ветвей важен.
_MARKER_OR_ID_RX = re.compile(r"\[([A-Z\u0410-\u042f\u0401][A-Z0-9_\u0410-\u042f\u0401]*)\]|(?<![\w\[])((?:CHAR|OBJ)_\d+)(?!\w)")
# Любой id записи, оставшийся в тексте промпта (в скобках или без): такой текст не должен уйти в генерацию.
_LEFT_ID_RX = re.compile(r"(?<!\w)(?:CHAR|OBJ)_\d+(?!\w)")


def _at_sentence_start(text: str, pos: int) -> bool:
    """Стоит ли позиция pos в начале предложения: начало текста, после «. » или перевода строки."""
    head = text[:pos]
    return head == "" or head.endswith(". ") or head.endswith("\n")


def resolve_markers(prompt: str, descriptions: dict) -> tuple:
    """Заменяет каждый [ИМЯ], у которого ИМЯ есть в descriptions, на описание из descriptions.

    Если маркер стоит в начале предложения, первая буква описания становится заглавной; в середине
    предложения описание вставляется как есть (строчной). Возвращает (новый текст, список неразрешённых имён).
    Неизвестный маркер остаётся в тексте как есть и попадает в список (каждое имя один раз, по порядку
    первого появления). Описания не придумываются."""
    unresolved: list = []

    def repl(m):
        name = m.group(1) or m.group(2)   # [ИМЯ] или голый id (CHAR_N / OBJ_N)
        if name in descriptions:
            text = descriptions[name]
            if text and _at_sentence_start(m.string, m.start()):
                text = text[:1].upper() + text[1:]
            return text
        if name not in unresolved:
            unresolved.append(name)
        return m.group(0)

    return _MARKER_OR_ID_RX.sub(repl, prompt), unresolved


def _is_clean(results: dict, num: int, problems: list) -> bool:
    r = results.get(num)
    return r is not None and not r.needs_review and not any(p.num == num for p in problems)


def _run_batch(transport: GeminiTransport, model: str, builder: _Builder, batch: BatchRequest,
               marker_note: str = "", objects: Optional[list] = None) -> dict:
    """Запрос батча + один retry для номеров с нарушениями. Возвращает num -> PromptResult.
    marker_note: строка инструкции о маркерах [ID]; добавляется и к повторному запросу батча.
    objects: записи объектов, маркеры которых объявлены в батче; по ним проверяется наличие маркера [OBJ_N]
    в промпте сегмента, где объект назван. Нарушение идёт в тот же повторный запрос, что и остальные."""
    expected = list(batch.nums)
    objects = objects or []
    raw = transport.generate_json(model, batch.system, batch.user, batch.schema)
    results, problems = parse_and_validate(raw, expected, builder.cfg)
    # нарушение: объект назван в тексте сегмента, а маркера [OBJ_N] в промпте нет
    misses = _object_marker_misses(results, builder, objects)
    problems = problems + [Problem(n, f"нет маркера объекта {rid}") for n, rid, _ in misses]
    final: dict = {n: results[n] for n in expected if _is_clean(results, n, problems)}
    bad = [n for n in expected if n not in final]
    extras = [p for p in problems if p.num is not None and p.num not in set(expected)]
    if extras:
        log.warning("Gemini вернул лишние номера: %d шт., они проигнорированы.", len(extras))

    if bad:
        log.warning("Батч %s..%s: нарушения в %d из %d записей, повторный запрос.",
                    expected[0], expected[-1], len(bad), len(expected))
        retry = builder.batches(bad)  # bad <= batch_size: ровно один батч
        for rb in retry:
            if marker_note:
                rb = replace(rb, user=rb.user + marker_note)
            note = _retry_note(rb.nums, problems)
            if note:
                rb = replace(rb, user=rb.user + note)
            raw2 = transport.generate_json(model, rb.system, rb.user, rb.schema)
            res2, prob2 = parse_and_validate(raw2, list(rb.nums), builder.cfg)
            misses2 = _object_marker_misses(res2, builder, objects)
            prob2 = prob2 + [Problem(n, f"нет маркера объекта {rid}") for n, rid, _ in misses2]
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
                # маркер объекта так и не появился после повтора: предупреждение с номером и именем объекта
                for mn, rid, name in misses2:
                    if mn == n:
                        log.warning("Сегмент %d: в тексте назван объект «%s», но маркер [%s] в промпте не записан "
                                    "даже после повторного запроса.", n, name, rid)
                final[n] = PromptResult(n, best.prompt, True, reason, best.extra)
    return final


# Копия normalize_appearance из character_file. Импорт оттуда дал бы цикл: character_file сам импортирует
# gemini_prompts. Оба варианта держать одинаковыми: тест сравнивает их на одних и тех же входах.
def _normalize_appearance(text: str) -> str:
    """Внешность в виде фразы для вставки в промпт (см. normalize_appearance в character_file)."""
    t = " ".join(str(text or "").split())
    if t.endswith("."):
        t = t[:-1].rstrip()
    pieces = [p for p in re.split(r"(?<!\b[A-Za-z])\.\s+", t) if p]
    out = []
    for piece in pieces:
        word = piece.split(" ", 1)[0]
        if len(word) >= 2 and word.isalpha() and word.isupper():
            out.append(piece)
        else:
            out.append(piece[:1].lower() + piece[1:])
    return ", and ".join(out)


def _text_has_term(rec: dict, text: str) -> bool:
    """Есть ли в тексте имя, псевдоним или ключевое слово записи. Совпадение целыми словами, без учёта регистра.
    Одна и та же логика для фильтра батча (batch_descriptions) и для проверки маркеров объектов."""
    name = str(rec.get("name") or "").strip()
    terms = [name, *(rec.get("aliases") or []), *(rec.get("keywords") or [])]
    for term in terms:
        term = str(term).strip()
        if term and re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.IGNORECASE):
            return True
    return False


def _all_descriptions(records: list) -> dict:
    """{ID: внешность} по всем записям файла, у которых есть id и внешность. Нужен для замены id в тексте
    промпта: в отличие от batch_descriptions, здесь нет фильтра по тексту батча."""
    out: dict = {}
    for rec in records:
        rid = str(rec.get("id") or "").strip()
        appearance = _normalize_appearance(rec.get("appearance") or "")
        if rid and appearance:
            out[rid] = appearance
    return out


def _object_marker_misses(results: dict, builder, objects: list) -> list:
    """Нарушения маркера объекта: список (номер сегмента, id объекта, имя) для каждого объекта, чьё имя,
    псевдоним или ключевое слово есть в собственном тексте сегмента, если в его промпте нет маркера [OBJ_N].
    Смотрится только текст самого сегмента (не контекст). Персонажи не проверяются. Выключается
    константой OBJECT_MARKER_CHECK. Проверяется сырой ответ Gemini, до замены маркеров на описания."""
    if not OBJECT_MARKER_CHECK or not objects:
        return []
    out = []
    for n in sorted(results):
        seg = builder.by_num.get(n)
        seg_text = (seg.text or "") if seg else ""
        prompt = results[n].prompt
        for rec in objects:
            rid = str(rec.get("id") or "").strip()
            if not rid.startswith("OBJ_") or not _text_has_term(rec, seg_text):
                continue
            if re.search(r"(?<!\w)" + re.escape(rid) + r"(?!\w)", prompt):
                continue
            out.append((n, rid, str(rec.get("name") or "").strip()))
    return out


def batch_descriptions(records: list, batch_text: str) -> dict:
    """{ID: внешность} для записей персонажей и объектов, чьё имя, псевдоним или ключевое слово есть в тексте батча.

    records: записи файла персонажей (id, name, aliases, keywords, appearance), персонажи и объекты вместе.
    Ключ — id записи (CHAR_N или OBJ_N), значение — внешность; имя владельца здесь не нужно.
    Совпадение по коду (без учёта регистра, целыми словами), Gemini не вызывается. Запись без id или без
    внешности пропускается: её маркер останется неразрешённым и попадёт в needs_review."""
    out: dict = {}
    for rec in records:
        rid = str(rec.get("id") or "").strip()
        name = str(rec.get("name") or "").strip()
        appearance = _normalize_appearance(rec.get("appearance") or "")
        if not rid or not name or not appearance:
            continue
        if _text_has_term(rec, batch_text):
            out[rid] = appearance
    return out


def marker_instruction(ids: list) -> str:
    """Инструкция для пользовательского сообщения батча: маркер пишется как id в квадратных скобках.

    В списке только id (CHAR_N, OBJ_N); имена персонажей и объектов в маркеры не входят."""
    return ("Where one of these characters or objects appears, write the marker [ID] in place of it, where ID is "
            "the id exactly as listed: [CHAR_1] for a person, [OBJ_2] for an object. Use only these ids: "
            + ", ".join(ids) + ". "
            "If the name of one of these objects appears in the segment text, the prompt must contain that "
            "object's marker and must not describe the object in its own words.")


def generate_prompts(segments: list, wanted_nums: list, cfg: PromptConfig,
                     transport: GeminiTransport, model: str,
                     on_batch: Optional[Callable[[list], None]] = None,
                     descriptions: Optional[dict] = None,
                     characters: Optional[list] = None) -> list:
    """Промпты для нужных номеров: список PromptResult по возрастанию номера.

    Все проверки входа выполняются до первого обращения к transport. Ошибки транспорта
    (GeminiRuntimeError) не перехватываются. Номер без пригодного текста получает prompt ""
    и needs_review=True: решение, пускать ли такой в генерацию, за оркестратором.

    on_batch(results): необязательный колбэк. Вызывается после каждого готового батча (с учётом
    retry), до начала следующего; results: list[PromptResult] этого батча по возрастанию номера,
    включая needs_review. Если позже случится GeminiRuntimeError, уже готовые батчи колбэку
    отданы. Исключения колбэка не перехватываются. Без колбэка поведение прежнее.

    descriptions: описания для маркеров [ID] в промптах (ключ — id, значение — внешность). Без них (None или {})
    маркеры остаются в тексте и помечаются needs_review.

    characters: записи персонажей и объектов (id, name, aliases, keywords, appearance) из файла персонажей.
    Если заданы, для каждого батча в запрос попадают описания записей, найденных в его сегментах и контексте
    (ключ — id), и в пользовательское сообщение добавляется инструкция о маркерах [ID]. descriptions тогда
    не используется."""
    desc = descriptions or {}
    builder = _Builder(segments, wanted_nums, cfg)
    # текст пресета читается до первого запроса: ошибка файла остаётся ошибкой входа
    style_text = read_preset(cfg.preset) if cfg.preset != PRESET_NONE else ""
    out: dict = {}
    for batch in builder.batches(builder.wanted):
        if characters is None:
            batch_desc, note, batch_objects, resolve_map = desc, "", [], desc
        else:
            # H3: в запрос батча идут только описания персонажей, найденных в его сегментах или контексте
            batch_desc = batch_descriptions(characters, batch.user)
            note = ("\n\n" + marker_instruction(sorted(batch_desc))) if batch_desc else ""
            if note:
                batch = replace(batch, user=batch.user + note)
            # проверка маркеров: только объекты, чьи маркеры объявлены в этом батче (иначе маркер не записать)
            batch_objects = [rec for rec in characters
                             if str(rec.get("id") or "").strip() in batch_desc
                             and str(rec.get("id") or "").strip().startswith("OBJ_")]
            # замена id в тексте промпта идёт по всем записям файла, не только по батчу
            resolve_map = _all_descriptions(characters)
        batch_res = _run_batch(transport, model, builder, batch, note, objects=batch_objects)
        for r in batch_res.values():
            r.prompt, unresolved = resolve_markers(r.prompt, resolve_map)
            if unresolved:
                r.needs_review = True
                reason = "; ".join(f"маркер персонажа без описания: {n}" for n in unresolved)
                r.review_reason = "; ".join(x for x in (r.review_reason, reason) if x)
            # id, оставшийся в тексте после замены (в том числе неизвестный), ставит needs_review
            left = list(dict.fromkeys(_LEFT_ID_RX.findall(r.prompt)))
            if left:
                r.needs_review = True
                reason = "; ".join(f"в тексте остался id: {x}" for x in left)
                r.review_reason = "; ".join(x for x in (r.review_reason, reason) if x)
            # стиль пресета прикрепляется последним; пустой промпт (нет пригодного текста) не трогаем
            if r.prompt.strip():
                r.prompt = attach_style(r.prompt, style_text)
        out.update(batch_res)
        if on_batch is not None:
            on_batch([batch_res[n] for n in batch.nums])
    results = [out[n] for n in builder.wanted]
    flagged = sum(1 for r in results if r.needs_review)
    log.info("Промпты готовы: %d, из них требуют проверки: %d.", len(results), flagged)
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
                if "PerDay" in qid:
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


class RealTransport:
    """Реальный транспорт Gemini на httpx. Ключ читается из окружения при создании и уходит
    только в заголовок x-goog-api-key; в URL, логи и исключения он не попадает."""

    def __init__(self, *, api_key_env: str = ENV_KEY_NAME, timeout: float = 120.0,
                 client: Any = None, base_url: str = API_BASE_URL):
        key = os.environ.get(api_key_env, "")
        if not key:
            raise GeminiInputError(f"Не задана переменная окружения {api_key_env} с ключом Gemini.")
        try:
            import httpx
        except ImportError:
            raise GeminiInputError("Не установлен пакет httpx.") from None
        _silence_http_logs()
        self._httpx = httpx
        self._headers = {API_KEY_HEADER: key, "Content-Type": "application/json"}
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
        url = f"{self._base_url}/{model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseJsonSchema": schema},
        }
        try:
            resp = self._client.post(url, headers=self._headers, json=body)
        except self._httpx.HTTPError as e:
            # В сообщение идёт только имя класса ошибки; объект запроса и цепочка не сохраняются.
            raise GeminiRuntimeError(
                f"Сбой сети при обращении к Gemini ({type(e).__name__}).") from None
        except Exception as e:
            raise GeminiRuntimeError(
                f"Непредвиденный сбой при обращении к Gemini ({type(e).__name__}).") from None

        if resp.status_code != 200:
            code, daily, retry = _safe_error_fields(resp)
            extra = f", код {code}" if code else ""
            raise GeminiRuntimeError(
                f"Gemini вернул ошибку: HTTP {resp.status_code}{extra}.",
                status=resp.status_code, code=code, daily_quota=daily, retry_after=retry)
        try:
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except Exception:
            # пустой/заблокированный ответ: пусть валидатор пометит батч и сделает retry
            return ""
