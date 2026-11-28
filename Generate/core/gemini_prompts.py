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
    from .cues_loader import build_batch_cue_guide, format_segment_cue
    from .model_constants import PRESET_NONE, PROFILE_NAME_PATTERN
    from .frame_rules import rephrase_note  # правило для запрещённых типов кадра (шаг 4)
    from .prompt_check import (PROBLEM_CAMERA_MOTION, find_camera_motion, find_problems,  # проверка промптов (шаг 6)
                           make_problem, problem_ru, rework_remark,
                           CONTENT_HINT_BY_REASON, content_reasons, ethnicity_reasons, ethnicity_hits, RELAXABLE_RULES,
                           CONTENT_REASON_RU, PROBLEM_PEOPLE_COUNT, PROBLEM_ETHNICITY, people_hint, problem_code, problem_label,
                           RulesValues, PEOPLE_LIMIT, EMOTION_VALUES, SUBJECT_SOURCE_VALUES,
                           PROBLEM_MANY_OF_ONE, MANY_OF_ONE_VEHICLES_HINT_EN, animal_plurals_by_count,
                           PROBLEM_REF_TAG, PROBLEM_DUPLICATE_PROMPT, validate_allowed_shot_items,
                           PROBLEM_DANGLING_TEXT, PROBLEM_LETTERING_NOT_IN_SEGMENT, frame_has_lettering)
except ImportError:  # pragma: no cover
    from srt_parser import Segment  # type: ignore
    from cues_loader import build_batch_cue_guide, format_segment_cue  # type: ignore
    from model_constants import PRESET_NONE, PROFILE_NAME_PATTERN  # type: ignore
    from frame_rules import rephrase_note  # type: ignore
    from prompt_check import (PROBLEM_CAMERA_MOTION, find_camera_motion, find_problems,  # type: ignore
                          make_problem, problem_ru, rework_remark,
                          CONTENT_HINT_BY_REASON, content_reasons, ethnicity_reasons, ethnicity_hits, RELAXABLE_RULES,
                          CONTENT_REASON_RU, PROBLEM_PEOPLE_COUNT, PROBLEM_ETHNICITY, people_hint, problem_code, problem_label,
                           RulesValues, PEOPLE_LIMIT, EMOTION_VALUES, SUBJECT_SOURCE_VALUES,
                           PROBLEM_MANY_OF_ONE, MANY_OF_ONE_VEHICLES_HINT_EN, animal_plurals_by_count,
                           PROBLEM_REF_TAG, PROBLEM_DUPLICATE_PROMPT, validate_allowed_shot_items,
                           PROBLEM_DANGLING_TEXT, PROBLEM_LETTERING_NOT_IN_SEGMENT, frame_has_lettering)

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
# cartoon_photos (batch_passage): сколько целых предложений контекста даётся до и после батча (только смысл)
PASSAGE_CONTEXT_SENTENCES = 2

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

# Лимит референсов (слоты графа ComfyUI). Выше PromptConfig: поле ref_max берёт значение отсюда.
REF_MAX = 10


@dataclass
class PromptConfig:
    style_file: str                  # путь к файлу стиля models/<профиль>/prompts/style.md (из профиля)
    media: str                       # тип промпта: "video" или "image"; без значения по умолчанию
    max_prompt_chars: int = 1500
    # max_characters: число персонажей в кадре (герой и остальные: люди и животные) из profile.yaml; None — PEOPLE_LIMIT
    max_characters: Optional[int] = None
    batch_size: int = 20
    response_format: Any = field(default_factory=lambda: [{"segment_index": int, "prompt": str}])
    # пресет профиля: Generate/model_profiles/presets/prompts/<preset>.md; «none» = без пресета
    preset: str = PRESET_NONE
    # refs_enabled: ответ с полем refs (номера референсов), метки <imageN> проверяются; по умолчанию выключено
    refs_enabled: bool = False
    # rework_style_flag_without_refs: клип с любым флагом разбора (стиль, длина, мусор, этнос, метки) идёт в очередь
    # исправлений и без refs; по умолчанию выключено (значение задаёт профиль через вызывающую сторону)
    rework_style_flag_without_refs: bool = False
    # on_unfixed: «keep» — клип с нарушением остаётся на проверке (прежнее поведение); «fail» — прогон останавливается
    on_unfixed: str = "keep"
    # fail_fast: первая проблема батча останавливает прогон сразу: без повтора и без исправлений; по умолчанию выключено
    fail_fast: bool = False
    # raw_dir: каталог для сырого ответа батча с проблемой (только при fail_fast); None = файлы не пишутся
    raw_dir: Optional[str] = None
    # relaxed_rules: отключённые профилем правила из RELAXABLE_RULES; по умолчанию ничего не отключено
    relaxed_rules: frozenset = frozenset()
    # text_language: None — как раньше; «srt» — язык текста в кадре из SRT; иначе название языка из профиля
    text_language: Optional[str] = None
    # frame_language: английское название языка текста в кадре; None, пока SRT не разобран
    frame_language: Optional[str] = None
    # ref_max: число допустимых референсов профиля (число ref_N.png, не больше REF_MAX); по умолчанию REF_MAX
    ref_max: int = REF_MAX
    # allowed_refs: разрешённые сочетания номеров refs (отсортированные кортежи) из profile.yaml; None — без проверки
    allowed_refs: Optional[frozenset] = None
    # niche_preset_file: prompts/preset.md папки ниши, если пресет «none»; None — файла нет
    niche_preset_file: Optional[str] = None
    # rules: значения prompts/rules.yaml папки ниши или модели; по умолчанию — значения прежнего поведения
    rules: RulesValues = field(default_factory=RulesValues)
    # niche: имя ниши (папка профиля) для имён файлов сырых ответов; задаёт профиль; пусто — файлы не пишутся
    niche: str = ""
    # dump_raw_responses: диагностика — сырой ответ каждого запроса (батч, повтор формата, исправление) в raw_dir;
    # по умолчанию выключено; разбор и тексты запросов от флага не зависят
    dump_raw_responses: bool = False
    # batch_passage: батч одним текстом (целые предложения контекста, строка на сегмент) и поле setting;
    # только cartoon_photos (profile.yaml prompt_batch_form: passage); по умолчанию выключено
    batch_passage: bool = False
    # fix_batch_size: размер пакета исправлений профиля (batch_size профиля задаёт и обычные батчи); None — FIX_BATCH_SIZE
    fix_batch_size: Optional[int] = None
    # I2V-режим: словарь {номер_сегмента: фото-промпт без хвоста preset.md} и {номер_сегмента: длительность_сек}
    source_photo_prompts: Optional[dict] = None
    segment_durations: Optional[dict] = None
    # Опциональные монтажные подсказки (montage_cues.json): словарь {номер_сегмента: SegmentCue}
    segment_cues: Optional[dict] = None

    def __post_init__(self) -> None:
        # Русский: fix_batch_size, если задан, — целое не меньше 1 (bool не принимается)
        if self.fix_batch_size is not None and (
                isinstance(self.fix_batch_size, bool) or not isinstance(self.fix_batch_size, int)
                or self.fix_batch_size < 1):
            raise GeminiInputError(
                f"Ошибка входных данных: fix_batch_size должен быть целым числом не меньше 1, а не {self.fix_batch_size!r}.")
        # max_characters задаётся в rules.yaml; поле cfg.max_characters — то же значение (сверяется)
        if self.max_characters is None:
            self.max_characters = self.rules.max_characters
        elif self.rules.max_characters is None:
            self.rules.max_characters = self.max_characters
        elif self.max_characters != self.rules.max_characters:
            raise GeminiInputError(
                f"Ошибка входных данных: max_characters в конфиге ({self.max_characters}) "
                f"не совпадает с rules.yaml ({self.rules.max_characters}).")


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

# Русский: полноширинные завершители CJK («。！？») — разбиение японских и китайских SRT на предложения
SENTENCE_END_CHARS = ".?!\u2026\u3002\uff01\uff1f"
SENTENCE_CLOSERS = "\"'\u00bb\u201d\u2019)]"
SENTENCE_OPENERS = "\"'\u00ab\u201c\u2018\u201e(["
NO_BREAK_ABBREVIATIONS = frozenset({
    "Mr.", "Mrs.", "Ms.", "Dr.", "St.", "No.", "Jr.", "Sr.", "vs.",
    # French
    "M.", "Mme.", "Mlle.", "MM.", "av.", "apr.", "env.", "cf.", "Ste.",
    # German
    "Hr.", "Fr.", "Prof.", "z.B.", "bzw.", "ca.",
    # Spanish
    "Sra.", "Srta.", "Ud.",
    # Russian
    "ул.", "см.",
})
LOWERCASE_CONTINUES_ABBREVIATIONS = frozenset({
    "etc.",
    # French
    "J.-C.",
    # Russian
    "г.", "гг.", "т.д.", "т.п.", "др.",
})


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
    return text.rstrip(SENTENCE_CLOSERS + " \u00a0\u202f")


def _ends_sentence(text: str, next_text: Optional[str]) -> bool:
    """Заканчивается ли предложение на этом сегменте (text непустой, next_text — ближайший
    следующий непустой сегмент или None)."""
    core = _strip_closers(text)
    if not core or core[-1] not in SENTENCE_END_CHARS:
        return False
    nxt = next_text.lstrip(SENTENCE_OPENERS + " ") if next_text else ""

    if core.endswith("\u2026") or core.endswith("..."):
        return next_text is None or nxt[:1].isupper()
    if core[-1] in "?!\uff01\uff1f\u3002":
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


def _format_segment_block(
    seg: Segment,
    sentences: list,
    by_segment: dict,
    allowed_items=frozenset(),
    *,
    source_photo: Optional[str] = None,
    duration_sec: Optional[float] = None,
    segment_cues: Optional[dict] = None,
) -> str:
    """Блок одного сегмента для пользовательского запроса. Номер идёт только как служебное
    поле segment_index (его модель вернёт в ответе), в смысловой текст он не входит.
    allowed_items: разрешённые типы кадра ниши (rules.yaml allowed_shot_items) для правила перефразирования.
    source_photo / duration_sec: опциональные данные I2V (фото-промпт без хвоста стиля и длительность клипа)."""
    sen = by_segment[seg.num]
    header = f"segment_index: {seg.num}"
    if segment_cues:
        cue_str = format_segment_cue(segment_cues.get(seg.num))
        if cue_str:
            header += f" | Cue: {cue_str}"
    lines = [header]
    if duration_sec is not None:
        lines.append(f"Duration: {duration_sec:.2f}s")
    if source_photo:
        lines.append(f"Source photo (animate ONLY what is already drawn here): {source_photo}")
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
    note = rephrase_note(text, allowed_items)
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


def _schema_prop(t: Any) -> dict:
    """Свойство схемы. Скаляр — как раньше; список из одного целого [int] (например refs) — массив целых."""
    if isinstance(t, list) and len(t) == 1 and t[0] is int:
        return {"type": "array", "items": {"type": "integer"}}
    return {"type": _schema_type(t)}


def response_schema(response_format: Any) -> dict:
    """JSON Schema массива объектов из примера формата: [{"segment_index": int, "prompt": str}]."""
    if (not isinstance(response_format, list) or len(response_format) != 1
            or not isinstance(response_format[0], dict)):
        raise GeminiInputError("response_format должен быть списком из одного объекта-примера.")
    item = response_format[0]
    if "segment_index" not in item or "prompt" not in item:
        raise GeminiInputError("В response_format обязательны поля segment_index и prompt.")
    props = {name: _schema_prop(t) for name, t in item.items()}
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


def read_niche_preset(path: str) -> str:
    """Текст файла пресета ниши (prompts/preset.md): USER_ONLY вырезан, как у именованного пресета."""
    file = Path(path)
    try:
        text = strip_user_only(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        raise GeminiInputError(f"Пресет ниши не удалось прочитать: {file}.") from None
    if not text:
        raise GeminiInputError(f"Пресет ниши пуст: в файле {file} нет текста.")
    return text


def preset_text(preset: str, niche_preset_file: Optional[str] = None) -> str:
    """Текст пресета для промптов: один путь для всех. Имя пресета задано — читается он, как прежде;
    «none» и есть файл пресета ниши — текст файла; иначе пусто."""
    if preset != PRESET_NONE:
        return read_preset(preset)
    if niche_preset_file:
        return read_niche_preset(niche_preset_file)
    return ""


def attach_style(prompt: str, style_text: str) -> str:
    """Дописывает текст стиля в конец готового промпта через «. ». Пустой стиль — промпт без изменений."""
    if not style_text:
        return prompt
    # модель уже дописала стиль в конец промпта: второй раз не добавляем (сравнение без точки на конце)
    body = prompt.rstrip()
    style_core = style_text.strip().rstrip(".")
    if style_core and body.rstrip(".").endswith(style_core):
        return body
    return prompt.rstrip().rstrip(".") + ". " + style_text


# ---------------------------------------------------------------------------
# Правила содержания из prompts/rules.yaml: только значения, английские ключи, без PyYAML.
# Строки с ключом подставляются из RulesValues; строки без ключа — фиксированные (safety и общие правила).
# ---------------------------------------------------------------------------

RULES_FILE_NAME = "rules.yaml"
RULES_KEYS = ("max_characters", "animals_count", "max_people_per_image", "small_objects", "speech", "weapons",
              "close_faces", "static_frame", "emotion", "subject_source", "horses_vehicles_animals_max", "children",
              "scene_motion_wording", "object_sets", "allowed_shot_items", "forbid_thick_vector", "sketch_style",
              "dangling_text_check", "fire_hands")
_RULES_INT_KEYS = ("max_characters", "max_people_per_image", "horses_vehicles_animals_max")
_RULES_BOOL_KEYS = ("animals_count", "static_frame", "forbid_thick_vector", "sketch_style", "dangling_text_check")
_RULES_SWITCH_FIELDS = {"small_objects": "small_objects_allowed", "speech": "speech_allowed",
                        "weapons": "weapons_allowed", "close_faces": "close_faces_allowed",
                        "object_sets": "object_sets_forbidden", "fire_hands": "fire_hands_allowed"}
SCENE_MOTION_VALUES = ("keep", "drop", "allow")
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
                 9: "nine", 10: "ten"}
_EMOTION_SENTENCE = {
    "posture": "Show emotion only through posture.",
    "posture_and_action": "Show emotion only through posture and visible actions.",
    "action": "Show emotion only through visible actions.",
}

# фиксированные строки содержания (ключа в rules.yaml нет)
_SHOW_ONLY_LINE = ("- Show only the people, animals and objects that the segment text names or directly implies. "
                   "Add no extra characters, animals, vehicles or props.\n")
_CHILDREN_LINE = ("- Never show children, babies, boys, girls, teenagers or young people. If the segment mentions "
                  "them, show adults or no people, and keep the meaning through the setting.\n")
_NUDITY_LINE = "- Every person wears named clothing. Never show anyone naked, shirtless or undressed.\n"
# Русский: cartoon_photos (max_characters): Type B носит одежду по имени, Type A (герой-палочка) — только чёрное туловище
_TYPE_AB_CLOTHING_LINE = ("- Every cartoon adult (Type B) wears their named clothing outfit; the stick-body main character "
                          "(Type A) has a plain solid-black stick torso with at most a situational hat, apron or backpack. "
                          "Never show anyone naked, shirtless or undressed.\n")
_SMALL_OBJECTS_LINE = ("- Show only large, clearly visible objects. No small objects and no hand actions with small things "
                       "(needles, thread, abacus, beads, cloth, bundles, cups, dung cakes; no stacking, stitching, tucking "
                       "or threading).\n")
_OBJECT_SET_LINE = ("- Show one object, or two or three objects of one kind. "
                    "Never show stacks, piles, rows or lines of objects.\n")
_SPEECH_FORBIDDEN_START = ("- No speech of any kind: no speaking, shouting, laughing, crying, singing, conversation, dialogue, "
                           "gestures for talking or moving lips. ")
_FIRE_LINE = "- Keep hands and fingers away from fire, flames, candles and lanterns.\n"
_CHAOS_LINE = ("- No battles, charges, gallops in formation, gunfire, explosions, smoke puffs or dust clouds. "
               "Show calm, static or slowly moving scenes.\n")
_WEAPONS_LINE = "- No weapons in the frame: no firearms, sabres, swords, spears or knives.\n"
_ABSENT_LINE = ("- When the text says something is absent (for example no saddle), describe the visible state "
                "instead (for example a bare horse back).\n")
_ETHNICITY_LINE = ("- Derive the place, period and people's appearance from the segment text and its setting. Do not add "
                   "a nationality, ethnicity or skin tone the text does not mention.\n")
_CLOSE_FACES_LINE = ("- Never show close-ups, extreme close-ups or portraits. Show people from a medium or wide "
                     "distance, with faces small in the frame.\n")
# Для профилей с max_characters (cartoon_photos): крупные планы рук, ног, предметов, еды, растений и врезки
# разрешены (как в style.md и prompt_check.py); запрещены только крупные планы и портреты лиц людей.
_CLOSE_FACES_OBJECTS_OK_LINE = ("- Never show close-ups or portraits of human faces. Show people from a medium or wide "
                                "distance, with faces small in the frame; close-ups of hands, feet, objects, food, "
                                "plants and cutaways are allowed.\n")
_MOOD_LINE = ("- Show moods and ideas only through visible physical cues. Never use mood or emotion "
              "words such as thoughtful, pensive, solemn, serene, tense, mysterious, majestic, epic, "
              "cinematic, sad, lonely, or phrases like \"quiet stillness\".\n")
_OUTFIT_LINE = ("- Give each person one consistent outfit; do not mix conflicting garments such as a "
                "suit and an overcoat unless the layering is stated clearly.\n")
_MEDIUM_LINE_VIDEO = ("- Describe only the scene and its motion. Never name a medium, rendering method, film look, "
                      "or camera or lens equipment; camera movement is the only camera wording allowed.\n")
_MEDIUM_LINE_IMAGE = ("- Describe only the scene and its motion. Never name a medium, rendering method, film look, "
                      "or camera or lens equipment; camera or lens equipment is forbidden.\n")
# scene_motion_wording drop: те же слова без «and its motion», остальная строка без изменений
_MEDIUM_LINE_IMAGE_STILL = _MEDIUM_LINE_IMAGE.replace(" and its motion", "", 1)
_PAPER_LINE = ("- No paper, papers, letters, passports, forms or documents of any kind in the frame, including stacks, "
               "sheets and folded paper. Show the place or the activity instead, without readable text.\n")
# Нишы с разрешёнными бумажными типами кадра (rules.yaml allowed_shot_items): если разрешены все семь имён
# набора ниже, запрет на бумагу, письма, документы и листы снимается. Запреты на паспорта, формы, флаги,
# ID-карты, гербы, эмблемы и пергамент остаются. Без полного набора строка _PAPER_LINE не меняется.
_PAPER_ALLOWED_ITEMS = frozenset({"paper", "sheet of paper", "blank sheet", "document", "newspaper", "letter",
                                  "spreadsheet"})
_PAPER_LINE_PAPER_ALLOWED = ("- No passports, forms, flags, ID cards, coats of arms, emblems or parchment of any kind "
                             "in the frame. Paper items such as letters, documents or sheets show only the short text "
                             "from the Text line. Show the place or the activity instead.\n")
_CONTRAST_LINE = ("- When the text contrasts two things (\"not X, but Y\", \"not X, rather Y\"), show only Y, as a visible "
                  "action or scene. Never describe X and never use a negation to refer to it.\n")
_ASPECT_LINE = ("- No aspect ratios, frame sizes, widescreen or framing terms, and no blank sheets, sheets of paper or "
                "sheets of parchment. Describe the visible scene only, without any format or sheet of writing material.\n")
# Нишы с разрешёнными листами (blank sheet и sheet of paper в rules.yaml allowed_shot_items): запрет на эти листы снят,
# запрет на 16:9/widescreen и на пергамент остаётся. Без обоих имён строка _ASPECT_LINE не меняется.
_SHEETS_ALLOWED_ITEMS = frozenset({"blank sheet", "sheet of paper"})
_ASPECT_LINE_SHEETS_ALLOWED = ("- No aspect ratios, frame sizes, widescreen or framing terms, and no sheets of parchment. "
                               "Describe the visible scene only, without any format or parchment.\n")
_ONE_CAMERA_MOVE_LINE = ("- Exactly one camera move. Finish the camera sentence with how the frame looks when "
                         "the move ends (for a static camera: what the frame holds on).\n")
_NO_CAMERA_MOVE_LINE = ("- Never describe camera movement: no pans, zooms, tracks, dolly, tilts, pushes, pulls, "
                        "or any motion of the camera or the frame. Describe the single still frame as a finished state.\n")
_FINAL_CHECK_LINE = ("- Before returning, check each prompt against the self-check checklist at the end of "
                     "the style guide and rewrite any prompt that fails it.")
_SUBJECT_TEXT_ONLY_LINE = ("Objects and on-frame text come from the Text line only; the Sentence, Before and After lines "
                           "are for understanding the meaning.")


# cartoon_photos (batch_passage): правило разнообразия сеттингов внутри батча; заменяет финальную самопроверку
# Фраза «must use different settings» сохранена (её проверяет тест); добавлены визуальный режим и композиция.
_SETTING_VARIETY_LINE = ("- Neighbouring clips in the same batch must use different settings, and must also differ in visual "
                         "mode, shot framing, focal subject and composition: alternate wide scenes, close-ups of hands or "
                         "objects, overhead views, cutaways, balance scales or diagrams, split comparisons and character "
                         "reactions. Never repeat the same visual setup or a centred stickman pose on adjacent segments, "
                         "even with a renamed place.")
# Подсказка батча: соседние сегменты одного предложения показывают разные стороны одной темы
_SAME_SENTENCE_NOTE = ("Several segments of this batch come from the same sentence: give each of them a different visual "
                       "facet of that sentence (for example scale, an interior cutaway, a close-up of hands, a comparison, "
                       "a diagram or a consequence), never one layout with renamed backdrops.")
# фрагмент без визуального контекста: до стольких слов или без конечного знака препинания — нужно целое предложение
_FRAGMENT_NOTE_MAX_WORDS = 5


def rules_error(source: str, text: str) -> GeminiInputError:
    """Ошибка rules.yaml по-русски: название файла и текст про ключ."""
    return GeminiInputError(f"Ошибка входных данных: в {source} {text}")


def _rule_scalar(key: str, value: str, source: str):
    """Значение одного ключа rules.yaml. Неверный вид — ошибка по-русски с именем ключа."""
    if key in _RULES_INT_KEYS:
        try:
            number = int(value)
        except ValueError:
            number = 0
        if number < 1:
            raise rules_error(source, f"ключ {key} должен быть целым числом не меньше 1, получено «{value}».")
        return number
    if key in _RULES_BOOL_KEYS:
        if value not in ("true", "false"):
            raise rules_error(source, f"ключ {key} должен быть true или false, получено «{value}».")
        return value == "true"
    if key == "allowed_shot_items":
        # разрешённые типы кадра через запятую; неизвестное имя — ошибка с перечнем допустимых
        names = [x.strip() for x in value.split(",") if x.strip()]
        try:
            return validate_allowed_shot_items(names)
        except ValueError as exc:
            raise rules_error(source, f"ключ allowed_shot_items: {exc}.") from exc
    if key == "object_sets":
        # обратная полярность: forbidden включает проверку наборов предметов (поле object_sets_forbidden)
        if value not in ("allowed", "forbidden"):
            raise rules_error(source, f"ключ object_sets должен быть allowed или forbidden, получено «{value}».")
        return value == "forbidden"
    if key in _RULES_SWITCH_FIELDS:
        if value not in ("allowed", "forbidden"):
            raise rules_error(source, f"ключ {key} должен быть allowed или forbidden, получено «{value}».")
        return value == "allowed"
    if key == "children":
        if value != "forbidden":
            raise rules_error(source, "ключ children фиксирован и не настраивается: допустимо только forbidden.")
        return None
    if key == "scene_motion_wording":
        if value not in SCENE_MOTION_VALUES:
            raise rules_error(source, f"ключ scene_motion_wording должен быть keep, drop или allow; получено «{value}».")
        return value
    choices = EMOTION_VALUES if key == "emotion" else SUBJECT_SOURCE_VALUES
    if value not in choices:
        raise rules_error(source, f"ключ {key} должен быть одним из: {', '.join(choices)}; получено «{value}».")
    return value


def parse_rules_text(text: str, source: str) -> RulesValues:
    """Разбор prompts/rules.yaml: только строки вида key: value (как в profile.yaml, без вложенности)."""
    fields = {}
    seen = set()
    for number, raw in enumerate(text.splitlines(), 1):
        body = raw.split("#", 1)[0].rstrip()
        if not body.strip():
            continue
        if body[0] in " \t":
            raise rules_error(source, f"строка {number}: вложенные значения не поддерживаются, нужен вид key: value.")
        key, sep, value = body.partition(":")
        if not sep:
            raise rules_error(source, f"строка {number}: нужен вид key: value.")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key not in RULES_KEYS:
            raise rules_error(source, f"неизвестный ключ «{key}». Допустимые ключи: {', '.join(RULES_KEYS)}.")
        if key in seen:
            raise rules_error(source, f"ключ «{key}» повторяется.")
        seen.add(key)
        parsed = _rule_scalar(key, value, source)
        if parsed is None:
            continue
        fields[_RULES_SWITCH_FIELDS.get(key, key)] = parsed
    return RulesValues(**fields)


def load_rules(path) -> RulesValues:
    """Читает prompts/rules.yaml папки ниши или модели. Файла нет — значения по умолчанию (прежнее поведение)."""
    rules_file = Path(path)
    if not rules_file.is_file():
        return RulesValues()
    return parse_rules_text(rules_file.read_text(encoding="utf-8"), str(rules_file))


def _count_word(n: int) -> str:
    return _NUMBER_WORDS.get(n, str(n))


def _people_line(media: str, max_characters: Optional[int], rules: RulesValues) -> str:
    if max_characters is None:
        limit = (rules.max_people_per_image if rules.max_people_per_image is not None
                 else PEOPLE_LIMIT.get(media, PEOPLE_LIMIT["video"]))
        phrase = "one person" if limit == 1 else f"at most {_count_word(limit)} people"
        return (f"- Show {phrase}, each named once and described distinctly (age, clothing colour, hair). "
                "Never show a crowd, a group, a row, a line or several people.\n")
    scope = "people and animals together" if rules.animals_count else "people only"
    return ("- Show at most " + str(max_characters) + " characters in total, " + scope + ", "
            "the main hero included. Describe every character other than the main hero fully. "
            "Never show a crowd, a group, a row, a line or several people.\n")


def _horses_line(limit: int) -> str:
    phrase = "one or two" if limit == 2 else _count_word(limit)
    return f"- Show at most {phrase} horses, vehicles or animals, each described distinctly. No herds or rows.\n"


def _speech_line(rules: RulesValues) -> str:
    emotion = _EMOTION_SENTENCE[rules.emotion]
    if rules.speech_allowed:
        return f"- {emotion}\n"
    return _SPEECH_FORBIDDEN_START + emotion + "\n"


def _content_rules(media: str, max_characters: Optional[int] = None, rules: Optional[RulesValues] = None) -> str:
    """Правила содержания кадра: строки из rules.yaml и фиксированные строки.
    Те же значения читает prompt_check.py: нарушение уходит на повтор, а не в GPU."""
    rv = rules or RulesValues()
    text = (
        _SHOW_ONLY_LINE
        + _people_line(media, max_characters, rv)
        + _CHILDREN_LINE
        + (_NUDITY_LINE if max_characters is None else _TYPE_AB_CLOTHING_LINE)
        + _horses_line(rv.horses_vehicles_animals_max)
        + ("" if rv.small_objects_allowed else _SMALL_OBJECTS_LINE)
        + (_OBJECT_SET_LINE if rv.object_sets_forbidden else "")
        + _speech_line(rv)
        + ("" if rv.fire_hands_allowed else _FIRE_LINE)
        + _CHAOS_LINE
        + ("" if rv.weapons_allowed else _WEAPONS_LINE)
        + _ABSENT_LINE
        + _ETHNICITY_LINE
    )
    if media == "image" and not rv.close_faces_allowed:
        # Профили с max_characters разрешают крупные планы предметов; прочие — прежняя строка без изменений.
        text += _CLOSE_FACES_LINE if max_characters is None else _CLOSE_FACES_OBJECTS_OK_LINE
    return text


_BATCH_TEXT_FRAMING = (
    "Over 90 percent of frames are completely text-free pure visual illustrations. Decide each segment independently "
    "and do not copy a text pattern from earlier items in the batch. Strictly forbid all floating or overlaid text "
    "carriers: never place label cards, title plaques, accent plaques, framed plaques, header bands, pointer-arrow "
    "labels, number cards or standalone text slides in any frame. Only rare physical diegetic in-world signs "
    "(such as a storefront shop sign, wooden harbor sign or street plaque mounted on a building or post) are allowed "
    "when the physical setting genuinely calls for it. Never split an incomplete modifier across segments: a lone "
    "adjective without its noun is forbidden. ")


def _output_language_sentence(cfg: PromptConfig) -> str:
    """Предложение о языке вывода для Task-блока. Без relaxed_rules и text_language — прежняя фраза.

    frame_text: текст в кадре пишется на языке из следующего правила; text_language: язык текста в кадре.
    Нет языка при заданном text_language — ошибка, а не тихое «English» (см. profile.yaml, text_language).
    Русский: русский пример (ЗЕМЛЯНАЯ ГРУША) выводится только при frame_language == Russian.
    """
    text = ""
    if "frame_text" in cfg.relaxed_rules:
        text += "Text that appears inside the frame is written in the on-frame language named in the next rule. "
    # незаконченный текст в кадре: одна фраза-инструкция модели только при rules dangling_text_check: true; иначе текст прежний
    # cartoon_photos (batch_passage): текст в кадре только из строки Text; у прочих профилей фраза прежняя
    frame_source = "Text line" if cfg.batch_passage else "subtitle line"
    if cfg.rules.dangling_text_check:
        if cfg.batch_passage:
            # Русский: рамка против потока табличек стоит до правил оформления
            text += _BATCH_TEXT_FRAMING
        russian = cfg.frame_language == "Russian"
        example = (" (for example, write ЗЕМЛЯНАЯ ГРУША for the segment words земляной грушей)" if russian else "")
        lead = "For inflected on-frame languages (such as Russian), " if russian else "For inflected on-frame languages, "
        text += ("On-frame text must be a complete phrase of 1 to 4 words copied exactly from the " + frame_source
                 + " of the current segment only, except for the nominative rule below. The Before and After context is "
                 "for understanding the meaning only; never copy words from it into on-frame text. Never end the text on a "
                 "preposition, article, conjunction, auxiliary verb or the middle of a sentence. If the current segment "
                 "has no complete phrase to copy, write no on-frame text for that frame. "
                 + lead + "write each on-frame phrase in its clean nominative singular or dictionary form, built only "
                 "from words that appear in the " + frame_source + " of the current segment" + example + ". Change only "
                 "the grammatical case or number; never invent, translate or add words, and never take words from the "
                 "Before or After lines. ")
    if cfg.source_photo_prompts is not None:
        if "frame_text" in cfg.relaxed_rules:
            return ("Keep any existing on-frame text from the Source photo stationary and sharp; "
                    "always write the animation prompt in English.")
        return "Always write the animation prompt in English."
    if cfg.text_language is None:
        return text + "Always write the output in English."
    if not cfg.frame_language:
        raise GeminiInputError(
            "Ошибка входных данных: text_language задан, но язык текста в кадре не определён (нет разобранного SRT).")
    return text + (f"Write all prompt text in English, except text that appears inside the frame, "
                   f"which must be written in {cfg.frame_language}.")


def _allowed_combo_lines(allowed) -> str:
    """Строки разрешённых сочетаний для системной инструкции: по одному на строку, номера через «, ».
    Берутся из тех же кортежей, что проверяет _combo_bad (cfg.allowed_refs); порядок — по возрастанию кортежей."""
    return "\n".join(", ".join(str(n) for n in combo) for combo in sorted(allowed))


def _contract_object(cfg: PromptConfig) -> str:
    """Описание одного объекта ответа в системной инструкции. У batch_passage есть поле setting (перед prompt)."""
    if cfg.batch_passage:
        return ("{\"segment_index\": <the integer given for that segment>, \"setting\": <a short English "
                "phrase in the form <visual mode>: <place or subject>, for example macro: hands peeling tuber, "
                "cutaway: soil roots, scale: mass vs calories or wide: potato rows>, \"prompt\": <string>}. ")
    return "{\"segment_index\": <the integer given for that segment>, \"prompt\": <string>}. "


def build_system_instruction(cfg: PromptConfig) -> str:
    style = strip_user_only(Path(cfg.style_file).read_text(encoding="utf-8"))
    rules = cfg.rules
    # строки art_medium и frame_text: при отключении правила строки нет; без отключения текст тот же, что раньше
    if cfg.media == "video":
        kind = "image-to-video animation prompt" if cfg.source_photo_prompts is not None else "text-to-video prompt"
        camera_end = _ONE_CAMERA_MOVE_LINE
        medium = "" if "art_medium" in cfg.relaxed_rules else _MEDIUM_LINE_VIDEO
    elif cfg.media == "image":
        # static_frame true: один неподвижный кадр и запрет движения камеры; false — без них (rules.yaml)
        kind = "text-to-image prompt, one still frame" if rules.static_frame else "text-to-image prompt"
        camera_end = _NO_CAMERA_MOVE_LINE if rules.static_frame else ""
        image_line = _MEDIUM_LINE_IMAGE if rules.scene_motion_wording in ("keep", "allow") else _MEDIUM_LINE_IMAGE_STILL
        medium = "" if "art_medium" in cfg.relaxed_rules else image_line
    else:
        raise GeminiInputError(f"media должен быть «video» или «image», получено: {cfg.media!r}.")
    # бумажные типы кадра: строка без запретов на бумагу только если разрешены все имена _PAPER_ALLOWED_ITEMS
    paper_line = _PAPER_LINE_PAPER_ALLOWED if _PAPER_ALLOWED_ITEMS <= rules.allowed_shot_items else _PAPER_LINE
    paper = "" if "frame_text" in cfg.relaxed_rules else paper_line
    aspect = _ASPECT_LINE_SHEETS_ALLOWED if _SHEETS_ALLOWED_ITEMS <= rules.allowed_shot_items else _ASPECT_LINE
    camera_rules = medium + _CONTRAST_LINE + paper + aspect + camera_end
    # subject_source text_only: объекты и текст в кадре только из строки Text; у text_and_context строки нет
    subject_line = ("\n" + _SUBJECT_TEXT_ONLY_LINE) if rules.subject_source == "text_only" else ""
    parts = [
        "# Task\n"
        "You receive subtitle segments of a narrated video. For every segment write ONE "
        + kind + ", in English, that visually illustrates what the segment says, "
        "following the style guide below. Use the Sentence, Before and After lines only to "
        "understand the meaning; the >>marked<< part is the passage to illustrate. The source subtitle text may be in any language. "
        + _output_language_sentence(cfg) + subject_line + "\n\n"
        "# Output contract\n"
        "Return a JSON array with exactly one object per requested segment: "
        + _contract_object(cfg)
        + "No other segments, no duplicates, no markdown, no code fences. "
        "Never write the segment number, the word \"segment\" or any numbering into the prompt text. "
        f"Each prompt must be at most {cfg.max_prompt_chars} characters.\n\n"
        "# Hard rules for every prompt\n"
        + _MOOD_LINE
        + "- No negations (no, not, without, never, nothing), no quotation marks, "
        "no keyword lists (4k, masterpiece, trending).\n"
        + camera_rules
        + _OUTFIT_LINE
        + _content_rules(cfg.media, cfg.max_characters, rules)
        # cartoon_photos (batch_passage): строка разнообразия сеттингов вместо финальной самопроверки
        + (_SETTING_VARIETY_LINE if cfg.batch_passage else _FINAL_CHECK_LINE),
        "# Style guide\n" + style,
    ]
    if cfg.refs_enabled:
        # контракт с референсами (только профили с refs_enabled): номера и метки должны совпадать
        if cfg.allowed_refs is None:
            # профили без allowed_refs: текст прежний, байт в байт
            refs_rule = "naming the reference images used for that segment (an empty list if none). "
            no_refs_rule = "A segment with no refs has no <imageN> tag."
            # текст метки без allowed_refs прежний, байт в байт
            tag_rule = ("In the prompt, label the reference images with the tags <image1>, <image2> … <imageN>, "
                        "where N is the number of entries in refs; the tag <imageK> is the K-th entry of the sorted refs list. ")
        else:
            # профили с allowed_refs: сочетания берутся из тех же кортежей, что проверяет _combo_bad
            refs_rule = ("naming the reference images used for that segment. "
                         "refs must be exactly one of the allowed combinations listed below, and never an empty list. "
                         "Allowed combinations, one per line:\n"
                         + _allowed_combo_lines(cfg.allowed_refs) + "\n")
            no_refs_rule = "Every segment has at least one reference, so every segment has at least one <imageN> tag."
            # Русский: метки с allowed_refs — по номеру референса (как в style.md), а не по позиции в refs;
            # нормализатор переводит номера в позиции до проверок. Только профили с allowed_refs (cartoon_photos).
            tag_rule = ("In the prompt, label each reference image with one tag for each entry in refs, "
                        "written with the position of that image in the sorted refs list: <image1> for the first entry, <image2> for the second. Tags name positions, never reference numbers. "
                        "List the tags from the lowest reference number to the highest. ")
        parts[0] += (
            "\n\n# Reference images\n"
            f"Each object also has \"refs\": a JSON list of distinct integers from 1 to {cfg.ref_max}, sorted ascending, "
            + refs_rule
            + tag_rule
            + "Use each tag at least once and no other tags. " + no_refs_rule)
    # пресет в системную инструкцию не входит: здесь только проверка имени и файла, текст дописывает generate_prompts
    if cfg.preset != PRESET_NONE:
        read_preset(cfg.preset)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Построение запросов
# ---------------------------------------------------------------------------

def _passage_context(sentences: list, first: int, last: int) -> tuple:
    """Два целых предложения до батча и два после. first и last — номера предложений первого и последнего
    сегмента батча; сами предложения батча сюда не попадают (их номера лежат внутри [first, last]).
    Пустые предложения пропускаются."""
    before: list = []
    i = first - 2                      # индекс предложения с номером first-1
    while i >= 0 and len(before) < PASSAGE_CONTEXT_SENTENCES:
        if sentences[i].text:
            before.append(sentences[i])
        i -= 1
    before.reverse()
    after: list = []
    i = last                           # индекс предложения с номером last+1
    while i < len(sentences) and len(after) < PASSAGE_CONTEXT_SENTENCES:
        if sentences[i].text:
            after.append(sentences[i])
        i += 1
    return before, after


def _needs_full_sentence(text: str) -> bool:
    """Фрагмент без визуального контекста: короткий (до _FRAGMENT_NOTE_MAX_WORDS слов) или без конечного знака."""
    core = _strip_closers(text)
    return len(text.split()) <= _FRAGMENT_NOTE_MAX_WORDS or not core or core[-1] not in SENTENCE_END_CHARS


def _shares_sentence(chunk: list, by_segment: dict) -> bool:
    """Есть ли два соседних по номеру сегмента батча из одного предложения."""
    nums = sorted(chunk)
    return any(by_segment[a].number == by_segment[b].number for a, b in zip(nums, nums[1:]))


def _format_passage(chunk: list, by_num: dict, sentences: list, by_segment: dict,
                    allowed_items=frozenset(), segment_cues: Optional[dict] = None) -> str:
    """Батч одним текстом для cartoon_photos: контекст до и после (только смысл, целые предложения),
    затем по одной строке на сегмент: segment_index, Text и Sentence с маркером >>...<<.
    Контекст не повторяется внутри батча. Правило перефразирования добавляется один раз, если нужно."""
    first = min(by_segment[n].number for n in chunk)
    last = max(by_segment[n].number for n in chunk)
    before, after = _passage_context(sentences, first, last)
    lines: list = []
    if before:
        lines.append("Before (context only): " + " ".join(x.text for x in before))
    notes: list = []
    for n in chunk:
        seg = by_num[n]
        sen = by_segment[n]
        text = " ".join((seg.text or "").split())
        if text:
            a, b = sen.spans[n]
            marked = f"{sen.text[:a]}>>{sen.text[a:b]}<<{sen.text[b:]}"
            line = f"segment_index: {n} | Text (what this segment says): {text} | Sentence: {marked}"
            if _needs_full_sentence(text):
                line += (" | Note: this is an incomplete fragment; base the visual on the whole Sentence, "
                         "not on these words alone.")
        else:
            line = f"segment_index: {n} | Text (what this segment says): (silence / no text)"
            if sen.text:
                line += f" | Sentence: {sen.text}"
        if segment_cues:
            cue_str = format_segment_cue(segment_cues.get(n))
            if cue_str:
                line += f" | Cue: {cue_str}"
        lines.append(line)
        note = rephrase_note(text, allowed_items)
        if note and note not in notes:
            notes.append(note)
    if after:
        lines.append("After (context only): " + " ".join(x.text for x in after))
    if _shares_sentence(chunk, by_segment):
        notes.append(_SAME_SENTENCE_NOTE)   # Русский: одна подсказка на батч, после строк сегментов
    lines.extend(notes)
    return "\n".join(lines)


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
        cues_map = self.cfg.segment_cues or None
        out = []
        for i in range(0, len(nums), size):
            chunk = nums[i:i + size]
            if self.cfg.batch_passage:
                # cartoon_photos: батч одним текстом, строка на сегмент (см. _format_passage)
                body = _format_passage(chunk, self.by_num, self.sentences, self.by_segment,
                                       self.cfg.rules.allowed_shot_items, segment_cues=cues_map)
            else:
                src_photos = self.cfg.source_photo_prompts or {}
                durs = self.cfg.segment_durations or {}
                blocks = [_format_segment_block(self.by_num[n], self.sentences, self.by_segment,
                                                self.cfg.rules.allowed_shot_items,
                                                source_photo=src_photos.get(n),
                                                duration_sec=durs.get(n),
                                                segment_cues=cues_map)
                          for n in chunk]
                body = "\n\n".join(blocks)
            user = ("Write prompts for the following segments. Return only the JSON array.\n\n" + body)
            if cues_map:
                cue_guide = build_batch_cue_guide(cues_map.get(n) for n in chunk)
                if cue_guide:
                    user += "\n\n" + cue_guide
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
# причина правила art_medium (отключается профилем: relaxed_rules); вынесена, чтобы пропуск знал, какое правило снять
_ART_MEDIUM_REASON = "стиль: художественная техника"
# причина правила emotion (отключается профилем: relaxed_rules); вынесена, чтобы пропуск знал, какое правило снять
_EMOTION_REASON = "стиль: эмоциональная метка"
_NEGATION_REASON = "стиль: отрицание"
# префикс причины, пришедшей с разбора ответа на исправление: такие причины не проходят через таблицы кодов нарушений
_PARSE_PREFIX = "разбор: "
# cartoon_photos (batch_passage): канон главного героя и позитивная формулировка вместо «free of X»
_HERO_CANON_REASON = "персонаж: неполное описание главного героя"
_FREE_OF_REASON = "стиль: «free of» вместо позитивного описания"
_HERO_NAME_RX = re.compile(r"\bmain character\b", re.I)
_HERO_CANON_PARTS = ("round head", "stick")
# «free of» и следом (до двух слов) визуальное существительное; «free of quotation marks» не ловится
# Русский: per-clip проверки batch_passage (без квот по пакету); английские подсказки — в _parse_hint_en
_BLANK_CARD_REASON = "надпись: пустая табличка или карточка без текста"
_UNAUTH_TEXT_REASON = "надпись: текст на обычной сцене без разрешения"
_OVERLAY_CARRIER_REASON = "надпись: запрещённая плашка или карточка поверх кадра"
_ZERO_PHRASE_REASON = "стиль: «zero» вместо позитивного описания"
_TYPE_B_ON_HERO_REASON = "персонаж: одежда типа B на главном герое типа A"
_MOUTH_REASON = "персонаж: рот противоречит себе или вынесен над головой"
_LONG_LABEL_REASON = "надпись: слишком длинная надпись (больше 4 слов или 26 символов)"
# внутренние метки типов персонажей («type A», «type C mascot») и служебное слово «situational» не должны попадать в кадр
_TYPE_LABEL_REASON = "персонаж: внутренняя метка типа (type A / type B / type C)"
_SITUATIONAL_REASON = "стиль: служебное слово «situational» в описании предмета"
_TYPE_LABEL_RX = re.compile(r"\btype\s+[abc]\b(?:\s+(?:mascot|character|cartoon|figure))?", re.IGNORECASE)
_SITUATIONAL_RX = re.compile(r"\bsituational\b", re.IGNORECASE)
_CARD_WORDS_RX = re.compile(r"\b(?:label card|title plaque|accent plaque|framed plaque|header band|nameplate)s?\b", re.I)
_OVERLAY_CARRIER_RX = re.compile(
    r"\b(?:label card|title plaque|accent plaque|framed plaque|header band|number card)s?\b", re.I)
_DIEGETIC_SIGN_RX = re.compile(
    r"\b(?:shop sign|storefront sign|street sign|harbor sign|harbour sign|wooden sign|signboard|"
    r"awning|street plaque|market sign|bakery sign)s?\b", re.I)
_READ_WORDS_RX = re.compile(r"\b(?:read|reads|reading)\b|:", re.I)
_ORDINARY_MODES = frozenset({"wide", "medium", "macro", "overhead", "spotlight", "podium", "neat"})
_ORDINARY_OPENER_RX = re.compile(r"^\W*(?:an?|the)?\s*(?:[\w-]+\s+){0,2}?(wide|medium|macro|overhead|spotlight|podium|neat)\b",
                                 re.I)
_ANALYTIC_RX = re.compile(r"\b(?:map|chart|timeline|split|scale|balance|diagram|cutaway|cross-section|checklist|process)\b"
                          r"|(?<!label )(?<!title )\bcard\b", re.I)
_ROMAN_TOKEN_RX = re.compile(r"\b[IVXLCDM]{2,6}\b")
_ZERO_RX = re.compile(r"\bzero\s+(?:characters?|people|persons?|figures?|text|labels?|words?|plaques?|signs?|numbers?|"
                      r"objects?|others?)\b", re.I)
_CLOTHING_TYPE_B_RX = re.compile(r"\b(?:shirts?|trousers|pants|sweaters?|jackets?|overalls|boots|lab coats?|suits?)\b", re.I)
_HERO_ROUND_HEAD_RX = re.compile(r"\bpure-white round head\b", re.I)
_MOUTH_ABOVE_RX = re.compile(r"\bmouth\b[^.]{0,25}\babove\b", re.I)
# Русский: четыре канонических типа рта из style.md; два разных типа у одного персонажа — конфликт, один тип дважды — нет
_MOUTH_TYPE_RXS = (
    re.compile(r"\bclosed-line mouth\b", re.I),
    re.compile(r"\bclosed-line smile\b", re.I),
    re.compile(r"\bopen O-shaped mouth\b", re.I),
    re.compile(r"\b(?:wavy worried|worried wavy) mouth\b", re.I),
)
# Русский: определённые слова перед ссылкой на персонажа — повтор того же персонажа («the farmer»), не второй
_DEFINITE_WORDS = frozenset({"the", "his", "her", "their", "its", "this", "that", "such"})
_GROUP_WORDS_RX = re.compile(r"\b(?:two|three|four|five|several|many|group|crowd|villagers|people|men|women|figures|children)\b",
                             re.I)
_SINGLE_CHARACTER_RX = re.compile(r"\bcharacter\b", re.I)
# Второй персонаж в кадре (названный по роли): «adult figure», «the farmer», «a historical figure» и т.п.
_SECOND_CHARACTER_RX = re.compile(r"\b(?:adults?|child|children|figures?|farmers?|scientists?|chefs?|miners?|men|man|"
                                  r"women|woman|persons?|microbes?|villagers?|boys?|girls?)\b", re.I)
# Ссылка на персонажа внутри предложения: по ней режем клаузы («..., and the adult has ...»)
_CHARACTER_REF_RX = re.compile(r"\b(?:main character|character|adults?|child|figures?|farmers?|scientists?|chefs?|"
                               r"miners?|men|man|women|woman|persons?|microbes?|villagers?|boys?|girls?)\b", re.I)
# Граница клаузы: «; », «, while» или «and» перед новой ссылкой на персонажа («and one», «and the other» — тоже)
_CLAUSE_SPLIT_RX = re.compile(r";|,\s*while\b|\s+and\s+(?=(?:the\s+|an?\s+)?(?:main character|character|adults?|child|"
                              r"figures?|farmers?|scientists?|chefs?|miners?|men|man|women|woman|persons?|microbes?|"
                              r"villagers?|boys?|girls?|one|another|other)\b)", re.I)
# «reads/reading <LABEL>»: метка до «in … lettering», запятой, точки, строчного продолжения или конца
# Французские апострофы внутри метки (L'EAU, D'OR, C'EST) не обрезают надпись; строчные латинские с диакритикой включены
_READ_LABEL_RX = re.compile(
    r"\b(?i:reads?|reading)\s+(?:the\s+)?(?:words?\s+|text\s+|label\s+)?[\"'\u201c\u00ab]?"
    r"([^\n,.;\"\u201d\u00bb]+?)"
    r"(?=\s+(?i:in)\s|\s+[a-z\u00e0-\u00ff\u0153\u0430-\u044f\u0451]|[\"'\u201d\u00bb]?\s*[,.;]|$)"
)
_FREE_OF_RX = re.compile(r"\bfree of\s+(?:[\w-]+\s+){0,2}?(?:words?|text|labels?|smoke|fire|wood|clothing|headwear|"
                         r"expressions?|smiling|smile|stick limbs?|crowds?)\b", re.I)
_STYLE_RULES = [
    (re.compile(r"\b(?:no|not|without|never|nothing|nobody|none|"
                r"(?:don|doesn|isn|aren|can|won)['\u2019]?t|cannot)\b", re.I),
     _NEGATION_REASON,
     "Remove negations (no, not, without, never, nothing); describe only what is present."),
    (re.compile(r"[\"\u201c\u201d\u00ab\u00bb]"),
     "стиль: кавычки",
     "Remove quotation marks and any spoken or written words."),
    (re.compile(r"\b(?:4k|8k|uhd|masterpiece|trending|ultra[- ]detailed|highly detailed|"
                r"photorealistic|hyperrealistic|render\s+(?:engine|farm)|rendered|octane|unreal[ -]engine)\b", re.I),
     "стиль: keyword spam",
     "Remove photo and render terms such as 4k, 8k, UHD, photorealistic, masterpiece, trending, "
     "rendered, rendering, render, octane, unreal engine."),
    (re.compile(r"\b(?:thoughtful|pensive|contemplative|melancholy|melancholic|wistful|"
                r"nostalgic|solemn|somber|sombre|serene|peaceful|tense|anxious|mysterious|"
                r"eerie|haunting|majestic|epic|cinematic|sad|lonely|hopeless|hopeful|"
                r"confused|happy|angry|afraid|scared|joyful|stillness)\b", re.I),
     _EMOTION_REASON,
     "Replace mood or emotion words (thoughtful, solemn, serene, tense, cinematic, stillness) "
     "with visible physical cues or concrete objects."),
    (re.compile(r"\b(?:film\s+(?:still|grain)|anamorphic|bokeh|depth\s+of\s+field|35\s?mm|"
                r"lens|shot\s+on\s+(?:a\s+|an\s+)?(?:camera|film)|aperture|focal\s+length)\b", re.I),
     "стиль: кино и камера",
     "Remove film and lens terms such as film grain, anamorphic, bokeh, depth of field, 35mm, lens, "
     "aperture, focal length, shot on. For a magnifying glass write round glass disc or magnifier, never lens. "
     "Describe only the scene and its one camera move."),
    (re.compile(r"\b(?:illustrations?|cartoons?|anime|watercolou?r|oil\s+painting|sketch(?:es)?|"
                r"stickmen?|pixel\s+art)\b", re.I),
     _ART_MEDIUM_REASON,
     "Remove art-medium words such as illustration, cartoon, anime, watercolor, oil painting, sketch, "
     "stickman, pixel art. Describe the real scene instead."),
]
_STYLE_HINTS = {reason: hint for _, reason, hint in _STYLE_RULES}


_STYLE_HINTS.update(CONTENT_HINT_BY_REASON)  # подсказки для повтора по содержанию кадра

# Проверка refs и меток <imageN> (только профили с refs_enabled). Причины — русские (журнал и повтор),
# подсказки — английские (текст повтора, уходит в модель).
REF_REASON_MISSING = "refs: поле отсутствует"
REF_REASON_NOT_LIST = "refs: не список"
REF_REASON_NOT_INT = "refs: не целые числа"
REF_REASON_RANGE = "refs: номер вне 1…10"
REF_REASON_DUP = "refs: повторяющиеся номера"
LABEL_REASON = "метки <imageN>: не совпадают с refs сегмента"
REF_REASON_COMBO = "refs: сочетание не разрешено профилем"
REF_COMBO_HINT = "Use only one of the allowed reference combinations listed in the style guide."
REF_CODE_INVALID = "refs_invalid"
REF_CODE_LABELS = "label_mismatch"
_LABEL_RX = re.compile(r"<image(\d+)>")
_REF_HINT_TEXT = ("Use the tags <image1>…<imageN>, where N is the number of entries in the refs list of this "
                  "segment, numbered from 1 in ascending order of the refs numbers.")
_STYLE_HINTS.update({
    REF_REASON_MISSING: "Return a refs list of integers for this segment.",
    REF_REASON_NOT_LIST: "Return refs as a JSON list of integers.",
    REF_REASON_NOT_INT: "Return refs as integers only.",
    REF_REASON_RANGE: "Each refs number must be between 1 and 10.",
    REF_REASON_DUP: "Return each refs number only once.",
    LABEL_REASON: _REF_HINT_TEXT,
    REF_REASON_COMBO: REF_COMBO_HINT,
})


def _range_reason(limit: int) -> str:
    """Причина «номер вне 1…limit». Для лимита REF_MAX — прежний текст REF_REASON_RANGE."""
    if limit == REF_MAX:
        return REF_REASON_RANGE
    return f"refs: номер вне 1…{limit} (в нише {limit} референсов)"


# подсказка для повтора при любом числе референсов профиля (1…REF_MAX); для REF_MAX текст тот же, что выше
_STYLE_HINTS.update({_range_reason(k): f"Each refs number must be between 1 and {k}." for k in range(1, REF_MAX + 1)})

# Метки <imageN> и ref_N без референсов (refs_enabled: false): причина для повтора и барьер перед выдачей.
# Для профилей с референсами эти проверки не выполняются.
_REF_TAG_RX = re.compile(r"<image\d+>|\bref_\d+\b")
_IMAGE_TAG_RX = re.compile(r"<image\d+>")
REF_TAG_REASON = "метки <imageN> или ref_N при выключенных референсах"
_STYLE_HINTS.update({REF_TAG_REASON: "Do not write <imageN> or ref_N tags; this niche has no reference images. "
                                     "Describe the scene in words only."})


def _check_refs(value: Any, limit: int = REF_MAX) -> tuple:
    """(отсортированный список или None, причины). Пустой список допустим (сегмент без референсов).
    limit: число референсов профиля (ref_max); без аргумента — REF_MAX, как раньше."""
    if value is None:
        return None, [REF_REASON_MISSING]
    if not isinstance(value, list):
        return None, [REF_REASON_NOT_LIST]
    if any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        return None, [REF_REASON_NOT_INT]
    if any(v < 1 or v > limit for v in value):
        return None, [_range_reason(limit)]
    if len(set(value)) != len(value):
        return None, [REF_REASON_DUP]
    return sorted(value), []


def _combo_bad(refs, cfg) -> bool:
    """True, если отсортированные refs сегмента не равны ни одному разрешённому сочетанию профиля.
    Без allowed_refs или при невалидном refs — False (проверка сочетания не применяется)."""
    if cfg.allowed_refs is None or refs is None:
        return False
    return tuple(sorted(refs)) not in cfg.allowed_refs


def _label_detail(prompt: str, refs: list) -> str:
    """Пусто, если метки <imageN> совпадают с 1…len(refs); иначе английское описание для повтора."""
    found = sorted(set(int(m) for m in _LABEL_RX.findall(prompt or "")))
    expected = list(range(1, len(refs) + 1))
    if found == expected:
        return ""
    tags = ", ".join(f"<image{n}>" for n in found) or "none"
    want = (f"<image1>…<image{len(refs)}>" if refs else "no <imageN> tags")
    return f"tags {tags} found, expected {want} for {len(refs)} refs"


def _normalise_labels(prompt: str, refs) -> str:
    """Метки <imageN> переводятся из номеров референсов в позиции в refs (только профили с refs_enabled).

    Модель пишет номер референса (<image6> для референса 6), а ComfyUI и проверка меток ждут позицию в refs
    (1…len(refs)). Если каждая метка — номер из refs, она заменяется на позицию + 1. Если номера не из refs,
    текст не меняется: модель могла написать позиции, и проверка меток решит. refs: проверенный список или None;
    при None текст не меняется, метки проверит _label_reasons как раньше.
    """
    if refs is None or not prompt:
        return prompt
    found = [int(m) for m in _LABEL_RX.findall(prompt)]
    if not found:
        return prompt
    pos = {n: i + 1 for i, n in enumerate(refs)}
    if all(n in pos for n in found):
        return _LABEL_RX.sub(lambda m: f"<image{pos[int(m.group(1))]}>", prompt)
    return prompt


def _label_reasons(prompt: str, refs) -> list:
    """Причина меток для parse_and_validate (русская). Только когда refs валидны."""
    return [LABEL_REASON] if refs is not None and _label_detail(prompt, refs) else []


def _ref_fix_problems(prompt: str, refs, invalid_reasons: list) -> list:
    """Строки очереди исправлений для меток и refs: «код: «описание»», описание английское."""
    out = []
    if refs is not None:
        detail = _label_detail(prompt, refs)
        if detail:
            out.append(f"{REF_CODE_LABELS}: «{detail}»")
    elif invalid_reasons:
        if REF_REASON_COMBO in invalid_reasons:
            out.append(f"{REF_CODE_INVALID}: «{REF_COMBO_HINT}»")
        else:
            out.append(f"{REF_CODE_INVALID}: «refs must be a list of distinct integers 1…10»")
    return out


def _is_ref_problem(problem: str) -> bool:
    return problem.partition(": ")[0] in (REF_CODE_LABELS, REF_CODE_INVALID)


_LENGTH_REASON_RX = re.compile(r"^длина \d+ больше лимита (\d+)")


def _parse_hint_en(item: str) -> str:
    """Английская подсказка для причины разбора: текст правила стиля, подсказка содержания (этнос), английские
    подсказки длины и служебного мусора; иначе общая (причина по-русски в запрос не идёт)."""
    reason = item[len(_PARSE_PREFIX):]
    for _rx, label, hint in _STYLE_RULES:
        if label == reason:
            return hint
    if reason == _HERO_CANON_REASON:
        return ("Describe the main character in full: a smooth pure-white round head with a crisp black outline, "
                "small solid black dot eyes, curved eyebrows, and a plain solid-black stick body with stick arms, "
                "hands and legs.")
    if reason == _FREE_OF_REASON:
        return ("Replace each phrase of the form free of X with a positive description of what IS visible instead, "
                "for example a blank dark screen, a bare stick body, clear air or neutral animal features; "
                "never name the excluded item.")
    if reason == _BLANK_CARD_REASON:
        return ("Either write valid uppercase text from the Text line on the card or plaque, or remove the card or "
                "plaque completely.")
    if reason == _UNAUTH_TEXT_REASON:
        return ("Remove all plaques, label cards, header bands, pointer-arrow text and lettering; keep this "
                "ordinary scene text-free.")
    if reason == _OVERLAY_CARRIER_REASON:
        return ("Remove all floating or overlaid text carriers (label cards, title plaques, accent plaques, "
                "framed plaques, header bands, number cards); keep the frame as a pure text-free visual illustration.")
    if reason == _ZERO_PHRASE_REASON:
        return ("Do not write zero with a count; describe only the visible elements in positive wording, for example "
                "an empty path or one figure.")
    if reason == _TYPE_B_ON_HERO_REASON:
        return ("The Type A stick-body main character has a bare solid-black stick body and wears at most a "
                "situational hat, apron or backpack; remove the shirt, trousers or other Type B clothing from it.")
    if reason == _MOUTH_REASON:
        return ("Give the single character one consistent mouth expression placed directly on its face; "
                "do not place a mouth above the head.")
    if reason == _TYPE_LABEL_REASON:
        return ("Remove internal character-type labels such as type A, type B, type C mascot or type C character; "
                "keep only the natural visual description of the character.")
    if reason == _SITUATIONAL_REASON:
        return ("Remove the word situational and name the item naturally, for example farmer hat instead of "
                "situational farmer hat.")
    if reason == _LONG_LABEL_REASON:
        return ("Shorten the label to 1 or 2 words, at most 26 characters, or remove it.")
    if reason in CONTENT_HINT_BY_REASON:
        return CONTENT_HINT_BY_REASON[reason]
    if reason.startswith("длина "):
        m = _LENGTH_REASON_RX.match(reason)
        if m:
            return (f"Shorten the prompt to at most {m.group(1)} characters; "
                    "keep the subject, count, action and setting.")
        return "Shorten the prompt; keep the subject, count, action and setting."
    if reason.startswith("служебный мусор:"):
        return ("Remove service text, markup, placeholders or code-like fragments; "
                "write plain descriptive prose only.")
    return "Fix the problem found when the previous version was checked; keep the rest of the prompt."


# Маркер вида кадра в фрагменте причины «дубликат промпта» от mark_monotony_for_rework: «N mode=<вид>»
_VISUAL_MODE_MARKER_RX = re.compile(r"\bmode=([a-z][a-z \-]{0,30})")

# Английский текст повтора для клипа, повторившего недавний вид кадра этого батча (без списка мест)
VISUAL_MODE_REWORK_TEXT = (
    "This frame repeats a recent {mode} scene of this batch. Rewrite it so the scene is visibly different: "
    "change the setting or location, the background, the props or the composition. "
    "Do not keep the same arrangement with only a few words swapped. "
    "Keep the segment's meaning and the character canon. Output only the rewritten prompt.")

# Английский текст повтора для клипа-дубликата. Текст модели: сцена должна стать заметно другой.
DUPLICATE_REWORK_TEXT = (
    "This prompt is the same as the prompt of another clip. Rewrite it so the scene is visibly different: "
    "change the layout, the main focal object, the setting or the character's pose and action. "
    "Do not merely swap left and right, cloud placement, or grass and dirt colours. "
    "Keep the segment's meaning and the subject type. Output only the rewritten prompt.")


def _rework_text(problems: list) -> str:
    """Английский текст повтора для строк очереди: причины разбора — подсказкой, остальное как раньше."""
    # Русский: дубликат промпта — свой текст повтора; прочие причины в той же строке идут обычным путём
    dup = [p for p in problems if problem_code(p) == PROBLEM_DUPLICATE_PROMPT]
    if dup:
        others = [p for p in problems if problem_code(p) != PROBLEM_DUPLICATE_PROMPT]
        head = _rework_text(others) if others else ""
        # Русский: повтор вида кадра из batch_passage — своя подсказка; точный дубликат — прежний текст
        tail = DUPLICATE_REWORK_TEXT
        for p in dup:
            m = _VISUAL_MODE_MARKER_RX.search(p)
            if m and m.group(1).strip():
                tail = VISUAL_MODE_REWORK_TEXT.format(mode=" ".join(m.group(1).split()))
                break
        return (head + " " if head else "") + tail
    parse = [p for p in problems if p.startswith(_PARSE_PREFIX)]
    rest = [p for p in problems if not p.startswith(_PARSE_PREFIX)]
    base = ("Rewrite this prompt. Keep the subject, count, action and setting from the segment. "
            "Output only the rewritten prompt.") if parse and not rest else _rework_text_core(rest)
    hints = " ".join(dict.fromkeys(_parse_hint_en(p) for p in parse))
    return base + (" " + hints if hints else "")


def _rework_text_core(problems: list) -> str:
    """Английский текст повтора для строк очереди. Строки меток и refs — свой текст; остальные — как раньше."""
    ref_items = [p for p in problems if _is_ref_problem(p)]
    if not ref_items:
        return rework_remark(problems)
    base = rework_remark([p for p in problems if not _is_ref_problem(p)]) if len(ref_items) < len(problems) \
        else rework_remark([])
    lines = [p.partition(": ")[2] for p in ref_items]
    if any(p.partition(": ")[0] == REF_CODE_LABELS for p in ref_items):
        return base + " Fix the image labels so that they match the refs list of this segment: " + \
            " ".join(lines) + " Return the refs list unchanged unless a fix needs it."
    # только невалидный refs (без меток): просим исправить сам список
    return base + " Fix the refs list of this segment: " + " ".join(lines) + \
        " Return refs as a JSON list of integers."


def _problem_ru_any(problem: str) -> str:
    if problem.startswith(_PARSE_PREFIX):
        return problem[len(_PARSE_PREFIX):]
    if _is_ref_problem(problem):
        code, _, rest = problem.partition(": ")
        if code == REF_CODE_LABELS:
            head = LABEL_REASON
        elif rest == f"«{REF_COMBO_HINT}»":
            head = REF_REASON_COMBO
        else:
            head = REF_REASON_NOT_LIST
        return f"{head}: {rest}"
    return problem_ru(problem)


def _only_caps_matches(prompt: str, rx) -> bool:
    # Исключение для текста на кадре: если профиль снимает frame_text, отрицание, написанное только заглавными
    # словами из двух и более букв (например, «NO TOOLS» на табличке), правилом стиля не считается;
    # строчные и смешанные совпадения (no, No) по-прежнему нарушают правило.
    found = [m.group(0) for m in rx.finditer(prompt)]
    return bool(found) and all(len(t) >= 2 and t.isupper() for t in found)


def _style_reasons(prompt: str, media: str = "video", relaxed=frozenset(), max_characters: Optional[int] = None,
                   rules: Optional[RulesValues] = None) -> list:
    """Стилевые причины и причины содержания кадра (дети, толпа, мелкие предметы, речь и т. д.).
    relaxed: отключённые профилем правила; art_medium снимает стилевое правило, small_objects — в содержании."""
    style = [reason for rx, reason, _ in _STYLE_RULES
             if rx.search(prompt) and not (reason == _ART_MEDIUM_REASON and "art_medium" in relaxed)
             and not (reason == _NEGATION_REASON and "frame_text" in relaxed and _only_caps_matches(prompt, rx))
             and not (reason == _EMOTION_REASON and "emotion" in relaxed)]
    return style + content_reasons(prompt, media, relaxed, max_characters, rules)


# ---------------------------------------------------------------------------
# Профиль ниши: relaxed_rules, text_language, референсы. Поля читаются строками profile.yaml, без PyYAML.
# ---------------------------------------------------------------------------

def _profile_value(lines: list, key: str) -> Optional[str]:
    """Сырое значение поля верхнего уровня: регистр сохраняется, комментарий и кавычки убраны. Нет поля — None."""
    for ln in lines:
        if ln.startswith(key + ":"):
            return ln.split(":", 1)[1].split("#", 1)[0].strip().strip("\"'") or None
    return None


def parse_flow_list(value: str, key: str, profile: str) -> list:
    """Список вида [a, b] из значения поля. Другой вид значения — ошибка по-русски."""
    text = (value or "").strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise GeminiInputError(
            f"Ошибка входных данных: в профиле {profile}/profile.yaml {key} должен быть списком вида [a, b].")
    return [item.strip().strip("\"'") for item in text[1:-1].split(",") if item.strip()]


def parse_relaxed_rules(names: list, profile: str) -> frozenset:
    """relaxed_rules профиля: только id из RELAXABLE_RULES. Иное имя — ошибка при загрузке профиля."""
    bad = [n for n in names if n not in RELAXABLE_RULES]
    if bad:
        raise GeminiInputError(
            f"Ошибка входных данных: в профиле {profile}/profile.yaml в relaxed_rules недопустимо: "
            f"{', '.join(bad)}. Допустимые значения: {', '.join(sorted(RELAXABLE_RULES))}.")
    return frozenset(names)


def profile_relaxation(lines: list, profile: str) -> tuple:
    """(relaxed_rules, text_language) из строк profile.yaml. Нет ключей — (пусто, None), как раньше.

    text_language: None; «srt» (язык берётся из SRT прогона); или название языка латиницей (задано явно).
    frame_text без text_language — ошибка: правило «следующее правило» без языка указывало бы в пустоту.
    """
    raw_rules = _profile_value(lines, "relaxed_rules")
    names = parse_flow_list(raw_rules, "relaxed_rules", profile) if raw_rules is not None else []
    relaxed = parse_relaxed_rules(names, profile)
    # allow_text: true — текст в кадре разрешён: правило frame_text отключено; без поля или false — как раньше
    allow_text = _profile_value(lines, "allow_text")
    if allow_text not in (None, "true", "false"):
        raise GeminiInputError(
            f"Ошибка входных данных: в профиле {profile}/profile.yaml allow_text должен быть true или false.")
    if allow_text == "true":
        relaxed = relaxed | {"frame_text"}
    text_language = _profile_value(lines, "text_language")
    if text_language is not None and text_language != "srt" and not re.fullmatch(r"[A-Za-z]+", text_language):
        raise GeminiInputError(
            f"Ошибка входных данных: в профиле {profile}/profile.yaml text_language должен быть «srt» "
            f"или название языка латиницей, например Russian.")
    if "frame_text" in relaxed and text_language is None:
        raise GeminiInputError(
            f"Ошибка входных данных: в профиле {profile}/profile.yaml relaxed_rules содержит frame_text, "
            f"но text_language не задан: нужен язык текста в кадре.")
    return relaxed, text_language


# Разрешённые сочетания refs (allowed_refs): строки в кавычках, в каждой номера через запятую.
# parse_flow_list здесь не годится: он режет по каждой запятой, и «1,2» распалось бы на два номера.
_QUOTED_ITEM_RX = re.compile(r"""\s*(?:"([^"]*)"|'([^']*)')\s*""")


def _allowed_error(profile: str, detail: str) -> GeminiInputError:
    return GeminiInputError(f"Ошибка входных данных: в профиле {profile}/profile.yaml allowed_refs {detail}.")


def parse_allowed_refs(lines: list, profile: str):
    """frozenset отсортированных кортежей номеров из allowed_refs; нет поля — None (проверки нет).

    Каждая запись — строка в кавычках: номера от 1 до REF_MAX через запятую, без повторов.
    Иное — ошибка при загрузке профиля."""
    raw = _profile_value(lines, "allowed_refs")
    if raw is None:
        return None
    text = raw.strip()
    form = "должен быть списком вида [\"1\", \"1,2\"]"
    if not (text.startswith("[") and text.endswith("]")):
        raise _allowed_error(profile, form)
    inner = text[1:-1]
    items, pos = [], 0
    while True:
        m = _QUOTED_ITEM_RX.match(inner, pos)
        if m is None:
            raise _allowed_error(profile, form)
        items.append(m.group(1) if m.group(1) is not None else m.group(2))
        pos = m.end()
        if inner[pos:].strip() == "":
            break
        if not inner[pos:].lstrip().startswith(","):
            raise _allowed_error(profile, form)
        pos = inner.index(",", pos) + 1
    combos = set()
    for item in items:
        nums = item.split(",") if re.fullmatch(r"\d+(,\d+)*", item) else None
        values = [int(x) for x in nums] if nums else []
        if not values or any(v < 1 or v > REF_MAX for v in values) or len(set(values)) != len(values):
            raise _allowed_error(profile, f"содержит недопустимую запись «{item}»: нужны номера от 1 до "
                                              f"{REF_MAX} через запятую без повторов, например «1,2»")
        combos.add(tuple(sorted(values)))
    return frozenset(combos)


# Определение языка текста по SRT: библиотека lingua (lingua-language-detector, Generate/requirements-lang.txt).
# Решение о языке принимает lingua (detect_language_of) с минимальным относительным отрывом LINGUA_MIN_GAP.
# LINGUA_MIN_CONFIDENCE не участвует в решении: только для строки в логе. Не определено — ошибка с просьбой
# задать text_language явно.
SRT_MIN_LETTERS = 20          # меньше букв в тексте — язык не определяется
LINGUA_MIN_CONFIDENCE = 0.80  # только для лога: уверенность выбранного языка пишется в строку лога
LINGUA_MIN_GAP = 0.10         # минимальный относительный отрыв лучшего языка от второго (задаётся детектору)

_SRT_INDEX_RX = re.compile(r"^\s*\d+\s*$")
_SRT_TIMING_RX = re.compile(r"^\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->")
_SRT_TAG_RX = re.compile(r"<[^>]*>|\{[^}]*\}")
_LINGUA_DETECTOR = None   # детектор строится один раз на процесс


def _lingua_detector():
    """Детектор lingua по всем языкам. Импорт здесь, а не на уровне модуля: без библиотеки модуль грузится."""
    global _LINGUA_DETECTOR
    if _LINGUA_DETECTOR is None:
        try:
            from lingua import LanguageDetectorBuilder
        except ImportError:
            raise GeminiInputError(
                "Ошибка входных данных: для языка текста по SRT нужна библиотека lingua-language-detector "
                "(Generate/requirements-lang.txt), она не установлена.") from None
        # минимальный относительный отрыв: lingua сама возвращает None, если отрыв меньше LINGUA_MIN_GAP
        _LINGUA_DETECTOR = (LanguageDetectorBuilder.from_all_languages()
                            .with_minimum_relative_distance(LINGUA_MIN_GAP)
                            .build())
    return _LINGUA_DETECTOR


def detect_srt_language(path) -> str:
    """Название языка текста в SRT по-английски («Russian»). Неясно — GeminiInputError по-русски.

    Строки-индексы, тайминги и теги отбрасываются. Букв меньше SRT_MIN_LETTERS — ошибка. Иначе решение о языке
    принимает lingua (detect_language_of): None — ошибка, язык не определён. Уверенность пишется только в лог.
    Текст SRT в лог не пишется.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        raise GeminiInputError(f"Ошибка входных данных: SRT для языка текста в кадре не прочитан: {path}") from None
    kept = []
    for line in raw.splitlines():
        if not line.strip() or _SRT_INDEX_RX.match(line) or _SRT_TIMING_RX.match(line):
            continue
        kept.append(_SRT_TAG_RX.sub("", line))
    text = "\n".join(kept)
    letters = sum(1 for ch in text if ch.isalpha())
    hint = "Задайте text_language в профиле явно (название языка латиницей, например Russian)."
    if letters < SRT_MIN_LETTERS:
        raise GeminiInputError(
            f"Ошибка входных данных: язык текста в SRT не определён, мало букв ({letters}). {hint}")
    detector = _lingua_detector()
    # решение lingua: None — смешанный текст или отрыв меньше LINGUA_MIN_GAP; порог уверенности здесь не проверяется
    language = detector.detect_language_of(text)
    if language is None:
        raise GeminiInputError(
            f"Ошибка входных данных: язык текста в SRT не определён: текст смешанный или язык неясен. {hint}")
    name = language.name.title()
    # уверенность нужна только для строки в лог; на решение не влияет
    values = detector.compute_language_confidence_values(text)
    confidence = next((cv.value for cv in values if cv.language == language), 0.0)
    log.info("Язык текста в кадре определён по SRT: %s, уверенность %.2f.", name, confidence)
    return name


_REF_FILE_RX = re.compile(r"ref_(\d+)\.png")


def count_ref_files(refs_dir, profile: str) -> int:
    """Число допустимых референсов: ref_1.png…ref_N.png в папке refs профиля, не больше REF_MAX.

    Нет папки, нет файлов или номера не идут подряд с 1 — ошибка при загрузке профиля.
    """
    folder = Path(refs_dir)
    if not folder.is_dir():
        raise GeminiInputError(
            f"Ошибка входных данных: в профиле {profile} refs_enabled: true, но папки refs нет: {folder}")
    nums = set()
    for p in folder.iterdir():
        m = _REF_FILE_RX.fullmatch(p.name)
        if m:
            nums.add(int(m.group(1)))
    nums = sorted(nums)
    if not nums:
        raise GeminiInputError(
            f"Ошибка входных данных: в профиле {profile} в папке refs нет файлов ref_N.png: {folder}")
    if nums != list(range(1, len(nums) + 1)):
        raise GeminiInputError(
            f"Ошибка входных данных: в профиле {profile} номера референсов должны идти подряд с 1 "
            f"(ref_1.png … ref_{len(nums)}.png), найдено: {nums}")
    return min(len(nums), REF_MAX)


def _retry_note(nums, problems: list, max_characters: Optional[int] = None, animals_count: bool = True) -> str:
    """Текст для повторного запроса: что именно исправить. Пусто, если стилевых нарушений нет.
    max_characters: при заданном лимите персонажей подсказка о людях и животных берётся из people_hint."""
    lines = []
    people_reason = CONTENT_REASON_RU[PROBLEM_PEOPLE_COUNT]
    many_reason = CONTENT_REASON_RU[PROBLEM_MANY_OF_ONE]
    by_count = animal_plurals_by_count(max_characters, animals_count)
    for n in nums:
        hints = []
        for p in problems:
            if p.num != n:
                continue
            if max_characters is not None and p.reason == people_reason:
                hints.append(people_hint(max_characters, animals_count))
            elif by_count and p.reason == many_reason:
                hints.append(MANY_OF_ONE_VEHICLES_HINT_EN)
            elif p.reason in _STYLE_HINTS:
                hints.append(_STYLE_HINTS[p.reason])
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
        if cfg.refs_enabled:
            # Русский: метки переводятся в позиции до всех проверок (длина, стиль, этнос, метки); refs берётся
            # из ответа только если он валиден, иначе текст не меняется и метки проверит _label_reasons
            prompt = _normalise_labels(prompt, _check_refs(item.get("refs"), cfg.ref_max)[0])
        if len(prompt) > cfg.max_prompt_chars:
            reasons.append(f"длина {len(prompt)} больше лимита {cfg.max_prompt_chars}")
        g = _garbage_reason(prompt)
        if g:
            reasons.append(g)
        if not cfg.refs_enabled and _REF_TAG_RX.search(prompt):
            reasons.append(REF_TAG_REASON)
        reasons.extend(_style_reasons(prompt, cfg.media, cfg.relaxed_rules, cfg.max_characters, cfg.rules))
        if texts is not None and n in texts:
            # Русский: корень этнического признака ищем в тексте сегмента и соседних (n-1, n+1); нет соседа — пусто
            window = " ".join(str(texts.get(k) or "") for k in (n - 1, n, n + 1))
            reasons.extend(ethnicity_reasons(prompt, window))
        refs, ref_reasons = None, []
        if cfg.refs_enabled:
            refs, ref_reasons = _check_refs(item.get("refs"), cfg.ref_max)
            reasons.extend(ref_reasons)
            if _combo_bad(refs, cfg):
                reasons.append(REF_REASON_COMBO)
            reasons.extend(_label_reasons(prompt, refs))
        extra = {k: v for k, v in item.items() if k not in ("segment_index", "prompt")}
        if cfg.refs_enabled:
            if refs is None:
                extra.pop("refs", None)   # невалидный refs в запись не попадает
            else:
                extra["refs"] = refs      # валидный — отсортирован; пишется в prompts.json через extra
        if "setting" in extra:
            # cartoon_photos: setting — одна строка без переводов строк; пустой остаётся пустой строкой
            extra["setting"] = " ".join(str(extra["setting"] or "").split())
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


def _only_negation_flag(review_reason: str) -> bool:
    """Единственная причина разбора — флаг стиля «отрицание» (причины разделены «; »).
    Очередь исправлений эту функцию больше не использует (любой флаг разбора идёт в очередь); оставлена для проверок."""
    parts = {x for x in review_reason.split("; ") if x}
    return parts == {_NEGATION_REASON}


def _rework_gate_open(cfg: PromptConfig, r: PromptResult) -> bool:
    """Открыт ли путь в очередь исправлений для клипа с текстом и флагом разбора.

    Закрыт при fail_fast, без needs_review и без причины. Открыт при refs_enabled (прежнее условие) или при
    rework_style_flag_without_refs: тогда любой флаг разбора (стиль, длина, мусор, этнос, метки) ведёт в очередь
    и без refs. Профиль без refs и без этого ключа — прежнее поведение: клип остаётся на проверке."""
    if cfg.fail_fast or not r.needs_review or not r.review_reason:
        return False
    return bool(cfg.refs_enabled or cfg.rework_style_flag_without_refs)


# Повтор со 2-го круга для клипа, у которого все находки — надписи: кадр без текста (только при dangling_text_check)
TEXTLESS_FROM_ROUND = 2
_LETTERING_CODES = frozenset({PROBLEM_LETTERING_NOT_IN_SEGMENT, PROBLEM_DANGLING_TEXT})
TEXTLESS_REWORK_TEXT = ("Rewrite this still frame with NO on-frame text at all. Keep the subject of the segment as a scene, "
                        "a diagram or a character, and describe the action or object in plain words. Do not quote any phrase.")


def _textless_round(cfg: PromptConfig, pending: list, req_round: int) -> bool:
    """Клип на запросе req_round (с 1): все находки pending — только надписи. Без dangling_text_check или до 4-го круга — False."""
    if not cfg.rules.dangling_text_check or req_round < TEXTLESS_FROM_ROUND or not pending:
        return False
    return {problem_code(p) for p in pending} <= _LETTERING_CODES


def _ordinary_layout(text: str, setting: str) -> bool:
    """Обычная сцена: вид кадра из setting (префикс до «:») или из начала промпта. Русский: без вида — False."""
    mode = (setting or "").split(":", 1)[0].strip().lower()
    if mode in _ORDINARY_MODES:
        return True
    m = _ORDINARY_OPENER_RX.match(text[:60])
    return bool(m)


def _blank_card(text: str) -> bool:
    """Табличка или карточка названа, но в её предложении нет «read/reads/reading» или «:» с текстом.

    Русский: окно — до 40 символов назад и до 120 вперёд, но не дальше границ предложения («.» и «;»),
    иначе шаблон «header band separated by a thin line reading <TEXT>» ложно считается пустым.
    """
    for m in _CARD_WORDS_RX.finditer(text):
        left = max(text.rfind(".", 0, m.start()), text.rfind(";", 0, m.start())) + 1
        right_candidates = [i for i in (text.find(".", m.end()), text.find(";", m.end())) if i != -1]
        right = min(right_candidates) if right_candidates else len(text)
        window = text[max(left, m.start() - 40):min(right, m.end() + 120)]
        if not _READ_WORDS_RX.search(window):
            return True
    return False


def _label_too_long(text: str) -> bool:
    """Извлечённая надпись «reading <LABEL>» длиннее 4 слов или 26 символов; цифры в надписи — не нарушение."""
    for m in _READ_LABEL_RX.finditer(text):
        label = " ".join(m.group(1).split())
        if not label or re.search(r"\d", label):
            continue
        if len(label.split()) > 4 or len(label) > 26:
            return True
    return False


def _type_b_on_hero(text: str) -> bool:
    """Одно предложение описывает главного героя (тип A) и называет одежду типа B."""
    for sentence in re.split(r"(?<=[.;])\s+", text):
        hero = _HERO_NAME_RX.search(sentence) or _HERO_ROUND_HEAD_RX.search(sentence)
        if hero and _CLOTHING_TYPE_B_RX.search(sentence):
            return True
    return False


def _mouth_types(text: str) -> set:
    """Русский: индексы канонических типов рта, найденных в тексте (порядок — как в _MOUTH_TYPE_RXS)."""
    return {i for i, rx in enumerate(_MOUTH_TYPE_RXS) if rx.search(text)}


def _has_multiple_characters(text: str) -> bool:
    """Русский: в кадре больше одного персонажа.

    Да, если есть группа; есть главный герой и второй персонаж по роли; или второй персонаж назван
    по роли два раза и более как новое введение. Повтор «the farmer» (определённое слово перед ролью)
    и соседние слова одной фразы («adult figure») вторым персонажем не считаются.
    """
    if _GROUP_WORDS_RX.search(text):
        return True
    if _HERO_NAME_RX.search(text) and _SECOND_CHARACTER_RX.search(text):
        return True
    introductions = 0
    last_end = None
    for m in _SECOND_CHARACTER_RX.finditer(text):
        adjacent = last_end is not None and m.start() - last_end <= 2   # «adult figure»: одна фраза
        last_end = m.end()
        if adjacent:
            continue
        before = re.findall(r"\b\w+\b", text[:m.start()])[-3:]
        if any(w.lower() in _DEFINITE_WORDS for w in before):           # «the farmer»: тот же персонаж
            continue
        introductions += 1
        if introductions >= 2:
            return True
    return False


def _mouth_conflict(text: str) -> bool:
    """Рот «выше» головы или два разных взаимоисключающих рта у одного персонажа.

    Русский: один персонаж (в том числе названный по роли и повторно: «a farmer … the farmer») — два разных
    типа рта в тексте конфликт. Повтор одного и того же типа не конфликт. Если персонажей несколько (группа
    или второй персонаж по роли), конфликт только когда оба рта стоят в одной клаузе с одной ссылкой на персонажа.
    """
    if _MOUTH_ABOVE_RX.search(text):
        return True
    if len(_mouth_types(text)) < 2:
        return False
    if not _has_multiple_characters(text):
        # Русский: одиночный персонаж — нужна хотя бы одна ссылка на персонажа в тексте
        return bool(_SINGLE_CHARACTER_RX.search(text) or _SECOND_CHARACTER_RX.search(text))
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for clause in _CLAUSE_SPLIT_RX.split(sentence):
            if len(_mouth_types(clause)) >= 2 and len(_CHARACTER_REF_RX.findall(clause)) <= 1:
                return True
    return False


def _batch_style_reasons(text: str, setting: str = "") -> list:
    """cartoon_photos (batch_passage): причины разбора по одному клипу. Пусто — всё в порядке.

    Русский: все проверки — по этому клипу (без квот на пакет): канон героя, «free of X», пустая табличка,
    надпись на обычной сцене, «zero …», одежда типа B на герое типа A, рот, длинная надпись.
    """
    reasons: list = []
    low = text.lower()
    if _HERO_NAME_RX.search(text) and not all(part in low for part in _HERO_CANON_PARTS):
        reasons.append(_HERO_CANON_REASON)
    if _FREE_OF_RX.search(text):
        reasons.append(_FREE_OF_REASON)
    if _blank_card(text):
        reasons.append(_BLANK_CARD_REASON)
    if _OVERLAY_CARRIER_RX.search(text):
        reasons.append(_OVERLAY_CARRIER_REASON)
    if (frame_has_lettering(text) and _ordinary_layout(text, setting)
            and not _DIEGETIC_SIGN_RX.search(text)
            and not _ANALYTIC_RX.search(text)
            and not _ANALYTIC_RX.search(setting or "")
            and not re.search(r"\d", text)
            and not _ROMAN_TOKEN_RX.search(text)):
        reasons.append(_UNAUTH_TEXT_REASON)
    if _ZERO_RX.search(text):
        reasons.append(_ZERO_PHRASE_REASON)
    if _type_b_on_hero(text):
        reasons.append(_TYPE_B_ON_HERO_REASON)
    if _mouth_conflict(text):
        reasons.append(_MOUTH_REASON)
    if _label_too_long(text):
        reasons.append(_LONG_LABEL_REASON)
    if _TYPE_LABEL_RX.search(text):
        reasons.append(_TYPE_LABEL_REASON)
    if _SITUATIONAL_RX.search(text):
        reasons.append(_SITUATIONAL_REASON)
    return reasons


def _find_all_problems(text: str, cfg: PromptConfig, refs=None, segment_text=None, textless: bool = False,
                       setting: str = "") -> list:
    """Нарушения промпта: запрещённые типы кадра, маркеры и содержание кадра; для фото ещё движение камеры.
    refs: валидный список номеров сегмента (только при refs_enabled): метки <imageN> должны совпадать с ним.
    segment_text: реплика текущего сегмента; надпись в кадре проверяется по ней (только при dangling_text_check)."""
    problems = find_problems(text, cfg.media, cfg.relaxed_rules, cfg.max_characters, cfg.rules, segment_text,
                             textless=textless)
    if cfg.media == "image":
        problems += [make_problem(PROBLEM_CAMERA_MOTION, m) for m in find_camera_motion(text)]
    if not cfg.refs_enabled:
        # метки без референсов — нарушение (очередь исправлений); барьер перед выдачей их удалит, если повтор не помог
        problems += [make_problem(PROBLEM_REF_TAG, m.group(0)) for m in _REF_TAG_RX.finditer(text)]
    if cfg.refs_enabled and refs is not None:
        problems += _ref_fix_problems(text, refs, [])
        if _combo_bad(refs, cfg):
            problems.append(f"{REF_CODE_INVALID}: «{REF_COMBO_HINT}»")
    if cfg.batch_passage:
        # Русский: причины разбора — как у прочих флагов стиля, чтобы повтор получил английскую подсказку
        problems += [_PARSE_PREFIX + r for r in _batch_style_reasons(text, setting)]
    return list(dict.fromkeys(problems))


def _strip_image_tags_without_refs(results: list, cfg: PromptConfig) -> int:
    """Барьер перед выдачей: при refs_enabled: false убирает все метки <imageN> из текста промптов.
    Выполняется после кругов исправления; для профилей с референсами ничего не делает. Возвращает число меток."""
    if cfg.refs_enabled:
        return 0
    removed = 0
    for r in results:
        new, n = _IMAGE_TAG_RX.subn("", r.prompt or "")
        if n:
            r.prompt = " ".join(new.split())
            removed += n
        if r.prompt_raw:
            r.prompt_raw = " ".join(_IMAGE_TAG_RX.sub("", r.prompt_raw).split())
    if removed:
        log.warning("Удалено меток <imageN> из промптов без референсов: %d.", removed)
    return removed


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


def _fail_unfixed(results: list, cfg: PromptConfig) -> None:
    """on_unfixed == «fail»: клипы, оставшиеся с нарушениями после всех кругов, останавливают прогон.

    Вызывается после обеих волн, до генерации фото и видео. Уже записанные в prompts.json промпты
    не удаляются: батчи сохраняются по ходу, поэтому повторный запуск продолжит с места остановки.
    """
    bad = [r for r in results if r.needs_review]
    if not bad:
        return
    parts = [f"сегмент {r.num}: {r.review_reason or 'нарушение'}" for r in bad]
    log.warning("Прогон остановлен (on_unfixed: fail): клипов с нарушениями %d.", len(bad))
    raise GeminiError(
        "Ошибка: после исправлений остались промпты с нарушениями (профиль с on_unfixed: fail), "
        "генерация не запущена. " + "; ".join(parts) + ". Уже сохранённые промпты в prompts.json не удалены.")


# fail_fast: фиксированные коды причин и их русские подписи. Текст причины в сообщение не попадает:
# он может содержать фрагмент промпта из ответа модели (make_problem), а сообщение печатается в журнал.
FAIL_FAST_LABELS = {
    "empty": "пустой ответ",
    "not_json": "ответ не является JSON",
    "not_array": "ожидался JSON-массив",
    "refs": "неверные refs или метки <imageN>",
    "segment": "нарушение у сегмента: номер, пустой текст, ограничения или стиль",
    "content": "нарушение в тексте промпта: запрещённый тип кадра, маркер или содержание",
    "other": "прочее нарушение ответа",
}


def _is_refs_reason(text: str) -> bool:
    return text.startswith("refs:") or _is_ref_problem(text)


def _parse_slug(problems: list, expected: list) -> str:
    """Код причины первой проблемы разбора (в том числе fail_all: пустой ответ, не JSON, не массив)."""
    if not problems:
        return "other"
    first = problems[0]
    if first.reason == "пустой ответ":
        return "empty"
    if first.reason == "ответ не является JSON":
        return "not_json"
    if first.reason == "ожидался JSON-массив":
        return "not_array"
    if _is_refs_reason(first.reason):
        return "refs"
    if first.num is not None and first.num in set(expected):
        return "segment"
    return "other"


def _content_slug(found: list) -> str:
    """Код причины нарушения в тексте промпта (шаг 6)."""
    return "refs" if _is_refs_reason(found[0]) else "content"


def _dump_raw(cfg: PromptConfig, batch_nums: list, slug: str, raw: str) -> None:
    """fail_fast: сырой ответ батча с проблемой пишется в <raw_dir>/batch_<первый>-<последний>_<код>.txt.
    Только при проблеме и только если raw_dir задан. В журнал текст не попадает: только путь файла.
    Сбой записи не скрывает ошибку батча."""
    if not cfg.raw_dir:
        return
    path = Path(cfg.raw_dir) / f"batch_{batch_nums[0]}-{batch_nums[-1]}_{slug}.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(raw if isinstance(raw, str) else "")
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось записать сырой ответ модели в файл %s (%s).", path, type(exc).__name__)


def _dump_debug(cfg: PromptConfig, system, user) -> None:
    """fail_fast: системная инструкция и текст запроса батча с проблемой пишутся в <родитель raw_dir>/debug/
    (system_instruction.txt, user_request.txt). Только при проблеме и только если raw_dir задан.
    В журнал содержимое не попадает: только путь к папке. Сбой записи не скрывает ошибку батча."""
    if not cfg.raw_dir:
        return
    folder = Path(cfg.raw_dir).parent / "debug"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for name, text in (("system_instruction.txt", system), ("user_request.txt", user)):
            if text is None:
                continue
            with open(folder / name, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
    except Exception:  # noqa: BLE001
        log.warning("Не удалось записать отладочные файлы промпта в папку %s.", folder)


def _dump_response(cfg: PromptConfig, wave: str, rnd: int, batch: str, raw: str) -> None:
    """Диагностика: сырой текст ответа модели пишется в <raw_dir>/<ниша>_<волна>_<круг>_<батч>.txt.

    Круг 0 — главные батчи (первый ответ), круг N — исправления раунда N; батч — метка батча или номер пакета
    исправления; повтор формата — метка с суффиксом r<k>. Только при dump_raw_responses и заданном raw_dir.
    Сбой записи не поднимается: предупреждение с путём, без текста ответа. Файлы не удаляются."""
    if not getattr(cfg, "dump_raw_responses", False) or not getattr(cfg, "raw_dir", None):
        return
    path = Path(cfg.raw_dir) / f"{cfg.niche}_{wave}_{rnd}_{batch}.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(raw if isinstance(raw, str) else "")
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось записать сырой ответ модели в файл %s (%s).", path, type(exc).__name__)


def _fail_fast_stop(cfg: PromptConfig, batch_nums: list, bad_nums: list, slug: str, raw: str,
                    rule_label: Optional[str] = None, system: Optional[str] = None,
                    user: Optional[str] = None) -> None:
    """fail_fast: первая проблема останавливает прогон. Сырой ответ уходит в файл, затем GeminiError. Повтора нет.
    rule_label: фиксированная русская подпись правила (из таблиц кодов); без неё — общая подпись по slug.
    system, user: системная инструкция и текст запроса батча для отладочных файлов (только при raw_dir)."""
    _dump_raw(cfg, batch_nums, slug, raw)
    _dump_debug(cfg, system, user)
    shown = _num_ranges(list(bad_nums)) or "нет"
    raise GeminiError(
        f"Ошибка: проверка промптов остановлена (fail_fast): батч {batch_nums[0]}-{batch_nums[-1]}, "
        f"клипы с нарушением: {shown}; причина: {rule_label or FAIL_FAST_LABELS[slug]}. "
        "Повтор и исправления не запускались.")


def _segment_rule_label(problems: list) -> Optional[str]:
    """Подпись правила для fail_fast по первой проблеме сегмента. Код в разборе не передаётся, поэтому подпись
    берётся только тогда, когда причина совпадает с фиксированной подписью правила; иначе None (общая подпись)."""
    if not problems:
        return None
    reason = problems[0].reason
    fixed = set(CONTENT_REASON_RU.values()) | {r for _, r, _ in _STYLE_RULES}
    return reason if reason in fixed else None


def _content_rule_label(found: list, max_characters: Optional[int] = None) -> Optional[str]:
    """Подпись правила для fail_fast по первому нарушению текста промпта (код до «: » в записи make_problem).
    Только при заданном max_characters (ниша с лимитом персонажей), иначе общая подпись — как раньше.
    None для refs и для неизвестного кода: тогда общая подпись."""
    if max_characters is None or _content_slug(found) != "content":
        return None
    return problem_label(problem_code(found[0]))


_CONTENT_CODE_BY_REASON = {reason: code for code, reason in CONTENT_REASON_RU.items()}


def ethnicity_problems(prompt: str, n: int, texts: Optional[dict]) -> list:
    """Русский: нарушения внешности клипа n с фрагментами, как «код: «фрагмент»» (очередь исправлений).

    Окно проверки — текст сегментов n-1, n, n+1 (как у разбора). Нет texts или номера в нём — пусто.
    Фрагменты берутся из промпта без стиля (prompt_raw или prompt до стиля)."""
    if texts is None or n not in texts:
        return []
    window = " ".join(str(texts.get(k) or "") for k in (n - 1, n, n + 1))
    return [make_problem(PROBLEM_ETHNICITY, frag) for frag in ethnicity_hits(prompt, window)]


_ETHNICITY_REASON = CONTENT_REASON_RU[PROBLEM_ETHNICITY]   # причина внешности без фрагмента (разбор ответа)


def _merge_parse_reasons(found: list, reasons) -> list:
    """Добавляет к находкам find_problems причины разбора («разбор: причина») и не дублирует уже покрытое.

    Причина пропускается, если находка с тем же смыслом уже есть: содержание с тем же кодом; тег <imageN> без
    референсов при PROBLEM_REF_TAG; refs и метки при коде refs_invalid или label_mismatch. Этнос не пропускается:
    find_problems его не проверяет. Порядок: сначала находки, затем причины разбора по порядку появления."""
    codes = {problem_code(p) for p in found}
    out = list(found)
    for reason in dict.fromkeys(x.strip() for x in reasons):
        if not reason:
            continue
        code = _CONTENT_CODE_BY_REASON.get(reason)
        if code is not None and code in codes:
            continue
        if reason == REF_TAG_REASON and PROBLEM_REF_TAG in codes:
            continue
        if (reason.startswith("refs:") or reason in (LABEL_REASON, REF_REASON_COMBO)) \
                and codes & {REF_CODE_INVALID, REF_CODE_LABELS}:
            continue
        out.append(_PARSE_PREFIX + reason)
    return list(dict.fromkeys(out))


_DUP_REASON = "номер повторяется в ответе"   # причина дубликата номера: относится к формату ответа


def _run_batch(transport: GeminiTransport, model: str, builder: _Builder, batch: BatchRequest,
               tag: str = "", dump: Optional[Callable[[str, str], None]] = None) -> dict:
    """Запрос батча + один retry для номеров с нарушениями. Возвращает num -> PromptResult.

    tag: префикс строк журнала батча (имя волны в пуле, например «[видео] »); без пула пустой.
    dump: необязательный приёмник сырого ответа (диагностика): dump(суффикс, текст); «» — первый ответ,
    «r<k>» — повтор формата. Без него поведение прежнее."""
    expected = list(batch.nums)
    raw = transport.generate_json(model, batch.system, batch.user, batch.schema)
    if dump is not None:
        dump("", raw)   # Русский: сырой ответ до разбора; разбор и повтор ниже не меняются
    results, problems = parse_and_validate(raw, expected, builder.cfg, builder.texts)
    final: dict = {n: results[n] for n in expected if _is_clean(results, n, problems)}
    bad = [n for n in expected if n not in final]
    invalid_refs: dict = {}   # номер -> причины refs, не исправленные повтором (только refs_enabled)
    extras = [p for p in problems if p.num is not None and p.num not in set(expected)]
    if extras:
        log.warning("%sGemini вернул лишние номера: %d шт., они проигнорированы.", tag, len(extras))
    if builder.cfg.fail_fast and (bad or problems):
        # fail_fast: первая проблема батча останавливает прогон; повтора и исправлений нет
        bad_nums = sorted({p.num for p in problems if p.num is not None} | set(bad))
        slug = _parse_slug(problems, expected)
        # подпись правила — только для ниши с max_characters; у остальных ниш сообщение прежнее
        label = _segment_rule_label(problems) if (slug == "segment" and builder.cfg.max_characters is not None) else None
        _fail_fast_stop(builder.cfg, expected, bad_nums, slug, raw, label, system=batch.system, user=batch.user)

    # нарушения правил (номер с текстом и причинами) повтора не получают: клип с needs_review уходит
    # в очередь исправлений шага 6, как и раньше. Повтор — только для ответа без пригодного текста (формат)
    fmt_bad = [n for n in bad if n not in results or _DUP_REASON in results[n].review_reason]
    for n in bad:
        if n in fmt_bad:
            continue
        best = results[n]
        final[n] = PromptResult(n, best.prompt, True, best.review_reason, best.extra)
        if builder.cfg.refs_enabled:
            ref_left = [p.reason for p in problems if p.num == n and p.reason.startswith("refs:")]
            if ref_left:
                invalid_refs[n] = ref_left
    if fmt_bad:
        log.warning("%sБатч %s..%s: нет пригодного ответа по %d из %d записей, повторный запрос.",
                    tag, expected[0], expected[-1], len(fmt_bad), len(expected))
        retry = builder.batches(fmt_bad)  # fmt_bad <= batch_size: ровно один батч
        for k, rb in enumerate(retry, 1):
            note = _retry_note(rb.nums, problems, builder.cfg.max_characters, builder.cfg.rules.animals_count)
            if note:
                rb = replace(rb, user=rb.user + note)
            raw2 = transport.generate_json(model, rb.system, rb.user, rb.schema)
            if dump is not None:
                dump(f"r{k}", raw2)   # Русский: сырой ответ повтора формата (диагностика)
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
                # невалидный refs, переживший повтор: как метки, уходит в очередь исправлений (шаг 6)
                if builder.cfg.refs_enabled:
                    src = prob2 if n in res2 else problems   # та же версия, что и best
                    ref_left = [p.reason for p in src if p.num == n and p.reason.startswith("refs:")]
                    if ref_left:
                        invalid_refs[n] = ref_left
    # шаг 6 без запроса: запрещённые типы и маркеры помечаются очередью исправлений (rework_pending);
    # текст пока без стиля, исправление идёт пакетами после главных батчей (_WaveJob.process_fix)
    for n in list(final):
        r = final[n]
        if r.prompt.strip():
            found = _find_all_problems(r.prompt, builder.cfg, r.extra.get("refs"), builder.texts.get(n),
                                       setting=str(r.extra.get("setting") or ""))
            if n in invalid_refs:
                found = list(dict.fromkeys(found + _ref_fix_problems(r.prompt, None, invalid_refs[n])))
            if _rework_gate_open(builder.cfg, r):
                # Русский: клип с флагом разбора (стиль, длина, мусор, этнос) идёт в очередь, даже если find_problems
                # нашёл свою находку; причины разбора, не покрытые ею, добавляются (_merge_parse_reasons).
                # Внешность с фрагментом (по тексту сегментов n-1..n+1) идёт в очередь как нарушение, а не причина разбора.
                parse_reasons = r.review_reason.split("; ")
                eth = ethnicity_problems(r.prompt, n, builder.texts) if _ETHNICITY_REASON in parse_reasons else []
                if eth:
                    parse_reasons = [x for x in parse_reasons if x != _ETHNICITY_REASON]
                    found = list(dict.fromkeys(found + eth))
                found = _merge_parse_reasons(found, parse_reasons)
            if found:
                if builder.cfg.fail_fast:
                    # fail_fast: исправлений нет; сырой ответ этого батча — первый (других запросов не было)
                    _fail_fast_stop(builder.cfg, expected, [n], _content_slug(found), raw,
                                    _content_rule_label(found, builder.cfg.max_characters),
                                    system=batch.system, user=batch.user)
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
            log.warning("%s%s %s: попытка %d не удалась, класс ошибки %s%s.",
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


def openrouter_wave_threads(key_count: int, sends_per_window: Optional[int], n_batches: int) -> int:
    """Потоки одной волны OpenRouter: min(ключи × отправки, число батчей волны), не меньше одного.

    Один поток нужен и при нуле батчей: очередь исправлений волны всё равно выполняется потоком."""
    return max(1, min(openrouter_thread_count(key_count, sends_per_window), n_batches))


EMPTY_LENGTH_CODE = "EMPTY_RESPONSE_LENGTH"            # код ошибки OpenRouter: пустой ответ по лимиту вывода
EMPTY_LENGTH_REASON = "пустой ответ по лимиту вывода"  # причина needs_review для половины, упавшей снова


def _nums_ranges(nums) -> str:
    """Русский: номера клипов диапазонами, например «1–3, 5, 7–8»."""
    xs = sorted(int(n) for n in nums)
    if not xs:
        return "нет"
    parts, start = [], xs[0]
    prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        parts.append(str(start) if start == prev else f"{start}–{prev}")
        start = prev = x
    parts.append(str(start) if start == prev else f"{start}–{prev}")
    return ", ".join(parts)


_SUBTITLE_NOTE_CODES = frozenset({PROBLEM_DUPLICATE_PROMPT, PROBLEM_LETTERING_NOT_IN_SEGMENT, PROBLEM_DANGLING_TEXT,
                                  PROBLEM_ETHNICITY})


def _rework_subtitle_note(c: PromptResult, texts: dict, by_segment: Optional[dict] = None) -> str:
    """Текст субтитра текущего сегмента для повтора: при дубликате, надписи не из реплики, незаконченном тексте
    и при этносе из очереди (причина разбора с префиксом тоже учитывается).

    Русский: при дубликате короткого или незаконченного фрагмента добавляется предложение-контекст (by_segment),
    только для смысла сцены; слова из него в надпись не переносятся. Без by_segment — прежний текст."""
    reasons = {p[len(_PARSE_PREFIX):] if p.startswith(_PARSE_PREFIX) else p for p in c.rework_pending}
    need = any(problem_code(p) in _SUBTITLE_NOTE_CODES for p in c.rework_pending) \
        or CONTENT_REASON_RU[PROBLEM_ETHNICITY] in reasons
    if not need:
        return ""
    seg_text = " ".join(str(texts.get(c.num) or "").split())
    if not seg_text:
        return ""
    note = f" Subtitle text for segment_index {c.num}: {seg_text}."
    dup = any(problem_code(p) == PROBLEM_DUPLICATE_PROMPT for p in c.rework_pending)
    sen = (by_segment or {}).get(c.num)
    fragment = len(seg_text.split()) <= 4 or seg_text[-1] not in SENTENCE_END_CHARS
    if dup and fragment and sen is not None and getattr(sen, "text", ""):
        sentence = " ".join(str(sen.text).split())
        if sentence and sentence != seg_text:
            note += (" Sentence (for visual meaning only, never copy words from it into on-frame text): "
                     f"{sentence}.")
    return note


_dup_subtitle_note = _rework_subtitle_note   # прежнее имя: совместимость


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
        # Русский: размер пакета исправлений — fix_batch_size профиля, иначе FIX_BATCH_SIZE (25, как раньше)
        fbs = getattr(getattr(builder, "cfg", None), "fix_batch_size", None)
        self.fix_batch_size = fbs if isinstance(fbs, int) and not isinstance(fbs, bool) and fbs >= 1 else FIX_BATCH_SIZE
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
        self.labels: list = [str(i + 1) for i in range(len(batches))]  # метка батча: номер или x.1 / x.2
        self.is_half: list = [False] * len(batches)  # половина разделённого по лимиту вывода батча: не делится снова
        self.base_count = len(batches)    # число батчей волны до разделения (для строк «из N»)
        self.pkg_round: dict = {}         # номер пакета -> номер раунда (для журнала)
        self.round_left: dict = {}        # раунд -> пакетов, ещё не выполненных
        self.round_stats: dict = {}       # раунд -> счётчики для строки итога раунда
        self.stat_lock = threading.Lock() # счётчики раунда; короткий замок, без вызова модели
        self.round_pkgs: list = []        # пакеты текущего раунда исправлений, ещё не выданные: (номер пакета, клипы)
        self.round_inflight = 0           # пакеты раунда, которые выполняются сейчас
        self.round_no = 0                 # номер последнего раунда исправлений волны (с 1)

    def process(self, logged, idx: int) -> None:
        """Выполняет батч idx: запрос, стиль, журнал, затем out и on_batch под out_lock. Исключения не глотает."""
        label = self.labels[idx]   # номер батча или x.1 / x.2 для половины
        batch_no = label           # имя батча в строках журнала ниже
        batch = self.batches[idx]
        logged.start_batch(label)
        log.info("%sБатч %s из %d: начало, клипов в батче: %d.", self.tag, label, self.base_count,
                 len(batch.nums), extra=KEY_EVENT_EXTRA)
        t_batch = time.monotonic()
        def dump(suffix: str, text: str) -> None:
            # Русский: сырой ответ батча (круг 0); имя — волна, круг 0, метка батча и суффикс повтора
            _dump_response(self.builder.cfg, getattr(self, "wave", ""), 0, label + suffix, text)

        try:
            batch_res = _run_batch(logged, self.model, self.builder, batch, self.tag, dump=dump)
        except GeminiRuntimeError as exc:
            if getattr(exc, "code", "") != EMPTY_LENGTH_CODE:
                raise  # прочие ошибки: как раньше, волна останавливается
            if not self.is_half[idx] and len(batch.nums) > 1:
                self._split_batch(idx, batch)  # батч заменяется двумя половинами; результата пока нет
                return
            # половина (или один клип), упавшая снова: без деления, клипы на проверку с причиной
            log.warning("%sБатч %s: пустой ответ по лимиту вывода и без деления, клипы на проверку: %s.",
                        self.tag, label, _num_ranges(sorted(batch.nums)))
            batch_res = {n: PromptResult(n, "", True, EMPTY_LENGTH_REASON) for n in batch.nums}
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
            log.warning("%sБатч %s: промпты требуют проверки, клипы: %s.",
                        self.tag, batch_no, ", ".join(str(n) for n in review_nums))
        # клипы с нарушениями: в запись уходят со стилем, сырой текст в prompt_raw; в очередь исправлений — после записи
        queued = [batch_res[n] for n in batch.nums if n in batch_res and batch_res[n].rework_pending]
        for r in queued:
            # Русский: причина клипа в очереди — по его текущему тексту, той же функцией, что у исправления;
            # раньше здесь оставалась причина разбора, и она могла не совпадать с текстом
            r.review_reason = "; ".join(_problem_ru_any(p) for p in r.rework_pending)
        if queued:
            log.info("%sБатч %s: клипов с нарушениями, в очереди исправлений: %d.", self.tag, batch_no, len(queued))
        with_text = sum(1 for r in batch_res.values() if r.prompt.strip())
        log.info("%sБатч %s: получено промптов с текстом: %d из %d, за %.1f с.",
                 self.tag, batch_no, with_text, len(batch_res), time.monotonic() - t_batch,
                 extra=KEY_EVENT_EXTRA)
        with self.out_lock:
            self.out.update(batch_res)
            if self.on_batch is not None:
                self.on_batch([batch_res[n] for n in batch.nums])
        self.enqueue(queued)

    def _split_batch(self, idx: int, batch) -> None:
        """Пустой ответ по лимиту вывода: батч idx заменяется половинами x.1 и x.2 в конце очереди волны.

        Номера делятся пополам: первая половина n // 2 номеров (для 25 клипов — 12), вторая остальные (13).
        Каждая половина — один батч (не больше batch_size). Очередь выдаёт их как обычные батчи, в том числе разным ключам.
        """
        nums = sorted(batch.nums)
        cut = len(nums) // 2
        label = self.labels[idx]
        first = self.builder.batches(nums[:cut])[0]
        second = self.builder.batches(nums[cut:])[0]
        items = [(f"{label}.1", first), (f"{label}.2", second)]
        if self.pool is not None:
            self.pool.add_batches(self, items)
        else:
            with self.sched_lock:
                self.append_batches(items)
        log.warning("%sБатч %s: пустой ответ по лимиту вывода, батч разделён на %s.1 (%d клипов) и %s.2 (%d клипов).",
                    self.tag, label, label, len(first.nums), label, len(second.nums))

    def append_batches(self, items: list) -> None:
        """Добавляет батчи-половины в конец очереди волны. Вызывается под замком очереди (пул или одиночная волна)."""
        for label, b in items:
            self.batches.append(b)
            self.labels.append(label)
            self.is_half.append(True)

    def start_round(self) -> None:
        """Новый раунд исправлений: снимок очереди режется на пакеты по FIX_BATCH_SIZE по порядку.

        Раунд — это клипы очереди на момент начала. Клипы, которые вернутся с нарушениями, попадут в очередь
        и пойдут в следующий раунд. Вызывается под замком очереди (пул) или без потоков волны (одиночная волна).
        """
        clips = list(self.rework_queue)
        self.rework_queue.clear()
        self.round_no += 1
        pkgs = []
        for i in range(0, len(clips), self.fix_batch_size):
            self.fix_no += 1   # номер пакета волны: растёт и здесь, под замком очереди
            pkgs.append((self.fix_no, clips[i:i + self.fix_batch_size]))
            self.pkg_round[self.fix_no] = self.round_no
        self.round_pkgs = pkgs
        self.round_left[self.round_no] = len(pkgs)
        self.round_stats[self.round_no] = {"in": len(clips), "fixed": 0, "returned": 0, "review": 0,
                                           "t0": time.monotonic()}
        log.info("%sРаунд исправлений %d: клипов %d, пакетов %d.", self.tag, self.round_no, len(clips), len(pkgs))

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
        left = list(self.rework_queue)
        for _no, clips in self.round_pkgs:   # пакеты раунда, не выданные из-за остановки
            left.extend(clips)
        self.round_pkgs = []
        self.rework_queue.clear()
        if not left:
            return
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
        if builder.cfg.fail_fast:
            # fail_fast: исправлений нет; до запроса к модели исключение (защита на случай очереди из прошлого запуска)
            raise GeminiError("Ошибка: исправление промптов не запускается в режиме fail_fast.")
        nums = [c.num for c in clips]
        logged.start_batch(fix_no, "Фикс")   # номер попытки и номер пакета в строках повторов
        rnd = self.pkg_round.get(fix_no, self.round_no)
        log.info("%sФикс: круг %d, пакет %d, клипы %s, начало, клипов в пакете: %d.",
                 self.tag, rnd, fix_no, _nums_ranges(nums), len(clips), extra=KEY_EVENT_EXTRA)
        src_photos = builder.cfg.source_photo_prompts or {}
        durs = builder.cfg.segment_durations or {}
        lines = []
        for c in clips:
            line = (f"Prompt for segment_index {c.num}: {c.prompt_raw} {_rework_text(c.rework_pending)}"
                    + _rework_subtitle_note(c, builder.texts, getattr(builder, "by_segment", None))
                    # Русский: на 4-м и далее круге клипа только с надписями — кадр без текста; иначе строка прежняя
                    + (" " + TEXTLESS_REWORK_TEXT
                       if _textless_round(builder.cfg, c.rework_pending, (c.rework_round or 0) + 1) else "")
                    + (f" Refs for segment_index {c.num}: "
                       f"{json.dumps(c.extra['refs']) if 'refs' in c.extra else 'missing or invalid, return a valid refs list'}."
                       if builder.cfg.refs_enabled else ""))
            if c.num in durs and durs[c.num] is not None:
                line += f" Duration: {durs[c.num]:.2f}s."
            if c.num in src_photos and src_photos[c.num]:
                line += f" Source photo (animate ONLY what is already drawn here): {src_photos[c.num]}"
            lines.append(line)
        user = "\n".join(lines) + "\n\nReturn only the JSON array with one item per segment_index."
        try:
            t_req = time.monotonic()
            raw = logged.generate_json(self.model, builder.system, user, builder.schema)
            # Русский: сырой ответ исправления — до разбора (диагностика); круг rnd, номер пакета fix_no
            _dump_response(builder.cfg, getattr(self, "wave", ""), rnd, str(fix_no), raw)
            results, fix_problems = parse_and_validate(raw, nums, builder.cfg, builder.texts)
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
            empty_fix = not new_text.strip()
            if new_text.strip():
                c.prompt_raw = new_text   # пустой ответ: прежний сырой текст остаётся
            c.rework_round = (c.rework_round or 0) + 1
            rr_fix = results.get(c.num)
            if builder.cfg.refs_enabled and rr_fix is not None and rr_fix.extra.get("refs") is not None:
                c.extra["refs"] = rr_fix.extra["refs"]   # новый валидный refs ответа заменяет старый
            if rr_fix is not None and rr_fix.extra.get("setting"):
                c.extra["setting"] = rr_fix.extra["setting"]   # Русский: setting из ответа исправления обновляет запись
            ref_bad = [p.reason for p in fix_problems if p.num == c.num and p.reason.startswith("refs:")]
            found = _find_all_problems(c.prompt_raw, builder.cfg, c.extra.get("refs"),
                                       builder.texts.get(c.num),  # проверка по сырому тексту, до стиля
                                       textless=_textless_round(builder.cfg, c.rework_pending, c.rework_round),
                                       setting=str(c.extra.get("setting") or ""))
            if ref_bad:
                found = list(dict.fromkeys(found + _ref_fix_problems(c.prompt_raw, None, ref_bad)))
            # Флаг разбора ответа на исправление (стиль, длина, формат): клип без находок по сырому тексту, но с флагом,
            # ставится в очередь с этой причиной; на REPAIR_ROUNDS помечается на проверку, как и остальные клипы
            rr_now = results.get(c.num)
            parse_reasons = [p.reason for p in fix_problems if p.num == c.num]
            if not parse_reasons and rr_now is not None and rr_now.needs_review:
                parse_reasons = [x for x in (rr_now.review_reason or "ответ помечен на проверку").split("; ") if x]
            if _ETHNICITY_REASON in parse_reasons:
                # Русский: внешность из разбора — нарушение с фрагментом по сырому тексту клипа; без фрагментов
                # (нет окна текста) остаётся причиной разбора, как раньше
                eth = ethnicity_problems(c.prompt_raw, c.num, builder.texts)
                if eth:
                    parse_reasons = [x for x in parse_reasons if x != _ETHNICITY_REASON]
                    found = list(dict.fromkeys(found + eth))
            if empty_fix and c.rework_pending:
                # Русский: модель не вернула текст сегмента — прежняя очередь остаётся, причина «номер отсутствует»
                # или «пустой prompt» не пишется; prompt_raw не менялся, поэтому очередь прежняя
                found = list(c.rework_pending)
                parse_reasons = []
            found = _merge_parse_reasons(found, parse_reasons)
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
                c.review_reason = "; ".join(_problem_ru_any(p) for p in found)
                log.warning("%sКлип %d: не исправлено за %d кругов: %s.",
                            self.tag, c.num, REPAIR_ROUNDS, c.review_reason)
            else:
                # Русский: в очередь уходит текст этого круга, поэтому причина — тоже этого круга; раньше
                # review_reason оставался от прошлого круга, и on_batch писал старую причину рядом с новым prompt
                c.rework_pending = found
                c.needs_review = False   # в очереди клип ещё не на проверке: остался круг
                c.review_reason = "; ".join(_problem_ru_any(p) for p in found)
                requeue.append(c)
        ordered = sorted(clips, key=lambda c: c.num)
        with self.out_lock:
            self.out.update({c.num: c for c in ordered})
            if self.on_batch is not None:
                self.on_batch(ordered)
        # Русский: строка пакета (запрос, исходы, причины по номерам) и итог раунда, когда раунд выполнен
        req_s = time.monotonic() - t_req
        back_ids = {id(c) for c in requeue}
        left = [c for c in clips if c.needs_review and id(c) not in back_ids]
        detail = "; ".join(f"{c.num}: {c.review_reason}" for c in left) or "нет"
        log.info("%sФикс: круг %d, пакет %d, запрос %.1f с, исправлено %d, в очередь %d, на проверке %d (%s).",
                 self.tag, rnd, fix_no, req_s, fixed, len(requeue), len(left), detail)
        with self.stat_lock:
            st = self.round_stats.get(rnd)
            if st is not None:
                st["fixed"] += fixed
                st["returned"] += len(requeue)
                st["review"] += len(left)
                self.round_left[rnd] -= 1
                if self.round_left[rnd] == 0:
                    log.info("Раунд исправлений %d: на входе клипов %d, исправлено %d, возвращено %d, "
                             "на проверке %d, длительность %.1f с.",
                             rnd, st["in"], st["fixed"], st["returned"], st["review"],
                             time.monotonic() - st["t0"])
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

    def __init__(self, shares: dict, bind: Optional[Callable] = None, per_wave_cap: bool = False) -> None:
        self._shares = dict(shares)
        self._bind = bind
        # per_wave_cap: число потоков волны — её доля (ключи × отправки), но не больше числа её батчей; чужая волна не делит
        self._per_wave_cap = per_wave_cap
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
            if self._per_wave_cap:
                # Русский: батчей может не быть (только исправления) — тогда потоки считаются по пакетам исправлений
                rework_pkgs = ((len(job.rework_queue) + job.fix_batch_size - 1) // job.fix_batch_size
                               if job.rework_queue else 0)
                return max(1, min(int(self._shares[wave]), max(len(job.batches), rework_pkgs)))
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

    def add_batches(self, job: "_WaveJob", items: list) -> None:
        """Добавляет половины батчей волны в очередь пула и будит потоки."""
        with self._cond:
            job.append_batches(items)
            self._cond.notify_all()

    def _take(self, own: str):
        """Следующая работа потока: ('main', волна, idx) или ('fix', волна, (номер пакета, клипы)); None, когда работы больше не будет.

        Порядок: 1) полный пакет исправлений (FIX_BATCH_SIZE клипов) — раньше главного батча, даже если главные ещё есть;
        2) главный батч (своя волна, затем чужая); 3) остаток исправлений (меньше FIX_BATCH_SIZE) — только когда нигде
        не выполняется главный батч. Иначе поток ждёт в wait(); замок при ожидании отпускается. Ошибка пула останавливает выдачу.
        """
        with self._cond:
            while True:
                if self._fatal is not None:
                    return None
                fix = self._pick_fix(own, remainder=False)
                if fix is not None:
                    job, clips, fix_no = fix
                    return "fix", job, (fix_no, clips)
                main = self._pick_main(own)
                if main is not None:
                    job, idx = main
                    return "main", job, idx
                fix = self._pick_fix(own, remainder=True)
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

    def _pick_fix(self, own: str, remainder: bool):
        """Под замком очереди: пакет исправлений одной волны (своя, затем чужая) или None.

        remainder=False: только полный пакет (FIX_BATCH_SIZE клипов). Полный пакет снимка раунда выдаётся сразу;
        новый раунд начинается, когда пакетов раунда не осталось, а в очереди не меньше FIX_BATCH_SIZE клипов.
        Барьера раунда нет: клип, вернувшийся с нарушением, сразу попадает в очередь и уходит в следующий пакет.
        remainder=True: остаток (меньше FIX_BATCH_SIZE) — только когда нигде не выполняется главный батч.
        """
        order = [own] + [w for w in self._shares if w != own]
        alive = []
        for w in order:
            job = self._jobs.get(w)
            if job is not None and not job.stopped:
                alive.append(job)
        if not remainder:
            for job in alive:
                if job.round_pkgs and len(job.round_pkgs[0][1]) >= job.fix_batch_size:
                    return self._cut_fix(job)
            for job in alive:
                if not job.round_pkgs and len(job.rework_queue) >= job.fix_batch_size:
                    return self._cut_fix(job)
            return None
        if any(j.main_inflight for j in self._jobs.values() if j is not None):
            return None
        for job in alive:
            if job.round_pkgs or job.rework_queue:
                return self._cut_fix(job)
        return None

    @staticmethod
    def _cut_fix(job: "_WaveJob"):
        """Под замком очереди: выдаёт следующий пакет раунда волны; раунд начинается здесь, если пакетов нет."""
        if not job.round_pkgs:
            job.start_round()
        fix_no, clips = job.round_pkgs.pop(0)
        job.inflight += 1
        job.round_inflight += 1
        return job, clips, fix_no

    def _idle(self) -> bool:
        """Под замком очереди: в не остановленных волнах не осталось ни батчей, ни исправлений, ни выполнений."""
        for job in self._jobs.values():
            if job is None or job.stopped:
                continue
            if job.next_idx < len(job.batches) or job.rework_queue or job.round_pkgs or job.inflight:
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
            if kind == "fix":
                job.round_inflight -= 1   # пакет раунда завершён (с ошибкой или без): барьер раунда
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
                or (job.next_idx >= len(job.batches) and not job.rework_queue and not job.round_pkgs)))

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
    style_text = preset_text(cfg.preset, cfg.niche_preset_file)
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
        if provider == "openrouter":
            log.info("Потоки OpenRouter: ключей %d, отправок в окне %s, потоков %d.",
                     key_count, sends_per_window if sends_per_window is not None else "не задано", n_workers)

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
        # исправления очереди этой волны: потоков уже нет; раунды по барьеру, пакеты раунда — параллельно
        # Русский: потоки исправлений берутся по ключам, а не по числу главных батчей (при пустых batches — тоже)
        if provider == "openrouter":
            fix_max_workers = max(1, max_workers)
        else:
            fix_max_workers = max(1, min(max_workers, MAX_PARALLEL_BATCHES))
        while job.rework_queue and not first_error:
            job.start_round()
            fix_threads = max(1, min(fix_max_workers, len(job.round_pkgs)))
            if fix_threads > 1:
                log.info("Исправления очереди выполняются в %d потоках, пакетов: %d.", fix_threads, len(job.round_pkgs))

            def fix_worker() -> None:
                logged_fix = _LoggedTransport(transport)   # журнал свой у каждого потока
                while True:
                    with queue_lock:
                        if first_error or not job.round_pkgs:
                            return
                        fix_no, clips = job.round_pkgs.pop(0)
                    try:
                        job.process_fix(logged_fix, clips, fix_no)
                    except BaseException as exc:
                        with queue_lock:
                            if not first_error:
                                first_error.append(exc)
                        return

            if fix_threads == 1:
                fix_worker()
            else:
                fix_ths = [threading.Thread(target=fix_worker, name=f"gemini-fix-{i + 1}")
                           for i in range(fix_threads)]
                for th in fix_ths:
                    th.start()
                for th in fix_ths:
                    th.join()
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
    # барьер после кругов исправления: без референсов метки <imageN> не уходят в рендер (для refs_enabled не выполняется)
    _strip_image_tags_without_refs(results, cfg)
    if cfg.on_unfixed == "fail":
        _fail_unfixed(results, cfg)
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