"""Файл персонажей и объектов одного SRT: characters.json (H3, часть A).

Окно = WINDOW_SEGMENTS сегментов подряд, на каждое окно один запрос Gemini. Ответы сливаются в записи:
персонажи и объекты получают стабильные id (CHAR_N, OBJ_N), известная запись сохраняет внешность,
к ней дописываются новые постоянные черты, противоречие не записывается и попадает в лог.
Запрос идёт через тот же транспорт, что и генерация промптов (метод generate_json; в оркестраторе это
ManagedTransport с лимитером и повторами). Ключ здесь не читается.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

try:  # пакетный и «плоский» импорт (как в остальных модулях core)
    from . import gemini_prompts as gp
    from .srt_parser import Segment, srt_hash
    from .gemini_orchestrator import atomic_write_json
except ImportError:  # pragma: no cover
    import gemini_prompts as gp  # type: ignore
    from srt_parser import Segment, srt_hash  # type: ignore
    from gemini_orchestrator import atomic_write_json  # type: ignore

log = logging.getLogger("Generate.character_file")

WINDOW_SEGMENTS = 200
CHARACTERS_FILENAME = "characters.json"
PARTIAL_FILENAME = "characters.partial.json"
MAX_NAME_LEN = 40
# Та же форма, что маркер [ИМЯ] в gemini_prompts: заглавные латинские и кириллические буквы, цифры, «_».
_NAME_RX = re.compile(r"[A-Z\u0410-\u042f\u0401][A-Z0-9_\u0410-\u042f\u0401]*")
_STR_LIST = {"type": "array", "items": {"type": "string"}}
_LIST_FIELDS = ("aliases", "keywords", "new_traits", "conflicts")
# Общая форма элемента персонажа и объекта.
_ITEM_PROPS = {"name": {"type": "string"}, "aliases": _STR_LIST, "keywords": _STR_LIST,
               "new_traits": _STR_LIST, "conflicts": _STR_LIST}
_ITEM_REQUIRED = ["name", "aliases", "keywords", "new_traits", "conflicts"]
# Владелец — обычная строка (3a): пустая строка означает «нет владельца». null в схеме не используется.
_OWNER_SCHEMA = {"type": "string"}

# Схема ответа одного окна (responseJsonSchema). Форма дублирует текст инструкции ниже.
WINDOW_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": dict(_ITEM_PROPS),
                "required": list(_ITEM_REQUIRED),
            },
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {**_ITEM_PROPS, "owner": _OWNER_SCHEMA},
                "required": [*_ITEM_REQUIRED, "owner"],
            },
        },
    },
    "required": ["characters"],
}


def build_character_instructions() -> str:
    """Полный текст инструкции для одного окна: роль, правила персонажей и объектов, правило внешности,
    правило владельца, форма ответа. Правила записаны здесь один раз."""
    lines = [
        "You list the people and the important objects that appear in the subtitle text of one video window. "
        "Each one is kept in a record that lasts across windows.",
        "Rules for characters:",
        "- name: one word in uppercase Latin or Cyrillic letters, no spaces, for example ANNA or АННА. "
        "For a character already in the record, use the existing name exactly.",
        "- aliases: other names this same person has in the text (may be empty).",
        "- keywords: words or short phrases that appear in the window text and identify this person.",
        "- new_traits: permanent appearance only: age range, build, hair, face, beard.",
        "- For a character already in the record, new_traits holds only permanent traits that the recorded "
        "appearance does not contain yet, and conflicts holds permanent traits that contradict it.",
        "- For a character not in the record, all permanent traits go into new_traits and conflicts is empty.",
        "- If the name is ambiguous, choose the most likely historical person from the context of the text; "
        "do not add any identifier beyond the name.",
        "Rules for objects:",
        "- List an object only when it matters to the plot of this window, not because a word repeats often. "
        "A background mention with no name and no further detail is not listed. "
        "An object central to the scene is always listed.",
        "- name: one word, same form as character names. For an object already in the record, use the "
        "existing name exactly.",
        "- aliases and keywords: as for characters.",
        "- new_traits: permanent properties only: material, shape, color, size, and other fixed features.",
        "- owner: the name of the character who owns the object, exactly as in the characters list, "
        "or an empty string when no listed character owns it.",
        "- For an object already in the record, new_traits and conflicts follow the same rules as for characters.",
        "- For an object not in the record, all permanent properties go into new_traits and conflicts is empty.",
        "Rules for everyone:",
        "- Do not describe clothing or temporary state (wet, tired, and similar), for people or objects.",
        "Appearance rule:",
        "- For every new character and every new object, new_traits always contains at least age range, build, "
        "hair, and facial features for a person, or the main fixed properties for an object, even when the "
        "window text has no visual details. Base it on what you know about the person or object by name and "
        "on the context of the subtitles.",
        "- Write each new_traits item as one short noun phrase in lowercase, one trait per item, with an article where it fits, "
        "with no final period, at most 25 words. Do not write full sentences.",
        "- Do not add any uncertainty marker or extra field.",
        "- If nothing relevant appears in the text, return empty lists.",
        "Return only one JSON object, with no markdown, no code fences and no text before or after it.",
        "Return only JSON in this exact shape:",
        '{"characters": [{"name": "ANNA", "aliases": ["ANNA"], "keywords": ["Anna"], '
        '"new_traits": ["a woman in her thirties with short dark hair"], "conflicts": []}], '
        '"objects": [{"name": "HOUSE", "aliases": [], "keywords": ["house"], '
        '"new_traits": ["a two-storey brick house with white shutters"], "conflicts": [], "owner": ""}]}',
    ]
    return "\n".join(lines)


def _is_valid_name(name: str) -> bool:
    return 0 < len(name) <= MAX_NAME_LEN and _NAME_RX.fullmatch(name) is not None


def _window_request(text: str, known: list, known_objects: list = ()) -> str:
    """Текст запроса одного окна: текст окна, уже известные персонажи и объекты с id, список имён.
    Правила здесь не пишутся: они в build_character_instructions()."""
    owner_names = {c["id"]: c["name"] for c in known if c.get("id")}
    recorded = "\n".join(f"{c.get('id', '')} {c['name']}: {c.get('appearance', '')}" for c in known) \
        or "(none yet)"
    objs = "\n".join(
        f"{o.get('id', '')} {o['name']} (owner: {owner_names.get(o.get('owner'), 'none')}): "
        f"{o.get('appearance', '')}" for o in known_objects) or "(none yet)"
    names = [c["name"] for c in known] + [o["name"] for o in known_objects]
    return (
        "Window text (one subtitle segment per line):\n" + text + "\n\n"
        "Characters already in the record (id name: recorded appearance):\n" + recorded + "\n\n"
        "Objects already in the record (id name (owner: character name): recorded appearance):\n" + objs + "\n\n"
        "Names already in the record: " + json.dumps(names, ensure_ascii=False)
    )


def _parse_items(items: list, low: str, what: str, *, owner: bool) -> list:
    """Разбор списка персонажей или объектов. Проверка формы; пустая внешность новой записи — после слияния."""
    out: list = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"элемент {what} не объект")
        name = item.get("name")
        if not isinstance(name, str):
            raise ValueError(f"поле name не строка ({what})")
        for key in _LIST_FIELDS:
            value = item.get(key)
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise ValueError(f"поле {key} не список строк")
        owner_name = None
        if owner:
            raw_owner = item.get("owner")
            if raw_owner is not None and not isinstance(raw_owner, str):
                raise ValueError("поле owner не строка и не null")
            owner_name = raw_owner.strip() if isinstance(raw_owner, str) and raw_owner.strip() else None
        nm = name.strip().upper()
        if not _is_valid_name(nm):
            log.warning("Имя (%s) не подходит по формату маркера, запись пропущена.", what)
            continue
        new_traits = [t.strip() for t in item["new_traits"] if t.strip()]
        entry = {
            "name": nm,
            "aliases": [a.strip() for a in item["aliases"] if a.strip()],
            "keywords": [k.strip() for k in item["keywords"] if k.strip() and k.strip().lower() in low],
            "new_traits": new_traits,
            "conflicts": [t.strip() for t in item["conflicts"] if t.strip()],
        }
        if owner:
            entry["owner"] = owner_name
        out.append(entry)
    return out


def parse_window_reply(raw: str, window_text: str) -> dict:
    """Строгий разбор ответа одного окна. Любое нарушение формы: ValueError с коротким описанием.

    Возвращает {characters: [...], objects: [...]}. Отсутствующий ключ objects читается как пустой список.
    Имя приводится к верхнему регистру и проверяется по форме маркера; неподходящее имя пропускается с записью
    в лог. Ключевые слова оставляются только те, что есть в тексте окна (без учёта регистра).
    Новая запись без постоянной внешности проверяется после слияния в _ask_window: тогда окно пойдёт повторно."""
    if not isinstance(raw, str):
        raise ValueError("ответ не строка")
    data = json.loads(raw)  # не JSON -> json.JSONDecodeError (подкласс ValueError)
    if not isinstance(data, dict) or not isinstance(data.get("characters"), list):
        raise ValueError("ожидался объект с массивом characters")
    objects_raw = data.get("objects", [])
    if not isinstance(objects_raw, list):
        raise ValueError("поле objects не список")
    low = window_text.lower()
    return {
        "characters": _parse_items(data["characters"], low, "персонаж", owner=False),
        "objects": _parse_items(objects_raw, low, "объект", owner=True),
    }


# ---------------------------------------------------------------------------
# id и миграция
# ---------------------------------------------------------------------------

def _id_number(rid: Any, prefix: str) -> int:
    """Номер из id вида PREFIX_N; 0, если id отсутствует или не той формы."""
    if not isinstance(rid, str):
        return 0
    m = re.fullmatch(prefix + r"_([0-9]+)", rid)
    return int(m.group(1)) if m else 0


def _new_id(records: list, prefix: str) -> str:
    """Следующий свободный id: на единицу больше наибольшего номера в записях."""
    top = max([_id_number(r.get("id"), prefix) for r in records] + [0])
    return f"{prefix}_{top + 1}"


def _assign_ids(records: list, prefix: str) -> None:
    """Дописывает id записям, у которых его нет, по порядку позиции в списке (меняет список)."""
    for rec in records:
        if not _id_number(rec.get("id"), prefix):
            rec["id"] = _new_id(records, prefix)


def migrate_record(record: dict) -> dict:
    """Миграция записи characters.json к виду с id и объектами. Вход не меняется.

    Записи персонажей без id получают CHAR_N, объекты без id получают OBJ_N, оба по порядку позиции в файле.
    Если ключа objects нет, он появляется пустым. Недостающие aliases, keywords, appearance (у объектов ещё
    owner) заполняются пустыми значениями."""
    out = copy.deepcopy(record)
    characters = out.get("characters") or []
    objects = out.get("objects") or []
    for rec in characters:
        rec.setdefault("aliases", [])
        rec.setdefault("keywords", [])
        rec.setdefault("appearance", "")
    _assign_ids(characters, "CHAR")
    for rec in objects:
        rec.setdefault("owner", None)
        rec.setdefault("aliases", [])
        rec.setdefault("keywords", [])
        rec.setdefault("appearance", "")
    _assign_ids(objects, "OBJ")
    out["characters"] = characters
    out["objects"] = objects
    return out


# ---------------------------------------------------------------------------
# Слияние (чистые функции: вход не меняется)
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .!?")


def _sentence(text: str) -> str:
    """Одна черта в виде фразы для записи: без точки в конце, первая буква заглавная.
    Несколько черт склеивает _add_trait через «, and »; точка между ними не ставится."""
    t = normalize_appearance(text)
    return t[:1].upper() + t[1:] if t else ""


def _lower_first(piece: str) -> str:
    """Первая буква строчная; первое слово-аббревиатуру (два и больше заглавных подряд) не трогаем."""
    word = piece.split(" ", 1)[0]
    if len(word) >= 2 and word.isalpha() and word.isupper():
        return piece
    return piece[:1].lower() + piece[1:]


def normalize_appearance(text: str) -> str:
    """Внешность в виде фразы для вставки в середину промпта.

    Убирает лишние пробелы и одну точку в конце. Внутренние «. » между чертами заменяются на «, and »,
    каждая черта начинается со строчной буквы: старые записи хранят черты предложениями
    («Man in his fifties. Grey beard.»), а в промпте нужна одна фраза. Точка не разрывает фразу, если стоит
    после одной буквы («U.S. army»). Аббревиатура в начале черты сохраняется.
    Читается при каждом использовании, поэтому старые characters.json работают без пересборки."""
    t = " ".join(str(text or "").split())
    if t.endswith("."):
        t = t[:-1].rstrip()
    pieces = [p for p in re.split(r"(?<!\b[A-Za-z])\.\s+", t) if p]
    return ", and ".join(_lower_first(p) for p in pieces)


def _add_trait(appearance: str, trait: str) -> str:
    """Дописывает черту к внешности через «, and », если её там ещё нет (по нормализованному тексту).
    Старая запись с точками читается через normalize_appearance и переписывается в новую форму."""
    sen = _sentence(trait)
    n = _norm(trait)
    if not sen or (n and n in _norm(appearance)):
        return appearance
    base = normalize_appearance(appearance)
    if not base:
        return sen
    return _sentence(base + ", and " + _lower_first(sen))


def _apply_traits(rec: dict, item: dict) -> None:
    """Общая часть слияния персонажа и объекта: ключевые слова, новые черты, конфликты (меняет rec)."""
    for k in item["keywords"]:
        if k not in rec["keywords"]:
            rec["keywords"].append(k)
    for t in item["new_traits"]:
        rec["appearance"] = _add_trait(rec["appearance"], t)
    for t in item["conflicts"]:
        log.warning("conflict, not recorded: %s, %s", rec["name"], t)


def merge_characters(characters: list, extracted: list) -> list:
    """Сливает разобранные элементы окна в список записей. Чистая функция, Gemini не вызывается.

    Правила:
    - персонаж ищется по имени или псевдониму; известный сохраняет id и внешность, новые постоянные черты
      дописываются в конец, черта, уже присутствующая в внешности, пропускается;
    - черта из conflicts не записывается и логируется: «conflict, not recorded: <имя>, <черта>»;
    - новый персонаж заводится новой записью с очередным id (CHAR_N) и всеми своими чертами.
    Единственный побочный эффект — строки лога."""
    out = copy.deepcopy(characters)
    _assign_ids(out, "CHAR")
    for rec in out:
        rec.setdefault("aliases", [])
        rec.setdefault("keywords", [])
        rec.setdefault("appearance", "")
    index: dict = {}  # имя или псевдоним в верхнем регистре -> позиция записи
    for pos, rec in enumerate(out):
        for key in [rec["name"], *rec["aliases"]]:
            index.setdefault(key.upper(), pos)

    for item in extracted:
        name = item["name"]
        pos = index.get(name.upper())
        if pos is None:
            aliases = list(dict.fromkeys(a for a in item["aliases"] if a.upper() != name.upper()))
            rec = {"id": _new_id(out, "CHAR"), "name": name, "aliases": aliases,
                   "keywords": [], "appearance": ""}
            _apply_traits(rec, item)
            out.append(rec)
            pos = len(out) - 1
            for key in [name, *aliases]:
                index.setdefault(key.upper(), pos)
            continue

        rec = out[pos]
        for a in item["aliases"]:
            if a.upper() not in index:
                rec["aliases"].append(a)
                index[a.upper()] = pos
        _apply_traits(rec, item)
    return out


def _owner_index(characters: list) -> dict:
    """Имя или псевдоним персонажа в верхнем регистре -> его id."""
    index: dict = {}
    for rec in characters:
        for key in [rec["name"], *rec.get("aliases", [])]:
            index.setdefault(key.upper(), rec["id"])
    return index


def merge_objects(objects: list, extracted: list, characters: list) -> list:
    """Сливает разобранные объекты окна в список записей. Чистая функция, Gemini не вызывается.

    Объект ищется по владельцу и имени (или псевдониму): владелец — id персонажа из characters, или None.
    Два объекта с одним именем и разными владельцами — две записи. Два объекта без владельца с одним именем —
    одна запись. Имя владельца, которого нет среди персонажей, даёт владельца None и лог
    «owner not found, set to null: <имя объекта>». Слияние черт и конфликтов то же, что у персонажей."""
    out = copy.deepcopy(objects)
    _assign_ids(out, "OBJ")
    for rec in out:
        rec.setdefault("owner", None)
        rec.setdefault("aliases", [])
        rec.setdefault("keywords", [])
        rec.setdefault("appearance", "")
    owners = _owner_index(characters)
    index: dict = {}  # (id владельца или None, имя или псевдоним в верхнем регистре) -> позиция записи
    for pos, rec in enumerate(out):
        for key in [rec["name"], *rec["aliases"]]:
            index.setdefault((rec["owner"], key.upper()), pos)

    for item in extracted:
        name = item["name"]
        owner = None
        if item.get("owner"):
            owner = owners.get(item["owner"].upper())
            if owner is None:
                log.warning("owner not found, set to null: %s", name)
        pos = index.get((owner, name.upper()))
        if pos is None:
            aliases = list(dict.fromkeys(a for a in item["aliases"] if a.upper() != name.upper()))
            rec = {"id": _new_id(out, "OBJ"), "owner": owner, "name": name, "aliases": aliases,
                   "keywords": [], "appearance": ""}
            _apply_traits(rec, item)
            out.append(rec)
            pos = len(out) - 1
            for key in [name, *aliases]:
                index.setdefault((owner, key.upper()), pos)
            continue

        rec = out[pos]
        for a in item["aliases"]:
            if (owner, a.upper()) not in index:
                rec["aliases"].append(a)
                index[(owner, a.upper())] = pos
        _apply_traits(rec, item)
    return out


# ---------------------------------------------------------------------------
# Окна, запросы, сборка записи
# ---------------------------------------------------------------------------

class CharacterWindowError(gp.GeminiError):
    """Окно сегментов не дало разобранного ответа даже после повтора. Прогон останавливается.

    characters.json в этом случае не пишется; готовые окна лежат в characters.partial.json.
    Поля: first и last, номера первого и последнего сегмента окна. Текст ответа Gemini не сохраняется."""
    exit_code = 3

    def __init__(self, first: int, last: int):
        super().__init__(f"Окно персонажей, сегменты {first}..{last}, не дало верного ответа после повтора. "
                         "Прогон остановлен, characters.json не записан.")
        self.first = first
        self.last = last


class EmptyAppearanceError(CharacterWindowError):
    """Новая запись (персонаж или объект) без внешности и после повтора окна. Запись не сохраняется.

    Поля: record_name — имя записи, first и last — сегменты окна. Наследует CharacterWindowError (код выхода 3)."""

    def __init__(self, record_name: str, first: int, last: int):
        gp.GeminiError.__init__(self, f"Запись «{record_name}» без внешности после повтора, окно сегментов "
                                f"{first}..{last}. Запись не сохранена, characters.json не записан.")
        self.record_name = record_name
        self.first = first
        self.last = last


def _first_empty_new(old_chars: list, old_objs: list, new_chars: list, new_objs: list) -> Optional[str]:
    """Имя первой новой записи без внешности или None. Новая — запись, чьего id не было до слияния."""
    for old, new in ((old_chars, new_chars), (old_objs, new_objs)):
        old_ids = {r.get("id") for r in old}
        for rec in new:
            if rec.get("id") not in old_ids and not str(rec.get("appearance") or "").strip():
                return rec.get("name")
    return None


def _ask_window(client: Any, model: str, text: str, characters: list, objects: list,
                first: int, last: int) -> tuple:
    """Один запрос на окно, слияние и проверка внешности новых записей. Возвращает (characters, objects).

    Нарушение формы или новая запись без внешности: ещё один запрос (всего два). Два провала: CharacterWindowError;
    если последний провал — пустая внешность, то EmptyAppearanceError с именем записи. Слияние выполняется здесь,
    чтобы проверка учитывала владельца объекта (4a).

    Ошибки транспорта (GeminiRuntimeError) не перехватываются: квота и сбои сети ManagedTransport
    обрабатывает сам и не превращает их в пропущенное окно."""
    system = build_character_instructions()
    user = _window_request(text, characters, objects)
    empty_name: Optional[str] = None
    for attempt in (1, 2):
        raw = client.generate_json(model, system, user, WINDOW_SCHEMA)
        try:
            extracted = parse_window_reply(raw, text)
        except ValueError as exc:
            empty_name = None
            log.warning("Окно сегментов %s..%s: ответ не прошёл проверку (%s), попытка %d из 2.",
                        first, last, exc, attempt)
            continue
        new_chars = merge_characters(characters, extracted["characters"])
        new_objs = merge_objects(objects, extracted["objects"], new_chars)
        empty_name = _first_empty_new(characters, objects, new_chars, new_objs)
        if empty_name is None:
            return new_chars, new_objs
        log.warning("Окно сегментов %s..%s: новая запись «%s» без внешности, попытка %d из 2.",
                    first, last, empty_name, attempt)
    if empty_name is not None:
        log.error("Окно сегментов %s..%s: запись «%s» без внешности после повтора, прогон остановлен.",
                  first, last, empty_name)
        raise EmptyAppearanceError(empty_name, first, last)
    log.error("Окно сегментов %s..%s: два ответа подряд не прошли проверку, прогон остановлен.", first, last)
    raise CharacterWindowError(first, last)


def _window_ranges(segments: list) -> list:
    """(первый номер, последний номер, текст окна) для каждого окна по WINDOW_SEGMENTS сегментов."""
    out = []
    for start in range(0, len(segments), WINDOW_SEGMENTS):
        window = segments[start:start + WINDOW_SEGMENTS]
        out.append((window[0].num, window[-1].num, "\n".join(s.text for s in window if s.text)))
    return out


def _load_finished(raw: Any) -> list:
    """Список завершённых окон из частичного файла: только пары целых чисел first и last."""
    out = []
    for item in raw or []:
        if isinstance(item, dict) and isinstance(item.get("first"), int) and isinstance(item.get("last"), int):
            out.append({"first": item["first"], "last": item["last"]})
    return out


def build_character_file(segments: list, client: Any, existing: dict, *, model: str = "",
                         on_window: Optional[Callable[[dict], None]] = None) -> dict:
    """Строит запись о персонажах и объектах по окнам, по одному запросу на окно с текстом.

    existing: частичная запись прошлого запуска ({srt_hash, characters, objects, finished}). Если её srt_hash не
    совпадает с текущим SRT, она отбрасывается. Старая запись без id и без objects мигрирует при чтении.
    Окна из finished пропускаются без запроса. Результат каждого окна сливается до следующего окна.
    После каждого успешного окна вызывается on_window с текущей записью. Ошибка окна (CharacterWindowError)
    выходит наружу без возврата. Возвращает dict: {srt_hash, characters, objects, finished}."""
    current = srt_hash(segments)
    if existing and existing.get("srt_hash") != current:
        log.warning("Частичный файл персонажей относится к другому SRT: запись строится заново.")
        existing = {}
    base = migrate_record(existing) if existing else {}
    characters = base.get("characters") or []
    objects = base.get("objects") or []
    finished = _load_finished(existing.get("finished")) if existing else []
    done = {(f["first"], f["last"]) for f in finished}

    for first, last, text in _window_ranges(segments):
        if (first, last) in done:
            continue
        if text:
            characters, objects = _ask_window(client, model, text, characters, objects, first, last)
        finished.append({"first": first, "last": last})
        done.add((first, last))
        if on_window is not None:
            on_window({"srt_hash": current, "characters": characters, "objects": objects, "finished": finished})
    return {"srt_hash": current, "characters": characters, "objects": objects, "finished": finished}


def load_character_file(path) -> Optional[dict]:
    """Читает characters.json. Нет файла или его форма неверна: None (повреждённый файл пересоберётся)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("Файл персонажей повреждён и будет построен заново: %s.", path.name)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("characters"), list) \
            or not isinstance(data.get("srt_hash"), str):
        log.warning("Файл персонажей имеет неверный формат и будет построен заново: %s.", path.name)
        return None
    return data


def load_partial(path) -> Optional[dict]:
    """Читает characters.partial.json. Нет файла или его форма неверна: None (окна пойдут заново)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("Частичный файл персонажей повреждён и будет построен заново: %s.", path.name)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("characters"), list) \
            or not isinstance(data.get("srt_hash"), str):
        log.warning("Частичный файл персонажей имеет неверный формат и будет построен заново: %s.", path.name)
        return None
    return data


def get_character_file(out_dir, segments: list, client: Any, *, model: str = "") -> dict:
    """Файл персонажей для SRT в папке out_dir (рядом с prompts.json).

    Если characters.json есть и его srt_hash совпадает с текущим, возвращается он; Gemini не вызывается.
    Иначе окна обрабатываются с частичным файлом characters.partial.json: готовые окна пропускаются,
    после каждого нового окна частичный файл записывается атомарно. characters.json пишется только
    когда все окна готовы; тогда частичный файл удаляется. При ошибке окна исключение выходит наружу,
    characters.json не пишется."""
    out_dir = Path(out_dir)
    path = out_dir / CHARACTERS_FILENAME
    partial_path = out_dir / PARTIAL_FILENAME
    current = srt_hash(segments)
    stored = load_character_file(path)
    if stored is not None and stored.get("srt_hash") == current:
        # 5a: миграция (id, objects, owner) при совпавшем srt_hash; запись обратно только если что-то изменилось
        migrated = migrate_record(stored)
        final = {"srt_hash": stored["srt_hash"], "characters": migrated["characters"],
                 "objects": migrated["objects"]}
        if final != stored:
            atomic_write_json(path, final)
            log.info("Файл персонажей дополнен id и объектами (миграция): %s.", CHARACTERS_FILENAME)
        log.info("Файл персонажей для этого SRT уже есть (%s): Gemini не вызывается.", CHARACTERS_FILENAME)
        return final

    partial = load_partial(partial_path)

    def save_partial(rec: dict) -> None:
        atomic_write_json(partial_path, rec)

    record = build_character_file(segments, client, partial or {}, model=model, on_window=save_partial)
    final = {"srt_hash": record["srt_hash"], "characters": record["characters"], "objects": record["objects"]}
    atomic_write_json(path, final)
    try:
        partial_path.unlink()
    except FileNotFoundError:
        pass
    log.info("Файл персонажей записан (%s): персонажей %d, объектов %d.",
             CHARACTERS_FILENAME, len(final["characters"]), len(final["objects"]))
    return final
