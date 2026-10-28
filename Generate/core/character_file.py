"""Файл персонажей и объектов одного SRT: characters.json (H3, часть A).

Окно = WINDOW_SEGMENTS сегментов подряд, на каждое окно один запрос Gemini. Записи обновляются, а не пишутся заново:
в запрос окна передаётся предыдущая внешность каждой известной записи, модель возвращает полную внешность по id.
Существующая запись (id из списка окна) меняется, только если возвращённая внешность отличается от записанной.
Новая запись (id — пустая строка) получает CHAR_N или OBJ_N по порядку появления. Ключевые слова и псевдонимы
дописываются. Ответ, не прошедший проверку, повторяется один раз; второй провал останавливает окно.
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
_LIST_FIELDS = ("aliases", "keywords")
# Общая форма элемента персонажа и объекта. id: у известной записи — её id из окна, у новой — пустая строка.
_ITEM_PROPS = {"id": {"type": "string"}, "name": {"type": "string"}, "aliases": _STR_LIST,
               "keywords": _STR_LIST, "appearance": {"type": "string"}}
_ITEM_REQUIRED = ["id", "name", "aliases", "keywords", "appearance"]
# Владелец — обычная строка: пустая строка означает «нет владельца». null в схеме не используется (3a).
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
    правило владельца, форма ответа. Правила записаны здесь один раз; запрос окна их не повторяет."""
    lines = [
        "You list the people and the important objects that appear in the subtitle text of one video window. "
        "Each one is kept in a record that lasts across windows.",
        "Return a record only for a person or an object that is visible in the window text, identified by its id.",
        "- id: for a record listed in the known records, return its id exactly as listed. "
        "For a new person or object, return an empty string. Do not invent any other id.",
        "Rules for characters:",
        "- name: one word in uppercase Latin or Cyrillic letters, no spaces, for example ANNA or АННА. "
        "For a record already listed, use its name exactly.",
        "- aliases: other names this same person has in the text (may be empty).",
        "- keywords: words or short phrases that appear in the window text and identify this person.",
        "- If the name is ambiguous, choose the most likely historical person from the context of the text; "
        "do not add any identifier beyond the name.",
        "Rules for objects:",
        "- List an object only when it matters to the plot of this window, not because a word repeats often. "
        "A background mention with no name and no further detail is not listed. "
        "An object central to the scene is always listed.",
        "- name: one word, same form as character names. For a record already listed, use its name exactly.",
        "- aliases and keywords: as for characters.",
        "- owner: the name of the character who owns the object, exactly as in the characters list, "
        "or an empty string when no listed character owns it.",
        "Appearance rule:",
        "- appearance is one complete phrase in lowercase, with an article and a preposition where it fits, "
        "for example \"a woman in her thirties with short dark hair\" or "
        "\"a two-storey brick house with white shutters\". "
        "It holds all permanent traits: age range, build, hair, and facial features for a person, "
        "or the main fixed properties of an object: material, shape, color, size. "
        "The traits are linked inside the phrase with \"with\" and \"and\". No final period. At most 30 words. "
        "Do not write full sentences.",
        "- For a new person or object, appearance is always filled, even when the window text has no visual "
        "details. Base it on what you know about the person or object by name and on the context of the subtitles.",
        "- For a record already listed, return its appearance in full, exactly as given. Change it only when a new "
        "permanent trait appears in the window text or the previous phrase contains a clear error. "
        "When you change it, keep every permanent trait of the previous phrase.",
        "- Do not describe clothing or temporary state (wet, tired, and similar), for people or objects.",
        "- Do not add any uncertainty marker or extra field.",
        "- If nothing relevant appears in the text, return empty lists.",
        "Return only one JSON object, with no markdown, no code fences and no text before or after it.",
        "Return only JSON in this exact shape:",
        '{"characters": [{"id": "", "name": "ANNA", "aliases": ["ANNA"], "keywords": ["Anna"], '
        '"appearance": "a woman in her thirties with short dark hair"}], '
        '"objects": [{"id": "", "name": "HOUSE", "aliases": [], "keywords": ["house"], '
        '"appearance": "a two-storey brick house with white shutters", "owner": ""}]}',
    ]
    return "\n".join(lines)


def _is_valid_name(name: str) -> bool:
    return 0 < len(name) <= MAX_NAME_LEN and _NAME_RX.fullmatch(name) is not None


def _window_request(text: str, known: list, known_objects: list = ()) -> str:
    """Текст запроса одного окна: текст окна, известные записи с предыдущей внешностью по id, список имён.
    Правила здесь не пишутся: они в build_character_instructions(). Блок известных записей содержит только данные."""
    lines = [f"id: {c.get('id', '')} | name: {c['name']} | previous appearance: {c.get('appearance', '')}"
             for c in known]
    for o in known_objects:
        owner = o.get("owner") or "none"
        lines.append(f"id: {o.get('id', '')} | name: {o['name']} | previous appearance: "
                     f"{o.get('appearance', '')} | owner: {owner}")
    block = "\n".join(lines) or "(none yet)"
    names = [c["name"] for c in known] + [o["name"] for o in known_objects]
    return (
        "Window text (one subtitle segment per line):\n" + text + "\n\n"
        "Known records (each one is separate; do not mix them):\n" + block + "\n\n"
        "Return a record only if its id is in this list or it is a new person or object in the window text. "
        "Copy the previous appearance into your reply unless a permanent trait is new or the previous appearance "
        "has a clear error.\n\n"
        "Names already in the record: " + json.dumps(names, ensure_ascii=False)
    )


class _EmptyNewAppearance(ValueError):
    """Новая запись (id пустой строкой) без внешности в ответе. Ответ повторяется; если повтор снова такой,
    окно даёт EmptyAppearanceError с именем этой записи."""

    def __init__(self, record_name: str):
        super().__init__(f"новая запись «{record_name}» без внешности")
        self.record_name = record_name


def _parse_items(items: list, low: str, what: str, *, owner: bool) -> list:
    """Разбор списка персонажей или объектов: проверка формы каждого поля и пустой внешности (5a)."""
    out: list = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"элемент {what} не объект")
        rid = item.get("id")
        if not isinstance(rid, str):
            raise ValueError(f"поле id не строка ({what})")
        rid = rid.strip()
        name = item.get("name")
        if not isinstance(name, str):
            raise ValueError(f"поле name не строка ({what})")
        for key in _LIST_FIELDS:
            value = item.get(key)
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise ValueError(f"поле {key} не список строк ({what} «{name}»)")
        appearance = item.get("appearance")
        if not isinstance(appearance, str):
            raise ValueError(f"поле appearance не строка, а список или другой тип ({what} «{name}»)")
        owner_name = ""
        if owner:
            raw_owner = item.get("owner")
            if not isinstance(raw_owner, str):
                raise ValueError(f"поле owner не строка ({what} «{name}»)")
            owner_name = raw_owner.strip()
        nm = name.strip().upper()
        if not _is_valid_name(nm):
            log.warning("Имя (%s) не подходит по формату маркера, запись пропущена.", what)
            continue
        app = appearance.strip()
        if not app:
            if rid == "":
                raise _EmptyNewAppearance(nm)
            raise ValueError(f"запись {rid} «{nm}»: внешность пустая")
        entry = {
            "id": rid,
            "name": nm,
            "aliases": [a.strip() for a in item["aliases"] if a.strip()],
            "keywords": [k.strip() for k in item["keywords"] if k.strip() and k.strip().lower() in low],
            "appearance": app,
        }
        if owner:
            entry["owner"] = owner_name
        out.append(entry)
    return out


def parse_window_reply(raw: str, window_text: str) -> dict:
    """Строгий разбор ответа одного окна. Любое нарушение формы: ValueError с коротким описанием на русском.

    Возвращает {characters: [...], objects: [...]}; у каждого элемента есть id (пустая строка для новой записи),
    name, aliases, keywords, appearance, у объектов ещё owner (строка, пустая — нет владельца). Отсутствующий ключ
    objects читается как пустой список. Имя приводится к верхнему регистру и проверяется по форме маркера;
    неподходящее имя пропускается с записью в лог. Ключевые слова оставляются только те, что есть в тексте окна.
    Соответствие id известным записям проверяется в _check_reply."""
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

def _sentence(text: str) -> str:
    """Внешность одной фразой для записи: пробелы по краям и одна точка в конце убираются, первая буква заглавная.
    Ничего не склеивает и точку не добавляет: фраза из ответа модели записывается как есть."""
    t = " ".join(str(text or "").split())
    if t.endswith("."):
        t = t[:-1].rstrip()
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


def _same_appearance(old: Any, new: str) -> bool:
    """Записанная и возвращённая внешность совпадают: строки после обрезки пробелов по краям (6a, 6b).
    Если normalize_appearance меняет записанную строку (старая запись: точка в конце, «. » между чертами,
    заглавная буква), обе строки сравниваются после нормализации (6f)."""
    old_s = str(old or "").strip()
    new_s = str(new or "").strip()
    if old_s == new_s:
        return True
    if normalize_appearance(old_s) != old_s:
        return normalize_appearance(old_s) == normalize_appearance(new_s)
    return False


def _name_keys(name: str, aliases: list) -> set:
    """Имя и псевдонимы в верхнем регистре. По ним ищется совпадение новой записи с существующей (2d)."""
    return {name.upper(), *(a.upper() for a in aliases)}


def _dedupe_aliases(name: str, aliases: list) -> list:
    """Псевдонимы новой записи без повторов и без имени записи."""
    return list(dict.fromkeys(a for a in aliases if a.upper() != name.upper()))


def _merge_known(rec: dict, item: dict) -> None:
    """Существующая запись (id из окна): ключевые слова и псевдонимы дописываются. Внешность меняется, только если
    возвращённая отличается от записанной (6a); при изменении пишется строка лога «appearance updated».
    Имя, id и владелец записи не меняются (6d)."""
    for k in item["keywords"]:
        if k not in rec["keywords"]:
            rec["keywords"].append(k)
    known_names = _name_keys(rec["name"], rec["aliases"])
    for a in item["aliases"]:
        if a.upper() not in known_names:
            rec["aliases"].append(a)
            known_names.add(a.upper())
    old = rec.get("appearance") or ""
    new = item["appearance"].strip()
    if _same_appearance(old, new):
        return
    log.info("appearance updated: %s %s: %s -> %s", rec["id"], rec["name"], str(old).strip(), new)
    rec["appearance"] = new


def merge_characters(characters: list, extracted: list) -> list:
    """Сливает разобранные элементы окна в список записей. Чистая функция, Gemini не вызывается.

    Правила:
    - известная запись ищется по id (6d), её имя и id не меняются, внешность меняется по правилу 6a;
    - новая запись (id пустой строкой) получает очередной CHAR_N и внешность из ответа с заглавной первой
      буквой, без точки в конце (6c);
    - единственный побочный эффект — строки лога."""
    out = copy.deepcopy(characters)
    _assign_ids(out, "CHAR")
    for rec in out:
        rec.setdefault("aliases", [])
        rec.setdefault("keywords", [])
        rec.setdefault("appearance", "")
    by_id = {rec["id"]: rec for rec in out}
    for item in extracted:
        if item["id"]:
            rec = by_id.get(item["id"])
            if rec is not None:
                _merge_known(rec, item)
            continue
        rec = {"id": _new_id(out, "CHAR"), "name": item["name"],
               "aliases": _dedupe_aliases(item["name"], item["aliases"]),
               "keywords": list(dict.fromkeys(item["keywords"])),
               "appearance": _sentence(item["appearance"])}
        out.append(rec)
        by_id[rec["id"]] = rec
    return out


def _owner_index(characters: list) -> dict:
    """Имя, псевдоним или id персонажа в верхнем регистре -> его id. По id тоже находится владелец: запрос окна
    показывает владельца по id, и модель может вернуть его так же."""
    index: dict = {}
    for rec in characters:
        for key in [rec["name"], *rec.get("aliases", []), rec.get("id") or ""]:
            if key:
                index.setdefault(key.upper(), rec["id"])
    return index


def merge_objects(objects: list, extracted: list, characters: list) -> list:
    """Сливает разобранные объекты окна в список записей. Чистая функция, Gemini не вызывается.

    Известный объект ищется по id (6d); его владелец, имя и id не меняются, внешность меняется по правилу 6a.
    Новый объект (id пустой строкой) получает очередной OBJ_N. Владелец — id персонажа из characters, найденный
    по имени, псевдониму или id (6e). Имя, которого нет среди персонажей, даёт владельца None и лог
    «owner not found, set to null: <имя объекта>»; пустая строка даёт None без записи в лог."""
    out = copy.deepcopy(objects)
    _assign_ids(out, "OBJ")
    for rec in out:
        rec.setdefault("owner", None)
        rec.setdefault("aliases", [])
        rec.setdefault("keywords", [])
        rec.setdefault("appearance", "")
    owners = _owner_index(characters)
    by_id = {rec["id"]: rec for rec in out}
    for item in extracted:
        if item["id"]:
            rec = by_id.get(item["id"])
            if rec is not None:
                _merge_known(rec, item)
            continue
        owner = None
        if item.get("owner"):
            owner = owners.get(item["owner"].upper())
            if owner is None:
                log.warning("owner not found, set to null: %s", item["name"])
        rec = {"id": _new_id(out, "OBJ"), "owner": owner, "name": item["name"],
               "aliases": _dedupe_aliases(item["name"], item["aliases"]),
               "keywords": list(dict.fromkeys(item["keywords"])),
               "appearance": _sentence(item["appearance"])}
        out.append(rec)
        by_id[rec["id"]] = rec
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


def _keep_known_ids(items: list, known: list, what: str) -> list:
    """Оставляет новые записи (id пустой строкой) и записи с id из этого окна. Остальные отбрасываются с записью
    в лог: «id not in window, dropped: <id> <имя>» (2b)."""
    ids = {r.get("id") for r in known}
    out = []
    for item in items:
        if item["id"] == "" or item["id"] in ids:
            out.append(item)
        else:
            log.warning("id not in window, dropped: %s %s", item["id"], item["name"])
    return out


def _check_reply(parsed: dict, characters: list, objects: list) -> dict:
    """Проверка разобранного ответа по известным записям. Неизвестный id отбрасывается (_keep_known_ids).
    Новая запись персонажа (id пустой), чьё имя или псевдоним совпадает с известным персонажем, — ошибка ответа
    (2d): такая запись уже есть и должна вернуться с её id. Два новых персонажа с одним именем в одном ответе
    — тоже ошибка (2b). Проверка объектов по владельцу здесь не делается: владелец известен только после
    слияния персонажей, см. _check_new_objects."""
    chars = _keep_known_ids(parsed["characters"], characters, "персонаж")
    objs = _keep_known_ids(parsed["objects"], objects, "объект")
    seen_new: list = []
    for item in chars:
        if item["id"] != "":
            continue
        keys = _name_keys(item["name"], item["aliases"])
        for rec in characters:
            if keys & _name_keys(rec["name"], rec.get("aliases", [])):
                raise ValueError(f"новая запись «{item['name']}» совпадает с записью {rec['id']}: "
                                 "её нужно вернуть с этим id")
        if any(keys & prev for prev in seen_new):
            raise ValueError(f"новая запись «{item['name']}» повторяется в этом ответе")
        seen_new.append(keys)
    return {"characters": chars, "objects": objs}


def _check_new_objects(items: list, objects: list, new_chars: list) -> None:
    """Проверка новых объектов ответа после слияния персонажей (2a). Владелец находится по new_chars, то есть
    id, который merge_characters только что назначил персонажам окна; неизвестное имя владельца даёт None,
    как в merge_objects. Дубль (2b): новый объект с тем же именем или псевдонимом, что у записи с тем же
    владельцем по id, или два новых объекта с тем же именем и тем же владельцем внутри этого ответа.
    Ошибка — ValueError: окно повторяется (2c)."""
    owners = _owner_index(new_chars)
    seen_new: list = []
    for item in items:
        if item["id"] != "":
            continue
        owner_id = owners.get(item["owner"].upper()) if item["owner"] else None
        keys = _name_keys(item["name"], item["aliases"])
        for rec in objects:
            if rec.get("owner") != owner_id:
                continue
            if keys & _name_keys(rec["name"], rec.get("aliases", [])):
                raise ValueError(f"новый объект «{item['name']}» совпадает с объектом {rec['id']} того же "
                                 "владельца: его нужно вернуть с этим id")
        for prev_owner, prev_keys, prev_name in seen_new:
            if prev_owner == owner_id and keys & prev_keys:
                raise ValueError(f"новый объект «{item['name']}» повторяет новый объект «{prev_name}» того же "
                                 "владельца в этом ответе")
        seen_new.append((owner_id, keys, item["name"]))


def _ask_window(client: Any, model: str, text: str, characters: list, objects: list,
                first: int, last: int) -> tuple:
    """Один запрос на окно: проверка ответа и слияние. Возвращает (characters, objects).

    Ответ проверяется parse_window_reply, _check_reply, слияние персонажей и _check_new_objects (владелец
    объекта — id после слияния). Нарушение — ещё один запрос (всего два). Если второй
    ответ тоже не прошёл, то при пустой внешности новой записи выходит EmptyAppearanceError, иначе
    CharacterWindowError. Слияние выполняется только для ответа, прошедшего проверку. Ошибки транспорта
    (GeminiRuntimeError) не перехватываются: квоту и сбои сети обрабатывает ManagedTransport."""
    system = build_character_instructions()
    user = _window_request(text, characters, objects)
    last_error: Optional[ValueError] = None
    for attempt in (1, 2):
        raw = client.generate_json(model, system, user, WINDOW_SCHEMA)
        try:
            extracted = _check_reply(parse_window_reply(raw, text), characters, objects)
            new_chars = merge_characters(characters, extracted["characters"])
            _check_new_objects(extracted["objects"], objects, new_chars)
        except ValueError as exc:
            last_error = exc
            log.warning("Окно сегментов %s..%s: ответ не прошёл проверку (%s), попытка %d из 2.",
                        first, last, exc, attempt)
            continue
        new_objs = merge_objects(objects, extracted["objects"], new_chars)
        return new_chars, new_objs
    if isinstance(last_error, _EmptyNewAppearance):
        log.error("Окно сегментов %s..%s: запись «%s» без внешности после повтора, прогон остановлен.",
                  first, last, last_error.record_name)
        raise EmptyAppearanceError(last_error.record_name, first, last)
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
