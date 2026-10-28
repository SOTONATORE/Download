"""Файл персонажей одного SRT: characters.json (H3, часть A).

Окно = WINDOW_SEGMENTS сегментов подряд, на каждое окно один запрос Gemini. Ответы сливаются в запись:
известный персонаж сохраняет внешность, к ней дописываются новые постоянные черты, противоречие
не записывается и попадает в лог. Запрос идёт через тот же транспорт, что и генерация промптов
(метод generate_json; в оркестраторе это ManagedTransport с лимитером и повторами). Ключ здесь не читается.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

try:  # пакетный и «плоский» импорт (как в остальных модулях core)
    from .srt_parser import Segment, srt_hash
    from .gemini_orchestrator import atomic_write_json
except ImportError:  # pragma: no cover
    from srt_parser import Segment, srt_hash  # type: ignore
    from gemini_orchestrator import atomic_write_json  # type: ignore

log = logging.getLogger("Generate.character_file")

WINDOW_SEGMENTS = 200
CHARACTERS_FILENAME = "characters.json"
MAX_NAME_LEN = 40
# Та же форма, что маркер [ИМЯ] в gemini_prompts: заглавные латинские и кириллические буквы, цифры, «_».
_NAME_RX = re.compile(r"[A-Z\u0410-\u042f\u0401][A-Z0-9_\u0410-\u042f\u0401]*")
_STR_LIST = {"type": "array", "items": {"type": "string"}}
_LIST_FIELDS = ("aliases", "keywords", "new_traits", "conflicts")

# Схема ответа одного окна (responseJsonSchema). Форма дублирует текст запроса ниже.
WINDOW_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "aliases": _STR_LIST,
                    "keywords": _STR_LIST,
                    "new_traits": _STR_LIST,
                    "conflicts": _STR_LIST,
                },
                "required": ["name", "aliases", "keywords", "new_traits", "conflicts"],
            },
        },
    },
    "required": ["characters"],
}

_SYSTEM = (
    "You list the people who appear in the subtitle text of one video window. "
    "Return only one JSON object, with no markdown, no code fences and no text before or after it."
)


def _is_valid_name(name: str) -> bool:
    return 0 < len(name) <= MAX_NAME_LEN and _NAME_RX.fullmatch(name) is not None


def _window_request(text: str, known: list) -> str:
    """Текст запроса одного окна: текст окна, уже известные имена с внешностью, правила, форма ответа."""
    names = [c["name"] for c in known]
    recorded = "\n".join(f"{c['name']}: {c.get('appearance', '')}" for c in known) or "(none yet)"
    return (
        "Window text (one subtitle segment per line):\n" + text + "\n\n"
        "Characters already in the record (name: recorded appearance):\n" + recorded + "\n"
        "Names already in the record: " + json.dumps(names, ensure_ascii=False) + "\n\n"
        "Rules:\n"
        "- name: one word in uppercase Latin or Cyrillic letters, no spaces, for example ANNA or АННА. "
        "For a person already in the record, use the existing name exactly.\n"
        "- aliases: other names this same person has in the text (may be empty).\n"
        "- keywords: words or short phrases that appear in the window text and identify this person.\n"
        "- new_traits: only permanent visible traits: age, build, hair, face, beard. Do not include "
        "clothing, anything that changes between scenes, or physical state such as wet or tired.\n"
        "- For a person already in the record, put in new_traits only permanent traits that the recorded "
        "appearance does not contain yet. Put in conflicts any permanent trait that contradicts the recorded "
        "appearance.\n"
        "- For a person not in the record, put all permanent traits in new_traits and leave conflicts empty.\n"
        "- If nobody appears in the text, return an empty list.\n\n"
        "Return only JSON in this exact shape:\n"
        '{"characters": [{"name": "ANNA", "aliases": ["ANNA"], "keywords": ["Anna"], '
        '"new_traits": ["woman in her thirties", "short dark hair"], "conflicts": []}]}'
    )


def parse_window_reply(raw: str, window_text: str) -> list:
    """Строгий разбор ответа одного окна. Любое нарушение формы: ValueError с коротким описанием.

    Возвращает список элементов {name, aliases, keywords, new_traits, conflicts}. Имя приводится к верхнему
    регистру и проверяется по форме маркера; неподходящее имя пропускается с записью в лог.
    Ключевые слова оставляются только те, что есть в тексте окна (без учёта регистра)."""
    if not isinstance(raw, str):
        raise ValueError("ответ не строка")
    data = json.loads(raw)  # не JSON -> json.JSONDecodeError (подкласс ValueError)
    if not isinstance(data, dict) or not isinstance(data.get("characters"), list):
        raise ValueError("ожидался объект с массивом characters")
    low = window_text.lower()
    out: list = []
    for item in data["characters"]:
        if not isinstance(item, dict):
            raise ValueError("элемент characters не объект")
        name = item.get("name")
        if not isinstance(name, str):
            raise ValueError("поле name не строка")
        for key in _LIST_FIELDS:
            value = item.get(key)
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise ValueError(f"поле {key} не список строк")
        nm = name.strip().upper()
        if not _is_valid_name(nm):
            log.warning("Имя персонажа не подходит по формату маркера, запись пропущена.")
            continue
        out.append({
            "name": nm,
            "aliases": [a.strip() for a in item["aliases"] if a.strip()],
            "keywords": [k.strip() for k in item["keywords"] if k.strip() and k.strip().lower() in low],
            "new_traits": [t.strip() for t in item["new_traits"] if t.strip()],
            "conflicts": [t.strip() for t in item["conflicts"] if t.strip()],
        })
    return out


# ---------------------------------------------------------------------------
# Слияние (чистая функция: вход не меняется)
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .!?")


def _sentence(text: str) -> str:
    t = " ".join(text.split())
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    return t if t[-1] in ".!?" else t + "."


def _add_trait(appearance: str, trait: str) -> str:
    """Дописывает черту в конец внешности, если её там ещё нет (по нормализованному тексту)."""
    sen = _sentence(trait)
    n = _norm(trait)
    if not sen or (n and n in _norm(appearance)):
        return appearance
    return (appearance + " " + sen).strip()


def merge_characters(characters: list, extracted: list) -> list:
    """Сливает разобранные элементы окна в список записей. Чистая функция, Gemini не вызывается.

    Правила:
    - известный персонаж (по имени или псевдониму) сохраняет внешность; новые постоянные черты
      дописываются в конец; черта, уже присутствующая в внешности, пропускается;
    - черта из conflicts не записывается и логируется: «conflict, not recorded: <имя>, <черта>»;
    - новый персонаж заводится новой записью со всеми своими чертами.
    Единственный побочный эффект — строки лога.
    """
    out = copy.deepcopy(characters)
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
            rec = {"name": name, "aliases": aliases,
                   "keywords": list(dict.fromkeys(item["keywords"])), "appearance": ""}
            for t in item["new_traits"]:
                rec["appearance"] = _add_trait(rec["appearance"], t)
            out.append(rec)
            pos = len(out) - 1
            for key in [name, *aliases]:
                index.setdefault(key.upper(), pos)
            for t in item["conflicts"]:
                log.warning("conflict, not recorded: %s, %s", name, t)
            continue

        rec = out[pos]
        for a in item["aliases"]:
            if a.upper() not in index:
                rec["aliases"].append(a)
                index[a.upper()] = pos
        for k in item["keywords"]:
            if k not in rec["keywords"]:
                rec["keywords"].append(k)
        for t in item["new_traits"]:
            rec["appearance"] = _add_trait(rec["appearance"], t)
        for t in item["conflicts"]:
            log.warning("conflict, not recorded: %s, %s", rec["name"], t)
    return out


# ---------------------------------------------------------------------------
# Окна, запросы, сборка записи
# ---------------------------------------------------------------------------

def _ask_window(client: Any, model: str, text: str, known: list, first: int, last: int) -> Optional[list]:
    """Один запрос на окно. Нарушение формы: ещё один запрос (всего два). Два провала: None (окно пропущено).

    Ошибки транспорта (GeminiRuntimeError) не перехватываются: квота и сбои сети ManagedTransport
    обрабатывает сам, и ошибка не должна превращаться в тихо пропущенное окно."""
    user = _window_request(text, known)
    for attempt in (1, 2):
        raw = client.generate_json(model, _SYSTEM, user, WINDOW_SCHEMA)
        try:
            return parse_window_reply(raw, text)
        except ValueError as exc:
            log.warning("Окно сегментов %s..%s: ответ не прошёл проверку (%s), попытка %d из 2.",
                        first, last, exc, attempt)
    log.error("Окно сегментов %s..%s пропущено: два ответа подряд не прошли проверку. "
              "Персонажи этого окна не записаны.", first, last)
    return None


def build_character_file(segments: list, client: Any, existing: dict, *, model: str = "") -> dict:
    """Строит запись о персонажах по окнам по WINDOW_SEGMENTS сегментов, по одному запросу на окно.

    existing: ранее сохранённая запись. Если её srt_hash не совпадает с текущим SRT, она отбрасывается
    (файл один на SRT) и запись строится заново. Результаты каждого окна сливаются до следующего окна,
    чтобы следующий запрос видел уже записанных персонажей.
    model: имя модели для транспорта. ManagedTransport его игнорирует (модель выбирает цепочка);
    для RealTransport указывать обязательно.
    Возвращает dict: {srt_hash, characters, skipped_windows}."""
    current = srt_hash(segments)
    if existing and existing.get("srt_hash") != current:
        log.warning("Файл персонажей относится к другому SRT: запись строится заново.")
        existing = {}
    characters = [copy.deepcopy(c) for c in (existing.get("characters") or [])] if existing else []
    skipped = list(existing.get("skipped_windows") or []) if existing else []

    for start in range(0, len(segments), WINDOW_SEGMENTS):
        window = segments[start:start + WINDOW_SEGMENTS]
        text = "\n".join(s.text for s in window if s.text)
        if not text:
            continue
        extracted = _ask_window(client, model, text, characters, window[0].num, window[-1].num)
        if extracted is None:
            skipped.append({"first": window[0].num, "last": window[-1].num})
            continue
        characters = merge_characters(characters, extracted)
    return {"srt_hash": current, "characters": characters, "skipped_windows": skipped}


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


def get_character_file(out_dir, segments: list, client: Any, *, model: str = "") -> dict:
    """Файл персонажей для SRT в папке out_dir (рядом с prompts.json).

    Если characters.json есть и его srt_hash совпадает с текущим, возвращается сохранённый файл,
    Gemini не вызывается. Иначе запись строится (build_character_file) и пишется атомарно."""
    path = Path(out_dir) / CHARACTERS_FILENAME
    current = srt_hash(segments)
    stored = load_character_file(path)
    if stored is not None and stored.get("srt_hash") == current:
        log.info("Файл персонажей для этого SRT уже есть (%s): Gemini не вызывается.", CHARACTERS_FILENAME)
        return stored
    record = build_character_file(segments, client, stored or {}, model=model)
    atomic_write_json(path, record)
    log.info("Файл персонажей записан (%s): персонажей %d, окон без результата %d.",
             CHARACTERS_FILENAME, len(record["characters"]), len(record["skipped_windows"]))
    return record
