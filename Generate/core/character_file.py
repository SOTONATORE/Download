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
    "For every new person listed, new_traits must contain a description of permanent appearance, even when the window text has no visual details. The description is based on what you know about the historical person by name and on the context of the subtitles. "
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
        "- For a new person, new_traits always contains at least age range, build, hair, and facial features, "
        "based on the historical person.\n"
        "- If the name is ambiguous, choose the most likely historical person from the context of the text; "
        "do not leave new_traits empty and do not add any identifier beyond the name.\n"
        "- Do not describe clothing or temporary state (wet, tired, and similar).\n"
        "- Do not add any uncertainty marker or extra field.\n"
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


def _ask_window(client: Any, model: str, text: str, known: list, first: int, last: int) -> list:
    """Один запрос на окно. Нарушение формы: ещё один запрос (всего два). Два провала: CharacterWindowError.

    Ошибки транспорта (GeminiRuntimeError) не перехватываются: квота и сбои сети ManagedTransport
    обрабатывает сам и не превращает их в пропущенное окно."""
    user = _window_request(text, known)
    for attempt in (1, 2):
        raw = client.generate_json(model, _SYSTEM, user, WINDOW_SCHEMA)
        try:
            return parse_window_reply(raw, text)
        except ValueError as exc:
            log.warning("Окно сегментов %s..%s: ответ не прошёл проверку (%s), попытка %d из 2.",
                        first, last, exc, attempt)
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
    """Строит запись о персонажах по окнам, по одному запросу на окно с текстом.

    existing: частичная запись прошлого запуска ({srt_hash, characters, finished}). Если её srt_hash не
    совпадает с текущим SRT, она отбрасывается. Окна из finished пропускаются без запроса.
    Результат каждого окна сливается до следующего окна. После каждого успешного окна вызывается
    on_window с текущей записью. Ошибка окна (CharacterWindowError) выходит наружу без возврата.
    Возвращает dict: {srt_hash, characters, finished}."""
    current = srt_hash(segments)
    if existing and existing.get("srt_hash") != current:
        log.warning("Частичный файл персонажей относится к другому SRT: запись строится заново.")
        existing = {}
    characters = [copy.deepcopy(c) for c in (existing.get("characters") or [])] if existing else []
    finished = _load_finished(existing.get("finished")) if existing else []
    done = {(f["first"], f["last"]) for f in finished}

    for first, last, text in _window_ranges(segments):
        if (first, last) in done:
            continue
        if text:
            extracted = _ask_window(client, model, text, characters, first, last)
            characters = merge_characters(characters, extracted)
        finished.append({"first": first, "last": last})
        done.add((first, last))
        if on_window is not None:
            on_window({"srt_hash": current, "characters": characters, "finished": finished})
    return {"srt_hash": current, "characters": characters, "finished": finished}


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
        log.info("Файл персонажей для этого SRT уже есть (%s): Gemini не вызывается.", CHARACTERS_FILENAME)
        return stored

    partial = load_partial(partial_path)

    def save_partial(rec: dict) -> None:
        atomic_write_json(partial_path, rec)

    record = build_character_file(segments, client, partial or {}, model=model, on_window=save_partial)
    final = {"srt_hash": record["srt_hash"], "characters": record["characters"]}
    atomic_write_json(path, final)
    try:
        partial_path.unlink()
    except FileNotFoundError:
        pass
    log.info("Файл персонажей записан (%s): персонажей %d.", CHARACTERS_FILENAME, len(final["characters"]))
    return final
