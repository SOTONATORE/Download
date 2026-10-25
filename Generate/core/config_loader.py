"""Загрузка и проверка Generate/config.default.yaml (SPEC.md, раздел 3).

Слои настроек (сверху вниз приоритет растёт):
  1. Generate/config.default.yaml: значения по умолчанию, лежат в репозитории;
  2. входы workflow / аргументы CLI: передаются параметром ``overrides``;
  3. секреты: только из окружения. Этот модуль окружение НЕ читает, и секреты
     в конфигурации запрещены (ключи с api_key/token/secret/password отвергаются).

Схема задаётся датаклассом ``Config``. Тип и границы каждого поля записаны в metadata.
Ошибки (неизвестный или отсутствующий ключ, неверный тип, выход за границы,
нарушение связей между полями, повторяющийся ключ, битый YAML) дают
``GeminiInputError`` с кодом выхода 2. В сообщении перечисляются все найденные
проблемы сразу. Значения из файла в текст ошибок не попадают (SPEC 0.1):
только имена ключей и правила.

Зависимость: PyYAML (safe_load). Сеть и окружение не используются.
"""
from __future__ import annotations

import difflib
import math
import re
import typing
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, get_args, get_origin, get_type_hints

import yaml

try:  # пакетный и «плоский» импорт (как в остальных модулях core)
    from .gemini_prompts import GeminiInputError
    from .model_constants import DEFAULT_MODEL_PROFILE, speed_threshold_mbps
except ImportError:  # pragma: no cover
    from gemini_prompts import GeminiInputError  # type: ignore
    from model_constants import DEFAULT_MODEL_PROFILE, speed_threshold_mbps  # type: ignore

__all__ = [
    "Config", "DEFAULT_CONFIG_PATH", "MODEL_PROFILES_DIR", "ACTIONS_JOB_LIMIT_MIN", "DEFAULT_MODEL_PROFILE",
    "parse_config", "load_config", "load_profile_number", "load_profile_frame_rule", "profile_exists",
]

GENERATE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = GENERATE_DIR / "config.default.yaml"
MODEL_PROFILES_DIR = GENERATE_DIR / "model_profiles"
# Лимит GitHub Actions на одно задание, минут (generate.yml: timeout-minutes: 360).
ACTIONS_JOB_LIMIT_MIN = 360

# Подстроки имён ключей: если неизвестный ключ похож на секрет, сообщаем об этом отдельно.
_SECRET_NAME_HINTS = ("api_key", "apikey", "token", "secret", "password")
_PROFILE_NAME_PATTERN = r"[A-Za-z0-9_\-]+"
# Режимы аренды карт Vast (rental_type): прерываемая (bid) и обычная (on-demand).
RENTAL_TYPES = ("interruptible", "on-demand")

_TYPE_WORDS = {bool: "логическое значение (true или false)", int: "целое число",
               float: "число", str: "строка"}


def _rule(**meta: Any):
    """Поле схемы без значения по умолчанию: ключ обязателен, правила лежат в metadata."""
    return field(metadata=meta)


@dataclass(frozen=True)
class Config:
    """Схема и значения конфигурации. Порядок полей совпадает с config.default.yaml."""

    # --- имена файлов и текст ---
    filename_template: str = _rule(nonempty=True, contains="{num}")
    clip_name_width: int = _rule(ge=0)

    # --- Gemini ---
    gemini_models: list[str] = _rule(min_items=1)
    gemini_batch_size: int = _rule(ge=1)
    gemini_rpm_limit: int = _rule(ge=1)
    gemini_tpm_limit: int = _rule(ge=1)
    gemini_rpd_limit: int = _rule(ge=1)

    # --- профиль модели и тайминги ---
    model_profile: str = _rule(pattern=_PROFILE_NAME_PATTERN,
                               hint="допустимы латинские буквы, цифры, «_» и «-»")
    # пресет профиля: только формат; значение читается, но пока не применяется
    preset: str = _rule(pattern=_PROFILE_NAME_PATTERN,
                        hint="допустимы латинские буквы, цифры, «_» и «-»")
    fps: int = _rule(ge=1)
    clip_min_sec: float = _rule(gt=0)
    clip_max_sec: float = _rule(gt=0)
    gpu_max_sec: Optional[float] = _rule(gt=0, optional=True)
    tail_pad_sec: float = _rule(ge=0)
    target_resolution: str = _rule(pattern=r"\d+x\d+", hint="формат ШИРИНАxВЫСОТА, например 1920x1080")

    # --- аренда карт Vast.ai ---
    gpu_name: str = _rule()
    rental_type: str = _rule(choices=RENTAL_TYPES)
    # Единственный ключ с умолчанием: отсутствие в файле читается как True, то есть 1 (SPEC 3).
    # В файле и во входах значение только 0 или 1 (целые); true и false не принимаются.
    # kw_only снимает правило «поля с умолчанием идут после полей без умолчания»; порядок как в файле.
    verified: bool = field(default=True, kw_only=True, metadata={"zero_one": True})
    min_price_per_hour: float = _rule(ge=0)
    max_price_per_hour: float = _rule(gt=0)
    min_reliability: float = _rule(ge=0, le=1)
    min_inet_mbps: float = _rule(ge=0)
    max_inet_price_per_tb: float = _rule(gt=0)
    max_cards: int = _rule(ge=1)
    budget_limit_usd: float = _rule(gt=0)
    min_balance_usd: float = _rule(ge=0)
    silent_host_timeout_min: int = _rule(ge=1)
    docker_image: str = _rule(pattern=r"\S+", hint="непустая строка без пробелов")
    disk_gb: int = _rule(ge=1)
    worker_idle_timeout_min: int = _rule(ge=1)
    # логи карты через Vast при снятии без логов воркера (SPEC 8.4): попытки и пауза между ними, с
    vast_logs_attempts: int = _rule(ge=1)
    vast_logs_pause_sec: float = _rule(ge=0)

    # --- выдача и запуск ---
    release_asset_limit: int = _rule(ge=1)
    limit_clips: int = _rule(ge=0)
    job_deadline_min: int = _rule(ge=1)

    # --- сбор цен, загрузка весов, мониторинг (v0.6) ---
    scan_sec: int = _rule(ge=1)
    scan_step_sec: int = _rule(ge=1)
    stable_max_spread: float = _rule(ge=0)
    max_rent_tries: int = _rule(ge=1)
    bid_margin: float = _rule(ge=0)
    speed_check_at_pct: int = _rule(ge=1, le=100)
    progress_step_pct: int = _rule(ge=1, le=50)
    stall_timeout_min: int = _rule(ge=1)
    stage_max_min: int = _rule(ge=1)
    worker_ready_timeout_min: int = _rule(ge=1)
    task_accept_timeout_min: int = _rule(ge=1)
    heartbeat_interval_sec: int = _rule(ge=1)
    monitor_interval_sec: int = _rule(ge=1)

    @property
    def interruptible(self) -> bool:
        """Производное от rental_type: единственный источник режима аренды (ключа interruptible в файле нет)."""
        return self.rental_type == "interruptible"


class _Bad(Exception):
    """Внутренняя ошибка одного поля; текст уходит в общий список проблем."""


def _check_value(name: str, value: Any, hint: Any, meta: Mapping[str, Any]) -> Any:
    optional = bool(meta.get("optional", False))
    base = hint
    if get_origin(hint) is typing.Union:
        base = next(a for a in get_args(hint) if a is not type(None))

    if value is None:
        if optional:
            return None
        raise _Bad(f"«{name}»: значение не может быть пустым (null)")

    if meta.get("zero_one"):
        # только целые 0 и 1: bool (True и False) и строки «0», «1», «true» не принимаются
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            raise _Bad(f"«{name}»: нужно 0 или 1")
        return bool(value)

    if base is bool:
        if not isinstance(value, bool):
            raise _Bad(f"«{name}»: нужно {_TYPE_WORDS[bool]}")
        return value

    if get_origin(base) is list:
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise _Bad(f"«{name}»: нужен список строк")
        if len(value) < meta.get("min_items", 0):
            raise _Bad(f"«{name}»: список не может быть пустым")
        if any(not x.strip() for x in value):
            raise _Bad(f"«{name}»: пустые элементы в списке недопустимы")
        return list(value)

    if base is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _Bad(f"«{name}»: нужно {_TYPE_WORDS[int]}")
        return _check_bounds(name, value, meta)

    if base is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _Bad(f"«{name}»: нужно {_TYPE_WORDS[float]}")
        return _check_bounds(name, float(value), meta)

    if base is str:
        if not isinstance(value, str):
            raise _Bad(f"«{name}»: нужна {_TYPE_WORDS[str]}")
        if meta.get("nonempty") and not value.strip():
            raise _Bad(f"«{name}»: строка не может быть пустой")
        if "pattern" in meta and not re.fullmatch(meta["pattern"], value):
            raise _Bad(f"«{name}»: не соответствует формату ({meta.get('hint', meta['pattern'])})")
        if "contains" in meta and meta["contains"] not in value:
            raise _Bad(f"«{name}»: должно содержать {meta['contains']}")
        if "choices" in meta and value not in meta["choices"]:
            # значение из файла в текст не выводим (SPEC 0.1): только список допустимых
            raise _Bad(f"«{name}»: допустимы значения: {', '.join(meta['choices'])}")
        return value

    raise TypeError(f"неподдерживаемый тип схемы для {name}: {hint!r}")  # ошибка разработчика


def _check_bounds(name: str, number: float, meta: Mapping[str, Any]) -> float:
    # Значение из файла в текст ошибки не выводим (SPEC 0.1, задание этапа 1): только имя ключа и правило.
    if "gt" in meta and not number > meta["gt"]:
        raise _Bad(f"«{name}»: должно быть больше {meta['gt']}")
    if "ge" in meta and not number >= meta["ge"]:
        raise _Bad(f"«{name}»: должно быть не меньше {meta['ge']}")
    if "le" in meta and not number <= meta["le"]:
        raise _Bad(f"«{name}»: должно быть не больше {meta['le']}")
    return number


def _cross_checks(v: Mapping[str, Any]) -> list[str]:
    problems = []
    if v["clip_max_sec"] < v["clip_min_sec"]:
        problems.append("«clip_max_sec» не может быть меньше «clip_min_sec»")
    if v["max_price_per_hour"] <= v["min_price_per_hour"]:
        problems.append("«max_price_per_hour» должен быть строго больше «min_price_per_hour»")
    if v["job_deadline_min"] >= ACTIONS_JOB_LIMIT_MIN:
        problems.append(f"«job_deadline_min» должен быть меньше лимита задания "
                        f"GitHub ({ACTIONS_JOB_LIMIT_MIN} мин)")
    # на запуск должно хватать всего бюджета: кредит (min_balance_usd) не меньше потолка расходов
    if v["min_balance_usd"] < v["budget_limit_usd"]:
        problems.append("«min_balance_usd» не может быть меньше «budget_limit_usd»")
    # порог скорости весов = половина входа min_inet_mbps, округлённая вниз (не из конфига): не меньше 1 Мбит/с.
    # Верхняя граница к min_inet_mbps больше не нужна: порог всегда не больше входа.
    if not math.isfinite(v["min_inet_mbps"]) or speed_threshold_mbps(v["min_inet_mbps"]) < 1:
        problems.append("«min_inet_mbps»: порог скорости весов (половина, округлённая вниз) должен быть "
                        "конечным и не меньше 1 Мбит/с")
    # общий срок этапа не короче срока зависания
    if v["stage_max_min"] < v["stall_timeout_min"]:
        problems.append("«stage_max_min» не может быть меньше «stall_timeout_min»")
    # за окно сбора цен должно набираться хотя бы 5 проб на предложение (минимум фильтра стабильности)
    if v["scan_sec"] < 5 * v["scan_step_sec"]:
        problems.append(f"«scan_sec» ({v['scan_sec']}) не может быть меньше 5 × «scan_step_sec» "
                        f"({v['scan_step_sec']}): за окно не наберётся 5 проб на предложение")
    return problems


def _format(source: str, problems: list[str]) -> str:
    body = "\n".join(f"  - {p}" for p in problems)
    return f"Ошибка конфигурации ({source}): проблем: {len(problems)}.\n{body}"


def parse_config(raw: Any, *, source: str = DEFAULT_CONFIG_PATH.name) -> Config:
    """Проверяет уже разобранный объект (dict из YAML) и возвращает Config.

    Бросает GeminiInputError со всеми найденными проблемами сразу.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise GeminiInputError(
            _format(source, ["верхний уровень должен быть набором пар «ключ: значение»"]))

    known = {f.name: f for f in fields(Config)}
    hints = get_type_hints(Config)
    problems: list[str] = []

    for key in raw:
        if not isinstance(key, str):
            problems.append(f"ключ {key!r} должен быть строкой")
        elif key not in known:
            if any(hint in key.lower() for hint in _SECRET_NAME_HINTS):
                problems.append(f"ключ «{key}»: секреты не хранятся в конфигурации, "
                                f"используйте переменную окружения (SPEC 0.1)")
            else:
                close = difflib.get_close_matches(key, list(known), n=1)
                tip = f" (возможно, «{close[0]}»)" if close else ""
                problems.append(f"неизвестный ключ «{key}»{tip}")

    values: dict[str, Any] = {}
    for name, f in known.items():
        if name not in raw:
            if f.default is not MISSING:  # ключ с умолчанием: отсутствие равно значению по умолчанию
                values[name] = f.default
                continue
            problems.append(f"отсутствует обязательный ключ «{name}»")
            continue
        try:
            values[name] = _check_value(name, raw[name], hints[name], f.metadata)
        except _Bad as e:
            problems.append(str(e))

    if not problems and len(values) == len(known):
        problems.extend(_cross_checks(values))

    if problems:
        raise GeminiInputError(_format(source, problems))
    return Config(**values)


class _DuplicateKeyError(Exception):
    def __init__(self, key: Any, line: int) -> None:
        super().__init__(key, line)
        self.key = key
        self.line = line


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader, который отвергает повторяющиеся ключи: иначе второй молча перетрёт первый."""


def _mapping_without_duplicates(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False):
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise _DuplicateKeyError(key, key_node.start_mark.line + 1)
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                              _mapping_without_duplicates)


def load_config(path: Optional[Path | str] = None,
                overrides: Optional[Mapping[str, Any]] = None) -> Config:
    """Читает YAML, накладывает входы (значения None пропускаются) и проверяет схему.

    path: файл конфигурации; по умолчанию Generate/config.default.yaml.
    overrides: значения входов workflow/CLI; проходят ту же проверку, что и файл.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    source = cfg_path.name
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise GeminiInputError(f"Файл конфигурации не найден: {source}") from None
    except (OSError, UnicodeDecodeError) as e:
        raise GeminiInputError(
            f"Файл конфигурации не удалось прочитать ({source}): {type(e).__name__}.") from None

    try:
        raw = yaml.load(text, Loader=_StrictLoader)
    except _DuplicateKeyError as e:
        raise GeminiInputError(
            _format(source, [f"повторяющийся ключ «{e.key}» (строка {e.line})"])) from None
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f", строка {mark.line + 1}" if mark is not None else ""
        raise GeminiInputError(f"Файл конфигурации не разобран как YAML ({source}{where}).") from None

    label = source
    if overrides and isinstance(raw, dict):
        raw = {**raw, **{k: v for k, v in overrides.items() if v is not None}}
        label = f"{source} с входами workflow/CLI"
    return parse_config(raw, source=label)


def profile_dir(profile: str) -> Path:
    """Папка профиля модели: Generate/model_profiles/models/<profile>/ (имя проверяется)."""
    if not re.fullmatch(_PROFILE_NAME_PATTERN, profile or ""):
        raise GeminiInputError("Недопустимое имя профиля модели.")
    return MODEL_PROFILES_DIR / "models" / profile


def profile_exists(profile: str) -> bool:
    """Есть ли файл profile.yaml у профиля модели."""
    return (profile_dir(profile) / "profile.yaml").is_file()


def _read_profile(profile: str) -> dict:
    path = profile_dir(profile) / "profile.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise GeminiInputError(f"Профиль модели не найден: {profile}/profile.yaml") from None
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        raise GeminiInputError(f"Профиль модели не удалось прочитать: {profile}/profile.yaml") from None
    return data if isinstance(data, dict) else {}


def load_profile_number(profile: str, key: str) -> float:
    """Положительное число из профиля модели: Generate/model_profiles/models/<profile>/profile.yaml."""
    value = _read_profile(profile).get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise GeminiInputError(
            f"В профиле {profile}/profile.yaml не задан положительный параметр «{key}».")
    return float(value)


def load_profile_frame_rule(profile: str) -> dict:
    """Правило кадров из профиля модели (ключ frame_rule), например {«kind»: «8k+1»}."""
    rule = _read_profile(profile).get("frame_rule")
    if not isinstance(rule, dict) or not isinstance(rule.get("kind"), str):
        raise GeminiInputError(
            f"В профиле {profile}/profile.yaml не задано правило кадров («frame_rule»).")
    return dict(rule)
