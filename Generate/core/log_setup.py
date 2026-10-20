"""Обработчики журнала шага run (SPEC 10, этап 7а).

Два обработчика на корневом логгере:
- консоль (stderr, окно GitHub Actions): уровень INFO всегда; сырой журнал обмена с Vast
  (строки «Запрос Vast #» и «Ответ Vast #») в консоль не попадает;
- файл <каталог артефакта>/run/debug.log: уровень из GEN_LOG_LEVEL, пишет все записи,
  включая DEBUG и сырой журнал Vast (--debug-logs). Создаётся только при заданном каталоге
  артефакта; без каталога (локальный запуск) файла нет.

Каждая строка маскируется до записи в любой из обработчиков (SPEC 0.1, п. 2, 11).
Маскирование идёт по именам (решение владельца 18.10.2026): модуль не читает окружение; значения секретов
ему передаёт оркестратор через set_known_secrets (этап 9а).
Формат строки совпадает с прежним (его разбирает фильтр в generate.yml).
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

try:  # пакетный и «плоский» импорт, как в cli.py
    from Generate.core.vast_client import redact_text
except ImportError:  # pragma: no cover
    from vast_client import redact_text  # type: ignore

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
CONSOLE_LEVEL = logging.INFO
# файл отладочного журнала внутри каталога артефакта (SPEC 10.4)
DEBUG_FILE_PARTS = ("run", "debug.log")
# строки сырого обмена с Vast: пишутся логгером клиента на уровне INFO (vast_client.py)
JOURNAL_LOGGER = "Generate.vast_client"
JOURNAL_PREFIXES = ("Запрос Vast #", "Ответ Vast #")
# Режим маскирования (SPEC 0.1, п. 14; решение владельца 18.10.2026).
# Режим A — текущий: маскируется пара ИМЯ=значение и "ИМЯ": "значение", если в имени есть часть,
# похожая на секрет (KEY, TOKEN, SECRET, PASSWORD, AUTH, CREDENTIAL, SESSION; без учёта регистра).
# Режим B — цель: маскируется любая такая пара. Переходить на B, когда белый список ALLOWED_NAMES полный.
# Переключение — одна константа здесь; тесты проверяют оба режима через mask_text_with_mode.
MASK_MODE = "A"
_MASK_MODES = ("A", "B")
if MASK_MODE not in _MASK_MODES:
    raise ValueError("Режим маскирования MASK_MODE должен быть A или B.")

# Белый список имён (сравнение без учёта регистра по полному имени). Пары с этими именами не маскируются
# ни в одном режиме; шаблоны по форме значения действуют всегда. Список короткий и явный: добавлять
# только имя, которое нужно сохранить в строке журнала и которое не является секретом.
# Список имён, которые владелец подтвердил как не секретные в логах запусков 18.10.2026;
# имя добавляется только явным решением владельца.
ALLOWED_NAMES = frozenset({
    "status", "ip", "size", "port", "msg", "is_bid", "dph", "run_id", "target", "gpu", "limit", "rc",
    "id", "name", "component", "identity", "address", "hash", "event", "connIndex", "threshold",
    "device", "protocol", "location",
    "max_tokens", "num_tokens", "token_count", "tokenizer",
})
# Сравнение без учёта регистра: имя connIndex в списке написано с заглавной буквой, сверяем в нижнем регистре.
_ALLOWED_LOWER = frozenset(n.lower() for n in ALLOWED_NAMES)

# имена, которые маскируются всегда, в любом режиме; список меняется только явным решением владельца
BLOCKED_NAMES = frozenset({"crisp_hmac", "hmac"})
_BLOCKED_LOWER = frozenset(n.lower() for n in BLOCKED_NAMES)
_OVERLAP = sorted(_BLOCKED_LOWER & _ALLOWED_LOWER)
if _OVERLAP:
    raise ValueError("Имена из чёрного списка не могут быть в белом списке: " + ", ".join(_OVERLAP))

# Имя с секретной частью (режим A) и любое имя (режим B). Значение: в кавычках до закрывающей кавычки,
# иначе до первого пробела, кавычки, запятой или конца строки.
_SECRET_WORDS = r"(?:KEY|TOKEN|SECRET|PASSWORD|AUTH|CREDENTIAL|SESSION)"
_NAME_A = r"[\w.\-]*" + _SECRET_WORDS + r"[\w.\-]*"
_NAME_B = r"[\w.\-]+"
_VALUE = r"\"[^\"\r\n]*\"?|'[^'\r\n]*'?|[^\s\"',]*"
_ASSIGN_A = re.compile(r"(?i)(?<![\w.\-])(?P<name>" + _NAME_A + r")=(?P<value>" + _VALUE + r")")
_ASSIGN_B = re.compile(r"(?i)(?<![\w.\-])(?P<name>" + _NAME_B + r")=(?P<value>" + _VALUE + r")")
_JSON_A = re.compile(r"(?i)\"(?P<name>" + _NAME_A + r")\"(?P<sep>\s*:\s*)\"[^\"\r\n]*\"?")
_JSON_B = re.compile(r"(?i)\"(?P<name>" + _NAME_B + r")\"(?P<sep>\s*:\s*)\"[^\"\r\n]*\"?")
_PAIR_PATTERNS = {"A": (_JSON_A, _ASSIGN_A), "B": (_JSON_B, _ASSIGN_B)}
# Чёрный список: полное имя (граница с обеих сторон), маскируется в любом режиме и раньше белого списка.
_BLOCKED_ALT = "|".join(re.escape(n) for n in sorted(BLOCKED_NAMES, key=len, reverse=True))
_BLOCK_ASSIGN = re.compile(r"(?i)(?<![\w.\-])(?P<name>" + _BLOCKED_ALT + r")(?![\w.\-])=(?P<value>" + _VALUE + r")")
_BLOCK_JSON = re.compile(r"(?i)\"(?P<name>" + _BLOCKED_ALT + r")\"(?P<sep>\s*:\s*)\"[^\"\r\n]*\"?")

# Пары с именами из белого списка для прохода redact_text (vast_client): такие пары остаются как есть.
_ALLOWED_PAIR_RE = re.compile(
    r"(?i)(?<![\w.\-])[\"']?(?:"
    + "|".join(re.escape(n) for n in sorted(ALLOWED_NAMES, key=len, reverse=True))
    + r")[\"']?\s*[:=]\s*(?:\"[^\"\r\n]*\"?|'[^'\r\n]*'?|[^\s\"',;&}\]]*)"
)

# Маскирование текста, пришедшего с карты (SPEC 0.1): карта может напечатать что угодно.
# Список шаблонов по форме значения живёт только здесь; оркестратор берёт его через mask_remote_text.
# Тот же набор, что в worker/server.py (воркер собирается отдельно, модуль не импортируется).
_SECRET_PATTERNS = (
    (re.compile(r"(?i)(authorization\s*:)[^\r\n]*"), True),
    (re.compile(r"(?i)(x-worker-token\s*[:=]\s*)\S+"), True),
    (re.compile(r"(?i)(bearer\s+)\S+"), True),
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), False),
    (re.compile(r"AIza[0-9A-Za-z_-]{20,}"), False),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), False),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), False),
)
_MASK = "***"

# Шаблоны по форме значения (любой режим, любое имя). Из _SECRET_PATTERNS берутся все;
# добавлены только github_pat_ и ghp_ с порогом 10.
_EXTRA_SHAPES = (
    (re.compile(r"github_pat_[A-Za-z0-9_]{10,}"), False),
    (re.compile(r"ghp_[A-Za-z0-9]{10,}"), False),
    # адрес почты (вида *@*) маскируется во всех логах как страховка: приватность, 20.10.2026
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), False),
)
_VALUE_SHAPES = tuple(_SECRET_PATTERNS) + _EXTRA_SHAPES


def _is_allowed(name: str) -> bool:
    return name.lower() in _ALLOWED_LOWER


def _is_blocked(name: str) -> bool:
    return name.lower() in _BLOCKED_LOWER


def _json_repl(m: "re.Match[str]") -> str:
    # чёрный список проверяется раньше белого: заблокированное имя не пропускается никогда
    if not _is_blocked(m.group("name")) and _is_allowed(m.group("name")):
        return m.group(0)
    return f"\"{m.group('name')}\"{m.group('sep')}\"{_MASK}\""


def _assign_repl(m: "re.Match[str]") -> str:
    # чёрный список проверяется раньше белого
    if not _is_blocked(m.group("name")) and _is_allowed(m.group("name")):
        return m.group(0)
    value = m.group("value")
    tail = value[0] + _MASK + value[0] if value[:1] in ('"', "'") else _MASK
    return m.group("name") + "=" + tail


def _redact_free_text(text: str) -> str:
    """Свободный текст через redact_text (vast_client): пары с секретными именами и «Bearer».
    Пары из белого списка вырезаются до вызова и возвращаются без изменений."""
    parts: list = []
    pos = 0
    for m in _ALLOWED_PAIR_RE.finditer(text):
        parts.append(redact_text(text[pos:m.start()]))
        parts.append(m.group(0))
        pos = m.end()
    parts.append(redact_text(text[pos:]))
    return "".join(parts)


# Значения секретов, которые передал оркестратор (set_known_secrets). Модуль сам их не ищет.
_KNOWN_SECRETS: tuple = ()
_MIN_KNOWN_SECRET_LEN = 8


def set_known_secrets(values) -> None:
    """Задаёт значения секретов (токен воркера, HF_TOKEN) для маски: каждое вхождение в тексте
    любой строки журнала и любого текста с карты заменяется на «***», даже без имени и без известной формы.

    Вызывает оркестратор один раз, когда значения известны (решение владельца 18.10.2026, этап 9а).
    Значения передаются аргументом; модуль их не читает ни из какого источника.
    Пустые значения и значения короче 8 символов игнорируются: короткое совпадение дало бы ложные маски.
    Повторный вызов заменяет прежний список (не добавляет к нему)."""
    global _KNOWN_SECRETS
    cleaned = {str(v).strip() for v in values if v is not None}
    cleaned = {v for v in cleaned if len(v) >= _MIN_KNOWN_SECRET_LEN}
    _KNOWN_SECRETS = tuple(sorted(cleaned, key=len, reverse=True))


def mask_text_with_mode(text: str, mode: str) -> str:
    """Маскирует текст в режиме "A" или "B" (см. выше). Модуль не читает окружение.

    Порядок: известные значения секретов (set_known_secrets), затем пары «имя»: значение и имя=значение
    по режиму, затем свободный текст (redact_text), затем шаблоны по форме значения.
    Белый список действует на пары; шаблоны по форме действуют всегда.
    """
    if mode not in _PAIR_PATTERNS:
        raise ValueError("Режим маскирования должен быть A или B.")
    json_re, assign_re = _PAIR_PATTERNS[mode]
    for secret in _KNOWN_SECRETS:
        text = text.replace(secret, _MASK)
    text = _BLOCK_JSON.sub(_json_repl, text)
    text = _BLOCK_ASSIGN.sub(_assign_repl, text)
    text = json_re.sub(_json_repl, text)
    text = assign_re.sub(_assign_repl, text)
    text = _redact_free_text(text)
    for pattern, keep_prefix in _VALUE_SHAPES:
        if keep_prefix:
            text = pattern.sub(lambda m: m.group(1) + _MASK, text)
        else:
            text = pattern.sub(_MASK, text)
    return text


def mask_text(text: str) -> str:
    """Маскирование текста в режиме MASK_MODE (сейчас A). Окружение не используется."""
    return mask_text_with_mode(text, MASK_MODE)


def mask_remote_text(text: str) -> str:
    """Общая функция маски для текста с карты (оркестратор, _mask_remote) и для журнала.

    Те же шаблоны по форме значения, пары режима MASK_MODE и белый список, что у журнала (SPEC 0.1, п. 14).
    Литеральные значения (токен воркера, HF_TOKEN, ключ Vast) оркестратор маскирует сам до вызова.
    """
    return mask_text(text)


class MaskFilter(logging.Filter):
    """Маскирует текст записи до того, как её запишет обработчик. Трейсбек не пишется (мог бы содержать секреты)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - сбой форматирования не должен ронять запуск
            message = "<сообщение не разобрано>"
        record.msg = mask_text(message)
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


class JournalFilter(logging.Filter):
    """Отсекает сырой журнал обмена с Vast: он идёт только в файл, не в консоль."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != JOURNAL_LOGGER:
            return True
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        return not message.startswith(JOURNAL_PREFIXES)


def setup_logging(level: str, artifact_dir: Optional[str]) -> tuple[list[logging.Handler], Optional[str]]:
    """Ставит обработчики на корневой логгер. Возвращает (обработчики, заметка или None).

    level: INFO или DEBUG (уже проверено в cli.py). artifact_dir: None = файла нет.
    Если файл не открылся, запуск продолжается с одной консолью; заметка для пользователя без пути.
    """
    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(CONSOLE_LEVEL)
    console.setFormatter(formatter)
    console.addFilter(JournalFilter())
    console.addFilter(MaskFilter())
    handlers: list[logging.Handler] = [console]

    note: Optional[str] = None
    if artifact_dir:
        try:
            path = Path(artifact_dir, *DEBUG_FILE_PARTS)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            file_handler.setLevel(getattr(logging, level))
            file_handler.setFormatter(formatter)
            file_handler.addFilter(MaskFilter())
            handlers.append(file_handler)
        except OSError:
            note = "Файл отладочного журнала не открыт: запуск продолжается, подробные строки не записываются."

    root = logging.getLogger()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(getattr(logging, level))
    return handlers, note


def teardown_logging(handlers: list[logging.Handler]) -> None:
    """Снимает обработчики с корневого логгера и закрывает файл."""
    root = logging.getLogger()
    for handler in handlers:
        root.removeHandler(handler)
        handler.close()
