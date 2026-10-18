"""Обработчики журнала шага run (SPEC 10, этап 7а).

Два обработчика на корневом логгере:
- консоль (stderr, окно GitHub Actions): уровень INFO всегда; сырой журнал обмена с Vast
  (строки «Запрос Vast #» и «Ответ Vast #») в консоль не попадает;
- файл <каталог артефакта>/run/debug.log: уровень из GEN_LOG_LEVEL, пишет все записи,
  включая DEBUG и сырой журнал Vast (--debug-logs). Создаётся только при заданном каталоге
  артефакта; без каталога (локальный запуск) файла нет.

Каждая строка маскируется до записи в любой из обработчиков (SPEC 0.1, п. 2, 11).
Маскирование идёт по именам (решение владельца 18.10.2026): модуль не читает окружение и не знает значений секретов.
Формат строки совпадает с прежним (его разбирает фильтр в generate.yml).
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

try:  # пакетный и «плоский» импорт, как в cli.py
    from Generate.core.orchestrator import _MASK, _SECRET_PATTERNS
    from Generate.core.vast_client import redact_text
except ImportError:  # pragma: no cover
    from orchestrator import _MASK, _SECRET_PATTERNS  # type: ignore
    from vast_client import redact_text  # type: ignore

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
CONSOLE_LEVEL = logging.INFO
# файл отладочного журнала внутри каталога артефакта (SPEC 10.4)
DEBUG_FILE_PARTS = ("run", "debug.log")
# строки сырого обмена с Vast: пишутся логгером клиента на уровне INFO (vast_client.py)
JOURNAL_LOGGER = "Generate.vast_client"
JOURNAL_PREFIXES = ("Запрос Vast #", "Ответ Vast #")
# Маскирование по именам (SPEC 0.1, п. 11; решение владельца 18.10.2026).
# ЗНАЧЕНИЯ ДЛЯ МАСКИРОВАНИЯ ИЗ ОКРУЖЕНИЯ НЕ ЧИТАЮТСЯ: маскируется то, что похоже на секрет по имени.
# Имя секретное, если содержит API_KEY, APIKEY, TOKEN, SECRET или PASSWORD (без учёта регистра).
# Значение заменяется на *** до первого пробела, кавычки, запятой или конца строки; значение в кавычках
# маскируется до закрывающей кавычки. Формы: NAME=значение и "NAME": "значение".
_SECRET_NAME = r"[\w.\-]*(?:API_?KEY|TOKEN|SECRET|PASSWORD)[\w.\-]*"
_ASSIGN_RE = re.compile(
    r"(?i)(?<![\w.\-])(" + _SECRET_NAME + r"=)(\"[^\"\r\n]*\"?|'[^'\r\n]*'?|[^\s\"',]*)"
)
_JSON_PAIR_RE = re.compile(r"(?i)(\"" + _SECRET_NAME + r"\"\s*:\s*)\"[^\"\r\n]*\"?")


def _assign_repl(m: "re.Match[str]") -> str:
    value = m.group(2)
    if value[:1] in ('"', "'"):
        return m.group(1) + value[0] + _MASK + value[0]
    return m.group(1) + _MASK


def mask_text(text: str) -> str:
    """Маскирует пары по именам (см. выше), пары «имя: значение» из redact_text (тоже по именам),
    заголовок Authorization и шаблоны ключей из orchestrator. Окружение не используется."""
    text = _JSON_PAIR_RE.sub(lambda m: m.group(1) + f'"{_MASK}"', text)
    text = _ASSIGN_RE.sub(_assign_repl, text)
    text = redact_text(text)
    for pattern, keep_prefix in _SECRET_PATTERNS:
        if keep_prefix:
            text = pattern.sub(lambda m: m.group(1) + _MASK, text)
        else:
            text = pattern.sub(_MASK, text)
    return text


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
