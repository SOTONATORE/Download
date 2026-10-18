"""Обработчики журнала шага run (SPEC 10, этап 7а).

Два обработчика на корневом логгере:
- консоль (stderr, окно GitHub Actions): уровень INFO всегда; сырой журнал обмена с Vast
  (строки «Запрос Vast #» и «Ответ Vast #») в консоль не попадает;
- файл <каталог артефакта>/run/debug.log: уровень из GEN_LOG_LEVEL, пишет все записи,
  включая DEBUG и сырой журнал Vast (--debug-logs). Создаётся только при заданном каталоге
  артефакта; без каталога (локальный запуск) файла нет.

Каждая строка маскируется до записи в любой из обработчиков (SPEC 0.1, п. 2, 11).
Значения секретов читаются из окружения только для маскирования; сами ключи здесь не используются.
Формат строки совпадает с прежним (его разбирает фильтр в generate.yml).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

try:  # пакетный и «плоский» импорт, как в cli.py
    from Generate.core.gemini_prompts import ENV_KEY_NAME as GEMINI_KEY_ENV
    from Generate.core.orchestrator import _MASK, _SECRET_PATTERNS
    from Generate.core.vast_client import ENV_KEY_NAME as VAST_KEY_ENV
    from Generate.core.vast_client import redact_text
except ImportError:  # pragma: no cover
    from gemini_prompts import ENV_KEY_NAME as GEMINI_KEY_ENV  # type: ignore
    from orchestrator import _MASK, _SECRET_PATTERNS  # type: ignore
    from vast_client import ENV_KEY_NAME as VAST_KEY_ENV  # type: ignore
    from vast_client import redact_text  # type: ignore

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
CONSOLE_LEVEL = logging.INFO
# файл отладочного журнала внутри каталога артефакта (SPEC 10.4)
DEBUG_FILE_PARTS = ("run", "debug.log")
# строки сырого обмена с Vast: пишутся логгером клиента на уровне INFO (vast_client.py)
JOURNAL_LOGGER = "Generate.vast_client"
JOURNAL_PREFIXES = ("Запрос Vast #", "Ответ Vast #")
# литеральные значения секретов короче этого не маскируются (как в generate.yml)
MIN_LITERAL_LEN = 8
HF_TOKEN_ENV = "HF_TOKEN"


def _secret_values() -> list[str]:
    """Значения секретов из окружения, длинные первыми (чтобы короткое не съело длинное)."""
    names = (VAST_KEY_ENV, GEMINI_KEY_ENV, HF_TOKEN_ENV)
    values = {os.environ.get(name, "").strip() for name in names} - {""}
    return sorted((v for v in values if len(v) >= MIN_LITERAL_LEN), key=len, reverse=True)


def mask_text(text: str, secrets: list[str]) -> str:
    """Маскирует литеральные значения, пары «имя: значение» с секретным именем, Bearer и шаблоны ключей."""
    for value in secrets:
        text = text.replace(value, _MASK)
    text = redact_text(text)
    for pattern, keep_prefix in _SECRET_PATTERNS:
        if keep_prefix:
            text = pattern.sub(lambda m: m.group(1) + _MASK, text)
        else:
            text = pattern.sub(_MASK, text)
    return text


class MaskFilter(logging.Filter):
    """Маскирует текст записи до того, как её запишет обработчик. Трейсбек не пишется (мог бы содержать секреты)."""

    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self._secrets = secrets

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - сбой форматирования не должен ронять запуск
            message = "<сообщение не разобрано>"
        record.msg = mask_text(message, self._secrets)
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
    secrets = _secret_values()
    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(CONSOLE_LEVEL)
    console.setFormatter(formatter)
    console.addFilter(JournalFilter())
    console.addFilter(MaskFilter(secrets))
    handlers: list[logging.Handler] = [console]

    note: Optional[str] = None
    if artifact_dir:
        try:
            path = Path(artifact_dir, *DEBUG_FILE_PARTS)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            file_handler.setLevel(getattr(logging, level))
            file_handler.setFormatter(formatter)
            file_handler.addFilter(MaskFilter(secrets))
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
