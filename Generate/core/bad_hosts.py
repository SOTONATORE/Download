"""Постоянный список плохих машин Vast.ai (SPEC 3, 8.2).

Файл ``Generate/bad_hosts.txt``: один ID машины на строку. ID — это «Machine ID» из консоли
Vast, а не ID хоста. Всё после ``#`` считается комментарием. Пустые строки и строки-комментарии
допустимы. Повторы сворачиваются в один элемент.

Список читается один раз при старте запуска (``cli.cmd_run``) и передаётся оркестратору.
Машины из него не арендуются ни в одном запуске. Это отдельно от ``_bad_machines`` оркестратора,
которое живёт только в пределах одного запуска (SPEC 8.2).

Путь строится от расположения этого файла, а не от текущего каталога. Переменные окружения
здесь не читаются. Секреты в файл не попадают и не должны попадать.
Сообщения на русском (SPEC 0.5).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger("Generate.bad_hosts")

GENERATE_DIR = Path(__file__).resolve().parent.parent
BAD_HOSTS_FILENAME = "bad_hosts.txt"
DEFAULT_BAD_HOSTS_PATH = GENERATE_DIR / BAD_HOSTS_FILENAME
EXIT_INPUT = 2

# только ASCII-цифры: без знаков, пробелов внутри и дробной части
_ID_RE = re.compile(r"[0-9]+")


class BadHostsError(Exception):
    """Неверный формат файла списка. Код выхода 2: до любых трат (SPEC 2, коды CLI)."""
    exit_code = EXIT_INPUT


def parse_bad_hosts(text: str) -> frozenset:
    """Разбирает текст файла. Все ошибки собираются и выводятся разом, с номерами строк.

    Содержимое строки в сообщение не попадает: в файле не должно быть ничего, кроме ID,
    а случайно вставленный секрет не должен попасть в лог.
    """
    ids: set = set()
    errors: list = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        body = raw.split("#", 1)[0].strip()
        if not body:
            continue
        if not _ID_RE.fullmatch(body):
            errors.append(f"строка {lineno}: ожидается одно целое положительное число "
                          f"(ID машины), без букв, знаков и пробелов внутри")
            continue
        value = int(body)
        if value <= 0:
            errors.append(f"строка {lineno}: ID машины должен быть больше нуля")
            continue
        ids.add(value)
    if errors:
        raise BadHostsError(
            f"неверный формат файла {BAD_HOSTS_FILENAME}: " + "; ".join(errors)
            + ". Исправьте файл и запустите снова.")
    return frozenset(ids)


def load_bad_hosts(path: Optional[Path] = None) -> frozenset:
    """Читает список плохих машин. Файла нет или он пуст: пустое множество, без ошибки.

    Ошибка формата или чтения: ``BadHostsError`` (код выхода 2).
    """
    p = Path(path) if path is not None else DEFAULT_BAD_HOSTS_PATH
    if not p.exists():
        ids = frozenset()
    else:
        try:
            # utf-8-sig: BOM, который добавляют некоторые редакторы, не считается ошибкой
            text = p.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            raise BadHostsError(
                f"файл {BAD_HOSTS_FILENAME} не читается как текст UTF-8") from None
        ids = parse_bad_hosts(text)
    if ids:
        log.info("Постоянный список плохих машин: %d шт. (ID: %s).",
                 len(ids), ", ".join(str(i) for i in sorted(ids)))
    else:
        log.info("Постоянный список плохих машин пуст (файл %s отсутствует или без записей).",
                 BAD_HOSTS_FILENAME)
    return ids
