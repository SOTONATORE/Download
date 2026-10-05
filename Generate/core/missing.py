"""Разбор файла «пропущенных» номеров сегментов (SPEC 9.6).

Чистый модуль: только стандартная библиотека, без сети и БД.
Путь к файлу передаётся аргументом, ничего не зашито в логике.
"""
from __future__ import annotations

import logging
from pathlib import Path

try:
    from .gemini_prompts import GeminiInputError
    from .srt_parser import Segment
except ImportError:  # запуск без пакета (плоская раскладка)
    from gemini_prompts import GeminiInputError  # type: ignore
    from srt_parser import Segment  # type: ignore

log = logging.getLogger("Generate.missing")

# Заглушка из SPEC 9.6: пропущенных сегментов нет.
STUB_LINE = "Пропущенных сегментов нет - все найдены и успешно скачаны."
_STUB_NORMALIZED = STUB_LINE.casefold()

_PREVIEW_LEN = 80


def _preview(line: str) -> str:
    return line if len(line) <= _PREVIEW_LEN else line[:_PREVIEW_LEN] + "…"


def _is_number(line: str) -> bool:
    # str.isdigit() пропускает «²» и подобное, поэтому дополнительно требуем ASCII
    return line.isascii() and line.isdigit()


def parse_missing(missing_path: str | Path, segments: list[Segment]) -> list[int]:
    """Читает файл пропущенных номеров и возвращает отсортированные уникальные номера.

    Возвращает [] если в файле только строка-заглушка (генерировать нечего).
    При любом нарушении формата бросает GeminiInputError (код выхода 2).
    """
    path = Path(missing_path)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise GeminiInputError(f"Файл пропущенных сегментов не найден: {path.name}") from None
    except UnicodeDecodeError:
        raise GeminiInputError(
            f"Файл пропущенных сегментов {path.name} не в кодировке UTF-8."
        ) from None
    except OSError as e:
        raise GeminiInputError(
            f"Не удалось прочитать файл пропущенных сегментов {path.name}: {e.strerror or 'ошибка ввода-вывода'}"
        ) from None

    lines = [ln.strip() for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln]

    if not lines:
        raise GeminiInputError(
            f"Файл пропущенных сегментов {path.name} пуст: нет ни номеров, ни строки "
            f"«{STUB_LINE}»."
        )

    numbers: list[int] = []
    has_stub = False
    for ln in lines:
        if _is_number(ln):
            numbers.append(int(ln))
        elif ln.casefold() == _STUB_NORMALIZED:
            has_stub = True
        else:
            raise GeminiInputError(
                f"В файле пропущенных сегментов {path.name} найдена недопустимая строка: "
                f"«{_preview(ln)}». Допустимы только целые номера или строка «{STUB_LINE}»."
            )

    if has_stub:
        if numbers:
            raise GeminiInputError(
                f"Файл пропущенных сегментов {path.name} содержит одновременно номера "
                f"и строку «{STUB_LINE}»."
            )
        log.info("Пропущенных сегментов нет: генерировать нечего.")
        return []

    unique = sorted(set(numbers))
    dup_count = len(numbers) - len(unique)
    if dup_count:
        log.warning("В файле пропущенных сегментов повторяющихся номеров: %d; дубли объединены.", dup_count)

    known = {s.num for s in segments}
    absent = [n for n in unique if n not in known]
    if absent:
        raise GeminiInputError(
            f"Номера из файла пропущенных сегментов отсутствуют в SRT: {absent}."
        )
    return unique
