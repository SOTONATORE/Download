"""Точка входа второй части: ``python -m Generate.cli <команда> [параметры]``.

Каркас (этап 3): реализована только проверка входных файлов ``check``,
команды ``run`` и ``kill-cards`` пока заглушки.

Коды завершения:
    0  успех (в том числе «генерировать нечего»);
    2  ошибка входных данных до любых трат (нет файла, битый SRT, неверные аргументы);
    3  прочая ошибка выполнения, включая «не реализовано».

Ключи на этом этапе не читаются (SPEC 0.1). Только стандартная библиотека.
Пути не зашиты: берутся из параметров.
"""
from __future__ import annotations

import argparse
import os
import sys

from Generate.core.srt_parser import SrtError, parse_srt, srt_hash

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_RUNTIME = 3

# значения по умолчанию для входов (раздел 0, п.7 SPEC: имена файлов в корне репозитория)
DEFAULT_SRT = "final.srt"
DEFAULT_MISSING = "missing.txt"

NOT_IMPLEMENTED = "Команда ещё не реализована."


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def cmd_check(args: argparse.Namespace) -> int:
    """Проверяет входные файлы: SRT разбирается, missing.txt существует (не разбирается)."""
    srt_path = args.srt
    missing_path = args.missing

    if not os.path.isfile(srt_path):
        _err(f"Ошибка входных данных: файл SRT не найден: {srt_path}")
        return EXIT_INPUT
    try:
        segments = parse_srt(srt_path)
    except SrtError as e:
        _err(f"Ошибка входных данных: файл SRT не разобран ({srt_path}). {e}")
        return EXIT_INPUT
    except (OSError, UnicodeDecodeError) as e:
        # из исключения берём только тип: текст может содержать лишнее
        _err(f"Ошибка входных данных: файл SRT не удалось прочитать "
             f"({srt_path}): {type(e).__name__}.")
        return EXIT_INPUT

    # содержимое missing.txt здесь НЕ разбирается (отдельный этап, раздел 9)
    if not os.path.isfile(missing_path):
        _err(f"Ошибка входных данных: файл missing не найден: {missing_path}")
        return EXIT_INPUT

    print(f"Проверка пройдена. Сегментов в SRT: {len(segments)}")
    print(f"srt_hash: {srt_hash(segments)}")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    """Заглушка: основной запуск генерации (этапы 4-9)."""
    _err(NOT_IMPLEMENTED)
    return EXIT_RUNTIME


def cmd_kill_cards(args: argparse.Namespace) -> int:
    """Заглушка: короткая команда для шага ``if: always()`` (SPEC 8.3 (в))."""
    _err(NOT_IMPLEMENTED)
    return EXIT_RUNTIME


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--srt", default=DEFAULT_SRT,
                        help=f"путь к файлу SRT (по умолчанию {DEFAULT_SRT})")
    common.add_argument("--missing", default=DEFAULT_MISSING,
                        help=f"путь к файлу missing (по умолчанию {DEFAULT_MISSING})")

    parser = argparse.ArgumentParser(
        prog="python -m Generate.cli",
        description="Вторая часть: генерация клипов по SRT.",
    )
    sub = parser.add_subparsers(dest="command", metavar="команда", required=True)

    # функции берутся из модуля при каждом вызове (удобно подменять в тестах)
    sub.add_parser("check", parents=[common],
                   help="проверить входные файлы (SRT разбирается, missing существует)"
                   ).set_defaults(func=cmd_check)
    sub.add_parser("run", parents=[common],
                   help="основной запуск (пока не реализовано)"
                   ).set_defaults(func=cmd_run)
    sub.add_parser("kill-cards", parents=[common],
                   help="уничтожить арендованные карты (пока не реализовано)"
                   ).set_defaults(func=cmd_kill_cards)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Возвращает код завершения. Трейсбеки пользователю не показываются."""
    try:
        parser = build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit as e:  # argparse: 2 при ошибке аргументов, 0 для --help
            code = e.code
            return code if isinstance(code, int) else EXIT_INPUT
        return int(args.func(args))
    except Exception as e:  # noqa: BLE001 - намеренно ловим всё непредвиденное
        # Печатаем только тип исключения: текст может содержать секреты (SPEC 0.1).
        _err(f"Непредвиденная ошибка выполнения ({type(e).__name__}).")
        return EXIT_RUNTIME


if __name__ == "__main__":
    # на Windows при перенаправлении вывод может быть не в UTF-8: не падаем на кириллице
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    sys.exit(main())
