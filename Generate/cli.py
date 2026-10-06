"""Точка входа второй части: ``python -m Generate.cli <команда> [параметры]``.

Команды:
    check       проверка входных файлов (SRT разбирается, missing.txt существует);
    run         основной запуск генерации (оркестратор, Gemini, Vast.ai, Release);
    kill-cards  уничтожение арендованных карт по метке (шаг ``if: always()``).

Коды завершения:
    0  успех (в том числе «генерировать нечего»);
    2  ошибка входных данных до любых трат (нет файла, битый SRT, неверные аргументы);
    3  прочая ошибка выполнения или частичный результат.

Ключи читаются только внутри клиентов и нигде не печатаются (SPEC 0.1).
Этот модуль сам ничего не читает из окружения (SPEC 0.2): репозиторий и номер
запуска передаются явными параметрами командной строки (--repo, --run-id).
Только стандартная библиотека + httpx. Пути не зашиты: берутся из параметров.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

from Generate.core.srt_parser import SrtError, parse_srt, srt_hash

try:  # пакетный и «плоский» импорт
    from Generate.core.orchestrator import run_generation
    from Generate.core.vast_client import VastClient, VastError
except ImportError:  # pragma: no cover
    from orchestrator import run_generation  # type: ignore
    from vast_client import VastClient, VastError  # type: ignore

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_RUNTIME = 3

# значения по умолчанию для входов (раздел 0, п.7 SPEC: имена файлов в корне репозитория)
DEFAULT_SRT = "final.srt"
DEFAULT_MISSING = "missing.txt"
DEFAULT_PROMPTS = "prompts.json"
DEFAULT_PROFILE = "ltx25"
STYLE_DIR = "Generate/model_profiles/prompt_styles"

# параметры таймингов по умолчанию (SPEC 3)
TIMING_DEFAULTS = {
    "fps": 24,
    "frame_rule": {"kind": "8k+1"},
    "clip_min_sec": 2.5,
    "clip_max_sec": 5.0,
    "gpu_max_sec": 10.0,
    "tail_pad_sec": 0.5,
}

_CORE_PREFIXES = ("Generate.core.", "")


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _find_symbol(name: str, modules: tuple[str, ...]):
    """Ищет класс в перечисленных модулях core (пакетный и плоский импорт)."""
    for mod in modules:
        for prefix in _CORE_PREFIXES:
            try:
                module = importlib.import_module(prefix + mod)
            except ImportError:
                continue
            obj = getattr(module, name, None)
            if obj is not None:
                return obj
    raise ImportError(f"не найден {name}")


def _build_timing():
    cls = _find_symbol("TimingParams", ("timing", "gemini_prompts", "gemini_orchestrator"))
    return cls(**TIMING_DEFAULTS)


def _build_prompt_config(profile: str, style_brief: str):
    cls = _find_symbol("PromptConfig", ("gemini_prompts", "gemini_orchestrator", "timing"))
    style_file = f"{STYLE_DIR}/{profile}.md"
    if not os.path.isfile(style_file):
        style_file = f"{STYLE_DIR}/{DEFAULT_PROFILE}.md"
    return cls(style_file=style_file, style_brief=style_brief)


def _make_transport():
    cls = _find_symbol("RealTransport", ("gemini_client", "gemini_transport",
                                          "gemini_orchestrator", "gemini_prompts"))
    return cls()  # ключ читается внутри класса


def _make_release_store(repo: str):
    cls = _find_symbol("GhReleaseStore", ("release_adapter",))
    return cls(repo=repo)


def _load_srt(srt_path: str):
    """Возвращает (сегменты, None) или (None, код_ошибки); сообщение уже напечатано."""
    if not os.path.isfile(srt_path):
        _err(f"Ошибка входных данных: файл SRT не найден: {srt_path}")
        return None, EXIT_INPUT
    try:
        return parse_srt(srt_path), None
    except SrtError as e:
        _err(f"Ошибка входных данных: файл SRT не разобран ({srt_path}). {e}")
        return None, EXIT_INPUT
    except (OSError, UnicodeDecodeError) as e:
        # из исключения берём только тип: текст может содержать лишнее
        _err(f"Ошибка входных данных: файл SRT не удалось прочитать "
             f"({srt_path}): {type(e).__name__}.")
        return None, EXIT_INPUT


def cmd_check(args: argparse.Namespace) -> int:
    """Проверяет входные файлы: SRT разбирается, missing.txt существует (не разбирается)."""
    segments, code = _load_srt(args.srt)
    if code is not None:
        return code

    # содержимое missing.txt здесь НЕ разбирается (отдельный этап, раздел 9)
    if not os.path.isfile(args.missing):
        _err(f"Ошибка входных данных: файл missing не найден: {args.missing}")
        return EXIT_INPUT

    print(f"Проверка пройдена. Сегментов в SRT: {len(segments)}")
    print(f"srt_hash: {srt_hash(segments)}")
    return EXIT_OK


def _print_summary(s) -> None:
    def g(name, default=0):
        return getattr(s, name, default)

    print("Итоги запуска:")
    print(f"  нужно сделать клипов: {g('total_needed')}")
    print(f"  готово: {g('completed')}")
    print(f"  с ошибкой: {g('failed')}")
    print(f"  не обработано: {g('pending')}")
    print(f"  уже было в Release: {g('skipped_done')}")
    print(f"  арендовано карт: {g('cards_rented')}")
    print(f"  потрачено, USD: {float(g('spent_usd', 0.0)):.2f}")
    if g("deadline_reached", False):
        print("  достигнут срок задания: новые клипы не брались")
    if g("budget_exceeded", False):
        print("  достигнут потолок расходов")
    if not g("cleanup_ok", True):
        print("  ВНИМАНИЕ: карты могли остаться арендованными, проверьте Vast.ai")


def cmd_run(args: argparse.Namespace) -> int:
    """Основной запуск генерации (этапы 4-9)."""
    segments, code = _load_srt(args.srt)
    if code is not None:
        return code
    if not os.path.isfile(args.missing):
        _err(f"Ошибка входных данных: файл missing не найден: {args.missing}")
        return EXIT_INPUT

    try:
        timing = _build_timing()
        prompt_cfg = _build_prompt_config(args.model_profile, args.style_brief)
        transport = _make_transport()
        store = _make_release_store(args.repo)
        vast = VastClient()
    except VastError as e:
        _err(f"Ошибка запуска: {e}")
        return EXIT_INPUT if e.exit_code == EXIT_INPUT else EXIT_RUNTIME
    except Exception as e:  # noqa: BLE001
        exit_code = EXIT_INPUT if getattr(e, "exit_code", None) == EXIT_INPUT else EXIT_RUNTIME
        # текст берём только у своих ошибок проекта (у них есть exit_code); иначе только тип
        if hasattr(e, "exit_code"):
            _err(f"Ошибка подготовки запуска: {e}")
        else:
            _err(f"Ошибка подготовки запуска ({type(e).__name__}).")
        return exit_code

    try:
        summary = run_generation(
            segments,
            args.missing,
            prompt_cfg,
            timing,
            store,
            vast,
            transport,
            args.prompts_path,
            max_cards=args.max_cards,
            budget_limit_usd=args.budget_limit,
            limit_clips=args.limit_clips,
            repo=args.repo,
            run_id=args.run_id,
            profile=args.model_profile,
            release_tag=args.release_tag,
        )
    finally:
        close = getattr(vast, "close", None)
        if callable(close):
            close()

    _print_summary(summary)
    return EXIT_OK if int(getattr(summary, "exit_code", EXIT_RUNTIME)) == EXIT_OK else EXIT_RUNTIME


def cmd_kill_cards(args: argparse.Namespace) -> int:
    """Уничтожает все карты запуска по метке (SPEC 8.3 (в))."""
    label = args.label or (f"gen-{args.run_id}" if args.run_id else "")
    if not label:
        _err("Ошибка входных данных: не задана метка (--label) и номер запуска (--run-id).")
        return EXIT_INPUT

    client = None
    try:
        client = VastClient()
        destroyed = client.destroy_by_label(label)
    except VastError as e:
        _err(f"Не удалось уничтожить карты: {e}")
        return EXIT_RUNTIME
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    print(f"Уничтожено карт: {len(destroyed)} (метка {label}).")
    return EXIT_OK


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
                   ).set_defaults(func=lambda a: cmd_check(a))

    run_p = sub.add_parser("run", parents=[common], help="основной запуск генерации")
    run_p.add_argument("--limit-clips", type=int, default=0,
                       help="0 = все номера; N > 0 = первые N (пробный прогон)")
    run_p.add_argument("--release-tag", default=None, help="тег Release (необязательно)")
    run_p.add_argument("--max-cards", type=int, default=4, help="потолок параллельных карт")
    run_p.add_argument("--budget-limit", type=float, default=5.0,
                       help="жёсткий потолок расходов на запуск, USD")
    run_p.add_argument("--model-profile", default=DEFAULT_PROFILE, help="профиль модели")
    run_p.add_argument("--style-brief", default="", help="общее описание стиля/мира")
    run_p.add_argument("--repo", default="",
                       help="репозиторий owner/name (передаётся явно)")
    run_p.add_argument("--run-id", default="",
                       help="номер запуска (передаётся явно)")
    run_p.add_argument("--prompts-path", default=DEFAULT_PROMPTS,
                       help=f"путь к файлу промптов (по умолчанию {DEFAULT_PROMPTS})")
    run_p.set_defaults(func=lambda a: cmd_run(a))

    kill_p = sub.add_parser("kill-cards", help="уничтожить арендованные карты по метке")
    kill_p.add_argument("--label", default=None, help="метка карт (по умолчанию gen-<run-id>)")
    kill_p.add_argument("--run-id", default="",
                        help="номер запуска (передаётся явно)")
    kill_p.set_defaults(func=lambda a: cmd_kill_cards(a))
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
