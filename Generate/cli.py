"""Точка входа второй части: ``python -m Generate.cli <команда> [параметры]``.

Команды:
    check       проверка конфигурации и входных файлов (SRT разбирается, missing.txt существует);
    run         основной запуск генерации (оркестратор, Gemini, Vast.ai, Release);
    kill-cards  уничтожение арендованных карт по метке (шаг ``if: always()``);
    finalize-release  выдача уже готовых клипов: prompts.json и generated_links.txt
                (шаг ``if: always()``, в том числе после отмены запуска).

Коды завершения:
    0  успех (в том числе «генерировать нечего»);
    2  ошибка входных данных или конфигурации до любых трат (нет файла, битый SRT,
       неверный config, неверные аргументы);
    3  прочая ошибка выполнения или частичный результат.

Настройки: значения по умолчанию берутся из Generate/config.default.yaml через
Generate/core/config_loader.py (схема проверяется, см. SPEC 3). Входы workflow
(аргументы CLI) переопределяют файл и проходят ту же проверку. Секреты сюда не попадают.

Ключи читаются только внутри клиентов и нигде не печатаются (SPEC 0.1).
Этот модуль сам ничего не читает из окружения (SPEC 0.2): репозиторий и номер
запуска передаются явными параметрами командной строки (--repo, --run-id).
Стандартная библиотека, httpx и PyYAML (конфиг). Пути не зашиты: берутся из параметров.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from Generate.core.srt_parser import SrtError, parse_srt, srt_hash

try:  # пакетный и «плоский» импорт
    from Generate.core.config_loader import (
        DEFAULT_CONFIG_PATH, Config, load_config, load_profile_number)
    from Generate.core.gemini_prompts import GeminiInputError
    from Generate.core.orchestrator import run_generation
    from Generate.core.release_adapter import (
        LINKS_FILENAME, PROMPTS_FILENAME, ReleaseUploader, make_tag)
    from Generate.core.vast_client import (
        OfferFilter, VastClient, VastError, VastRuntimeError)
except ImportError:  # pragma: no cover
    from config_loader import (  # type: ignore
        DEFAULT_CONFIG_PATH, Config, load_config, load_profile_number)
    from gemini_prompts import GeminiInputError  # type: ignore
    from orchestrator import run_generation  # type: ignore
    from release_adapter import (  # type: ignore
        LINKS_FILENAME, PROMPTS_FILENAME, ReleaseUploader, make_tag)
    from vast_client import (  # type: ignore
        OfferFilter, VastClient, VastError, VastRuntimeError)

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_RUNTIME = 3

log = logging.getLogger("Generate.cli")

# имена входных файлов по умолчанию (раздел 0, п.7 SPEC: в корне репозитория; это не настройка)
DEFAULT_SRT = "final.srt"
DEFAULT_MISSING = "missing.txt"
DEFAULT_PROMPTS = "prompts.json"
STYLE_DIR = "Generate/model_profiles/prompt_styles"
# запасной стиль, если файла стиля для профиля нет (прежнее поведение)
FALLBACK_STYLE_PROFILE = "ltx25"
# значения по умолчанию, которых нет в config.default.yaml (описаны в профиле модели и SPEC 5)
TIMING_FRAME_RULE = {"kind": "8k+1"}


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _usd_floor(value: float) -> str:
    """Баланс в USD до цента с округлением вниз: 4.996 не покажется как «5.00» при минимуме 5."""
    return f"{math.floor(value * 100) / 100:.2f}"


def _balance_preflight(vast, min_balance_usd: float) -> Optional[int]:
    """Предстартовая проверка баланса Vast.ai (SPEC 8.3, е).

    Вызывается до поиска предложений и до любого вызова Gemini. Возвращает None, если
    запуск можно продолжать, иначе код завершения (сообщение уже напечатано):
    2 — баланса не хватает, 3 — баланс не удалось проверить. В обоих случаях ничего не потрачено.
    """
    try:
        balance = vast.get_balance_usd()
    except VastRuntimeError as e:
        # текст ошибки клиента безопасен: ключ в нём уже замаскирован (SPEC 0.1)
        _err(f"Не удалось проверить баланс Vast.ai: {e}. "
             f"Запуск не начат, карты не арендованы.")
        return EXIT_RUNTIME
    if balance < min_balance_usd:
        _err(f"Недостаточно средств на Vast.ai: баланс {_usd_floor(balance)} USD, "
             f"нужно не менее {min_balance_usd:.2f} USD. Пополните баланс и запустите снова.")
        return EXIT_INPUT
    log.info("Баланс Vast.ai: %s USD, минимум %.2f USD.", _usd_floor(balance), min_balance_usd)
    return None


def _vast_preflight(vast, min_balance_usd: float) -> Optional[int]:
    """Предстартовая проверка Vast API (SPEC 8.3, е): только чтение, до любого поиска и аренды.

    Сначала эндпоинт списка экземпляров: без него уборка (kill-cards) не сможет найти карты,
    и ошибка выплыла бы уже после аренды. Затем баланс. Возвращает None (можно продолжать)
    или код завершения; сообщение уже напечатано. В обоих случаях ничего не арендовано.
    """
    try:
        vast.check_instances_endpoint()
    except VastRuntimeError as e:
        # текст ошибки клиента безопасен: ключ в нём уже замаскирован, путь эндпоинта назван (SPEC 0.1)
        _err(f"Предстартовая проверка Vast API не пройдена: {e}. "
             f"Запуск не начат, карты не арендованы.")
        return EXIT_RUNTIME
    log.info("Эндпоинт списка экземпляров Vast отвечает.")
    return _balance_preflight(vast, min_balance_usd)


def _find_symbol(name: str, modules: tuple[str, ...]):
    """Ищет класс в перечисленных модулях core (пакетный и плоский импорт)."""
    for mod in modules:
        for prefix in ("Generate.core.", ""):
            try:
                module = importlib.import_module(prefix + mod)
            except ImportError:
                continue
            obj = getattr(module, name, None)
            if obj is not None:
                return obj
    raise ImportError(f"не найден {name}")


def _load_cfg(path: Optional[str], overrides: Optional[dict] = None) -> tuple[Optional[Config], Optional[int]]:
    """Возвращает (конфиг, None) или (None, код_ошибки); сообщение уже напечатано."""
    try:
        return load_config(path, overrides), None
    except GeminiInputError as e:
        _err(str(e))
        return None, EXIT_INPUT


def _build_timing(cfg: Config, profile: str):
    cls = _find_symbol("TimingParams", ("timing", "gemini_prompts", "gemini_orchestrator"))
    # gpu_max_sec в конфиге может быть null: тогда берётся из профиля модели
    gpu_max = (cfg.gpu_max_sec if cfg.gpu_max_sec is not None
               else load_profile_number(profile, "gpu_max_sec"))
    return cls(fps=cfg.fps, frame_rule=dict(TIMING_FRAME_RULE),
               clip_min_sec=cfg.clip_min_sec, clip_max_sec=cfg.clip_max_sec,
               gpu_max_sec=gpu_max, tail_pad_sec=cfg.tail_pad_sec)


def _build_prompt_config(profile: str, style_brief: str, batch_size: int):
    cls = _find_symbol("PromptConfig", ("gemini_prompts", "gemini_orchestrator", "timing"))
    style_file = f"{STYLE_DIR}/{profile}.md"
    if not os.path.isfile(style_file):
        style_file = f"{STYLE_DIR}/{FALLBACK_STYLE_PROFILE}.md"
    return cls(style_file=style_file, style_brief=style_brief, batch_size=batch_size)


def _gemini_options(cfg: Config) -> dict:
    return {"models": tuple(cfg.gemini_models), "rpm_limit": cfg.gemini_rpm_limit,
            "tpm_limit": cfg.gemini_tpm_limit, "rpd_limit": cfg.gemini_rpd_limit}


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
    """Проверяет конфигурацию, SRT и наличие missing.txt (без разбора missing)."""
    _cfg, code = _load_cfg(args.config)
    if code is not None:
        return code

    segments, code = _load_srt(args.srt)
    if code is not None:
        return code

    # содержимое missing.txt здесь НЕ разбирается (отдельный этап, раздел 9)
    if not os.path.isfile(args.missing):
        _err(f"Ошибка входных данных: файл missing не найден: {args.missing}")
        return EXIT_INPUT

    print(f"Конфигурация в порядке: {Path(args.config or DEFAULT_CONFIG_PATH).name}")
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
    # входы workflow (None = не задано) переопределяют config.default.yaml и проверяются так же
    overrides = {
        "limit_clips": args.limit_clips,
        "max_cards": args.max_cards,
        "budget_limit_usd": args.budget_limit,
        "min_balance_usd": args.min_balance_usd,
        "model_profile": args.model_profile,
        "style_brief": args.style_brief,
        "gpu_name": args.gpu_name,
        "min_price_per_hour": args.min_price_per_hour,
        "max_price_per_hour": args.max_price_per_hour,
        "min_reliability": args.min_reliability,
        "min_inet_mbps": args.min_inet_mbps,
        "docker_image": args.docker_image,
        "disk_gb": args.disk_gb,
        "silent_host_timeout_min": args.silent_host_timeout_min,
    }
    cfg, code = _load_cfg(args.config, overrides)
    if code is not None:
        return code

    segments, code = _load_srt(args.srt)
    if code is not None:
        return code
    if not os.path.isfile(args.missing):
        _err(f"Ошибка входных данных: файл missing не найден: {args.missing}")
        return EXIT_INPUT

    try:
        timing = _build_timing(cfg, cfg.model_profile)
        prompt_cfg = _build_prompt_config(cfg.model_profile, cfg.style_brief, cfg.gemini_batch_size)
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

    offer_filter = OfferFilter(
        gpu_name=cfg.gpu_name or None,
        min_price=cfg.min_price_per_hour,
        max_price=cfg.max_price_per_hour,
        min_reliability=cfg.min_reliability,
        min_inet_mbps=cfg.min_inet_mbps,
        interruptible=cfg.interruptible,
    )

    try:
        # предстартовая проверка Vast API и баланса: до поиска предложений и любого вызова Gemini
        code = _vast_preflight(vast, cfg.min_balance_usd)
        if code is not None:
            return code
        summary = run_generation(
            segments,
            args.missing,
            prompt_cfg,
            timing,
            store,
            vast,
            transport,
            args.prompts_path,
            max_cards=cfg.max_cards,
            budget_limit_usd=cfg.budget_limit_usd,
            limit_clips=cfg.limit_clips,
            repo=args.repo,
            run_id=args.run_id,
            profile=cfg.model_profile,
            release_tag=args.release_tag,
            offer_filter=offer_filter,
            docker_image=cfg.docker_image,
            disk_gb=cfg.disk_gb,
            silent_host_timeout_min=cfg.silent_host_timeout_min,
            job_deadline_min=cfg.job_deadline_min,
            card_max_lifetime_min=cfg.card_max_lifetime_min,
            worker_idle_timeout_min=cfg.worker_idle_timeout_min,
            filename_template=cfg.filename_template,
            clip_name_width=cfg.clip_name_width,
            release_asset_limit=cfg.release_asset_limit,
            gemini_options=_gemini_options(cfg),
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


def cmd_finalize_release(args: argparse.Namespace) -> int:
    """Доводит выдачу до конца: prompts.json и generated_links.txt по уже загруженным клипам.

    Безопасна при повторном запуске и после отмены: если Release ещё не создан,
    завершается с кодом 0 (финализировать нечего).
    """
    cfg, code = _load_cfg(args.config, {"model_profile": args.model_profile})
    if code is not None:
        return code

    segments, code = _load_srt(args.srt)
    if code is not None:
        return code

    try:
        tag = make_tag(segments, args.release_tag)
        store = _make_release_store(args.repo)
        if not store.find_releases(tag):
            print(f"Release {tag} ещё не создан: финализировать нечего.")
            return EXIT_OK

        uploader = ReleaseUploader(store, segments, cfg.model_profile,
                                   release_tag=args.release_tag,
                                   release_asset_limit=cfg.release_asset_limit,
                                   filename_template=cfg.filename_template)
        uploader.prepare()
        clips = len(uploader.done)

        with tempfile.TemporaryDirectory() as tmp:
            prompts_uploaded = False
            if os.path.isfile(args.prompts_path):
                # служебный файл должен называться строго prompts.json
                prompts_copy = os.path.join(tmp, PROMPTS_FILENAME)
                shutil.copyfile(args.prompts_path, prompts_copy)
                uploader.upload_service_file(prompts_copy)
                prompts_uploaded = True
            else:
                print(f"Файл промптов не найден ({args.prompts_path}): он не загружен.")

            links_uploaded = False
            if args.repo and uploader.done:
                links_path = os.path.join(tmp, LINKS_FILENAME)
                with open(links_path, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(uploader.links_text(args.repo))
                uploader.upload_service_file(links_path)
                links_uploaded = True
    except Exception as e:  # noqa: BLE001
        # текст берём только у своих ошибок проекта (у них есть exit_code); иначе только тип
        if hasattr(e, "exit_code"):
            _err(f"Не удалось завершить выдачу клипов: {e}")
        else:
            _err(f"Не удалось завершить выдачу клипов ({type(e).__name__}).")
        return EXIT_RUNTIME

    print("Выдача клипов завершена.")
    print(f"  релиз: {tag}")
    print(f"  клипов в релизах: {clips}")
    print(f"  prompts.json загружен: {'да' if prompts_uploaded else 'нет'}")
    print(f"  generated_links.txt загружен: {'да' if links_uploaded else 'нет'}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--srt", default=DEFAULT_SRT,
                        help=f"путь к файлу SRT (по умолчанию {DEFAULT_SRT})")
    common.add_argument("--missing", default=DEFAULT_MISSING,
                        help=f"путь к файлу missing (по умолчанию {DEFAULT_MISSING})")
    common.add_argument("--config", default=None,
                        help="путь к файлу конфигурации (по умолчанию Generate/config.default.yaml)")

    parser = argparse.ArgumentParser(
        prog="python -m Generate.cli",
        description="Вторая часть: генерация клипов по SRT.",
    )
    sub = parser.add_subparsers(dest="command", metavar="команда", required=True)

    # функции берутся из модуля при каждом вызове (удобно подменять в тестах)
    sub.add_parser("check", parents=[common],
                   help="проверить конфигурацию и входные файлы"
                   ).set_defaults(func=lambda a: cmd_check(a))

    # Значения None = «не задано во входах»: берутся из конфига (см. cmd_run).
    run_p = sub.add_parser("run", parents=[common], help="основной запуск генерации")
    run_p.add_argument("--limit-clips", type=int, default=None,
                       help="0 = все номера; N > 0 = первые N (пробный прогон); по умолчанию из конфига")
    run_p.add_argument("--release-tag", default=None, help="тег Release (необязательно)")
    run_p.add_argument("--max-cards", type=int, default=None,
                       help="потолок параллельных карт; по умолчанию из конфига")
    run_p.add_argument("--budget-limit", type=float, default=None,
                       help="жёсткий потолок расходов на запуск, USD; по умолчанию из конфига")
    run_p.add_argument("--min-balance-usd", type=float, default=None,
                       help="минимальный баланс Vast.ai для старта, USD; по умолчанию из конфига")
    run_p.add_argument("--model-profile", default=None,
                       help="профиль модели; по умолчанию из конфига")
    run_p.add_argument("--style-brief", default=None,
                       help="общее описание стиля/мира; по умолчанию из конфига")
    run_p.add_argument("--repo", default="",
                       help="репозиторий owner/name (передаётся явно)")
    run_p.add_argument("--run-id", default="",
                       help="номер запуска (передаётся явно)")
    run_p.add_argument("--prompts-path", default=DEFAULT_PROMPTS,
                       help=f"путь к файлу промптов (по умолчанию {DEFAULT_PROMPTS})")
    run_p.add_argument("--gpu-name", default=None,
                       help="модель GPU для поиска; пустая строка = любая; по умолчанию из конфига")
    run_p.add_argument("--min-price-per-hour", type=float, default=None,
                       help="нижняя граница цены, USD/час; по умолчанию из конфига")
    run_p.add_argument("--max-price-per-hour", type=float, default=None,
                       help="верхняя граница цены, USD/час; по умолчанию из конфига")
    run_p.add_argument("--min-reliability", type=float, default=None,
                       help="минимальная надёжность хоста, от 0 до 1; по умолчанию из конфига")
    run_p.add_argument("--min-inet-mbps", type=float, default=None,
                       help="минимальная скорость входящего канала, Мбит/с; по умолчанию из конфига")
    run_p.add_argument("--docker-image", default=None,
                       help="образ Docker для карты Vast.ai; по умолчанию из конфига")
    run_p.add_argument("--disk-gb", type=int, default=None,
                       help="размер диска для карты Vast.ai, ГБ; по умолчанию из конфига")
    run_p.add_argument("--silent-host-timeout-min", type=int, default=None,
                       help="таймаут первого ответа хоста, минут; по умолчанию из конфига")
    run_p.set_defaults(func=lambda a: cmd_run(a))

    kill_p = sub.add_parser("kill-cards", help="уничтожить арендованные карты по метке")
    kill_p.add_argument("--label", default=None, help="метка карт (по умолчанию gen-<run-id>)")
    kill_p.add_argument("--run-id", default="",
                        help="номер запуска (передаётся явно)")
    kill_p.set_defaults(func=lambda a: cmd_kill_cards(a))

    fin_p = sub.add_parser("finalize-release",
                           help="загрузить prompts.json и generated_links.txt в Release")
    fin_p.add_argument("--srt", default=DEFAULT_SRT,
                       help=f"путь к файлу SRT (по умолчанию {DEFAULT_SRT})")
    fin_p.add_argument("--config", default=None,
                       help="путь к файлу конфигурации (по умолчанию Generate/config.default.yaml)")
    fin_p.add_argument("--repo", default="", help="репозиторий owner/name (передаётся явно)")
    fin_p.add_argument("--model-profile", default=None,
                       help="профиль модели; по умолчанию из конфига")
    fin_p.add_argument("--release-tag", default=None, help="тег Release (необязательно)")
    fin_p.add_argument("--prompts-path", default=DEFAULT_PROMPTS,
                       help=f"путь к файлу промптов (по умолчанию {DEFAULT_PROMPTS})")
    fin_p.set_defaults(func=lambda a: cmd_finalize_release(a))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Возвращает код завершения. Трейсбеки пользователю не показываются."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
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
