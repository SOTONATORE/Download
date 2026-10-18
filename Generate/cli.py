"""Точка входа второй части: ``python -m Generate.cli <команда> [параметры]``.

Команды:
    check       проверка конфигурации и входных файлов (SRT разбирается, missing.txt существует);
    run         основной запуск генерации (оркестратор, Gemini, Vast.ai, Release);
    kill-cards  уничтожение арендованных карт по метке (шаг ``if: always()``);
    watch-heartbeat  наблюдатель heartbeat: фоновый процесс шага run (SPEC 8.3 а);
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
Из окружения читается только GEN_LOG_LEVEL (INFO или DEBUG, по умолчанию INFO);
он же передаётся карте. Репозиторий и номер запуска передаются явными параметрами
командной строки (--repo, --run-id). Стандартная библиотека, httpx и PyYAML (конфиг).
Пути не зашиты: берутся из параметров.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from Generate.core.srt_parser import SrtError, parse_srt, srt_hash

try:  # пакетный и «плоский» импорт
    from Generate.core.bad_hosts import BadHostsError, load_bad_hosts
    from Generate.core.config_loader import (
        DEFAULT_CONFIG_PATH, Config, load_config, load_profile_number)
    from Generate.core.gemini_prompts import GeminiInputError
    from Generate.core.log_setup import mask_text, setup_logging, teardown_logging
    from Generate.core.orchestrator import (
        ReleaseStartError, RunPublisher, run_generation)
    from Generate.core.release_adapter import (
        LINKS_FILENAME, PROMPTS_FILENAME, ReleaseUploader, make_tag)
    from Generate.core.vast_client import (
        OfferFilter, VastClient, VastError, VastRuntimeError)
except ImportError:  # pragma: no cover
    from bad_hosts import BadHostsError, load_bad_hosts  # type: ignore
    from config_loader import (  # type: ignore
        DEFAULT_CONFIG_PATH, Config, load_config, load_profile_number)
    from gemini_prompts import GeminiInputError  # type: ignore
    from log_setup import mask_text, setup_logging, teardown_logging  # type: ignore
    from orchestrator import (  # type: ignore
        ReleaseStartError, RunPublisher, run_generation)
    from release_adapter import (  # type: ignore
        LINKS_FILENAME, PROMPTS_FILENAME, ReleaseUploader, make_tag)
    from vast_client import (  # type: ignore
        OfferFilter, VastClient, VastError, VastRuntimeError)

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_RUNTIME = 3

log = logging.getLogger("Generate.cli")

# уровень логов: единственная переменная окружения, которую читает этот модуль (SPEC 10)
LOG_LEVEL_ENV = "GEN_LOG_LEVEL"
LOG_LEVELS = ("INFO", "DEBUG")

# имена входных файлов по умолчанию (раздел 0, п.7 SPEC: в корне репозитория; это не настройка)
DEFAULT_SRT = "final.srt"
DEFAULT_MISSING = "missing.txt"
DEFAULT_PROMPTS = "prompts.json"
STYLE_DIR = "Generate/model_profiles/prompt_styles"
# запасной стиль, если файла стиля для профиля нет (прежнее поведение)
FALLBACK_STYLE_PROFILE = "ltx25"
# значения по умолчанию, которых нет в config.default.yaml (описаны в профиле модели и SPEC 5)
TIMING_FRAME_RULE = {"kind": "8k+1"}
# наблюдатель heartbeat (SPEC 8.3 а): пороги по умолчанию и формат времени в файле heartbeat
WATCH_DEFAULT_SILENCE_SEC = 120
WATCH_DEFAULT_CHECK_SEC = 30
HEARTBEAT_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
# метка «запуск завершён»: файл рядом с heartbeat (SPEC 8.3 а)
FINISHED_SUFFIX = ".finished"


def _err(text: str) -> None:
    # текст проходит маску до печати: в сообщении об ошибке может оказаться значение секрета (SPEC 0.1; этап 9а).
    # Код выхода не меняется; текст не обрезается.
    print(mask_text(text), file=sys.stderr)


def _usd_floor(value: float) -> str:
    """Кредит в USD до цента с округлением вниз: 4.996 не покажется как «5.00» при минимуме 5."""
    return f"{math.floor(value * 100) / 100:.2f}"


def _parse_verified(value: str) -> bool:
    """Вход --verified: только «1» (только проверенные хосты) или «0» (без фильтра по проверке).

    Любое другое значение, включая true, false и пустую строку, — ошибка аргументов (код 2),
    это происходит до первого вызова Vast. Введённое значение в сообщении не печатаем.
    Отсутствие флага (None) означает «взять из конфига» и сюда не попадает.
    """
    if value == "1":
        return True
    if value == "0":
        return False
    raise argparse.ArgumentTypeError(
        "допустимы только значения 1 и 0: 1 = только проверенные хосты, 0 = без фильтра")


def _parse_rental_type(value: str) -> Optional[str]:
    """Вход --rental-type: interruptible или on-demand. Пустая строка = не задано (конфиг не переопределяется).

    Любое другое значение — ошибка аргументов (код 2), до первого вызова Vast. Значение в сообщении не печатаем.
    """
    if value == "":
        return None
    if value in ("interruptible", "on-demand"):
        return value
    raise argparse.ArgumentTypeError("допустимы значения: interruptible, on-demand")


# Сигналы остановки (SPEC 8.3 (в)): отмена запуска или SIGTERM переводят конвейер в режим уборки.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def _install_stop_handlers(stop: threading.Event) -> dict:
    """Ставит обработчики SIGINT и SIGTERM: они только выставляют флаг. Возвращает прежние обработчики.

    Логирование внутри обработчика не делается: оркестратор пишет строку, увидев флаг в цикле.
    Вне главного потока signal не ставится; тогда возвращается пустой словарь.
    """
    previous: dict = {}

    def _on_signal(signum, frame):  # noqa: ARG001 - сигнатура обязательна для signal
        stop.set()

    for sig in STOP_SIGNALS:
        try:
            previous[sig] = signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass
    return previous


def _restore_handlers(previous: dict) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def _balance_preflight(vast, min_balance_usd: float) -> Optional[int]:
    """Предстартовая проверка расходуемого кредита Vast.ai (SPEC 8.3, е).

    Кредит берётся через vast.get_balance_usd(), которое читает поле credit (не balance).
    Вызывается до поиска предложений и до любого вызова Gemini. Возвращает None, если
    запуск можно продолжать, иначе код завершения (сообщение уже напечатано):
    2 — кредита не хватает, 3 — кредит не удалось проверить. В обоих случаях ничего не потрачено.
    """
    try:
        credit = vast.get_balance_usd()
    except VastRuntimeError as e:
        # текст ошибки клиента безопасен: ключ в нём уже замаскирован (SPEC 0.1)
        _err(f"Не удалось проверить кредит Vast.ai: {e}. "
             f"Запуск не начат, карты не арендованы.")
        return EXIT_RUNTIME
    if credit < min_balance_usd:
        _err(f"Недостаточно средств на Vast.ai: кредит {_usd_floor(credit)} USD, "
             f"нужно не менее {min_balance_usd:.2f} USD. Пополните баланс и запустите снова.")
        return EXIT_INPUT
    log.info("Кредит Vast.ai: %s USD, минимум %.2f USD.", _usd_floor(credit), min_balance_usd)
    return None


def _vast_preflight(vast, min_balance_usd: float) -> Optional[int]:
    """Предстартовая проверка Vast API (SPEC 8.3, е): только чтение, до любого поиска и аренды.

    Сначала эндпоинт списка экземпляров: без него уборка (kill-cards) не сможет найти карты,
    и ошибка выплыла бы уже после аренды. Затем кредит. Возвращает None (можно продолжать)
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


def _resolve_docker_image(docker_image: str) -> str:
    """Имя образа карты (SPEC 8.4, вариант A): значение возвращается как есть.

    Проверяется только, что строка непустая и без пробельных символов. ValueError несёт текст
    для пользователя.
    """
    if not docker_image:
        raise ValueError("образ Docker пуст: укажите имя образа без пробелов.")
    if any(ch.isspace() for ch in docker_image):
        raise ValueError("имя образа Docker содержит пробельные символы.")
    return docker_image


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

    avg = float(g("avg_render_sec", 0.0))
    print("Итоги запуска:")
    print(f"  нужно сделать клипов: {g('total_needed')}")
    print(f"  готово: {g('completed')}")
    print(f"  с ошибкой: {g('failed')}")
    print(f"  не обработано: {g('pending')}")
    print(f"  уже было в Release: {g('skipped_done')}")
    print(f"  арендовано карт: {g('cards_rented')}")
    print(f"  уничтожено карт: {g('destroyed_cards')}")
    print(f"  среднее время рендера клипа, с: {avg:.1f}" if avg > 0
          else "  среднее время рендера клипа, с: нет данных")
    print(f"  потрачено, USD: {float(g('spent_usd', 0.0)):.2f}")
    if g("deadline_reached", False):
        print("  достигнут срок задания: новые клипы не брались")
    if g("budget_exceeded", False):
        print("  достигнут потолок расходов")
    if not g("cleanup_ok", True):
        print("  ВНИМАНИЕ: карты могли остаться арендованными, проверьте Vast.ai")


def _pilot_mode(args: argparse.Namespace) -> bool:
    """Флаг пилота ComfyUI; у старых вызовов без атрибута (тесты, прежние аргументы) — выключен."""
    return bool(getattr(args, "pilot", False))


def cmd_run(args: argparse.Namespace) -> int:
    """Основной запуск генерации (этапы 4-9)."""
    # None = вход не задан, образ берётся из конфига (стоковый образ, SPEC 8.4, вариант A)
    docker_image = args.docker_image
    if _pilot_mode(args) and not (docker_image or "").strip():
        _err("Ошибка входных данных: пилот ComfyUI требует явного образа (--docker-image).")
        return EXIT_INPUT
    if docker_image is not None and not _pilot_mode(args):
        try:
            docker_image = _resolve_docker_image(docker_image)
        except ValueError as e:
            _err(f"Ошибка входных данных: {e}")
            return EXIT_INPUT
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
        "max_inet_price_per_tb": args.max_inet_price_per_tb,
        "docker_image": docker_image,
        "disk_gb": args.disk_gb,
        "silent_host_timeout_min": args.silent_host_timeout_min,
        # схема конфига принимает только целые 0 и 1 (не bool): флаг переводится в 1 или 0
        "verified": None if args.verified is None else int(args.verified),
        # режим аренды: None (не задан) не переопределяет config.default.yaml
        "rental_type": args.rental_type,
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
    # постоянный список плохих машин (SPEC 3, 8.2): читается один раз, до любых трат
    try:
        excluded_machines = load_bad_hosts()
    except BadHostsError as e:
        _err(f"Ошибка входных данных: {e}")
        return EXIT_INPUT

    try:
        timing = _build_timing(cfg, cfg.model_profile)
        prompt_cfg = _build_prompt_config(cfg.model_profile, cfg.style_brief, cfg.gemini_batch_size)
        # пилот: ни Gemini, ни Release не нужны (клиенты не создаются)
        transport = None if _pilot_mode(args) else _make_transport()
        store = None if _pilot_mode(args) else _make_release_store(args.repo)
        vast = VastClient(debug_logs=bool(getattr(args, "debug_logs", False)))
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
        max_inet_price_per_tb=cfg.max_inet_price_per_tb,
        interruptible=cfg.interruptible,
        verified=cfg.verified,
    )

    stop = threading.Event()
    previous_handlers = _install_stop_handlers(stop)
    try:
        # предстартовая проверка Vast API и кредита: до поиска предложений и любого вызова Gemini
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
            # номер сборки воркера (SPEC 8.4): проверяется оркестратором до трат; пилот его не использует
            worker_build=args.worker_build,
            profile=cfg.model_profile,
            release_tag=args.release_tag,
            offer_filter=offer_filter,
            excluded_machines=excluded_machines,
            docker_image=cfg.docker_image,
            disk_gb=cfg.disk_gb,
            silent_host_timeout_min=cfg.silent_host_timeout_min,
            job_deadline_min=cfg.job_deadline_min,
            worker_idle_timeout_min=cfg.worker_idle_timeout_min,
            # мониторинг весов и готовности (SPEC 8.2): пороги из конфига, без литералов в оркестраторе
            speed_check_at_pct=cfg.speed_check_at_pct,
            speed_min_mbps=cfg.speed_min_mbps,
            progress_step_pct=cfg.progress_step_pct,
            stall_timeout_min=cfg.stall_timeout_min,
            stage_max_min=cfg.stage_max_min,
            worker_ready_timeout_min=cfg.worker_ready_timeout_min,
            task_accept_timeout_min=cfg.task_accept_timeout_min,
            heartbeat_interval_sec=cfg.heartbeat_interval_sec,
            monitor_interval_sec=cfg.monitor_interval_sec,
            min_inet_mbps=cfg.min_inet_mbps,
            # файл heartbeat для наблюдателя (этап 7, SPEC 8.3 а); по умолчанию не пишется
            heartbeat_path=getattr(args, "heartbeat_path", None),
            filename_template=cfg.filename_template,
            clip_name_width=cfg.clip_name_width,
            release_asset_limit=cfg.release_asset_limit,
            gemini_options=_gemini_options(cfg),
            log_level=getattr(args, "log_level", "INFO"),
            stop_event=stop,
            pilot=_pilot_mode(args),
            # пустая команда не передаётся дальше (None): onstart в payload не попадает
            pilot_start_cmd=(getattr(args, "pilot_start_cmd", "") or "").strip() or None,
            # артефакт запуска (SPEC 10.4): каталог задаёт workflow; пусто = не ведётся
            artifact_dir=(getattr(args, "artifact_dir", "") or None),
        )
    except ReleaseStartError as e:
        # служебный релиз или файл запуска не загружены: остановка до трат, код 2 (SPEC 9.1, E1)
        _err(f"Ошибка выдачи в релиз: {e}")
        return EXIT_INPUT
    except GeminiInputError as e:
        # ошибка входа из оркестратора (run_id, номер сборки): код 2, текст на русском, карт не арендовано
        _err(f"Ошибка входных данных: {e}")
        return EXIT_INPUT
    finally:
        _restore_handlers(previous_handlers)
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
        # отладочный журнал Vast здесь не включается (по умолчанию выключен; флага у kill-cards нет)
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
    if destroyed:
        # токен воркера создаётся в процессе run и здесь неизвестен: логи карт не запрашиваются
        print("Логи карт в этой команде не выгружаются: токен воркера есть только у процесса run.")
    return EXIT_OK


def finished_marker_path(heartbeat_path) -> Path:
    """Метка «запуск завершён»: файл рядом с файлом heartbeat (SPEC 8.3 а)."""
    hb = Path(heartbeat_path)
    return hb.with_name(hb.name + FINISHED_SUFFIX)


def _mark_run_finished(heartbeat_path) -> None:
    """Пишет метку конца запуска при любом исходе run. Ошибка записи не фатальна:
    тогда наблюдатель выйдет по тишине файла heartbeat."""
    try:
        marker = finished_marker_path(heartbeat_path)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("finished\n", encoding="utf-8")
    except OSError:
        _err("Не удалось записать метку конца запуска для наблюдателя heartbeat.")


def _read_heartbeat_utc(path: Path) -> Optional[datetime]:
    """Время записи heartbeat (UTC) или None: файла нет, он не читается или поле utc не разобрано.

    Содержимое файла не печатается: в нём адреса и номера карт.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return datetime.strptime(str(data["utc"]), HEARTBEAT_UTC_FORMAT).replace(tzinfo=timezone.utc)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _watch_kill(run_id: str) -> int:
    """Уборка для наблюдателя: та же логика, что у kill-cards, по метке gen-<run_id>."""
    code = cmd_kill_cards(argparse.Namespace(label=None, run_id=run_id))
    if code != EXIT_OK:
        log.error("Наблюдатель: уничтожение карт не удалось (код %d). Проверьте Vast.ai и шаг уборки.", code)
    return EXIT_OK


def cmd_watch_heartbeat(args: argparse.Namespace) -> int:
    """Наблюдатель heartbeat (SPEC 8.3 а): фоновый процесс шага run в generate.yml.

    Каждые check_every_sec читает файл heartbeat. Если метки конца запуска нет и запись старше
    max_silence_sec, либо файл пропал после того, как появлялся, уничтожает карты запуска
    (как kill-cards) и выходит с кодом 0. Если метка конца запуска появилась, выходит с кодом 0
    без уборки. Пока файла нет, ждёт: оркестратор пишет его в цикле аренды, а карты до этого
    не арендуются. Ключ Vast и адреса карт не печатаются.
    """
    if not args.run_id:
        _err("Ошибка входных данных: наблюдателю нужен номер запуска (--run-id).")
        return EXIT_INPUT
    if not args.heartbeat_path:
        _err("Ошибка входных данных: наблюдателю нужен путь файла heartbeat (--heartbeat-path).")
        return EXIT_INPUT
    if args.max_silence_sec <= 0 or args.check_every_sec <= 0:
        _err("Ошибка входных данных: пороги наблюдателя должны быть больше нуля.")
        return EXIT_INPUT

    heartbeat = Path(args.heartbeat_path)
    marker = finished_marker_path(heartbeat)
    log.info("Наблюдатель heartbeat запущен: запуск %s, порог тишины %d с, проверка каждые %d с.",
             args.run_id, args.max_silence_sec, args.check_every_sec)
    seen = False
    while True:
        if marker.exists():
            log.info("Метка конца запуска найдена: наблюдатель завершается, карты не трогаются.")
            return EXIT_OK
        beat = _read_heartbeat_utc(heartbeat)
        if beat is not None:
            seen = True
            silence = (datetime.now(timezone.utc) - beat).total_seconds()
            if silence <= args.max_silence_sec:
                log.info("heartbeat свежий: последняя запись %d с назад.", int(silence))
            else:
                log.error("heartbeat не обновлялся %d с (порог %d с): уничтожаем карты запуска.",
                          int(silence), args.max_silence_sec)
                return _watch_kill(args.run_id)
        elif seen:
            log.error("Файл heartbeat пропал или не читается: уничтожаем карты запуска.")
            return _watch_kill(args.run_id)
        else:
            log.info("Файла heartbeat ещё нет: карты запуска не арендовались, наблюдатель ждёт.")
        time.sleep(args.check_every_sec)


def cmd_finalize_release(args: argparse.Namespace) -> int:
    """Доводит выдачу до конца: prompts.json и generated_links.txt по уже загруженным клипам.

    Та же логика выдачи, что у оркестратора (RunPublisher, SPEC 9.1, 9.4, 9.7; F2).
    Безопасна при повторном запуске: если клип-релиз ещё не создан, завершается с кодом 0.
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
                                   filename_template=cfg.filename_template,
                                   run_id=args.run_id or None)
        uploader.prepare()
        clips = len(uploader.done)

        with tempfile.TemporaryDirectory() as tmp:
            pub = RunPublisher(uploader, args.repo, tmp)
            pub.finish(prompts_path=args.prompts_path)
    except Exception as e:  # noqa: BLE001
        # текст берём только у своих ошибок проекта (у них есть exit_code); иначе только тип
        if hasattr(e, "exit_code"):
            _err(f"Не удалось завершить выдачу клипов: {e}")
        else:
            _err(f"Не удалось завершить выдачу клипов ({type(e).__name__}).")
        return EXIT_RUNTIME

    prompts_uploaded = PROMPTS_FILENAME in pub.uploaded
    links_uploaded = LINKS_FILENAME in pub.uploaded
    if not os.path.isfile(args.prompts_path):
        print(f"Файл промптов не найден ({args.prompts_path}): он не загружен.")
    print("Выдача клипов завершена.")
    print(f"  релиз: {tag}")
    print(f"  клипов в релизах: {clips}")
    print(f"  prompts.json загружен: {'да' if prompts_uploaded else 'нет'}")
    print(f"  generated_links.txt загружен: {'да' if links_uploaded else 'нет'}")
    for name in pub.failures:
        _err(f"Не удалось загрузить {name}.")
    return EXIT_RUNTIME if pub.failures else EXIT_OK


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
    check_p = sub.add_parser("check", parents=[common],
                             help="проверить конфигурацию и входные файлы")
    check_p.set_defaults(func=lambda a: cmd_check(a))

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
                       help="минимальный кредит Vast.ai для старта, USD; по умолчанию из конфига")
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
    run_p.add_argument("--max-inet-price-per-tb", type=float, default=None,
                       help="потолок цены трафика карты, USD за ТБ (загрузка и выгрузка); по умолчанию из конфига")
    run_p.add_argument("--docker-image", default=None,
                       help="образ Docker для карты Vast.ai; не задано = из конфига (стоковый образ)")
    run_p.add_argument("--worker-build", default="",
                       help="номер коммита сборки воркера (SHA) для оркестратора и onstart")
    run_p.add_argument("--artifact-dir", default="",
                       help="каталог артефакта запуска (SPEC 10.4); пусто = не вести")
    run_p.add_argument("--heartbeat-path", default=None,
                       help="путь файла heartbeat (JSON, для наблюдателя); по умолчанию не пишется")
    run_p.add_argument("--disk-gb", type=int, default=None,
                       help="размер диска для карты Vast.ai, ГБ; по умолчанию из конфига")
    run_p.add_argument("--silent-host-timeout-min", type=int, default=None,
                       help="таймаут первого ответа хоста, минут; по умолчанию из конфига")
    run_p.add_argument("--rental-type", type=_parse_rental_type, default=None,
                       help="режим аренды карт: interruptible или on-demand (пусто = из конфига)")
    run_p.add_argument("--verified", type=_parse_verified, default=None,
                       help="только проверенные хосты Vast: 1 = да, 0 = нет; "
                            "другие значения недопустимы; без флага берётся из конфига")
    run_p.add_argument("--pilot", action="store_true", default=False,
                       help="пилот ComfyUI: одна карта, без воркера, весов и Release; метка pilot-<run-id>")
    run_p.add_argument("--pilot-start-cmd", default="",
                       help="команда старта ComfyUI в пилоте (UNVERIFIED); пусто = не передавать, образ стартует сам")
    run_p.add_argument("--debug-logs", action="store_true", default=False,
                       help="сырой журнал обмена с Vast: запросы и ответы в лог, секреты маскируются")
    run_p.set_defaults(func=lambda a: cmd_run(a))

    kill_p = sub.add_parser("kill-cards", help="уничтожить арендованные карты по метке")
    kill_p.add_argument("--label", default=None, help="метка карт (по умолчанию gen-<run-id>)")
    kill_p.add_argument("--run-id", default="",
                        help="номер запуска (передаётся явно)")
    kill_p.set_defaults(func=lambda a: cmd_kill_cards(a))

    watch_p = sub.add_parser("watch-heartbeat",
                             help="наблюдатель heartbeat: уничтожает карты запуска при тишине")
    watch_p.add_argument("--heartbeat-path", default="",
                         help="путь файла heartbeat (тот же, что у run)")
    watch_p.add_argument("--run-id", default="", help="номер запуска: метка карт gen-<номер>")
    watch_p.add_argument("--max-silence-sec", type=int, default=WATCH_DEFAULT_SILENCE_SEC,
                         help=f"порог тишины, с; по умолчанию {WATCH_DEFAULT_SILENCE_SEC}")
    watch_p.add_argument("--check-every-sec", type=int, default=WATCH_DEFAULT_CHECK_SEC,
                         help=f"период проверки, с; по умолчанию {WATCH_DEFAULT_CHECK_SEC}")
    watch_p.set_defaults(func=lambda a: cmd_watch_heartbeat(a))
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
    fin_p.add_argument("--run-id", default="",
                       help="номер запуска: нужен для служебного и итогового релизов (необязательно)")
    fin_p.set_defaults(func=lambda a: cmd_finalize_release(a))
    return parser


def _read_log_level() -> Optional[str]:
    """INFO или DEBUG; пусто = INFO; иное = None (значение в сообщении не печатаем, SPEC 0.1)."""
    value = os.environ.get(LOG_LEVEL_ENV, "").strip()
    if not value:
        return "INFO"
    return value if value in LOG_LEVELS else None


def main(argv: list[str] | None = None) -> int:
    """Возвращает код завершения. Трейсбеки пользователю не показываются.

    Обработчики журнала (SPEC 10): консоль на INFO всегда; файл отладочного журнала в каталоге
    артефакта (--artifact-dir) на уровне GEN_LOG_LEVEL. Настройка и снятие: core/log_setup.py.
    """
    try:
        parser = build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit as e:  # argparse: 2 при ошибке аргументов, 0 для --help
            code = e.code
            return code if isinstance(code, int) else EXIT_INPUT
        level = _read_log_level()
        if level is None:
            _err(f"Ошибка входных данных: {LOG_LEVEL_ENV} допускает только значения INFO и DEBUG.")
            return EXIT_INPUT
        # HTTP-библиотека на DEBUG выводит детали соединений: оставляем её на WARNING
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        args.log_level = level
        handlers, note = setup_logging(level, getattr(args, "artifact_dir", "") or None)
        try:
            if note:
                _err(note)
            try:
                return int(args.func(args))
            finally:
                # метка конца запуска для наблюдателя heartbeat (SPEC 8.3 а): при любом исходе run
                heartbeat = getattr(args, "heartbeat_path", None)
                if args.command == "run" and heartbeat:
                    _mark_run_finished(heartbeat)
        finally:
            teardown_logging(handlers)
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
