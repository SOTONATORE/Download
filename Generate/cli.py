"""Точка входа второй части: ``python -m Generate.cli <команда> [параметры]``.

Команды:
    check       проверка конфигурации и входных файлов (SRT разбирается, missing.txt существует);
    run         основной запуск генерации (оркестратор, Gemini, Vast.ai, Release);
    kill-cards  уничтожение арендованных карт по метке (шаг ``if: always()``);
    watch-heartbeat  наблюдатель heartbeat: фоновый процесс шага run (SPEC 8.3 а);
    cleanup-vast  уборка забытых карт по метке и статусу запуска (SPEC 8.3 г; cleanup_vast.yml);
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
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from Generate.core.srt_parser import SrtError, parse_srt, srt_hash

try:  # пакетный и «плоский» импорт
    from Generate.core.bad_hosts import BadHostsError, load_bad_hosts
    from Generate.core.config_loader import (
        DEFAULT_CONFIG_PATH, Config, load_config, load_profile_number)
    from Generate.core.gemini_prompts import GeminiInputError, GeminiRuntimeError
    from Generate.core.log_setup import mask_text, setup_logging, teardown_logging
    from Generate.core.orchestrator import (
        ReleaseStartError, RunPublisher, run_generation)
    from Generate.core.release_adapter import (
        LINKS_FILENAME, PROMPTS_FILENAME, GhCommandRunner, ReleaseUploader, make_tag)
    from Generate.core.vast_client import (
        OfferFilter, VastClient, VastError, VastInstanceNotFound, VastRuntimeError)
except ImportError:  # pragma: no cover
    from bad_hosts import BadHostsError, load_bad_hosts  # type: ignore
    from config_loader import (  # type: ignore
        DEFAULT_CONFIG_PATH, Config, load_config, load_profile_number)
    from gemini_prompts import GeminiInputError, GeminiRuntimeError  # type: ignore
    from log_setup import mask_text, setup_logging, teardown_logging  # type: ignore
    from orchestrator import (  # type: ignore
        ReleaseStartError, RunPublisher, run_generation)
    from release_adapter import (  # type: ignore
        LINKS_FILENAME, PROMPTS_FILENAME, GhCommandRunner, ReleaseUploader, make_tag)
    from vast_client import (  # type: ignore
        OfferFilter, VastClient, VastError, VastInstanceNotFound, VastRuntimeError)

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
# наблюдатель: сколько ждать первый файл heartbeat (решение владельца 18.10.2026); до первой записи карты не арендуются
WATCH_FIRST_HEARTBEAT_LIMIT_SEC = 600
# уборка по расписанию (SPEC 8.3 г): аварийный предел возраста карты и метки карт
MAX_AGE_SEC = 6 * 3600
GEN_LABEL_RE = re.compile(r"gen-([0-9]+)")
PILOT_LABEL_RE = re.compile(r"pilot-(.+)")
CARD_LABEL_PREFIXES = ("gen-", "pilot-")
# статусы запуска GitHub Actions, при которых запуск ещё идёт (SPEC 8.3 г)
ACTIVE_RUN_STATUSES = ("in_progress", "queued", "waiting", "requested", "pending")
GH_STATUS_TIMEOUT_SEC = 60


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


def _parse_scan_sec(value: str) -> Optional[int]:
    """Вход --scan-sec: целое число секунд больше 0. Пустая строка = не задано (берётся из конфига).

    Нечисловое или неположительное значение — ошибка аргументов (код 2), до первого вызова Vast.
    Введённое значение в сообщении не печатаем. Правило «не меньше 5 × scan_step_sec»
    проверяет config_loader уже на итоговом значении (после накладывания входов).
    """
    text = (value or "").strip()
    if text == "":
        return None
    if not re.fullmatch(r"[0-9]+", text) or int(text) < 1:
        raise argparse.ArgumentTypeError("допустимо только целое число секунд больше 0")
    return int(text)


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
    # --scan-sec проверяется здесь же, до любого вызова Vast (предстартовая проверка workflow)
    _cfg, code = _load_cfg(args.config, {"scan_sec": getattr(args, "scan_sec", None)})
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


# onstart пилота: карта скачивает pilot_start.sh из коммита, который тестируется, и запускает его с bash
PILOT_START_PATH = "Generate/worker/pilot_start.sh"
PILOT_ONSTART_LOG = "/opt/pilot/onstart.log"
_PILOT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_PILOT_REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def pilot_onstart_cmd(sha: str, repo: str) -> str:
    """Текст onstart пилота для --pilot-start-cmd (workflow подставляет GITHUB_SHA и REPO).

    Скрипт скачивается по raw-адресу коммита и запускается в фоне (setsid nohup bash). Вывод скачивания и скрипта
    дописывается в PILOT_ONSTART_LOG на карте. COMFYUI_ARGS в команду не попадает: он приходит в окружении карты.
    Токенов и секретов в тексте нет. Приватный репозиторий не поддерживается: токен здесь не передаётся.
    Некорректные sha или repo вызывают ValueError до аренды.
    """
    if not isinstance(sha, str) or _PILOT_SHA_RE.fullmatch(sha) is None:
        raise ValueError("коммит пилота (SHA): нужны 40 символов 0-9 и a-f")
    if not isinstance(repo, str) or _PILOT_REPO_RE.fullmatch(repo) is None:
        raise ValueError("репозиторий пилота: нужен вид владелец/имя")
    url = f"https://raw.githubusercontent.com/{repo}/{sha}/{PILOT_START_PATH}"
    return (
        f"mkdir -p /opt/pilot && curl -fsSL --max-time 60 {url} -o /opt/pilot/pilot_start.sh"
        f" >> {PILOT_ONSTART_LOG} 2>&1"
        f" && setsid nohup bash /opt/pilot/pilot_start.sh >> {PILOT_ONSTART_LOG} 2>&1 </dev/null &"
    )


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
        # окно сбора цен, с: None (не задано) не переопределяет config.default.yaml
        "scan_sec": getattr(args, "scan_sec", None),
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
        # диск предложения не меньше запрошенного (disk_gb входа); пилот этим фильтром не ограничивается
        min_disk_gb=0.0 if _pilot_mode(args) else float(cfg.disk_gb),
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
            # надбавка к ставке bid из конфига (bid_margin, как MARGIN в vast_stable_run.sh)
            bid_margin=cfg.bid_margin,
            excluded_machines=excluded_machines,
            docker_image=cfg.docker_image,
            disk_gb=cfg.disk_gb,
            silent_host_timeout_min=cfg.silent_host_timeout_min,
            job_deadline_min=cfg.job_deadline_min,
            worker_idle_timeout_min=cfg.worker_idle_timeout_min,
            # мониторинг весов и готовности (SPEC 8.2): пороги из конфига, без литералов в оркестраторе
            speed_check_at_pct=cfg.speed_check_at_pct,
            progress_step_pct=cfg.progress_step_pct,
            stall_timeout_min=cfg.stall_timeout_min,
            stage_max_min=cfg.stage_max_min,
            worker_ready_timeout_min=cfg.worker_ready_timeout_min,
            task_accept_timeout_min=cfg.task_accept_timeout_min,
            heartbeat_interval_sec=cfg.heartbeat_interval_sec,
            monitor_interval_sec=cfg.monitor_interval_sec,
            min_inet_mbps=cfg.min_inet_mbps,
            # логи карты через Vast при снятии без логов воркера (SPEC 8.4): из конфига
            vast_logs_attempts=cfg.vast_logs_attempts,
            vast_logs_pause_sec=cfg.vast_logs_pause_sec,
            # окно сбора цен перед первой арендой (SPEC 8.2): значения из конфига
            scan_sec=cfg.scan_sec,
            scan_step_sec=cfg.scan_step_sec,
            # лимит запросов аренды за запуск (9е-3, max_rent_tries из конфига)
            max_rent_tries=cfg.max_rent_tries,
            # фильтр стабильности (этап 9е-2, как PARSE_RANK в vast_stable_run.sh): порог и разброс из конфига
            stable_min_price=cfg.min_price_per_hour,
            stable_max_spread=cfg.stable_max_spread,
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


def read_card_ids(card_dir) -> list[int]:
    """ID экземпляров из meta.json карт артефакта запуска (cards/*/meta.json, пишет оркестратор при аренде).

    Читаются только номера экземпляров: адреса, ключи и токены из этих файлов не берутся.
    Битый файл или поле без целого положительного номера пропускается.
    """
    ids: list[int] = []
    for meta in sorted(Path(card_dir).glob("cards/*/meta.json")):
        try:
            value = json.loads(meta.read_text(encoding="utf-8")).get("instance_id")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        if value not in ids:
            ids.append(value)
    return ids


def cmd_kill_cards(args: argparse.Namespace) -> int:
    """Уничтожает все карты запуска по метке (SPEC 8.3 (в)).

    Основной путь: уничтожение по метке gen-<run_id>. Резерв (при --card-dir): карты из meta.json
    артефакта запуска, которые метка не нашла; их ID записал оркестратор сразу после аренды.
    Код 3, если хотя бы одно уничтожение не прошло (по метке или по ID).
    """
    run_id = str(getattr(args, "run_id", "") or "").strip()
    label = args.label or (f"gen-{run_id}" if run_id else "")
    if not label:
        _err("Ошибка входных данных: не задана метка (--label) и номер запуска (--run-id).")
        return EXIT_INPUT
    card_dir = str(getattr(args, "card_dir", "") or "")

    errors: list[str] = []
    by_label: list[int] = []
    by_id: list[int] = []
    already_gone: list[int] = []
    failed_ids: list[int] = []
    card_ids: list[int] = []
    client = None
    try:
        # отладочный журнал Vast здесь не включается (по умолчанию выключен; флага у kill-cards нет)
        client = VastClient()
        try:
            by_label = list(client.destroy_by_label(label))
        except VastError as e:
            # частичный результат (VastDestroyError) сохраняем: эти карты уже уничтожены
            by_label = list(getattr(e, "destroyed_ids", []) or [])
            errors.append(str(e))

        if card_dir:
            card_ids = read_card_ids(card_dir)
            for inst_id in (i for i in card_ids if i not in by_label):
                try:
                    client.get_instance(inst_id)
                except VastInstanceNotFound:
                    already_gone.append(inst_id)
                    continue
                except VastError:
                    failed_ids.append(inst_id)
                    continue
                try:
                    client.destroy_instance(inst_id)
                    by_id.append(inst_id)
                except VastError as e:
                    failed_ids.append(inst_id)
                    errors.append(str(e))

        if not by_label and not by_id:
            # метка ничего не нашла: печатаем, сколько карт вообще видно аккаунту (без адресов и ключей)
            try:
                everything = client.list_instances()
                tagged = sum(1 for i in everything if (i.label or "").startswith(CARD_LABEL_PREFIXES))
                print(f"Всего экземпляров в аккаунте: {len(everything)}; с метками gen-/pilot-: {tagged}.")
            except Exception:  # диагностика не должна менять исход уборки
                pass
    except VastError as e:
        _err(f"Не удалось уничтожить карты: {e}")
        return EXIT_RUNTIME
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    total = len(by_label) + len(by_id)
    print(f"Уничтожено карт: {total} (метка {label}).")
    if card_dir:
        print(f"Резерв по ID из артефакта: записано карт {len(card_ids)}, уничтожено по ID {len(by_id)}, "
              f"уже не было {len(already_gone)}, не удалось {len(failed_ids)}.")
    if total:
        # токен воркера создаётся в процессе run и здесь неизвестен: логи карт не запрашиваются
        print("Логи карт в этой команде не выгружаются: токен воркера есть только у процесса run.")
    for text in errors:
        _err(f"Не удалось уничтожить карты: {text}")
    if errors or failed_ids:
        return EXIT_RUNTIME
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
    """Уборка для наблюдателя: та же логика, что у kill-cards, по метке gen-<run_id>.

    Возвращает EXIT_OK, если уничтожение прошло, иначе EXIT_RUNTIME (решение владельца 18.10.2026).
    """
    code = cmd_kill_cards(argparse.Namespace(label=None, run_id=run_id))
    if code != EXIT_OK:
        log.error("Наблюдатель: уничтожение карт запуска %s не удалось (код %d). "
                  "Проверьте Vast.ai и шаг уборки.", run_id, code)
        return EXIT_RUNTIME
    return EXIT_OK


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def watch_heartbeat(run_id: str, heartbeat: Path, *, max_silence_sec: int, check_every_sec: int,
                    first_limit_sec: int = WATCH_FIRST_HEARTBEAT_LIMIT_SEC,
                    kill=None, clock=time.monotonic, sleep=time.sleep,
                    utc_now=_utc_now) -> int:
    """Цикл наблюдателя (SPEC 8.3 а). Часы, сон и уборка передаются параметрами (для тестов).

    Порядок в каждом обходе:
      - есть метка конца запуска: выход с кодом 0, карты не трогаются;
      - файл heartbeat читается и свежий: ждём дальше;
      - файл heartbeat старше max_silence_sec или пропал после первой записи: уборка gen-<run_id>
        (одна на эпизод тишины, без повтора в следующих обходах); после уборки наблюдатель продолжает
        работать. Повторная уборка возможна только после свежего heartbeat и новой тишины (F3);
      - первого файла ещё нет: ждём не дольше first_limit_sec; по истечении выход с кодом 3,
        уборка не запускается (карт до первой записи нет).
    Выход по метке конца запуска: код 3, если хоть одна уборка не удалась (G11), иначе 0. Выход по первому-лимиту —
    код 3. Цикл не завершается после уборки; остановка процесса — шаг «Stop heartbeat observer».
    Ключ Vast, адреса и содержимое файла не печатаются.
    """
    run_kill = kill if kill is not None else _watch_kill
    marker = finished_marker_path(heartbeat)
    started = clock()
    seen = False
    # F3: уборка уже сделана на текущем эпизоде тишины. Сбрасывается только свежим heartbeat.
    stalled = False
    # G11 сохранён: если хоть одна уборка не удалась, наблюдатель при выходе по метке даёт код 3
    kill_failed = False
    while True:
        if marker.exists():
            log.info("Метка конца запуска найдена: наблюдатель завершается, карты не трогаются.")
            return EXIT_RUNTIME if kill_failed else EXIT_OK
        beat = _read_heartbeat_utc(heartbeat)
        if beat is not None:
            seen = True
            silence = (utc_now() - beat).total_seconds()
            if silence <= max_silence_sec:
                if stalled:
                    stalled = False
                    log.info("heartbeat снова свежий: новая уборка возможна только после новой тишины.")
                log.info("heartbeat свежий: последняя запись %d с назад.", int(silence))
            elif not stalled:
                log.error("heartbeat не обновлялся %d с (порог %d с): уничтожаем карты запуска.",
                          int(silence), max_silence_sec)
                kill_failed = run_kill(run_id) != EXIT_OK or kill_failed
                stalled = True
        elif seen:
            if not stalled:
                log.error("Файл heartbeat пропал или не читается: уничтожаем карты запуска.")
                kill_failed = run_kill(run_id) != EXIT_OK or kill_failed
                stalled = True
        else:
            waited = clock() - started
            if waited >= first_limit_sec:
                log.error("Файл heartbeat не появился за %d с: карты запуска не арендовались, "
                          "наблюдатель завершается с кодом 3.", first_limit_sec)
                return EXIT_RUNTIME
            log.info("Файла heartbeat ещё нет (ждём не дольше %d с): карты запуска не арендовались.",
                     first_limit_sec)
        sleep(check_every_sec)


def cmd_watch_heartbeat(args: argparse.Namespace) -> int:
    """Наблюдатель heartbeat (SPEC 8.3 а): фоновый процесс шага run в generate.yml.

    Коды выхода: 0 — метка конца запуска, уборок не было или все прошли; 3 — метка конца запуска, а уборка
    не удалась, либо первый файл heartbeat не появился за WATCH_FIRST_HEARTBEAT_LIMIT_SEC; 2 — неверные аргументы.
    Уборка по тишине сама наблюдателя не завершает (F3).
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

    log.info("Наблюдатель heartbeat запущен: запуск %s, порог тишины %d с, проверка каждые %d с.",
             args.run_id, args.max_silence_sec, args.check_every_sec)
    return watch_heartbeat(args.run_id, Path(args.heartbeat_path),
                           max_silence_sec=args.max_silence_sec,
                           check_every_sec=args.check_every_sec)


# ---------------------------------------------------------------- уборка по расписанию (SPEC 8.3 г)

def parse_card_label(label: Optional[str]) -> Optional[str]:
    """Номер запуска из метки карты: gen-<номер> или pilot-<что угодно>. None, если метка не разобрана."""
    text = label or ""
    m = GEN_LABEL_RE.fullmatch(text) or PILOT_LABEL_RE.fullmatch(text)
    return m.group(1) if m else None


def classify_run_status(status: Optional[str]) -> str:
    """'finished' (completed или missing), 'active' (идёт или в очереди) или 'unknown' (статус не прочитан).

    missing (запуска нет, HTTP 404) считается завершённым: карта уничтожается независимо от возраста.
    """
    if status in ("completed", "missing"):
        return "finished"
    if status in ACTIVE_RUN_STATUSES:
        return "active"
    return "unknown"


def card_age_sec(inst, now: float) -> Optional[float]:
    """Возраст карты, сек, по полю start_date ответа Vast; None, если поля нет или оно не читается."""
    try:
        start = float(inst.raw.get("start_date"))
    except (AttributeError, TypeError, ValueError):
        return None
    return max(0.0, now - start)


def decide_card(label: Optional[str], run_status: Optional[str],
                age_sec: Optional[float]) -> tuple[bool, str]:
    """Решение по карте: (уничтожить, причина). Чистая функция, без обращения к сети (SPEC 8.3 г).

    Порядок: метка не разобрана -> уничтожить; запуск завершён -> уничтожить;
    возраст больше MAX_AGE_SEC -> уничтожить; запуск идёт -> оставить;
    статус не прочитан -> оставить до следующей проверки.
    """
    run_id = parse_card_label(label)
    if run_id is None:
        return True, "метка не разобрана"
    if run_status == "missing":
        return True, f"запуск {run_id} не найден (считается завершённым)"
    state = classify_run_status(run_status)
    if state == "finished":
        return True, f"запуск {run_id} завершён"
    if age_sec is not None and age_sec > MAX_AGE_SEC:
        return True, f"карта старше {MAX_AGE_SEC // 3600} ч (предел по возрасту)"
    if state == "active":
        return False, f"запуск {run_id} ещё выполняется"
    return False, f"статус запуска {run_id} не прочитан"


def gh_run_status(repo: str, run_id: str, runner=None) -> Optional[str]:
    """Статус запуска GitHub Actions через gh.

    Возвращает статус из ответа gh, 'missing' (запуска нет: в ошибке gh есть «HTTP 404» или «not found»)
    или None (статус не прочитан: другая ошибка gh, таймаут, нет поля).

    gh запускается через GhCommandRunner (core/release_adapter.py): окружение только PATH, HOME, GH_TOKEN.
    Текст ошибок gh не печатается; для решения используется только признак 404 / not found.
    """
    call = runner if runner is not None else GhCommandRunner(timeout=GH_STATUS_TIMEOUT_SEC)
    try:
        data = json.loads(call(["run", "view", run_id, "--repo", repo, "--json", "status,conclusion"]))
    except GeminiRuntimeError as exc:
        # Признак отсутствующего запуска ищется в тексте stderr gh (GhCommandRunner кладёт его в сообщение).
        # Зависим только от строк «HTTP 404» и «not found» (без учёта регистра). Если gh напишет для
        # несуществующего запуска что-то другое, результат будет «неизвестно», карта останется до 6 часов:
        # это безопасная сторона, карта не уничтожается по ошибочному признаку.
        text = str(exc).lower()
        if "http 404" in text or "not found" in text:
            return "missing"
        return None
    except (GeminiInputError, ValueError, AttributeError):
        return None
    status = data.get("status") if isinstance(data, dict) else None
    return status if isinstance(status, str) else None


@dataclass
class CleanupResult:
    kept: int = 0
    destroyed: int = 0
    failed: int = 0
    unclassified: int = 0

    @property
    def exit_code(self) -> int:
        """0, если все уничтожения прошли или уничтожать нечего; 3 при неудаче или карте без номера."""
        return EXIT_RUNTIME if (self.failed or self.unclassified) else EXIT_OK

    def summary_lines(self) -> list[str]:
        return [
            "## Уборка карт Vast",
            f"- Оставлено (запуск идёт или статус не прочитан): {self.kept}",
            f"- Уничтожено карт: {self.destroyed}",
            f"- Ошибок уничтожения: {self.failed}",
            f"- Карт без номера, не обработано: {self.unclassified}",
        ]


def cleanup_cards(client, *, run_status_fn: Callable[[str], Optional[str]], now: float,
                  out: Callable[[str], None] = print) -> CleanupResult:
    """Уборка карт с метками gen-* и pilot-* (SPEC 8.3 г). Остальные карты аккаунта не трогаются.

    run_status_fn(run_id) возвращает статус запуска или None. Решение по каждой карте печатается
    с номером карты (не адресом) и причиной.
    """
    result = CleanupResult()
    cards = [i for i in client.list_instances() if (i.label or "").startswith(CARD_LABEL_PREFIXES)]
    if not cards:
        out("Карт с метками gen-* и pilot-* нет — убирать нечего.")
    for inst in cards:
        iid = inst.instance_id
        if not iid:
            result.unclassified += 1
            out("Карта без номера в ответе Vast: уничтожить нельзя, пропущена.")
            continue
        run_id = parse_card_label(inst.label)
        run_status = run_status_fn(run_id) if run_id is not None else None
        destroy, reason = decide_card(inst.label, run_status, card_age_sec(inst, now))
        if not destroy:
            result.kept += 1
            out(f"Карта {iid}: оставлена, {reason}.")
            continue
        out(f"Карта {iid}: уничтожается, {reason}.")
        try:
            client.destroy_instance(iid)
            result.destroyed += 1
        except VastError as exc:
            result.failed += 1
            out(f"Карта {iid}: ошибка уничтожения ({type(exc).__name__}).")
    return result


def cmd_cleanup_vast(args: argparse.Namespace) -> int:
    """Уборка забытых карт (SPEC 8.3 г): вызывается из cleanup_vast.yml по расписанию и вручную."""
    if not args.repo:
        _err("Ошибка входных данных: не задан репозиторий (--repo).")
        return EXIT_INPUT
    client = None
    try:
        client = VastClient()
        result = cleanup_cards(
            client,
            run_status_fn=lambda rid: gh_run_status(args.repo, rid),
            now=time.time(),
        )
    except VastError as e:
        _err(f"Ошибка клиента Vast: {e}")
        return EXIT_INPUT if e.exit_code == EXIT_INPUT else EXIT_RUNTIME
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    lines = result.summary_lines()
    print("\n".join(lines))
    if args.summary_file:
        try:
            with open(args.summary_file, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError:
            _err("Не удалось записать итоги уборки в файл сводки.")
    return result.exit_code



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
    check_p.add_argument("--scan-sec", type=_parse_scan_sec, default=None,
                         help="окно сбора цен, с (целое > 0); пусто = из конфига")
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
    run_p.add_argument("--scan-sec", type=_parse_scan_sec, default=None,
                       help="окно сбора цен перед первой арендой, с (целое > 0); пусто = из конфига")
    run_p.add_argument("--rental-type", type=_parse_rental_type, default=None,
                       help="режим аренды карт: interruptible или on-demand (пусто = из конфига)")
    run_p.add_argument("--verified", type=_parse_verified, default=None,
                       help="только проверенные хосты Vast: 1 = да, 0 = нет; "
                            "другие значения недопустимы; без флага берётся из конфига")
    run_p.add_argument("--pilot", action="store_true", default=False,
                       help="пилот ComfyUI: одна карта, без воркера, весов и Release; метка pilot-<run-id>")
    run_p.add_argument("--pilot-start-cmd", default="",
                       help="команда старта ComfyUI в пилоте (UNVERIFIED); пусто = не передавать; пилот запускает pilot_start.sh из коммита")
    run_p.add_argument("--debug-logs", action="store_true", default=False,
                       help="сырой журнал обмена с Vast: запросы и ответы в лог, секреты маскируются")
    run_p.set_defaults(func=lambda a: cmd_run(a))

    kill_p = sub.add_parser("kill-cards", help="уничтожить арендованные карты по метке")
    kill_p.add_argument("--label", default=None, help="метка карт (по умолчанию gen-<run-id>)")
    kill_p.add_argument("--run-id", default="",
                        help="номер запуска (передаётся явно)")
    kill_p.add_argument("--card-dir", default="",
                        help="каталог артефакта запуска: резерв, ID карт из cards/*/meta.json")
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
    clean_p = sub.add_parser("cleanup-vast",
                             help="уборка забытых карт Vast по метке и статусу запуска (SPEC 8.3 г)")
    clean_p.add_argument("--repo", default="", help="репозиторий owner/name (передаётся явно)")
    clean_p.add_argument("--summary-file", default="",
                         help="файл итогов (в Actions: $GITHUB_STEP_SUMMARY); пусто = не писать")
    clean_p.set_defaults(func=lambda a: cmd_cleanup_vast(a))
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
