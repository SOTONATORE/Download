"""Оркестратор генерации клипов (этап 8, часть 2; SPEC 3, 4, 8.2, 8.3, 9, 0.1, 0.2, 0.5).

Конвейер: missing.txt -> проверка Release -> промпты Gemini -> аренда карт Vast.ai ->
рендер на воркерах -> немедленная выдача в Release -> уничтожение карт.

Только стандартная библиотека + httpx. Все сообщения на русском (SPEC 0.5).
Секреты (токен воркера, ключи API) не попадают в логи, repr и тексты ошибок (SPEC 0.1).
Время и пауза внедряются (clock/sleep), поэтому модуль детерминирован в тестах.

Протокол воркера (HTTP, порт 8000, заголовок X-Worker-Token):
  POST /task          {num, prompt, num_frames, seed} -> {"task_id": ...}  (допустим и {"id": ...})
  GET  /task/{id}     -> {"state"|"status": "queued|running|done|error|failed", "error": ...}
  GET  /file/{id}     -> байты mp4
  POST /ack/{id}      воркер удаляет клип со своего диска
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import zipfile
from collections import deque
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import httpx

try:  # пакетный и «плоский» импорт (как в остальных модулях core)
    from . import gemini_orchestrator as go
    from . import gemini_prompts as gp
    from . import weights_log as wl
    from .budget import BudgetError, BudgetTracker
    from .log_setup import _MASK, mask_remote_text, set_known_secrets
    from .missing import parse_missing
    from .naming import clip_filename
    from .release_adapter import (  # noqa: F401
        LINKS_FILENAME, PROMPTS_FILENAME, ReleaseInfo, ReleaseStore, ReleaseUploader,
    )
    from .vast_client import (  # noqa: F401
        WORKER_HTTP_PORT, GpuOffer, InstanceInfo, OfferFilter, VastClient, VastDestroyError,
        VastError, VastInstanceNotFound, remove_instance_fields,
    )
except ImportError:  # pragma: no cover
    import gemini_orchestrator as go  # type: ignore
    import gemini_prompts as gp  # type: ignore
    import weights_log as wl  # type: ignore
    from budget import BudgetError, BudgetTracker  # type: ignore
    from log_setup import _MASK, mask_remote_text, set_known_secrets  # type: ignore
    from missing import parse_missing  # type: ignore
    from naming import clip_filename  # type: ignore
    from release_adapter import (  # type: ignore  # noqa: F401
        LINKS_FILENAME, PROMPTS_FILENAME, ReleaseInfo, ReleaseStore, ReleaseUploader,
    )
    from vast_client import (  # type: ignore  # noqa: F401
        WORKER_HTTP_PORT, GpuOffer, InstanceInfo, OfferFilter, VastClient, VastDestroyError,
        VastError, VastInstanceNotFound, remove_instance_fields,
    )

log = logging.getLogger("Generate.orchestrator")

MAX_ATTEMPTS = 3          # попыток рендера одного клипа
# Пилот ComfyUI (pilot-comfyui.yml): проверка инфраструктуры Vast без воркера (Generate/PILOT_REPORT.md).
PILOT_HTTP_PORT = 18188   # порт ComfyUI внутри образа vastai/comfy (supervisor); внешний порт выдаёт Vast
# команды старта по умолчанию нет: образ сам запускает ComfyUI. onstart уходит только при непустом --pilot-start-cmd
PILOT_PROBE_PATH = "/system_stats"   # эндпоинт ComfyUI для проверки готовности
PILOT_AUTH_CODES = frozenset({401, 403})  # ComfyUI отвечает, но просит токен: отдельный результат auth_required
PILOT_PROBE_TIMEOUT_SEC = 5.0
MAX_NET_FAILS = 6         # подряд сбоев связи с воркером до снятия карты
EXIT_OK = 0
EXIT_PARTIAL = 3
LOGS_TIMEOUT_SEC = 10.0   # таймаут GET /logs (периодический и перед уничтожением карты), SPEC 8.4
LOG_PULL_INTERVAL_SEC = 60.0  # периодическая выгрузка логов готовой карты: не чаще раза в минуту
LOG_LEVELS = ("INFO", "DEBUG")
PROBE_TIMEOUT_SEC = 5.0      # таймаут проверки готовности и heartbeat (SPEC 8.2, 8.3 а)
# Аргументы ComfyUI образа воркера (SPEC 8.5). Порт 18188 — порт ComfyUI образа; наружу карта публикует только 8000.
COMFYUI_ARGS = "--disable-auto-launch --disable-xformers --port 18188 --enable-cors-header"
BID_MARGIN_DEFAULT = 0.03  # надбавка к ставке bid по умолчанию; то же значение, что bid_margin в config.default.yaml

# Фильтр стабильности предложений (этап 9е-2; как PARSE_RANK в vast_stable_run.sh).
STABLE_WINDOW = 15            # последние N проб min_bid на один ключ (offer_id, machine_id)
STABLE_MIN_SAMPLES = 5        # меньше проб: предложение не допускается
STABLE_MIN_PRICE_DEFAULT = 0.35
STABLE_MAX_SPREAD_DEFAULT = 0.05


def record_scan_samples(hist: dict, offers) -> None:
    """Добавляет пробу min_bid каждого предложения в историю по ключу (offer_id, machine_id).
    На ключ хранится не больше STABLE_WINDOW последних проб (deque с ограничением): память ограничена."""
    for o in offers:
        key = (o.offer_id, o.machine_id)
        dq = hist.get(key)
        if dq is None:
            dq = hist[key] = deque(maxlen=STABLE_WINDOW)
        dq.append((float(o.price_per_hr), o))


def rank_stable_offers(hist: dict, min_price: float, max_spread: float) -> list:
    """Стабильные предложения по истории, по возрастанию min_bid. Допускается предложение, у которого
    одновременно: проб не меньше STABLE_MIN_SAMPLES; (max - min) / min по последним STABLE_WINDOW пробам
    не больше max_spread; последняя проба min_bid не меньше min_price."""
    rows = []
    for dq in hist.values():
        if len(dq) < STABLE_MIN_SAMPLES:
            continue
        vals = [b for b, _ in list(dq)[-STABLE_WINDOW:]]
        last_bid, last_offer = dq[-1]
        if last_bid < min_price:
            continue
        lo = min(vals)
        if (max(vals) - lo) / lo > max_spread:
            continue
        rows.append(last_offer)
    rows.sort(key=lambda o: o.price_per_hr)
    return rows
READY_REASON = "воркер не готов"
ACCEPT_REASON = "задача не принята"
SSH_NOT_IMPLEMENTED = "SSH-проверка не реализована (SPEC 8.2)"
PORTAL_NOT_IMPLEMENTED = "проверка через портал не реализована (SPEC 8.2, 13.1)"

_DONE_STATUSES = frozenset({"done", "completed", "ready"})
_FAILED_STATUSES = frozenset({"failed", "error"})

# Маркеры ошибки в тексте статуса Vast (SPEC 8.2, неудачный старт), одна константа.
# «failed to create», «OCI runtime», «CDI», «shim» основаны на наблюдаемой ошибке (карта 55771098,
# хост 362355, 13.10.2026): «failed to create task … failed to create shim task: OCI runtime create
# failed: … failed to inject CDI devices: unresolvable CDI devices … gpu=2: unknown».
# «failed to», «unresolvable», «error» — общие слова ошибки. Сравнение без учёта регистра и по подстроке.
# Текст «Preparing GPUs...» и подобный прогресс маркеров не содержит и карту не снимает.
START_FAILURE_MARKERS = ("failed to", "unresolvable", "error", "failed to create", "OCI runtime", "CDI", "shim")
# Статусы Vast, которые после начала старта (loading или created) означают неудачный старт (SPEC 8.2)
STOPPED_AFTER_START_STATUSES = ("offline", "exited")
START_FAILURE_REASON = "старт карты не удался"
# снятие карты при перебитии ставки (9е-3): Vast остановил прерываемую карту; машина и bad_hosts.txt не трогаются
OUTBID_REASON = "ставка bid перебита, Vast остановил карту"
# Сборка воркера (SPEC 8.4): номер коммита, 7-64 символа 0-9 и a-f
WORKER_BUILD_RE = re.compile(r"[0-9a-f]{7,64}")
WORKER_ARCHIVE_URL = "https://codeload.github.com/SOTONATORE/Download/tar.gz/{sha}"
WORKER_SRC_DIR = "/workspace/src"
WORKER_BOOTSTRAP_LOG = "/workspace/bootstrap.log"

# Сколько ответов опроса подряд с «экземпляр не найден» (VastInstanceNotFound) терпим, прежде чем снять
# карту (SPEC 8.3). Это число подряд идущих ответов опроса, не секунды; любой успешный ответ обнуляет счётчик.
NOT_FOUND_LIMIT = 10
NOT_FOUND_REASON = "экземпляр не найден в Vast"

# Шаблоны маскирования текста с карты живут в core/log_setup.py (mask_remote_text): один список на журнал и на _mask_remote.


class _WorkerNetError(Exception):
    """Сбой связи с воркером (без деталей: они могут содержать заголовки)."""


class _WorkerHttpError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


class _LogCursorError(Exception):
    """Ответ GET /logs без корректного заголовка X-Log-Next: курсор неизвестен, строки не печатаются."""


@dataclass
class RunSummary:
    total_needed: int = 0      # сколько клипов нужно было сделать в этом запуске
    completed: int = 0
    failed: int = 0
    pending: int = 0
    skipped_done: int = 0      # уже лежали в Release
    spent_usd: float = 0.0
    exit_code: int = EXIT_OK
    deadline_reached: bool = False
    budget_exceeded: bool = False
    cleanup_ok: bool = True
    cards_rented: int = 0
    destroyed_cards: int = 0   # карты, уничтоженные в этом запуске (по ходу и при уборке)
    avg_render_sec: float = 0.0  # среднее время рендера клипа: от отправки до выдачи, с

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Task:
    num: int
    id: str
    started: float


@dataclass
class _CardSlot:
    index: int
    generation: int
    instance_id: int
    machine_id: int
    price: float
    created: float
    state: str = "booting"     # booting -> ready
    url: str = ""
    contacted: bool = False    # воркер хотя бы раз ответил
    net_fail: int = 0
    vast_status: str = ""      # последнее состояние Vast, для строк о переходах (не на каждый опрос)
    task: Optional[_Task] = None
    log_next: Optional[int] = None    # курсор логов: номер последней строки, полученной с карты
    log_pulled: Optional[float] = None  # время последней попытки периодической выгрузки
    offer_id: int = 0                 # предложение, по которому арендована карта (для исключения)
    saw_loading: bool = False         # Vast хотя бы раз показал «loading»: старт начался (SPEC 8.2)
    saw_created: bool = False         # Vast показал «created»: старт начался (SPEC 8.2)
    bid_requested: bool = False       # карта арендована как прерываемая (ставка bid, SPEC 8.2)
    dph_search: float = 0.0           # dph_total из поиска: запасная цена, если bid не подтверждён
    mode_mismatch: bool = False       # Vast не подтвердил bid (is_bid не True): бюджет по dph_total
    not_found: int = 0                # подряд идущих ответов «экземпляр не найден» (NOT_FOUND_LIMIT, SPEC 8.3)
    cost_checked: bool = False        # цена по ответу Vast уже учтена или признана недоступной (SPEC 8.2)
    monitor: Optional[object] = None  # монитор загрузки весов (core/weights_log.py), создаётся по требованию
    ready_at: float = 0.0             # момент перехода в ready (срок принятия первой задачи, SPEC 8.2)
    accepted: bool = False            # воркер принял хотя бы одну задачу (POST /task)
    ready_wait_since: Optional[float] = None  # с какого момента ждём готовность воркера (SPEC 8.2)
    last_beat: float = -1e9           # время последнего heartbeat-запроса (SPEC 8.3 а)
    last_ok_utc: str = ""              # UTC последнего успешного heartbeat (файл heartbeat)
    weights_reason: Optional[str] = None  # причина из разбора логов весов, ждёт снятия карты

    @property
    def started(self) -> bool:
        """Старт карты начался: Vast показал «loading» или «created» (SPEC 8.2)."""
        return self.saw_loading or self.saw_created


def _prompt_ready(entry) -> bool:
    if not isinstance(entry, dict):
        return False
    num_frames = entry.get("num_frames")
    return (entry.get("status") == "ready"
            and bool(str(entry.get("prompt", "")).strip())
            and isinstance(num_frames, int) and not isinstance(num_frames, bool)
            and num_frames > 0)


def _has_start_marker(text: str) -> bool:
    """Есть ли в тексте статуса маркер ошибки рантайма (без учёта регистра, по подстроке)."""
    low = text.lower()
    return any(marker.lower() in low for marker in START_FAILURE_MARKERS)


def _one_line(text: str) -> str:
    """Текст в одну строку: пробельные символы сворачиваются, длина не ограничивается."""
    return " ".join(text.split())


def _mapped_port(info: InstanceInfo, port: int) -> Optional[int]:
    """Внешний порт Vast, проброшенный на внутренний port (поле ports экземпляра); None, если проброса нет."""
    ports = info.raw.get("ports")
    if isinstance(ports, str):
        try:
            ports = json.loads(ports)
        except ValueError:
            return None
    if not isinstance(ports, dict):
        return None
    mapped = ports.get(f"{port}/tcp")
    if isinstance(mapped, list) and mapped and isinstance(mapped[0], dict):
        try:
            return int(mapped[0].get("HostPort"))
        except (TypeError, ValueError):
            return None
    return None


def _pilot_verdict(code: Optional[int]) -> Optional[str]:
    """Вердикт пробы GET /system_stats: ok (200), auth_required (401/403), None — ждём дальше."""
    if code == 200:
        return "ok"
    if code in PILOT_AUTH_CODES:
        return "auth_required"
    return None


def _valid_worker_build(value) -> bool:
    return isinstance(value, str) and WORKER_BUILD_RE.fullmatch(value) is not None


def _worker_onstart(worker_build: str) -> str:
    """Команда onstart карты воркера (SPEC 8.1, вариант A).

    Номер коммита проверяется до сборки текста. Архив коммита скачивается с codeload.github.com и
    распаковывается в WORKER_SRC_DIR без верхней папки (Generate окажется в WORKER_SRC_DIR/Generate).
    bootstrap.sh запускается в фоне, вывод уходит в WORKER_BOOTSTRAP_LOG. REPO_DIR передаётся явно,
    чтобы bootstrap не скачивал тот же архив второй раз. В тексте команды нет токенов и секретов:
    они приходят в окружении карты (WORKER_TOKEN, HF_TOKEN).
    """
    if not _valid_worker_build(worker_build):
        raise gp.GeminiInputError("Номер сборки воркера (--worker-build) должен быть из 7-64 символов 0-9 и a-f.")
    url = WORKER_ARCHIVE_URL.format(sha=worker_build)
    src = WORKER_SRC_DIR
    return (f"mkdir -p {src} && curl -fsSL --retry 3 {url} | tar -xz --strip-components=1 -C {src} && "
            f"(REPO_DIR={src}/Generate nohup bash {src}/Generate/worker/bootstrap.sh "
            f"> {WORKER_BOOTSTRAP_LOG} 2>&1 &)")


def _port_keys_text(info: InstanceInfo) -> str:
    """Имена ключей поля ports для лога. Значения (HostPort и пр.) не выводятся."""
    ports = info.raw.get("ports")
    if isinstance(ports, str):
        try:
            ports = json.loads(ports)
        except ValueError:
            return "поле ports не разобрано"
    if not isinstance(ports, dict):
        return "поля ports нет"
    keys = sorted(str(k) for k in ports)
    return ", ".join(keys) if keys else "пусто"


def _card_name(slot, *, cap: bool = False) -> str:
    """Имя карты в логах (SPEC 8.2): «карта <слот> поколение <поколение>», номер экземпляра Vast в скобках."""
    name = f"карта {slot.index} поколение {slot.generation} (экземпляр {slot.instance_id})"
    return name[:1].upper() + name[1:] if cap else name


def _utc_now() -> str:
    """Текущее время UTC, ISO 8601 с суффиксом Z (для файла heartbeat)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Артефакт запуска и выдача служебных файлов (SPEC 9.1, 9.4, 10.4, этап 6c)
ARTIFACT_SAVE_LIMIT_SEC = 60.0      # SPEC 8.3 (в), шаг 1: сохранение логов и JSON до уничтожения карт
REPORT_FILENAME = "summary.md"
FILES_LIST_NAME = "FILES.txt"
LOG_BUNDLE_NAME = "run-logs.zip"
MISSING_FILENAME = "missing.txt"
SERVICE_DIR = "service"


class ReleaseStartError(gp.GeminiInputError):
    """Служебный релиз не открыт или файл запуска не загружен: остановка до трат, код 2 (SPEC 9.1, E1)."""


_SCAN_CHUNK = 1 << 20


def _file_has_any(path: Path, needles) -> bool:
    """Есть ли в файле хотя бы одна из строк needles. Файл читается кусками (SPEC 10.3), не целиком."""
    keep = max(len(n) for n in needles) - 1  # хвост предыдущего куска: строка могла разорваться на границе
    tail = ""
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        while True:
            chunk = fh.read(_SCAN_CHUNK)
            if not chunk:
                return False
            window = tail + chunk
            if any(n in window for n in needles):
                return True
            tail = window[-keep:] if keep > 0 else ""


class RunArtifact:
    """Артефакт запуска (SPEC 10.4): run/, cards/, service/, FILES.txt.

    Строки пишутся на диск по мере поступления (без накопления в памяти); каждая строка
    маскируется до записи (SPEC 0.1). Файлы открываются на дозапись, поэтому данные не теряются
    при падении процесса.
    """

    def __init__(self, root, mask: Callable[[str], str]) -> None:
        self.root = Path(root)
        self._mask = mask
        # адреса карт (IP и IP:порт), известные оркестратору; файл, где адрес остался, в архив не идёт (этап 9а)
        self._addresses: set = set()
        for sub in ("run", "cards", SERVICE_DIR):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def add_address(self, value: str) -> None:
        """Запоминает адрес карты (этап 9а). Текст, который пишется после этого, маскируется оркестратором;
        файл, записанный раньше с адресом внутри, исключается из run-logs.zip (make_bundle)."""
        if value:
            self._addresses.add(str(value))

    def _append(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(self._mask(text) + "\n")

    def _write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._mask(text), encoding="utf-8", newline="\n")

    def card_dir(self, slot) -> Path:
        return self.root / "cards" / f"slot{slot.index}_generation{slot.generation}"

    def card_line(self, slot, line: str) -> None:
        self._append(self.card_dir(slot) / "worker_log.txt", line)

    def card_status(self, slot, text: str) -> None:
        self._append(self.card_dir(slot) / "status.txt", f"{_utc_now()} {text}")

    def card_meta(self, slot, data: dict) -> None:
        self._write(self.card_dir(slot) / "meta.json", json.dumps(data, ensure_ascii=False, indent=2))

    def card_instance(self, slot, raw: dict) -> None:
        # единственный писатель instance.json: поля адреса, портов, ssh и env удаляются до записи (этап 9а)
        self._write(self.card_dir(slot) / "instance.json",
                    json.dumps(remove_instance_fields(raw), ensure_ascii=False, indent=2, default=str))

    def service_copy(self, src: str, name: str) -> None:
        shutil.copyfile(src, self.root / SERVICE_DIR / name)

    def write_report(self, text: str) -> None:
        self._write(self.root / "run" / REPORT_FILENAME, text)

    def write_files_list(self) -> None:
        # Список строится обходом каталога. Единственный журнал запуска: run/debug.log (пишет cli через
        # core/log_setup.py, SPEC 10.4); run/run.log оркестратор больше не пишет (этап 7б).
        rows = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and p.name != FILES_LIST_NAME:
                rows.append(f"{p.relative_to(self.root).as_posix()}\t{p.stat().st_size}")
        self._write(self.root / FILES_LIST_NAME, "\n".join(rows) + "\n")

    def make_bundle(self, dest: str) -> None:
        """Архив каталога. Файл, в котором остался адрес карты, в архив не попадает; одна строка в журнал на файл (этап 9а)."""
        needles = sorted(self._addresses, key=len, reverse=True)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(self.root.rglob("*")):
                if not p.is_file():
                    continue
                name = p.relative_to(self.root).as_posix()
                if needles and _file_has_any(p, needles):
                    log.warning("Файл %s не включён в архив: в нём есть адрес карты.", name)
                    continue
                zf.write(p, name)


class RunPublisher:
    """Выдача служебных файлов запуска в релизы (SPEC 9.1, 9.4, 9.7; этап 6c, E1-E4).

    Общая логика для оркестратора и для `finalize-release` (F2, без дублирования).
    start(): missing.txt и prompts.json до трат; ошибка -> ReleaseStartError (код 2).
    finish(): отчёт и архив логов -> служебный релиз, prompts.json -> служебный,
    generated_links.txt -> итоговый. Ошибка одного файла: одна строка в лог с именем и причиной,
    файл попадает в failures, остальные файлы всё равно пробуются (E4).
    """

    def __init__(self, uploader, repo: str, tmp_dir: str, *, artifact: Optional[RunArtifact] = None,
                 mask: Optional[Callable[[str], str]] = None) -> None:
        self.up = uploader
        self.repo = repo
        self.tmp = tmp_dir
        self.artifact = artifact
        self._mask = mask or (lambda text: text)
        self.uploaded: list = []
        self.failures: list = []

    def _reason(self, exc: Exception) -> str:
        if isinstance(exc, (gp.GeminiInputError, gp.GeminiRuntimeError)):
            return self._mask(str(exc))
        return type(exc).__name__

    def _copy(self, src: str, name: str) -> str:
        dst = os.path.join(self.tmp, name)
        shutil.copyfile(src, dst)
        if self.artifact is not None:
            self.artifact.service_copy(dst, name)
        return dst

    def start(self, missing_path, prompts_path) -> None:
        """E1: файлы запуска в служебный релиз до любых трат. Первая ошибка останавливает запуск."""
        if os.path.isfile(str(missing_path)):
            dst = self._copy(str(missing_path), MISSING_FILENAME)
            self._required(lambda: self.up.upload_run_file(dst), MISSING_FILENAME)
        if os.path.isfile(str(prompts_path)):
            dst = self._copy(str(prompts_path), PROMPTS_FILENAME)
            self._required(lambda: self.up.upload_service_file(dst), PROMPTS_FILENAME)

    def _required(self, fn: Callable[[], None], name: str) -> None:
        try:
            fn()
        except Exception as exc:
            raise ReleaseStartError(
                f"Не удалось загрузить {name} в служебный релиз: {self._reason(exc)}. "
                "Запуск остановлен до трат.") from None
        self.uploaded.append(name)

    def _soft(self, fn: Callable[[], None], name: str) -> None:
        try:
            fn()
        except Exception as exc:
            self.failures.append(name)
            log.error("Не удалось загрузить %s: %s.", name, self._reason(exc))
            return
        self.uploaded.append(name)

    def finish(self, *, prompts_path=None, report_path=None, bundle_path=None) -> None:
        """E2, E4: служебные файлы в служебный релиз, generated_links.txt в итоговый."""
        if report_path:
            self._soft(lambda: self.up.upload_run_file(report_path), os.path.basename(report_path))
        if bundle_path:
            self._soft(lambda: self.up.upload_run_file(bundle_path), os.path.basename(bundle_path))
        if prompts_path and os.path.isfile(str(prompts_path)):
            dst = self._copy(str(prompts_path), PROMPTS_FILENAME)
            self._soft(lambda: self.up.upload_service_file(dst), PROMPTS_FILENAME)
        if not self.repo:
            log.warning("Репозиторий не задан: generated_links.txt не создан.")
        elif not (getattr(self.up, "done", None) or getattr(self.up, "asset_tags", None)):
            log.info("Нет готовых клипов: generated_links.txt не выкладывается.")  # SPEC 9.4
        else:
            dst = os.path.join(self.tmp, LINKS_FILENAME)
            Path(dst).write_text(self.up.links_text(self.repo), encoding="utf-8", newline="\n")
            self._soft(lambda: self.up.upload_service_file(dst), LINKS_FILENAME)


class Orchestrator:
    """Координатор: промпты, карты Vast, рендер, выдача в Release, уборка."""

    def __init__(
        self,
        segments: list,
        missing_path,
        cfg,
        timing_params,
        release_store,
        vast_client,
        gemini_transport,
        prompts_path,
        *,
        max_cards: int = 4,
        budget_limit_usd: float = 2.0,
        job_deadline_min: int = 330,
        silent_host_timeout_min: int = 20,
        worker_idle_timeout_min: int = 2,
        offer_filter: Optional[OfferFilter] = None,
        bid_margin: float = BID_MARGIN_DEFAULT,
        excluded_machines: Optional[frozenset] = None,
        docker_image: str = "",
        disk_gb: int = 100,
        limit_clips: int = 0,
        repo: str = "",
        run_id: str = "",
        profile: str = "",
        filename_template: str = "{num}.mp4",
        clip_name_width: int = 0,
        release_tag: Optional[str] = None,
        release_asset_limit: int = 900,
        gemini_options: Optional[dict] = None,
        http_client: Optional[httpx.Client] = None,
        work_dir=None,
        worker_token: Optional[str] = None,
        seed_base: int = 0,
        poll_interval_sec: float = 10.0,
        rent_retry_sec: float = 30.0,
        max_rent_tries: int = 5,
        scan_sec: float = 0.0,
        scan_step_sec: float = 20.0,
        stable_min_price: float = STABLE_MIN_PRICE_DEFAULT,
        stable_max_spread: float = STABLE_MAX_SPREAD_DEFAULT,
        drain_grace_sec: float = 900.0,
        task_timeout_sec: float = 1200.0,
        render_estimate_sec: float = 120.0,
        rent_estimate_sec: float = 1800.0,
        http_timeout_sec: float = 60.0,
        log_level: str = "INFO",
        clock: Callable[[], float] = time.monotonic,
        sleep: Optional[Callable[[float], None]] = None,
        stop_event: Optional[threading.Event] = None,
        pilot: bool = False,
        pilot_start_cmd: str = "",
        worker_build: Optional[str] = None,
        speed_check_at_pct: int = 10,
        speed_min_mbps: float = 400.0,
        progress_step_pct: int = 10,
        stall_timeout_min: int = 3,
        stage_max_min: int = 20,
        worker_ready_timeout_min: int = 10,
        task_accept_timeout_min: int = 3,
        heartbeat_interval_sec: int = 10,
        monitor_interval_sec: int = 10,
        min_inet_mbps: float = 800.0,
        heartbeat_path=None,
        artifact_dir=None,
    ) -> None:
        if max_cards <= 0:
            raise gp.GeminiInputError("max_cards должен быть больше нуля.")
        if job_deadline_min <= 0 or silent_host_timeout_min <= 0:
            raise gp.GeminiInputError("Сроки (дедлайн, молчание хоста) должны быть положительными.")
        # run_id обязателен (SPEC 9.1, 8.3 г): по нему карты помечаются меткой для уборки (kill-cards, cleanup_vast)
        if run_id is None or not str(run_id).strip():
            raise gp.GeminiInputError("Не задан номер запуска (run_id): без него карты нельзя пометить меткой для уборки.")
        # номер сборки воркера (SPEC 8.4): проверяется до любых трат; пилот его не использует
        if not pilot and not _valid_worker_build(worker_build):
            raise gp.GeminiInputError("Номер сборки воркера (--worker-build) должен быть из 7-64 символов 0-9 и a-f.")
        if log_level not in LOG_LEVELS:
            raise gp.GeminiInputError("Уровень логов должен быть INFO или DEBUG.")
        # пороги мониторинга (SPEC 8.2): значения приходят из конфига; здесь только проверка знаков
        sec_values = (stall_timeout_min, stage_max_min, worker_ready_timeout_min, task_accept_timeout_min,
                      heartbeat_interval_sec, monitor_interval_sec)
        if min(sec_values) < 1 or speed_min_mbps <= 0 or speed_check_at_pct < 1 or progress_step_pct < 1:
            raise gp.GeminiInputError("Пороги мониторинга весов и готовности должны быть положительными.")
        if scan_sec < 0 or scan_step_sec <= 0:
            raise gp.GeminiInputError("Окно сбора цен: scan_sec не может быть отрицательным, "
                                      "scan_step_sec должен быть больше нуля.")
        if int(max_rent_tries) < 1:
            raise gp.GeminiInputError("Лимит запросов аренды (max_rent_tries) должен быть не меньше 1.")
        if stage_max_min < stall_timeout_min:
            raise gp.GeminiInputError("«stage_max_min» не может быть меньше «stall_timeout_min».")
        self.log_level = log_level
        self.segments = segments
        self.missing_path = missing_path
        self.cfg = cfg
        self.timing_params = timing_params
        self.release_store = release_store
        self.vast = vast_client
        self.gemini_transport = gemini_transport
        self.prompts_path = Path(prompts_path)

        self.max_cards = int(max_cards)
        self.job_deadline_min = job_deadline_min
        self.worker_idle_timeout_min = worker_idle_timeout_min
        # пороги мониторинга (SPEC 8.2, 8.3 а); значения из конфига
        self.speed_check_at_pct = int(speed_check_at_pct)
        self.speed_min_mbps = float(speed_min_mbps)
        self.progress_step_pct = int(progress_step_pct)
        self.min_inet_mbps = float(min_inet_mbps)   # ожидаемая скорость; правило скорости берёт speed_min_mbps
        self.monitor_interval_sec = int(monitor_interval_sec)  # в 6б не используется: снимки делает воркер (SPEC 10)
        self._stall_sec = stall_timeout_min * 60.0
        self._stage_max_sec = stage_max_min * 60.0
        self._ready_sec = worker_ready_timeout_min * 60.0
        self._accept_sec = task_accept_timeout_min * 60.0
        self._heartbeat_sec = float(heartbeat_interval_sec)
        self.heartbeat_path = Path(heartbeat_path) if heartbeat_path else None
        self._last_heartbeat_tick = -1e9
        self._probe_notes_logged = False
        self._weights_source_noted = False
        self._deadline_sec = job_deadline_min * 60.0
        self._silent_sec = silent_host_timeout_min * 60.0
        self.offer_filter = offer_filter or OfferFilter()
        self.bid_margin = float(bid_margin)  # надбавка к min_bid для ставки прерываемой аренды (SPEC 8.2)
        self.docker_image = docker_image
        self.disk_gb = disk_gb
        self.limit_clips = limit_clips
        self.repo = repo
        self.pilot = bool(pilot)
        # пустая команда не передаётся дальше: onstart в payload не попадает
        self.pilot_start_cmd = (pilot_start_cmd or "").strip() or None
        self.label = f"{'pilot' if self.pilot else 'gen'}-{run_id}"
        self.worker_build = "" if self.pilot else worker_build
        self._onstart = None if self.pilot else _worker_onstart(worker_build)
        self.profile = profile
        self.template = filename_template
        self.width = clip_name_width
        self.release_tag = release_tag
        self.release_asset_limit = release_asset_limit
        self.gemini_options = dict(gemini_options or {})
        self.work_dir = work_dir
        self.seed_base = seed_base
        self.poll_interval = poll_interval_sec
        self.rent_retry = rent_retry_sec
        # лимит запросов аренды за запуск (9е-3): считается каждый запрос, успешный и неудачный
        self.max_rent_tries = int(max_rent_tries)
        # окно сбора цен перед первой арендой запуска (SPEC 8.2): 0 = окно выключено
        self.scan_sec = float(scan_sec)
        self.scan_step = float(scan_step_sec)
        # фильтр стабильности (этап 9е-2): порог последней пробы и допустимый разброс, из конфига
        self.stable_min_price = float(stable_min_price)
        self.stable_max_spread = float(stable_max_spread)
        self.drain_grace = drain_grace_sec
        self.task_timeout = task_timeout_sec
        self.render_estimate = render_estimate_sec
        self.rent_estimate = rent_estimate_sec
        self.http_timeout = http_timeout_sec
        self.clock = clock
        # сигнал остановки (SPEC 8.3 (в)): cli выставляет флаг; ожидания между шагами прерываются им
        self._stop = stop_event if stop_event is not None else threading.Event()
        # пауза по умолчанию прерываемая: Event.wait возвращается сразу при установленном флаге
        self._sleep = sleep if sleep is not None else self._stop.wait

        self._token = worker_token or secrets.token_urlsafe(24)
        self._headers = {"X-Worker-Token": self._token}
        # значения секретов известны: маска журнала и текста с карты получает их явно, из окружения модуль не читает
        # (решение владельца 18.10.2026, риск F11; этап 9а). Вызов один раз на запуск.
        set_known_secrets([self._token, os.environ.get("HF_TOKEN", "")])
        # адреса карт (IP и IP:порт), которые маскируются в тексте с карты и не попадают в архив (этап 9а)
        self._card_addresses: set = set()
        self._owns_http = http_client is None
        self._http = http_client if http_client is not None else httpx.Client(timeout=http_timeout_sec)
        self.budget = BudgetTracker(budget_limit_usd, clock=clock)

        # состояние запуска
        self._t0 = 0.0
        self._tmp = ""
        self._uploader: Optional[ReleaseUploader] = None
        self._state: dict = {"prompts": {}}
        self._prompts: dict = self._state["prompts"]
        self._queue: deque = deque()
        self._slots: dict = {}
        self._gens: dict = {}
        self._attempts: dict = {}
        self._done: set = set()
        self._failed: set = set()
        self._durations: list = []
        self._bad_machines: set = set()
        self._bad_offers: set = set()
        # предложения, по которым отправлен запрос аренды (успешный или нет): в этом запуске не выбираются снова (9е-3)
        self._used_offers: set = set()
        self._rent_requests = 0          # запросов аренды отправлено за запуск
        self._rent_exhausted = False     # лимит max_rent_tries исчерпан: новых аренд нет, запуск завершается
        self._next_wait = 0.0            # пауза до следующего поиска, сек; выставляет _rent_one перед возвратом retry
        # постоянный список плохих машин (bad_hosts.txt, SPEC 8.2): живёт между запусками,
        # в отличие от _bad_machines; в лог пишется одна строка на запуск
        self._excluded_machines: frozenset = frozenset(int(m) for m in (excluded_machines or ()))
        self._excluded_logged = False
        self._next_rent = 0.0
        # окно сбора цен: только до первой аренды запуска; предложения копятся по ID
        self._scan_started: Optional[float] = None
        self._scan_end = 0.0
        self._scan_next = 0.0
        self._scan_steps = 0
        self._scan_failed = 0
        self._scan_hist: dict = {}  # пробы min_bid по ключу (offer_id, machine_id), до STABLE_WINDOW на ключ
        self._scan_waiting = False  # окно закрыто, стабильных предложений пока нет: ждём и опрашиваем снова
        self._scan_done = False
        self._budget_blocked = False
        self._cards_rented = 0
        self._destroyed = 0
        self.deadline_reached = False
        self.budget_exceeded = False
        self.cleanup_ok = True
        self.run_id = str(run_id).strip()
        self._publisher: Optional[RunPublisher] = None
        # артефакт запуска и пилота (SPEC 10.4, этап 7в); без --artifact-dir артефакта нет
        self._artifact = RunArtifact(artifact_dir, self._mask_remote) if artifact_dir else None

    def __repr__(self) -> str:
        return f"Orchestrator(метка={self.label}, карт<={self.max_cards})"

    __str__ = __repr__

    # --- служебное ---------------------------------------------------------
    def _redact(self, text: str) -> str:
        return text.replace(self._token, "***") if self._token else text

    def _remember_card_address(self, ip, port) -> None:
        """Запоминает адрес карты (IP и IP:порт) для маски и для архива (SPEC 0.1, 10.4; этап 9а)."""
        if not ip:
            return
        found = {str(ip)}
        if port:
            found.add(f"{ip}:{port}")
        self._card_addresses |= found
        if self._artifact is not None:
            for value in found:
                self._artifact.add_address(value)

    def _mask_remote(self, text: str) -> str:
        """Маскирует текст, пришедший с карты (SPEC 0.1, п. 14).

        Здесь литеральные значения, известные оркестратору: адреса карт (этап 9а), токен воркера,
        HF_TOKEN, ключ Vast. Шаблоны и правила пар берутся из core/log_setup.py (mask_remote_text), как у журнала.
        """
        for addr in sorted(self._card_addresses, key=len, reverse=True):
            text = text.replace(addr, _MASK)
        text = self._redact(text)
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if len(hf_token) >= 8:
            text = text.replace(hf_token, _MASK)
        vast_redact = getattr(self.vast, "_redact", None)  # метод клиента Vast маскирует свой ключ
        if callable(vast_redact):
            text = vast_redact(text)
        return mask_remote_text(text)

    def _monitor(self, slot: _CardSlot):
        """Монитор загрузки весов карты (SPEC 8.2): создаётся по требованию, у каждой карты свой."""
        if slot.monitor is None:
            def emit(text: str, warn: bool = False, _slot=slot) -> None:
                msg = f"{_card_name(_slot, cap=True)}: {text}"
                if warn:
                    log.warning("%s", msg)
                else:
                    log.info("%s", msg)
            slot.monitor = wl.WeightsMonitor(
                clock=self.clock, log=emit, speed_check_pct=self.speed_check_at_pct,
                speed_min_mbps=self.speed_min_mbps, progress_step_pct=self.progress_step_pct,
                stall_sec=self._stall_sec, stage_max_sec=self._stage_max_sec)
        return slot.monitor

    def _pull_card_logs(self, slot: _CardSlot, *, final: bool, timeout: float = LOGS_TIMEOUT_SEC) -> bool:
        """GET /logs карты с курсора предыдущего успешного ответа (SPEC 8.4).

        Печатаются только строки после курсора, каждая с префиксом «[карта N]» и после маскирования.
        Курсор двигается только при успехе, поэтому при сбое строки запрашиваются снова и дублей нет.
        Сбой даёт одно предупреждение с именем типа исключения; текст исключения не печатается:
        в нём может быть адрес с заголовками.
        """
        params = {} if slot.log_next is None else {"since": slot.log_next}
        try:
            resp = self._http.get(slot.url + "/logs", headers=self._headers, params=params,
                                  timeout=timeout)
            if resp.status_code != 200:
                raise _WorkerHttpError(resp.status_code)
            cursor = resp.headers.get("X-Log-Next", "")
            if not cursor.isdigit():
                raise _LogCursorError()
            text = resp.text
        except (httpx.HTTPError, _WorkerHttpError, _LogCursorError) as exc:
            log.warning("Не удалось получить логи карты %d: %s.", slot.instance_id, type(exc).__name__)
            return False
        lines = self._mask_remote(text).splitlines()
        slot.log_next = int(cursor)
        if self._artifact is not None:  # строки уже замаскированы; пишутся по мере получения (SPEC 10.3)
            for line in lines:
                self._artifact.card_line(slot, line)
        if not final and lines:
            reason = self._monitor(slot).feed(lines)
            if reason:
                slot.weights_reason = reason  # снятие карты делает _pull_due_logs, не здесь (нет рекурсии)
        if final:
            log.info("Логи карты %d перед уничтожением: новых строк %d.", slot.instance_id, len(lines))
        for line in lines:
            log.info("[карта %d] %s", slot.instance_id, line)
        return True

    def _pull_due_logs(self, now: float) -> None:
        """Периодическая выгрузка: готовые карты, не чаще LOG_PULL_INTERVAL_SEC на карту.

        Уничтоженная карта исчезает из _slots и больше не опрашивается. Любой сбой не выходит
        из цикла: цикл продолжается, уничтожение не задерживается.
        """
        for slot in list(self._slots.values()):
            if slot.state not in ("booting", "ready") or not slot.url:  # C1: booting с адресом тоже
                continue
            if slot.log_pulled is not None and now - slot.log_pulled < LOG_PULL_INTERVAL_SEC:
                continue
            slot.log_pulled = now  # время фиксируется и при сбое: повтор не раньше чем через минуту
            try:
                self._pull_card_logs(slot, final=False)
                if slot.weights_reason and self._slots.get(slot.index) is slot:
                    reason, slot.weights_reason = slot.weights_reason, None
                    self._retire_bad(slot, reason)
            except Exception as exc:  # выгрузка логов не должна ронять конвейер
                log.warning("Не удалось получить логи карты %d: %s.", slot.instance_id, type(exc).__name__)

    def _info(self, msg: str) -> None:
        log.info("%s", msg)

    # --- запуск -------------------------------------------------------------
    def run(self) -> RunSummary:
        self._t0 = self.clock()
        return self._run_body()

    def _run_body(self) -> RunSummary:
        try:
            if self.pilot:
                return self._run_pilot()
            missing = parse_missing(self.missing_path, self.segments)
            if not missing:
                log.info("Пропущенных сегментов нет: генерировать нечего.")
                return RunSummary()
            if self.limit_clips > 0:
                missing = missing[: self.limit_clips]
                log.info("Пробный прогон: берём первые %d номеров.", len(missing))
            if self.work_dir:
                Path(self.work_dir).mkdir(parents=True, exist_ok=True)
            self._tmp = tempfile.mkdtemp(prefix="gen-", dir=str(self.work_dir) if self.work_dir else None)
            return self._run_inner(missing)
        finally:
            if self._tmp:
                shutil.rmtree(self._tmp, ignore_errors=True)
            if self._owns_http:
                self._http.close()

    def _run_inner(self, missing: list) -> RunSummary:
        uploader = ReleaseUploader(
            self.release_store, self.segments, self.profile, release_tag=self.release_tag,
            release_asset_limit=self.release_asset_limit, filename_template=self.template,
            log=self._info, run_id=self.run_id)
        try:
            done = uploader.prepare()  # открывает служебный релиз run-<run_id>-service (SPEC 9.1, E1)
        except gp.GeminiRuntimeError as exc:
            raise ReleaseStartError(
                f"Не удалось открыть релиз запуска: {self._mask_remote(str(exc))}. "
                "Запуск остановлен до трат.") from None
        self._uploader = uploader
        self._publisher = RunPublisher(uploader, self.repo, self._tmp, artifact=self._artifact,
                                       mask=self._mask_remote)
        needed = [n for n in missing if n not in done]
        summary = RunSummary(total_needed=len(needed), skipped_done=len(missing) - len(needed))
        # E1: missing.txt и prompts.json до любых трат; ошибка загрузки останавливает запуск (код 2)
        self._publisher.start(self.missing_path, self.prompts_path)
        if not needed:
            log.info("Все запрошенные клипы уже в Release (%d): аренда карт не требуется.", len(missing))
            self._fill_summary(summary, needed)
            self._finish(summary)
            return summary
        # SPEC 8.3 а: первый heartbeat пишется до фазы Gemini, чтобы наблюдатель не ждал дольше 10 минут
        self._write_heartbeat()
        try:
            try:
                ready = self._generate(needed)
                self._work(ready)
            finally:
                self._cleanup()  # F1 (1)-(2): логи и JSON с лимитом времени, затем уничтожение по метке
        finally:
            self._fill_summary(summary, needed)
            self._finish(summary)  # F1 (3)-(4): отчёт и архив, итоговый релиз, итоговая строка
        return summary

    def _fill_summary(self, summary: RunSummary, needed: list) -> None:
        summary.completed = len(self._done)
        summary.failed = len(self._failed)
        summary.pending = len(needed) - summary.completed - summary.failed
        summary.spent_usd = self.budget.current_spent(self.clock())
        summary.deadline_reached = self.deadline_reached
        summary.budget_exceeded = self.budget_exceeded
        summary.cleanup_ok = self.cleanup_ok
        summary.cards_rented = self._cards_rented
        summary.destroyed_cards = self._destroyed
        summary.avg_render_sec = (sum(self._durations) / len(self._durations)
                                  if self._durations else 0.0)
        full = summary.completed == len(needed) and self.cleanup_ok
        summary.exit_code = EXIT_OK if full else EXIT_PARTIAL

    def _report_text(self, summary: RunSummary) -> str:
        """Отчёт запуска (run/summary.md): только числа и итоговые признаки, без адресов и токенов."""
        lines = [
            f"# Итог запуска {self.run_id}",
            "",
            f"- нужно сделать клипов: {summary.total_needed}",
            f"- готово: {summary.completed}",
            f"- с ошибкой: {summary.failed}",
            f"- не обработано: {summary.pending}",
            f"- уже было в Release: {summary.skipped_done}",
            f"- арендовано карт: {summary.cards_rented}",
            f"- уничтожено карт: {summary.destroyed_cards}",
            f"- среднее время клипа, с: {summary.avg_render_sec:.1f}",
            f"- потрачено, USD: {summary.spent_usd:.4f}",
            f"- код завершения (до загрузки отчёта): {summary.exit_code}",
        ]
        if not summary.cleanup_ok:
            lines.append("- ВНИМАНИЕ: уничтожение карт не подтверждено, проверьте Vast.ai")
        return "\n".join(lines) + "\n"

    def _finish(self, summary: RunSummary) -> None:
        """F1 (3)-(4) и E2: отчёт, архив логов и служебные файлы; итоговая строка последней."""
        self._log_budget()
        pub = self._publisher
        if pub is not None and self._tmp:
            report_path = os.path.join(self._tmp, REPORT_FILENAME)
            text = self._report_text(summary)
            Path(report_path).write_text(self._mask_remote(text), encoding="utf-8", newline="\n")
            bundle_path = None
            if self._artifact is not None:
                self._artifact.write_report(text)
                if os.path.isfile(str(self.prompts_path)):
                    self._artifact.service_copy(str(self.prompts_path), PROMPTS_FILENAME)
                self._artifact.write_files_list()
                bundle_path = os.path.join(self._tmp, LOG_BUNDLE_NAME)
                self._artifact.make_bundle(bundle_path)
            pub.finish(prompts_path=str(self.prompts_path), report_path=report_path,
                       bundle_path=bundle_path)
            if pub.failures:  # E4: частичный итог, каждая ошибка уже записана одной строкой
                summary.exit_code = EXIT_PARTIAL
        log.info("Итог: готово %d из %d, ошибок %d, не начато %d, арендовано карт %d, "
                 "уничтожено карт %d, среднее время клипа %.1f с, потрачено %.4f USD, код %d.",
                 summary.completed, summary.total_needed, summary.failed, summary.pending,
                 summary.cards_rented, summary.destroyed_cards, summary.avg_render_sec,
                 summary.spent_usd, summary.exit_code)

    # --- пилот ComfyUI (pilot-comfyui.yml) -----------------------------------
    def _run_pilot(self) -> RunSummary:
        """Пилот: одна карта, ожидание статуса running и ответа ComfyUI на порту 18188, замер времени, уборка.

        Итог: ok (HTTP 200), auth_required (HTTP 401/403: ComfyUI отвечает, нужен токен), not_ready (иначе).
        Без missing.txt, Gemini, Release и воркера; весов не загружает. Уборка (destroy_by_label)
        и учёт бюджета — те же методы, что в основном пути. Все строки начинаются с «PILOT».
        """
        summary = RunSummary(total_needed=1)
        t_start = self.clock()
        result = "timeout"
        rent_at: Optional[float] = None
        running_at: Optional[float] = None
        port_at: Optional[float] = None
        slot: Optional[_CardSlot] = None
        last_code: Optional[int] = None
        keys_logged = False
        try:
            # аренда: повторы как в основном пути, пока не истёк срок молчания
            while slot is None:
                if self._stop.is_set():
                    result = "stop"
                    break
                state = self._rent_one(self.clock())
                if state == "ok":
                    slot = next(iter(self._slots.values()))
                    rent_at = slot.created
                elif state in ("budget", "stop", "exhausted"):
                    result = state
                    break
                elif self.clock() - t_start >= self._silent_sec:
                    log.error("PILOT: карту не удалось арендовать за %d мин.", int(self._silent_sec // 60))
                    break
                else:
                    self._sleep(self.rent_retry)

            while slot is not None and rent_at is not None:
                now = self.clock()
                if self._stop.is_set():
                    result = "stop"
                    break
                if now - rent_at >= self._silent_sec and port_at is None:
                    result = "timeout"
                    log.error("PILOT: порт %d не ответил за %d мин после аренды.",
                              PILOT_HTTP_PORT, int(self._silent_sec // 60))
                    log.info("PILOT-МАРКЕР порт %d: HTTP %s через %.0f с после аренды.", PILOT_HTTP_PORT,
                             "нет ответа" if last_code is None else last_code, now - rent_at)
                    break
                try:
                    info = self.vast.get_instance(slot.instance_id)
                except VastError as exc:  # VastInstanceNotFound тоже сюда: опрос повторяется
                    log.warning("PILOT: не удалось получить статус карты %d: %s",
                                slot.instance_id, type(exc).__name__)
                    info = None
                if info is not None:
                    if info.actual_status != slot.vast_status:
                        slot.vast_status = info.actual_status
                        log.info("Карта %d: состояние Vast «%s».", slot.instance_id,
                                 info.actual_status or "неизвестно")
                    if info.actual_status in STOPPED_AFTER_START_STATUSES or info.intended_status == "stopped":
                        result = "card_stopped"
                        break
                    if running_at is None and info.actual_status == "running":
                        running_at = self.clock()
                        log.info("PILOT-МАРКЕР running: карта %d через %.0f с после аренды.",
                                 slot.instance_id, running_at - rent_at)
                    if running_at is not None and info.public_ip:
                        port = _mapped_port(info, PILOT_HTTP_PORT)
                        if port is None:
                            # ключа нет: в лог идут только имена ключей ports, значения не печатаем
                            if not keys_logged:
                                keys_logged = True
                                log.error("PILOT: в поле ports нет ключа %d/tcp. Пришедшие ключи: %s.",
                                          PILOT_HTTP_PORT, _port_keys_text(info))
                        else:
                            self._remember_card_address(info.public_ip, port)
                            code = self._probe_pilot(f"http://{info.public_ip}:{port}")
                            last_code = code
                            verdict = _pilot_verdict(code)
                            if verdict is not None:
                                port_at = self.clock()
                                result = verdict
                                log.info("PILOT-МАРКЕР порт %d: HTTP %s через %.0f с после аренды.",
                                         PILOT_HTTP_PORT, code, port_at - rent_at)
                                break
                            log.debug("PILOT: порт %d пока не отвечает (код %s).", PILOT_HTTP_PORT, code)
                self._sleep(self.poll_interval)
        finally:
            # уборка по метке pilot-<run_id>: та же функция, что у основного пути; карты гасятся всегда
            self._cleanup()

        # замер: интервалы от аренды; None, если этап не наступил
        to_running = (running_at - rent_at) if (running_at is not None and rent_at is not None) else None
        to_port = (port_at - rent_at) if (port_at is not None and rent_at is not None) else None
        answered = result in ("ok", "auth_required")
        verdict = result if answered else "not_ready"
        reason = "" if answered else f"; причина={result}"
        log.info("PILOT-ИТОГ: результат=%s%s; аренда→running=%s с; аренда→ответ=%s с; "
                 "карт арендовано %d, уничтожено %d; уборка %s.", verdict, reason,
                 "нет" if to_running is None else f"{to_running:.0f}",
                 "нет" if to_port is None else f"{to_port:.0f}",
                 self._cards_rented, self._destroyed,
                 "выполнена" if self.cleanup_ok else "НЕ ПОДТВЕРЖДЕНА")
        # ok — успех; auth_required — не провал и не успех (ComfyUI отвечает, нужен токен), код 0
        summary.completed = 1 if result == "ok" else 0
        summary.failed = 0 if answered else 1
        summary.cards_rented = self._cards_rented
        summary.destroyed_cards = self._destroyed
        summary.cleanup_ok = self.cleanup_ok
        summary.spent_usd = self.budget.current_spent(self.clock())
        summary.budget_exceeded = self.budget_exceeded
        summary.exit_code = EXIT_OK if (answered and self.cleanup_ok) else EXIT_PARTIAL
        self._log_budget()
        self._finish_pilot_artifact(summary, verdict, result)
        return summary

    def _finish_pilot_artifact(self, summary: RunSummary, verdict: str, result: str) -> None:
        """Артефакт пилота (SPEC 10.4, этап 7в): run/summary.md и FILES.txt тем же путём, что у основного запуска.
        Файлов, которых пилот не формирует (prompts.json, missing.txt), в service/ нет: пропуск одного файла
        записывается в журнал, артефакт в целом не пропускается. Ошибка записи не меняет итог пилота."""
        if self._artifact is None:
            return
        skipped = []
        try:
            for path, name in ((self.prompts_path, PROMPTS_FILENAME), (self.missing_path, MISSING_FILENAME)):
                if os.path.isfile(str(path)):
                    self._artifact.service_copy(str(path), name)
                else:
                    skipped.append(name)
            if skipped:
                log.info("PILOT: в service/ не попали: %s (пилот эти файлы не формирует).", ", ".join(skipped))
            self._artifact.write_report(self._pilot_report_text(summary, verdict, result))
            self._artifact.write_files_list()
        except OSError as exc:
            log.warning("PILOT: артефакт пилота записан не полностью: %s.", type(exc).__name__)

    def _pilot_report_text(self, summary: RunSummary, verdict: str, result: str) -> str:
        """Итог пилота для run/summary.md: вердикт, счётчики карт, уборка. Адресов и токенов нет."""
        lines = [
            f"# Итог пилота {self.run_id}",
            "",
            f"- результат: {verdict}",
        ]
        if verdict == "not_ready":
            lines.append(f"- причина: {result}")
        lines += [
            f"- карт арендовано: {summary.cards_rented}",
            f"- уничтожено карт: {summary.destroyed_cards}",
            f"- потрачено, USD: {summary.spent_usd:.4f}",
            f"- уборка: {'выполнена' if self.cleanup_ok else 'НЕ ПОДТВЕРЖДЕНА, проверьте Vast.ai'}",
            f"- код завершения: {summary.exit_code}",
        ]
        return "\n".join(lines) + "\n"

    def _probe_pilot(self, base: str) -> Optional[int]:
        """GET PILOT_PROBE_PATH на порту ComfyUI. Возвращает код HTTP или None, если ответа нет."""
        try:
            resp = self._http.get(base + PILOT_PROBE_PATH, timeout=PILOT_PROBE_TIMEOUT_SEC)
        except httpx.HTTPError:
            return None
        return resp.status_code

    # --- промпты -------------------------------------------------------------
    def _read_state(self) -> dict:
        try:
            state = json.loads(self.prompts_path.read_text(encoding="utf-8"))
            if isinstance(state, dict) and isinstance(state.get("prompts"), dict):
                return state
        except (OSError, ValueError):
            pass
        return {"prompts": {}}

    def _generate(self, needed: list) -> list:
        try:
            state = go.generate_and_store(
                self.segments, needed, self.cfg, self.gemini_transport, self.prompts_path,
                profile=self.profile, timing=self.timing_params, **self.gemini_options)
        except gp.GeminiRuntimeError as exc:
            log.error("Генерация промптов прервана: %s. Используем уже сохранённые.", exc)
            state = self._read_state()
        self._state = state
        self._prompts = state.setdefault("prompts", {})
        ready = [n for n in needed if _prompt_ready(self._prompts.get(str(n)))]
        for n in needed:
            if n not in ready:
                self._failed.add(n)
                self._mark(n, "failed")
        if len(ready) < len(needed):
            log.warning("Без готового промпта осталось клипов: %d.", len(needed) - len(ready))
        return ready

    def _save_state(self) -> None:
        try:
            go.atomic_write_json(self.prompts_path, self._state)
        except OSError:
            log.warning("Не удалось сохранить файл промптов.")

    def _mark(self, num: int, status: str) -> None:
        entry = self._prompts.get(str(num))
        if isinstance(entry, dict):
            entry["render_status"] = status
            self._save_state()

    # --- основной цикл -------------------------------------------------------
    def _inflight(self) -> int:
        return sum(1 for s in self._slots.values() if s.task is not None)

    def _work(self, ready: list) -> None:
        self._queue = deque(ready)
        drain_since: Optional[float] = None
        while True:
            if self._stop.is_set():
                # сигнал остановки: новых поисков и аренд нет, дозаливки нет; карты уничтожает _cleanup
                # по метке запуска (как kill-cards) и закрывает их записи бюджета
                log.error("Получен сигнал остановки: новые поиски и аренды не запускаются, "
                          "арендованные карты уничтожаются.")
                self._scan_abort("сигнал остановки")
                break
            now = self.clock()
            deadline_hit = (now - self._t0) >= self._deadline_sec
            if deadline_hit and drain_since is None:
                drain_since = now
                self.deadline_reached = True
                log.warning("Достигнут срок job (%d мин): новые клипы не берём, дозаливаем начатое.",
                            self.job_deadline_min)
                self._scan_abort("срок job")
            if not self.budget_exceeded and self.budget.is_exceeded(now):
                self.budget_exceeded = True
                log.error("Бюджет исчерпан: снимаем все карты, новые не арендуем.")
                for slot in list(self._slots.values()):
                    self._retire(slot, "бюджет исчерпан")
            stopping = deadline_hit or self.budget_exceeded

            for slot in list(self._slots.values()):
                self._step_slot(slot, now, stopping)
            self._pull_due_logs(now)
            self._heartbeat(now)

            inflight = self._inflight()
            if stopping:
                if inflight == 0:
                    break
                if drain_since is not None and now - drain_since >= self.drain_grace:
                    log.error("Время дозаливки вышло: незавершённых клипов %d.", inflight)
                    break
            else:
                if not self._queue and inflight == 0:
                    break
                self._maybe_rent(now)
                if self._rent_exhausted and not self._slots:
                    break  # лимит запросов аренды исчерпан, активных карт нет: завершаем (rent_error)
                if self._budget_blocked and not self._slots:
                    log.error("Бюджет не позволяет арендовать карту, активных карт нет: завершаем.")
                    break
            self._sleep(self.poll_interval)

    # --- аренда --------------------------------------------------------------
    def _desired(self, now: float) -> int:
        remaining = len(self._queue) + self._inflight()
        if remaining <= 0:
            return 0
        avg = sum(self._durations) / len(self._durations) if self._durations else self.render_estimate
        time_left = max(1.0, self._deadline_sec - (now - self._t0))
        need = math.ceil(remaining * avg / (time_left * 0.8))
        return max(1, min(self.max_cards, need, remaining))

    # --- окно сбора цен (SPEC 8.2, v0.6) ---------------------------------------
    def _scan_wanted(self) -> bool:
        return (not self.pilot and not self._scan_done and self._cards_rented == 0
                and self.scan_sec > 0)

    def _scan_open(self, now: float) -> None:
        self._scan_started = now
        self._scan_next = now
        self._scan_end = now + self.scan_sec
        log.info("Окно сбора цен открыто: %d с, шаг опроса %d с, до первой аренды.",
                 int(self.scan_sec), int(self.scan_step))

    def _scan_step_if_due(self, now: float) -> bool:
        """True — если шаг опроса выполнен и пробы записаны."""
        if now < self._scan_next:
            return False
        self._scan_steps += 1
        step_no = self._scan_steps
        # график без накопления дрейфа: следующий шаг через scan_step от запланированного времени
        self._scan_next = max(self._scan_next + self.scan_step, now)
        try:
            offers = self.vast.search_offers(self.offer_filter)
        except VastError as exc:
            self._scan_failed += 1
            log.warning("Окно сбора цен, шаг %d: поиск не удался: %s", step_no, exc)
            return False
        record_scan_samples(self._scan_hist, offers)
        cheapest = min((o.price_per_hr for o in offers), default=None)
        log.info("Окно сбора цен, шаг %d (через %.0f с): найдено %d, минимальная цена %s, "
                 "накоплено уникальных %d.",
                 step_no, now - self._scan_started, len(offers),
                 f"{cheapest:.4f} USD/ч" if cheapest is not None else "нет",
                 len(self._scan_hist))
        return True

    def _stable_ranking(self) -> list:
        return rank_stable_offers(self._scan_hist, self.stable_min_price, self.stable_max_spread)

    def _scan_close(self, now: float) -> list:
        self._scan_done = True
        pool = self._stable_ranking()
        self._scan_hist = {}
        cheapest = pool[0].price_per_hr if pool else None
        log.info("Окно сбора цен закрыто: %.0f с, шагов %d (неуспешных %d), стабильных предложений %d, "
                 "минимальная цена %s.",
                 now - self._scan_started, self._scan_steps, self._scan_failed, len(pool),
                 f"{cheapest:.4f} USD/ч" if cheapest is not None else "нет")
        return pool

    def _scan_abort(self, reason: str) -> None:
        if self._scan_started is None or self._scan_done:
            return
        self._scan_done = True
        self._scan_hist = {}
        log.error("Окно сбора цен прервано (%s): до первой аренды не дошло, карта в этом окне не арендуется.",
                  reason)

    def _maybe_rent(self, now: float) -> None:
        if self._budget_blocked or self._rent_exhausted or now < self._next_rent:
            return
        offers = None  # список окна передаётся только первой аренде запуска
        if self._scan_wanted():
            if self._scan_started is None:
                self._scan_open(now)
            if now < self._scan_end:
                self._scan_step_if_due(now)
                return
            # окно закрыто: если стабильных предложений нет, ждём scan_step_sec и проверяем снова, не останавливаясь
            if self._scan_waiting and not self._scan_step_if_due(now):
                return
            if not self._stable_ranking():
                self._scan_waiting = True
                log.warning("Стабильных предложений нет (проб по %d предложениям), жду %d с и проверю снова.",
                            len(self._scan_hist), int(self.scan_step))
                return
            offers = self._scan_close(now)
        while len(self._slots) < self._desired(now):
            res = self._rent_one(now, offers)
            offers = None  # остальные аренды этого же прохода идут обычным поиском (без паузы)
            if res in ("stop", "exhausted"):
                return
            if res == "retry":
                self._next_rent = now + self._next_wait
                return
            if res == "budget":
                self._budget_blocked = True
                return

    def _note_excluded(self, listed: list) -> None:
        """Одна строка в лог на запуск о предложениях, отброшенных по постоянному списку."""
        if self._excluded_logged:
            return
        self._excluded_logged = True
        ids = sorted({o.machine_id for o in listed})
        log.warning("Предложения отброшены по постоянному списку плохих машин (bad_hosts.txt): "
                    "машины %s, предложений в этом поиске %d.",
                    ", ".join(str(i) for i in ids), len(listed))

    def _rent_one(self, now: float, offers: Optional[list] = None) -> str:
        if self._stop.is_set():
            return "stop"
        if self._rent_requests >= self.max_rent_tries:
            if not self._rent_exhausted:
                log.error("Исчерпан лимит запросов аренды max_rent_tries (%d): новые запросы не отправляются, "
                          "запуск завершается ошибкой аренды (rent_error).", self.max_rent_tries)
            self._rent_exhausted = True
            return "exhausted"
        from_scan = offers is not None  # список окна сбора цен уже получен, поиска не делаем
        if offers is None:
            try:
                offers = self.vast.search_offers(self.offer_filter)
            except VastError as exc:
                log.warning("Поиск предложений Vast не удался: %s", exc)
                self._next_wait = self.rent_retry
                return "retry"
        # постоянный список плохих машин (SPEC 8.2): отсеивается до аренды, по ID машины
        listed = [o for o in offers if o.machine_id in self._excluded_machines]
        if listed:
            self._note_excluded(listed)
            offers = [o for o in offers if o.machine_id not in self._excluded_machines]
        offers = [o for o in offers if o.offer_id not in self._bad_offers and o.offer_id not in self._used_offers
                  and not (o.machine_id and o.machine_id in self._bad_machines)]
        if not offers:
            # нет подходящего непробованного предложения: ждём scan_step_sec и ищем снова (9е-3)
            self._next_wait = self.scan_step
            log.warning("Подходящих предложений Vast нет, повтор позже.")
            return "retry"
        est = self.rent_estimate
        # цена по режиму аренды: бюджет и запрос аренды считаются по одной и той же цене (SPEC 8.2)
        priced = [(o, self._rental_price(o)) for o in offers]
        affordable = [(o, p) for o, p in priced if self.budget.can_afford(p, est, now)]
        if not affordable:
            log.warning("Бюджета не хватает ни на одно предложение (остаток %.4f USD).",
                        self.budget.remaining_budget(now))
            return "budget"
        # лимит max_price_per_hour проверяется до запроса аренды: ставка bid выше лимита — предложение пропускается
        limit = self.offer_filter.max_price
        within = []
        for o, p in affordable:
            if o.is_interruptible and p > limit:
                log.warning("Предложение %d пропущено: ставка bid %.3f USD/ч выше лимита "
                            "max_price_per_hour %.2f USD/ч; аренда не отправлена.", o.offer_id, p, limit)
                continue
            within.append((o, p))
        if not within:
            self._next_wait = self.scan_step  # ни одно предложение не в лимите цены: ждём scan_step_sec (9е-3)
            log.warning("Ни одно подходящее предложение не укладывается в лимит max_price_per_hour "
                        "%.2f USD/ч, повтор позже.", limit)
            return "retry"
        offer, price = within[0]
        if self._stop.is_set():  # сигнал пришёл во время поиска: новая аренда не начинается
            return "stop"
        if self.pilot:
            # пилот: ни токена воркера, ни HF_TOKEN (загрузка весов отключена); открыт только порт ComfyUI
            env = {
                "GEN_LOG_LEVEL": self.log_level,
                f"-p {PILOT_HTTP_PORT}:{PILOT_HTTP_PORT}": "1",
            }
        else:
            env = {
                "WORKER_TOKEN": self._token,
                "WORKER_BUILD": self.worker_build,  # номер коммита: первая строка логов воркера (SPEC 8.4)
                "WORKER_IDLE_TIMEOUT_MIN": str(self.worker_idle_timeout_min),
                "GEN_LOG_LEVEL": self.log_level,  # тот же уровень, что у запуска (SPEC 10)
                # аргументы ComfyUI образа vastai/comfy (SPEC 8.5): порт 18188, без автозапуска браузера
                "COMFYUI_ARGS": COMFYUI_ARGS,
                f"-p {WORKER_HTTP_PORT}:{WORKER_HTTP_PORT}": "1",
            }
            # HF_TOKEN пробрасывается в контейнер для скачивания закрытых моделей LTX-2.5
            # (значение токена не логируется и не печатается).
            hf_token = os.environ.get("HF_TOKEN", "").strip()
            if hf_token:
                env["HF_TOKEN"] = hf_token
        # поле «price» уходит только при прерываемой аренде: ставка bid (та же цена, что в бюджете)
        bid_price = price if offer.is_interruptible else None
        # запрос аренды считается до отправки; предложение больше не выбирается в этом запуске (9е-3)
        self._rent_requests += 1
        self._used_offers.add(offer.offer_id)
        try:
            inst_id = self.vast.create_instance(
                offer.offer_id, self.docker_image, self.disk_gb, env_vars=env, label=self.label,
                bid_price=bid_price, onstart=self.pilot_start_cmd if self.pilot else self._onstart)
        except VastError as exc:
            # ошибка API при аренде: повтор, как и раньше, через rent_retry_sec и со следующим предложением
            self._bad_offers.add(offer.offer_id)
            self._next_wait = self.rent_retry
            log.warning("Не удалось создать карту по предложению %d: %s", offer.offer_id, exc)
            return "retry"
        self.budget.track_instance_start(inst_id, price, now)
        # слоты нумеруются с 1 (карта 1, карта 2, …) до max_cards; поколение слота растёт с 1
        index = next(i for i in range(1, self.max_cards + 1) if i not in self._slots)
        gen = self._gens.get(index, 0) + 1
        self._gens[index] = gen
        slot = _CardSlot(index, gen, inst_id, offer.machine_id,
                         price, now, offer_id=offer.offer_id,
                         bid_requested=offer.is_interruptible,
                         dph_search=offer.dph_total)
        self._slots[index] = slot
        self._cards_rented += 1
        mode = "прерываемая, ставка bid" if offer.is_interruptible else "обычная"
        log.info("Арендована %s по предложению %d (%s): %.4f USD/ч.",
                 _card_name(slot), offer.offer_id, mode, price)
        if from_scan:
            log.info("Окно сбора цен: выбрано предложение %d, dph_total %.4f USD/ч, "
                     "загрузка интернета %.1f Мбит/с; от старта окна до аренды %.0f с.",
                     offer.offer_id, offer.dph_total, offer.inet_down_mbps,
                     self.clock() - self._scan_started)
        log.info("%s создана, ждём загрузку образа.", _card_name(slot, cap=True))
        if self._artifact is not None:  # meta.json без ключей и токенов (SPEC 10.4)
            self._artifact.card_meta(slot, {
                "slot": slot.index, "generation": slot.generation, "instance_id": inst_id,
                "machine_id": offer.machine_id, "offer_id": offer.offer_id,
                "price_per_hr": price, "mode": mode, "created_utc": _utc_now()})
        return "ok"

    def _rental_price(self, offer) -> float:
        """Цена карты для бюджета и запроса аренды (SPEC 8.2).

        Прерываемая аренда: ставка bid = min_bid предложения + bid_margin, округлённая до 3 знаков,
        как PRICE в vast_stable_run.sh (round(BID + MARGIN, 3)). Обычная аренда: dph_total предложения.
        """
        if not offer.is_interruptible:
            return offer.price_per_hr
        return round(float(offer.price_per_hr) + self.bid_margin, 3)

    def _retire(self, slot: _CardSlot, reason: str, *, bad: bool = False) -> None:
        log.warning("%s снимается: %s.", _card_name(slot, cap=True), reason)
        self._card_status(slot, f"снята: {reason}")
        self._budget_blocked = False
        task = slot.task
        slot.task = None  # клип больше не назначен этой карте до возврата в очередь (SPEC 8.2)
        if task is not None:
            if bad:
                # аварийно завершившаяся карта засчитывается как попытка (MAX_ATTEMPTS),
                # иначе «отравленный» клип может зацикливать аренду карт
                self._fail_clip(task.num, f"карта аварийно завершилась: {reason}")
            else:
                self._requeue_free(task.num)
        if bad and slot.machine_id:
            self._bad_machines.add(slot.machine_id)
        self._dump_before_destroy(slot)
        # стоимость карты фиксируется в момент снятия, независимо от исхода уничтожения (SPEC 8.3).
        # Повторное закрытие невозможно: финальная уборка закрывает только открытые записи.
        try:
            self.budget.track_instance_stop(slot.instance_id, self.clock())
        except BudgetError:
            pass
        try:
            self.vast.destroy_instance(slot.instance_id)
        except VastError as exc:
            log.warning("Не удалось уничтожить %s (машина %s): %s. Стоимость зафиксирована "
                        "на этот момент; уничтожение не подтверждено, карта будет проверена при "
                        "финальной уборке.", _card_name(slot), slot.machine_id, exc)
        else:
            self._destroyed += 1
        self._slots.pop(slot.index, None)

    def _dump_before_destroy(self, slot: _CardSlot, timeout: float = LOGS_TIMEOUT_SEC) -> None:
        """Логи карты запрашиваются до уничтожения; без адреса воркера запрос невозможен.

        Любая ошибка здесь гасится: уничтожение карты от логов не зависит.
        """
        try:
            if slot.url:
                self._pull_card_logs(slot, final=True, timeout=timeout)
            else:
                log.info("%s: адрес воркера неизвестен, логи не запрошены.", _card_name(slot, cap=True))
        except Exception as exc:  # уничтожение не должно ждать и падать из-за логов
            log.warning("Не удалось получить логи карты %d: %s.", slot.instance_id, type(exc).__name__)

    def _check_start_failure(self, slot: _CardSlot, prev_status: str, info: InstanceInfo) -> bool:
        """Неудачный старт карты (SPEC 8.2). Возвращает True, если карта снята.

        Судится только после начала старта (вызывающий код проверяет slot.started: «loading» или «created»).
        Признаки, любой из двух:
          - текст статуса (status_msg, строка) содержит маркер START_FAILURE_MARKERS (без учёта регистра,
            по подстроке);
          - статус Vast стал «offline» или «exited».
        Статус «created» сам по себе признаком не является: прогресс («Preparing GPUs...») карту не снимает,
        она ждёт до silent_host_timeout_min. Пустой статус, отсутствующий или нестроковый текст
        признака не дают (fail-closed).
        Снятие идёт через _retire(bad=True): хост и предложение исключаются на весь запуск, отдельного пути нет.
        """
        new_status = info.actual_status
        raw_msg = info.status_msg.strip() if isinstance(info.status_msg, str) else ""
        by_message = bool(raw_msg) and _has_start_marker(raw_msg)
        by_status = new_status in STOPPED_AFTER_START_STATUSES
        if not (by_message or by_status):
            return False
        # маскирование всего текста и сворачивание пробелов (SPEC 0.1)
        text = _one_line(self._mask_remote(raw_msg)) if raw_msg else ""
        detail = f"статус «{prev_status or 'неизвестно'}» → «{new_status or 'неизвестно'}»"
        if text:
            detail += f"; текст: {text}"
        if slot.machine_id:
            host = f"хост {slot.machine_id} исключён из этого запуска"
        else:
            host = "номер хоста неизвестен, исключено только предложение"
        log.error("%s: старт не удался (%s). Карта снимается, %s.",
                  _card_name(slot, cap=True), detail, host)
        if slot.offer_id:
            self._bad_offers.add(slot.offer_id)
        self._retire(slot, START_FAILURE_REASON, bad=True)
        return True

    def _retire_not_found(self, slot: _CardSlot) -> None:
        """Vast не находит карту NOT_FOUND_LIMIT раз подряд (SPEC 8.2, 8.3): снятие как при неудачном старте.

        Хост исключается только на этот запуск (_bad_machines, _bad_offers). bad_hosts.txt не меняется:
        он постоянный и правится вручную.
        """
        log.warning("%s на хосте %s: не найдена в Vast в %d ответах подряд; "
                    "карта снимается, хост исключён из этого запуска.",
                    _card_name(slot, cap=True), slot.machine_id, slot.not_found)
        if slot.offer_id:
            self._bad_offers.add(slot.offer_id)
        self._retire(slot, NOT_FOUND_REASON, bad=True)

    def _check_rental_mode(self, slot: _CardSlot, info: InstanceInfo) -> None:
        """Проверка режима после аренды (SPEC 8.2): карта арендована как bid, но Vast не подтвердил is_bid.

        Это ошибка нашего кода или ответа API. Запуск не прерывается; бюджет считается по dph_total,
        min_bid не используется (см. _adopt_vast_price).
        """
        if not slot.bid_requested or info.is_bid is True:
            return
        slot.mode_mismatch = True
        log.error("%s: запрошен bid, Vast вернул is_bid=%s; бюджет считается по dph_total.",
                  _card_name(slot), info.is_bid)

    def _adopt_vast_price(self, slot: _CardSlot, info: InstanceInfo) -> None:
        """Цена карты по ответу Vast (dph_total) вместо цены поиска (SPEC 8.2, бюджет).

        Вызывается один раз на карту, на первом успешном ответе get_instance. Если dph нулевой или
        отсутствует, остаётся цена поиска и пишется одно предупреждение. При несовпадении режима
        (mode_mismatch) запасная цена — dph_total из поиска, но не min_bid. Иначе ставка открытой записи
        бюджета меняется через set_dph: запись одна, стоимость до этого момента сохраняется.
        Лимит бюджета и расчёт резерва не меняются.
        """
        slot.cost_checked = True
        price = info.dph
        if not price > 0 and slot.mode_mismatch and slot.dph_search > 0:
            price = slot.dph_search
        if not price > 0:
            # текст зависит от режима: предупреждение про bid пишется только для прерываемой аренды
            if slot.bid_requested:
                log.warning("%s: Vast не вернул цену (dph_total) ни в ответе, ни в поиске; "
                            "бюджет считается по ставке bid из поиска, проверьте Billing.",
                            _card_name(slot, cap=True))
            else:
                log.warning("%s: Vast не вернул цену (dph_total) ни в ответе, ни в поиске; "
                            "бюджет обычной аренды считается по dph_total из поиска, проверьте Billing.",
                            _card_name(slot, cap=True))
            return
        if not self.budget.set_dph(slot.instance_id, price, self.clock()):
            log.warning("%s: нет записи учёта бюджета, учитывается цена поиска %.4f USD/ч.",
                        _card_name(slot, cap=True), slot.price)
            return
        slot.price = price

    # --- шаг по карте ---------------------------------------------------------
    def _step_slot(self, slot: _CardSlot, now: float, stopping: bool) -> None:
        age = now - slot.created
        if slot.state == "booting":
            # правила (б) зависания и (в) лимита этапа весов (SPEC 8.2)
            reason = self._monitor(slot).check()
            if reason:
                self._retire_bad(slot, reason)
                return
            info = None
            try:
                info = self.vast.get_instance(slot.instance_id)
            except VastInstanceNotFound as exc:
                # подряд идущие «не найдена» считаются отдельно (SPEC 8.3); остальные ошибки ниже, как раньше
                slot.not_found += 1
                log.warning("Не удалось получить статус %s: %s", _card_name(slot), exc)
                if slot.not_found >= NOT_FOUND_LIMIT:
                    self._retire_not_found(slot)
                    return
            except VastError as exc:
                log.warning("Не удалось получить статус %s: %s", _card_name(slot), exc)
            if info is not None:
                slot.not_found = 0
                if not slot.cost_checked:
                    self._check_rental_mode(slot, info)
                    self._adopt_vast_price(slot, info)
                prev_status = slot.vast_status
                # строка только при смене состояния Vast (загрузка, запуск), не на каждый опрос
                if info.actual_status != slot.vast_status:
                    slot.vast_status = info.actual_status
                    log.info("%s: состояние Vast «%s».", _card_name(slot, cap=True),
                             info.actual_status or "неизвестно")
                    self._card_status(slot, f"состояние Vast «{info.actual_status or 'неизвестно'}»")
                if info.actual_status == "loading":
                    slot.saw_loading = True
                if info.actual_status == "created":
                    slot.saw_created = True
                # перебитие ставки (9е-3, как в vast_stable_run.sh): карта снимается, предложение исключается на запуск
                if self._is_outbid(info):
                    self._retire_outbid(slot)
                    return
                # признаки неудачного старта судятся только после начала старта (loading или created)
                if slot.started and self._check_start_failure(slot, prev_status, info):
                    return
                if info.actual_status == "exited" or info.intended_status == "stopped":
                    self._retire(slot, "карта остановлена хостом", bad=True)
                    return
                if info.actual_status == "running" and info.public_ip and info.direct_port:
                    slot.url = f"http://{info.public_ip}:{info.direct_port}"
                    self._remember_card_address(info.public_ip, info.direct_port)
                    # готовность только после ответа воркера (не по статусу Vast), SPEC 8.2
                    if self._probe_ready(slot):
                        slot.state = "ready"
                        slot.ready_at = now
                        slot.contacted = True
                        slot.net_fail = 0
                        log.info("%s: healthcheck OK через %.0f с после создания, "
                                 "готова к приёму задач.", _card_name(slot, cap=True), age)
                    elif self._track_ready_wait(slot, now):
                        return
        if not slot.contacted and age >= self._silent_sec:
            log.error("%s: healthcheck не прошёл за %d мин (таймаут).",
                      _card_name(slot, cap=True), int(self._silent_sec // 60))
            self._retire(slot, f"хост не вышел на связь за {int(self._silent_sec // 60)} мин", bad=True)
            return
        if slot.net_fail >= MAX_NET_FAILS:
            self._retire(slot, "воркер недоступен", bad=True)
            return
        if slot.state != "ready":
            return
        if slot.task is not None:
            self._poll_task(slot, now)
        if self._slots.get(slot.index) is not slot or slot.task is not None:
            return
        if not stopping and self._queue:
            if not slot.accepted and self._accept_timed_out(slot, now):
                return
            self._dispatch(slot, now)
        elif stopping or not self._queue:
            self._retire(slot, "работы для карты больше нет")

    # --- воркер ----------------------------------------------------------------
    def _call(self, slot: _CardSlot, method: str, path: str, payload: Optional[dict] = None):
        try:
            resp = self._http.request(method, slot.url + path, json=payload,
                                      headers=self._headers, timeout=self.http_timeout)
        except httpx.HTTPError as exc:
            raise _WorkerNetError(type(exc).__name__) from None
        if resp.status_code >= 400:
            raise _WorkerHttpError(resp.status_code)
        return resp

    def _note_unimplemented(self) -> None:
        """SSH и портал не проверяются в 6б (SPEC 8.2, 13.1): одна строка на запуск."""
        if self._probe_notes_logged:
            return
        self._probe_notes_logged = True
        log.info("%s.", SSH_NOT_IMPLEMENTED)
        log.info("%s.", PORTAL_NOT_IMPLEMENTED)

    def _probe_ready(self, slot: _CardSlot) -> bool:
        """Проверка готовности воркера (SPEC 8.2): только GET /health самого воркера.

        Готовность: /health отвечает JSON с ok == true (воркер выставляет его, когда setup.sh создал файл
        готовности, SPEC 8.1). Порт ComfyUI снаружи оркестратор не проверяет (SPEC 8.5).
        Голый код 200 без ok == true не засчитывается. Сбой связи пишется как «запрос не дошёл».
        """
        self._note_unimplemented()
        return self._probe_health(slot)

    def _probe_health(self, slot: _CardSlot) -> bool:
        name = _card_name(slot, cap=True)
        start = self.clock()
        try:
            resp = self._http.get(slot.url + "/health", headers=self._headers, timeout=PROBE_TIMEOUT_SEC)
        except httpx.HTTPError as exc:
            log.warning("%s: проверка /health: запрос не дошёл (%s).", name, type(exc).__name__)
            return False
        data = self._json(resp)
        ok = resp.status_code == 200 and data.get("ok") is True
        body = _one_line(self._mask_remote(resp.text))  # /health короткий: пишется целиком
        log.info("%s: проверка /health: код %d, ответ %s, за %.1f с, %s.", name, resp.status_code,
                 body or "пусто", self.clock() - start, "готов" if ok else "не готов")
        return ok

    def _track_ready_wait(self, slot: _CardSlot, now: float) -> bool:
        """Правило (г), SPEC 8.2: готовность не подтверждена за worker_ready_timeout_min.

        Таймер запускается, когда загрузка весов завершена, или когда строк загрузки нет вовсе
        (тогда таймер идёт от первого адреса карты с public IP). True: карта снята.
        """
        mon = self._monitor(slot)
        if slot.ready_wait_since is None and (mon.stage_done or not mon.seen_start):
            slot.ready_wait_since = now
            if not mon.seen_start:
                self._note_weights_source()
        if slot.ready_wait_since is None or now - slot.ready_wait_since < self._ready_sec:
            return False
        log.error("%s: воркер не готов за %d мин: /health не ответил ok=true. Карта снимается, "
                  "хост исключён из этого запуска.", _card_name(slot, cap=True), int(self._ready_sec // 60))
        self._retire_bad(slot, READY_REASON)
        return True

    def _note_weights_source(self) -> None:
        """Строки загрузки весов не пришли по GET /logs: один раз на запуск (SPEC 8.2, 13.1 п. 5)."""
        if self._weights_source_noted:
            return
        self._weights_source_noted = True
        log.warning("Строки загрузки весов не получены по GET /logs: правила загрузки весов (SPEC 8.2) "
                    "в этом запуске могут не действовать; см. раздел 13.1.")

    def _accept_timed_out(self, slot: _CardSlot, now: float) -> bool:
        """Правило (д), SPEC 8.2: первая задача не принята за task_accept_timeout_min после готовности."""
        if now - slot.ready_at < self._accept_sec:
            return False
        log.error("%s: задача не принята за %d мин после готовности. Карта снимается, хост исключён "
                  "из этого запуска.", _card_name(slot, cap=True), int(self._accept_sec // 60))
        self._retire_bad(slot, ACCEPT_REASON)
        return True

    # --- heartbeat (SPEC 8.3 а) -----------------------------------------------
    def _heartbeat(self, now: float) -> None:
        """Каждые heartbeat_interval_sec: запрос с токеном к каждой карте с адресом, затем файл heartbeat.

        Любой запрос с верным токеном сбрасывает дед-свитч воркера, так что это и есть heartbeat карты.
        Сбой связи пишется в отладочный журнал и на карту не влияет (решают net_fail и таймауты).
        """
        if now - self._last_heartbeat_tick < self._heartbeat_sec:
            return
        self._last_heartbeat_tick = now
        for slot in list(self._slots.values()):
            if not slot.url:
                continue
            slot.last_beat = now
            try:
                resp = self._http.get(slot.url + "/health", headers=self._headers, timeout=PROBE_TIMEOUT_SEC)
            except httpx.HTTPError:
                log.debug("%s: heartbeat: запрос не дошёл.", _card_name(slot, cap=True))
                continue
            if resp.status_code == 200:
                slot.last_ok_utc = _utc_now()
        self._write_heartbeat()

    def _write_heartbeat(self) -> None:
        """Файл heartbeat для наблюдателя (этап 7). Без токенов и без адресов карт (SPEC 0.1)."""
        if self.heartbeat_path is None:
            return
        record = {
            "utc": _utc_now(),
            "label": self.label,
            "cards": [
                {"slot": s.index, "generation": s.generation, "instance_id": s.instance_id,
                 "state": s.state, "last_ok_utc": s.last_ok_utc}
                for s in self._slots.values()
            ],
        }
        try:
            go.atomic_write_json(self.heartbeat_path, record)
        except OSError as exc:
            log.warning("Не удалось записать файл heartbeat: %s.", type(exc).__name__)

    @staticmethod
    def _is_outbid(info) -> bool:
        """Перебитие ставки (9е-3): is_bid true и карта остановлена (cur_state или intended_status), как в vast_stable_run.sh."""
        raw = info.raw if isinstance(info.raw, dict) else {}
        return info.is_bid is True and (raw.get("cur_state") == "stopped" or info.intended_status == "stopped")

    def _retire_outbid(self, slot: _CardSlot) -> None:
        """Снятие карты при перебитии ставки (9е-3): предложение исключается на этот запуск; хост не исключается,
        bad_hosts.txt не пишется. Клип возвращается в очередь без записи попытки (bad=False): вина не в клипе."""
        if slot.offer_id:
            self._bad_offers.add(slot.offer_id)
        self._retire(slot, OUTBID_REASON)

    def _retire_bad(self, slot: _CardSlot, reason: str) -> None:
        """Снятие карты по правилам 8.2 (мониторинг, готовность, принятие задачи): хост и предложение
        исключаются только на этот запуск (bad_hosts.txt не пишется)."""
        if slot.offer_id:
            self._bad_offers.add(slot.offer_id)
        self._retire(slot, reason, bad=True)


    @staticmethod
    def _json(resp) -> dict:
        try:
            data = resp.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def _assigned_slot(self, num: int) -> Optional[_CardSlot]:
        """Слот, на котором сейчас назначен клип (состояние assigned, SPEC 8.2); None, если клип free."""
        for slot in self._slots.values():
            if slot.task is not None and slot.task.num == num:
                return slot
        return None

    def _requeue_free(self, num: int) -> None:
        """Клип возвращается в состояние free: в очередь, если он не done, не назначен и уже не в очереди."""
        if num in self._done or num in self._failed or num in self._queue or self._assigned_slot(num):
            return
        if self._attempts.get(num, 0) > 0:
            self._attempts[num] -= 1
        self._queue.appendleft(num)

    def _fail_clip(self, num: int, reason: str) -> None:
        if num in self._done or self._assigned_slot(num) is not None:
            return  # готовый или назначенный клип не возвращается в очередь
        used = self._attempts.get(num, 0)
        if used >= MAX_ATTEMPTS:
            log.error("Клип %d: попытки исчерпаны (%d), помечен как неудачный. Причина: %s",
                      num, used, reason)
            self._failed.add(num)
            self._mark(num, "failed")
        else:
            log.warning("Клип %d: неудачная попытка %d из %d (%s), возвращён в очередь.",
                        num, used, MAX_ATTEMPTS, reason)
            if num not in self._queue:
                self._queue.append(num)

    def _dispatch(self, slot: _CardSlot, now: float) -> None:
        num = self._queue.popleft()
        owner = self._assigned_slot(num)
        if owner is not None or num in self._done:
            # повторный POST /task для назначенного клипа создал бы дубль на воркере (SPEC 8.2, 13.1 п. 3)
            log.warning("Клип %d уже назначен или готов (%s): повторной отправки нет.", num,
                        _card_name(owner) if owner is not None else "готов")
            return
        entry = self._prompts.get(str(num))
        entry = entry if isinstance(entry, dict) else {}
        num_frames = entry.get("num_frames")
        if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
            # Подставлять значение по умолчанию нельзя: клип сразу считается неудачным.
            self._attempts[num] = MAX_ATTEMPTS
            self._fail_clip(num, "в prompts.json нет корректного num_frames")
            return
        payload = {"num": num, "prompt": entry["prompt"],
                   "num_frames": num_frames, "seed": self.seed_base + num}
        try:
            resp = self._call(slot, "POST", "/task", payload)
        except _WorkerNetError as exc:
            slot.net_fail += 1
            self._queue.appendleft(num)
            log.warning("Воркер %s не принял клип %d (сбой связи: %s).",
                        _card_name(slot), num, exc)
            return
        except _WorkerHttpError as exc:
            slot.net_fail += 1
            self._attempts[num] = self._attempts.get(num, 0) + 1
            self._fail_clip(num, f"воркер ответил {exc}")
            return
        slot.contacted = True
        slot.net_fail = 0
        self._attempts[num] = self._attempts.get(num, 0) + 1
        data = self._json(resp)
        tid = str(data.get("id") or data.get("task_id") or num)
        slot.task = _Task(num, tid, now)
        slot.accepted = True
        # текст промпта не логируем: только длина и короткий SHA-256 (SPEC 0.1)
        prompt = str(payload["prompt"])
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        log.info("Клип %d отправлен: %s (попытка %d, промпт %d симв., sha256 %s).",
                 num, _card_name(slot), self._attempts[num], len(prompt), digest)

    def _poll_task(self, slot: _CardSlot, now: float) -> None:
        task = slot.task
        if now - task.started >= self.task_timeout:
            slot.task = None
            self._fail_clip(task.num, "таймаут рендера")
            self._retire(slot, "воркер завис", bad=True)
            return
        try:
            resp = self._call(slot, "GET", f"/task/{task.id}")
        except _WorkerNetError:
            slot.net_fail += 1
            return
        except _WorkerHttpError as exc:
            slot.task = None
            self._fail_clip(task.num, f"статус задачи: {exc}")
            return
        slot.net_fail = 0
        slot.contacted = True
        data = self._json(resp)
        status = str(data.get("status") or data.get("state") or "").lower()
        if status in _DONE_STATUSES:
            self._finish_clip(slot, now)
        elif status in _FAILED_STATUSES:
            slot.task = None
            err = self._mask_remote(str(data.get("error", "")))  # ошибка воркера целиком, полная маска с карты
            self._fail_clip(task.num, f"воркер сообщил об ошибке: {err}")

    def _download(self, slot: _CardSlot, tid: str, path: str) -> None:
        try:
            with self._http.stream("GET", f"{slot.url}/file/{tid}", headers=self._headers,
                                   timeout=self.http_timeout) as resp:
                if resp.status_code >= 400:
                    raise _WorkerHttpError(resp.status_code)
                with open(path, "wb") as fh:
                    for chunk in resp.iter_bytes():
                        fh.write(chunk)
        except httpx.HTTPError as exc:
            raise _WorkerNetError(type(exc).__name__) from None

    def _finish_clip(self, slot: _CardSlot, now: float) -> None:
        task = slot.task
        path = os.path.join(self._tmp, clip_filename(task.num, self.width, self.template))
        reason = ""
        try:
            self._download(slot, task.id, path)
            self._uploader.upload_clip(path)
        except (_WorkerNetError, _WorkerHttpError) as exc:
            reason = f"не удалось скачать клип с воркера ({exc})"
        except gp.GeminiRuntimeError as exc:
            reason = f"не удалось загрузить клип в Release: {exc}"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        slot.task = None
        if reason:
            self._fail_clip(task.num, reason)
            return
        self._done.add(task.num)
        duration = max(0.0, now - task.started)
        self._durations.append(duration)
        self._mark(task.num, "done")
        log.info("Клип %d выдан в Release (рендер: %s, %.1f с).",
                 task.num, _card_name(slot), duration)
        try:
            self._call(slot, "POST", f"/ack/{task.id}")
        except (_WorkerNetError, _WorkerHttpError):
            log.warning("Воркер не подтвердил удаление клипа %d с диска карты.", task.num)

    # --- финал -------------------------------------------------------------------
    def _cleanup(self) -> None:
        # F1 (1): логи и JSON живых карт до уборки по метке, не дольше ARTIFACT_SAVE_LIMIT_SEC
        deadline = self.clock() + ARTIFACT_SAVE_LIMIT_SEC
        for slot in list(self._slots.values()):
            left = deadline - self.clock()
            if left <= 0:
                log.warning("Лимит %d с на сохранение логов исчерпан: карта %d без выгрузки логов.",
                            int(ARTIFACT_SAVE_LIMIT_SEC), slot.instance_id)
                continue
            self._dump_before_destroy(slot, timeout=min(LOGS_TIMEOUT_SEC, left))
            self._save_instance(slot)
        try:
            ids = self.vast.destroy_by_label(self.label)
            self._destroyed += len(ids)
            log.info("Уничтожено карт по метке %s: %d.", self.label, len(ids))
        except Exception as exc:  # уборка не должна маскировать исходную ошибку
            self.cleanup_ok = False
            log.error("Не удалось уничтожить карты по метке %s: %s", self.label,
                      exc if isinstance(exc, VastError) else type(exc).__name__)
            if isinstance(exc, VastDestroyError):
                # уничтоженные карты не живут: их учёт закрываем так же, как при обычной уборке
                self._destroyed += len(exc.destroyed_ids)
                self._close_budget_records(exc.destroyed_ids, self.clock())
            return
        now = self.clock()
        for rec in self.budget.summary(now)["breakdown"]:
            if rec["active"]:
                try:
                    self.budget.track_instance_stop(rec["instance_id"], now)
                except BudgetError:
                    pass

    def _log_budget(self) -> None:
        """Одна строка в конце запуска: режим аренды, расход по картам и итог (SPEC 10)."""
        now = self.clock()
        rows = self.budget.summary(now)["breakdown"]
        per_card = ", ".join(f"карта {r['instance_id']}: {r['cost_usd']:.4f} USD" for r in rows)
        mode = "прерываемая (bid)" if self.offer_filter.interruptible else "обычная (on-demand)"
        log.info("Бюджет запуска: режим аренды %s; по картам: %s; всего %.4f USD из лимита %.2f USD.",
                 mode, per_card or "карт не арендовано", self.budget.current_spent(now),
                 self.budget.budget_limit_usd)

    def _close_budget_records(self, instance_ids: list[int], now: float) -> None:
        """Закрывает открытые записи бюджета только у перечисленных карт (те же вызовы, что в уборке)."""
        for rec in self.budget.summary(now)["breakdown"]:
            if rec["active"] and rec["instance_id"] in instance_ids:
                try:
                    self.budget.track_instance_stop(rec["instance_id"], now)
                except BudgetError:
                    pass

    def _card_status(self, slot: _CardSlot, text: str) -> None:
        if self._artifact is not None:
            self._artifact.card_status(slot, text)

    def _save_instance(self, slot: _CardSlot) -> None:
        """JSON экземпляра перед уничтожением (SPEC 8.3 в, 10.2); ошибка не мешает уборке."""
        if self._artifact is None:
            return
        try:
            info = self.vast.get_instance(slot.instance_id)
            self._artifact.card_instance(slot, dict(getattr(info, "raw", None) or {}))
        except Exception as exc:
            log.warning("Не удалось сохранить JSON карты %d: %s.", slot.instance_id, type(exc).__name__)


def run_generation(*args, **kwargs) -> RunSummary:
    """Удобная обёртка: создаёт Orchestrator и запускает конвейер."""
    return Orchestrator(*args, **kwargs).run()
