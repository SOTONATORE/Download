"""Телеметрия воркера: снимки ресурсов в лог (SPEC 8.4, раздел 10).

Только стандартная библиотека (SPEC 0.2). Все сообщения на русском (SPEC 0.5).
PyTorch в процессе воркера не импортируется: воркер не выделяет память на GPU, а память ComfyUI
видна через GET /system_stats самого ComfyUI (см. Generate/reports/telemetry_verification.md).

Источники снимка. Поле, чей источник недоступен, в строку не попадает (и не выдумывается):
- видеопамять: ComfyUI GET /system_stats, devices[0] (vram_total, vram_free, torch_vram_total, torch_vram_free);
- ОЗУ: /proc/meminfo (MemTotal, MemAvailable) и предел cgroup, как читает его сам ComfyUI (comfy/system_memory.py);
- ЦП: /proc/stat, два замера; первый замер даёт только базу;
- диск: shutil.disk_usage (os.statvfs) по пути вывода.
Пик памяти за клип в снимок не попадает: в процессе воркера нет счётчика пика, который можно сбросить.
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("Generate.worker.telemetry")

GIB = 1024 ** 3
INTERVAL_INFO_SEC = 60.0      # уровень INFO: один снимок в минуту
INTERVAL_DEBUG_SEC = 10.0     # уровень DEBUG (GEN_LOG_LEVEL=DEBUG): один снимок каждые 10 с
CHECK_INTERVAL_SEC = 1.0      # как часто поток проверяет расписание и очередь событий
MEMINFO_PATH = "/proc/meminfo"
PROC_STAT_PATH = "/proc/stat"
# Порядок как у ComfyUI: сначала cgroup v2, затем v1
CGROUP_LIMIT_PATHS = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
UNKNOWN = "н/д"
REASON_START = "запуск воркера"
REASON_SCHEDULED = "по расписанию"


# ---------------------------------------------------------------------------
# Форматирование
# ---------------------------------------------------------------------------

def fmt_gib(value_bytes: int) -> str:
    """Байты -> «18.42 ГиБ» (два знака)."""
    return f"{value_bytes / GIB:.2f} ГиБ"


def fmt_pct(value: float) -> str:
    """Проценты с одним знаком: «12.4%»."""
    return f"{value:.1f}%"


# ---------------------------------------------------------------------------
# Чтение источников (только stdlib; ошибки поднимаются, ловит SystemSampler)
# ---------------------------------------------------------------------------

def _int_value(source: dict, key: str) -> int:
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"Поле «{key}» отсутствует или некорректно.")
    return int(value)


def gpu_memory_values(stats: dict) -> Dict[str, int]:
    """Память устройства из ответа ComfyUI GET /system_stats (байты).

    Формулы — из comfy/model_management.py (get_free_memory и get_total_memory):
      vram_free       = свободно_драйвер + (reserved - active)   (torch-процесс ComfyUI)
      torch_vram_free = reserved - active
      torch_vram_total = reserved
    Отсюда: занято драйвером = vram_total - (vram_free - torch_vram_free);
            выделено PyTorch = torch_vram_total - torch_vram_free.
    """
    devices = stats.get("devices")
    if not isinstance(devices, list) or not devices or not isinstance(devices[0], dict):
        raise ValueError("В ответе /system_stats нет устройств.")
    dev = devices[0]
    total = _int_value(dev, "vram_total")
    free = _int_value(dev, "vram_free")
    torch_total = _int_value(dev, "torch_vram_total")
    torch_free = _int_value(dev, "torch_vram_free")
    driver_free = free - torch_free
    if not (0 <= driver_free <= total) or torch_free > torch_total:
        raise ValueError("Поля памяти ComfyUI противоречат друг другу.")
    return {
        "driver_used": total - driver_free,
        "total": total,
        "torch_allocated": torch_total - torch_free,
        "torch_reserved": torch_total,
    }


def read_meminfo(path: str = MEMINFO_PATH) -> Tuple[int, int]:
    """(MemTotal, MemAvailable) в байтах из /proc/meminfo."""
    total = avail = None
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b"MemTotal:"):
                total = _kib_field(line)
            elif line.startswith(b"MemAvailable:"):
                avail = _kib_field(line)
            if total is not None and avail is not None:
                break
    if total is None or avail is None:
        raise ValueError("В /proc/meminfo нет MemTotal или MemAvailable.")
    return total, avail


def _kib_field(line: bytes) -> int:
    parts = line.split()
    if len(parts) < 2 or parts[2:3] not in ([], [b"kB"]):
        raise ValueError("Неожиданный формат строки /proc/meminfo.")
    return int(parts[1]) * 1024


def parse_cpu_counters(line: bytes) -> Tuple[int, int]:
    """(всего тиков, тиков простоя) из первой строки /proc/stat.

    Берутся поля user nice system idle iowait irq softirq steal. Поля guest и guest_nice не
    суммируются: ядро уже учло их в user и nice. Так же считает psutil (_cpu_tot_time, _cpu_busy_time).
    """
    parts = line.split()
    if not parts or parts[0] != b"cpu" or len(parts) < 9:
        raise ValueError("Неожиданный формат первой строки /proc/stat.")
    vals = [int(x) for x in parts[1:9]]
    return sum(vals), vals[3] + vals[4]  # idle + iowait


def read_cpu_counters(path: str = PROC_STAT_PATH) -> Tuple[int, int]:
    with open(path, "rb") as fh:
        return parse_cpu_counters(fh.readline())


def cpu_percent(prev: Optional[Tuple[int, int]], cur: Tuple[int, int]) -> Optional[float]:
    """Загрузка между двумя замерами: 100 * (Δвсего - Δпростой) / Δвсего.

    None, если базы нет (первый замер) или между замерами не прошло тиков.
    """
    if prev is None:
        return None
    total_d = cur[0] - prev[0]
    if total_d <= 0:
        return None
    idle_d = cur[1] - prev[1]
    return max(0.0, min(100.0, 100.0 * (total_d - idle_d) / total_d))


def read_cgroup_limit(paths: Sequence[str] = CGROUP_LIMIT_PATHS) -> Optional[int]:
    """Предел памяти контейнера в байтах; None, если файла нет или лимита нет («max»)."""
    for raw_path in paths:
        try:
            raw = Path(raw_path).read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return None


# ---------------------------------------------------------------------------
# Сборка строки снимка
# ---------------------------------------------------------------------------

class SystemSampler:
    """Собирает одну строку снимка. Ни один источник не может сорвать сборку строки:
    недоступный источник даёт одно предупреждение на имя источника и пропускается."""

    def __init__(self, *, gpu_stats: Optional[Callable[[], dict]], disk_path,
                 meminfo_path: str = MEMINFO_PATH, stat_path: str = PROC_STAT_PATH,
                 cgroup_paths: Sequence[str] = CGROUP_LIMIT_PATHS) -> None:
        self._gpu_stats = gpu_stats
        self._disk_path = str(disk_path)
        self._meminfo_path = meminfo_path
        self._stat_path = stat_path
        self._cgroup_paths = tuple(cgroup_paths)
        self._prev_cpu: Optional[Tuple[int, int]] = None
        self._warned: set = set()
        self._lock = threading.Lock()

    def snapshot(self, reason: str) -> str:
        with self._lock:
            fields: List[str] = []
            sources = (
                ("видеопамять (ComfyUI /system_stats)", self._gpu_fields),
                ("ОЗУ (/proc/meminfo)", self._ram_fields),
                ("ЦП (/proc/stat)", self._cpu_fields),
                ("диск (статистика файловой системы)", self._disk_fields),
            )
            for name, collect in sources:
                try:
                    fields.extend(collect())
                except Exception as exc:  # источник не должен ронять снимок
                    self._warn_once(name, exc)
        body = ", ".join(fields) if fields else "нет данных ни из одного источника"
        return f"Снимок ({reason}): {body}."

    def _warn_once(self, name: str, exc: Exception) -> None:
        if name in self._warned:
            return
        self._warned.add(name)
        log.warning("Снимок: источник «%s» недоступен (%s), поле пропущено.", name, type(exc).__name__)

    def _gpu_fields(self) -> List[str]:
        if self._gpu_stats is None:
            raise ValueError("источник видеопамяти не задан")
        values = gpu_memory_values(self._gpu_stats())
        return [
            f"видеопамять_занято={fmt_gib(values['driver_used'])}",
            f"видеопамять_всего={fmt_gib(values['total'])}",
            f"PyTorch_выделено={fmt_gib(values['torch_allocated'])}",
            f"PyTorch_зарезервировано={fmt_gib(values['torch_reserved'])}",
        ]

    def _ram_fields(self) -> List[str]:
        total, avail = read_meminfo(self._meminfo_path)
        out = [f"ОЗУ_доступно={fmt_gib(avail)}", f"ОЗУ_всего={fmt_gib(total)}"]
        limit = read_cgroup_limit(self._cgroup_paths)
        if limit is not None and limit < total:
            out.append(f"ОЗУ_лимит_контейнера={fmt_gib(limit)}")
        return out

    def _cpu_fields(self) -> List[str]:
        cur = read_cpu_counters(self._stat_path)  # при ошибке база не меняется
        pct = cpu_percent(self._prev_cpu, cur)
        self._prev_cpu = cur
        return [f"ЦП={fmt_pct(pct) if pct is not None else UNKNOWN}"]

    def _disk_fields(self) -> List[str]:
        return [f"диск_свободно={fmt_gib(shutil.disk_usage(self._disk_path).free)}"]


# ---------------------------------------------------------------------------
# Расписание и поток
# ---------------------------------------------------------------------------

class TelemetrySampler:
    """Расписание снимков и очередь событий.

    notify() ставит снимок в очередь и сразу возвращается: его вызывает путь задачи, поэтому
    он не делает ни одного внешнего запроса. Сам снимок пишет поток телеметрии.
    step(now) — одна итерация с явным временем (для тестов с подменным часом).
    """

    def __init__(self, snapshot: Callable[[str], str], level: str, *,
                 clock: Callable[[], float] = time.monotonic,
                 check_interval: float = CHECK_INTERVAL_SEC) -> None:
        self._snapshot = snapshot
        self.interval = INTERVAL_DEBUG_SEC if level == "DEBUG" else INTERVAL_INFO_SEC
        self._clock = clock
        self._check = check_interval
        self._pending: List[str] = []
        self._pending_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._next_due: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._warned = False

    def arm(self, now: float) -> None:
        """Первый плановый снимок через interval от момента now."""
        self._next_due = now + self.interval

    def notify(self, reason: str) -> None:
        with self._pending_lock:
            self._pending.append(reason)
        self._wake.set()

    def step(self, now: float) -> List[str]:
        """Записывает снимки для очереди и, если пора, плановый. Возвращает строки снимков."""
        with self._pending_lock:
            reasons, self._pending = self._pending, []
        written = [self._emit(reason) for reason in reasons]
        if self._next_due is not None and now >= self._next_due:
            written.append(self._emit(REASON_SCHEDULED))
            self._next_due = now + self.interval
        return written

    def _emit(self, reason: str) -> str:
        try:
            line = self._snapshot(reason)
        except Exception as exc:  # снимок не должен ронять ни поток, ни задачу
            if not self._warned:
                self._warned = True
                log.warning("Снимок не сформирован: %s. Снимки продолжаются по расписанию.",
                            type(exc).__name__)
            return ""
        log.info("%s", line)
        return line

    def start(self) -> None:
        if self._thread is not None:
            return
        self.arm(self._clock())
        self.notify(REASON_START)
        self._thread = threading.Thread(target=self._loop, name="telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()  # сброс до шага: событие, пришедшее во время шага, не теряется
            try:
                self.step(self._clock())
            except Exception as exc:  # поток телеметрии не должен завершаться
                if not self._warned:
                    self._warned = True
                    log.warning("Ошибка телеметрии: %s. Снимки продолжаются.", type(exc).__name__)
            self._wake.wait(self._check)
