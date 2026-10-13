"""Офлайн-тесты телеметрии воркера и вывода ComfyUI (SPEC 8.4, раздел 10). ComfyUI подменён, сети нет."""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest

try:
    from Generate.worker import comfy_output as co
    from Generate.worker import server
    from Generate.worker import telemetry as tm
    from Generate.worker.server import WorkerApp
except ImportError:  # pragma: no cover
    from worker import comfy_output as co
    from worker import server
    from worker import telemetry as tm
    from worker.server import WorkerApp

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
GIB = 1024 ** 3
KIB_PER_GIB = 1024 * 1024
TOKEN = "TELEMETRY-TOKEN-4242"
WORKFLOW = {"1": {"inputs": {}}}
BINDINGS = {
    "prompt": {"node_id": "1", "field": "text"},
    "num_frames": {"node_id": "1", "field": "frames"},
    "seed": {"node_id": "1", "field": "seed"},
    "filename_prefix": {"node_id": "1", "field": "prefix"},
}
TASK = {"num": 7, "prompt": "кот на крыше", "num_frames": 9, "seed": 1}


def make_stats(total=32 * GIB, driver_used=18 * GIB, torch_reserved=8 * GIB, torch_active=6 * GIB):
    """Ответ GET /system_stats ComfyUI с полями, которые даёт comfy/server.py (system_stats)."""
    torch_free = torch_reserved - torch_active          # reserved - active
    driver_free = total - driver_used
    return {"devices": [{
        "name": "FakeGPU", "type": "cuda", "index": 0,
        "vram_total": total,
        "vram_free": driver_free + torch_free,          # mem_free_cuda + mem_free_torch
        "torch_vram_total": torch_reserved,
        "torch_vram_free": torch_free,
    }]}


def write_meminfo(path: Path, total_gib: int = 64, avail_gib: int = 52) -> Path:
    path.write_text(f"MemTotal:       {total_gib * KIB_PER_GIB} kB\n"
                    f"MemFree:        1000 kB\n"
                    f"MemAvailable:   {avail_gib * KIB_PER_GIB} kB\n", encoding="utf-8")
    return path


def write_stat(path: Path, counters) -> Path:
    user, nice, system, idle, iowait, irq, softirq, steal = counters
    path.write_text(f"cpu  {user} {nice} {system} {idle} {iowait} {irq} {softirq} {steal} 0 0\n",
                    encoding="utf-8")
    return path


class FakeComfy:
    def __init__(self, out_dir, stats=None, gate=None):
        self.out_dir = Path(out_dir)
        self.stats = stats if stats is not None else make_stats()
        self.gate = gate
        self._n = 0

    def check_health(self):
        return self.stats

    def queue_prompt(self, graph):
        return "p1"

    def wait_for_completion(self, prompt_id, timeout=600.0, poll_interval=1.0):
        if self.gate is not None:
            self.gate.wait(10)
        self._n += 1
        sub = self.out_dir / "sub"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"clip_{self._n}.mp4").write_bytes(b"0123456789")
        return f"sub/clip_{self._n}.mp4"


@pytest.fixture
def make_app(tmp_path):
    apps = []

    def factory(comfy=None, **kw):
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        comfy = comfy or FakeComfy(out)
        kw.setdefault("idle_timeout_seconds", 3600)
        kw.setdefault("on_idle_timeout", lambda: None)
        app = WorkerApp(TOKEN, comfy, host="127.0.0.1", port=0, workflow=WORKFLOW,
                        bindings=BINDINGS, output_dir=out, **kw)
        app.start_background()
        apps.append(app)
        return app, comfy

    yield factory
    for a in apps:
        a.close()


def client_for(app):
    return httpx.Client(base_url=f"http://127.0.0.1:{app.port}",
                        headers={"X-Worker-Token": TOKEN}, trust_env=False, timeout=10)


def wait_for(predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# Источники и формулы
# ---------------------------------------------------------------------------

def test_gpu_values_follow_comfy_formulas():
    values = tm.gpu_memory_values(make_stats(total=32 * GIB, driver_used=18 * GIB,
                                             torch_reserved=8 * GIB, torch_active=6 * GIB))
    assert values == {"driver_used": 18 * GIB, "total": 32 * GIB,
                      "torch_allocated": 6 * GIB, "torch_reserved": 8 * GIB}


@pytest.mark.parametrize("mutate", [
    lambda d: d["devices"][0].pop("torch_vram_free"),                 # нет поля
    lambda d: d["devices"][0].update(torch_vram_free=9 * GIB),         # torch_free > reserved
    lambda d: d["devices"][0].update(vram_free=40 * GIB),              # драйверная свободная > всего
    lambda d: d["devices"][0].update(vram_total=True),                 # bool не число
    lambda d: d.update(devices=[]),                                    # нет устройств
    lambda d: d.update(devices=["не словарь"]),
])
def test_gpu_values_reject_inconsistent_stats(mutate):
    stats = make_stats()
    mutate(stats)
    with pytest.raises(ValueError):
        tm.gpu_memory_values(stats)


def test_cpu_counters_and_percent_formula_matches_psutil_rule():
    first = tm.parse_cpu_counters(b"cpu  100 0 50 800 50 0 0 0 0 0\n")
    assert first == (1000, 850)                      # idle + iowait
    second = tm.parse_cpu_counters(b"cpu  150 0 80 840 60 0 0 0 0 0\n")
    assert second == (1130, 900)
    # Δвсего=130, Δпростой=50 -> занято 80 из 130
    assert tm.cpu_percent(first, second) == pytest.approx(100.0 * 80 / 130)


def test_cpu_guest_fields_are_not_summed():
    plain = tm.parse_cpu_counters(b"cpu  100 0 50 800 50 0 0 0\n")
    with_guest = tm.parse_cpu_counters(b"cpu  100 0 50 800 50 0 0 0 999 999\n")
    assert plain == with_guest


def test_cpu_first_sample_has_no_percent_and_zero_delta_is_none():
    cur = (1000, 850)
    assert tm.cpu_percent(None, cur) is None          # первый замер: базы нет
    assert tm.cpu_percent(cur, cur) is None           # между замерами не прошло тиков


def test_cpu_parse_rejects_unexpected_format():
    with pytest.raises(ValueError):
        tm.parse_cpu_counters(b"intr 1 2 3\n")
    with pytest.raises(ValueError):
        tm.parse_cpu_counters(b"cpu 1 2\n")


def test_meminfo_parse_and_missing_field(tmp_path):
    assert tm.read_meminfo(str(write_meminfo(tmp_path / "m", 64, 52))) == (
        64 * GIB, 52 * GIB)
    bad = tmp_path / "bad"
    bad.write_text("MemTotal: 100 kB\n", encoding="utf-8")
    with pytest.raises(ValueError):
        tm.read_meminfo(str(bad))


def test_cgroup_limit_parsing(tmp_path):
    v2 = tmp_path / "memory.max"
    v1 = tmp_path / "memory.limit_in_bytes"
    v2.write_text("max\n")
    assert tm.read_cgroup_limit([str(v2)]) is None           # «max» = лимита нет
    v2.write_text("8589934592\n")
    assert tm.read_cgroup_limit([str(v2)]) == 8 * GIB
    assert tm.read_cgroup_limit([str(tmp_path / "нет"), str(v1)]) is None
    v1.write_text("9223372036854771712\n")                   # v1 без лимита: огромное число
    assert tm.read_cgroup_limit([str(tmp_path / "нет"), str(v1)]) == 9223372036854771712


def test_format_helpers():
    assert tm.fmt_gib(18 * GIB) == "18.00 ГиБ"
    assert tm.fmt_gib(int(18.424 * GIB)) == "18.42 ГиБ"
    assert tm.fmt_pct(12.44) == "12.4%"


def test_snapshot_fields_in_russian_no_peak_and_first_cpu_na(tmp_path):
    meminfo = write_meminfo(tmp_path / "meminfo", 64, 52)
    stat = write_stat(tmp_path / "stat", (100, 0, 50, 800, 50, 0, 0, 0))
    sampler = tm.SystemSampler(gpu_stats=make_stats, disk_path=tmp_path,
                               meminfo_path=str(meminfo), stat_path=str(stat),
                               cgroup_paths=[str(tmp_path / "нет")])
    first = sampler.snapshot("тест")
    assert first.startswith("Снимок (тест): ")
    for key in ("видеопамять_занято=18.00 ГиБ", "видеопамять_всего=32.00 ГиБ",
                "PyTorch_выделено=6.00 ГиБ", "PyTorch_зарезервировано=8.00 ГиБ",
                "ОЗУ_доступно=52.00 ГиБ", "ОЗУ_всего=64.00 ГиБ", "ЦП=н/д", "диск_свободно="):
        assert key in first, key
    assert "пик" not in first and "%" not in first.split("ЦП=")[0]

    write_stat(stat, (150, 0, 80, 840, 60, 0, 0, 0))  # Δвсего 130, Δпростой 50
    second = sampler.snapshot("тест")
    assert "ЦП=61.5%" in second


def test_container_ram_limit_shown_only_when_below_host(tmp_path):
    meminfo = write_meminfo(tmp_path / "meminfo", 64, 52)
    stat = write_stat(tmp_path / "stat", (1, 0, 1, 1, 0, 0, 0, 0))
    limit = tmp_path / "memory.max"
    limit.write_text("8589934592\n")
    sampler = tm.SystemSampler(gpu_stats=make_stats, disk_path=tmp_path, meminfo_path=str(meminfo),
                               stat_path=str(stat), cgroup_paths=[str(limit)])
    assert "ОЗУ_лимит_контейнера=8.00 ГиБ" in sampler.snapshot("тест")
    limit.write_text("max\n")
    assert "ОЗУ_лимит_контейнера" not in sampler.snapshot("тест")


def test_failing_sources_are_dropped_warned_once_and_never_raise(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)

    def broken_gpu():
        raise RuntimeError("секретный текст ошибки")

    sampler = tm.SystemSampler(gpu_stats=broken_gpu, disk_path=tmp_path / "нет-такого",
                               meminfo_path=str(tmp_path / "нет-meminfo"),
                               stat_path=str(tmp_path / "нет-stat"),
                               cgroup_paths=[str(tmp_path / "нет")])
    lines = [sampler.snapshot("тест") for _ in range(3)]
    assert all(line == "Снимок (тест): нет данных ни из одного источника." for line in lines)
    assert caplog.text.count("недоступен") == 4          # по одному предупреждению на источник
    assert "RuntimeError" in caplog.text
    assert "секретный текст ошибки" not in caplog.text    # в лог только имя типа исключения


def test_gpu_failure_drops_only_gpu_fields(tmp_path):
    def broken_gpu():
        raise ConnectionError("нет связи")

    meminfo = write_meminfo(tmp_path / "meminfo")
    stat = write_stat(tmp_path / "stat", (1, 0, 1, 1, 0, 0, 0, 0))
    sampler = tm.SystemSampler(gpu_stats=broken_gpu, disk_path=tmp_path, meminfo_path=str(meminfo),
                               stat_path=str(stat), cgroup_paths=[str(tmp_path / "нет")])
    line = sampler.snapshot("тест")
    assert "видеопамять" not in line and "PyTorch" not in line
    assert "ОЗУ_доступно=" in line and "ЦП=" in line and "диск_свободно=" in line


def test_telemetry_module_has_no_torch_import():
    text = (WORKER_DIR / "telemetry.py").read_text(encoding="utf-8")
    assert "import torch" not in text and "from torch" not in text


def test_disk_free_matches_statvfs(tmp_path):
    sampler = tm.SystemSampler(gpu_stats=None, disk_path=tmp_path)
    line = sampler.snapshot("тест")
    assert "диск_свободно=" in line
    free_gib = float(re.search(r"диск_свободно=([0-9.]+) ГиБ", line).group(1))
    assert abs(free_gib - shutil.disk_usage(tmp_path).free / GIB) < 0.5  # одна точка во времени


# ---------------------------------------------------------------------------
# Расписание и очередь (подменный час)
# ---------------------------------------------------------------------------

def test_scheduler_info_one_per_minute_on_injected_clock():
    sampler = tm.TelemetrySampler(lambda reason: reason, "INFO")
    sampler.arm(0.0)
    written = []
    for second in range(0, 301):
        written += [line for line in sampler.step(float(second)) if line]
    assert written == ["по расписанию"] * 5           # 60, 120, 180, 240, 300 с


def test_scheduler_debug_one_per_ten_seconds_on_injected_clock():
    sampler = tm.TelemetrySampler(lambda reason: reason, "DEBUG")
    assert sampler.interval == 10.0
    sampler.arm(0.0)
    written = []
    for second in range(0, 301):
        written += [line for line in sampler.step(float(second)) if line]
    assert written == ["по расписанию"] * 30         # 10, 20, ..., 300 с


def test_queued_events_written_in_order_without_schedule():
    seen = []
    sampler = tm.TelemetrySampler(lambda reason: seen.append(reason) or reason, "INFO")
    sampler.notify("начало клипа 1")
    sampler.notify("конец клипа 1")
    assert sampler.step(0.0) == ["начало клипа 1", "конец клипа 1"]
    assert sampler.step(1.0) == []                   # расписание не взведено


def test_snapshot_exception_warned_once_and_sampling_goes_on(caplog):
    caplog.set_level(logging.DEBUG)
    calls = {"n": 0}

    def flaky(reason):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("сбой")
        return f"ok {reason}"

    sampler = tm.TelemetrySampler(flaky, "INFO")
    sampler.notify("a")
    sampler.notify("b")
    assert sampler.step(0.0) == ["", "ok b"]          # первый снимок пропущен, второй записан
    sampler.notify("c")
    assert sampler.step(0.0) == ["ok c"]
    assert caplog.text.count("Снимок не сформирован") == 1


def test_notify_returns_at_once_while_snapshot_is_slow():
    entered = threading.Event()
    release = threading.Event()

    def slow(reason):
        entered.set()
        release.wait(5)
        return reason

    sampler = tm.TelemetrySampler(slow, "INFO", check_interval=0.05)
    sampler.start()
    try:
        assert entered.wait(5)                        # поток телеметрии застрял в снимке
        t0 = time.monotonic()
        sampler.notify("начало клипа 9")
        assert time.monotonic() - t0 < 0.1
    finally:
        release.set()
        sampler.stop()


# ---------------------------------------------------------------------------
# Воркер: снимки в буфер, ответы HTTP при медленном снимке
# ---------------------------------------------------------------------------

def test_endpoint_answers_while_sample_is_slow(make_app):
    entered = threading.Event()

    def slow_stats():
        entered.set()
        time.sleep(2.0)                               # медленный снимок
        return make_stats()

    app, _ = make_app(telemetry_gpu_stats=slow_stats, telemetry_interval_check=0.05)
    assert entered.wait(5)
    with client_for(app) as c:
        t0 = time.monotonic()
        resp = c.get("/health")
        assert resp.status_code == 200 and resp.json()["ok"] is True
        assert time.monotonic() - t0 < 1.0            # HTTP не ждёт снимка
        tid = c.post("/task", json=TASK).json()["task_id"]
        for _ in range(100):
            if c.get(f"/task/{tid}").json()["state"] == "done":
                break
            time.sleep(0.02)
        assert c.get(f"/task/{tid}").json()["state"] == "done"  # задача не ждёт снимка
    assert wait_for(lambda: any("Снимок (запуск воркера)" in s for s in app.logbuf.tail()), 6.0)


def test_snapshots_on_task_start_and_end_reach_get_logs(make_app):
    app, _ = make_app(telemetry_interval_check=0.05)
    with client_for(app) as c:
        tid = c.post("/task", json=TASK).json()["task_id"]
        for _ in range(200):
            if c.get(f"/task/{tid}").json()["state"] == "done":
                break
            time.sleep(0.02)
        assert wait_for(lambda: any("Снимок (начало клипа 7)" in s for s in app.logbuf.tail()))
        assert wait_for(lambda: any("Снимок (конец клипа 7)" in s for s in app.logbuf.tail()))
        body = c.get("/logs").text
    assert "Снимок (начало клипа 7):" in body and "видеопамять_занято=" in body
    assert "Снимок (запуск воркера):" in body
    assert TOKEN not in body


def test_comfy_output_reaches_buffer_masked_with_marker(make_app, tmp_path):
    comfy_log = tmp_path / "comfyui.log"
    comfy_log.write_text(
        "Total VRAM 32000 MB, total RAM 65000 MB\n"
        "Requested to load LTXVModel\n"
        "\x1b[1m\x1b[32m[INFO]\x1b[0m Unloaded partially: 1.00 MB freed\n"
        f"Authorization: Bearer {TOKEN}\n"
        "ключ sk-TESTKEY123456789 в выводе\n",
        encoding="utf-8")
    app, _ = make_app(comfy_log_path=str(comfy_log))
    assert wait_for(lambda: any("Authorization" in s for s in app.logbuf.tail()))
    lines = app.logbuf.tail()
    joined = "\n".join(lines)
    assert "Вывод ComfyUI: Requested to load LTXVModel" in joined
    assert "Вывод ComfyUI: [INFO] Unloaded partially: 1.00 MB freed" in joined   # ANSI убран
    assert "\x1b" not in joined
    assert TOKEN not in joined and "TESTKEY123456789" not in joined
    assert "Authorization:***" in joined
    assert "ключ *** в выводе" in joined
    with client_for(app) as c:
        body = c.get("/logs").text
    assert "Вывод ComfyUI: Requested to load LTXVModel" in body
    assert TOKEN not in body


def test_comfy_lines_do_not_reach_root_logger(make_app, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    comfy_log = tmp_path / "comfyui.log"
    comfy_log.write_text("Requested to load X\n", encoding="utf-8")
    app, _ = make_app(comfy_log_path=str(comfy_log))
    assert wait_for(lambda: any("Вывод ComfyUI: Requested to load X" in s for s in app.logbuf.tail()))
    assert "Вывод ComfyUI" not in caplog.text          # контейнерный лог получает их от tee, не от воркера
    assert logging.getLogger(server.COMFY_LOG_LOGGER).propagate is False


def test_comfy_logger_restored_on_close(tmp_path):
    comfy_log = tmp_path / "comfyui.log"
    comfy_log.write_text("", encoding="utf-8")
    app = WorkerApp(TOKEN, FakeComfy(tmp_path), host="127.0.0.1", port=0,
                    output_dir=tmp_path, comfy_log_path=str(comfy_log))
    app.start_background()
    assert logging.getLogger(server.COMFY_LOG_LOGGER).propagate is False
    app.close()
    assert logging.getLogger(server.COMFY_LOG_LOGGER).propagate is True


# ---------------------------------------------------------------------------
# Вывод ComfyUI: очистка строк и чтение файла
# ---------------------------------------------------------------------------

def identity(text):
    return text


def test_clean_line_strips_ansi_and_keeps_last_progress_state():
    assert co.clean_comfy_line("\x1b[1m\x1b[32m[INFO]\x1b[0m Requested to load X", identity) == \
        "[INFO] Requested to load X"
    assert co.clean_comfy_line("progress 10%\rprogress 50%\rprogress 100%", identity) == "progress 100%"


def test_clean_line_truncates_to_300_characters():
    out = co.clean_comfy_line("a" * 500, identity)
    assert len(out) == 300 and out.endswith("…")
    assert co.clean_comfy_line("b" * 300, identity) == "b" * 300


def test_clean_line_masks_before_truncation():
    secret = "TESTSECRET-VALUE-XYZ"
    server.register_secret(secret)
    raw = "x" * 296 + secret + " tail"            # граница обрезки (299 символов) попадает внутрь секрета
    out = co.clean_comfy_line(raw, server.mask_secrets)
    # маска сначала: 296 x + «***» + « tail» = 304 символа, усечение до 299 + «…»
    assert out == "x" * 296 + "***…"
    assert "TES" not in out and "XYZ" not in out  # обрывка секрета нет
    assert len(out) == 300


def test_clean_line_masks_key_shapes_and_authorization():
    out = co.clean_comfy_line("Authorization: Bearer hf_abcdefghijklmnopqrstuvwxyz0", server.mask_secrets)
    assert "hf_abcdef" not in out and out.startswith("Authorization:")
    out = co.clean_comfy_line("key sk-TESTKEY123456789 end", server.mask_secrets)
    assert "TESTKEY" not in out and "***" in out


def test_clean_line_skips_blank_lines_in_reader(tmp_path):
    path = tmp_path / "c.log"
    path.write_text("\n   \nnot blank\n", encoding="utf-8")
    got = []
    reader = co.ComfyOutputReader(path, got.append, identity)
    assert reader.poll_once() == 1 and got == ["not blank"]


def test_reader_partial_line_truncation_and_missing_file(tmp_path):
    path = tmp_path / "c.log"
    got = []
    reader = co.ComfyOutputReader(path, got.append, identity)
    assert reader.poll_once() == 0                       # файла ещё нет
    path.write_text("line1\nline2", encoding="utf-8")
    assert reader.poll_once() == 1 and got == ["line1"]  # «line2» ждёт перевода строки
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("-end\n")
    assert reader.poll_once() == 1 and got == ["line1", "line2-end"]
    path.write_text("new\n", encoding="utf-8")           # файл усечён
    assert reader.poll_once() == 1 and got[-1] == "new"


def test_reader_keeps_capturing_after_traceback(tmp_path):
    path = tmp_path / "c.log"
    path.write_text("Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\n"
                    "ValueError: boom\nRequested to load LTXV\n", encoding="utf-8")
    got = []
    reader = co.ComfyOutputReader(path, got.append, identity)
    reader.poll_once()
    assert got == ["Traceback (most recent call last):", "  File \"x.py\", line 1, in <module>",
                   "ValueError: boom", "Requested to load LTXV"]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("Prompt executed in 3.00 seconds\n")
    reader.poll_once()
    assert got[-1] == "Prompt executed in 3.00 seconds"


def test_reader_gives_up_on_endless_line_in_chunks(tmp_path):
    path = tmp_path / "c.log"
    path.write_text("x" * (co.MAX_PENDING + 10), encoding="utf-8")
    got = []
    co.ComfyOutputReader(path, got.append, identity).poll_once()
    assert len(got) == 1 and len(got[0]) == 300 and got[0].endswith("…")


# ---------------------------------------------------------------------------
# Структура: Dockerfile, setup.sh
# ---------------------------------------------------------------------------

def test_dockerfile_copies_every_worker_module():
    flat = (WORKER_DIR / "Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")
    for py in sorted(WORKER_DIR.glob("*.py")):
        if py.name == "__init__.py":
            continue
        assert f"Generate/worker/{py.name}" in flat, f"{py.name} не копируется в образ"
    assert "COMFY_LOG_PATH=/workspace/comfyui.log" in flat


def test_setup_sh_captures_comfy_output_and_is_valid_bash():
    script = WORKER_DIR / "setup.sh"
    text = script.read_text(encoding="utf-8")
    assert "tee -a" in text and "COMFY_LOG_PATH" in text
    assert "--verbose DEBUG" in text and 'GEN_LOG_LEVEL:-INFO}" = "DEBUG"' in text
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash недоступен")
    result = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
