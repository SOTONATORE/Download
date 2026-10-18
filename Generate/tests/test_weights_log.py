"""Офлайн-тесты разбора строк загрузки весов и правил (SPEC 8.2, этап 6б).

Строки «[setup] ...» повторяют вывод `Generate/worker/setup.sh` (функция log() и download_model).
Формат строк aria2 UNVERIFIED (см. докстринг core/weights_log.py): тесты фиксируют ожидаемый формат,
а не подтверждённый вывод aria2.
Функции без pytest: запускаются и pytest, и `python -m Generate.tests.test_weights_log`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from Generate.core import weights_log as wl
except ImportError:  # pragma: no cover
    from core import weights_log as wl  # type: ignore

NAME = "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
MIB = 1024 ** 2


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _monitor(clock=None, logs=None, **kw):
    clock = clock or Clock()
    logs = logs if logs is not None else []
    params = dict(speed_check_pct=10, speed_min_mbps=400.0, progress_step_pct=10,
                  stall_sec=180.0, stage_max_sec=1200.0)
    params.update(kw)
    return wl.WeightsMonitor(clock=clock, log=lambda text, warn=False: logs.append(text), **params), clock, logs


def _progress(pct: int, speed_mib: float = 500.0) -> str:
    total = 21.0
    done = total * pct / 100.0
    return f"[#2089b0 {done:.2f}GiB/{total:g}GiB({pct}%) CN:16 DL:{speed_mib:g}MiB ETA:40s]"


def _start(name: str = NAME, idx: int = 1, total: int = 5) -> str:
    return f"[setup] [{idx}/{total}] Начинаю загрузку: {name}"


# --- разбор строк setup.sh ---------------------------------------------------

def test_setup_lines_are_parsed_as_written_by_setup_sh():
    assert wl.parse_line(_start()).kind == "start"
    assert wl.parse_line(_start()).name == NAME
    skip = f"[setup] [2/5] Файл уже существует, пропускаю: text_encoders/x.safetensors"
    assert wl.parse_line(skip).kind == "skip"
    done = f"[setup] [1/5] Файл успешно загружен: {NAME} (123456 байт)"
    assert wl.parse_line(done).kind == "done"
    err = f"[setup] [1/5] ОШИБКА: не удалось загрузить файл {NAME} (код 4)."
    ev = wl.parse_line(err)
    assert ev.kind == "error" and ev.name == NAME
    assert wl.parse_line("[setup] Все файлы модели LTX-2.5 на месте.").kind == "stage_done"


def test_unrelated_setup_lines_are_other():
    for line in ("[setup] Запуск контейнера воркера. Уровень логов: INFO.",
                 "[setup] Запускаю ComfyUI (127.0.0.1:8188)...",
                 "[setup] [1/5] ОШИБКА: что-то другое",
                 "просто строка без префикса"):
        assert wl.parse_line(line).kind == "other", line


def test_aria2_line_with_unknown_shape_is_flagged_not_guessed():
    # похоже на строку aria2, но размер не читается: не выдаём процент
    ev = wl.parse_line("[#2089b0 1.2XiB/21GiB(5%) CN:16 DL:450MiB ETA:40s]")
    assert ev.kind == "aria_unknown"
    assert ev.pct == -1


def test_aria2_progress_line_without_eta_is_recognised():
    ev = wl.parse_line("[#2089b0 1.2GiB/21GiB(5%) CN:16 DL:450MiB]")
    assert ev.kind == "progress" and ev.pct == 5


def test_size_and_speed_conversion():
    assert wl.size_to_bytes("1KiB") == 1024
    assert wl.size_to_bytes("450MiB") == 450 * MIB
    # 50 МиБ/с ~ 419 Мбит/с (десятичные мегабиты)
    assert round(wl.bytes_per_sec_to_mbps(50 * MIB)) == 419


# --- прогресс каждые 10% без дублей ---------------------------------------------

def test_progress_logged_at_every_step_without_duplicates():
    mon, clock, logs = _monitor()
    mon.feed([_start()])
    for pct in (3, 3, 5, 9, 10, 10, 11, 15, 19, 20, 20, 33, 47, 60, 99):
        clock.now += 10
        assert mon.feed([_progress(pct)]) is None
    lines = [m for m in logs if "%," in m]
    pcts = [re.search(r"(\d+)%,", m).group(1) for m in lines]
    # 3..9 — без строки; 10 (первый шаг), 20, затем 33, 47, 60 (шаг 70 не достигнут), 99 (последний)
    assert pcts == ["10", "20", "33", "47", "60", "99"]
    assert len(pcts) == len(set(pcts)), "процент не должен повторяться"
    assert "Мбит/с" in lines[0] and "средняя" in lines[0] and "осталось" in lines[0]


def test_same_percent_is_not_logged_twice_and_does_not_move_timer():
    mon, clock, logs = _monitor(stall_sec=60.0)
    mon.feed([_start()])
    clock.now += 5
    mon.feed([_progress(10)])
    before = len(logs)
    for _ in range(5):
        clock.now += 10
        mon.feed([_progress(10)])
    assert len(logs) == before
    # последнее изменение процента было на +5 с; сейчас прошло 55 с (меньше 60): ещё не зависание
    assert mon.check() is None
    clock.now += 20   # 75 с без изменения процента
    assert mon.check() == wl.REASON_STALL


# --- правило (а): скорость на срезе 10% -------------------------------------------

def test_speed_below_threshold_at_10_percent_retires_once():
    mon, clock, logs = _monitor()
    mon.feed([_start()])
    clock.now += 10
    reason = mon.feed([_progress(10, speed_mib=20)])   # ~168 Мбит/с < 400
    assert reason == "скорость загрузки весов ниже порога"


def test_speed_above_threshold_at_10_percent_passes():
    mon, clock, logs = _monitor()
    mon.feed([_start()])
    clock.now += 10
    assert mon.feed([_progress(10, speed_mib=200)]) is None   # ~1677 Мбит/с


def test_speed_is_checked_only_once_per_file():
    mon, clock, logs = _monitor()
    mon.feed([_start()])
    clock.now += 10
    assert mon.feed([_progress(10, speed_mib=200)]) is None
    # позже скорость упала: правило (а) уже использовано для этого файла
    clock.now += 10
    assert mon.feed([_progress(20, speed_mib=1)]) is None


def test_speed_rule_applies_to_next_file_again():
    mon, clock, logs = _monitor()
    mon.feed([_start(NAME, 1)])
    clock.now += 10
    mon.feed([_progress(10, speed_mib=200)])
    mon.feed([f"[setup] [1/5] Файл успешно загружен: {NAME} (1 байт)"])
    mon.feed([_start("vae/a.safetensors", 3)])
    clock.now += 10
    assert mon.feed([_progress(10, speed_mib=1)]) == "скорость загрузки весов ниже порога"


# --- правила (б), (в): зависание и лимит этапа -----------------------------------

def test_stall_retires_when_percent_unchanged_too_long():
    mon, clock, logs = _monitor(stall_sec=180.0)
    mon.feed([_start()])
    clock.now += 170
    mon.feed([_progress(10, speed_mib=200)])
    clock.now += 179
    assert mon.check() is None
    clock.now += 2
    assert mon.check() == "загрузка весов зависла"


def test_stall_does_not_fire_between_files():
    mon, clock, logs = _monitor(stall_sec=60.0)
    mon.feed([_start()])
    mon.feed([f"[setup] [1/5] Файл успешно загружен: {NAME} (1 байт)"])
    clock.now += 600
    assert mon.check() is None   # нет активного файла: зависания нет


def test_stage_max_retires_even_with_progress():
    mon, clock, logs = _monitor(stage_max_sec=1200.0, stall_sec=600.0)
    mon.feed([_start()])
    for step in range(0, 1300, 60):
        clock.now += 60
        mon.feed([_progress(min(99, 10 + step // 60), speed_mib=200)])
    assert mon.check() == "этап загрузки весов превысил лимит"


def test_stage_done_stops_all_timers():
    mon, clock, logs = _monitor(stage_max_sec=100.0, stall_sec=50.0)
    mon.feed([_start()])
    mon.feed([f"[setup] [1/5] Файл успешно загружен: {NAME} (1 байт)"])
    mon.feed(["[setup] Все файлы модели LTX-2.5 на месте."])
    clock.now += 10_000
    assert mon.check() is None


# --- правило (е): ошибка файла ---------------------------------------------------

def test_file_error_line_retires_with_file_name_in_reason():
    mon, clock, logs = _monitor()
    mon.feed([_start()])
    reason = mon.feed([f"[setup] [1/5] ОШИБКА: не удалось загрузить файл {NAME} (код 4)."])
    assert reason == "ошибка загрузки весов: " + NAME


# --- нераспознанный формат --------------------------------------------------------

def test_unrecognised_progress_warns_once_and_gives_no_percent():
    mon, clock, logs = _monitor()
    mon.feed([_start()])
    mon.feed(["[#2089b0 1.2XiB/21GiB(5%) CN:16 DL:450MiB ETA:40s]"])
    mon.feed(["[#2089b0 1.3XiB/21GiB(6%) CN:16 DL:450MiB ETA:40s]"])
    warnings = [m for m in logs if "не распознан" in m]
    assert len(warnings) == 1
    assert not any("%," in m for m in logs), "проценты не должны выдумываться"
    # без распознанного прогресса остаётся только лимит этапа
    clock.now += 10_000
    assert mon.check() == "этап загрузки весов превысил лимит"


# --- запуск как скрипт ----------------------------------------------------------------

def _run_all() -> int:
    failed = 0
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        try:
            globals()[name]()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(names) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
