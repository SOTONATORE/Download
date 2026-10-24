"""Офлайн-тесты разбора строк загрузки весов и правил (SPEC 8.2, этап 6б).

Строки «[setup] ...» повторяют вывод `Generate/worker/setup.sh` (функция log() и download_model).
Строки aria2 повторяют прогон владельца (aria2c 1.37) с префиксом «[file=<имя>] », который ставит setup.sh.
В фикстурах нет адресов с query-частью. Строки «[setup] ОШИБКА...» префикса не получают (setup.sh их не трогает).
Правила: скорость — агрегат всех файлов за окно (MIN_WINDOW_SEC, MIN_SAMPLES), зависание — по файлу и тоже
только при полном окне. Консоль получает начало этапа, шаги процентов (10%) и «загружены»; подробное — в журнал.
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


def _monitor(clock=None, logs=None, consoles=None, **kw):
    """Возвращает (монитор, часы, журнал, консоль). Журнал: подробные строки; консоль: (вид, проценты)."""
    clock = clock or Clock()
    logs = logs if logs is not None else []
    consoles = consoles if consoles is not None else []
    params = dict(speed_check_pct=10, speed_min_mbps=wl.speed_threshold_mbps(800.0), progress_step_pct=10,
                  stall_sec=180.0, stage_max_sec=1200.0)
    params.update(kw)
    mon = wl.WeightsMonitor(
        clock=clock,
        log=lambda text, warn=False: logs.append(text),
        console=lambda kind, pct=0: consoles.append((kind, pct)),
        **params)
    return mon, clock, logs, consoles


def _progress(pct: int, speed_mib: float = 500.0) -> str:
    total = 21.0
    done = total * pct / 100.0
    return f"[#2089b0 {done:.2f}GiB/{total:g}GiB({pct}%) CN:16 DL:{speed_mib:g}MiB ETA:40s]"


def _aria(done_mib: float, total_mib: float, speed_mib: float, file: str = "") -> str:
    """Строка прогресса aria2 в формате владельца; file — имя без подкаталога (префикс setup.sh)."""
    pct = int(done_mib * 100 // total_mib)
    line = f"[#2089b0 {done_mib:g}MiB/{total_mib:g}MiB({pct}%) CN:16 DL:{speed_mib:g}MiB ETA:40s]"
    return f"[file={file}] {line}" if file else line


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


def test_one_mbit_is_125000_bytes_per_second():
    assert wl.bytes_per_sec_to_mbps(125_000) == 1.0
    assert wl.bytes_per_sec_to_mbps(1 * MIB) * 125_000 == 1 * MIB     # 1 МиБ/с в байтах, без потерь


# --- прогресс и шаги процентов для консоли ---------------------------------------

def test_console_progress_once_per_10_percent_step():
    """Шаги 10..100 по одной строке каждый; повторы процента и шаги внутри 10% не дают строк."""
    mon, clock, logs, consoles = _monitor()
    mon.feed([_start(total=1)])
    for done in range(0, 1001, 5):                     # 1000 МиБ, по 5 МиБ, с повторами процента
        clock.now += 1
        mon.feed([_aria(done, 1000, 1)])
    pcts = [pct for kind, pct in consoles if kind == "progress"]
    assert pcts == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert len(pcts) == len(set(pcts))


def test_console_progress_never_printed_without_known_total():
    """Пока не известны размеры всех файлов этапа, процентов в консоли нет (и «ожидания» тоже)."""
    a_key = NAME.rsplit("/", 1)[-1]
    mon, clock, logs, consoles = _monitor()
    mon.feed([_start(NAME, 1, 2), _start(VAE_NAME, 2, 2)])             # номера известны, размера нет
    clock.now += 5
    mon.feed([_aria(500, 1000, 50, file=a_key)])                     # A 50%, размер B неизвестен
    clock.now += 5
    mon.feed([_aria(1000, 1000, 50, file=a_key)])                    # A 100%, B всё ещё без размера
    assert [pct for kind, pct in consoles if kind == "progress"] == []
    # B получает размер: общий итог 2000 МиБ, сделано 1500 МиБ = 75%, печатается шаг 70% (без повторов)
    mon.feed([_aria(500, 1000, 50, file=VAE_FILE)])
    assert [pct for kind, pct in consoles if kind == "progress"] == [70]
    mon.feed([_aria(1000, 1000, 50, file=VAE_FILE)])
    assert [pct for kind, pct in consoles if kind == "progress"] == [70, 100]


def test_console_start_once_and_loaded_at_stage_end():
    mon, clock, logs, consoles = _monitor()
    mon.feed([_start(NAME, 1, 2), _start(VAE_NAME, 2, 2)])
    mon.feed(["[setup] [1/2] Файл успешно загружен: " + NAME + " (1 байт)",
              "[setup] [2/2] Файл успешно загружен: " + VAE_NAME + " (1 байт)",
              "[setup] Все файлы модели LTX-2.5 на месте."])
    # оба файла полностью на месте: итог 100% печатается один раз до строки «загружены»
    assert consoles == [("start", 0), ("progress", 100), ("loaded", 0)], consoles


# --- правило (а): агрегатная скорость за окно ---------------------------------------

def test_no_retirement_before_window_is_full_even_at_very_low_speed():
    mon, clock, logs, consoles = _monitor()
    mon.feed([_start(total=1)])
    for k in range(1, 10):                              # замеры каждые 10 с: 10..90 с от старта
        clock.now += 10
        assert mon.feed([_aria(10 * k, 1000, 1)]) is None   # 1 МиБ/с — очень мало, но окно не набрано
        assert mon.check() is None
    # первый замер на +10 с, последний на +90 с: окно 80 с < 90 с, решения нет
    assert sum(1 for m in logs if m == wl.WINDOW_WAIT_NOTE) == 1


def test_window_needs_three_samples_not_only_time():
    mon, clock, logs, consoles = _monitor()
    mon.feed([_start(total=1)])
    clock.now += 1
    assert mon.feed([_aria(1, 1000, 1)]) is None                 # замер 1
    clock.now += 95                                              # время набрано, замеров мало
    assert mon.feed([_aria(96, 1000, 1)]) is None                # замер 2: окно не набрано
    # ещё замер: 3 замера и 95 с — окно полное, скорость (96-1) МиБ / 95 с ≈ 1 МиБ/с, ниже порога
    clock.now += 1
    reason = mon.feed([_aria(97, 1000, 1)])
    assert reason == wl.REASON_SPEED


def test_aggregate_speed_decides_not_per_file():
    """Файл A медленный (1 МиБ/с), B быстрый (60 МиБ/с): сумма ~61 МиБ/с (~512 Мбит/с) выше порога 400."""
    a_key = NAME.rsplit("/", 1)[-1]
    mon, clock, logs, consoles = _monitor()
    mon.feed([_start(NAME, 1, 2), _start(VAE_NAME, 2, 2)])
    for k in range(1, 13):                                            # 120 с
        clock.now += 10
        assert mon.feed([_aria(10 * k, 1000, 1, file=a_key),          # A: +10 МиБ за 10 с
                         _aria(600 * k, 9000, 60, file=VAE_FILE)]) is None   # B: +600 МиБ за 10 с
        assert mon.check() is None
    # медленный файл не снимает карту сам по себе: его скорость только в журнале
    assert any(NAME in m and "Мбит/с" in m and "только для журнала" in m for m in logs)

    # те же числа, но сумма ниже порога (оба по 1 МиБ/с): снимается ОДИН раз
    mon2, clock2, logs2, consoles2 = _monitor()
    mon2.feed([_start(NAME, 1, 2), _start(VAE_NAME, 2, 2)])
    reasons = []
    for k in range(1, 13):
        clock2.now += 10
        r = mon2.feed([_aria(10 * k, 1000, 1, file=a_key), _aria(10 * k, 9000, 1, file=VAE_FILE)])
        if r:
            reasons.append(r)
        r = mon2.check()
        if r:
            reasons.append(r)
    assert reasons == [wl.REASON_SPEED], reasons


def test_fast_aggregate_above_threshold_does_not_retire():
    mon, clock, logs, consoles = _monitor()
    mon.feed([_start(total=1)])
    clock.now += 1
    mon.feed([_aria(10, 10000, 200)])
    for k in range(1, 11):
        clock.now += 10
        assert mon.feed([_aria(10 + 500 * k, 10000, 200)]) is None   # ~50 МиБ/с ≈ 419 Мбит/с, выше порога


# --- правило (б): зависание, только при полном окне --------------------------------------

def test_stall_respects_window_and_stays_per_file():
    """Файл A стоит с 1005 с, B идёт быстро. Зависание не раньше полного окна (1005 + 90 с)."""
    mon, clock, logs, consoles = _monitor(stall_sec=60.0, stage_max_sec=10_000.0)
    a_key = NAME.rsplit("/", 1)[-1]
    mon.feed([_start(NAME, 1, 2), _start(VAE_NAME, 2, 2)])
    clock.now = 1005.0
    assert mon.feed([_aria(100, 1000, 0, file=a_key)]) is None      # A: 10% в 1005 с, потом стоит
    window_open = 1005.0
    reason = None
    for k in range(1, 40):
        clock.now = 1005.0 + 10 * k
        assert mon.feed([_aria(600 * k, 9000, 60, file=VAE_FILE)]) is None
        r = mon.check()
        if clock.now - window_open < 90:
            assert r is None, (clock.now, r)                        # окно не набрано: ни зависания, ни скорости
        if r:
            reason = r
            reason_at = clock.now
            break
    assert reason == wl.REASON_STALL
    assert reason_at - window_open >= 90                            # зависание только после полного окна
    stall_logs = [m for m in logs if "не менялся" in m]
    assert len(stall_logs) == 1 and a_key in stall_logs[0] and VAE_FILE not in stall_logs[0]


def test_stall_does_not_fire_between_files():
    mon, clock, logs, consoles = _monitor(stall_sec=60.0)
    mon.feed([_start()])
    mon.feed([f"[setup] [1/5] Файл успешно загружен: {NAME} (1 байт)"])
    clock.now += 600
    assert mon.check() is None   # нет активного файла: зависания нет


# --- правило (в): лимит этапа ------------------------------------------------------------

def test_stage_max_retires_even_with_fast_progress():
    """Скорость 50 МиБ/с (~419 Мбит/с): правило скорости не срабатывает, срабатывает лимит этапа."""
    mon, clock, logs, consoles = _monitor(stage_max_sec=1200.0, stall_sec=600.0)
    mon.feed([_start(total=1)])
    reasons = []
    for k in range(1, 22):
        clock.now += 60
        r = mon.feed([_aria(3000 * k, 100_000, 50)])
        if r:
            reasons.append(r)
        r = mon.check()
        if r:
            reasons.append(r)
    assert reasons == [wl.REASON_STAGE]


def test_stage_done_stops_all_timers():
    mon, clock, logs, consoles = _monitor(stage_max_sec=100.0, stall_sec=50.0)
    mon.feed([_start()])
    mon.feed([f"[setup] [1/5] Файл успешно загружен: {NAME} (1 байт)"])
    mon.feed(["[setup] Все файлы модели LTX-2.5 на месте."])
    clock.now += 10_000
    assert mon.check() is None


# --- правило (е): ошибка файла -----------------------------------------------------------

def test_file_error_line_retires_with_file_name_in_reason():
    mon, clock, logs, consoles = _monitor()
    mon.feed([_start()])
    reason = mon.feed([f"[setup] [1/5] ОШИБКА: не удалось загрузить файл {NAME} (код 4)."])
    assert reason == "ошибка загрузки весов: " + NAME


# --- строки, которые не разобрались, ничего не решают ------------------------------------

def test_unrecognised_progress_like_line_decides_nothing():
    mon, clock, logs, consoles = _monitor()
    mon.feed([_start(total=1)])
    clock.now += 10
    mon.feed(["[#2089b0 1.2XiB/21GiB(5%) CN:16 DL:450MiB ETA:40s]"])     # похоже на прогресс, не разобрано
    clock.now += 200
    assert mon.feed(["[#2089b0 1.3XiB/21GiB(6%) CN:16 DL:450MiB ETA:40s]"]) is None
    clock.now += 100
    assert mon.feed(["[#2089b0 без процента]"]) is None                  # «[#», но без «%»: не участвует
    assert mon.check() is None                                           # ни скорости, ни зависания по ним нет
    warnings = [m for m in logs if "не распознан" in m]
    assert len(warnings) == 1, warnings                                  # предупреждение один раз на этап
    assert [kind for kind, _ in consoles] == ["start"]                  # ни процентов, ни шагов
    assert mon.unrecognised == 2                                         # две строки с «%» посчитаны
    clock.now += 10_000
    assert mon.check() == wl.REASON_STAGE                                # остаётся только лимит этапа


def test_unrecognised_lines_are_counted_only_with_percent_sign():
    mon, clock, logs, consoles = _monitor()
    mon.feed([_start(total=1)])
    mon.feed(["[#2089b0 ??? ]"])                             # «[#», нет «%»: не считается
    assert mon.unrecognised == 0


def test_unprefixed_progress_with_two_active_files_is_ignored():
    mon, clock, logs, consoles = _monitor()
    mon.feed([_start("vae/a.safetensors", 3, 5), _start("vae/b.safetensors", 4, 5)])
    clock.now += 10
    assert mon.feed([_progress(50)]) is None
    assert not any("Веса: файл" in m for m in logs)


# --- средняя скорость и оставшееся время: фиксированные отметки времени -------------------

def test_average_and_remaining_use_fixed_timestamps():
    mon, clock, logs, consoles = _monitor()
    clock.now = 1000.0
    mon.feed([_start(total=1)])                             # этап начинается в 1000 с
    clock.now = 1010.0
    mon.feed([_aria(10, 1000, 1)])                          # окно открыто: замер 1, 10 МиБ
    clock.now = 1050.0
    mon.feed([_aria(50, 1000, 1)])                          # замер 2: окно 40 с — не набрано
    status_1050 = [m for m in logs if m.startswith("Веса: этап:")][-1]
    assert "окно не набрано" in status_1050
    assert "осталось: оценка недоступна" in status_1050
    assert "средняя оценка недоступна" in status_1050       # 50 с < 90 с: среднюю не печатаем
    clock.now = 1100.0
    mon.feed([_aria(100, 1000, 1)])                         # замер 3: окно 90 с, полное
    status_1100 = [m for m in logs if m.startswith("Веса: этап:")][-1]
    assert "загружено 100.0 из 1000.0 МиБ" in status_1100
    assert "скорость 8 Мбит/с" in status_1100              # (100-10) МиБ / 90 с = 1 МиБ/с
    assert "средняя 8 Мбит/с" in status_1100               # 100 МиБ / 100 с = 1 МиБ/с
    assert "осталось ~15.0 мин" in status_1100             # (1000-100) МиБ / 1 МиБ/с = 900 с = 15 мин


def test_average_is_never_negative_and_remaining_needs_total():
    mon, clock, logs, consoles = _monitor()
    clock.now = 1000.0
    mon.feed([_start(NAME, 1, 2)])                          # вторая часть этапа ещё не названа
    clock.now = 1020.0
    mon.feed([_aria(20, 1000, 1)])
    status = [m for m in logs if m.startswith("Веса: этап:")][-1]
    assert "итог неизвестен" in status and "осталось: оценка недоступна" in status
    assert "средняя оценка недоступна" in status             # 20 с < 90 с
    assert "-" not in status.split("средняя")[1].split(",")[0]


# --- строки aria2 с префиксом [file=...] (этап 6б, параллельные загрузки) ----------------

VAE_FILE = "ltx-2.5-video-vae-bf16.safetensors"          # имя без подкаталога, как в префиксе setup.sh
VAE_NAME = "vae/" + VAE_FILE                             # имя с подкаталогом, как в строках «[setup]»

# 8 строк: прогресс с ETA, без ETA, перенаправление, сводка, завершение файла, три формы ошибки.
# Строки «[setup] ОШИБКА...» префикса [file=...] не получают: их печатает setup.sh без aria2-префикса.
FIXTURE = [
    f"[file={VAE_FILE}] [#6df3a8 488MiB/522MiB(93%) CN:3 DL:9.7MiB ETA:3s]",
    f"[file={VAE_FILE}] [#6df3a8 519MiB/522MiB(99%) CN:1 DL:8.1MiB]",
    f"[file={VAE_FILE}] 10/24 14:41:15 [NOTICE] CUID#9 - Redirecting to https://cdn.example.test/ltx/{VAE_FILE}",
    f"[file={VAE_FILE}] *** Download Progress Summary as of 10/24 14:42:07 ***",
    f"[file={VAE_FILE}] 10/24 14:42:07 [NOTICE] Download complete: /opt/worker/models/vae/{VAE_FILE}",
    "[setup] [3/5] ОШИБКА: не удалось загрузить файл vae/x.safetensors (код 4).",
    "[setup] [3/5] ОШИБКА: не удалось загрузить файл vae/x.safetensors (код aria2 3).",
    "[setup] [3/5] ОШИБКА: не удалось загрузить файл vae/x.safetensors (код curl 22).",
]


def test_fixture_is_prefixed_and_has_no_query_strings():
    assert len(FIXTURE) == 8
    assert all("?" not in line for line in FIXTURE)
    for line in FIXTURE[:5]:
        assert line.startswith(f"[file={VAE_FILE}] "), line


def test_fixture_progress_with_and_without_eta():
    with_eta = wl.parse_line(FIXTURE[0])
    no_eta = wl.parse_line(FIXTURE[1])
    assert (with_eta.kind, with_eta.pct, with_eta.file) == ("progress", 93, VAE_FILE)
    assert (no_eta.kind, no_eta.pct, no_eta.file) == ("progress", 99, VAE_FILE)
    assert with_eta.done_bytes == 488 * MIB and with_eta.total_bytes == 522 * MIB
    assert no_eta.done_bytes == 519 * MIB and no_eta.speed_bps == 8.1 * MIB


def test_redirect_and_summary_are_ignored_and_complete_marks_file():
    assert wl.parse_line(FIXTURE[2]).kind == "ignore"
    assert wl.parse_line(FIXTURE[3]).kind == "ignore"
    done = wl.parse_line(FIXTURE[4])
    assert done.kind == "aria_done" and done.name == VAE_FILE

    mon, clock, logs, consoles = _monitor(stall_sec=60.0, stage_max_sec=100_000.0)
    mon.feed([_start(VAE_NAME, 3)])
    clock.now += 5
    mon.feed([FIXTURE[4]])
    clock.now += 10_000
    assert mon.check() is None          # файл завершён: зависать нечему


def test_three_error_forms_give_the_same_error_event():
    for line in FIXTURE[5:8]:
        ev = wl.parse_line(line)
        assert ev.kind == "error" and ev.name == "vae/x.safetensors", line
        mon, clock, logs, consoles = _monitor()
        mon.feed([_start("vae/x.safetensors", 3)])
        assert mon.feed([line]) == "ошибка загрузки весов: vae/x.safetensors", line


def test_two_parallel_files_only_the_stalled_one_is_reported():
    """Файл A стоит, B идёт быстро (~60 МиБ/с): снимается зависание A, и только оно, и не раньше окна."""
    mon, clock, logs, consoles = _monitor(stall_sec=180.0, stage_max_sec=10_000.0)
    a_key = NAME.rsplit("/", 1)[-1]
    mon.feed([_start(NAME, 1, 3), _start(VAE_NAME, 3, 3)])
    clock.now += 5
    assert mon.feed([_aria(100, 1000, 0, file=a_key)]) is None      # A: 10%, потом стоит
    reason = None
    for k in range(1, 40):
        clock.now += 10
        assert mon.feed([_aria(600 * k, 9000, 60, file=VAE_FILE)]) is None
        reason = mon.check()
        if reason:
            break
    assert reason == wl.REASON_STALL
    stall_logs = [m for m in logs if "не менялся" in m]
    assert len(stall_logs) == 1
    assert NAME in stall_logs[0] and VAE_FILE not in stall_logs[0]


def test_kib_and_gib_parse_to_the_same_bytes_as_mib():
    kib = wl.parse_line("[file=x.safetensors] [#a1b2c3 1536KiB/2048KiB(75%) CN:4 DL:512KiB]")
    mib = wl.parse_line("[file=x.safetensors] [#a1b2c3 1.5MiB/2MiB(75%) CN:4 DL:0.5MiB]")
    assert (kib.kind, kib.pct) == ("progress", 75)
    assert kib.done_bytes == mib.done_bytes == 1.5 * MIB
    assert kib.total_bytes == mib.total_bytes == 2 * MIB
    assert kib.speed_bps == mib.speed_bps == 0.5 * MIB
    gib = wl.parse_line("[file=x.safetensors] [#a1b2c3 1GiB/2GiB(50%) CN:4 DL:1GiB ETA:1m20s]")
    mib2 = wl.parse_line("[file=x.safetensors] [#a1b2c3 1024MiB/2048MiB(50%) CN:4 DL:1024MiB]")
    assert gib.done_bytes == mib2.done_bytes == 1024 * MIB
    assert gib.total_bytes == mib2.total_bytes == 2048 * MIB
    assert gib.speed_bps == mib2.speed_bps == 1024 * MIB


def test_query_part_of_an_address_is_dropped_before_parsing():
    # адрес собран в коде теста, в фикстурах query-строк нет
    url = "https://cdn.example.test/ltx/f.bin" + "?" + "X-Amz-Signature=placeholder"
    ev = wl.parse_line(f"[file=f.bin] 10/24 14:41:15 [NOTICE] CUID#9 - Redirecting to {url}")
    assert ev.kind == "ignore"
    assert wl._drop_query(url) == "https://cdn.example.test/ltx/f.bin"


# --- выгрузка как замер; средняя только после окна; нераспознанные строки в журнал (C5) -----------

def test_average_is_not_printed_before_90_seconds_and_is_after_them():
    mon, clock, logs, consoles = _monitor()
    clock.now = 1000.0
    mon.feed([_start(total=1)])
    clock.now = 1005.0
    mon.feed([_aria(2, 1000, 1)])                       # 2 МиБ за 5 с: прежняя формула дала бы ~3 Мбит/с
    assert "средняя оценка недоступна" in [m for m in logs if m.startswith("Веса: этап:")][-1]
    clock.now = 1089.0
    mon.feed([_aria(178, 1000, 1)])                     # 89 с от начала этапа: всё ещё до окна
    assert "средняя оценка недоступна" in [m for m in logs if m.startswith("Веса: этап:")][-1]
    clock.now = 1090.0
    mon.feed([_aria(180, 1000, 1)])                     # 90 с: 180 МиБ / 90 с = 2 МиБ/с ≈ 17 Мбит/с
    status = [m for m in logs if m.startswith("Веса: этап:")][-1]
    assert "средняя 17 Мбит/с" in status
    assert "оценка недоступна" not in status.split("средняя")[1].split(",")[0]


def test_one_pull_with_many_lines_is_one_sample_and_causes_no_start_burst():
    mon, clock, logs, consoles = _monitor()
    clock.now = 1000.0
    mon.feed([_start(total=1)])
    clock.now = 1010.0
    burst = [_aria(mib, 1000, 1) for mib in range(1, 501)]   # 500 строк одного ответа GET /logs
    assert mon.feed(burst) is None
    assert mon.window_samples == 1                           # одна выгрузка — один замер, а не 500
    assert mon.window_first == 1010.0
    assert mon.window_first_bytes == 500 * MIB               # окно открыто с уже накопленного: всплеска скорости нет
    assert len([m for m in logs if m.startswith("Веса: файл")]) == 1   # в журнал — последнее значение файла
    clock.now = 1020.0
    assert mon.feed([_aria(501, 1000, 1)]) is None
    assert mon.window_samples == 2
    assert mon.check() is None                               # окно не набрано: решения нет


def test_unparsed_progress_line_goes_to_log_in_full_once_per_form_and_decides_nothing():
    mon, clock, logs, consoles = _monitor()
    mon.feed([_start(total=1)])
    clock.now += 10
    url = "https://cdn.example.test/ltx/f.bin?X-Amz-Signature=placeholder"
    assert mon.feed([f"[#2089b0 1.2XiB/21GiB(5%) CN:16 DL:450MiB ETA:40s] {url}"]) is None
    clock.now += 200
    assert mon.feed(["[#2089b0 1.3XiB/21GiB(6%) CN:16 DL:450MiB ETA:40s]"]) is None   # та же форма: в журнал не пишется
    forms = [m for m in logs if "полный текст для разбора формата" in m]
    assert len(forms) == 1, forms
    assert "[#2089b0 1.2XiB/21GiB(5%) CN:16 DL:450MiB ETA:40s]" in forms[0]
    assert "<адрес>" in forms[0]
    assert "placeholder" not in forms[0] and "https://" not in forms[0] and "?" not in forms[0]
    assert mon.check() is None                               # ни скорости, ни зависания по ней
    assert [kind for kind, _ in consoles] == ["start"]
    assert mon.unrecognised == 2



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
