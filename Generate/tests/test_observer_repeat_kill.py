"""F3: наблюдатель heartbeat не выходит после уборки (cli.watch_heartbeat).

Повторная уборка того же запуска — только после свежего heartbeat и новой тишины; внутри одного эпизода
тишины уборки нет; выход — по метке конца запуска (или по первому-лимиту до первого файла).
Часы и сон подставные: тест не ждёт реального времени.
"""
import json
from datetime import datetime, timedelta, timezone

try:
    from Generate import cli
except ImportError:  # pragma: no cover - плоский импорт
    import cli  # type: ignore

BASE = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, tmp_path):
        self.t = 0.0
        self.sleeps = 0
        self.kill_times = []
        self.hb = tmp_path / "hb" / "heartbeat.json"
        self.on_sleep = None

    def clock(self):
        return self.t

    def utc_now(self):
        return BASE + timedelta(seconds=self.t)

    def sleep(self, sec):
        self.sleeps += 1
        if self.sleeps > 1000:
            raise AssertionError("наблюдатель не завершился")
        self.t += sec
        if self.on_sleep is not None:
            self.on_sleep(self)

    def kill(self, run_id):
        self.kill_times.append(self.t)
        return cli.EXIT_OK

    def write_hb(self, at_sec):
        """Запись heartbeat с временем BASE + at_sec (как пишет оркестратор)."""
        self.hb.parent.mkdir(parents=True, exist_ok=True)
        when = BASE + timedelta(seconds=at_sec)
        self.hb.write_text(json.dumps({"utc": when.strftime(cli.HEARTBEAT_UTC_FORMAT), "label": "gen-123"}),
                           encoding="utf-8")

    def finish(self):
        cli.finished_marker_path(self.hb).write_text("finished\n", encoding="utf-8")


def _run(c):
    return cli.watch_heartbeat("123", c.hb, max_silence_sec=120, check_every_sec=30,
                               kill=c.kill, clock=c.clock, sleep=c.sleep, utc_now=c.utc_now)


def test_observer_keeps_running_after_kill(tmp_path):
    c = Clock(tmp_path)
    c.write_hb(-200)                      # стало сразу: первая уборка на первом обходе

    def on_sleep(x):
        if x.t >= 600:
            x.finish()

    c.on_sleep = on_sleep
    assert _run(c) == 0
    assert c.kill_times == [0.0]          # одна уборка
    assert c.sleeps >= 20                 # после уборки цикл продолжал работать до метки


def test_second_kill_only_after_fresh_heartbeat_and_new_silence(tmp_path):
    c = Clock(tmp_path)
    c.write_hb(-200)

    def on_sleep(x):
        if x.sleeps == 3:                 # t=90: пришёл свежий heartbeat
            x.write_hb(x.t)
        if x.t >= 400:
            x.finish()

    c.on_sleep = on_sleep
    assert _run(c) == 0
    # первая уборка на t=0; вторая — когда после свежей записи (t=90) тишина превысила 120 с: t=240
    assert c.kill_times == [0.0, 240.0], c.kill_times


def test_no_second_kill_while_heartbeat_stays_stale(tmp_path):
    c = Clock(tmp_path)
    c.write_hb(-200)

    def on_sleep(x):
        if x.t >= 900:
            x.finish()

    c.on_sleep = on_sleep
    assert _run(c) == 0
    assert c.kill_times == [0.0]          # без свежего heartbeat повторной уборки нет, сколько бы обходов ни было


def test_no_kill_between_two_kills_of_one_episode(tmp_path):
    """Свежий heartbeat, но тишина короче порога: уборки нет; новая тишина после него — одна уборка."""
    c = Clock(tmp_path)
    c.write_hb(-200)

    def on_sleep(x):
        if x.sleeps == 3:
            x.write_hb(x.t)               # свежий (t=90)
        if x.t >= 200:                    # до t=240 тишина после свежей записи не превышает 120 с
            x.finish()

    c.on_sleep = on_sleep
    assert _run(c) == 0
    assert c.kill_times == [0.0]          # без второй уборки: тишина короче порога
