"""Офлайн-тесты исправлений P1, P2 и P4 (распределение по нужным клипам, фото-профиль, порог наблюдателя).

Без сети и без Vast: распределение подменяется, файлы пишутся только в tmp_path, часы и сон подставные.
Исправление P3 кода не меняет: переключение модели перезапускает ComfyUI (см. отчёт).
"""
import json
from datetime import datetime, timedelta, timezone

try:
    from Generate import cli
    from Generate.core import orchestrator as orch
except ImportError:  # запуск из каталога Generate
    import cli  # type: ignore
    from core import orchestrator as orch  # type: ignore

BASE = datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc)


def _bare_orc(photo_profile, split_result, previews=None):
    """Orchestrator без __init__: распределение подменено, фиксируется аргумент, с которым его вызвали."""
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    o.photo_profile = photo_profile
    o._previews = previews if previews is not None else {}
    o._photo_queue = orch.deque()
    o._video_queue = orch.deque()
    o._split = None
    o._photo_nums = set()
    o.calls = []

    def fake_split(needed):
        o.calls.append(list(needed))
        return split_result(needed)

    o._split_for_run = fake_split
    return o


# --- P1: распределение строится только по нужным клипам ---

def test_split_uses_only_needed_clips(tmp_path):
    # клипы 1 и 2 уже в релизе; распределение должно видеть только 3 и 4
    o = _bare_orc("zimage_turbo", lambda needed: {"video": list(needed), "photo": []})
    o._apply_run_split([3, 4])
    assert o.calls == [[3, 4]]
    assert o._split == {"video": [3, 4], "photo": []}


def test_photo_numbers_come_from_needed_split():
    o = _bare_orc("zimage_turbo", lambda needed: {"video": [needed[0]], "photo": needed[1:]})
    o._apply_run_split([5, 6, 7])
    assert o._photo_nums == {6, 7}
    assert o.photo_profile == "zimage_turbo"  # фото нужны: профиль остаётся


def test_nothing_needed_gives_empty_split_without_calling_split():
    o = _bare_orc("zimage_turbo", lambda needed: {"video": [], "photo": [9]})
    o._apply_run_split([])
    assert o.calls == []                      # распределение не строится, Gemini не вызывается
    assert o._split == {"video": [], "photo": []}


# --- P2: фото-профиль снимается, если фото в запуске нет ---

def test_photo_profile_dropped_when_no_photo_clips():
    o = _bare_orc("zimage_turbo", lambda needed: {"video": list(needed), "photo": []})
    o._apply_run_split([1, 2])
    assert o.photo_profile is None
    assert o._photo_nums == set()
    assert o._profile_for_role("photo") != "zimage_turbo"  # роль фото не получает фото-модель


def test_photo_profile_kept_when_previews_need_it():
    o = _bare_orc("zimage_turbo", lambda needed: {"video": list(needed), "photo": []},
                  previews={-1: object()})
    o._apply_run_split([1, 2])
    assert o.photo_profile == "zimage_turbo"  # превью идут в очередь фото и требуют профиль


def test_video_only_run_never_returns_photo_role():
    o = _bare_orc("zimage_turbo", lambda needed: {"video": list(needed), "photo": []})
    o._apply_run_split([1, 2])
    o._role_plan = None
    o._split = {"video": [1, 2], "photo": []}
    assert o._next_role(0.0) == "video"


def test_tie_with_empty_queues_and_no_photo_is_video():
    # резервный выбор роли: обе очереди пусты, фото в распределении нет -> видео (исправление P2)
    o = _bare_orc("zimage_turbo", lambda needed: {"video": [], "photo": []})
    o._split = {"video": [], "photo": []}
    o._photo_nums = set()
    o._role_plan = []
    o._plan_idx = 0
    assert o._next_role(0.0) == "video"


def test_tie_with_empty_queues_and_photo_in_split_keeps_photo():
    # при наличии фото в распределении поведение прежнее: при равенстве очередей — фото
    o = _bare_orc("zimage_turbo", lambda needed: {"video": [], "photo": [4]})
    o._split = {"video": [], "photo": [4]}
    o._photo_nums = {4}
    o._role_plan = []
    o._plan_idx = 0
    assert o._next_role(0.0) == "photo"


# --- P4: порог тишины наблюдателя для switching / booting ---

class _Clock:
    def __init__(self, tmp_path, cards_state):
        self.t = 0.0
        self.sleeps = 0
        self.kills = []
        self.hb = tmp_path / "hb" / "heartbeat.json"
        self.state = cards_state

    def clock(self):
        return self.t

    def utc_now(self):
        return BASE + timedelta(seconds=self.t)

    def sleep(self, sec):
        self.sleeps += 1
        self.t += sec
        if self.sleeps > 5:  # 5 обходов по 30 с: проверяем окно до 150 с после первого обхода
            raise _Stop()

    def kill(self, run_id):
        self.kills.append(self.t)
        return cli.EXIT_OK

    def write_hb(self, at_sec, state):
        self.hb.parent.mkdir(parents=True, exist_ok=True)
        when = BASE + timedelta(seconds=at_sec)
        record = {"utc": when.strftime(cli.HEARTBEAT_UTC_FORMAT), "label": "gen-123",
                  "cards": [{"slot": 1, "state": state}]}
        self.hb.write_text(json.dumps(record), encoding="utf-8")


class _Stop(Exception):
    pass


def _run(c):
    try:
        return cli.watch_heartbeat("123", c.hb, max_silence_sec=120, check_every_sec=30,
                                   kill=c.kill, clock=c.clock, sleep=c.sleep, utc_now=c.utc_now)
    except _Stop:
        return "stopped"


def test_limit_constants():
    assert cli.WATCH_LONG_STATE_SILENCE_SEC == 900
    assert cli.WATCH_LONG_STATE_SILENCE_SEC == orch.SWITCH_TIMEOUT_SEC
    assert cli.WATCH_LONG_STATES == ("switching", "booting")


def test_switching_card_is_not_killed_while_silence_under_900s(tmp_path):
    c = _Clock(tmp_path, "switching")
    c.write_hb(-300, "switching")             # тишина 300 с при состоянии switching: 120 с был бы превышен
    assert cli._silence_limit_sec(c.hb, 120) == 900
    assert _run(c) == "stopped"               # наблюдатель не уничтожил карты
    assert c.kills == []


def test_booting_card_is_not_killed_while_silence_under_900s(tmp_path):
    c = _Clock(tmp_path, "booting")
    c.write_hb(-300, "booting")
    assert cli._silence_limit_sec(c.hb, 120) == 900
    assert _run(c) == "stopped"
    assert c.kills == []


def test_ready_card_keeps_120s_limit(tmp_path):
    c = _Clock(tmp_path, "ready")
    c.write_hb(-200, "ready")                 # тишина 200 с у карты «ready»: уборка на первом обходе
    assert cli._silence_limit_sec(c.hb, 120) == 120
    assert _run(c) == "stopped"
    assert c.kills == [0.0]


def test_limit_without_cards_field_is_unchanged(tmp_path):
    hb = tmp_path / "heartbeat.json"
    hb.parent.mkdir(parents=True, exist_ok=True)
    hb.write_text(json.dumps({"utc": "2026-11-08T12:00:00Z", "label": "gen-1"}), encoding="utf-8")
    assert cli._silence_limit_sec(hb, 120) == 120


def test_limit_with_unreadable_file_is_unchanged(tmp_path):
    hb = tmp_path / "missing.json"
    assert cli._silence_limit_sec(hb, 120) == 120
