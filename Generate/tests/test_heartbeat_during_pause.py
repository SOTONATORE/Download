"""F2: heartbeat во время пауз между попытками логов Vast (403 и др.), core/orchestrator.py.

Пауза (vast_logs_pause_sec, 30 с) идёт кусками не длиннее VAST_LOGS_PAUSE_SLICE_SEC (10 с); между кусками
и до первой / после последней попытки пишется файл heartbeat. Общая длина паузы и число попыток не меняются.
Часы и сон подставные: тест не ждёт реального времени. Сеть не используется: клиент Vast — подставка.
"""
import json
import pathlib
import sys

try:
    from Generate.core import orchestrator as orch
except ImportError:  # pragma: no cover - плоский импорт
    from core import orchestrator as orch  # type: ignore


class FakeVast:
    """Подставка клиента Vast: request_logs и download_logs. codes — коды ответов загрузки по порядку."""

    def __init__(self, codes):
        self.codes = list(codes)
        self.requests = 0
        self.downloads = 0

    def request_logs(self, instance_id, *, tail=2000, daemon_logs=False):
        self.requests += 1
        return "https://result.example.test/logs/f.txt"

    def download_logs(self, url, dest):
        self.downloads += 1
        code = self.codes.pop(0) if self.codes else 200
        if code == 200:
            with open(dest, "wb") as fh:
                fh.write(b"line one\n")
        return code


class Recorder:
    """Журнал событий по порядку: ('sleep', сек) и ('write', None). Сон двигает подставные часы."""

    def __init__(self):
        self.events = []

    def sleep(self, sec):
        self.events.append(("sleep", float(sec)))


def _slot():
    return orch._CardSlot(index=1, generation=1, instance_id=77, machine_id=5, price=0.1, created=0.0)


def _orc(tmp_path, vast, rec, *, attempts=5, pause=30.0):
    orc = orch.Orchestrator.__new__(orch.Orchestrator)
    orc._token = ""
    orc._card_addresses = set()
    orc.vast = vast
    orc.pilot = False
    orc._artifact = orch.RunArtifact(tmp_path / "artifact", orc._mask_remote)
    orc._vast_logs_attempts = attempts
    orc._vast_logs_pause = pause
    orc._sleep = rec.sleep
    orc._slots = {}
    orc.heartbeat_path = tmp_path / "hb" / "heartbeat.json"
    orc.label = "gen-1"
    real_write = orc._write_heartbeat

    def counting_write():
        rec.events.append(("write", None))
        real_write()          # пишет настоящий файл heartbeat (время текущее)

    orc._write_heartbeat = counting_write
    return orc


def test_thirty_second_pause_writes_heartbeat_at_least_three_times(tmp_path):
    rec = Recorder()
    orc = _orc(tmp_path, FakeVast([403, 200]), rec)
    assert orc._fetch_vast_logs(_slot()) is True
    sleeps = [v for k, v in rec.events if k == "sleep"]
    assert sleeps == [10.0, 10.0, 10.0]                       # одна пауза 30 с, кусками по 10 с
    # записи внутри паузы: в её начале и после каждого куска, т.е. не реже трёх раз на 30 с
    first_sleep = rec.events.index(("sleep", 10.0))
    last_sleep = len(rec.events) - 1 - rec.events[::-1].index(("sleep", 10.0))
    writes_in_pause = [k for k, _ in rec.events[first_sleep - 1:last_sleep + 1] if k == "write"]
    assert len(writes_in_pause) >= 3, rec.events


def test_heartbeat_file_is_written_with_utc_field(tmp_path):
    rec = Recorder()
    orc = _orc(tmp_path, FakeVast([403, 200]), rec)
    orc._fetch_vast_logs(_slot())
    data = json.loads(pathlib.Path(orc.heartbeat_path).read_text(encoding="utf-8"))
    assert "utc" in data and data["label"] == "gen-1"


def test_total_pause_and_attempt_count_are_unchanged(tmp_path):
    rec = Recorder()
    vast = FakeVast([403, 403, 200])
    orc = _orc(tmp_path, vast, rec, attempts=5, pause=30.0)
    assert orc._fetch_vast_logs(_slot()) is True
    sleeps = [v for k, v in rec.events if k == "sleep"]
    assert sum(sleeps) == 60.0                                # две паузы по 30 с, как и раньше
    assert vast.requests == 3 and vast.downloads == 3         # три попытки, как и раньше
    assert vast.codes == []


def test_slices_never_longer_than_ten_seconds_and_pause_zero_adds_none(tmp_path):
    rec = Recorder()
    orc = _orc(tmp_path, FakeVast([403, 200]), rec, pause=7.5)
    orc._fetch_vast_logs(_slot())
    assert [v for k, v in rec.events if k == "sleep"] == [7.5]

    rec2 = Recorder()
    orc2 = _orc(tmp_path, FakeVast([403, 200]), rec2, pause=0.0)
    assert orc2._fetch_vast_logs(_slot()) is True
    assert [v for k, v in rec2.events if k == "sleep"] == []   # без паузы: ни сна, ни лишнего шага
