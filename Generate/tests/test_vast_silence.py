"""Тесты: периодические логи Vast у карты без ответа /health и логи Vast при уничтожении.

Сеть не используется: часы подставные (числа), клиент Vast и клиент воркера работают поверх httpx.MockTransport.
Канареечные значения заведомо фальшивые (адреса из диапазона документации).
"""
import logging

import httpx
import pytest

try:
    from Generate.core import orchestrator as orch
except ImportError:
    from core import orchestrator as orch  # type: ignore

RESULT_URL = "https://result.example.test/logs/f.txt?X-Amz-Signature=URLCANARY-987654"
URL_CANARY = "URLCANARY-987654"
CARD_IP = "203.0.113.77"
CARD_PORT = 47123
CARD_URL = f"http://{CARD_IP}:{CARD_PORT}"


class FakeLogsVast:
    """Подставка клиента Vast: request_logs и download_logs. codes — коды ответов загрузки по порядку."""

    def __init__(self, codes=None, body=b"line one\n"):
        self.codes = list(codes) if codes is not None else []
        self.body = body
        self.requests = []
        self.downloads = []

    def request_logs(self, instance_id, *, tail=2000, daemon_logs=False):
        self.requests.append((instance_id, tail, daemon_logs))
        return RESULT_URL

    def download_logs(self, url, dest):
        self.downloads.append(url)
        code = self.codes.pop(0) if self.codes else 200
        if code == 200:
            with open(dest, "wb") as fh:
                fh.write(self.body)
        return code


class ReentrantVast(FakeLogsVast):
    """Во время первой выгрузки снова вызывает периодический путь (как при повторном заходе цикла)."""

    def __init__(self):
        super().__init__(codes=[200])
        self.orc = None
        self._nested_done = False

    def download_logs(self, url, dest):
        if not self._nested_done and self.orc is not None:
            self._nested_done = True
            self.orc._pull_silent_vast_logs(20.0)  # выгрузка ещё идёт; 20 с — интервал уже прошёл, пропускает busy
        return super().download_logs(url, dest)


def _bare_orc(tmp_path, vast, *, attempts=5, pause=30.0, sleeps=None, http=None):
    """Оркестратор без конструктора: только то, что нужно путям снятия, периодике и маске."""
    orc = orch.Orchestrator.__new__(orch.Orchestrator)
    orc._token = ""
    orc._card_addresses = set()
    orc.vast = vast
    orc.pilot = False  # генерация: __init__ не вызывается, поэтому флаг задаётся явно
    orc._artifact = orch.RunArtifact(tmp_path / "artifact", orc._mask_remote)
    orc._vast_logs_attempts = attempts
    orc._vast_logs_pause = pause
    orc._sleep = (sleeps.append if sleeps is not None else (lambda sec: None))
    orc._slots = {}
    orc._headers = {}
    orc._http = http if http is not None else httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    return orc


def _slot(url="", state="booting"):
    slot = orch._CardSlot(index=1, generation=1, instance_id=77, machine_id=5,
                          price=0.1, created=0.0)
    slot.url = url
    slot.state = state
    return slot


def _card_file(tmp_path):
    return tmp_path / "artifact" / "cards" / "slot1_generation1" / orch.VAST_LOGS_FILE


# ---------------------------------------------------------------- периодика: частота и одна попытка

def test_silence_fetches_about_every_ten_seconds(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    vast = FakeLogsVast(codes=[200] * 10)
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    for t in range(0, 36):                 # шаг шага цикла: 1 с, часы подставные
        orc._pull_silent_vast_logs(float(t))
    # t = 0: PUT; t = 10: загрузка; t = 20: PUT; t = 30: загрузка (два шага в разных тиках)
    assert [r for r in vast.requests] == [(77, orch.VAST_LOGS_TAIL_LINES, False)] * 2
    assert len(vast.downloads) == 2
    assert slot.vast_silence_busy is False


def test_silence_tick_is_one_attempt_without_retry_or_pause(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    sleeps = []
    vast = FakeLogsVast(codes=[500, 500, 500])   # ошибка, а повторов в периодике нет
    orc = _bare_orc(tmp_path, vast, attempts=5, pause=30.0, sleeps=sleeps)
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)              # PUT
    orc._pull_silent_vast_logs(10.0)             # загрузка: одна попытка
    assert len(vast.downloads) == 1
    assert sleeps == []
    assert not _card_file(tmp_path).exists()     # неудача файл не трогает


def test_no_second_fetch_while_one_is_in_progress(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    vast = ReentrantVast()
    orc = _bare_orc(tmp_path, vast)
    vast.orc = orc
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)              # PUT
    orc._pull_silent_vast_logs(10.0)             # загрузка; внутри неё вложенный вход на 20 с
    assert len(vast.downloads) == 1              # вложенный вход пропущен (busy), второй загрузки нет
    assert len(vast.requests) == 1
    assert slot.vast_silence_busy is False       # флаг снят после выгрузки
    orc._pull_silent_vast_logs(20.0)             # следующий тик уже разрешён: PUT
    assert len(vast.requests) == 2


def test_silence_stops_when_health_answers_ok(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    vast = FakeLogsVast(codes=[200] * 5)
    orc = _bare_orc(tmp_path, vast)
    slot = _slot(url=CARD_URL, state="booting")
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)
    assert len(vast.requests) == 1
    slot.state = "ready"                         # так _step_slot отмечает /health ok
    orc._pull_silent_vast_logs(10.0)
    orc._pull_silent_vast_logs(20.0)
    assert len(vast.requests) == 1


def test_silence_stops_when_card_is_retired(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    vast = FakeLogsVast(codes=[200] * 5)
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)
    orc._slots.pop(slot.index)                   # снятие убирает карту из _slots
    orc._pull_silent_vast_logs(10.0)
    assert len(vast.requests) == 1


# ---------------------------------------------------------------- файл: перезапись снимком

def test_artifact_is_overwritten_by_latest_snapshot_and_masked(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    caplog.set_level(logging.DEBUG)
    vast = FakeLogsVast(codes=[200, 200], body=b"old snapshot\n")
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)              # PUT
    orc._pull_silent_vast_logs(10.0)             # загрузка старого снимка
    assert "old snapshot" in _card_file(tmp_path).read_text(encoding="utf-8")
    vast.body = f"new snapshot addr {CARD_IP}:{CARD_PORT} {RESULT_URL}\n".encode("utf-8")
    orc._pull_silent_vast_logs(20.0)             # PUT
    orc._pull_silent_vast_logs(30.0)             # загрузка нового снимка
    text = _card_file(tmp_path).read_text(encoding="utf-8")
    assert "new snapshot" in text and "old snapshot" not in text   # перезапись, не дописывание
    for needle in (CARD_IP, str(CARD_PORT), RESULT_URL, URL_CANARY, "result.example.test"):
        assert needle not in text, f"в файле осталось {needle!r}"
        assert needle not in caplog.text, f"в журнале осталось {needle!r}"


# ---------------------------------------------------------------- уничтожение: логи воркера и Vast

def _worker_http(status=None, exc=None):
    def handler(request):
        if exc is not None:
            raise exc
        return httpx.Response(status, text="")
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_destroy_fetches_vast_when_worker_logs_fail_with_known_address(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    vast = FakeLogsVast(codes=[200], body=b"container line\n")
    orc = _bare_orc(tmp_path, vast, http=_worker_http(status=500))
    orc._dump_before_destroy(_slot(url=CARD_URL))
    assert len(vast.requests) == 1               # адрес известен, но GET /logs не удался: Vast всё равно
    assert _card_file(tmp_path).read_text(encoding="utf-8") == "container line\n"


def test_destroy_fetches_vast_when_worker_logs_raise_unexpected_error(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    vast = FakeLogsVast(codes=[200])
    orc = _bare_orc(tmp_path, vast, http=_worker_http(exc=RuntimeError("boom")))
    orc._dump_before_destroy(_slot(url=CARD_URL))   # не должно упасть наружу
    assert len(vast.requests) == 1                  # сбой логов воркера не пропускает логи Vast


def test_destroy_skips_vast_when_worker_logs_succeed(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    vast = FakeLogsVast(codes=[200])
    orc = _bare_orc(tmp_path, vast, http=_worker_http(status=200))
    orc._http = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text="", headers={"X-Log-Next": "3"})))
    orc._dump_before_destroy(_slot(url=CARD_URL))
    assert vast.requests == []                      # логи воркера получены: Vast не нужен


def test_destroy_without_address_still_fetches_vast(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    vast = FakeLogsVast(codes=[200])
    orc = _bare_orc(tmp_path, vast)
    orc._dump_before_destroy(_slot(url=""))
    assert len(vast.requests) == 1
