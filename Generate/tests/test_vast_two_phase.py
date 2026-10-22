"""Тесты: периодическая выгрузка логов Vast в два шага (PUT на одном тике, загрузка на следующем).

Сеть не используется: клиент Vast подставной, часы подставные (числа). Канарейка заведомо фальшивая.
time.sleep патчится так, что любой вызов бросает исключение: периодический путь не должен ждать.
"""
import logging

import pytest

try:
    from Generate.core import orchestrator as orch
except ImportError:
    from core import orchestrator as orch  # type: ignore

CANARY_URL = "https://example.invalid/logs?sig=CANARY-12345"
CANARY = "CANARY-12345"


class TwoPhaseVast:
    """Подставка клиента Vast.

    codes: по одному элементу на загрузку; int — код ответа, исключение — бросается из download_logs.
    request_errors: исключения, которые бросает request_logs по порядку (PUT до успешного).
    """

    def __init__(self, codes=(), body=b"log line\n", request_errors=()):
        self.codes = list(codes)
        self.body = body
        self.request_errors = list(request_errors)
        self.requests = []
        self.downloads = []

    def request_logs(self, instance_id, *, tail=2000, daemon_logs=False):
        self.requests.append(instance_id)
        if self.request_errors:
            raise self.request_errors.pop(0)
        return CANARY_URL

    def download_logs(self, url, dest):
        self.downloads.append(url)
        code = self.codes.pop(0) if self.codes else 200
        if isinstance(code, Exception):
            raise code
        if code == 200:
            with open(dest, "wb") as fh:
                fh.write(self.body)
        return code


def _no_sleep(sec):
    raise AssertionError("периодический путь не должен вызывать sleep")


def _bare_orc(tmp_path, vast):
    """Оркестратор без конструктора: только то, что нужно периодике, снятию и маске."""
    orc = orch.Orchestrator.__new__(orch.Orchestrator)
    orc._token = ""
    orc._card_addresses = set()
    orc.vast = vast
    orc._artifact = orch.RunArtifact(tmp_path / "artifact", orc._mask_remote)
    orc._vast_logs_attempts = 5
    orc._vast_logs_pause = 30.0
    orc._sleep = _no_sleep
    orc._slots = {}
    return orc


def _slot(state="booting"):
    slot = orch._CardSlot(index=1, generation=1, instance_id=77, machine_id=5,
                          price=0.1, created=0.0)
    slot.state = state
    return slot


def _card_file(tmp_path):
    return tmp_path / "artifact" / "cards" / "slot1_generation1" / orch.VAST_LOGS_FILE


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)


# ---------------------------------------------------------------- два шага в разных тиках

def test_tick_one_requests_logs_and_does_not_download(tmp_path):
    vast = TwoPhaseVast(codes=[200])
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)
    assert len(vast.requests) == 1
    assert vast.downloads == []                         # загрузки на этом тике нет
    assert slot.vast_logs_phase == "pending"
    assert slot.vast_logs_url == CANARY_URL             # адрес запомнен в памяти
    assert not _card_file(tmp_path).exists()


def test_tick_two_downloads_stored_url_and_200_writes_file(tmp_path):
    vast = TwoPhaseVast(codes=[200], body=b"fresh container line\n")
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)
    orc._pull_silent_vast_logs(10.0)
    assert vast.downloads == [CANARY_URL]               # загрузка сохранённого адреса
    assert len(vast.requests) == 1                      # PUT не повторялся
    assert _card_file(tmp_path).read_text(encoding="utf-8") == "fresh container line\n"
    assert slot.vast_logs_phase == "idle"
    assert slot.vast_logs_url is None


def test_403_on_tick_two_clears_url_and_next_put_is_on_tick_three(tmp_path):
    vast = TwoPhaseVast(codes=[403])
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)                     # PUT
    orc._pull_silent_vast_logs(10.0)                    # 403: логов ещё нет
    assert vast.downloads == [CANARY_URL]
    assert slot.vast_logs_url is None
    assert slot.vast_logs_phase == "idle"
    assert not _card_file(tmp_path).exists()
    orc._pull_silent_vast_logs(20.0)                    # tick 3: новый PUT, без загрузки
    assert len(vast.requests) == 2
    assert len(vast.downloads) == 1
    assert slot.vast_logs_phase == "pending"


def test_second_step_waits_at_least_ten_seconds(tmp_path):
    vast = TwoPhaseVast(codes=[200])
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)
    orc._pull_silent_vast_logs(5.0)                     # интервал не прошёл
    assert vast.downloads == []
    assert slot.vast_logs_phase == "pending"
    orc._pull_silent_vast_logs(10.0)
    assert len(vast.downloads) == 1


def test_failed_put_stays_idle_and_retries_on_a_later_tick(tmp_path):
    vast = TwoPhaseVast(codes=[200], request_errors=[orch.VastRuntimeError("сбой PUT")])
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)                     # PUT не удался
    assert slot.vast_logs_phase == "idle"
    assert slot.vast_logs_url is None
    assert vast.downloads == []
    orc._pull_silent_vast_logs(10.0)                    # повтор PUT на следующем тике
    assert len(vast.requests) == 2
    assert vast.downloads == []
    assert slot.vast_logs_phase == "pending"


def test_network_error_on_download_returns_to_idle(tmp_path):
    vast = TwoPhaseVast(codes=[orch.VastRuntimeError("сеть")])
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)
    orc._pull_silent_vast_logs(10.0)
    assert slot.vast_logs_phase == "idle"
    assert slot.vast_logs_url is None
    assert not _card_file(tmp_path).exists()
    orc._pull_silent_vast_logs(20.0)
    assert len(vast.requests) == 2


# ---------------------------------------------------------------- без ожидания

def test_periodic_path_never_calls_time_sleep(tmp_path, monkeypatch):
    def boom(sec):
        raise AssertionError("time.sleep в периодике")
    monkeypatch.setattr(orch.time, "sleep", boom)
    vast = TwoPhaseVast(codes=[403, 500, 200])          # несколько циклов: 403, 500, успех
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    for t in range(0, 61):
        orc._pull_silent_vast_logs(float(t))
    assert len(vast.downloads) == 3
    assert _card_file(tmp_path).read_text(encoding="utf-8") == "log line\n"


# ---------------------------------------------------------------- остановка

def test_leaving_booting_clears_url(tmp_path):
    vast = TwoPhaseVast(codes=[200])
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)                     # PUT: адрес в памяти
    assert slot.vast_logs_url == CANARY_URL
    slot.state = "ready"                                # так _step_slot отмечает /health ok
    orc._pull_silent_vast_logs(10.0)
    assert vast.downloads == []
    assert slot.vast_logs_url is None
    assert slot.vast_logs_phase == "idle"


def test_retire_clears_url(tmp_path, monkeypatch):
    vast = TwoPhaseVast(codes=[200])
    vast.destroy_instance = lambda instance_id: None
    orc = _bare_orc(tmp_path, vast)

    class _Budget:
        def track_instance_stop(self, instance_id, now):
            return None

    orc.budget = _Budget()
    orc.clock = lambda: 0.0
    orc._budget_blocked = False
    orc._bad_machines = set()
    orc._destroyed = 0
    orc._dump_before_destroy = lambda slot, timeout=0: None
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)                     # PUT: адрес в памяти
    assert slot.vast_logs_url == CANARY_URL
    orc._retire(slot, "тест снятия")
    assert slot.vast_logs_url is None
    assert slot.vast_logs_phase == "idle"
    assert slot.index not in orc._slots


# ---------------------------------------------------------------- адрес не попадает в журнал и файлы

def test_canary_url_never_reaches_logs_or_artifact(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    # ошибки с адресом в тексте: журнал пишет только тип исключения и код, текст не выводится
    vast = TwoPhaseVast(
        codes=[orch.VastRuntimeError(f"сбой загрузки {CANARY_URL}"), 200],
        body=b"container line\n",
        request_errors=[orch.VastRuntimeError(f"отказ PUT {CANARY_URL}")],
    )
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)                     # PUT с ошибкой
    orc._pull_silent_vast_logs(10.0)                    # PUT успешный
    orc._pull_silent_vast_logs(20.0)                    # загрузка с ошибкой сети
    orc._pull_silent_vast_logs(30.0)                    # PUT
    orc._pull_silent_vast_logs(40.0)                    # загрузка 200
    assert _card_file(tmp_path).read_text(encoding="utf-8") == "container line\n"
    assert CANARY not in caplog.text
    assert "example.invalid" not in caplog.text
    assert CANARY not in _card_file(tmp_path).read_text(encoding="utf-8")


def test_403_log_says_logs_not_ready_and_has_no_url(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    vast = TwoPhaseVast(codes=[403])
    orc = _bare_orc(tmp_path, vast)
    slot = _slot()
    orc._slots = {slot.index: slot}
    orc._pull_silent_vast_logs(0.0)
    orc._pull_silent_vast_logs(10.0)
    assert "ещё не готовы" in caplog.text
    assert "не получены после" not in caplog.text
    assert CANARY not in caplog.text
