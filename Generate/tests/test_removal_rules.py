"""Правила снятия карт: Правило A (ComfyUI не ответил за 5 мин по comfy_alive) и Правило B (403 от request_logs).

Оффлайн: HTTP воркера и клиент Vast подставлены, сеть не используется. Значения секретов не нужны.
"""
import logging
import types

import httpx

try:
    from Generate.core import orchestrator as orch
    from Generate.core import vast_client as vc
except ImportError:
    from core import orchestrator as orch  # type: ignore
    from core import vast_client as vc  # type: ignore

WORKER_URL = "http://worker.example.test:8000"
INSTANCE = 77
RESULT_URL = "https://result.example.test/logs/f.txt"
RULE_A_REASON = "ComfyUI не ответил за 5 мин"


class _HttpStub:
    """Подставка HTTP-клиента воркера: отдаёт заданный ответ либо бросает заданное исключение."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return httpx.Response(200, json=self.answer)


class _Budget:
    def track_instance_stop(self, instance_id, when):
        pass


class _Vast403:
    """Клиент Vast: PUT request_logs отвечает 403; загрузка при вызове не должна была бы понадобиться."""

    def __init__(self):
        self.request_calls = 0
        self.download_calls = 0
        self.destroyed = []

    def request_logs(self, instance_id, *, tail=2000, daemon_logs=False):
        self.request_calls += 1
        raise vc.VastStatusError("доступ запрещён", status=403)

    def download_logs(self, url, dest):
        self.download_calls += 1
        return 200

    def destroy_instance(self, instance_id):
        self.destroyed.append(instance_id)


class _VastDownload:
    """Клиент Vast: запрос логов успешен, загрузка result_url отдаёт коды по порядку."""

    def __init__(self, codes):
        self.codes = list(codes)
        self.request_calls = 0
        self.download_calls = 0

    def request_logs(self, instance_id, *, tail=2000, daemon_logs=False):
        self.request_calls += 1
        return RESULT_URL

    def download_logs(self, url, dest):
        self.download_calls += 1
        code = self.codes.pop(0) if self.codes else 200
        if code == 200:
            with open(dest, "wb") as fh:
                fh.write(b"container log line\n")
        return code


def _orc(tmp_path, *, http=None, vast=None, now=0.0, silent_sec=1800.0):
    """Оркестратор без конструктора: только то, что трогают проверка /health, шаг карты и снятие."""
    orc = orch.Orchestrator.__new__(orch.Orchestrator)
    orc._http = http
    orc._headers = {}
    orc._token = ""
    orc._card_addresses = set()
    orc.pilot = False
    orc.vast = vast
    orc.clock = lambda: now
    orc._sleep = lambda sec: None
    orc._silent_sec = silent_sec
    orc._vast_logs_attempts = 5
    orc._vast_logs_pause = 30.0
    orc._artifact = orch.RunArtifact(tmp_path / "artifact", orc._mask_remote) if tmp_path else None
    orc._bad_machines = set()
    orc._bad_offers = set()
    orc._slots = {}
    orc._destroyed = 0
    orc._budget_blocked = False
    orc.budget = _Budget()
    orc._card_status = lambda slot, text: None
    orc._lookup_status = lambda slot: (None, None)
    orc._monitor = lambda slot: types.SimpleNamespace(check=lambda: None)
    return orc


def _slot(url=WORKER_URL, created=0.0):
    return orch._CardSlot(index=1, generation=1, instance_id=INSTANCE, machine_id=5,
                          price=0.1, created=created, url=url)


def _step_calls(orc, slot, now):
    """Один шаг карты в состоянии booting; возвращает список снятий (причина, bad)."""
    calls = []
    orc._retire = lambda s, reason, *, bad=False: calls.append((reason, bad))
    orc._step_slot(slot, now, False)
    return calls


# ---------------------------------------------------------------- Правило A

def test_comfy_alive_true_marks_answered_and_rule_a_does_not_fire(tmp_path):
    orc = _orc(tmp_path, http=_HttpStub({"ok": False, "comfy_alive": True}))
    slot = _slot()
    orc._probe_health(slot)
    assert slot.comfy_checked is True and slot.comfy_answered is True
    assert _step_calls(orc, slot, now=400.0) == []


def test_comfy_alive_false_fires_rule_a_at_300_seconds(tmp_path):
    orc = _orc(tmp_path, http=_HttpStub({"ok": False, "comfy_alive": False}))
    slot = _slot()
    orc._probe_health(slot)
    assert slot.comfy_checked is True and slot.comfy_answered is False
    assert _step_calls(orc, slot, now=299.0) == []                       # до 300 с не снимается
    assert _step_calls(orc, slot, now=300.0) == [(RULE_A_REASON, True)]  # на 300 с снимается, bad=True


def test_200_without_comfy_alive_logs_once_and_does_not_fire_rule_a(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    orc = _orc(tmp_path, http=_HttpStub({"ok": False}))
    slot = _slot()
    orc._probe_health(slot)
    orc._probe_health(slot)
    line = f"Rule A: воркер не отдаёт comfy_alive, карта {INSTANCE}"
    assert caplog.text.count(line) == 1                                  # одна строка на карту
    assert slot.comfy_checked is False and slot.comfy_answered is False
    assert _step_calls(orc, slot, now=400.0) == []


def test_connection_error_sets_neither_comfy_field(tmp_path):
    orc = _orc(tmp_path, http=_HttpStub(httpx.ConnectError("нет связи")))
    slot = _slot()
    assert orc._probe_health(slot) is False
    assert slot.comfy_checked is False and slot.comfy_answered is False
    assert _step_calls(orc, slot, now=400.0) == []


# ---------------------------------------------------------------- Правило B

def test_403_on_request_logs_makes_one_request_and_removes_card(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    vast = _Vast403()
    orc = _orc(tmp_path, vast=vast)
    slot = _slot(url="")  # без адреса воркера: снятие сразу берёт логи Vast
    orc._retire(slot, "тест 403", bad=True)
    assert vast.request_calls == 1
    assert vast.download_calls == 0
    assert vast.destroyed == [INSTANCE]                                  # карта снята (DELETE ушёл)
    assert caplog.text.count("код 403 на запрос логов") == 1             # одна строка о 403


def test_403_on_request_logs_stays_at_one_request_on_untimed_path(tmp_path):
    """Путь «хост не вышел на связь» и «воркер не готов» вызывают _fetch_vast_logs без бюджета, затем снятие."""
    vast = _Vast403()
    orc = _orc(tmp_path, vast=vast)
    slot = _slot(url="")
    assert orc._fetch_vast_logs(slot) is False
    orc._retire(slot, "хост не вышел на связь", bad=True)
    assert vast.request_calls == 1                                       # повтора нет ни в одном пути
    assert vast.download_calls == 0
    assert vast.destroyed == [INSTANCE]


def test_403_on_result_download_is_still_retried(tmp_path):
    vast = _VastDownload([403, 200])
    orc = _orc(tmp_path, vast=vast)
    slot = _slot(url="")
    assert orc._fetch_vast_logs(slot) is True
    assert vast.request_calls == 2                                       # запрос на каждую попытку, как раньше
    assert vast.download_calls == 2                                      # 403 загрузки повторяется
