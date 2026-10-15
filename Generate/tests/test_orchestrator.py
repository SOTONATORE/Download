"""Офлайн-тесты оркестратора (этап 8, часть 2).

Никакой сети и реальных пауз: воркер — httpx.MockTransport, Release — in-memory,
Vast — заглушка, время — фальшивые часы (sleep двигает часы).
Сообщения на русском (SPEC 0.5); секреты в логи не попадают (SPEC 0.1).
"""
from __future__ import annotations

import json
import logging
import re
from collections import deque
import sys
import types
from pathlib import Path

import httpx
import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from Generate.core import orchestrator as orch
    from Generate.core import release_adapter as ra
    from Generate.core import vast_client as vc
except ImportError:  # pragma: no cover
    from core import orchestrator as orch  # type: ignore
    from core import release_adapter as ra  # type: ignore
    from core import vast_client as vc  # type: ignore

FULL_HASH = "ab12cd34ef56" + "0" * 52
TAG = "gen-ab12cd34ef56"
TOKEN = "WTOK-secret-0123456789"
VAST_KEY = "vast-key-SECRET-777"
GEMINI_KEY = "gemini-key-SECRET-888"
LABEL = "gen-test"
SECRETS = (TOKEN, VAST_KEY, GEMINI_KEY)


# ---------------------------------------------------------------------------
# Заглушки
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, sec: float) -> None:
        self.now += sec


class FakeStore(orch.ReleaseStore):
    def __init__(self, events: list) -> None:
        self.events = events
        self.releases: dict = {}
        self.created: list = []
        self.uploads: list = []   # (tag, имя, файл существовал в момент загрузки)
        self.blobs: dict = {}

    def preload(self, assets) -> None:
        self.releases[TAG] = orch.ReleaseInfo(
            tag=TAG, title="t", body=f"hash: {FULL_HASH}\nmodel: ltx25", assets=list(assets))

    def find_releases(self, tag: str) -> list:
        rel = self.releases.get(tag)
        return [rel] if rel else []

    def create_release(self, tag: str, title: str, body: str) -> None:
        self.events.append("release_create")
        self.created.append(tag)
        self.releases[tag] = orch.ReleaseInfo(tag=tag, title=title, body=body, assets=[])

    def upload_asset(self, tag: str, path: str) -> None:
        import os
        name = os.path.basename(path)
        self.events.append(f"upload:{name}")
        self.uploads.append((tag, name, os.path.exists(path)))
        with open(path, "rb") as fh:
            self.blobs[(tag, name)] = fh.read()
        self.releases[tag].assets.append(name)

    def clip_uploads(self) -> list:
        return [n for _, n, _ in self.uploads if n.endswith(".mp4")]


def make_offer(offer_id: int, machine_id: int, price: float = 0.3) -> "vc.GpuOffer":
    return vc.GpuOffer(offer_id=offer_id, gpu_name="RTX 4090", num_gpus=1, vram_gb=24.0,
                       price_per_hr=price, reliability=0.99, inet_down_mbps=3000.0,
                       inet_up_mbps=3000.0, machine_id=machine_id, is_interruptible=True)


class FakeVast:
    def __init__(self, offers: list, events: list) -> None:
        self.offers = offers
        self.events = events
        self.created: list = []        # словари аргументов create_instance
        self.destroyed: list = []
        self.label_destroyed: list = []
        self.search_calls = 0
        self.stuck_offers: set = set()  # предложения, чьи карты не выходят из loading
        self.create_exc = None
        self.fail_offers: set = set()   # предложения, создание которых завершается ошибкой
        self.label_exc = None
        # предложение -> сценарий опросов [(actual_status, status_msg), ...]; последний шаг повторяется
        self.script: dict = {}
        self._polls: dict = {}
        self._offer_of: dict = {}
        self.dph: dict = {}             # предложение -> dph_total в ответе карты (нет записи = 0.0, поля нет)
        self.destroy_exc = None         # если задано, destroy_instance бросает эту ошибку
        self._next = 100

    def search_offers(self, filters):
        self.search_calls += 1
        return list(self.offers)

    def create_instance(self, offer_id, image, disk_gb, env_vars=None, label=None, onstart=None):
        if self.create_exc is not None:
            raise self.create_exc
        if offer_id in self.fail_offers:
            raise vc.VastRuntimeError("тестовый сбой создания карты")
        self.events.append("vast_create")
        self._next += 1
        self._offer_of[self._next] = offer_id
        self.created.append({"id": self._next, "offer_id": offer_id,
                             "image": image, "disk_gb": disk_gb,
                             "env": dict(env_vars or {}), "label": label})
        return self._next

    def get_instance(self, instance_id):
        offer = self._offer_of.get(instance_id)
        if offer in self.script:
            steps = self.script[offer]
            i = self._polls.get(instance_id, 0)
            self._polls[instance_id] = i + 1
            status, msg = steps[min(i, len(steps) - 1)]
            up = status == "running"
            extra = {} if msg is None else {"status_msg": msg}   # None: поля статуса нет совсем
            return vc.InstanceInfo(
                instance_id=instance_id, actual_status=status, intended_status="running",
                public_ip="10.0.0.1" if up else None, direct_port=18000 if up else None,
                dph=self.dph.get(offer, 0.0), **extra)
        stuck = offer in self.stuck_offers
        return vc.InstanceInfo(
            instance_id=instance_id,
            actual_status="loading" if stuck else "running",
            intended_status="running",
            public_ip=None if stuck else "10.0.0.1",
            direct_port=None if stuck else 18000,
            dph=self.dph.get(offer, 0.0))

    def destroy_instance(self, instance_id):
        if self.destroy_exc is not None:
            raise self.destroy_exc
        self.destroyed.append(instance_id)
        return True

    def destroy_by_label(self, label):
        self.events.append("destroy_by_label")
        self.label_destroyed.append(label)
        if self.label_exc is not None:
            raise self.label_exc
        return [c["id"] for c in self.created if c["id"] not in self.destroyed]


class GeminiSpy:
    """Подменяет go.generate_and_store: промпты готовы, сеть не нужна."""

    def __init__(self, events: list) -> None:
        self.events = events
        self.calls = 0

    def __call__(self, segments, needed, cfg, transport, prompts_path, **kw):
        self.calls += 1
        self.events.append("gemini")
        state = {"srt_hash": FULL_HASH, "model": "fake", "profile": kw.get("profile", ""),
                 "prompts": {str(n): {"num": n, "status": "ready", "prompt": f"промпт {n}",
                                      "num_frames": 49} for n in sorted(needed)}}
        orch.go.atomic_write_json(prompts_path, state)
        return state


class FakeWorker:
    """HTTP-воркер: POST /task, GET /task/{id}, GET /file/{id}, POST /ack/{id}."""

    def __init__(self, clock, token: str, work_dir: Path, events: list,
                 render_sec: float = 15.0, key: str = "state",
                 error_text: str = "ошибка рендера") -> None:
        self.clock, self.token, self.work_dir, self.events = clock, token, work_dir, events
        self.render_sec, self.key, self.error_text = render_sec, key, error_text
        self.plan: dict = {}        # num -> список исходов по попыткам: "ok" | "error" | "hang"
        self.tasks: dict = {}
        self.posted: list = []
        self.acked: list = []
        self.files_left_at_ack: list = []
        self.urls: list = []
        self.bad_token = 0
        self._seq = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        if request.headers.get("X-Worker-Token") != self.token:
            self.bad_token += 1
            return httpx.Response(401, json={"error": "токен"})
        path, method = request.url.path, request.method
        # проверка готовности воркера (healthcheck gate: booting -> ready)
        if method == "GET" and path == "/health":
            return httpx.Response(200, json={"ok": True, "gpu": "FakeGPU", "vram_free": 12345})
        if method == "POST" and path == "/task":
            body = json.loads(request.content)
            num = body["num"]
            self.posted.append(num)
            self._seq += 1
            tid = f"t{self._seq}"
            plan = self.plan.get(num, [])
            outcome = plan.pop(0) if plan else "ok"
            self.tasks[tid] = {"num": num, "started": self.clock(), "outcome": outcome}
            return httpx.Response(200, json={"id": tid})
        if method == "GET" and path.startswith("/task/"):
            t = self.tasks[path.rsplit("/", 1)[1]]
            if t["outcome"] == "error":
                return httpx.Response(200, json={self.key: "error", "error": f"сбой {self.error_text}"})
            if t["outcome"] == "ok" and self.clock() - t["started"] >= self.render_sec:
                return httpx.Response(200, json={self.key: "done"})
            return httpx.Response(200, json={self.key: "running"})
        if method == "GET" and path.startswith("/file/"):
            t = self.tasks[path.rsplit("/", 1)[1]]
            return httpx.Response(200, content=f"MP4-{t['num']}".encode())
        if method == "POST" and path.startswith("/ack/"):
            t = self.tasks[path.rsplit("/", 1)[1]]
            self.acked.append(t["num"])
            self.events.append(f"ack:{t['num']}")
            self.files_left_at_ack.extend(self.work_dir.glob(f"gen-*/{t['num']}.mp4"))
            return httpx.Response(200, json={})
        return httpx.Response(404)


class Rig:
    def __init__(self, tmp_path, monkeypatch, *, missing=(1, 2), offers=None,
                 worker_kw=None, preload=None, token=TOKEN, **orch_kw) -> None:
        self.events: list = []
        self.clock = FakeClock()
        self.work_dir = tmp_path / "work"
        self.prompts_path = tmp_path / "prompts.json"
        self.worker = FakeWorker(self.clock, token or "", self.work_dir, self.events,
                                 **(worker_kw or {}))
        self.store = FakeStore(self.events)
        if preload is not None:
            self.store.preload(preload)
        self.vast = FakeVast(offers or [make_offer(1, 11)], self.events)
        self.gemini = GeminiSpy(self.events)
        self.missing = list(missing)

        monkeypatch.setattr(orch, "parse_missing", lambda path, segs: list(self.missing))
        monkeypatch.setattr(orch.go, "generate_and_store", self.gemini)
        monkeypatch.setattr(ra, "srt_hash", lambda segs: FULL_HASH)

        self.http = httpx.Client(transport=httpx.MockTransport(self.worker))
        kw = dict(repo="owner/repo", run_id="test", profile="ltx25", docker_image="img:latest",
                  worker_token=token, poll_interval_sec=10.0, work_dir=self.work_dir)
        kw.update(orch_kw)
        segs = [types.SimpleNamespace(num=n) for n in range(1, 6)]
        self.orc = orch.Orchestrator(
            segs, tmp_path / "missing.txt", None, None, self.store, self.vast, object(),
            self.prompts_path, http_client=self.http, clock=self.clock, sleep=self.clock.sleep, **kw)

    def run(self):
        return self.orc.run()

    def assert_clean_workdir(self) -> None:
        assert not self.work_dir.exists() or list(self.work_dir.iterdir()) == []


@pytest.fixture(autouse=True)
def _secrets_in_env(monkeypatch):
    monkeypatch.setenv("GEN_" + "VAST_API_KEY", VAST_KEY)
    monkeypatch.setenv("GEN_" + "GEMINI_API_KEY", GEMINI_KEY)


def _no_secrets(text: str) -> None:
    for s in SECRETS:
        assert s not in text


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2))
    summary = rig.run()

    assert summary.exit_code == 0
    assert (summary.completed, summary.failed, summary.pending) == (2, 0, 0)
    assert summary.total_needed == 2 and summary.cleanup_ok
    assert summary.spent_usd > 0

    assert rig.store.created == [TAG]
    assert rig.store.clip_uploads() == ["1.mp4", "2.mp4"]
    # клип существовал на диске в момент загрузки
    assert all(existed for _, name, existed in rig.store.uploads if name.endswith(".mp4"))
    assert rig.store.blobs[(TAG, "1.mp4")] == b"MP4-1"
    # временные файлы удалены сразу: к моменту ack на диске клипа уже нет
    assert rig.worker.files_left_at_ack == []
    rig.assert_clean_workdir()
    assert rig.worker.acked == [1, 2]

    # служебные файлы и ссылки
    links = rig.store.blobs[(TAG, "generated_links.txt")].decode()
    assert f"https://github.com/owner/repo/releases/download/{TAG}/1.mp4" in links
    assert (TAG, "prompts.json") in rig.store.blobs

    # карты уничтожены по метке
    assert rig.vast.label_destroyed == [LABEL]
    assert rig.vast.created[0]["label"] == LABEL
    assert rig.vast.destroyed  # карта снята по завершении работы

    # порядок этапов
    ev = rig.events
    assert ev.index("release_create") < ev.index("gemini") < ev.index("vast_create")
    assert ev.index("vast_create") < ev.index("upload:1.mp4") < ev.index("ack:1")
    assert ev.index("ack:2") < ev.index("destroy_by_label")
    assert rig.worker.bad_token == 0
    _no_secrets(caplog.text)


@pytest.mark.parametrize("key", ["state", "status"])
def test_poll_reads_state_and_status(tmp_path, monkeypatch, key):
    rig = Rig(tmp_path, monkeypatch, missing=(1,), worker_kw={"key": key})
    summary = rig.run()
    assert summary.exit_code == 0 and summary.completed == 1
    assert rig.store.clip_uploads() == ["1.mp4"]


# ---------------------------------------------------------------------------
# Ранние выходы
# ---------------------------------------------------------------------------

def test_stub_missing_exits_without_work(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=())   # строка-заглушка даёт пустой список
    summary = rig.run()
    assert summary.exit_code == 0 and summary.total_needed == 0
    assert rig.gemini.calls == 0
    assert rig.vast.search_calls == 0 and rig.vast.created == []
    assert rig.store.created == [] and rig.store.uploads == []
    assert rig.vast.label_destroyed == []


def test_all_clips_already_in_release(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2), preload=["1.mp4", "2.mp4"])
    summary = rig.run()
    assert summary.exit_code == 0
    assert summary.skipped_done == 2 and summary.total_needed == 0
    assert rig.gemini.calls == 0
    assert rig.vast.search_calls == 0 and rig.vast.created == []
    assert rig.store.created == []
    assert rig.store.clip_uploads() == []


def test_no_clips_means_no_links_upload(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2))

    def gemini_unavailable(*args, **kwargs):
        raise orch.gp.GeminiRuntimeError("Сервис Gemini временно недоступен", status=503)
    monkeypatch.setattr(orch.go, "generate_and_store", gemini_unavailable)

    summary = rig.run()
    assert summary.completed == 0 and rig.vast.created == []
    assert rig.store.clip_uploads() == []
    assert rig.store.uploads == []                          # ни prompts.json, ни generated_links.txt
    assert (TAG, "generated_links.txt") not in rig.store.blobs
    assert caplog.text.count("Нет готовых клипов: generated_links.txt не выкладывается.") == 1
    _no_secrets(caplog.text)


def test_resume_renders_only_missing(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2), preload=["1.mp4"])
    summary = rig.run()
    assert summary.exit_code == 0 and summary.skipped_done == 1 and summary.completed == 1
    assert rig.worker.posted == [2]
    assert rig.store.clip_uploads() == ["2.mp4"]


# ---------------------------------------------------------------------------
# Бюджет
# ---------------------------------------------------------------------------

def test_budget_blocks_renting_when_unaffordable(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[make_offer(1, 11, price=1.0)],
              budget_limit_usd=0.1)
    summary = rig.run()
    assert rig.vast.created == []
    assert summary.completed == 0 and summary.pending == 1
    assert summary.exit_code == 3
    assert rig.vast.label_destroyed == [LABEL]


def test_budget_exceeded_mid_run_stops_everything(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[make_offer(1, 11, price=0.9)],
              budget_limit_usd=0.5, task_timeout_sec=1e9, card_max_lifetime_min=300,
              worker_kw={"render_sec": 1e9})
    rig.worker.plan[1] = ["hang"] * 5
    summary = rig.run()
    assert summary.budget_exceeded is True
    assert len(rig.vast.created) == 1            # новых карт после превышения нет
    assert len(rig.vast.destroyed) == 1
    assert summary.completed == 0 and summary.pending == 1
    assert summary.exit_code == 3
    assert rig.vast.label_destroyed == [LABEL]
    assert "Бюджет исчерпан" in caplog.text


# ---------------------------------------------------------------------------
# Молчащий хост
# ---------------------------------------------------------------------------

def test_silent_host_is_retired_and_replaced(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    offers = [make_offer(1, 11, price=0.2), make_offer(2, 22, price=0.3)]
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=offers, silent_host_timeout_min=1)
    rig.vast.stuck_offers = {1}
    summary = rig.run()

    assert [c["offer_id"] for c in rig.vast.created] == [1, 2]   # плохой хост не выбран повторно
    first_id = rig.vast.created[0]["id"]
    assert first_id in rig.vast.destroyed
    assert summary.completed == 1 and summary.exit_code == 0
    assert summary.cards_rented == 2
    assert "хост не вышел на связь" in caplog.text


def test_failed_cheapest_offer_climbs_to_next_cheapest(tmp_path, monkeypatch):
    offers = [make_offer(1, 11, price=0.2), make_offer(2, 22, price=0.3),
              make_offer(3, 33, price=0.4)]
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=offers)
    rig.vast.fail_offers = {1}
    summary = rig.run()

    # самое дешёвое предложение не создалось, следующее по цене выбрано сразу
    assert [c["offer_id"] for c in rig.vast.created] == [2]
    assert 1 in rig.orc._bad_offers
    assert summary.completed == 1 and summary.exit_code == 0
    assert summary.cards_rented == 1


# ---------------------------------------------------------------------------
# Дедлайн job
# ---------------------------------------------------------------------------

def test_job_deadline_drains_inflight_and_returns_partial(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2), max_cards=1, job_deadline_min=1,
              worker_kw={"render_sec": 80.0})
    summary = rig.run()

    assert summary.deadline_reached is True
    assert rig.worker.posted == [1]                   # второй клип не взят
    assert rig.store.clip_uploads() == ["1.mp4"]      # начатый клип дозалит
    assert (summary.completed, summary.pending) == (1, 1)
    assert summary.exit_code == 3
    assert rig.vast.label_destroyed == [LABEL]
    assert "Достигнут срок job" in caplog.text


def test_deadline_drain_grace_expires(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,), max_cards=1, job_deadline_min=1,
              drain_grace_sec=30.0, task_timeout_sec=1e9, worker_kw={"render_sec": 1e9})
    rig.worker.plan[1] = ["hang"]
    summary = rig.run()
    assert summary.deadline_reached and summary.completed == 0
    assert summary.exit_code == 3
    assert rig.vast.label_destroyed == [LABEL]


# ---------------------------------------------------------------------------
# Повторы
# ---------------------------------------------------------------------------

def test_task_retry_then_success(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), worker_kw={"error_text": TOKEN})
    rig.worker.plan[1] = ["error", "error"]
    summary = rig.run()
    assert rig.worker.posted == [1, 1, 1]
    assert summary.completed == 1 and summary.failed == 0 and summary.exit_code == 0
    assert rig.store.clip_uploads() == ["1.mp4"]
    _no_secrets(caplog.text)       # токен из текста ошибки воркера вычищен
    assert "***" in caplog.text


def test_task_fails_after_three_attempts(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.worker.plan[1] = ["error"] * 5
    summary = rig.run()
    assert rig.worker.posted == [1, 1, 1]
    assert (summary.completed, summary.failed) == (0, 1)
    assert summary.exit_code == 3
    assert rig.store.clip_uploads() == []
    prompts = json.loads(rig.prompts_path.read_text(encoding="utf-8"))["prompts"]
    assert prompts["1"]["render_status"] == "failed"
    assert rig.vast.label_destroyed == [LABEL]


# ---------------------------------------------------------------------------
# Гарантированная уборка
# ---------------------------------------------------------------------------

def test_cleanup_runs_in_finally_on_unexpected_error(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.vast.create_exc = RuntimeError("неожиданный сбой")
    with pytest.raises(RuntimeError) as ei:
        rig.run()
    assert rig.vast.label_destroyed == [LABEL]
    rig.assert_clean_workdir()
    _no_secrets(str(ei.value))


def test_cleanup_failure_gives_partial_exit(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.vast.label_exc = vc.VastRuntimeError("нет связи с Vast")
    summary = rig.run()
    assert rig.vast.label_destroyed == [LABEL]
    assert summary.completed == 1
    assert summary.cleanup_ok is False and summary.exit_code == 3
    assert "Не удалось уничтожить карты" in caplog.text


# ---------------------------------------------------------------------------
# Безопасность секретов
# ---------------------------------------------------------------------------

def test_tokens_never_leak(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2), worker_kw={"error_text": TOKEN})
    rig.worker.plan[2] = ["error"]
    summary = rig.run()

    _no_secrets(caplog.text)
    _no_secrets(repr(rig.orc))
    _no_secrets(str(rig.orc))
    _no_secrets(str(summary.as_dict()))
    # токен не в URL, только в заголовке (воркер отверг бы запросы без него)
    assert all(TOKEN not in u for u in rig.worker.urls)
    assert rig.worker.bad_token == 0
    # токен доходит до воркера единственным путём: переменной окружения карты
    assert rig.vast.created[0]["env"]["WORKER_TOKEN"] == TOKEN
    others = {k: v for k, v in rig.vast.created[0]["env"].items() if k != "WORKER_TOKEN"}
    assert TOKEN not in json.dumps(others)
    # в Release токен не попал
    for blob in rig.store.blobs.values():
        _no_secrets(blob.decode("utf-8", errors="ignore"))


def test_generated_token_not_in_repr(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,), token=None)
    token = rig.orc._token
    assert token and token != TOKEN
    assert token not in repr(rig.orc) and token not in str(rig.orc)


# ---------------------------------------------------------------------------
# Строгая проверка num_frames и проброс параметров
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", [
    {"num": 1, "status": "ready", "prompt": "промпт 1", "num_frames": None},
    {"num": 1, "status": "ready", "prompt": "промпт 1", "num_frames": 0},
    {"num": 1, "status": "ready", "prompt": "промпт 1", "num_frames": -5},
    {"num": 1, "status": "ready", "prompt": "промпт 1", "num_frames": "49"},
    {"num": 1, "status": "ready", "prompt": "промпт 1"},
], ids=["none", "zero", "negative", "string", "missing"])
def test_missing_or_invalid_num_frames_fails_immediately(tmp_path, monkeypatch, caplog, entry):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    orc = rig.orc
    prompts = {"1": dict(entry)}
    orc._state = {"prompts": prompts}
    orc._prompts = prompts
    orc._queue = deque([1])
    slot = orch._CardSlot(index=0, generation=1, instance_id=101, machine_id=11,
                          price=0.3, created=rig.clock())

    orc._dispatch(slot, rig.clock())

    assert rig.worker.posted == []                      # воркеру ничего не отправлено
    assert slot.task is None
    assert 1 in orc._failed
    assert list(orc._queue) == []                       # без повторных попыток
    assert prompts["1"]["render_status"] == "failed"
    assert "в prompts.json нет корректного num_frames" in caplog.text
    _no_secrets(caplog.text)


@pytest.mark.parametrize("num_frames, ready", [
    (49, True), (97, True), (1, True),
    (0, False), (-1, False), (None, False), ("49", False), (True, False), (49.0, False),
])
def test_prompt_ready_requires_positive_int_num_frames(num_frames, ready):
    entry = {"status": "ready", "prompt": "промпт", "num_frames": num_frames}
    assert orch._prompt_ready(entry) is ready
    assert orch._prompt_ready({"status": "ready", "prompt": "промпт"}) is False
    assert orch._prompt_ready(None) is False


def test_valid_num_frames_is_sent_to_worker_as_is(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.run()
    assert rig.worker.posted == [1]
    assert rig.store.clip_uploads() == ["1.mp4"]


def test_orchestrator_defaults_for_new_parameters(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    # Rig задаёт docker_image явно; остальные значения — по умолчанию.
    assert rig.orc.docker_image == "img:latest"
    assert rig.orc.disk_gb == 100
    assert rig.orc._silent_sec == 20 * 60.0


def test_run_generation_passes_new_parameters(tmp_path, monkeypatch):
    captured: dict = {}

    class _Recorder:
        def __init__(self, *args, **kwargs) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

        def run(self):
            return "итог"

    monkeypatch.setattr(orch, "Orchestrator", _Recorder)
    result = orch.run_generation(
        [], tmp_path / "missing.txt", None, None, object(), object(), object(),
        tmp_path / "prompts.json",
        docker_image="custom/img:1", disk_gb=77, silent_host_timeout_min=2)
    assert result == "итог"
    kw = captured["kwargs"]
    assert (kw["docker_image"], kw["disk_gb"], kw["silent_host_timeout_min"]) == (
        "custom/img:1", 77, 2)


def test_new_parameters_reach_vast_and_silent_timeout(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    offers = [make_offer(1, 11, price=0.2), make_offer(2, 22, price=0.3)]
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=offers,
              docker_image="custom/img:1", disk_gb=77, silent_host_timeout_min=2)
    rig.vast.stuck_offers = {1}
    assert rig.orc._silent_sec == 120.0
    summary = rig.run()

    assert [c["offer_id"] for c in rig.vast.created] == [1, 2]
    assert all(c["image"] == "custom/img:1" and c["disk_gb"] == 77 for c in rig.vast.created)
    assert rig.vast.created[0]["id"] in rig.vast.destroyed       # молчащий хост снят
    assert "хост не вышел на связь за 2 мин" in caplog.text
    assert summary.completed == 1 and summary.exit_code == 0
    _no_secrets(caplog.text)


# ---------------------------------------------------------------------------
# Неудачный старт карты (SPEC 8.2): снятие сразу, хост не арендуется снова в этом запуске
# ---------------------------------------------------------------------------

START_REASON = orch.START_FAILURE_REASON
CDI_MSG = ("Error response from daemon: failed to create task for container: failed to create "
           "shim task: OCI runtime create failed: failed to inject CDI devices: unresolvable CDI "
           "devices nvidia.com/gpu=2: unknown")
TWO_OFFERS = [make_offer(1, 11, price=0.2), make_offer(2, 22, price=0.3)]


def _spy_retire(orc):
    """Записывает каждое снятие карты: причину, bad, хост, время создания и время снятия."""
    calls: list = []
    original = orc._retire

    def spy(slot, reason, *, bad=False):
        calls.append({"instance": slot.instance_id, "machine": slot.machine_id,
                      "reason": reason, "bad": bad, "created": slot.created, "at": orc.clock()})
        return original(slot, reason, bad=bad)

    orc._retire = spy
    return calls


def test_loading_then_created_retires_card_at_once(tmp_path, monkeypatch, caplog):
    """Статус поля нет совсем (None): работает только правило перехода loading -> created."""
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=list(TWO_OFFERS))
    rig.vast.script[1] = [("created", None), ("loading", None), ("created", None)]
    calls = _spy_retire(rig.orc)
    summary = rig.run()

    first = calls[0]
    assert first["reason"] == START_REASON and first["bad"] is True
    assert first["instance"] == rig.vast.created[0]["id"]
    assert first["at"] - first["created"] <= 60          # сразу, а не через 20 минут молчания
    assert rig.vast.created[0]["id"] in rig.vast.destroyed
    assert 11 in rig.orc._bad_machines                   # хост 11 исключён из этого запуска
    assert [c["offer_id"] for c in rig.vast.created] == [1, 2]
    assert summary.completed == 1 and summary.exit_code == 0
    assert "старт не удался (статус «loading» → «created»)" in caplog.text
    assert "хост 11 исключён из этого запуска" in caplog.text


def test_created_only_is_not_judged_as_failed_start(tmp_path, monkeypatch, caplog):
    """Карта ни разу не была в loading: новое правило её не трогает, даже с текстом CDI."""
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=list(TWO_OFFERS), silent_host_timeout_min=1)
    rig.vast.script[1] = [("created", CDI_MSG)]
    calls = _spy_retire(rig.orc)
    summary = rig.run()

    assert calls[0]["reason"] == "хост не вышел на связь за 1 мин"   # старое правило, не новое
    assert calls[0]["bad"] is True
    assert START_REASON not in caplog.text and "старт не удался" not in caplog.text
    assert summary.completed == 1


def test_normal_start_loading_running_is_unchanged(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.vast.script[1] = [("created", None), ("loading", None), ("running", None)]
    calls = _spy_retire(rig.orc)
    summary = rig.run()

    assert summary.completed == 1 and summary.exit_code == 0
    assert len(rig.vast.created) == 1
    assert all(not c["bad"] for c in calls)              # ни одного аварийного снятия
    assert rig.orc._bad_machines == set() and rig.orc._bad_offers == set()
    assert "старт не удался" not in caplog.text
    assert "healthcheck OK" in caplog.text


def test_cdi_message_while_loading_retires_at_once(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=list(TWO_OFFERS))
    rig.vast.script[1] = [("created", None), ("loading", CDI_MSG)]
    calls = _spy_retire(rig.orc)
    summary = rig.run()

    assert calls[0]["reason"] == START_REASON and calls[0]["bad"] is True
    assert calls[0]["at"] - calls[0]["created"] <= 30    # первый же опрос после loading
    assert 11 in rig.orc._bad_machines
    assert summary.completed == 1
    assert "старт не удался (статус «created» → «loading»; текст: Error response from daemon" in caplog.text


def test_unknown_previous_status_is_named_in_log(tmp_path, monkeypatch, caplog):
    """Первый же опрос уже «loading» (предыдущего статуса нет): в строке «неизвестно», не пусто."""
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=list(TWO_OFFERS))
    rig.vast.script[1] = [("loading", CDI_MSG)]
    rig.run()
    assert "старт не удался (статус «неизвестно» → «loading»; текст:" in caplog.text


def test_status_message_is_redacted_before_log(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    hf_fake = "hf-fake-SECRET-554433"
    monkeypatch.setenv("HF_TOKEN", hf_fake)
    # секреты в начале текста: в строке лога они должны быть уже замаскированы
    msg = f"token={TOKEN} hf={hf_fake} {CDI_MSG}"
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=list(TWO_OFFERS))
    rig.vast.script[1] = [("loading", msg)]
    summary = rig.run()

    _no_secrets(caplog.text)
    assert hf_fake not in caplog.text
    assert "token=***" in caplog.text and "hf=***" in caplog.text
    assert summary.completed == 1


def test_status_message_over_200_chars_is_logged_in_full_and_redacted(tmp_path, monkeypatch, caplog):
    """Длинный текст ошибки попадает в строку лога целиком; секрет в его хвосте маскируется."""
    caplog.set_level(logging.INFO)
    fake_hf = "hf-fake-TAIL-554466"
    monkeypatch.setenv("HF_TOKEN", fake_hf)
    long_msg = ("CDI " + "длинный текст " * 40 + "\nвторая строка; "
                + f"hf={fake_hf}" + " конец ошибки")
    assert len(long_msg) > 200
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=list(TWO_OFFERS))
    rig.vast.script[1] = [("loading", long_msg)]
    summary = rig.run()

    line = next(r.getMessage() for r in caplog.records if "старт не удался" in r.getMessage())
    assert "\n" not in line
    m = re.search(r"; текст: (.*)\)\. Карта снимается", line)
    assert m is not None
    text = m.group(1)
    assert len(text) > 200                                   # без обрезки
    assert text.startswith("CDI длинный текст")
    assert text.endswith("вторая строка; hf=*** конец ошибки")  # хвост на месте, секрет замаскирован
    assert fake_hf not in caplog.text
    _no_secrets(caplog.text)
    assert summary.completed == 1


@pytest.mark.parametrize("text,expected", [
    ("Error: failed to create shim task", True),
    ("OCI runtime create failed", True),
    ("unresolvable cdi devices", True),
    ("Shim crashed", True),
    ("Container is running", False),
    ("", False),
])
def test_start_markers_match_case_insensitively(text, expected):
    assert orch._has_start_marker(text) is expected


def test_one_line_collapses_whitespace_and_never_cuts():
    assert orch._one_line("a  b\n c") == "a b c"
    long_text = "x" * 1000
    assert orch._one_line(long_text) == long_text


def test_empty_verified_pool_rents_nothing(tmp_path, monkeypatch, caplog):
    """Пустой пул (поиск с verified не нашёл предложений): карта не арендуется, запуск
    завершается по существующей логике (срок job_deadline_min), непроверенный хост не берётся."""
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), job_deadline_min=5)
    rig.vast.offers = []
    summary = rig.run()

    assert rig.vast.created == []
    assert "Подходящих предложений Vast нет, повтор позже." in caplog.text
    assert summary.completed == 0


def test_failed_start_machine_and_offer_are_not_rented_again(tmp_path, monkeypatch):
    offers = [make_offer(1, 11, price=0.2),    # плохая карта: хост 11, предложение 1
              make_offer(3, 11, price=0.25),   # другое предложение того же хоста 11
              make_offer(2, 22, price=0.3)]    # другой хост
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=offers)
    rig.vast.script[1] = [("loading", None), ("created", None)]
    summary = rig.run()

    assert [c["offer_id"] for c in rig.vast.created] == [1, 2]   # предложение 3 не арендовано
    assert 1 in rig.orc._bad_offers and 11 in rig.orc._bad_machines
    assert summary.completed == 1 and summary.exit_code == 0


def test_rent_one_filters_bad_machine_and_bad_offer(tmp_path, monkeypatch):
    offers = [make_offer(1, 11, price=0.2), make_offer(3, 11, price=0.25),
              make_offer(2, 22, price=0.3)]
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=offers)
    rig.orc._bad_machines.add(11)
    rig.orc._bad_offers.add(2)
    assert rig.orc._rent_one(rig.clock.now) == "retry"   # хост 11 и предложение 2 отброшены
    assert rig.vast.created == []

    rig.orc._bad_offers.clear()
    assert rig.orc._rent_one(rig.clock.now) == "ok"      # снова доступно только предложение 2
    assert [c["offer_id"] for c in rig.vast.created] == [2]


def test_failed_card_cost_is_stopped_like_any_retirement(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=list(TWO_OFFERS))
    rig.vast.script[1] = [("loading", None), ("created", None)]
    calls = _spy_retire(rig.orc)
    rig.run()

    failed_id = rig.vast.created[0]["id"]
    rec = next(r for r in rig.orc.budget._records if r.instance_id == failed_id)
    assert not rec.is_active
    assert rec.stop_time == calls[0]["at"]               # учёт остановлен в момент снятия
    assert rec.accrued_cost(rec.stop_time + 3600) == rec.accrued_cost(rec.stop_time)
    assert rec.accrued_cost(rec.stop_time) <= 0.2 * 60 / 3600 + 1e-9


# ---------------------------------------------------------------------------
# Счётчик «экземпляр не найден» (SPEC 8.2, 8.3)
# ---------------------------------------------------------------------------

NF_MSG = "Экземпляр Vast {} не найден в ответе API"


class _Responses:
    """Сценарий ответов get_instance для одной карты.

    'nf' — VastInstanceNotFound; 'timeout' — VastRuntimeError по таймауту;
    любое другое значение — обычный ответ «loading» (карта ещё стартует). Шаги кончились — «nf».
    """

    def __init__(self, steps) -> None:
        self.steps = list(steps)
        self.calls = 0

    def __call__(self, instance_id):
        self.calls += 1
        step = self.steps.pop(0) if self.steps else "nf"
        if step == "nf":
            raise vc.VastInstanceNotFound(NF_MSG.format(instance_id))
        if step == "timeout":
            raise vc.VastRuntimeError(f"Таймаут запроса к Vast API: GET /api/v0/instances/{instance_id}/")
        return vc.InstanceInfo(instance_id=instance_id, actual_status="loading",
                               intended_status="running", dph=0.0)


def _booting_slot(rig, *, instance_id=500, machine_id=11, offer_id=1, price=0.3):
    """Карта в состоянии booting, учтённая в бюджете, как после аренды."""
    slot = orch._CardSlot(0, 1, instance_id, machine_id, price, rig.clock.now, offer_id=offer_id)
    rig.orc._slots[0] = slot
    rig.orc.budget.track_instance_start(instance_id, price, rig.clock.now)
    return slot


def _poll(rig, slot, times) -> None:
    """Опросы карты с шагом 10 с, как в основном цикле; стоп, если карта снята."""
    for _ in range(times):
        if rig.orc._slots.get(slot.index) is not slot:
            break
        rig.orc._step_slot(slot, rig.clock.now, False)
        rig.clock.sleep(10.0)


def test_nine_not_found_responses_do_not_retire(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    slot = _booting_slot(rig)
    rig.vast.get_instance = _Responses(["nf"] * 9)
    calls = _spy_retire(rig.orc)
    _poll(rig, slot, 9)

    assert calls == []
    assert rig.orc._slots.get(0) is slot and slot.not_found == 9
    assert rig.orc._bad_machines == set() and rig.orc._bad_offers == set()
    assert orch.NOT_FOUND_REASON not in caplog.text


def test_ten_not_found_responses_retire_and_exclude_machine_for_run(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    offers = [make_offer(1, 11, price=0.2),    # карта на хосте 11
              make_offer(3, 11, price=0.25),   # другое предложение того же хоста 11
              make_offer(2, 22, price=0.3)]    # другой хост
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=offers)
    slot = _booting_slot(rig, machine_id=11, offer_id=1, price=0.2)
    rig.vast.get_instance = _Responses(["nf"] * 10)
    calls = _spy_retire(rig.orc)
    _poll(rig, slot, 10)

    assert len(calls) == 1
    assert calls[0]["reason"] == orch.NOT_FOUND_REASON and calls[0]["bad"] is True
    assert rig.orc._slots.get(0) is None
    assert 11 in rig.orc._bad_machines and 1 in rig.orc._bad_offers
    assert 500 in rig.vast.destroyed           # снятие идёт через обычный _retire

    warnings = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "в 10 ответах подряд" in r.getMessage()]
    assert len(warnings) == 1                  # одно предупреждение на снятие
    assert "Слот 0" in warnings[0] and "500" in warnings[0] and "11" in warnings[0]
    assert "bad_hosts.txt" not in warnings[0]  # постоянный список не трогается

    # следующая аренда не берёт хост 11, включая другое предложение того же хоста (offer 3)
    assert rig.orc._rent_one(rig.clock.now) == "ok"
    assert [c["offer_id"] for c in rig.vast.created] == [2]


def test_normal_response_between_not_found_resets_count(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    slot = _booting_slot(rig)
    rig.vast.get_instance = _Responses(["nf"] * 9 + ["loading"] + ["nf"] * 9)
    calls = _spy_retire(rig.orc)
    _poll(rig, slot, 19)

    assert calls == []                          # 9 + 1 обычный + 9: серия прервана
    assert slot.not_found == 9
    assert rig.orc._slots.get(0) is slot


def test_timeout_does_not_increment_not_found_counter(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    slot = _booting_slot(rig)
    rig.vast.get_instance = _Responses(["nf"] * 9 + ["timeout"] * 5)
    calls = _spy_retire(rig.orc)
    _poll(rig, slot, 14)

    assert calls == []
    assert slot.not_found == 9                  # таймауты счётчик не увеличили

    rig.vast.get_instance = _Responses(["nf"])  # следующий «не найдена» — десятый подряд
    _poll(rig, slot, 1)
    assert calls[0]["reason"] == orch.NOT_FOUND_REASON


# ---------------------------------------------------------------------------
# Цена карты по ответу Vast (SPEC 8.2, бюджет)
# ---------------------------------------------------------------------------

def test_budget_uses_vast_dph_after_first_poll(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[make_offer(1, 11, price=0.3)])
    rig.vast.dph[1] = 0.4398                    # Vast берёт больше, чем показал поиск
    summary = rig.run()

    inst = rig.vast.created[0]["id"]
    recs = [r for r in rig.orc.budget._records if r.instance_id == inst]
    assert len(recs) == 1                                      # одна запись, без переоткрытия
    assert round(recs[0].dph, 4) == 0.4398                     # ставка из ответа карты
    assert summary.completed == 1 and summary.exit_code == 0


def test_zero_dph_keeps_search_price_and_warns_once(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[make_offer(1, 11, price=0.3)])
    rig.vast.dph[1] = 0.0                       # поле есть, но цена нулевая
    summary = rig.run()

    inst = rig.vast.created[0]["id"]
    recs = [r for r in rig.orc.budget._records if r.instance_id == inst]
    assert [r.dph for r in recs] == [0.3]      # цена поиска сохранена
    assert caplog.text.count("Vast не вернул цену") == 1
    assert summary.completed == 1


# ---------------------------------------------------------------------------
# Снятие карты: стоимость фиксируется в момент снятия (SPEC 8.2, 8.3)
# ---------------------------------------------------------------------------

def _card_records(rig, instance_id=500):
    return [r for r in rig.orc.budget._records if r.instance_id == instance_id]


def test_failed_destroy_at_retirement_fixes_cost_at_that_moment(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.vast.destroy_exc = vc.VastRuntimeError("тестовый сбой уничтожения")
    slot = _booting_slot(rig, price=0.3)
    rig.clock.sleep(600.0)
    rig.orc._retire(slot, "тест")
    now = rig.clock.now

    recs = _card_records(rig)
    assert len(recs) == 1 and not recs[0].is_active and recs[0].stop_time == now
    first = rig.orc.budget.summary(now)["breakdown"][0]
    later = rig.orc.budget.summary(now + 7200)["breakdown"][0]
    assert later["active"] is False
    assert later["cost_usd"] == first["cost_usd"]              # после снятия стоимость не растёт
    assert rig.orc._slots.get(0) is None                       # прогон продолжается, слот освобождён
    assert "Стоимость зафиксирована на этот момент" in caplog.text
    assert "карта будет проверена при финальной уборке" in caplog.text


def test_cleanup_after_failed_destroy_does_not_add_cost_again(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.vast.destroy_exc = vc.VastRuntimeError("тестовый сбой уничтожения")
    slot = _booting_slot(rig, price=0.3)
    rig.clock.sleep(600.0)
    rig.orc._retire(slot, "тест")
    before = rig.orc.budget.summary(rig.clock.now)["total_spent"]

    rig.clock.sleep(3600.0)
    rig.orc._cleanup()                                         # не бросает
    after = rig.orc.budget.summary(rig.clock.now)["total_spent"]
    assert after == pytest.approx(before)
    assert len(_card_records(rig)) == 1                        # вторая запись не появилась


def test_successful_destroy_at_retirement_closes_record_once(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    slot = _booting_slot(rig, price=0.3)
    rig.clock.sleep(600.0)
    rig.orc._retire(slot, "тест")
    now = rig.clock.now

    recs = _card_records(rig)
    assert len(recs) == 1 and recs[0].stop_time == now and not recs[0].is_active
    assert rig.vast.destroyed == [500] and rig.orc._destroyed == 1
    total = rig.orc.budget.summary(now)["total_spent"]
    rig.clock.sleep(3600.0)
    rig.orc._cleanup()
    assert rig.orc.budget.summary(rig.clock.now)["total_spent"] == pytest.approx(total)
    assert len(_card_records(rig)) == 1


class _LiveVast(FakeVast):
    """Vast, который по метке отдаёт живые карты (live_ids), как list_instances(label=...) + destroy."""

    def __init__(self, offers, events, live_ids) -> None:
        super().__init__(offers, events)
        self.live = list(live_ids)
        self.attempts: list = []

    def destroy_instance(self, instance_id):
        self.attempts.append(instance_id)
        return super().destroy_instance(instance_id)

    def destroy_by_label(self, label):
        self.events.append("destroy_by_label")
        self.label_destroyed.append(label)
        if self.label_exc is not None:
            raise self.label_exc
        destroyed, failed = [], []
        for iid in self.live:
            try:
                self.destroy_instance(iid)
                destroyed.append(iid)
            except vc.VastRuntimeError:
                failed.append(iid)
        if failed:
            raise vc.VastRuntimeError(f"Не удалось уничтожить экземпляры Vast: {failed}")
        return destroyed


def test_cleanup_destroys_card_whose_retirement_destroy_failed(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.vast.destroy_exc = vc.VastRuntimeError("тестовый сбой уничтожения")
    slot = _booting_slot(rig, instance_id=500, machine_id=11, price=0.3)
    rig.clock.sleep(600.0)
    rig.orc._retire(slot, "тест")                              # уничтожение при снятии не прошло
    assert rig.vast.destroyed == []
    warnings = [r for r in caplog.records
                if r.levelno == logging.WARNING and "машина 11" in r.getMessage()]
    assert len(warnings) == 1
    assert "500" in warnings[0].getMessage()
    assert "уничтожение не подтверждено, карта будет проверена при финальной уборке" \
        in warnings[0].getMessage()

    live = _LiveVast(rig.vast.offers, rig.events, live_ids=[500])   # Vast по-прежнему видит карту
    rig.orc.vast = live
    before = rig.orc.budget.summary(rig.clock.now)["total_spent"]
    rig.clock.sleep(3600.0)
    rig.orc._cleanup()
    assert live.attempts == [500] and live.destroyed == [500]      # уборка уничтожила карту
    assert live.label_destroyed == [rig.orc.label]
    assert rig.orc._destroyed == 1 and rig.orc.cleanup_ok is True
    assert rig.orc.budget.summary(rig.clock.now)["total_spent"] == pytest.approx(before)
    assert len(_card_records(rig)) == 1                            # стоимость не добавлена, запись не открыта


def test_cleanup_destroys_live_card_without_budget_record(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    live = _LiveVast(rig.vast.offers, rig.events, live_ids=[777])  # в бюджете записи о 777 нет
    rig.orc.vast = live
    assert rig.orc.budget._records == []
    rig.orc._cleanup()
    assert live.destroyed == [777] and rig.orc._destroyed == 1
    assert rig.orc.cleanup_ok is True
    assert rig.orc.budget._records == []                           # уборка не создаёт записей бюджета


class _PartialDestroyVast(FakeVast):
    """Как реальный destroy_by_label: пробует все живые карты, при сбоях бросает VastDestroyError."""

    def __init__(self, offers, events, live_ids, fail_ids) -> None:
        super().__init__(offers, events)
        self.live = list(live_ids)
        self.fail_ids = set(fail_ids)

    def destroy_instance(self, instance_id):
        if instance_id in self.fail_ids:
            raise vc.VastRuntimeError("тестовый сбой уничтожения")
        self.destroyed.append(instance_id)
        return True

    def destroy_by_label(self, label):
        self.events.append("destroy_by_label")
        self.label_destroyed.append(label)
        destroyed, failed = [], []
        for iid in self.live:
            try:
                self.destroy_instance(iid)
                destroyed.append(iid)
            except vc.VastRuntimeError:
                failed.append(iid)
        if failed:
            raise vc.VastDestroyError(
                f"Не удалось уничтожить экземпляры Vast: {failed}; уничтожены: {destroyed}",
                destroyed_ids=destroyed, failed_ids=failed)
        return destroyed


def _record_of(rig, instance_id):
    return next(r for r in rig.orc.budget._records if r.instance_id == instance_id)


def test_cleanup_closes_destroyed_card_and_keeps_failed_card_open(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.orc.budget.track_instance_start(501, 0.3, rig.clock.now)   # карта A: уничтожится
    rig.orc.budget.track_instance_start(502, 0.3, rig.clock.now)   # карта B: не уничтожится
    rig.clock.sleep(600.0)
    vast = _PartialDestroyVast(rig.vast.offers, rig.events, live_ids=[501, 502], fail_ids=[502])
    rig.orc.vast = vast
    rig.orc._cleanup()                                             # исключение не выходит

    now = rig.clock.now
    rec_a, rec_b = _record_of(rig, 501), _record_of(rig, 502)
    assert not rec_a.is_active and rec_a.stop_time == now          # учёт A закрыт в момент уборки
    assert rec_b.is_active and rec_b.stop_time is None             # B жива: запись не закрыта
    assert vast.destroyed == [501]
    assert rig.orc.cleanup_ok is False and rig.orc._destroyed == 1
    assert "502" in caplog.text                                    # упавший id есть в логе, как раньше
    assert vast.label_destroyed == [rig.orc.label]


def test_cleanup_partial_failure_does_not_stop_closed_card_again(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.orc.budget.track_instance_start(501, 0.3, rig.clock.now)
    rig.orc.budget.track_instance_start(502, 0.3, rig.clock.now)
    rig.clock.sleep(600.0)
    rig.orc.budget.track_instance_stop(501, rig.clock.now)         # A уже закрыта при снятии
    stop_a = _record_of(rig, 501).stop_time
    cost_a = _record_of(rig, 501).accrued_cost(stop_a)

    rig.clock.sleep(3600.0)
    rig.orc.vast = _PartialDestroyVast(rig.vast.offers, rig.events, live_ids=[501, 502], fail_ids=[502])
    rig.orc._cleanup()                                             # без исключения и повторной остановки

    rec_a = _record_of(rig, 501)
    assert rec_a.stop_time == stop_a                               # время остановки не переписано
    assert rec_a.accrued_cost(rig.clock.now) == pytest.approx(cost_a)
    assert len(_card_records(rig, 501)) == 1
    assert _record_of(rig, 502).is_active                          # B по-прежнему открыта
    assert rig.orc.cleanup_ok is False


def test_cleanup_all_destroys_fail_closes_no_record(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.orc.budget.track_instance_start(501, 0.3, rig.clock.now)
    rig.orc.budget.track_instance_start(502, 0.3, rig.clock.now)
    rig.clock.sleep(600.0)
    rig.orc.vast = _PartialDestroyVast(rig.vast.offers, rig.events, live_ids=[501, 502],
                                       fail_ids=[501, 502])
    rig.orc._cleanup()                                             # исключение не выходит

    assert _record_of(rig, 501).is_active and _record_of(rig, 502).is_active
    assert rig.orc._destroyed == 0 and rig.orc.cleanup_ok is False
