"""Офлайн-тесты оркестратора (этап 8, часть 2).

Никакой сети и реальных пауз: воркер — httpx.MockTransport, Release — in-memory,
Vast — заглушка, время — фальшивые часы (sleep двигает часы).
Сообщения на русском (SPEC 0.5); секреты в логи не попадают (SPEC 0.1).
"""
from __future__ import annotations

import json
import logging
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
        self._offer_of: dict = {}
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
                             "env": dict(env_vars or {}), "label": label})
        return self._next

    def get_instance(self, instance_id):
        stuck = self._offer_of.get(instance_id) in self.stuck_offers
        return vc.InstanceInfo(
            instance_id=instance_id,
            actual_status="loading" if stuck else "running",
            intended_status="running",
            public_ip=None if stuck else "10.0.0.1",
            direct_port=None if stuck else 18000)

    def destroy_instance(self, instance_id):
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
