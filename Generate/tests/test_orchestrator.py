"""Офлайн-тесты оркестратора (этап 8, часть 2).

Никакой сети и реальных пауз: воркер — httpx.MockTransport, Release — in-memory,
Vast — заглушка, время — фальшивые часы (sleep двигает часы).
Сообщения на русском (SPEC 0.5); секреты в логи не попадают (SPEC 0.1).
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import re
import threading
from collections import deque
import sys
import types
from pathlib import Path
import zipfile

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
TAG = "clips-ab12cd34ef56"
TOKEN = "WTOK-secret-0123456789"
VAST_KEY = "vast-key-SECRET-777"
GEMINI_KEY = "gemini-key-SECRET-888"
LABEL = "gen-test"
SECRETS = (TOKEN, VAST_KEY, GEMINI_KEY)
# Этап 9а: адрес карты, порт, ssh и ключ экземпляра в ответах Vast (документационные значения)
CARD_IP = "203.0.113.77"
CARD_PORT = 47123
SSH_HOST = "ssh-gw.example.invalid"
SSH_PORT = 39871
INSTKEY = "INSTKEY-98765"
CANARY = "TESTKEY-12345"
VALID_BUILD = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"   # номер коммита сборки воркера (SPEC 8.4)


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
        self.is_bid: dict = {}          # предложение -> is_bid в ответе карты (нет записи = True)
        self._next = 100
        self.raw_env: dict = {}         # переменные окружения карты, которые Vast вернул бы в ответе (этап 9а)

    def _raw(self, instance_id: int, up: bool) -> dict:
        """Ответ Vast со всеми полями, которых не должно быть в артефакте, логах и релизах (этап 9а)."""
        port = CARD_PORT if up else None
        return {
            "id": instance_id, "label": LABEL, "is_bid": True, "dph_total": 0.3,
            "actual_status": "running" if up else "loading", "intended_status": "running",
            "public_ipaddr": CARD_IP if up else None,
            "ports": {f"{vc.WORKER_HTTP_PORT}/tcp": [{"HostIp": "0.0.0.0", "HostPort": str(port)}]} if up else {},
            "direct_port_start": port, "direct_port_end": port, "direct_port_count": 1 if up else 0,
            "ssh_host": SSH_HOST, "ssh_port": SSH_PORT,
            "env": dict(self.raw_env),
            "extra_env": [[k, v] for k, v in self.raw_env.items()],
            "instance_api_key": INSTKEY,
        }

    def _redact(self, text: str) -> str:
        """Как у VastClient._redact: ключ Vast в тексте заменяется на ***."""
        return text.replace(VAST_KEY, "***")

    def search_offers(self, filters):
        self.search_calls += 1
        return list(self.offers)

    def create_instance(self, offer_id, image, disk_gb, env_vars=None, label=None, onstart=None,
                        bid_price=None):
        if self.create_exc is not None:
            raise self.create_exc
        if offer_id in self.fail_offers:
            raise vc.VastRuntimeError("тестовый сбой создания карты")
        self.events.append("vast_create")
        self._next += 1
        self._offer_of[self._next] = offer_id
        self.created.append({"id": self._next, "offer_id": offer_id,
                             "image": image, "disk_gb": disk_gb,
                             "env": dict(env_vars or {}), "label": label,
                             "bid_price": bid_price})
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
                public_ip=CARD_IP if up else None, direct_port=CARD_PORT if up else None,
                dph=self.dph.get(offer, 0.0), is_bid=self.is_bid.get(offer, True),
                raw=self._raw(instance_id, up), **extra)
        stuck = offer in self.stuck_offers
        return vc.InstanceInfo(
            instance_id=instance_id,
            actual_status="loading" if stuck else "running",
            intended_status="running",
            public_ip=None if stuck else CARD_IP,
            direct_port=None if stuck else CARD_PORT,
            dph=self.dph.get(offer, 0.0), is_bid=self.is_bid.get(offer, True),
            raw=self._raw(instance_id, not stuck))

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
        self.health_reply = None     # None: как раньше (ok true); иначе (код, тело) или исключение
        self.stats_reply = None      # ответ /system_stats (код, тело) или None = 404
        self.logs_text = ""          # текст GET /logs (строки лога карты)
        self.health_hits = []        # время каждого GET /health с верным токеном

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        # /system_stats живёт на порту ComfyUI, а не воркера: токен там не нужен и не проверяется
        if request.url.path != "/system_stats" and request.headers.get("X-Worker-Token") != self.token:
            self.bad_token += 1
            return httpx.Response(401, json={"error": "токен"})
        path, method = request.url.path, request.method
        # проверка готовности воркера (healthcheck gate: booting -> ready)
        if method == "GET" and path == "/health":
            self.health_hits.append(self.clock())
            if isinstance(self.health_reply, Exception):
                raise self.health_reply
            if self.health_reply is not None:
                code, body = self.health_reply
                return httpx.Response(code, text=body)
            return httpx.Response(200, json={"ok": True, "gpu": "FakeGPU", "vram_free": 12345})
        if method == "GET" and path == "/system_stats":
            if self.stats_reply is None:
                return httpx.Response(404)
            code, body = self.stats_reply
            return httpx.Response(code, text=body)
        if method == "GET" and path == "/logs":
            return httpx.Response(200, text=self.logs_text, headers={"X-Log-Next": str(len(self.logs_text.splitlines()))})
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
                  worker_token=token, poll_interval_sec=10.0, work_dir=self.work_dir,
                  worker_build=VALID_BUILD)
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

    # релизы запуска: клип-релиз, служебный (в начале), итоговый (в конце, SPEC 9.1)
    assert rig.store.created == [TAG, ra.service_tag("test"), ra.final_tag("test")]
    assert rig.store.clip_uploads() == ["1.mp4", "2.mp4"]
    # клип существовал на диске в момент загрузки
    assert all(existed for _, name, existed in rig.store.uploads if name.endswith(".mp4"))
    assert rig.store.blobs[(TAG, "1.mp4")] == b"MP4-1"
    # временные файлы удалены сразу: к моменту ack на диске клипа уже нет
    assert rig.worker.files_left_at_ack == []
    rig.assert_clean_workdir()
    assert rig.worker.acked == [1, 2]

    # служебные файлы и ссылки
    # generated_links.txt -> итоговый релиз; prompts.json -> служебный (SPEC 9.1, 9.4, 9.7)
    links = rig.store.blobs[(ra.final_tag("test"), "generated_links.txt")].decode()
    assert f"https://github.com/owner/repo/releases/download/{TAG}/1.mp4" in links
    assert (ra.service_tag("test"), "prompts.json") in rig.store.blobs
    assert (TAG, "generated_links.txt") not in rig.store.blobs  # E3: клип-релиз без служебных файлов
    assert (TAG, "prompts.json") not in rig.store.blobs

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
    # служебный релиз открывается в начале, итоговый создаётся для ссылок (SPEC 9.1, 9.4)
    assert rig.store.created == [ra.service_tag("test"), ra.final_tag("test")]
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
    # клип-релиз не получает служебных файлов (E3); links пустые и не выкладываются
    assert [u for u in rig.store.uploads if u[0] == TAG] == []
    assert not any(name == "generated_links.txt" for _, name, _ in rig.store.uploads)
    assert (ra.final_tag("test"), "generated_links.txt") not in rig.store.blobs
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
              budget_limit_usd=0.5, task_timeout_sec=1e9, offer_filter=vc.OfferFilter(max_price=1.0),
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
    slot = orch._CardSlot(index=1, generation=1, instance_id=101, machine_id=11,
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


def test_created_with_progress_text_is_not_retired(tmp_path, monkeypatch, caplog):
    """Прогресс «created» («Preparing GPUs...») после loading карту не снимает: ждём таймаут молчания (SPEC 8.2)."""
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=list(TWO_OFFERS), silent_host_timeout_min=1)
    rig.vast.script[1] = [("created", None), ("loading", None), ("created", "Preparing GPUs...")]
    calls = _spy_retire(rig.orc)
    summary = rig.run()

    assert calls[0]["reason"] == "хост не вышел на связь за 1 мин"
    assert calls[0]["bad"] is True
    assert calls[0]["at"] - calls[0]["created"] >= 60     # не раньше таймаута молчания
    assert START_REASON not in caplog.text and "старт не удался" not in caplog.text
    assert summary.completed == 1


def test_created_with_error_marker_retires_at_once(tmp_path, monkeypatch, caplog):
    """Маркер ошибки в тексте статуса «created» снимает карту сразу (статус уже начал старт)."""
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=list(TWO_OFFERS))
    rig.vast.script[1] = [("created", CDI_MSG)]
    calls = _spy_retire(rig.orc)
    summary = rig.run()

    assert calls[0]["reason"] == START_REASON and calls[0]["bad"] is True
    assert calls[0]["at"] - calls[0]["created"] <= 30
    assert 11 in rig.orc._bad_machines
    assert "статус «неизвестно» → «created»; текст: Error response from daemon" in caplog.text
    assert summary.completed == 1


def test_offline_after_loading_retires_as_failed_start(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=list(TWO_OFFERS))
    rig.vast.script[1] = [("loading", None), ("offline", None)]
    calls = _spy_retire(rig.orc)
    summary = rig.run()

    assert calls[0]["reason"] == START_REASON and calls[0]["bad"] is True
    assert "статус «loading» → «offline»" in caplog.text
    assert 11 in rig.orc._bad_machines
    assert summary.completed == 1


def test_bid_not_confirmed_by_vast_recomputes_budget_from_dph_total(tmp_path, monkeypatch, caplog):
    """Запрошен bid, Vast ответил is_bid не True: строка об ошибке, бюджет по dph_total ответа, не по min_bid."""
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[make_offer(1, 11, price=0.2)])
    rig.vast.is_bid[1] = None                 # поля is_bid в ответе нет
    rig.vast.dph[1] = 0.5                     # dph_total в ответе карты
    rig.orc.run()

    assert "карта " in caplog.text and "запрошен bid, Vast вернул is_bid=None; бюджет считается по dph_total." in caplog.text
    assert rig.vast.created[0]["bid_price"] == 0.23         # bid = min_bid 0.2 + bid_margin 0.03
    rows = rig.orc.budget.summary(rig.clock.now)["breakdown"]
    assert rows[0]["dph"] == pytest.approx(0.5)


def test_on_demand_rental_sends_no_bid_price(tmp_path, monkeypatch):
    offer = make_offer(1, 11, price=0.3)
    offer.is_interruptible = False
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[offer])
    rig.orc.run()
    assert rig.vast.created[0]["bid_price"] is None


def test_stop_signal_rents_nothing_and_cleans_up_by_label(tmp_path, monkeypatch, caplog):
    """Сигнал остановки до работы: ни поиска, ни аренды; уборка идёт по метке запуска (как kill-cards)."""
    caplog.set_level(logging.INFO)
    stop = threading.Event()
    stop.set()
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=list(TWO_OFFERS), stop_event=stop)
    rig.run()

    assert rig.vast.search_calls == 0 and rig.vast.created == []
    assert rig.vast.label_destroyed == [rig.orc.label]
    assert "Получен сигнал остановки" in caplog.text


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


def test_status_message_500_chars_is_logged_in_full_and_redacted(tmp_path, monkeypatch, caplog):
    """Текст статуса маскируется целиком и в строке лога не обрезается (SPEC 8.2, owner decision)."""
    caplog.set_level(logging.INFO)
    fake_hf = "hf-fake-TAIL-554466"
    monkeypatch.setenv("HF_TOKEN", fake_hf)
    head = "CDI " + "длинный текст " * 30
    tail = " конец ошибки ХВОСТ-OK"
    long_msg = head + "\nвторая строка; " + f"hf={fake_hf}" + tail
    assert len(long_msg) > 450
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=list(TWO_OFFERS))
    rig.vast.script[1] = [("loading", long_msg)]
    rig.run()

    line = next(r.getMessage() for r in caplog.records if "старт не удался" in r.getMessage())
    assert "\n" not in line
    m = re.search(r"; текст: (.*)\)\. Карта снимается", line)
    assert m is not None
    text = m.group(1)
    assert "…" not in text                                   # обрезки нет
    assert text.endswith("конец ошибки ХВОСТ-OK")            # хвост на месте
    assert text.startswith("CDI длинный текст")
    assert "hf=***" in text and fake_hf not in caplog.text
    _no_secrets(caplog.text)


@pytest.mark.parametrize("text,expected", [
    ("cdi", True),
    ("Failed To Create", True),
    ("Preparing GPUs...", False),
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
    rig.vast.script[1] = [("loading", None), ("created", "failed to create shim task")]
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
    rig.vast.script[1] = [("loading", None), ("created", "failed to create shim task")]
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
    slot = orch._CardSlot(1, 1, instance_id, machine_id, price, rig.clock.now, offer_id=offer_id)
    rig.orc._slots[1] = slot
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
    assert rig.orc._slots.get(1) is slot and slot.not_found == 9
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
    assert rig.orc._slots.get(1) is None
    assert 11 in rig.orc._bad_machines and 1 in rig.orc._bad_offers
    assert 500 in rig.vast.destroyed           # снятие идёт через обычный _retire

    warnings = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "в 10 ответах подряд" in r.getMessage()]
    assert len(warnings) == 1                  # одно предупреждение на снятие
    assert "Карта 1 поколение 1 (экземпляр 500)" in warnings[0] and "хосте 11" in warnings[0]
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
    assert rig.orc._slots.get(1) is slot


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
    assert [r.dph for r in recs] == [0.33]     # записана ставка bid: цена поиска + bid_margin 0.03
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
    assert rig.orc._slots.get(1) is None                       # прогон продолжается, слот освобождён
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


# ---------------------------------------------------------------------------
# Пилот ComfyUI (образ vastai/comfy, внутренний порт 18188). Офлайн: Vast и HTTP — заглушки.
# ---------------------------------------------------------------------------

class PilotVast(FakeVast):
    """Vast для пилота: статусы идут по сценарию, поле ports задаёт тест (None = поля нет)."""

    def __init__(self, offers, events, ports=None, steps=("loading", "running")) -> None:
        super().__init__(offers, events)
        self.ports = ports
        self.steps = list(steps)

    def get_instance(self, instance_id):
        i = self._polls.get(instance_id, 0)
        self._polls[instance_id] = i + 1
        status = self.steps[min(i, len(self.steps) - 1)]
        up = status == "running"
        raw = {} if self.ports is None else {"ports": self.ports}
        # адрес SSH есть у запущенной карты: проба пилота идёт через туннель (SPEC 8.5), внешний порт не нужен
        return vc.InstanceInfo(instance_id=instance_id, actual_status=status, intended_status="running",
                               public_ip="10.0.0.1" if up else None,
                               ssh_host="ssh.invalid" if up else None, ssh_port=2222 if up else None,
                               dph=0.0, is_bid=True, raw=raw)


PILOT_TUNNEL_PORT = 45678  # локальный порт фиктивного туннеля: тесты пилота ssh не запускают


class FakeTunnel:
    """Подмена ssh_probe.open_tunnel: считает открытия и закрытия, ssh не запускает.

    opened: (адрес, порт SSH, удалённый порт) — в ассертах не печатаются. closed: число закрытий.
    """

    def __init__(self) -> None:
        self.opened: list = []
        self.closed = 0

    def __call__(self, host, port, key_text, remote_port, ready_timeout_sec=30.0):
        self.opened.append((host, port, remote_port))
        return self._open()

    @contextlib.contextmanager
    def _open(self):
        try:
            yield PILOT_TUNNEL_PORT
        finally:
            self.closed += 1


def _pilot_orc(tmp_path, monkeypatch, *, vast, status=200, probe_urls=None, pilot_start_cmd="", tunnel=None, **kw):
    """Оркестратор в пилотном режиме; /system_stats отвечает кодом status, адреса проб пишутся в probe_urls.

    Туннель к карте подменён: tunnel (FakeTunnel) или новый FakeTunnel по умолчанию.
    """
    urls = probe_urls if probe_urls is not None else []
    monkeypatch.setattr(orch.ssh_probe, "open_tunnel", tunnel if tunnel is not None else FakeTunnel())

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if request.url.path == "/system_stats":
            return httpx.Response(status, json={"ok": True} if status == 200 else {"error": "нужен токен"})
        return httpx.Response(404)

    clock = FakeClock()
    http = httpx.Client(transport=httpx.MockTransport(handler))
    segs = [types.SimpleNamespace(num=n) for n in range(1, 6)]
    orc_kw = dict(repo="owner/repo", run_id="pilot-test", profile="ltx25", docker_image="comfy:img",
                  poll_interval_sec=10.0, silent_host_timeout_min=1, pilot=True,
                  pilot_start_cmd=pilot_start_cmd, work_dir=tmp_path / "work")
    orc_kw.update(kw)
    return orch.Orchestrator(
        segs, tmp_path / "missing.txt", None, None, FakeStore([]), vast, object(),
        tmp_path / "prompts.json", http_client=http, clock=clock, sleep=clock.sleep, **orc_kw)


PILOT_PORTS_OK = {"18188/tcp": [{"HostIp": "0.0.0.0", "HostPort": "31888"}]}


def test_pilot_probe_goes_through_tunnel_to_loopback(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    urls: list = []
    tunnel = FakeTunnel()
    vast = PilotVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    summary = _pilot_orc(tmp_path, monkeypatch, vast=vast, probe_urls=urls, tunnel=tunnel).run()
    assert urls, "проба не выполнялась"
    assert all(u == f"http://127.0.0.1:{PILOT_TUNNEL_PORT}/system_stats" for u in urls)
    assert tunnel.opened and all(t[2] == 18188 for t in tunnel.opened)  # туннель на внутренний порт 18188
    assert tunnel.closed == len(tunnel.opened)
    assert not any(":8188/" in u for u in urls)
    assert summary.exit_code == orch.EXIT_OK
    assert "PILOT-ИТОГ: результат=ok" in caplog.text
    assert "PILOT-МАРКЕР порт 18188: HTTP 200" in caplog.text
    assert "PILOT-МАРКЕР running: карта" in caplog.text
    assert "аренда→ответ=" in caplog.text
    assert "карт арендовано 1, уничтожено 1" in caplog.text


@pytest.mark.parametrize("code", [401, 403])
def test_auth_codes_give_auth_required_neither_failure_nor_success(tmp_path, monkeypatch, caplog, code):
    caplog.set_level(logging.INFO)
    vast = PilotVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    summary = _pilot_orc(tmp_path, monkeypatch, vast=vast, status=code).run()
    assert "PILOT-ИТОГ: результат=auth_required" in caplog.text
    assert f"PILOT-МАРКЕР порт 18188: HTTP {code}" in caplog.text
    assert summary.completed == 0 and summary.failed == 0
    assert summary.exit_code == orch.EXIT_OK


def test_other_code_until_timeout_gives_not_ready(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    vast = PilotVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    summary = _pilot_orc(tmp_path, monkeypatch, vast=vast, status=500).run()
    assert "PILOT-ИТОГ: результат=not_ready; причина=timeout" in caplog.text
    assert "HTTP 500" in caplog.text
    assert summary.failed == 1
    assert summary.exit_code == orch.EXIT_PARTIAL


def test_pilot_marker_check_never_builds_url_from_external_port(tmp_path, monkeypatch, caplog):
    """Маркер ComfyUI: туннель к 127.0.0.1 карты, URL из внешнего порта Vast не строится; туннель закрывается."""
    caplog.set_level(logging.INFO)
    urls: list = []
    tunnel = FakeTunnel()
    vast = PilotVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    _pilot_orc(tmp_path, monkeypatch, vast=vast, probe_urls=urls, tunnel=tunnel, status=500).run()
    assert tunnel.opened, "туннель не открывался"
    assert urls and all(u.startswith("http://127.0.0.1:") for u in urls)
    assert not any("31888" in u for u in urls)
    assert tunnel.closed == len(tunnel.opened)  # туннель закрыт после каждой пробы, в том числе неудачной
    assert "PILOT-ИТОГ: результат=not_ready" in caplog.text
    assert "PILOT-МАРКЕР порт 18188: HTTP 500" in caplog.text


def test_pilot_without_ssh_address_sends_no_probe(tmp_path, monkeypatch, caplog):
    """Пилот проверяет ComfyUI только через SSH-туннель: без адреса SSH туннеля и пробы нет (внешний порт не нужен)."""
    caplog.set_level(logging.INFO)
    urls: list = []
    tunnel = FakeTunnel()

    class NoSsh(PilotVast):
        def get_instance(self, instance_id):
            info = super().get_instance(instance_id)
            info.ssh_host = None
            info.ssh_port = None
            return info

    vast = NoSsh([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    _pilot_orc(tmp_path, monkeypatch, vast=vast, probe_urls=urls, tunnel=tunnel).run()
    assert urls == [] and tunnel.opened == []
    assert "PILOT-ИТОГ: результат=not_ready" in caplog.text


def test_pilot_without_worker_address_does_not_request_logs(tmp_path, monkeypatch, caplog):
    """Пилот: адреса воркера нет, логи не запрашиваются ни у воркера, ни через Vast (request_logs не вызывается)."""
    caplog.set_level(logging.INFO)
    calls: list = []

    class Spy(PilotVast):
        def request_logs(self, instance_id, **kwargs):
            calls.append("request_logs")
            raise AssertionError("request_logs не должен вызываться в пилоте")

        def download_logs(self, *args, **kwargs):
            calls.append("download_logs")
            raise AssertionError("download_logs не должен вызываться в пилоте")

    vast = Spy([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    orc = _pilot_orc(tmp_path, monkeypatch, vast=vast)
    slot = orch._CardSlot(index=1, generation=1, instance_id=77, machine_id=11, price=0.3, created=0.0)
    orc._dump_before_destroy(slot)
    assert calls == []
    assert "логи не запрошены (пилот)" in caplog.text


def test_pilot_copies_onstart_log_into_card_folder(tmp_path, monkeypatch, caplog):
    """onstart.log карты пилота копируется в cards/slot1_generation1/onstart.log артефакта (один артефакт)."""
    caplog.set_level(logging.INFO)
    seen: list = []

    def fake_probe(host, port, key_text, command_name, timeout_sec=20):
        seen.append(command_name)
        return orch.ssh_probe.ProbeResult(ok=True, returncode=0, output="[onstart] итог: код 0",
                                          failure_class="ok", output_length=19)

    monkeypatch.setattr(orch.ssh_probe, "run_probe", fake_probe)
    art = tmp_path / "art"
    vast = PilotVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    summary = _pilot_orc(tmp_path, monkeypatch, vast=vast, artifact_dir=art).run()
    assert "onstart_log" in seen
    copied = art / "cards" / "slot1_generation1" / "onstart.log"
    assert copied.read_text(encoding="utf-8").startswith("[onstart] итог")
    assert "PILOT: onstart.log скопирован в артефакт" in caplog.text
    assert summary.exit_code == orch.EXIT_OK


def test_pilot_onstart_copy_failure_is_warning_and_run_continues(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    def failing_probe(host, port, key_text, command_name, timeout_sec=20):
        return orch.ssh_probe.ProbeResult(ok=False, returncode=1, output="", failure_class="command_failed",
                                          output_length=0)

    monkeypatch.setattr(orch.ssh_probe, "run_probe", failing_probe)
    art = tmp_path / "art"
    vast = PilotVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    summary = _pilot_orc(tmp_path, monkeypatch, vast=vast, artifact_dir=art).run()
    assert not (art / "cards" / "slot1_generation1" / "onstart.log").exists()
    assert "PILOT: onstart.log не скопирован: command_failed" in caplog.text
    assert "PILOT-ИТОГ: результат=ok" in caplog.text
    assert summary.exit_code == orch.EXIT_OK


@pytest.mark.parametrize(("code", "verdict"), [
    (200, "ok"), (401, "auth_required"), (403, "auth_required"),
    (404, None), (500, None), (302, None), (None, None),
])
def test_pilot_verdict_mapping(code, verdict):
    assert orch._pilot_verdict(code) == verdict


def test_port_keys_text_has_names_only():
    info = vc.InstanceInfo(instance_id=1, actual_status="running", intended_status="running",
                           raw={"ports": {"22/tcp": [{"HostPort": "50022"}], "18188/tcp": [{"HostPort": "31888"}]}})
    text = orch._port_keys_text(info)
    assert text == "18188/tcp, 22/tcp"
    assert "50022" not in text and "31888" not in text


@pytest.mark.parametrize("start_cmd", ["", "   "])
def test_pilot_empty_start_cmd_sends_no_onstart(tmp_path, monkeypatch, start_cmd):
    received: list = []

    class Spy(PilotVast):
        def create_instance(self, offer_id, image, disk_gb, env_vars=None, label=None, onstart=None,
                            bid_price=None):
            received.append(onstart)
            return super().create_instance(offer_id, image, disk_gb, env_vars=env_vars, label=label,
                                           onstart=onstart, bid_price=bid_price)

    vast = Spy([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    _pilot_orc(tmp_path, monkeypatch, vast=vast, pilot_start_cmd=start_cmd).run()
    assert received == [None]


def test_pilot_nonempty_start_cmd_is_passed_as_onstart(tmp_path, monkeypatch):
    received: list = []

    class Spy(PilotVast):
        def create_instance(self, offer_id, image, disk_gb, env_vars=None, label=None, onstart=None,
                            bid_price=None):
            received.append(onstart)
            return super().create_instance(offer_id, image, disk_gb, env_vars=env_vars, label=label,
                                           onstart=onstart, bid_price=bid_price)

    vast = Spy([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    _pilot_orc(tmp_path, monkeypatch, vast=vast, pilot_start_cmd="echo start").run()
    assert received == ["echo start"]


def test_pilot_env_has_no_worker_hf_gemini_gh_or_comfy_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-PILOT-secret-4242")
    monkeypatch.setenv("GH_TOKEN", "gh-PILOT-secret-4242")
    vast = PilotVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    _pilot_orc(tmp_path, monkeypatch, vast=vast).run()
    env = vast.created[0]["env"]
    forbidden = {"WORKER_TOKEN", "HF_TOKEN", "GEN_" + "GEMINI_API_KEY", "GH_TOKEN"}
    assert not (forbidden & set(env))
    assert not any(k.startswith("OPEN_BUTTON_") for k in env)
    assert not any(k.startswith("PORTAL_CONFIG") and k != "PORTAL_CONFIG" for k in env)
    assert env["PORTAL_CONFIG"] == "localhost:18188:18188:/:ComfyUI"
    assert [k for k in env if k.startswith("COMFYUI_")] == ["COMFYUI_ARGS"]
    assert not any("8000" in k for k in env)          # -p 8000:8000 не добавляется
    assert not any(k.startswith("-p ") for k in env)  # порт 18188 не публикуется: проба идёт через SSH-туннель
    assert set(env) <= {"GEN_LOG_LEVEL", "COMFYUI_ARGS", "PORTAL_CONFIG"}
    _no_secrets(json.dumps(env))


def test_pilot_passes_exact_comfyui_args(tmp_path, monkeypatch):
    """Пилот передаёт COMFYUI_ARGS с точным значением: без них образ не поднимает ComfyUI на 18188."""
    vast = PilotVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    _pilot_orc(tmp_path, monkeypatch, vast=vast).run()
    env = vast.created[0]["env"]
    assert env["COMFYUI_ARGS"] == "--disable-auto-launch --disable-xformers --port 18188 --enable-cors-header"


def test_generation_env_comfyui_args_unchanged(tmp_path, monkeypatch):
    """Генерация не меняется пилотом: env карты воркера содержит то же значение COMFYUI_ARGS, что и раньше,
    и не получает пилотских ключей (GEN_LOG_LEVEL и COMFYUI_ARGS у пилота; порта пилота в env нет)."""
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.run()
    env = rig.vast.created[0]["env"]
    assert env["COMFYUI_ARGS"] == "--disable-auto-launch --disable-xformers --port 18188 --enable-cors-header"
    assert "-p 18188:18188" not in env


def test_pilot_passes_portal_config_for_comfyui(tmp_path, monkeypatch):
    """Пилот передаёт PORTAL_CONFIG: образ пишет его в /etc/portal.yaml, без записи с label ComfyUI не стартует ComfyUI."""
    vast = PilotVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    _pilot_orc(tmp_path, monkeypatch, vast=vast).run()
    env = vast.created[0]["env"]
    assert env["PORTAL_CONFIG"] == "localhost:18188:18188:/:ComfyUI"
    assert env["PORTAL_CONFIG"].rsplit(":", 1)[-1] == "ComfyUI"


def test_generation_env_has_no_portal_config(tmp_path, monkeypatch):
    """Генерация не получает PORTAL_CONFIG: её env остаётся прежним."""
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.run()
    env = rig.vast.created[0]["env"]
    assert "PORTAL_CONFIG" not in env


def test_pilot_writes_artifact_layout_and_notes_skipped_files(tmp_path, monkeypatch, caplog):
    """Этап 7в (SPEC 10.4): пилот ведёт артефакт тем же путём: run/summary.md, FILES.txt, cards/, service/.
    prompts.json и missing.txt пилот не формирует: их нет в service/, это записано в журнал. Вердикт не меняется."""
    caplog.set_level(logging.INFO)
    art = tmp_path / "art"
    vast = PilotVast([make_offer(1, 11)], [], ports=PILOT_PORTS_OK)
    summary = _pilot_orc(tmp_path, monkeypatch, vast=vast, artifact_dir=art).run()
    assert summary.exit_code == orch.EXIT_OK
    assert "PILOT-ИТОГ: результат=ok" in caplog.text
    assert (art / "run" / "summary.md").is_file()
    assert "Итог пилота pilot-test" in (art / "run" / "summary.md").read_text(encoding="utf-8")
    assert not (art / "run" / "run.log").exists()
    assert (art / "cards").is_dir() and (art / "service").is_dir()
    assert not any((art / "service").iterdir())
    assert "PILOT: в service/ не попали" in caplog.text and "prompts.json" in caplog.text
    files_txt = (art / "FILES.txt").read_text(encoding="utf-8")
    assert "run/summary.md" in files_txt and "FILES.txt" not in files_txt
    for p in art.rglob("*"):
        if p.is_file() and p.suffix != ".zip":
            _no_secrets(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Этап 6а: слоты, назначение клипов, поколения, сборка воркера, run_id, onstart (SPEC 8.2, 8.3, 8.4)
# ---------------------------------------------------------------------------

def _ready_slot(rig, index, *, instance_id, machine_id=11, offer_id=1, price=0.3, generation=1):
    """Карта в состоянии ready с адресом воркера, как после healthcheck (для тестов отправки)."""
    slot = orch._CardSlot(index, generation, instance_id, machine_id, price, rig.clock.now, offer_id=offer_id)
    slot.state = "ready"
    slot.contacted = True
    slot.url = "http://10.0.0.1:18000"
    rig.orc._slots[index] = slot
    rig.orc.budget.track_instance_start(instance_id, price, rig.clock.now)
    return slot


def _prompt(rig, num=1):
    rig.orc._prompts = {str(num): {"status": "ready", "prompt": f"промпт {num}", "num_frames": 49}}


def test_one_clip_is_never_assigned_to_two_slots(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,), max_cards=2)
    _prompt(rig)
    s1 = _ready_slot(rig, 1, instance_id=501)
    s2 = _ready_slot(rig, 2, instance_id=502, offer_id=2, machine_id=22)
    rig.orc._queue = deque([1])

    rig.orc._dispatch(s1, rig.clock.now)
    assert s1.task is not None and s1.task.num == 1

    # клип снова оказался в очереди, пока назначен первому слоту: второй слот его не берёт
    rig.orc._queue.append(1)
    rig.orc._dispatch(s2, rig.clock.now)
    assert s2.task is None
    assert rig.worker.posted == [1]                       # одна отправка на воркер, дубля нет
    holders = [s for s in (s1, s2) if s.task is not None and s.task.num == 1]
    assert holders == [s1]
    assert rig.orc._assigned_slot(1) is s1


def test_retired_slot_returns_its_clip_to_free(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,), max_cards=2)
    _prompt(rig)
    s1 = _ready_slot(rig, 1, instance_id=501)
    rig.orc._queue = deque([1])
    rig.orc._dispatch(s1, rig.clock.now)

    rig.orc._retire(s1, "тест")
    assert list(rig.orc._queue) == [1]                    # клип free и снова в очереди
    assert rig.orc._assigned_slot(1) is None
    assert 501 in rig.vast.destroyed

    s2 = _ready_slot(rig, 2, instance_id=502, offer_id=2, machine_id=22)
    rig.orc._dispatch(s2, rig.clock.now)
    assert s2.task is not None and s2.task.num == 1
    assert rig.worker.posted == [1, 1]                    # вторая отправка после возврата, не дубль в слот
    assert rig.orc._attempts[1] == 1                      # возврат из-за снятия не тратит попытку


def test_done_clip_is_never_sent_again(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    _prompt(rig)
    s1 = _ready_slot(rig, 1, instance_id=501)
    rig.orc._done.add(1)
    rig.orc._queue = deque([1])
    rig.orc._dispatch(s1, rig.clock.now)
    assert rig.worker.posted == [] and s1.task is None


def test_failed_or_done_clip_is_not_requeued_by_fail_clip(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.orc._done.add(1)
    rig.orc._fail_clip(1, "тест")
    assert list(rig.orc._queue) == []


@pytest.mark.parametrize("max_cards", [1, 3, 101])
def test_slots_start_at_one_and_generation_counts_replacements(tmp_path, monkeypatch, max_cards):
    # разные предложения на каждую аренду: предложение в этом запуске не выбирается повторно (9е-3)
    offers = [make_offer(i, 10 + i) for i in range(1, max_cards + 3)]
    rig = Rig(tmp_path, monkeypatch, missing=(1,), max_cards=max_cards, budget_limit_usd=100000.0,
              offers=offers, max_rent_tries=max_cards + 2)
    for _ in range(max_cards):
        assert rig.orc._rent_one(rig.clock.now) == "ok"
    assert sorted(rig.orc._slots) == list(range(1, max_cards + 1))
    assert all(s.generation == 1 for s in rig.orc._slots.values())
    # пересоздание слота 1 даёт следующее поколение того же слота
    old = rig.orc._slots[1]
    rig.orc._retire(old, "тест")
    assert rig.orc._rent_one(rig.clock.now) == "ok"
    new = rig.orc._slots[1]
    assert new.generation == 2 and new.instance_id != old.instance_id
    assert sorted(rig.orc._slots) == list(range(1, max_cards + 1))


def test_card_name_in_logs_is_slot_and_generation(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[make_offer(1, 11), make_offer(2, 22)])
    assert rig.orc._rent_one(rig.clock.now) == "ok"
    assert "Арендована карта 1 поколение 1 (экземпляр 101)" in caplog.text
    rig.orc._retire(rig.orc._slots[1], "тест")
    assert "Карта 1 поколение 1 (экземпляр 101) снимается: тест." in caplog.text
    assert rig.orc._rent_one(rig.clock.now) == "ok"
    assert "Арендована карта 1 поколение 2" in caplog.text


def test_card_older_than_90_minutes_is_not_retired_by_age(tmp_path, monkeypatch, caplog):
    """Срока жизни 90 минут больше нет: карта живёт, пока её клип не готов (SPEC 8.3 б)."""
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), budget_limit_usd=10.0, task_timeout_sec=1e9,
              worker_kw={"render_sec": 6000.0})
    summary = rig.run()
    assert summary.completed == 1 and summary.exit_code == 0
    assert "истёк" not in caplog.text and "предельный срок" not in caplog.text


def test_lifetime_parameter_and_worker_env_limit_are_removed(tmp_path, monkeypatch):
    with pytest.raises(TypeError):
        Rig(tmp_path, monkeypatch, missing=(1,), card_max_lifetime_min=90)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    assert not hasattr(rig.orc, "_lifetime_sec")
    assert not hasattr(rig.orc, "card_max_lifetime_min")
    rig.run()
    env = rig.vast.created[0]["env"]
    assert "WORKER_MAX_LIFETIME_MIN" not in env
    assert env["WORKER_BUILD"] == VALID_BUILD


def test_worker_env_has_exact_comfyui_args_and_no_forbidden_keys(tmp_path, monkeypatch):
    """SPEC 8.5, 0.1 (5c): единственный ключ COMFYUI_* в env карты воркера — COMFYUI_ARGS с точным значением;
    порталы, Jupyter, данные и провизия образа не передаются; наружу публикуется только 8000."""
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.run()
    env = rig.vast.created[0]["env"]
    assert env["COMFYUI_ARGS"] == "--disable-auto-launch --disable-xformers --port 18188 --enable-cors-header"
    assert [k for k in env if k.startswith("COMFYUI_")] == ["COMFYUI_ARGS"]
    forbidden = ("COMFYUI_API_BASE", "PORTAL_CONFIG", "OPEN_BUTTON_PORT", "OPEN_BUTTON_TOKEN",
                 "JUPYTER_DIR", "DATA_DIRECTORY", "PROVISIONING_COMFYUI_WORKFLOWS")
    for key in forbidden:
        assert key not in env, key
    assert not any(k.startswith(("PORTAL_CONFIG", "OPEN_BUTTON_", "JUPYTER_", "DATA_DIRECTORY"))
                   for k in env)
    assert [k for k in env if k.startswith("-p ")] == ["-p 8000:8000"]


@pytest.mark.parametrize("bad", [
    None, "", "ZZZ", "A1B2C3D4", "abc123", "x" * 40, "a" * 65,
])
def test_invalid_worker_build_is_refused_in_russian(tmp_path, monkeypatch, bad):
    with pytest.raises(orch.gp.GeminiInputError) as ei:
        Rig(tmp_path, monkeypatch, missing=(1,), worker_build=bad)
    assert "--worker-build" in str(ei.value)


@pytest.mark.parametrize("run_id", [None, ""])
def test_run_id_is_required_and_label_is_gen_run_id(tmp_path, monkeypatch, run_id):
    with pytest.raises(orch.gp.GeminiInputError) as ei:
        Rig(tmp_path, monkeypatch, missing=(1,), run_id=run_id)
    assert "run_id" in str(ei.value)
    assert Rig(tmp_path, monkeypatch, missing=(1,)).orc.label == "gen-test"


def test_worker_onstart_has_sha_and_no_secret(tmp_path, monkeypatch):
    canary = "TESTKEY-12345"
    received: list = []

    class Spy(FakeVast):
        def create_instance(self, offer_id, image, disk_gb, env_vars=None, label=None, onstart=None,
                            bid_price=None):
            received.append(onstart)
            return super().create_instance(offer_id, image, disk_gb, env_vars=env_vars, label=label,
                                           onstart=onstart, bid_price=bid_price)

    rig = Rig(tmp_path, monkeypatch, missing=(1,), token=canary)
    spy = Spy([make_offer(1, 11)], rig.events)
    rig.orc.vast = spy
    rig.orc.run()

    onstart = received[0]
    assert VALID_BUILD in onstart
    assert f"https://codeload.github.com/SOTONATORE/Download/tar.gz/{VALID_BUILD}" in onstart
    assert "/opt/worker/src" in onstart and "worker/bootstrap.sh" in onstart
    assert "/opt/worker/bootstrap.log" in onstart and "2>&1 &" in onstart   # в фоне, вывод в файл
    assert canary not in onstart
    assert "set -x" not in onstart and "printenv" not in onstart
    assert re.search(r"\benv\b", onstart) is None
    assert canary in json.dumps(spy.created[0]["env"])      # токен идёт только через окружение карты


def test_worker_onstart_refuses_non_hex_build():
    with pytest.raises(orch.gp.GeminiInputError):
        orch._worker_onstart("x; rm -rf /")


@pytest.mark.parametrize(("interruptible", "expected", "forbidden"), [
    (True, "ставке bid", "обычной аренды"),
    (False, "обычной аренды", "ставке bid"),
])
def test_zero_dph_warning_text_follows_rental_mode(tmp_path, monkeypatch, caplog,
                                                   interruptible, expected, forbidden):
    caplog.set_level(logging.INFO)
    offer = make_offer(1, 11, price=0.3)
    offer.is_interruptible = interruptible
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[offer])
    rig.vast.dph[1] = 0.0
    rig.run()
    warn = [r.getMessage() for r in caplog.records if "не вернул цену" in r.getMessage()]
    assert len(warn) == 1
    assert expected in warn[0] and forbidden not in warn[0]
    assert ("bid" in warn[0]) is interruptible
    assert rig.vast.created[0]["bid_price"] == (0.33 if interruptible else None)


@pytest.mark.parametrize(("interruptible", "price", "expected_bid"), [(True, 0.2, 0.23), (False, 0.45, None)])
def test_budget_price_follows_rental_mode(tmp_path, monkeypatch, interruptible, price, expected_bid):
    offer = make_offer(1, 11, price=price)
    offer.is_interruptible = interruptible
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[offer])
    assert rig.orc._rent_one(rig.clock.now) == "ok"
    rec = rig.orc.budget._records[0]
    sent = expected_bid if interruptible else price
    assert rec.dph == pytest.approx(sent)                       # учёт по цене режима: ставка bid или dph_total
    assert rig.vast.created[0]["bid_price"] == expected_bid     # поле price уходит только при bid

# ---------------------------------------------------------------------------
# Ставка bid и лимит max_price_per_hour (этап 9е-1; как vast_stable_run.sh: PRICE = round(BID + MARGIN, 3))
# ---------------------------------------------------------------------------

def test_bid_is_min_bid_plus_margin(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[make_offer(1, 11, price=0.2)])
    assert rig.orc._rent_one(rig.clock.now) == "ok"
    assert rig.vast.created[0]["bid_price"] == 0.23


def test_bid_is_rounded_to_three_decimals_as_in_script(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[make_offer(1, 11, price=0.2345)])
    assert rig.orc._rent_one(rig.clock.now) == "ok"
    # round(float('0.2345') + float('0.03'), 3) == 0.264: та же формула, что PRICE в vast_stable_run.sh
    assert rig.vast.created[0]["bid_price"] == 0.264


def test_bid_margin_comes_from_constructor_parameter(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[make_offer(1, 11, price=0.2)], bid_margin=0.05)
    assert rig.orc._rent_one(rig.clock.now) == "ok"
    assert rig.vast.created[0]["bid_price"] == 0.25


def test_bid_above_max_price_is_skipped_before_request(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[make_offer(1, 11, price=0.54)])
    assert rig.orc._rent_one(rig.clock.now) == "retry"
    assert rig.vast.created == []                               # запрос аренды не отправлен
    assert "Предложение 1 пропущено" in caplog.text
    assert "0.570" in caplog.text and "0.55" in caplog.text     # ставка и лимит в строке лога


def test_skipped_offer_falls_through_to_next_affordable_offer(tmp_path, monkeypatch):
    offers = [make_offer(1, 11, price=0.54), make_offer(2, 22, price=0.3)]
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=offers)
    assert rig.orc._rent_one(rig.clock.now, offers=offers) == "ok"
    assert len(rig.vast.created) == 1
    assert rig.vast.created[0]["bid_price"] == 0.33             # взято второе предложение: 0.3 + 0.03


def test_budget_records_the_bid_that_was_sent(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[make_offer(1, 11, price=0.2)])
    assert rig.orc._rent_one(rig.clock.now) == "ok"
    sent = rig.vast.created[0]["bid_price"]
    assert sent == 0.23
    assert rig.orc.budget._records[0].dph == pytest.approx(sent)
    assert rig.orc._slots[1].price == pytest.approx(sent)


def test_bad_host_is_excluded_for_this_run_only_and_no_file_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), offers=[make_offer(1, 11)])
    slot = _ready_slot(rig, 1, instance_id=501, machine_id=11, offer_id=1)
    rig.orc._retire(slot, "тест", bad=True)
    assert 11 in rig.orc._bad_machines                     # хост исключён на этот запуск
    assert list(tmp_path.rglob("bad_hosts.txt")) == []     # постоянный список не пишется

    (tmp_path / "second").mkdir()
    rig2 = Rig(tmp_path / "second", monkeypatch, missing=(1,), offers=[make_offer(1, 11)])
    assert rig2.orc._bad_machines == set() and rig2.orc._bad_offers == set()
    assert rig2.orc._rent_one(rig2.clock.now) == "ok"      # в новом запуске хост снова доступен
    assert list(tmp_path.rglob("bad_hosts.txt")) == []


# ---------------------------------------------------------------------------
# Этап 6б: мониторинг весов, готовность, heartbeat, принятие задачи (SPEC 8.2, 8.3 а)
# ---------------------------------------------------------------------------

SETUP_FILE = "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
MIB6 = 1024 ** 2


def _slot_with_url(rig, *, instance_id=500, machine_id=11, offer_id=1):
    slot = orch._CardSlot(1, 1, instance_id, machine_id, 0.3, rig.clock.now, offer_id=offer_id)
    slot.url = "http://10.0.0.1:18000"
    rig.orc._slots[1] = slot
    rig.orc.budget.track_instance_start(instance_id, 0.3, rig.clock.now)
    return slot


def test_health_ok_true_makes_card_ready(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    slot = _slot_with_url(rig)
    assert rig.orc._probe_ready(slot) is True
    assert "проверка /health: код 200" in caplog.text
    assert "готов" in caplog.text


@pytest.mark.parametrize("reply", [(200, '{"ok": false}'), (200, "OK"), (503, '{"ok": true}')])
def test_probe_rejects_bare_200_and_false_ok(tmp_path, monkeypatch, caplog, reply):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.worker.health_reply = reply
    slot = _slot_with_url(rig)
    assert rig.orc._probe_ready(slot) is False


def test_network_error_is_logged_and_not_success(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.worker.health_reply = httpx.HTTPError("сброс соединения")
    slot = _slot_with_url(rig)
    assert rig.orc._probe_ready(slot) is False
    assert "запрос не дошёл" in caplog.text


def test_system_stats_alone_does_not_make_card_ready(tmp_path, monkeypatch, caplog):
    """SPEC 8.2 (5c): ответ /system_stats не засчитывается; оркестратор его не запрашивает."""
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.worker.health_reply = (200, '{"ok": false}')
    rig.worker.stats_reply = (200, '{"system": {"os": "linux"}, "devices": []}')
    slot = _slot_with_url(rig)
    assert rig.orc._probe_ready(slot) is False
    assert not any(u.endswith("/system_stats") for u in rig.worker.urls)
    assert "/system_stats" not in caplog.text


def test_readiness_probes_only_worker_health_not_comfy_port(tmp_path, monkeypatch):
    """SPEC 8.2, 8.5 (5c): готовность — только /health воркера; порт 8188 и /system_stats не опрашиваются."""
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    slot = _slot_with_url(rig)
    assert rig.orc._probe_ready(slot) is True
    assert rig.worker.urls and all(u == slot.url + "/health" for u in rig.worker.urls)
    assert not any(":8188" in u or "system_stats" in u for u in rig.worker.urls)


def test_probe_has_no_comfy_port_state(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    slot = _slot_with_url(rig)
    assert not hasattr(slot, "comfy_url") and not hasattr(slot, "comfy_noted")
    assert not hasattr(orch, "COMFY_PORT") and not hasattr(orch, "_probe_comfy")


def test_ssh_and_portal_are_noted_once_per_run(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    slot = _slot_with_url(rig)
    rig.orc._probe_ready(slot)
    rig.orc._probe_ready(slot)
    assert caplog.text.count("SSH-проверка не реализована (SPEC 8.2)") == 1
    assert caplog.text.count("проверка через портал не реализована") == 1


def test_heartbeat_reaches_card_every_interval(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    slot = _ready_slot(rig, 1, instance_id=500)
    for _ in range(6):                       # шесть шагов по 10 с, как в основном цикле
        rig.orc._heartbeat(rig.clock.now)
        rig.clock.sleep(10.0)
    gaps = [b - a for a, b in zip(rig.worker.health_hits, rig.worker.health_hits[1:])]
    assert len(rig.worker.health_hits) == 6
    assert gaps and all(g == 10.0 for g in gaps)
    assert slot.last_ok_utc.endswith("Z")


def test_heartbeat_file_is_written_without_secrets(tmp_path, monkeypatch):
    hb = tmp_path / "heartbeat.json"
    rig = Rig(tmp_path, monkeypatch, missing=(1,), token="TESTKEY-12345", heartbeat_path=hb)
    _ready_slot(rig, 1, instance_id=500)
    rig.orc._heartbeat(rig.clock.now)
    text = hb.read_text(encoding="utf-8")
    record = json.loads(text)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record["utc"])
    assert record["cards"][0]["instance_id"] == 500
    assert "TESTKEY-12345" not in text      # канарейка: токен воркера в файл не попадает
    _no_secrets(text)
    assert "http" not in text, "адрес карты в файл не пишем"


def test_heartbeat_file_absent_by_default(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.orc._heartbeat(rig.clock.now)
    assert list(tmp_path.rglob("heartbeat*.json")) == []


class _SlowGeminiSpy(GeminiSpy):
    """GeminiSpy, который «долго работает» по фейковым часам и запоминает файл heartbeat на первом вызове."""

    def __init__(self, events: list, clock: FakeClock, hb: Path) -> None:
        super().__init__(events)
        self.clock = clock
        self.hb = hb
        self.exists_at_first = None
        self.text_at_first = None

    def __call__(self, *args, **kwargs):
        if self.calls == 0:
            self.exists_at_first = self.hb.exists()
            self.text_at_first = self.hb.read_text(encoding="utf-8") if self.exists_at_first else None
        self.clock.sleep(900.0)              # Gemini занимает 15 минут по фейковым часам, без реального сна
        return super().__call__(*args, **kwargs)


def test_first_heartbeat_exists_before_gemini_call(tmp_path, monkeypatch):
    """SPEC 8.3 а: первый heartbeat пишется до фазы Gemini, наблюдатель не ждёт карту 10 минут зря."""
    hb = tmp_path / "heartbeat.json"
    rig = Rig(tmp_path, monkeypatch, missing=(1,), heartbeat_path=hb)
    spy = _SlowGeminiSpy(rig.events, rig.clock, hb)
    monkeypatch.setattr(orch.go, "generate_and_store", spy)
    rig.run()
    assert spy.calls >= 1
    assert spy.exists_at_first is True, "heartbeat должен существовать к первому вызову Gemini"
    record = json.loads(spy.text_at_first)
    assert record["cards"] == [], "на старте карт ещё нет"


def test_first_heartbeat_has_no_secrets_no_addresses_and_utc_format(tmp_path, monkeypatch):
    hb = tmp_path / "heartbeat.json"
    rig = Rig(tmp_path, monkeypatch, missing=(1,), token="TESTKEY-12345", heartbeat_path=hb)
    spy = _SlowGeminiSpy(rig.events, rig.clock, hb)
    monkeypatch.setattr(orch.go, "generate_and_store", spy)
    rig.run()
    text = spy.text_at_first
    assert text is not None
    record = json.loads(text)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record["utc"])
    assert "TESTKEY-12345" not in text, "канарейка: токен воркера в файл не попадает"
    _no_secrets(text)
    assert "http" not in text and "10.0.0.1" not in text, "адреса карт в файл не пишем"


def test_speed_below_threshold_in_log_retires_card_and_host(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    slot = _ready_slot(rig, 1, instance_id=500, machine_id=11, offer_id=7)
    rig.worker.logs_text = (f"[setup] [1/5] Начинаю загрузку: {SETUP_FILE}\n"
                            f"[#2089b0 2.1GiB/21GiB(10%) CN:16 DL:20MiB ETA:40s]\n")
    rig.orc._pull_due_logs(rig.clock.now)
    assert rig.orc._slots.get(1) is None
    assert "скорость загрузки весов ниже порога" in caplog.text
    assert 11 in rig.orc._bad_machines and 7 in rig.orc._bad_offers
    assert list(tmp_path.rglob("bad_hosts.txt")) == []     # только на этот запуск


def test_file_error_line_retires_card_with_file_name(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    _ready_slot(rig, 1, instance_id=500, machine_id=11)
    rig.worker.logs_text = f"[setup] [1/5] ОШИБКА: не удалось загрузить файл {SETUP_FILE} (код 4).\n"
    rig.orc._pull_due_logs(rig.clock.now)
    assert rig.orc._slots.get(1) is None
    assert ("ошибка загрузки весов: " + SETUP_FILE) in caplog.text
    assert 11 in rig.orc._bad_machines


def test_unrecognised_weights_format_warns_once(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    _ready_slot(rig, 1, instance_id=500)
    rig.worker.logs_text = f"[setup] [1/5] Начинаю загрузку: {SETUP_FILE}\n[#2089b0 ??? ]\n"
    rig.orc._pull_due_logs(rig.clock.now)
    rig.clock.sleep(60.0)
    rig.orc._pull_due_logs(rig.clock.now)
    assert caplog.text.count("загрузки весов не распознан") == 1


def test_stage_limit_retires_booting_card(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), stage_max_min=20, stall_timeout_min=3)
    slot = _slot_with_url(rig)
    slot.state = "booting"
    mon = rig.orc._monitor(slot)
    mon.feed([f"[setup] [1/5] Начинаю загрузку: {SETUP_FILE}"])
    rig.clock.sleep(20 * 60.0 + 1)
    rig.orc._step_slot(slot, rig.clock.now, False)
    assert rig.orc._slots.get(1) is None
    assert "этап загрузки весов превысил лимит" in caplog.text


def test_worker_ok_false_without_logs_retires_after_stage_cap(tmp_path, monkeypatch, caplog):
    """/health отвечает ok: false и строк загрузки нет: карта живёт до stage_max_min, а не до
    worker_ready_timeout_min (fix 3). Шаг опроса 10 с, таймер весов стартует с первого опроса."""
    caplog.set_level(logging.INFO)
    stage_max_min = 20  # config.default.yaml: stage_max_min: 20 (то же значение по умолчанию в Orchestrator)
    rig = Rig(tmp_path, monkeypatch, missing=(1,), worker_ready_timeout_min=10,
              silent_host_timeout_min=60, stage_max_min=stage_max_min)
    rig.worker.health_reply = (200, '{"ok": false}')
    slot = _booting_slot(rig, instance_id=500, machine_id=11)
    _poll(rig, slot, 70)                      # ~11.7 мин при шаге 10 с: лимит ещё не вышел
    assert rig.orc._slots.get(1) is slot
    assert 11 not in rig.orc._bad_machines
    # далее до stage_max_min + 1 мин запаса; всего (20 * 60 + 60) / 10 = 126 шагов
    _poll(rig, slot, (stage_max_min * 60 + 60) // 10 - 70)
    assert rig.orc._slots.get(1) is None
    assert f"загрузка весов без строк в GET /logs дольше {stage_max_min} мин" in caplog.text
    assert 11 in rig.orc._bad_machines


def test_task_not_accepted_within_timeout_retires_card(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2), task_accept_timeout_min=3)
    rig.orc._queue = deque([1, 2])
    slot = _ready_slot(rig, 1, instance_id=500, machine_id=11, offer_id=3)
    slot.ready_at = rig.clock.now
    assert rig.orc._accept_timed_out(slot, rig.clock.now + 179.0) is False
    assert rig.orc._slots.get(1) is slot
    assert rig.orc._accept_timed_out(slot, rig.clock.now + 181.0) is True
    assert rig.orc._slots.get(1) is None
    assert "задача не принята" in caplog.text
    assert 11 in rig.orc._bad_machines and 3 in rig.orc._bad_offers


def test_accepted_card_is_not_retired_by_accept_timer(tmp_path, monkeypatch, caplog):
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2), task_accept_timeout_min=3)
    rig.orc._prompts = {str(n): {"status": "ready", "prompt": "p", "num_frames": 49} for n in (1, 2)}
    rig.orc._queue = deque([1, 2])
    slot = _ready_slot(rig, 1, instance_id=500)
    slot.ready_at = rig.clock.now - 3600.0
    slot.accepted = True
    rig.orc._step_slot(slot, rig.clock.now, False)
    assert rig.orc._slots.get(1) is slot
    assert "задача не принята" not in caplog.text


def test_per_run_bad_host_is_not_kept_for_next_run(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    slot = _ready_slot(rig, 1, instance_id=500, machine_id=11, offer_id=7)
    rig.worker.logs_text = f"[setup] [1/5] ОШИБКА: не удалось загрузить файл {SETUP_FILE} (код 4).\n"
    rig.orc._pull_due_logs(rig.clock.now)
    rig2 = Rig(tmp_path / "second", monkeypatch, missing=(1,))
    assert rig2.orc._bad_machines == set() and rig2.orc._bad_offers == set()
    assert list(tmp_path.rglob("bad_hosts.txt")) == []


# ---------------------------------------------------------------------------
# Этап 6c: C1 (логи booting-карт), E1-E5 (релизы и артефакт), F1 (порядок отмены)
# ---------------------------------------------------------------------------

def _slot(index=1, generation=1, instance_id=77, url="http://10.0.0.1:18000", state="booting"):
    return orch._CardSlot(index, generation, instance_id, 11, 0.3, 0.0, state=state, url=url)


def test_booting_card_with_url_pulls_logs_and_weights_see_them(tmp_path, monkeypatch, caplog):
    """C1: карта в booting с адресом отдаёт GET /logs; строки загрузки весов доходят до правил."""
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.orc._slots[1] = _slot()
    rig.worker.logs_text = "[setup] [1/5] Начинаю загрузку: diffusion_models/x.safetensors\n"
    rig.orc._pull_due_logs(rig.clock.now)
    assert any(u.endswith("/logs") for u in rig.worker.urls)
    assert "Начинаю загрузку" in caplog.text


def test_booting_card_without_url_is_not_pulled(tmp_path, monkeypatch):
    """C2: без адреса воркера GET /logs не выполняется."""
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.orc._slots[1] = _slot(url="")
    rig.orc._pull_due_logs(rig.clock.now)
    assert not any(u.endswith("/logs") for u in rig.worker.urls)


def test_service_release_opened_and_start_files_uploaded_before_spending(tmp_path, monkeypatch):
    """E1: служебный релиз и missing.txt до аренды; prompts.json в служебный релиз."""
    (tmp_path / "missing.txt").write_text("1\n2\n", encoding="utf-8")
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2))
    rig.run()
    svc = ra.service_tag("test")
    assert (svc, "missing.txt") in rig.store.blobs
    assert (svc, "prompts.json") in rig.store.blobs
    assert rig.events.index("upload:missing.txt") < rig.events.index("vast_create")
    assert rig.store.created[0] == TAG                       # клип-релиз проверяется первым
    assert svc in rig.store.created


def test_service_start_upload_failure_stops_before_any_spending(tmp_path, monkeypatch):
    """E1: сбой загрузки missing.txt в начале: ReleaseStartError (код 2), Vast и Gemini не вызваны."""
    (tmp_path / "missing.txt").write_text("1\n", encoding="utf-8")
    rig = Rig(tmp_path, monkeypatch, missing=(1,))

    def failing(tag, path):
        import os
        if os.path.basename(path) == "missing.txt":
            raise orch.gp.GeminiRuntimeError("Команда gh завершилась с кодом 1: сбой")
        rig.store.releases[tag].assets.append(os.path.basename(path))
    monkeypatch.setattr(rig.store, "upload_asset", failing)
    with pytest.raises(orch.ReleaseStartError) as exc:
        rig.run()
    assert "missing.txt" in str(exc.value) and "трат" in str(exc.value)
    assert rig.vast.search_calls == 0 and rig.vast.created == []
    assert rig.gemini.calls == 0


def test_final_release_links_cover_clip_release_and_parts(tmp_path, monkeypatch):
    """E2: generated_links.txt в итоговом релизе строится по основному релизу и всем частям."""
    rig = Rig(tmp_path, monkeypatch, missing=(2,), preload=["1.mp4"])
    part2 = ra.part_tag(TAG, 2)
    rig.store.releases[part2] = orch.ReleaseInfo(
        tag=part2, title="p2", body=f"hash: {FULL_HASH}\nmodel: ltx25", assets=["5.mp4"])
    rig.run()
    links = rig.store.blobs[(ra.final_tag("test"), "generated_links.txt")].decode()
    assert f"/releases/download/{TAG}/1.mp4" in links
    assert f"/releases/download/{part2}/5.mp4" in links
    assert f"/releases/download/{part2}/2.mp4" in links
    assert (part2, "generated_links.txt") not in rig.store.blobs
    assert not any(t in (TAG, part2) and n in ("prompts.json", "generated_links.txt", "summary.md")
                   for t, n, _ in rig.store.uploads)


def test_upload_failure_gives_exit_3_and_one_line_with_name(tmp_path, monkeypatch, caplog):
    """E4: сбой выгрузки prompts.json: код 3, одна строка с именем файла; отчёт всё равно уходит."""
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    real = rig.store.upload_asset

    def failing(tag, path):
        import os
        if os.path.basename(path) == "prompts.json":
            raise orch.gp.GeminiRuntimeError("Команда gh завершилась с кодом 1: сбой выгрузки")
        real(tag, path)
    monkeypatch.setattr(rig.store, "upload_asset", failing)
    summary = rig.run()
    assert summary.exit_code == 3
    lines = [r for r in caplog.records if "prompts.json" in r.getMessage() and "Не удалось загрузить" in r.getMessage()]
    assert len(lines) == 1
    names = [n for _, n, _ in rig.store.uploads]
    assert "summary.md" in names and "generated_links.txt" in names
    _no_secrets(caplog.text)


def test_artifact_layout_masking_and_no_secrets(tmp_path, monkeypatch, caplog):
    """E5 и 0.1: run/, cards/, service/, FILES.txt; строки замаскированы; ключей нет нигде."""
    caplog.set_level(logging.DEBUG)
    art = tmp_path / "art"
    (tmp_path / "missing.txt").write_text("1\n2\n", encoding="utf-8")
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2), artifact_dir=art)
    rig.worker.logs_text = ("[setup] [1/5] Начинаю загрузку: m.safetensors\n"
                            "[comfyui] Requested to load X\n"
                            f"в логе токен {TOKEN} и ключ {VAST_KEY}\n")
    rig.run()
    assert (art / "run" / "summary.md").is_file()
    assert not (art / "run" / "run.log").exists()  # единственный журнал запуска: run/debug.log (этап 7б)
    assert "Итог" in caplog.text
    card = art / "cards" / "slot1_generation1"
    assert (card / "meta.json").is_file() and (card / "status.txt").is_file()
    wl = (card / "worker_log.txt").read_text(encoding="utf-8")
    assert "[comfyui] Requested to load X" in wl and "[setup] [1/5]" in wl
    assert TOKEN not in wl and VAST_KEY not in wl
    assert (art / "service" / "missing.txt").is_file()
    files_txt = (art / "FILES.txt").read_text(encoding="utf-8")
    assert "run/summary.md" in files_txt and "cards/slot1_generation1/meta.json" in files_txt
    # каждый файл артефакта и каждая загрузка без секретов
    for p in art.rglob("*"):
        if p.is_file() and p.suffix != ".zip":
            _no_secrets(p.read_text(encoding="utf-8"))
    for (_, _), blob in rig.store.blobs.items():
        _no_secrets(blob.decode("utf-8", errors="replace"))
    # журнал запуска и загрузки без клипов в служебном релизе
    assert (ra.service_tag("test"), "run-logs.zip") in rig.store.blobs
    _no_secrets(caplog.text)


def test_canary_in_hf_token_never_reaches_artifact_or_uploads(tmp_path, monkeypatch):
    """0.1 п. 10: канарейка TESTKEY-12345 (HF_TOKEN) не попадает в артефакт и в загрузки."""
    canary = "TESTKEY-12345"
    monkeypatch.setenv("HF_TOKEN", canary)
    art = tmp_path / "art"
    rig = Rig(tmp_path, monkeypatch, missing=(1,), artifact_dir=art)
    rig.worker.logs_text = f"строка с {canary}\n"
    rig.run()
    for p in art.rglob("*"):
        if p.is_file() and p.suffix != ".zip":
            assert canary not in p.read_text(encoding="utf-8")
    for blob in rig.store.blobs.values():
        assert canary.encode() not in blob


def test_cancel_saves_logs_destroys_then_uploads_then_summary_last(tmp_path, monkeypatch, caplog):
    """F1: при остановке порядок: логи и JSON, уничтожение по метке, отчёт и архив, итоговая строка."""
    caplog.set_level(logging.INFO)
    art = tmp_path / "art"
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2), artifact_dir=art)
    rig.orc._sleep = lambda sec: (rig.orc._stop.set(), rig.clock.sleep(sec))  # сигнал после первой итерации
    summary = rig.run()
    assert summary.exit_code == 3
    assert rig.vast.label_destroyed == [LABEL]
    assert (art / "cards" / "slot1_generation1" / "instance.json").is_file()
    ev = rig.events
    assert ev.index("vast_create") < ev.index("destroy_by_label")
    assert ev.index("destroy_by_label") < ev.index("upload:summary.md")
    assert ev.index("upload:summary.md") < ev.index("upload:run-logs.zip")
    assert caplog.records[-1].getMessage().startswith("Итог:")


def test_cancel_log_saving_respects_60_second_limit(tmp_path, monkeypatch, caplog):
    """F1 (1): при исчерпанном лимите 60 с логи карты не запрашиваются, уборка идёт дальше."""
    caplog.set_level(logging.WARNING)
    rig = Rig(tmp_path, monkeypatch, missing=(1,))
    rig.orc._slots[1] = _slot(url="http://10.0.0.1:18000", state="ready")
    rig.orc._cleanup_started = None
    orig_clock = rig.orc.clock
    calls = {"n": 0}

    def fast_clock():
        calls["n"] += 1
        return orig_clock() + (0.0 if calls["n"] == 1 else orch.ARTIFACT_SAVE_LIMIT_SEC + 1)
    rig.orc.clock = fast_clock
    rig.orc._cleanup()
    assert not any(u.endswith("/logs") for u in rig.worker.urls)
    assert "Лимит 60 с на сохранение логов исчерпан" in caplog.text
    assert rig.vast.label_destroyed == [LABEL]



# ---------------------------------------------------------------------------
# Этап 9а: адреса карт, поля экземпляра Vast, значения токенов в артефакте, архиве и релизах
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_known_secrets_after_test():
    """Значения токенов, переданные оркестратором (set_known_secrets), не переживают тест."""
    yield
    orch.set_known_secrets([])


# поля, которых не должно быть в артефакте (SPEC 10.4; этап 9а, п. 1)
FORBIDDEN_FIELD_NAMES = ('"public_ipaddr"', '"direct_port_start"', '"direct_port_end"', '"direct_port_count"',
                         '"ssh_host"', '"ssh_port"', '"env"', '"extra_env"', '"instance_api_key"')


def _leak_markers() -> tuple:
    """Значения, которых не должно быть ни в одном файле артефакта, архиве, загрузке и релизе."""
    return (CARD_IP, f"{CARD_IP}:{CARD_PORT}", str(CARD_PORT), SSH_HOST, str(SSH_PORT), INSTKEY, CANARY)


def _assert_no_markers(text: str, markers, *, where: str) -> None:
    # сообщение не печатает само значение: тестовые секреты не должны попадать и в отчёт теста
    for marker in markers:
        assert marker not in text, f"в {where} найдено значение из списка утечек"


def test_instance_json_has_no_address_port_ssh_or_env(tmp_path, monkeypatch):
    """9а п. 1: instance.json без адреса, портов, ssh, env, extra_env и ключа экземпляра; значения не попадают."""
    art = tmp_path / "art"
    rig = Rig(tmp_path, monkeypatch, missing=(1,), artifact_dir=art)
    rig.vast.raw_env = {"WORKER_TOKEN": CANARY, "HF_TOKEN": CANARY}
    rig.orc._save_instance(_slot(url=f"http://{CARD_IP}:{CARD_PORT}", state="ready"))
    text = (art / "cards" / "slot1_generation1" / "instance.json").read_text(encoding="utf-8")
    parsed = json.loads(text)                                  # JSON цел
    assert parsed["id"] == 77 and parsed["label"] == LABEL     # прочие поля остались
    for field in FORBIDDEN_FIELD_NAMES:
        assert field not in text
    _assert_no_markers(text, _leak_markers(), where="instance.json")


def test_artifact_bundle_and_releases_have_no_card_fields_or_canary(tmp_path, monkeypatch, caplog):
    """9а п. 3, 9, 10: файлы артефакта, run-logs.zip, все загрузки, названия и описания релизов и журнал
    не содержат адреса карты, порта, ssh, полей Vast и канарейку (токен воркера и HF_TOKEN)."""
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("HF_TOKEN", CANARY)      # до создания оркестратора: он передаёт значение маске
    art = tmp_path / "art"
    (tmp_path / "missing.txt").write_text("1\n2\n", encoding="utf-8")
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2), artifact_dir=art, token=CANARY)
    rig.vast.raw_env = {"WORKER_TOKEN": CANARY, "HF_TOKEN": CANARY}
    rig.worker.logs_text = ("[setup] [1/5] Начинаю загрузку: m.safetensors\n"
                            f"адрес карты {CARD_IP}:{CARD_PORT} и токен {CANARY}\n")
    rig.run()
    markers = _leak_markers() + FORBIDDEN_FIELD_NAMES
    # 1. файлы артефакта на диске (кроме архива)
    for p in art.rglob("*"):
        if p.is_file() and p.suffix != ".zip":
            _assert_no_markers(p.read_text(encoding="utf-8"), markers, where=p.name)
    # 2. архив run-logs.zip, который уходит в служебный релиз
    bundle = rig.store.blobs[(ra.service_tag("test"), "run-logs.zip")]
    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        names = zf.namelist()
        assert names and "run/summary.md" in names
        for name in names:
            _assert_no_markers(zf.read(name).decode("utf-8", errors="replace"), markers, where=f"архив: {name}")
    # 3. все загрузки в релизы
    for (_tag, name), blob in rig.store.blobs.items():
        _assert_no_markers(blob.decode("utf-8", errors="replace"), markers, where=f"загрузка {name}")
    # 4. названия и описания релизов
    for rel in rig.store.releases.values():
        _assert_no_markers(f"{rel.title}\n{rel.body}", markers, where=f"релиз {rel.tag}")
    # 5. журнал
    _assert_no_markers(caplog.text, markers, where="журнал")


def test_token_values_reach_mask_without_name_or_shape(tmp_path, monkeypatch):
    """9а п. 4: оркестратор передаёт токен воркера и HF_TOKEN в модуль маски; значение без имени и без формы маскируется."""
    monkeypatch.setenv("HF_TOKEN", "HF-VALUE-PLAIN-12345")
    Rig(tmp_path, monkeypatch, token=CANARY)
    line = f"свободный текст: {CANARY} и HF-VALUE-PLAIN-12345 конец"
    out = orch.mask_remote_text(line)
    assert CANARY not in out and "HF-VALUE-PLAIN-12345" not in out and "***" in out


def test_bundle_excludes_file_with_card_address_and_logs_once(tmp_path, caplog):
    """9а п. 3: файл, где адрес карты остался (записан до регистрации адреса), в архив не идёт; одна строка в журнал."""
    caplog.set_level(logging.WARNING)
    art = orch.RunArtifact(tmp_path / "art", lambda text: text)     # без маски: адрес ещё не известен
    art.card_line(_slot(), f"строка до регистрации адреса {CARD_IP}")
    art.add_address(CARD_IP)
    art.write_report("итог без адреса")
    dest = tmp_path / "run-logs.zip"
    art.make_bundle(str(dest))
    with zipfile.ZipFile(dest) as zf:
        names = zf.namelist()
    assert "run/summary.md" in names
    assert "cards/slot1_generation1/worker_log.txt" not in names
    notes = [r for r in caplog.records if "worker_log.txt" in r.getMessage()]
    assert len(notes) == 1


def test_canary_absent_from_status_meta_prompts_and_links(tmp_path, monkeypatch, caplog):
    """9а п. 10 (a–d): канарейка TESTKEY-12345 как WORKER_TOKEN и как HF_TOKEN не попадает в
    status.txt, meta.json, prompts.json (копия для служебного релиза) и generated_links.txt.
    run/debug.log проверяется в test_logging.py (файловый обработчик cli)."""
    canary = "TESTKEY-12345"
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("HF_TOKEN", canary)            # до создания оркестратора: он передаёт значение маске
    art = tmp_path / "art"
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2), artifact_dir=art, token=canary)
    rig.vast.raw_env = {"WORKER_TOKEN": canary, "HF_TOKEN": canary}
    rig.worker.logs_text = f"строка с {canary} без имени\n"
    rig.run()
    card = art / "cards" / "slot1_generation1"
    status = (card / "status.txt").read_text(encoding="utf-8")
    meta = (card / "meta.json").read_text(encoding="utf-8")
    prompts = (art / "service" / "prompts.json").read_text(encoding="utf-8")
    links = rig.store.blobs[(ra.final_tag("test"), "generated_links.txt")].decode("utf-8")
    assert status.strip() and meta.strip() and prompts.strip()        # файлы не пустые: проверка значима
    assert "https://github.com/" in links
    for name, text in (("status.txt", status), ("meta.json", meta),
                       ("prompts.json", prompts), ("generated_links.txt", links)):
        assert canary not in text, f"канарейка найдена в {name}"
