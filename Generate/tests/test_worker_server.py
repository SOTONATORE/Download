"""Офлайн-тесты HTTP-сервиса воркера (SPEC 8.1, 8.3(а), 8.4). ComfyUI подменён."""
from __future__ import annotations

import logging
import threading
import time

import httpx
import pytest

try:
    from Generate.worker.comfy_client import ComfyExecutionError
    from Generate.worker.server import DeadManSwitch, WorkerApp, parse_range
except ImportError:
    from worker.comfy_client import ComfyExecutionError
    from worker.server import DeadManSwitch, WorkerApp, parse_range

TOKEN = "SECRET-TOKEN-123"
WRONG = "WRONG-TOKEN-999"
PAYLOAD_BYTES = bytes(range(256)) * 40  # 10240 байт
WORKFLOW = {"1": {"inputs": {}}}
BINDINGS = {
    "prompt": {"node_id": "1", "field": "text"},
    "num_frames": {"node_id": "1", "field": "frames"},
    "seed": {"node_id": "1", "field": "seed"},
    "filename_prefix": {"node_id": "1", "field": "prefix"},
}
TASK = {"num": 7, "prompt": "кот на крыше", "num_frames": 97, "seed": 42}


class FakeComfy:
    def __init__(self, out_dir, gate=None, fail=None):
        self.out_dir = out_dir
        self.gate = gate if gate is not None else threading.Event()
        if gate is None:
            self.gate.set()
        self.fail = fail
        self.started = threading.Event()
        self.graphs = []
        self._n = 0

    def check_health(self):
        return {"devices": [{"name": "FakeGPU", "vram_free": 12345, "vram_total": 99999}]}

    def queue_prompt(self, graph):
        self.graphs.append(graph)
        return f"pid-{len(self.graphs)}"

    def wait_for_completion(self, prompt_id, timeout=600.0, poll_interval=1.0):
        self.started.set()
        self.gate.wait(10)
        if self.fail:
            raise ComfyExecutionError(self.fail)
        self._n += 1
        sub = self.out_dir / "sub"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"clip_{self._n}.mp4").write_bytes(PAYLOAD_BYTES)
        return f"sub/clip_{self._n}.mp4"


@pytest.fixture
def make_app(tmp_path):
    apps = []

    def factory(comfy=None, **kw):
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        comfy = comfy or FakeComfy(out)
        kw.setdefault("idle_timeout_seconds", 3600)
        kw.setdefault("on_idle_timeout", lambda: None)
        app = WorkerApp(TOKEN, comfy, host="127.0.0.1", port=0, workflow=WORKFLOW,
                        bindings=BINDINGS, output_dir=out, **kw)
        app.start_background()
        apps.append(app)
        return app, comfy

    yield factory
    for a in apps:
        a.close()


def client_for(app, token=TOKEN):
    headers = {"X-Worker-Token": token} if token else {}
    return httpx.Client(base_url=f"http://127.0.0.1:{app.port}", headers=headers,
                        trust_env=False, timeout=10)


def wait_state(cli, task_id, state, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        body = cli.get(f"/task/{task_id}").json()
        if body["state"] == state:
            return body
        time.sleep(0.02)
    raise AssertionError(f"Состояние {state} не достигнуто, последнее: {body}")


def submit_done(cli, **over):
    resp = cli.post("/task", json={**TASK, **over})
    assert resp.status_code == 200
    tid = resp.json()["task_id"]
    wait_state(cli, tid, "done")
    return tid


# --- аутентификация ---

def test_auth_missing_invalid_valid(make_app):
    app, _ = make_app()
    with client_for(app, None) as c:
        assert c.get("/health").status_code == 401
    with client_for(app, WRONG) as c:
        assert c.get("/health").status_code == 403
    with client_for(app) as c:
        assert c.get("/health").status_code == 200


def test_invalid_token_does_not_create_task(make_app):
    app, comfy = make_app()
    with client_for(app, WRONG) as c:
        assert c.post("/task", json=TASK).status_code == 403
    assert comfy.graphs == []


def test_unknown_route_and_method(make_app):
    app, _ = make_app()
    with client_for(app) as c:
        assert c.get("/nope").status_code == 404
        assert c.get("/task").status_code == 405


# --- задачи ---

def test_task_transitions_and_injection(make_app, tmp_path):
    gate = threading.Event()
    app, comfy = make_app(FakeComfy(tmp_path / "out", gate=gate))
    with client_for(app) as c:
        tid = c.post("/task", json=TASK).json()["task_id"]
        assert comfy.started.wait(5)
        running = wait_state(c, tid, "running")
        assert running["size_bytes"] is None and running["error"] is None
        assert isinstance(running["stage"], str) and "log_tail" in running
        gate.set()
        done = wait_state(c, tid, "done")
        assert done["size_bytes"] == len(PAYLOAD_BYTES)
        assert done["error"] is None
    inputs = comfy.graphs[0]["1"]["inputs"]
    assert inputs["text"] == TASK["prompt"]
    assert inputs["frames"] == 97 and inputs["seed"] == 42
    assert inputs["prefix"].startswith("clip_7_")


def test_task_error_state(make_app, tmp_path):
    app, _ = make_app(FakeComfy(tmp_path / "out", fail="сбой сэмплера"))
    with client_for(app) as c:
        tid = c.post("/task", json=TASK).json()["task_id"]
        body = wait_state(c, tid, "error")
        assert "сбой сэмплера" in body["error"]
        assert c.get(f"/file/{tid}").status_code == 404


@pytest.mark.parametrize("bad", [
    {"num": "1", "prompt": "x", "num_frames": 9, "seed": 1},
    {"num": 1, "prompt": "", "num_frames": 9, "seed": 1},
    {"num": 1, "prompt": "x", "num_frames": 0, "seed": 1},
    {"num": 1, "prompt": "x", "num_frames": 9},
])
def test_invalid_payload(make_app, bad):
    app, _ = make_app()
    with client_for(app) as c:
        assert c.post("/task", json=bad).status_code == 400


def test_unknown_task_404(make_app):
    app, _ = make_app()
    with client_for(app) as c:
        assert c.get("/task/unknown").status_code == 404


def test_backlog_429_and_release_after_ack(make_app):
    app, _ = make_app()
    with client_for(app) as c:
        ids = [submit_done(c, num=i) for i in range(3)]
        resp = c.post("/task", json=TASK)
        assert resp.status_code == 429
        assert resp.json()["error"] == "Очередь заполнена, ожидается скачивание готовых клипов"
        assert c.post(f"/ack/{ids[0]}").json() == {"ok": True}
        assert c.post("/task", json=TASK).status_code == 200


# --- файлы ---

def test_file_full_and_ranges(make_app):
    app, _ = make_app()
    with client_for(app) as c:
        tid = submit_done(c)
        full = c.get(f"/file/{tid}")
        assert full.status_code == 200
        assert full.content == PAYLOAD_BYTES
        assert full.headers["content-length"] == str(len(PAYLOAD_BYTES))

        part = c.get(f"/file/{tid}", headers={"Range": "bytes=100-199"})
        assert part.status_code == 206
        assert part.content == PAYLOAD_BYTES[100:200]
        assert part.headers["content-range"] == f"bytes 100-199/{len(PAYLOAD_BYTES)}"

        tail = c.get(f"/file/{tid}", headers={"Range": "bytes=10000-"})
        assert tail.status_code == 206 and tail.content == PAYLOAD_BYTES[10000:]

        suffix = c.get(f"/file/{tid}", headers={"Range": "bytes=-50"})
        assert suffix.status_code == 206 and suffix.content == PAYLOAD_BYTES[-50:]

        clipped = c.get(f"/file/{tid}", headers={"Range": "bytes=10000-99999"})
        assert clipped.headers["content-range"] == "bytes 10000-10239/10240"

        bad = c.get(f"/file/{tid}", headers={"Range": "bytes=99999-"})
        assert bad.status_code == 416
        assert bad.headers["content-range"] == f"bytes */{len(PAYLOAD_BYTES)}"


def test_file_not_ready_and_unknown(make_app, tmp_path):
    gate = threading.Event()
    app, comfy = make_app(FakeComfy(tmp_path / "out", gate=gate))
    with client_for(app) as c:
        tid = c.post("/task", json=TASK).json()["task_id"]
        assert comfy.started.wait(5)
        assert c.get(f"/file/{tid}").status_code == 404
        assert c.get("/file/unknown").status_code == 404
        gate.set()


def test_ack_deletes_file(make_app, tmp_path):
    app, _ = make_app()
    with client_for(app) as c:
        tid = submit_done(c)
        files = list((tmp_path / "out").rglob("*.mp4"))
        assert len(files) == 1 and files[0].exists()
        assert c.post(f"/ack/{tid}").json() == {"ok": True}
        assert not files[0].exists()
        assert c.get(f"/file/{tid}").status_code == 404
        assert c.post(f"/ack/{tid}").status_code == 200  # идемпотентно
        assert c.post("/ack/unknown").status_code == 404


def test_health(make_app):
    app, _ = make_app()
    with client_for(app) as c:
        body = c.get("/health").json()
    assert body == {"ok": True, "gpu": "FakeGPU", "vram_free": 12345, "busy": False}


# --- Dead-Man Switch ---

def test_deadman_unit_with_fake_clock():
    now = [0.0]
    fired = []
    dms = DeadManSwitch(300, lambda: fired.append(1), clock=lambda: now[0])
    now[0] = 299
    assert dms.check() is False
    dms.touch()
    now[0] = 598
    assert dms.check() is False  # таймер сброшен
    now[0] = 700
    assert dms.check() is True and dms.check() is True
    assert fired == [1]  # сработал ровно один раз


def test_deadman_fires_after_inactivity(make_app):
    fired = threading.Event()
    app, _ = make_app(idle_timeout_seconds=0.3, check_interval=0.05,
                      on_idle_timeout=fired.set)
    assert fired.wait(3)


def test_deadman_reset_by_authenticated_request_only(make_app):
    fired = threading.Event()
    app, _ = make_app(idle_timeout_seconds=0.6, check_interval=0.05,
                      on_idle_timeout=fired.set)
    with client_for(app) as good, client_for(app, WRONG) as bad:
        for _ in range(8):  # ~0.96 с > таймаута, но запросы валидны
            assert good.get("/health").status_code == 200
            time.sleep(0.12)
        assert not fired.is_set()
        for _ in range(8):  # неавторизованные запросы таймер не сбрасывают
            assert bad.get("/health").status_code == 403
            time.sleep(0.12)
    assert fired.wait(3)


def test_default_callback_stops_server(tmp_path):
    app = WorkerApp(TOKEN, FakeComfy(tmp_path), host="127.0.0.1", port=0,
                    idle_timeout_seconds=0.2, check_interval=0.05)
    thread = threading.Thread(target=app.httpd.serve_forever, daemon=True)
    app.manager.start()
    app.deadman.start()
    thread.start()
    thread.join(timeout=3)
    assert not thread.is_alive()
    app.close()


# --- безопасность токена ---

def test_token_never_leaks(make_app, tmp_path, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    app, _ = make_app()
    bodies = []
    with client_for(app) as good, client_for(app, WRONG) as bad, client_for(app, None) as none:
        bodies += [bad.get("/health").text, none.get("/health").text,
                   bad.post("/task", json=TASK).text, good.get("/nope").text]
        tid = submit_done(good)
        bodies += [good.get(f"/task/{tid}").text, good.post(f"/ack/{tid}").text]
    app2, _ = make_app(FakeComfy(tmp_path / "out", fail="сбой"))
    with client_for(app2) as c:
        tid = c.post("/task", json=TASK).json()["task_id"]
        wait_state(c, tid, "error")
        bodies.append(c.get(f"/task/{tid}").text)
    out = capsys.readouterr()
    for text in (caplog.text, out.out, out.err, *bodies):
        assert TOKEN not in text
        assert WRONG not in text


def test_empty_token_refused(tmp_path):
    with pytest.raises(ValueError):
        WorkerApp("", FakeComfy(tmp_path), host="127.0.0.1", port=0)


# --- разбор Range ---

@pytest.mark.parametrize("header,expected", [
    (None, None), ("bytes=0-9", (0, 9)), ("bytes=5-", (5, 99)), ("bytes=-10", (90, 99)),
    ("bytes=90-500", (90, 99)), ("garbage", None), ("bytes=0-1,5-6", None),
])
def test_parse_range(header, expected):
    assert parse_range(header, 100) == expected
