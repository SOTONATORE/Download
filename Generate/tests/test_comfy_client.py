"""Офлайн-тесты клиента ComfyUI (httpx.MockTransport, без сети)."""
from __future__ import annotations

import json
import socket

import httpx
import pytest

try:
    from Generate.worker import comfy_client as cc
except ImportError:  # pragma: no cover
    from worker import comfy_client as cc  # type: ignore


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def _blocked(*a, **k):
        raise AssertionError("Сетевые обращения в тестах запрещены")
    monkeypatch.setattr(socket.socket, "connect", _blocked)


GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 20}},
    "5": {"class_type": "Latent", "inputs": {"length": 1, "width": 1920}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "старое"}},
    "9": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "old"}},
}
BINDINGS = {
    "prompt": {"node_id": "6", "field": "text"},
    "num_frames": {"node_id": "5", "field": "length"},
    "seed": {"node_id": "3", "field": "seed"},
    "filename_prefix": {"node_id": "9", "field": "filename_prefix"},
    "steps": {"node_id": "3", "field": "steps"},
}


def make_client(handler, **kw):
    return cc.ComfyClient(client=httpx.Client(transport=httpx.MockTransport(handler)), **kw)


def test_inject_workflow_replaces_values_and_copies():
    out = cc.inject_workflow(GRAPH, BINDINGS, "новый", 121, 42, "clip_7", extra_inputs={"steps": 30})
    assert out["6"]["inputs"]["text"] == "новый"
    assert out["5"]["inputs"]["length"] == 121
    assert out["3"]["inputs"]["seed"] == 42
    assert out["3"]["inputs"]["steps"] == 30
    assert out["9"]["inputs"]["filename_prefix"] == "clip_7"
    assert GRAPH["6"]["inputs"]["text"] == "старое"  # исходный граф не изменён
    assert out["5"]["inputs"]["width"] == 1920


def test_inject_workflow_errors_in_russian():
    with pytest.raises(cc.ComfyError, match="обязательных"):
        cc.inject_workflow(GRAPH, {"prompt": BINDINGS["prompt"]}, "x", 1, 1, "p")
    bad = dict(BINDINGS, seed={"node_id": "99", "field": "seed"})
    with pytest.raises(cc.ComfyError, match="нет узла"):
        cc.inject_workflow(GRAPH, bad, "x", 1, 1, "p")
    with pytest.raises(cc.ComfyError, match="нет привязки"):
        cc.inject_workflow(GRAPH, BINDINGS, "x", 1, 1, "p", extra_inputs={"cfg": 1})


def test_check_health_ok():
    stats = {"system": {"os": "posix"}, "devices": [{"name": "RTX", "vram_free": 100}]}
    def h(req):
        assert req.method == "GET" and req.url.path == "/system_stats"
        return httpx.Response(200, json=stats)
    with make_client(h) as c:
        assert c.check_health()["devices"][0]["name"] == "RTX"


def test_check_health_connection_error():
    def h(req):
        raise httpx.ConnectError("refused")
    with make_client(h) as c, pytest.raises(cc.ComfyConnectionError, match="недоступен"):
        c.check_health()


def test_queue_prompt_payload_and_id():
    seen = {}
    def h(req):
        seen["body"] = json.loads(req.content)
        assert req.method == "POST" and req.url.path == "/prompt"
        return httpx.Response(200, json={"prompt_id": "abc", "number": 1})
    with make_client(h) as c:
        assert c.queue_prompt(GRAPH) == "abc"
    assert seen["body"] == {"prompt": GRAPH}


def test_queue_prompt_rejected():
    def h(req):
        return httpx.Response(400, json={"error": {"message": "плохой граф"}})
    with make_client(h) as c, pytest.raises(cc.ComfyExecutionError, match="отклонил"):
        c.queue_prompt(GRAPH)


def test_queue_prompt_no_id():
    with make_client(lambda r: httpx.Response(200, json={})) as c, pytest.raises(cc.ComfyError, match="prompt_id"):
        c.queue_prompt(GRAPH)


def test_wait_success_extracts_video(caplog):
    calls = {"n": 0}
    done = {"p1": {"status": {"status_str": "success", "completed": True},
                   "outputs": {"9": {"images": [{"filename": "p.png"}],
                                     "gifs": [{"filename": "clip_00007.mp4", "subfolder": "vid",
                                               "type": "output"}]}}}}
    def h(req):
        calls["n"] += 1
        return httpx.Response(200, json={} if calls["n"] < 3 else done)
    with caplog.at_level("INFO"), make_client(h) as c:
        path = c.wait_for_completion("p1", timeout=5, poll_interval=0)
    assert path == "vid/clip_00007.mp4"
    text = caplog.text
    assert "очередь" in text and "сэмплирование" in text and "завершено" in text


def test_wait_execution_error():
    err = {"p2": {"status": {"status_str": "error", "messages": [
        ["execution_error", {"exception_message": "Нехватка памяти", "node_id": "3"}]]}}}
    with make_client(lambda r: httpx.Response(200, json=err)) as c, \
            pytest.raises(cc.ComfyExecutionError, match="Нехватка памяти"):
        c.wait_for_completion("p2", timeout=5, poll_interval=0)


def test_wait_no_video_in_outputs():
    done = {"p3": {"status": {"status_str": "success", "completed": True}, "outputs": {}}}
    with make_client(lambda r: httpx.Response(200, json=done)) as c, \
            pytest.raises(cc.ComfyExecutionError, match="видеофайл"):
        c.wait_for_completion("p3", timeout=5, poll_interval=0)


def test_wait_timeout():
    with make_client(lambda r: httpx.Response(200, json={})) as c, \
            pytest.raises(cc.ComfyTimeoutError, match="не завершилась"):
        c.wait_for_completion("p4", timeout=0.05, poll_interval=0.01)


def test_context_manager_closes_owned_client():
    c = cc.ComfyClient()
    with c:
        pass
    assert c._client.is_closed
    ext = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with cc.ComfyClient(client=ext):
        pass
    assert not ext.is_closed
    ext.close()
