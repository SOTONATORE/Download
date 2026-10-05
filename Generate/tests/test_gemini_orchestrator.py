"""Тесты core/gemini_orchestrator (этап 4б). Сеть не используется, время подменено."""
import io
import json
import logging
import os
import pathlib
import socket

import pytest

from Generate.core import gemini_orchestrator as go
from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import GeminiInputError, GeminiRuntimeError, PromptConfig
from Generate.core.srt_parser import Segment

ENV_NAME = "GEN_" + "GEMINI_API_KEY"
TESTKEY = "TESTKEY-12345"
M1, M2 = "model-a", "model-b"
TIMING = go.TimingParams(24, {"kind": "8k+1"}, 2.5, 5.0, 10.0, 0.5)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_json(self, model, system, user, schema):
        self.calls.append(model)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)


def segs(n=4):
    return [Segment(i, i * 3000, i * 3000 + 1000, f"Sentence number {i}.") for i in range(1, n + 1)]


def cfg(tmp_path, **kw):
    style = tmp_path / "style.md"
    style.write_text("# Style\nWrite one paragraph.\n", encoding="utf-8")
    kw.setdefault("style_file", str(style))
    kw.setdefault("max_prompt_chars", 300)
    kw.setdefault("batch_size", 10)
    return PromptConfig(**kw)


def ok(nums, text="A wide shot of a quiet street at dawn."):
    return [{"segment_index": n, "prompt": text} for n in nums]


def quota():
    return GeminiRuntimeError("суточная квота", status=429, code="RESOURCE_EXHAUSTED", daily_quota=True)


def rpm429(retry_after=7.0):
    return GeminiRuntimeError("лимит в минуту", status=429, code="RESOURCE_EXHAUSTED", retry_after=retry_after)


def run(tmp_path, transport, wanted, segments=None, clock=None, path=None, **kw):
    clock = clock or FakeClock()
    path = path or tmp_path / "prompts.json"
    kw.setdefault("models", (M1, M2))
    kw.setdefault("clock", clock.now)
    kw.setdefault("sleep", clock.sleep)
    c = kw.pop("cfg", None) or cfg(tmp_path)
    return go.generate_and_store(segments or segs(), wanted, c, transport, path,
                                 profile="ltx25", timing=TIMING, **kw)


def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ состояние

def test_fresh_run_writes_full_entries(tmp_path):
    t = FakeTransport([ok([1, 2])])
    state = run(tmp_path, t, [1, 2])
    data = read(tmp_path / "prompts.json")
    assert data == json.loads(json.dumps(state))
    assert data["profile"] == "ltx25" and data["model"] == M1 and len(data["srt_hash"]) == 64
    e = data["prompts"]["1"]
    assert set(e) == {"num", "start_ms", "end_ms", "clip_ms", "num_frames", "prompt",
                      "status", "attempts", "error", "needs_review", "extra"}
    # сегмент 1: raw=3000 мс -> 72 кадра -> по правилу 8k+1 = 73; 73/24 = 3041.67 -> 3042
    assert (e["start_ms"], e["end_ms"], e["num_frames"], e["clip_ms"]) == (3000, 4000, 73, 3042)
    assert e["status"] == "ready" and e["attempts"] == 1 and not e["needs_review"]


def test_hash_mismatch_raises_before_transport(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"srt_hash": "0" * 64, "model": M1, "profile": "x", "prompts": {}}))
    before = path.read_bytes()
    t = FakeTransport([])
    with pytest.raises(GeminiInputError) as ei:
        run(tmp_path, t, [1, 2], path=path)
    assert ei.value.exit_code == 2 and "SRT" in str(ei.value) and "другого" in str(ei.value)
    assert t.calls == [] and path.read_bytes() == before


def test_corrupt_file_is_input_error(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text("{not json")
    with pytest.raises(GeminiInputError):
        run(tmp_path, FakeTransport([]), [1], path=path)


def test_resume_sends_only_missing(tmp_path):
    run(tmp_path, FakeTransport([ok([1, 2], "First version of prompt.")]), [1, 2])
    t = FakeTransport([ok([3, 4])])
    run(tmp_path, t, [1, 2, 3, 4])
    assert len(t.calls) == 1
    data = read(tmp_path / "prompts.json")
    assert sorted(map(int, data["prompts"])) == [1, 2, 3, 4]
    assert data["prompts"]["1"]["prompt"] == "First version of prompt."
    assert data["prompts"]["1"]["attempts"] == 1


def test_all_present_no_calls_file_untouched(tmp_path):
    run(tmp_path, FakeTransport([ok([1, 2])]), [1, 2])
    path = tmp_path / "prompts.json"
    before = path.read_bytes()
    t = FakeTransport([])
    run(tmp_path, t, [1, 2])
    assert t.calls == [] and path.read_bytes() == before


def test_empty_prompt_entry_is_regenerated(tmp_path):
    run(tmp_path, FakeTransport([["junk"], ["junk"]]), [1])  # оба ответа непригодны
    data = read(tmp_path / "prompts.json")
    assert data["prompts"]["1"]["prompt"] == "" and data["prompts"]["1"]["needs_review"]
    assert data["prompts"]["1"]["status"] != "ready"
    run(tmp_path, FakeTransport([ok([1])]), [1])
    e = read(tmp_path / "prompts.json")["prompts"]["1"]
    assert e["prompt"] and e["status"] == "ready" and e["attempts"] == 2


def test_input_errors_before_transport(tmp_path):
    t = FakeTransport([])
    with pytest.raises(GeminiInputError):
        run(tmp_path, t, [99])
    with pytest.raises(GeminiInputError):
        run(tmp_path, t, [])
    assert t.calls == []


# ------------------------------------------------------------ цепочка моделей

def test_fallback_on_daily_quota(tmp_path):
    t = FakeTransport([quota(), ok([1, 2])])
    run(tmp_path, t, [1, 2])
    assert t.calls == [M1, M2]
    assert read(tmp_path / "prompts.json")["model"] == M2


def test_rpm_429_retries_same_model(tmp_path):
    clock = FakeClock()
    t = FakeTransport([rpm429(7.0), ok([1, 2])])
    run(tmp_path, t, [1, 2], clock=clock)
    assert t.calls == [M1, M1]
    assert 7.0 in clock.sleeps


def test_rpm_429_without_retry_after_uses_backoff(tmp_path):
    clock = FakeClock()
    t = FakeTransport([rpm429(None), ok([1])])
    run(tmp_path, t, [1], clock=clock)
    assert t.calls == [M1, M1] and clock.sleeps and clock.sleeps[0] >= 1


def test_rpm_retries_are_bounded(tmp_path):
    t = FakeTransport([rpm429(1.0)] * 3)
    with pytest.raises(GeminiRuntimeError) as ei:
        run(tmp_path, t, [1], max_rate_retries=2)
    assert ei.value.status == 429 and not ei.value.daily_quota and len(t.calls) == 3


def test_all_models_exhausted_keeps_progress(tmp_path):
    t = FakeTransport([ok([1, 2]), quota(), quota()])
    with pytest.raises(GeminiRuntimeError) as ei:
        run(tmp_path, t, [1, 2, 3, 4], cfg=cfg(tmp_path, batch_size=2))
    assert ei.value.exit_code == 3 and ei.value.daily_quota
    assert t.calls == [M1, M1, M2]
    assert sorted(map(int, read(tmp_path / "prompts.json")["prompts"])) == [1, 2]


def test_rpd_counter_switches_model(tmp_path):
    t = FakeTransport([ok([1, 2]), ok([3, 4])])
    run(tmp_path, t, [1, 2, 3, 4], cfg=cfg(tmp_path, batch_size=2), rpd_limit=1)
    assert t.calls == [M1, M2]


def test_rpd_used_from_previous_runs(tmp_path):
    t = FakeTransport([ok([1])])
    run(tmp_path, t, [1], rpd_used={M1: 500})
    assert t.calls == [M2]


def test_rpd_exhausted_everywhere_before_first_call(tmp_path):
    t = FakeTransport([])
    with pytest.raises(GeminiRuntimeError):
        run(tmp_path, t, [1], rpd_used={M1: 500, M2: 500})
    assert t.calls == []


# ------------------------------------------------------------------- лимитер

def test_15_rpm_throttling():
    c = FakeClock()
    rl = go.RateLimiter(15, 250_000, c.now, c.sleep)
    start = c.t
    for _ in range(15):
        rl.acquire(10)
    assert c.sleeps == []
    rl.acquire(10)
    assert c.t - start >= 60.0 and len(c.sleeps) == 1


def test_window_slides():
    c = FakeClock()
    rl = go.RateLimiter(2, 10_000, c.now, c.sleep)
    rl.acquire(1)
    c.t += 30
    rl.acquire(1)
    rl.acquire(1)          # ждёт, пока уйдёт первое событие (ещё 30 с)
    assert round(c.sleeps[0]) == 30


def test_tpm_throttling():
    c = FakeClock()
    rl = go.RateLimiter(15, 100, c.now, c.sleep)
    rl.acquire(60)
    rl.acquire(30)
    assert c.sleeps == []
    rl.acquire(60)
    assert sum(c.sleeps) >= 60


def test_oversized_request_does_not_deadlock():
    c = FakeClock()
    go.RateLimiter(15, 100, c.now, c.sleep).acquire(10_000)


def test_managed_transport_paces_requests(tmp_path):
    c = FakeClock()
    t = FakeTransport([ok([1]), ok([2]), ok([3])])
    start = c.t
    run(tmp_path, t, [1, 2, 3], clock=c, rpm_limit=2, cfg=cfg(tmp_path, batch_size=1))
    assert len(t.calls) == 3 and c.t - start >= 60.0


# --------------------------------------------------------------- атомарность

def test_atomic_write_leaves_no_tmp(tmp_path):
    path = tmp_path / "p.json"
    go.atomic_write_json(path, {"a": "б"})
    assert read(path) == {"a": "б"} and os.listdir(tmp_path) == ["p.json"]


def test_atomic_write_uses_same_dir_and_replace(tmp_path, monkeypatch):
    seen = []
    real = os.replace

    def spy(src, dst):
        seen.append((os.path.dirname(src), os.path.dirname(dst)))
        real(src, dst)
    monkeypatch.setattr(os, "replace", spy)
    go.atomic_write_json(tmp_path / "p.json", {"x": 1})
    assert seen and seen[0][0] == seen[0][1] == str(tmp_path)


def test_failed_replace_keeps_old_file(tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    go.atomic_write_json(path, {"v": 1})

    def boom(*a):
        raise OSError("сбой")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        go.atomic_write_json(path, {"v": 2})
    assert read(path) == {"v": 1} and os.listdir(tmp_path) == ["p.json"]


def test_incremental_save_before_next_batch(tmp_path):
    path = tmp_path / "prompts.json"
    snapshots = []

    class Spy(FakeTransport):
        def generate_json(self, *a):
            snapshots.append(sorted(map(int, read(path)["prompts"])) if path.exists() else [])
            return super().generate_json(*a)
    run(tmp_path, Spy([ok([1, 2]), ok([3, 4])]), [1, 2, 3, 4], cfg=cfg(tmp_path, batch_size=2))
    assert snapshots == [[], [1, 2]]


# ------------------------------------------------------------- безопасность

def test_key_does_not_leak(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_NAME, TESTKEY)
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.DEBUG)
    lg = logging.getLogger("Generate")
    old = lg.level
    lg.setLevel(logging.DEBUG)
    lg.addHandler(h)
    try:
        t = FakeTransport([ok([1, 2]), quota(), quota()])
        with pytest.raises(GeminiRuntimeError) as ei:
            run(tmp_path, t, [1, 2, 3, 4], cfg=cfg(tmp_path, batch_size=2))
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    assert TESTKEY not in str(ei.value) and TESTKEY not in buf.getvalue()
    assert TESTKEY not in (tmp_path / "prompts.json").read_text(encoding="utf-8")
    assert "A wide shot" not in buf.getvalue()  # тексты промптов в логи не пишутся


def test_orchestrator_source_has_no_key_access():
    src = pathlib.Path(go.__file__).read_text(encoding="utf-8")
    assert "environ" not in src and "getenv" not in src and ENV_NAME not in src
    assert "github" not in src.lower()


def test_messages_are_russian(tmp_path):
    with pytest.raises(GeminiRuntimeError) as ei:
        run(tmp_path, FakeTransport([quota(), quota()]), [1])
    assert any("а" <= ch <= "я" for ch in str(ei.value).lower())
