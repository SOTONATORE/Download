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


# ------------------------------------------------ временные ошибки (500/503/504, сеть)

def unavail(status=503, code="UNAVAILABLE"):
    return GeminiRuntimeError("тестовая временная ошибка сервера", status=status, code=code)


def net_fail():
    return GeminiRuntimeError("Сбой сети при обращении к Gemini (ConnectError).")


SWITCH_TEXT = "переход на модель"


def _switch_lines(caplog):
    return [r.getMessage() for r in caplog.records if SWITCH_TEXT in r.getMessage()]


@pytest.mark.parametrize("err, expected", [
    (unavail(500), True),
    (unavail(503), True),
    (unavail(504), True),
    (net_fail(), True),
    (unavail(502), False),
    (unavail(503, ""), True),
    (GeminiRuntimeError("x", status=503, daily_quota=True), False),
    (GeminiRuntimeError("x", status=429, code="RESOURCE_EXHAUSTED"), False),
    (GeminiRuntimeError("x", status=400), False),
    (GeminiRuntimeError("x", status=401), False),
    (GeminiRuntimeError("x", status=403), False),
    (GeminiRuntimeError("x", status=404), False),
])
def test_is_transient_classification(err, expected):
    assert go.is_transient(err) is expected


def test_503_then_success_same_model_one_warning(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    t = FakeTransport([unavail(), ok([1, 2])])
    state = run(tmp_path, t, [1, 2], clock=clock)
    assert t.calls == [M1, M1]
    assert clock.sleeps == [3.0]
    assert state["model"] == M1 and read(tmp_path / "prompts.json")["model"] == M1
    warnings = [r for r in caplog.records
                if r.name == "Generate.gemini_orchestrator" and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert _switch_lines(caplog) == []


def test_five_503_then_switch_for_rest_of_run(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    t = FakeTransport([unavail() for _ in range(5)] + [ok([1, 2]), ok([3, 4])])
    state = run(tmp_path, t, [1, 2, 3, 4], clock=clock, cfg=cfg(tmp_path, batch_size=2))
    assert t.calls == [M1] * 5 + [M2, M2]
    assert clock.sleeps == [3.0, 6.0, 12.0, 24.0]     # без паузы при смене модели
    assert state["model"] == M2
    lines = _switch_lines(caplog)
    assert len(lines) == 1 and M1 in lines[0] and M2 in lines[0]


def test_after_switch_transient_on_second_model_does_not_go_back(tmp_path):
    clock = FakeClock()
    t = FakeTransport([unavail() for _ in range(5)] + [ok([1, 2]), unavail(), ok([3, 4])])
    run(tmp_path, t, [1, 2, 3, 4], clock=clock, cfg=cfg(tmp_path, batch_size=2))
    assert t.calls[:5] == [M1] * 5
    assert t.calls[5:] == [M2, M2, M2]
    assert read(tmp_path / "prompts.json")["model"] == M2


def test_fresh_request_gets_five_attempts_on_current_model(tmp_path):
    clock = FakeClock()
    replies = [unavail() for _ in range(5)] + [ok([1, 2])] + [unavail() for _ in range(4)] + [ok([3, 4])]
    t = FakeTransport(replies)
    run(tmp_path, t, [1, 2, 3, 4], clock=clock, cfg=cfg(tmp_path, batch_size=2))
    assert t.calls == [M1] * 5 + [M2] * 6
    assert clock.sleeps == [3.0, 6.0, 12.0, 24.0] * 2


def test_503_on_every_model_raises_after_five_per_model(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(ENV_NAME, TESTKEY)
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    t = FakeTransport([unavail() for _ in range(10)])
    with pytest.raises(GeminiRuntimeError) as ei:
        run(tmp_path, t, [1], clock=clock)
    e = ei.value
    assert t.calls == [M1] * 5 + [M2] * 5
    assert e.exit_code == 3 and e.status == 503 and e.code == "UNAVAILABLE" and not e.daily_quota
    assert "временно недоступен на всех моделях" in str(e)
    assert clock.sleeps == [3.0, 6.0, 12.0, 24.0] * 2
    assert TESTKEY not in str(e) and TESTKEY not in caplog.text
    assert not (tmp_path / "prompts.json").exists()   # ни одного батча не готово: файл не создан


def test_last_status_kept_on_final_failure_for_network(tmp_path):
    t = FakeTransport([net_fail() for _ in range(10)])
    with pytest.raises(GeminiRuntimeError) as ei:
        run(tmp_path, t, [1])
    assert ei.value.status is None and "сбой сети" in str(ei.value)


def test_pauses_follow_schedule(tmp_path):
    clock = FakeClock()
    t = FakeTransport([unavail() for _ in range(4)] + [ok([1])])
    run(tmp_path, t, [1], clock=clock)
    assert clock.sleeps == [3.0, 6.0, 12.0, 24.0]


@pytest.mark.parametrize("retry_after, expected", [(2.5, [2.5]), (500.0, [go.MAX_WAIT_SEC])])
def test_transient_pause_honours_retry_after_capped(tmp_path, retry_after, expected):
    clock = FakeClock()
    t = FakeTransport([GeminiRuntimeError("тест", status=503, retry_after=retry_after), ok([1])])
    run(tmp_path, t, [1], clock=clock)
    assert clock.sleeps == expected


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_raise_at_once_without_switch(tmp_path, status, caplog):
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    t = FakeTransport([GeminiRuntimeError("тест", status=status), ok([1])])
    with pytest.raises(GeminiRuntimeError) as ei:
        run(tmp_path, t, [1], clock=clock)
    assert ei.value.status == status
    assert t.calls == [M1] and clock.sleeps == []
    assert _switch_lines(caplog) == []


def test_network_error_is_transient_fake_transport(tmp_path):
    clock = FakeClock()
    t = FakeTransport([net_fail(), ok([1])])
    run(tmp_path, t, [1], clock=clock)
    assert t.calls == [M1, M1] and clock.sleeps == [3.0]


def test_network_error_from_real_transport_is_retried(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setenv(ENV_NAME, TESTKEY)
    good = json.dumps(ok([1]))
    seen = []

    def handler(request):
        seen.append(1)
        if len(seen) == 1:
            raise httpx.ConnectError("обрыв соединения", request=request)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": good}]}}]})

    real = gp.RealTransport(client=httpx.Client(transport=httpx.MockTransport(handler)))
    clock = FakeClock()
    run(tmp_path, real, [1], clock=clock)
    assert len(seen) == 2 and clock.sleeps == [3.0]
    assert TESTKEY not in (tmp_path / "prompts.json").read_text(encoding="utf-8")


def test_limiter_runs_before_every_transient_attempt(tmp_path, monkeypatch):
    calls = []
    real_acquire = go.RateLimiter.acquire

    def spy(self, tokens):
        calls.append(tokens)
        return real_acquire(self, tokens)
    monkeypatch.setattr(go.RateLimiter, "acquire", spy)
    t = FakeTransport([unavail(), unavail(), unavail(), ok([1])])
    run(tmp_path, t, [1])
    assert len(calls) == 4 and len(set(calls)) == 1


def test_transient_errors_leave_no_key_or_prompt_text_in_logs(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(ENV_NAME, TESTKEY)
    caplog.set_level(logging.DEBUG)
    t = FakeTransport([unavail() for _ in range(5)] + [ok([1]), unavail(504), ok([2])])
    run(tmp_path, t, [1, 2], cfg=cfg(tmp_path, batch_size=1))
    assert TESTKEY not in caplog.text and "A wide shot" not in caplog.text
    assert "HTTP 503, код UNAVAILABLE" in caplog.text


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


def test_success_logs_model_time_and_attempts_without_prompt_text(tmp_path, caplog):
    # SPEC 7 / 0.1 п.24: в лог идут модель, время и число сбоев; текст промпта не идёт
    caplog.set_level(logging.INFO)
    run(tmp_path, FakeTransport([unavail(), ok([1, 2])]), [1, 2])
    lines = [r.getMessage() for r in caplog.records if r.name == "Generate.gemini_orchestrator"]
    # первый вызов падает (503), второй успешен: в логе успеха один предшествующий сбой
    assert any(l.startswith(f"Модель {M1}: ответ получен за") and "сбоев подряд до этого: 1" in l
               for l in lines)
    assert "A wide shot" not in caplog.text
