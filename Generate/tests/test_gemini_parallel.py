"""Параллельные батчи generate_prompts и замок ManagedTransport. Сеть, Gemini и реальные ключи не используются."""
import json
import re
import socket
import threading
import time

import pytest

from Generate.core.gemini_orchestrator import ManagedTransport
from Generate.core.gemini_prompts import GeminiRuntimeError, PromptConfig, generate_prompts
from Generate.core.srt_parser import Segment

SEG_TEXT = "A wide shot of a quiet street at dawn, a baker lifts a tray."
WANTED = [1, 2, 3, 4, 6, 7]   # номер 5 без текста: в эти тесты не берём


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


def seg(num, text, start=0):
    return Segment(num, start, start + 1000, text)


def make_segments():
    return [
        seg(1, "The old city woke slowly."),
        seg(2, "Merchants opened their stalls"),
        seg(3, "and carts rolled in from the hills."),
        seg(4, "Dr. Hale watched from a window."),
        seg(5, ""),
        seg(6, "Then the bells began to ring!"),
        seg(7, "Nobody ran."),
    ]


def make_cfg(tmp_path, **kw):
    style = tmp_path / "style.md"
    style.write_text("# Style guide\nWrite one paragraph.\n", encoding="utf-8")
    kw.setdefault("style_file", str(style))
    kw.setdefault("max_prompt_chars", 300)
    kw.setdefault("batch_size", 10)
    kw.setdefault("media", "video")
    return PromptConfig(**kw)


class FakeParallelTransport:
    """Ответ зависит только от номеров батча, а не от порядка вызовов: поэтому потоки не меняют результат.
    fail_on: номер, на котором запрос падает GeminiRuntimeError. delay: номер -> пауза, не больше 0.01 с."""

    def __init__(self, fail_on=None, delay=None):
        self.fail_on = fail_on
        self.delay = delay or {}
        self.started = []
        self.finished = []
        self._lock = threading.Lock()

    def generate_json(self, model, system, user, schema):
        nums = [int(x) for x in re.findall(r"(?m)^segment_index: (\d+)$", user)]
        with self._lock:
            self.started.append(tuple(nums))
        try:
            time.sleep(self.delay.get(nums[0], 0.0))
            if self.fail_on in nums:
                raise GeminiRuntimeError("сбой", status=500)
            return json.dumps([{"segment_index": n, "prompt": SEG_TEXT} for n in nums])
        finally:
            with self._lock:
                self.finished.append(tuple(nums))


def test_results_sorted_by_segment_when_batches_finish_out_of_order(tmp_path):
    cfg = make_cfg(tmp_path, batch_size=1)
    # меньшие номера медленнее: батчи завершаются не по порядку
    t = FakeParallelTransport(delay={n: 0.01 - 0.001 * n for n in WANTED})
    res = generate_prompts(make_segments(), WANTED, cfg, t, "m", max_workers=3)
    assert [r.num for r in res] == WANTED
    assert all(r.prompt and not r.needs_review for r in res)


def test_hard_error_in_one_worker_stops_new_batches_and_raises(tmp_path):
    cfg = make_cfg(tmp_path, batch_size=1)
    t = FakeParallelTransport(fail_on=1, delay={n: 0.005 for n in WANTED})
    with pytest.raises(GeminiRuntimeError):
        generate_prompts(make_segments(), WANTED, cfg, t, "m", max_workers=2)
    # новые батчи не брались: запущено меньше, чем всего
    assert len(t.started) < len(WANTED)
    # запущенные батчи дошли до конца
    assert len(t.started) == len(t.finished)


def test_parallel_result_equals_sequential_with_three_keys(tmp_path):
    cfg = make_cfg(tmp_path, batch_size=2)
    seq = generate_prompts(make_segments(), WANTED, cfg, FakeParallelTransport(), "m", max_workers=1)
    par = generate_prompts(make_segments(), WANTED, cfg,
                           FakeParallelTransport(delay={n: 0.003 for n in WANTED}), "m", max_workers=3)
    assert [(r.num, r.prompt, r.needs_review) for r in par] == \
           [(r.num, r.prompt, r.needs_review) for r in seq]


class _NeverCalledInner:
    def generate_json(self, *a, **k):
        raise AssertionError("в этом тесте сетевой вызов не нужен")


def test_managed_active_model_not_corrupted_by_two_threads():
    m = ManagedTransport(_NeverCalledInner(), models=("model-a", "model-b", "model-c"))
    m._exhausted.add("model-a")   # первая модель исчерпана: выбор должен всегда давать model-b
    errors = []
    barrier = threading.Barrier(2)

    def run():
        try:
            barrier.wait()
            for _ in range(500):
                assert m._pick_model() == "model-b"
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    assert m.active_model == "model-b"
