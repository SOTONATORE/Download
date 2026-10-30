"""Тесты H3, часть B: недостающие промпты, батчи по 50.

Сеть не используется: транспорт подменён, время подменено.
"""
import json
import pathlib
import socket

import pytest

from Generate.core import gemini_orchestrator as go
from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import PromptConfig
from Generate.core.srt_parser import Segment, srt_hash

TIMING = go.TimingParams(24, {"kind": "8k+1"}, 2.5, 5.0, 10.0, 0.5)
GOOD_PROMPT = "A wide shot of a quiet street at dawn."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


class RecordingTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.users = []
        self.systems = []

    def generate_json(self, model, system, user, schema):
        self.users.append(user)
        self.systems.append(system)
        return self.replies.pop(0)


def cfg(tmp_path, **kw):
    style = tmp_path / "style.md"
    style.write_text("# Style\nWrite one paragraph.\n", encoding="utf-8")
    kw.setdefault("style_file", str(style))
    kw.setdefault("max_prompt_chars", 300)
    kw.setdefault("batch_size", 10)
    return PromptConfig(**kw)


def make_segs(n):
    return [Segment(i, i * 3000, i * 3000 + 1000, f"Sentence number {i}.") for i in range(1, n + 1)]


def _ready(n, text):
    return {"num": n, "start_ms": 0, "end_ms": 1, "clip_ms": 1, "num_frames": 1, "prompt": text,
            "status": "ready", "attempts": 1, "error": "", "needs_review": False, "extra": {}}


# ------------------------------------------------------------------ недостающие номера (B2)

def test_missing_segments_computed():
    prompts = {str(n): _ready(n, f"p{n}") for n in (1, 2, 4, 5)}
    assert go.missing_segments(prompts, [1, 2, 3, 4, 5]) == [3]


def test_failed_entry_counts_as_missing():
    prompts = {"1": {"status": "failed", "prompt": ""}}
    assert go.missing_segments(prompts, [1]) == [1]


def test_generate_only_missing_and_keep_others(tmp_path):
    segs = make_segs(5)
    path = tmp_path / "prompts.json"
    existing = {"srt_hash": srt_hash(segs), "model": "m", "profile": "ltx25",
                "prompts": {str(n): _ready(n, f"old {n}") for n in (1, 2, 4, 5)}}
    path.write_text(json.dumps(existing), encoding="utf-8")
    t = RecordingTransport([json.dumps([{"segment_index": 3, "prompt": GOOD_PROMPT}])])
    go.generate_and_store(segs, [1, 2, 3, 4, 5], cfg(tmp_path), t, path, profile="ltx25", timing=TIMING,
                          clock=lambda: 1000.0, sleep=lambda s: None)
    assert len(t.users) == 1
    assert "segment_index: 3\n" in t.users[0]
    assert "segment_index: 1\n" not in t.users[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data["prompts"]) == ["1", "2", "3", "4", "5"]
    for n in (1, 2, 4, 5):
        assert data["prompts"][str(n)]["prompt"] == f"old {n}"
    assert data["prompts"]["3"]["prompt"] == GOOD_PROMPT


# ------------------------------------------------------------------ батчи по 50 (B1)

def test_batches_of_fifty(tmp_path):
    segs = make_segs(120)
    reqs = gp.build_requests(segs, list(range(1, 121)), cfg(tmp_path, batch_size=50))
    assert [len(r.nums) for r in reqs] == [50, 50, 20]


def test_config_default_batch_size_is_fifty():
    text = (pathlib.Path(__file__).resolve().parents[1] / "config.default.yaml").read_text(encoding="utf-8")
    assert "gemini_batch_size: 50\n" in text
