"""Режим fail_fast: заглушка транспорта со счётчиком вызовов, сеть не используется.

Проверяют: поведение по умолчанию (один повтор), остановку без повтора, файл сырого ответа
только при проблеме, отсутствие текста ответа в сообщении и журнале, вход в verify_prompts.yml.
"""
import json
import logging
import pathlib
import socket

import pytest
import yaml

from Generate.core.gemini_prompts import GeminiError, PromptConfig, generate_prompts
from Generate.core.srt_parser import Segment

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "verify_prompts.yml"
# Метка в тексте ответа модели: не должна попадать ни в сообщение об ошибке, ни в журнал
MARK = "SECRET-RAW-MARKER"
GOOD = "A wide shot of a quiet street at dawn, a baker lifts a tray."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


def segments():
    return [
        Segment(1, 0, 1000, "The old city woke slowly."),
        Segment(2, 1000, 2000, "Merchants opened their stalls"),
        Segment(3, 2000, 3000, "and carts rolled in from the hills."),
    ]


def make_cfg(tmp_path, **kw):
    style = tmp_path / "style.md"
    style.write_text("# Style guide\nWrite one paragraph.\n", encoding="utf-8")
    kw.setdefault("style_file", str(style))
    kw.setdefault("max_prompt_chars", 300)
    kw.setdefault("batch_size", 10)
    kw.setdefault("media", "video")
    return PromptConfig(**kw)


class CountingTransport:
    """Отдаёт заранее заданные ответы по порядку и считает вызовы. Лишний вызов даёт IndexError."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate_json(self, model, system, user, schema):
        self.calls += 1
        return self.replies.pop(0)


def ok(nums, text=GOOD):
    return json.dumps([{"segment_index": n, "prompt": text} for n in nums])


# ---------------------------------------------------------------- режим по умолчанию

def test_default_off_bad_answer_is_retried_once(tmp_path):
    cfg = make_cfg(tmp_path)  # fail_fast по умолчанию выключен
    t = CountingTransport([ok([1]), ok([3])])  # в первом ответе нет номера 3; повтор только по нему
    res = generate_prompts(segments(), [1, 3], cfg, t, "m", max_workers=1)
    assert t.calls == 2
    assert [r.num for r in res] == [1, 3]
    assert not any(r.needs_review for r in res)


def test_defaults_keep_fail_fast_off_and_no_raw_dir():
    cfg = PromptConfig(style_file="x", media="video")
    assert cfg.fail_fast is False
    assert cfg.raw_dir is None


# ---------------------------------------------------------------- fail_fast: остановка

def test_fail_fast_bad_answer_one_call_and_error(tmp_path):
    cfg = make_cfg(tmp_path, fail_fast=True)
    t = CountingTransport([ok([1])])  # нет номера 3; повтора быть не должно (второго ответа нет)
    with pytest.raises(GeminiError):
        generate_prompts(segments(), [1, 3], cfg, t, "m", max_workers=1)
    assert t.calls == 1


def test_fail_fast_good_answer_no_file_no_error(tmp_path):
    raw_dir = tmp_path / "raw"
    cfg = make_cfg(tmp_path, fail_fast=True, raw_dir=str(raw_dir))
    t = CountingTransport([ok([1, 3])])
    res = generate_prompts(segments(), [1, 3], cfg, t, "m", max_workers=1)
    assert t.calls == 1
    assert [r.num for r in res] == [1, 3]
    assert not raw_dir.exists()


# ---------------------------------------------------------------- fail_fast: файл сырого ответа

def test_fail_fast_bad_answer_writes_one_raw_file(tmp_path):
    raw_dir = tmp_path / "raw"
    cfg = make_cfg(tmp_path, fail_fast=True, raw_dir=str(raw_dir))
    raw_answer = "это не JSON " + MARK
    t = CountingTransport([raw_answer])
    with pytest.raises(GeminiError):
        generate_prompts(segments(), [1, 3], cfg, t, "m", max_workers=1)
    files = sorted(raw_dir.iterdir())
    assert len(files) == 1
    # имя: диапазон номеров батча и код причины
    assert files[0].name == "batch_1-3_not_json.txt"
    assert files[0].read_text(encoding="utf-8") == raw_answer


def test_fail_fast_bad_answer_without_raw_dir_writes_nothing(tmp_path):
    cfg = make_cfg(tmp_path, fail_fast=True)  # raw_dir не задан
    before = sorted(p.name for p in tmp_path.iterdir())
    t = CountingTransport(["это не JSON " + MARK])
    with pytest.raises(GeminiError):
        generate_prompts(segments(), [1, 3], cfg, t, "m", max_workers=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# ---------------------------------------------------------------- fail_fast: текст ответа не утекает

def test_fail_fast_message_and_log_have_no_raw_text(tmp_path, caplog):
    cfg = make_cfg(tmp_path, fail_fast=True, raw_dir=str(tmp_path / "raw"))
    # сегмент 1 в ответе с меткой; номера 3 нет: нарушение по 3
    raw_answer = json.dumps([{"segment_index": 1, "prompt": GOOD + " " + MARK}])
    t = CountingTransport([raw_answer])
    caplog.set_level(logging.DEBUG)
    with pytest.raises(GeminiError) as info:
        generate_prompts(segments(), [1, 3], cfg, t, "m", max_workers=1)
    message = str(info.value)
    assert message.startswith("Ошибка")
    assert "fail_fast" in message
    assert MARK not in message
    assert MARK not in caplog.text


# ---------------------------------------------------------------- workflow

def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _dispatch_inputs(data):
    # PyYAML читает ключ on как True (YAML 1.1); берём оба варианта
    on = data["on"] if "on" in data else data[True]
    return on["workflow_dispatch"]["inputs"]


def test_workflow_input_fail_fast_is_boolean_default_false():
    inp = _dispatch_inputs(_workflow())["fail_fast"]
    assert inp["type"] == "boolean"
    assert inp["default"] is False


def test_workflow_step_env_passes_fail_fast():
    # env задан на шаге проверки (в этом workflow нет env на уровне job)
    steps = _workflow()["jobs"]["verify"]["steps"]
    env = {}
    for step in steps:
        if "FAIL_FAST" in (step.get("env") or {}):
            env = step["env"]
    assert env.get("FAIL_FAST") == "${{ inputs.fail_fast }}"
