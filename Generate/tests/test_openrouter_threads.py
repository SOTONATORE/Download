"""Потоки OpenRouter: openrouter_sends_per_window из конфига доходит до числа потоков одиночной волны.

Сеть, ключи и реальные API не используются: ключи подменяются, транспорт фейковый, лог читается через мок.
"""
import json
import re
import socket
from unittest import mock

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import PromptConfig, generate_prompts, openrouter_thread_count
from Generate.core.srt_parser import Segment

SEG_TEXT = "A wide shot of a quiet street at dawn, a baker lifts a tray."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


class _FakeTransport:
    """Ответ зависит только от номеров батча: как в test_gemini_parallel."""

    def generate_json(self, model, system, user, schema):
        nums = [int(x) for x in re.findall(r"(?m)^segment_index: (\d+)$", user)]
        return json.dumps([{"segment_index": n, "prompt": SEG_TEXT} for n in nums])


def _make_cfg(tmp_path, **kw):
    style = tmp_path / "style.md"
    style.write_text("# Style guide\nWrite one paragraph.\n", encoding="utf-8")
    kw.setdefault("style_file", str(style))
    kw.setdefault("max_prompt_chars", 300)
    kw.setdefault("batch_size", 1)
    kw.setdefault("media", "video")
    return PromptConfig(**kw)


def test_thread_count_multiplies_keys_by_sends_per_window():
    assert openrouter_thread_count(10, 10) == 100
    assert openrouter_thread_count(9, 10) == 90


def test_thread_count_without_value_is_one_thread_per_key():
    assert openrouter_thread_count(10) == 10


def test_generate_prompts_openrouter_uses_sends_per_window_per_key(tmp_path, monkeypatch):
    # 2 ключа × 10 отправок = 20 потоков; 25 номеров по одному в батч дают 25 батчей, так что 20 достижимо
    monkeypatch.setattr(gp, "_provider_key_count", lambda provider: 2)
    logger = mock.MagicMock()
    monkeypatch.setattr(gp, "log", logger)
    segments = [Segment(n, 0, 1000, f"Scene {n} shows a quiet street.") for n in range(1, 26)]
    wanted = list(range(1, 26))

    res = generate_prompts(segments, wanted, _make_cfg(tmp_path), _FakeTransport(), "m",
                           provider="openrouter", sends_per_window=10)

    assert [r.num for r in res] == wanted
    # число батчей берём из строки "=== Генерация ... батчей N ..." (аргументы: вид, клипов, батчей, ключей)
    start_calls = [c for c in logger.info.call_args_list if "Генерация" in str(c.args[0])]
    batches = start_calls[0].args[3]
    assert batches >= 20
    worker_calls = [c for c in logger.info.call_args_list if c.args and "Батчи выполняются в" in c.args[0]]
    assert len(worker_calls) == 1
    assert worker_calls[0].args[1] == min(20, batches) == 20
