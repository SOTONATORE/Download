"""Параллельная выдача батчей и исправлений: потоки OpenRouter, нарушения без повтора, деление по лимиту вывода,
раунды исправлений по барьеру. Транспорт фейковый, сеть не используется, ключи подменяются."""
import json
import re
import socket
import threading
from unittest import mock

import pytest

from Generate.cli import _ReworkCounter
from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import GeminiRuntimeError, PromptConfig, generate_prompts
from Generate.core.srt_parser import Segment

CLEAN = "A wide shot of a quiet street at dawn, a baker lifts a tray."
MAP = "A detailed map of the Balkans on a wooden table."   # запрещённый тип кадра: нарушение шага 6

MAIN_RE = re.compile(r"(?m)^segment_index: (\d+)$")   # строки батча основного запроса
FIX_RE = re.compile(r"Prompt for segment_index (\d+):")  # строки пакета исправления


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


def _cfg(tmp_path, batch_size):
    style = tmp_path / "style.md"
    style.write_text("# Style guide\nWrite one paragraph.\n", encoding="utf-8")
    return PromptConfig(media="video", style_file=str(style), max_prompt_chars=300, batch_size=batch_size)


def _segs(count):
    return [Segment(i, 0, 1000, f"Scene {i} shows a quiet street.") for i in range(1, count + 1)]


def _answer(nums, text=CLEAN):
    return json.dumps([{"segment_index": n, "prompt": text} for n in nums])


def _main_nums(user):
    return [int(x) for x in MAIN_RE.findall(user)]


class _Inner:
    def generate_json(self, model, system, user, schema):
        return "[]"


class _OpenRouterInner(_Inner):
    provider = "openrouter"


# а. Обёртка отдаёт провайдер внутреннего транспорта

def test_wrapper_exposes_openrouter_provider():
    assert _ReworkCounter(_OpenRouterInner()).provider == "openrouter"


def test_wrapper_defaults_to_gemini_provider():
    assert _ReworkCounter(_Inner()).provider == "gemini"


def test_wrapper_forwards_other_attributes():
    inner = _Inner()
    inner.active_model = "model-x"
    assert _ReworkCounter(inner).active_model == "model-x"


# б. Число потоков волны OpenRouter

@pytest.mark.parametrize("keys, sends, batches, expected", [
    (10, 10, 21, 21),
    (10, 10, 5, 5),
    (1, 10, 21, 10),
])
def test_wave_thread_count(keys, sends, batches, expected):
    assert gp.openrouter_wave_threads(keys, sends, batches) == expected


# в. Нарушение правил: один вызов модели, затем очередь исправлений

class _Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_json(self, model, system, user, schema):
        self.calls.append(user)
        return self.replies.pop(0)


def test_violation_is_called_once_and_goes_to_fix_queue(tmp_path):
    cfg = _cfg(tmp_path, 10)
    builder = gp._Builder(_segs(2), [1], cfg)
    batches = builder.batches([1])
    t = _Scripted([_answer([1], MAP)])
    job = gp._WaveJob("видео", builder, batches, t, "m", "", None)
    job.process(gp._LoggedTransport(t), 0)
    assert len(t.calls) == 1
    assert [c.num for c in job.rework_queue] == [1]


# Фейковый транспорт для пунктов г–д: пустой ответ по лимиту вывода по правилу fail(nums)

class _LengthTransport:
    def __init__(self, fail):
        self.fail = fail
        self.calls = []

    def generate_json(self, model, system, user, schema):
        nums = _main_nums(user)
        self.calls.append(nums)
        if self.fail(nums):
            raise GeminiRuntimeError("OpenRouter исчерпал лимит вывода без текста.",
                                     status=422, code="EMPTY_RESPONSE_LENGTH")
        return _answer(nums)


def _one_key(monkeypatch):
    monkeypatch.setattr(gp, "_provider_key_count", lambda provider: 1)


# г. Деление по лимиту вывода: x.1 и x.2 по 12 и 13 клипов, по одному вызову

def test_length_error_splits_batch_into_halves(tmp_path, monkeypatch):
    _one_key(monkeypatch)
    logger = mock.MagicMock()
    monkeypatch.setattr(gp, "log", logger)
    t = _LengthTransport(fail=lambda nums: nums == list(range(1, 26)))
    res = generate_prompts(_segs(30), list(range(1, 31)), _cfg(tmp_path, 25), t, "m")

    assert [r.num for r in res] == list(range(1, 31))
    assert not any(r.needs_review for r in res)
    assert len(t.calls) == 4                                  # батч 1 (упал), батч 2, половины по одному разу
    assert t.calls.count(list(range(1, 13))) == 1
    assert t.calls.count(list(range(13, 26))) == 1
    assert t.calls.count(list(range(26, 31))) == 1
    split = [c.args[0] % c.args[1:] for c in logger.warning.call_args_list if "разделён" in c.args[0]]
    assert len(split) == 1
    assert "1.1" in split[0] and "1.2" in split[0] and "12 клипов" in split[0] and "13 клипов" in split[0]


# д. Половина, упавшая снова, не делится: клипы на проверку, прогон идёт дальше

def test_half_failing_again_is_flagged_and_run_continues(tmp_path, monkeypatch):
    _one_key(monkeypatch)
    monkeypatch.setattr(gp, "log", mock.MagicMock())
    t = _LengthTransport(fail=lambda nums: 1 in nums)         # номер 1 есть в батче и в первой половине
    res = generate_prompts(_segs(30), list(range(1, 31)), _cfg(tmp_path, 25), t, "m")

    assert t.calls.count(list(range(1, 13))) == 1             # первая половина не делится снова
    assert len(t.calls) == 4
    by = {r.num: r for r in res}
    assert [r.num for r in res] == list(range(1, 31))
    assert all(by[n].needs_review and by[n].review_reason == "пустой ответ по лимиту вывода" for n in range(1, 13))
    assert not any(by[n].needs_review for n in range(13, 31))  # вторая половина и батч 2 — в порядке


# е. Раунды исправлений в пуле: второй раунд стартует, пока пакет первого раунда ещё выполняется

class _FixOverlapTransport:
    """Пакет 1–25 первого раунда ждёт старта пакета второго раунда (клипы 26–50); если его нет, ждёт 5 с."""

    def __init__(self):
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.waited_ok = False
        self.calls = []          # номера клипов каждого пакета исправления, по порядку вызовов
        self.lo26 = 0

    def generate_json(self, model, system, user, schema):
        fixed = [int(x) for x in FIX_RE.findall(user)]
        if not fixed:
            return _answer(_main_nums(user), MAP)              # основной запрос: все клипы с нарушением
        with self.lock:
            self.calls.append(sorted(fixed))
            if min(fixed) == 26:
                self.lo26 += 1
                second = self.lo26 == 2
            else:
                second = False
        if second:
            self.release.set()                                  # пакет второго раунда стартовал
            return _answer(fixed, CLEAN)
        if min(fixed) == 26:
            return _answer(fixed, MAP)                          # первый раунд: 26–50 остаются с нарушением
        self.waited_ok = self.release.wait(timeout=5)           # 1–25 ждут старта второго раунда
        return _answer(fixed, CLEAN)


def test_fix_round_barrier(tmp_path, monkeypatch):
    # Комментарий: раньше второй раунд ждал конца первого (round_inflight == 0). Теперь клипы, вернувшиеся
    # с нарушением, уходят в следующий пакет сразу. Проверяем: пакет второго раунда стартовал при выполняющемся
    # пакете первого; ни один клип не потерян; ни один клип не исправлен дважды в одном пакете; rework_round
    # увеличен ровно на одно исправление каждого клипа.
    monkeypatch.setattr(gp, "log", mock.MagicMock())
    t = _FixOverlapTransport()
    pool = gp.SharedBatchPool({"photo": 2})
    res = gp.generate_prompts(_segs(50), list(range(1, 51)), _cfg(tmp_path, 25), t, "m",
                              pool=pool, wave="photo")
    by = {r.num: r for r in res}
    assert t.waited_ok is True                                  # второй раунд стартовал, пока 1–25 выполнялся
    assert sorted(by) == list(range(1, 51))                     # ни один клип не потерян
    assert all(len(set(c)) == len(c) for c in t.calls)          # ни один клип не исправлен дважды в пакете
    assert by[1].rework_round == 1 and by[26].rework_round == 2  # одно исправление на клип за круг


# ж. Прочая ошибка транспорта по-прежнему останавливает прогон

def test_non_length_runtime_error_still_stops(tmp_path, monkeypatch):
    _one_key(monkeypatch)
    monkeypatch.setattr(gp, "log", mock.MagicMock())

    class _Failing:
        calls = 0

        def generate_json(self, model, system, user, schema):
            self.calls += 1
            raise GeminiRuntimeError("сбой", status=500, code="EMPTY_RESPONSE")

    t = _Failing()
    with pytest.raises(GeminiRuntimeError):
        generate_prompts(_segs(30), list(range(1, 31)), _cfg(tmp_path, 25), t, "m")
    assert t.calls == 1
