"""Тесты пула ключей в Download/generate_queries.py: без сети, без Gemini.

genai.Client и время подменяются фейками. Значения ключей тестовые и в ассертах не печатаются.
"""
import importlib.util
import json
import pathlib
import sys
import threading

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "Download" / "generate_queries.py"


def _load_module():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("generate_queries_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gq = _load_module()
from Generate.core.gemini_pool import KeyPool  # noqa: E402


class FakeClock:
    """Управляемое время: не спит, только растёт по требованию."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


def _pool(keys, clock, rpd=500):
    return KeyPool(
        keys,
        rpm_limit=10**6,
        tpm_limit=10**12,
        rpd_limit=rpd,
        clock=clock,
        poll_interval=0.001,
    )


def _clients(keys):
    # Фейковые клиенты: считаются, сами по себе ничего не делают.
    return {name: object() for name, _ in keys}


KEYS = [(f"GEMINI{i}", f"fake-value-{i}") for i in range(1, 4)]


# ---------------------------------------------------------------------------
# 1. Батчи распределяются по ключам
# ---------------------------------------------------------------------------

def test_batches_distributed_across_keys():
    clock = FakeClock()
    pool = _pool(KEYS, clock)
    used_keys = set()
    lock = threading.Lock()
    barrier = threading.Barrier(len(KEYS), timeout=5)

    def run_one(client, model, task):
        # Первые len(KEYS) заданий ждут друг друга, поэтому каждый получает отдельный ключ.
        if task < len(KEYS):
            barrier.wait()
        with lock:
            used_keys.add(id(client))
        return {str(task): {"ok": True}}

    results: dict = {}
    code, _ = gq._run_pool_tasks(
        pool, _clients(KEYS), list(range(9)), run_one, threading.Lock(),
        on_done=lambda t, r: results.update(r), model_for_pool="m", max_workers=10,
    )
    assert code is None
    assert len(results) == 9
    assert len(used_keys) == len(KEYS), "каждый ключ должен получить работу"


# ---------------------------------------------------------------------------
# 2. Чекпоинт не портится при записи из нескольких потоков
# ---------------------------------------------------------------------------

def test_checkpoint_valid_json_under_threads(tmp_path):
    clock = FakeClock()
    pool = _pool(KEYS, clock)
    cp = tmp_path / "cp.json"
    results: dict = {}
    exhausted: dict = {}
    lock = threading.Lock()

    def run_one(client, model, task):
        return {str(task): {"scene": f"scene {task}", "visual_value": 50}}

    def on_done(task, batch_result):
        results.update(batch_result)
        gq.save_checkpoint(str(cp), "hash", results, exhausted)

    code, _ = gq._run_pool_tasks(
        pool, _clients(KEYS), list(range(60)), run_one, lock, on_done,
        model_for_pool="m", max_workers=10, exhausted_pairs=exhausted,
    )
    assert code is None
    data = json.loads(cp.read_text(encoding="utf-8"))  # не должен бросить исключение
    assert data["source_hash"] == "hash"
    assert len(data["results"]) == 60


# ---------------------------------------------------------------------------
# 3. Дневная квота одного ключа не гасит остальные
# ---------------------------------------------------------------------------

def test_daily_quota_marks_only_one_key():
    clock = FakeClock()
    pool = _pool(KEYS, clock)
    bad_name = KEYS[0][0]
    exhausted: dict = {}
    seen_by_bad = []

    def run_one(client, model, task):
        # Клиент, помеченный как бракованный, отвечает дневной квотой.
        if client is _clients(KEYS)[bad_name]:
            seen_by_bad.append(task)
            raise gq.DailyQuotaExceededError(model, "quotaId perday")
        return {str(task): {"ok": True}}

    clients = _clients(KEYS)
    results: dict = {}
    code, _ = gq._run_pool_tasks(
        pool, clients, list(range(6)), run_one, threading.Lock(),
        on_done=lambda t, r: results.update(r), model_for_pool="m", max_workers=10,
        exhausted_pairs=exhausted,
    )
    assert code is None, "пока два ключа живы, код 3 не должен выставляться"
    assert len(results) == 6
    assert f"{bad_name}|m" in exhausted
    assert all(not k.startswith(f"{KEYS[1][0]}|") for k in exhausted)
    assert all(not k.startswith(f"{KEYS[2][0]}|") for k in exhausted)


def test_daily_quota_for_every_key_returns_3():
    clock = FakeClock()
    keys = KEYS[:2]
    pool = _pool(keys, clock)

    def run_one(client, model, task):
        raise gq.DailyQuotaExceededError(model, "quotaId perday")

    code, _ = gq._run_pool_tasks(
        pool, _clients(keys), list(range(4)), run_one, threading.Lock(),
        on_done=lambda t, r: None, model_for_pool="m", max_workers=10,
        exhausted_pairs={},
    )
    assert code == 3


# ---------------------------------------------------------------------------
# 4. Коды 1, 3, 4 сохраняются
# ---------------------------------------------------------------------------

def test_hard_error_returns_1_and_stops_new_tasks():
    clock = FakeClock()
    pool = _pool(KEYS, clock)
    started = []
    lock = threading.Lock()

    def run_one(client, model, task):
        with lock:
            started.append(task)
        if task == 0:
            raise ValueError("структурная ошибка")
        return {str(task): {}}

    code, _ = gq._run_pool_tasks(
        pool, _clients(KEYS), list(range(50)), run_one, threading.Lock(),
        on_done=lambda t, r: None, model_for_pool="m", max_workers=3,
    )
    assert code == 1
    assert len(started) < 50, "после жёсткой ошибки новые задания не стартуют"


def test_exit_code_4_from_model_unavailable_is_unchanged():
    # Код 4 формируется в pick_working_model/main по ModelUnavailableError - проверяем, что
    # класс сохранён и отличается от квоты (код 3).
    err = gq.ModelUnavailableError({"m": "ServerError: 503"}, [])
    assert isinstance(err, Exception)
    assert not isinstance(err, gq.AllModelsQuotaExhaustedError)


def test_exit_code_3_from_all_models_quota_is_unchanged():
    assert issubclass(gq.AllModelsQuotaExhaustedError, RuntimeError)
    assert not issubclass(gq.AllModelsQuotaExhaustedError, gq.ModelUnavailableError)


# ---------------------------------------------------------------------------
# 5. REPAIR: чанки идут через пул, порядок сегментов сохраняется
# ---------------------------------------------------------------------------

def _seg(i):
    return gq.Segment(index=i, start="00:00:00,000", end="00:00:01,000", text=f"line {i}")


def test_repair_chunks_go_through_pool_and_keep_order(tmp_path):
    clock = FakeClock()
    pool = _pool(KEYS, clock)
    segments = [_seg(i) for i in range(1, 301)]
    # Все записи изначально с нарушением (пустые sites), REPAIR их исправляет.
    results = {str(i): {"scene": "x", "sites": [], "query_narrow": "", "query_medium": "",
                        "query_broad": None, "type": "image", "is_entity": False,
                        "entity_keywords": [], "visual_value": 10} for i in range(1, 301)}
    seen_chunks = []
    lock = threading.Lock()

    def fake_call(client, model, chunk_segs, ctx_before, ctx_after, **kw):
        with lock:
            seen_chunks.append([s.index for s in chunk_segs])
        out = {}
        for s in chunk_segs:
            out[str(s.index)] = {"scene": "fixed", "sites": ["pexels"], "query_narrow": "a",
                                 "query_medium": "a", "query_broad": None, "type": "image",
                                 "is_entity": False, "entity_keywords": [], "visual_value": 99}
        return out

    cp = tmp_path / "cp.json"
    code, _ = gq._run_repair_round(
        pool, _clients(KEYS), "m", segments, results, {}, str(cp), "hash",
        gq.SOURCES_MIX, list(range(1, 301)), 1, call_batch_fn=fake_call,
    )
    assert code is None
    assert len(seen_chunks) >= 2, "чанков должно быть несколько"
    flat = [i for chunk in seen_chunks for i in chunk]
    assert sorted(flat) == list(range(1, 301))
    # visual_value при REPAIR не меняется
    assert all(results[str(i)]["visual_value"] == 10 for i in range(1, 301))
    # и чекпоинт валидный JSON
    json.loads(cp.read_text(encoding="utf-8"))
