"""Новые проверки: этнический признак по названию места и по окну соседних сегментов, очередь исправлений
без устаревших причин, порядок выдачи пакетов в пуле и завершение пула на пустой очереди.
Фейковый транспорт, сеть не используется, ключи не нужны."""
import json
import socket
import threading
from unittest import mock

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core import prompt_check as pc
from Generate.core.gemini_prompts import PromptConfig
from Generate.core.srt_parser import Segment

CLEAN = "A wide shot of a quiet street at dawn, a baker lifts a tray."
MAP = "A detailed map of the Balkans on a wooden table."
CHILD = "A young boy rides a horse across the steppe."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


def _cfg(tmp_path, batch_size=10):
    style = tmp_path / "style.md"
    style.write_text("# Style guide\nWrite one paragraph.\n", encoding="utf-8")
    return PromptConfig(media="video", style_file=str(style), max_prompt_chars=300, batch_size=batch_size)


def _segs(count):
    return [Segment(i, 0, 1000, f"Scene {i} shows a quiet street.") for i in range(1, count + 1)]


def _answer(nums, text=CLEAN):
    return json.dumps([{"segment_index": n, "prompt": text} for n in nums])


class _Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_json(self, model, system, user, schema):
        self.calls.append(user)
        return self.replies.pop(0)


ETH = pc.CONTENT_REASON_RU[pc.PROBLEM_ETHNICITY]
SEG_RU = "Тажигали шёл по степи и думал о сыне."


# --- 1.1 названия мест и животных не дают этнического признака ---------------------------------

def test_african_place_name_gives_no_reason():
    # Комментарий: «East African grassland» — название места; признак внешности не должен срабатывать.
    assert gp.ethnicity_reasons("An East African grassland with acacia trees.", SEG_RU) == []


def test_african_person_still_gives_reason():
    # Комментарий: «African man» описывает человека; без корня Африки в тексте сегмента причина нужна.
    assert gp.ethnicity_reasons("An African man walks across the savanna.", SEG_RU) == [ETH]


# --- 1.2 окно соседних сегментов -----------------------------------------------------------------

def test_ethnic_root_in_next_segment_counts(tmp_path):
    # Комментарий: корень стоит только в сегменте n+1; текст соседа учитывается, причина не ставится.
    raw = json.dumps([{"segment_index": 7, "prompt": "An African man walks across the steppe."}])
    texts = {6: "Сцена шесть.", 7: SEG_RU, 8: "Африканец ждёт у реки."}
    res, _ = gp.parse_and_validate(raw, [7], _cfg(tmp_path), texts)
    assert ETH not in res[7].review_reason


def test_no_ethnic_root_in_window_flags(tmp_path):
    # Комментарий: ни в одном из трёх сегментов корня нет — причина ставится.
    raw = json.dumps([{"segment_index": 7, "prompt": "An African man walks across the steppe."}])
    texts = {6: "Сцена шесть.", 7: SEG_RU, 8: "Ветер дует."}
    res, _ = gp.parse_and_validate(raw, [7], _cfg(tmp_path), texts)
    assert ETH in res[7].review_reason


# --- 1.3 причина в записи принадлежит тексту записи ----------------------------------------------

def test_requeued_clip_carries_reason_of_its_own_text(tmp_path):
    # Комментарий: клип вернулся в очередь с новым текстом (мальчик). Причина в поле записи должна быть
    # причиной именно этого текста, а не причиной карты из первого прохода.
    cfg = _cfg(tmp_path, 10)
    builder = gp._Builder(_segs(2), [1], cfg)
    batches = builder.batches([1])
    t = _Scripted([_answer([1], MAP), _answer([1], CHILD)])
    job = gp._WaveJob("видео", builder, batches, t, "m", "", None)
    job.process(gp._LoggedTransport(t), 0)
    clip = job.rework_queue[0]
    assert clip.review_reason != ""                              # первый проход: причина карты
    job.rework_queue.clear()
    job.process_fix(gp._LoggedTransport(t), [clip], 2)
    assert job.rework_queue == [clip]                            # клип вернулся в очередь
    expected = "; ".join(gp._problem_ru_any(p) for p in gp._find_all_problems(clip.prompt_raw, cfg))
    assert clip.review_reason == expected
    assert pc.CONTENT_REASON_RU[pc.PROBLEM_CHILD] in clip.review_reason
    assert clip.needs_review is True


# --- 2.1 полный пакет раньше главного батча; остаток — когда главных нет в работе -----------------

def _rework_clips(count):
    return [gp.PromptResult(num=100 + i, prompt="x", needs_review=True, review_reason="r",
                            rework_pending=[], rework_round=0, prompt_raw="x") for i in range(count)]


def test_full_fix_package_goes_before_main(tmp_path):
    # Комментарий: очередь из 25 клипов и один непринятый главный батч — поток берёт полный пакет первым.
    cfg = _cfg(tmp_path)
    builder = gp._Builder(_segs(2), [1], cfg)
    batches = builder.batches([1])
    job = gp._WaveJob("фото", builder, batches, _Scripted([]), "m", "", None, rework=_rework_clips(25))
    pool = gp.SharedBatchPool({"фото": 1})
    pool._jobs["фото"] = job
    job.pool = pool
    kind, _, payload = pool._take("фото")
    assert kind == "fix"
    assert len(payload[1]) == 25


def test_remainder_waits_for_main_then_runs(tmp_path):
    # Комментарий: остаток (5 клипов) не берётся, пока главный батч выполняется; после его конца — берётся.
    cfg = _cfg(tmp_path)
    builder = gp._Builder(_segs(2), [1], cfg)
    batches = builder.batches([1])
    job = gp._WaveJob("фото", builder, batches, _Scripted([]), "m", "", None, rework=_rework_clips(5))
    pool = gp.SharedBatchPool({"фото": 1})
    pool._jobs["фото"] = job
    job.pool = pool
    kind1, _, _ = pool._take("фото")
    assert kind1 == "main"
    assert pool._pick_fix("фото", remainder=True) is None        # главный батч в работе: остаток ждёт
    pool._finish(job, None, kind="main")
    kind2, _, payload = pool._take("фото")
    assert kind2 == "fix" and len(payload[1]) == 5


# --- 2.4 пустая очередь: поток и ожидание завершаются ----------------------------------------------

def test_empty_queue_worker_and_wait_terminate(tmp_path):
    # Комментарий: без батчей, без исправлений и без работы в полёте поток выходит, wait_job не виснет.
    cfg = _cfg(tmp_path)
    builder = gp._Builder(_segs(2), [], cfg)
    job = gp._WaveJob("фото", builder, [], _Scripted([]), "m", "", None)
    pool = gp.SharedBatchPool({"фото": 1})
    pool._jobs["фото"] = job
    job.pool = pool
    done = []

    def run():
        pool.run_worker("фото")
        pool.wait_job(job)
        done.append(True)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout=5)
    assert done == [True]


# --- 3.x диапазоны номеров в журнале --------------------------------------------------------------

def test_nums_ranges_format():
    # Комментарий: номера клипов в строках журнала сворачиваются в диапазоны, пустой список — «нет».
    assert gp._nums_ranges([1, 2, 3, 5, 7, 8]) == "1–3, 5, 7–8"
    assert gp._nums_ranges([]) == "нет"
