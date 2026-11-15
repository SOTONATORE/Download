"""Согласованность исправлений и исключение для заглавного текста на кадре.

Проверяется:
- профиль с frame_text: заглавные отрицания (текст на кадре «NO TOOLS») не считаются нарушением стиля;
  строчные и смешанные — считаются; без frame_text правило прежнее;
- клип, чей ответ на исправление помечен при разборе (стиль), но без находок по сырому тексту, ставится
  в очередь и получает второй круг; на REPAIR_ROUNDS помечается на проверку и прогон останавливается
  при on_unfixed: fail;
- _run_body работает у объекта без атрибута segments_all (флаг считается False).

Без сети и без реального Gemini: транспорт и ответы модели подменяются. Тесты не запускались при подготовке
изменения (владелец запускает их на GitHub).
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Generate.core import gemini_prompts as gp  # noqa: E402
from Generate.core import orchestrator as orch  # noqa: E402
from Generate.core import prompt_check as pc  # noqa: E402

STYLE = _HERE.parents[1] / "model_profiles" / "models" / "zimage_turbo" / "prompts" / "style.md"
NEGATION = "стиль: отрицание"


# --- а/б: правило отрицания и текст на кадре ----------------------------------------------------

def test_caps_negation_on_frame_passes_when_frame_text_relaxed():
    text = "Left panel titled NO TOOLS NO FOOD"
    reasons = gp._style_reasons(text, "image", frozenset({"frame_text"}))
    assert NEGATION not in reasons


def test_lowercase_or_mixed_negation_still_flagged_when_frame_text_relaxed():
    # строчное no и смешанное No остаются нарушением, как и раньше: исключение только для заглавных слов
    text = "Left panel titled NO TOOLS, no food"
    reasons = gp._style_reasons(text, "image", frozenset({"frame_text"}))
    assert NEGATION in reasons


def test_caps_negation_flagged_without_frame_text():
    # профиль без frame_text: то же самое отрицание по-прежнему нарушает правило
    text = "Left panel titled NO TOOLS NO FOOD"
    reasons = gp._style_reasons(text, "image", frozenset())
    assert NEGATION in reasons


def test_caps_word_sign_flagged_without_frame_text():
    # б: без frame_text заглавное NO в описании знака нарушает правило стиля
    reasons = gp._style_reasons("a sign that says NO", "image", frozenset())
    assert NEGATION in reasons


# --- в/г: очередь исправлений и остановка ------------------------------------------------------

class _Logged:
    """Заглушка запроса: отдаёт заранее заданный ответ модели на исправление и считает вызовы."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def start_batch(self, fix_no, label) -> None:
        return None

    def generate_json(self, model, system, user, schema) -> str:
        self.calls += 1
        return self.reply


def _job(clip: gp.PromptResult, queued: list):
    cfg = gp.PromptConfig(media="image", style_file=str(STYLE))
    builder = SimpleNamespace(system="sys", schema={}, cfg=cfg, texts={clip.num: "Текст сегмента."})
    job = gp._WaveJob.__new__(gp._WaveJob)  # без полного конструктора: нужны поля, которые читает process_fix
    job.builder = builder
    job.model = "m"
    job.tag = ""
    job.style_text = ""
    job.out = {}
    job.out_lock = threading.Lock()
    job.on_batch = None
    job.pool = None
    job.rework_queue = []
    job.sched_lock = threading.Lock()
    job.enqueue = lambda clips: queued.append(list(clips))   # очередь исправлений: что вернули в очередь
    return job


def _clip(rework_round: int) -> gp.PromptResult:
    # клип с находкой по сырому тексту (больше одного человека), как в очереди исправлений
    prompt = "A crowd of villagers gathers in the square."
    return gp.PromptResult(
        num=8, prompt=prompt, needs_review=False, review_reason="",
        rework_pending=pc.find_problems(prompt, "image"), rework_round=rework_round, prompt_raw=prompt)


def test_parse_flagged_fix_is_requeued_and_gets_second_round():
    # Ответ на первый круг: без находок по тексту, но с отрицанием (стиль) при разборе.
    queued: list = []
    clip = _clip(rework_round=1)
    round1 = "A man in a grey coat stands by a wall with no hat."
    assert pc.find_problems(round1, "image") == []   # предусловие: флаг только от разбора, не от find_problems
    job = _job(clip, queued)
    logged1 = _Logged(json.dumps([{"segment_index": 8, "prompt": round1}]))
    gp._WaveJob.process_fix(job, logged1, [clip], 1)
    assert queued == [[clip]]                        # вернули в очередь, а не сохранили как исправленный
    assert clip.needs_review is False                # на проверку не помечен: остался круг
    assert clip.rework_round == 2
    assert gp._PARSE_PREFIX + NEGATION in clip.rework_pending

    # второй круг: исправленный ответ проходит все проверки, клип снимается с очереди
    queued.clear()
    round2 = "One adult man stands in a grey coat by a wall."
    logged2 = _Logged(json.dumps([{"segment_index": 8, "prompt": round2}]))
    gp._WaveJob.process_fix(job, logged2, [clip], 2)
    assert logged2.calls == 1                        # второй запрос к модели был, прогон не остановился
    assert queued == [[]]
    assert clip.needs_review is False
    assert clip.rework_pending == []


def test_parse_flagged_clip_after_repair_rounds_is_marked_and_run_stops():
    queued: list = []
    clip = _clip(rework_round=gp.REPAIR_ROUNDS - 1)  # следующий круг последний
    still_bad = "A man in a grey coat stands by a wall with no hat."
    job = _job(clip, queued)
    gp._WaveJob.process_fix(job, _Logged(json.dumps([{"segment_index": 8, "prompt": still_bad}])), [clip], 1)
    assert queued == [[]]                            # больше в очередь не возвращается
    assert clip.rework_round == gp.REPAIR_ROUNDS
    assert clip.needs_review is True
    assert NEGATION in clip.review_reason            # причина с разбора, по-русски
    assert gp._PARSE_PREFIX + NEGATION in clip.rework_pending
    cfg = gp.PromptConfig(media="image", on_unfixed="fail")
    with pytest.raises(gp.GeminiError, match="on_unfixed"):
        gp._fail_unfixed([clip], cfg)


# --- д: _run_body без атрибута segments_all ----------------------------------------------------

def test_run_body_without_segments_all_attribute_uses_false(tmp_path, monkeypatch):
    seen: list = []

    def fake_wanted(segments, path, flag):
        seen.append(flag)
        return [1]

    monkeypatch.setattr(orch, "wanted_segment_numbers", fake_wanted)
    orc = orch.Orchestrator.__new__(orch.Orchestrator)  # без __init__: атрибута segments_all нет
    orc.pilot = False
    orc.segments = []
    orc.missing_path = tmp_path / "missing.txt"
    orc.limit_clips = 0
    orc.work_dir = str(tmp_path)
    orc._tmp = None
    orc._owns_http = False
    orc._prepare_previews = lambda: None
    orc._run_inner = lambda missing: orch.RunSummary()
    assert not hasattr(orc, "segments_all")
    orc._run_body()
    assert seen == [False]
