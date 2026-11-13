"""Флаг needs_review после исправления очереди (process_fix).

Раньше флаг снимался только у клипов с причиной «исправление прервано» или «очередь не выполнена».
Клип с причиной содержания кадра (например «слишком много людей»), который исправлен и прошёл проверку,
оставался на проверке навсегда. Тест проверяет, что исправленный текст снимает флаг, а текст с нарушением
оставляет флаг с актуальной причиной.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Generate.core import gemini_prompts as gp  # noqa: E402
from Generate.core import prompt_check as pc  # noqa: E402

STYLE = _HERE.parents[1] / "model_profiles" / "models" / "zimage_turbo" / "prompts" / "style.md"


class _Logged:
    """Заглушка запроса: отдаёт заранее заданный ответ модели на исправление."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def start_batch(self, fix_no, label) -> None:  # noqa: D401
        return None

    def generate_json(self, model, system, user, schema) -> str:
        return self.reply


def _job(reply: str, clip: gp.PromptResult):
    cfg = gp.PromptConfig(media="image", style_file=str(STYLE))
    builder = SimpleNamespace(system="sys", schema={}, cfg=cfg, texts={clip.num: "Текст сегмента."})
    job = gp._WaveJob.__new__(gp._WaveJob)  # без полного конструктора: нужны только поля, которые читает process_fix
    job.builder = builder
    job.model = "m"
    job.tag = ""
    job.style_text = ""
    job.out = {}
    job.out_lock = __import__("threading").Lock()
    job.on_batch = None
    job.pool = None
    job.rework_queue = []
    job.sched_lock = __import__("threading").Lock()
    return job


def _clip(prompt: str, reason: str) -> gp.PromptResult:
    return gp.PromptResult(
        num=8,
        prompt=prompt,
        needs_review=True,
        review_reason=reason,
        rework_pending=pc.find_problems(prompt, "image"),
        rework_round=1,
        prompt_raw=prompt,
    )


def test_fixed_clip_clears_content_flag():
    clip = _clip("A crowd of villagers gathers in the square.", pc.CONTENT_REASON_RU[pc.PROBLEM_PEOPLE_COUNT])
    good = json.dumps([{"segment_index": 8, "prompt": "One adult man stands in a grey coat by a wall."}])
    job = _job(good, clip)
    gp._WaveJob.process_fix(job, _Logged(good), [clip], 1)
    assert clip.needs_review is False
    assert clip.review_reason == ""
    assert clip.rework_pending == []


def test_still_bad_clip_keeps_flag_with_current_reason():
    clip = _clip("A crowd of villagers gathers in the square.", "старая причина")
    clip.rework_round = gp.REPAIR_ROUNDS - 1  # следующий круг последний: причина пишется сразу
    bad = json.dumps([{"segment_index": 8, "prompt": "A crowd of villagers stands in the square."}])
    job = _job(bad, clip)
    gp._WaveJob.process_fix(job, _Logged(bad), [clip], 1)
    assert clip.needs_review is True
    assert clip.review_reason != "старая причина"
    assert pc.CONTENT_REASON_RU[pc.PROBLEM_PEOPLE_COUNT] in clip.review_reason
