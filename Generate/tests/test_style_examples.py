"""Образцы «Good» из style.md не должны нарушать правила содержания.

Модель копирует образцы: если образец сам нарушает правило (толпа, речь, мелкие предметы), модель
будет получать нарушения при каждом прогоне. Также проверяются ложные срабатывания на слова
«shoots» (ростки) и «lips press» (губы без речи).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Generate.core import prompt_check as pc  # noqa: E402

LTX_STYLE = _HERE.parents[1] / "model_profiles" / "models" / "ltx25" / "prompts" / "style.md"


def _good_examples() -> list:
    text = LTX_STYLE.read_text(encoding="utf-8")
    return re.findall(r"Good:\s*(.+?)(?=\n|$)", text)


def test_good_examples_present():
    assert len(_good_examples()) >= 10


def test_good_examples_pass_content_rules():
    bad = [(g[:60], [pc.problem_code(p) for p in pc.find_problems(g, "video")]) for g in _good_examples()]
    bad = [b for b in bad if b[1]]
    assert bad == []


def test_plant_shoot_is_not_gunfire():
    assert pc.content_hits("The camera holds on the small green shoot as sunlight reaches it.", "video") == []


def test_lips_pressed_is_not_speech():
    assert pc.content_hits("Her lips press together and she lowers her head.", "video") == []


def test_lips_moving_is_speech():
    assert pc.PROBLEM_SPEECH in {c for c, _ in pc.content_hits("Her lips move as she speaks.", "video")}
