"""Исключение emotion для ниши cartoon_photos, очередь исправлений через общий пул и on_unfixed: keep.

Проверяется:
- a) профиль с emotion снимает причину «стиль: эмоциональная метка» (в том числе для «body tense with eyes wide»);
- b) без emotion та же фраза даёт причину «стиль: эмоциональная метка»;
- c) parse_relaxed_rules принимает id emotion; неизвестное имя — ошибка на русском;
- d) клип 17 после первого круга с причиной разбора уходит во второй круг через общий пул: два запроса, два раунда;
- e) on_unfixed: keep — клип 17, оставшийся с нарушением после последнего круга, не поднимает исключение
  и остаётся на проверке.

Без сети: транспорт подменён заглушкой, которая считает вызовы и отдаёт заранее заданные ответы.
Тесты написаны при подготовке изменения и не запускались (владелец запускает их на GitHub).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Generate.core import gemini_prompts as gp  # noqa: E402
from Generate.core.srt_parser import Segment  # noqa: E402

GEN = _HERE.parents[1]
NICHE = GEN / "Nishes" / "cartoon_photos"
STYLE = GEN / "model_profiles" / "models" / "zimage_turbo" / "prompts" / "style.md"
TENSE_PHRASES = ["The main character stands, body tense", "The main character stands, body tense with eyes wide"]


class _Transport:
    """Заглушка транспорта: считает вызовы, отдаёт ответы по порядку, запоминает тексты запросов."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.users: list = []

    def generate_json(self, model, system, user, schema) -> str:
        self.users.append(user)
        return self.replies[len(self.users) - 1]

    @property
    def calls(self) -> int:
        return len(self.users)


def _answer(text: str) -> str:
    return json.dumps([{"segment_index": 17, "prompt": text}])


def _cfg(on_unfixed: str = "keep", relaxed=frozenset()) -> gp.PromptConfig:
    return gp.PromptConfig(media="image", style_file=str(STYLE), on_unfixed=on_unfixed, relaxed_rules=relaxed)


def _seed(prompt: str, rework_round: int = 0) -> gp.PromptResult:
    # клип из очереди исправлений: текст уже есть, батчей для него нет (rework=), сырой текст сохранён
    return gp.PromptResult(num=17, prompt=prompt, needs_review=False, review_reason="",
                           rework_pending=[], rework_round=rework_round, prompt_raw=prompt)


def _segments():
    return [Segment(17, 0, 1000, "The main character stands in a quiet room.")]


def _run_pool(cfg, transport, seed):
    # одиночная волна в общем пуле: одна волна «photo», один поток, батчей нет — работают только исправления
    pool = gp.SharedBatchPool({"photo": 1})
    return gp.generate_prompts(_segments(), [], cfg, transport, "m", pool=pool, wave="photo", rework=[seed])


# --- а/б/в: исключение emotion в проверке стиля и разбор профиля --------------------------------

def test_a_emotion_relaxed_passes_style_check():
    lines = (NICHE / "profile.yaml").read_text(encoding="utf-8").splitlines()
    relaxed, _ = gp.profile_relaxation(lines, "cartoon_photos")
    assert "emotion" in relaxed          # профиль ниши действительно включает emotion
    for text in TENSE_PHRASES:
        reasons = gp._style_reasons(text, "image", frozenset({"emotion"}))
        assert gp._EMOTION_REASON not in reasons
        assert [r for r in reasons if r.startswith("стиль")] == []


def test_b_emotion_not_relaxed_still_flagged():
    for text in TENSE_PHRASES:
        reasons = gp._style_reasons(text, "image", frozenset())
        assert "стиль: эмоциональная метка" in reasons


def test_c_parse_relaxed_rules_accepts_emotion_and_rejects_unknown():
    assert "emotion" in gp.parse_relaxed_rules(["emotion"], "cartoon_photos")
    with pytest.raises(gp.GeminiInputError, match="недопустимо"):
        gp.parse_relaxed_rules(["child_in_frame"], "cartoon_photos")


# --- д: очередь исправлений через общий пул ---------------------------------------------------

def test_d_parse_flagged_clip_17_gets_second_round_in_pool(monkeypatch):
    # предусловие: первый ответ клипа 17 без находок по тексту, но с причиной разбора (эмоция), профиль без emotion
    round1 = "The main character stands in a quiet room, body tense."
    round2 = "One adult man stands in a quiet room."
    assert gp._EMOTION_REASON in gp._style_reasons(round1, "image", frozenset())
    rounds: list = []
    orig_start = gp._WaveJob.start_round

    def counting_start(self):
        orig_start(self)
        rounds.append(self.round_no)

    monkeypatch.setattr(gp._WaveJob, "start_round", counting_start)
    transport = _Transport([_answer(round1), _answer(round2)])
    seed = _seed("The main character stands in a quiet room.")
    results = _run_pool(_cfg(on_unfixed="keep"), transport, seed)
    assert transport.calls == 2                             # два запроса: первый круг и второй
    assert rounds == [1, 2]                                 # два раунда исправлений
    assert f"segment_index 17: {round1}" in transport.users[1]   # второй круг получил текст первого ответа
    assert results[0].num == 17
    assert results[0].needs_review is False
    assert "One adult man" in results[0].prompt


# --- е: on_unfixed: keep — клип остаётся на проверке и прогон не останавливается -----------------

def test_e_on_unfixed_keep_leaves_clip_flagged_without_raise():
    still_bad = "The main character stands in a quiet room, body tense."
    transport = _Transport([_answer(still_bad)])
    seed = _seed("The main character stands in a quiet room.", rework_round=gp.REPAIR_ROUNDS - 1)
    results = _run_pool(_cfg(on_unfixed="keep"), transport, seed)   # исключения нет
    assert transport.calls == 1                             # последний круг: больше запросов нет
    assert results[0].needs_review is True
    assert gp._EMOTION_REASON in results[0].review_reason
