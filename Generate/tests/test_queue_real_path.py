"""Очередь исправлений на реальном пути проверки промптов (verify_prompts) для ниши cartoon_photos.

Путь реального прогона (cli.prepare_prompts_only -> prompt_pipeline.build_prompts -> _photo_step):
- gp.generate_prompts(segments, pending, photo_cfg, managed, model, on_batch=on_batch) без pool, wave, rework
  и max_workers: одиночная волна, клипы попадают в очередь из главного батча, а не из сохранённого состояния;
- photo_cfg собирает cli._build_prompt_config(<ниша>, batch_size, preset, fail_fast=False): refs_enabled,
  relaxed_rules, rules.yaml, allowed_refs и on_unfixed берутся из profile.yaml ниши, а не из заглушки.

Существующий тест очереди (test_emotion_and_queue.py) идёт другим путём: общий пул, клип из rework=, PromptConfig
без refs. Здесь путь тот же, что у реального прогона.

Сценарий: батч из 25 клипов, клипы 11-14 первым ответом получают только причину разбора (стиль: keyword spam;
находок по содержанию нет). Ожидание: клипы идут в исправление; первый ответ исправления снова с причиной
разбора; второй ответ чистый; значит два раунда и три запроса, а в итоге клипы 11-14 без needs_review.

Причина разбора здесь «keyword spam», а не эмоциональное слово: профиль ниши отключает правило emotion
(relaxed_rules), поэтому слово «tense» в этой нише причиной не является.

Без сети: транспорт подменён заглушкой, которая отдаёт заранее заданные ответы по порядку.
Тест написан при подготовке изменения и не запускался (владелец запускает его на GitHub).
"""

from __future__ import annotations

import dataclasses
import json
import re
import socket
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Generate import cli  # noqa: E402
from Generate.core import gemini_prompts as gp  # noqa: E402
from Generate.core.srt_parser import Segment  # noqa: E402

NICHE = "cartoon_photos"
NUMS = list(range(1, 26))            # 25 клипов, как в прогоне
FLAGGED = [11, 12, 13, 14]           # клипы, помеченные первым ответом
SPAM_REASON = "стиль: keyword spam"  # причина разбора без находок по содержанию

CLEAN = "One adult man stands in a room <image1>."
BATCH_BAD = "One adult man stands in a room <image1>, photorealistic."    # ответ батча 1
FIX1_BAD = "One adult man stands in a room <image1>, highly detailed."    # первый ответ исправления
FIX2_OK = "One adult man stands by a window <image1>."                    # второй ответ исправления


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


class _Transport:
    """Заглушка транспорта: ответы по порядку, тексты запросов запоминаются; лишний запрос — IndexError."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.users: list = []

    def generate_json(self, model, system, user, schema) -> str:
        self.users.append(user)
        return self.replies[len(self.users) - 1]

    @property
    def calls(self) -> int:
        return len(self.users)


def _answer(texts: dict) -> str:
    # ниша с refs_enabled: у каждого элемента список refs; сочетание «1» разрешено профилем ниши
    return json.dumps([{"segment_index": n, "prompt": t, "refs": [1]} for n, t in texts.items()])


def _segments() -> list:
    return [Segment(n, (n - 1) * 1000, n * 1000, "A man stands in a room.") for n in NUMS]


def _real_cfg() -> gp.PromptConfig:
    # те же аргументы, что в prepare_prompts_only: профиль фото, размер батча, пресет «none», fail_fast выключен
    cfg = cli._build_prompt_config(NICHE, 25, "none", fail_fast=False, raw_dir=None, srt_path=None)
    # язык текста в кадре: английский (без разбора SRT поле пустое)
    return dataclasses.replace(cfg, frame_language="English")


def test_parse_flagged_batch_clips_get_two_fix_rounds_without_pool(monkeypatch):
    cfg = _real_cfg()
    # предусловия: конфиг действительно из profile.yaml ниши, а тексты ответов дают именно причину разбора
    assert cfg.refs_enabled is True and cfg.on_unfixed == "keep" and "emotion" in cfg.relaxed_rules
    style = lambda t: gp._style_reasons(t, "image", cfg.relaxed_rules, cfg.max_characters, cfg.rules)  # noqa: E731
    for bad in (BATCH_BAD, FIX1_BAD):
        assert SPAM_REASON in style(bad)
        assert gp._find_all_problems(bad, cfg, [1]) == []     # по содержанию находок нет: только разбор
    for ok in (CLEAN, FIX2_OK):
        assert style(ok) == []
        assert gp._find_all_problems(ok, cfg, [1]) == []

    rounds: list = []
    orig_start = gp._WaveJob.start_round

    def counting_start(self):
        orig_start(self)
        rounds.append(self.round_no)

    monkeypatch.setattr(gp._WaveJob, "start_round", counting_start)

    batch1 = _answer({n: (BATCH_BAD if n in FLAGGED else CLEAN) for n in NUMS})
    fix1 = _answer({n: FIX1_BAD for n in FLAGGED})
    fix2 = _answer({n: FIX2_OK for n in FLAGGED})
    transport = _Transport([batch1, fix1, fix2])
    saved: list = []

    # вызов как в _photo_step: без pool, wave, rework, max_workers; on_batch — колбэк сохранения
    results = gp.generate_prompts(_segments(), NUMS, cfg, transport, "m", on_batch=lambda rs: saved.append(list(rs)))

    assert transport.calls == 3                                  # батч, исправление 1, исправление 2
    assert rounds == [1, 2]                                      # два раунда исправлений
    first_fix = [int(x) for x in re.findall(r"Prompt for segment_index (\d+):", transport.users[1])]
    assert first_fix == FLAGGED                                  # в первый пакет вошли только клипы 11-14
    for n in FLAGGED:
        assert f"segment_index {n}: {FIX1_BAD}" in transport.users[2]   # второй круг получил текст первого ответа
    by_num = {r.num: r for r in results}
    for n in FLAGGED:
        assert by_num[n].needs_review is False
        assert by_num[n].review_reason == ""
        assert by_num[n].rework_pending == []
        assert "by a window" in by_num[n].prompt
    for n in NUMS:
        if n not in FLAGGED:
            assert by_num[n].needs_review is False
            assert by_num[n].rework_round is None                # клипы без нарушений очередь не проходили
