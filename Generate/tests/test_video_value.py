"""Офлайн-тесты core/video_value.py (STEP 3). Фейковый send и временные файлы; сети нет."""
from __future__ import annotations

import json

import pytest

try:
    from Generate.core.segment_split import Segment
    from Generate.core.video_value import (
        build_value_prompt,
        parse_value_response,
        score_segments,
        write_distribution,
    )
    from Generate.core.segment_split import select_video_segments
except ImportError:  # запуск из каталога Generate
    from core.segment_split import Segment, select_video_segments
    from core.video_value import (
        build_value_prompt,
        parse_value_response,
        score_segments,
        write_distribution,
    )

SEGS = [
    Segment(id=1, start=0.0, end=4.0, text="Человек бежит"),
    Segment(id=2, start=4.0, end=8.0, text="Пустой стол"),
]


def test_prompt_contains_russian_instruction_and_anchors():
    prompt = build_value_prompt(SEGS)
    assert "video_value" in prompt
    assert "JSON" in prompt
    assert "движение или действие: 70–100" in prompt
    assert "статичный кадр: 0–30" in prompt
    assert "смешанный случай: 31–69" in prompt
    assert "Человек бежит" in prompt


def test_parse_accepts_fenced_json():
    text = '```json\n[{"id": 1, "video_value": 80}, {"id": 2, "video_value": 10}]\n```'
    assert parse_value_response(text, [1, 2]) == {1: 80, 2: 10}


def test_parse_accepts_plain_json():
    assert parse_value_response('[{"id": 1, "video_value": 0}]', [1]) == {1: 0}


@pytest.mark.parametrize(
    "text",
    [
        '[{"id": 1, "video_value": 80}]',                                   # нет id 2
        '[{"id": 1, "video_value": 80}, {"id": 2, "video_value": 1}, {"id": 3, "video_value": 1}]',  # лишний id
        '[{"id": 1, "video_value": 101}, {"id": 2, "video_value": 1}]',    # 101
        '[{"id": 1, "video_value": "80"}, {"id": 2, "video_value": 1}]',   # не int
        '[{"id": 1, "video_value": 80.0}, {"id": 2, "video_value": 1}]',   # float
        '[{"id": 1, "video_value": true}, {"id": 2, "video_value": 1}]',   # bool
        "не JSON",
    ],
)
def test_parse_rejects_bad_responses(text):
    with pytest.raises(ValueError) as exc:
        parse_value_response(text, [1, 2])
    # сообщение на русском: есть хотя бы одна кириллическая буква
    assert any("а" <= ch.lower() <= "я" for ch in str(exc.value))


def test_score_segments_retries_once_then_succeeds():
    calls = []

    def fake_send(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise RuntimeError("временный сбой")
        return '[{"id": 1, "video_value": 70}, {"id": 2, "video_value": 20}]'

    assert score_segments(SEGS, fake_send) == {1: 70, 2: 20}
    assert len(calls) == 2


def test_score_segments_bad_then_good_response():
    answers = iter(["не json", '[{"id": 1, "video_value": 50}, {"id": 2, "video_value": 60}]'])
    assert score_segments(SEGS, lambda p: next(answers)) == {1: 50, 2: 60}


def test_score_segments_always_fails_raises_russian_message():
    calls = []

    def fake_send(prompt):
        calls.append(prompt)
        return "ерунда"

    with pytest.raises(ValueError) as exc:
        score_segments(SEGS, fake_send)
    assert str(exc.value) == "Не удалось получить video_value для сегментов: 1, 2"
    assert len(calls) == 2


def test_write_distribution_keys(tmp_path):
    values = {1: 80, 2: 20}
    result = select_video_segments(SEGS, values, video_percent=50, intro_seconds=0)
    out = tmp_path / "distribution.json"
    write_distribution(out, result, SEGS, values)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"budget_seconds", "video_seconds", "intro_seconds_used", "segments"}
    assert data["budget_seconds"] == 4.0
    assert data["video_seconds"] == 4.0
    assert data["intro_seconds_used"] == 0
    assert [r["id"] for r in data["segments"]] == [1, 2]
    first = data["segments"][0]
    assert set(first) == {"id", "start", "end", "duration", "video_value", "mode"}
    assert first["mode"] == "video"
    assert first["video_value"] == 80
    assert first["duration"] == 4.0
    assert data["segments"][1]["mode"] == "photo"
