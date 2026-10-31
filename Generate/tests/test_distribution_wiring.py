"""Офлайн-тесты STEP 3b: связка распределения видео/фото с запуском Generate.

Без сети и без реального Gemini: отправитель send — подставная функция, файлы — во временном каталоге.
"""
from __future__ import annotations

import json
import re

import pytest

try:
    from Generate.core.segment_split import Segment, to_segments
    from Generate.core.srt_parser import Segment as SrtSegment
    from Generate.core import gemini_prompts as gp
    from Generate.core.orchestrator import split_for_run
except ImportError:  # запуск из каталога Generate
    from core.segment_split import Segment, to_segments  # type: ignore
    from core.srt_parser import Segment as SrtSegment  # type: ignore
    from core import gemini_prompts as gp  # type: ignore
    from core.orchestrator import split_for_run  # type: ignore


def srt_segments():
    """Пять сегментов SRT по 10 с каждый: номера 1..5."""
    return [
        SrtSegment(num=n, start_ms=(n - 1) * 10_000, end_ms=n * 10_000, text=f"текст {n}")
        for n in range(1, 6)
    ]


MISSING = [2, 4]


def fake_send_factory(calls: list):
    """Отправитель: записывает вызовы и возвращает корректный JSON для всех id из промпта."""
    def send(prompt: str) -> str:
        calls.append(prompt)
        ids = [int(m) for m in re.findall(r"^id (\d+) \(", prompt, flags=re.MULTILINE)]
        # id 2 получает высокую оценку, id 4 низкую
        return json.dumps([{"id": i, "video_value": 90 if i == 2 else 10} for i in ids])
    return send


def test_to_segments_keeps_only_missing_and_converts_ms_to_seconds():
    out = to_segments(srt_segments(), MISSING)
    assert [s.id for s in out] == [2, 4]
    assert all(isinstance(s, Segment) for s in out)
    assert out[0].start == 10.0 and out[0].end == 20.0
    assert out[0].text == "текст 2"
    assert out[1].start == 30.0 and out[1].end == 40.0


def test_to_segments_raises_for_unknown_missing_num():
    with pytest.raises(ValueError, match="отсутствуют в SRT"):
        to_segments(srt_segments(), [2, 99])


def test_none_branch_is_all_photo_and_never_calls_gemini(tmp_path):
    calls: list = []
    dist = tmp_path / "distribution.json"
    result = split_for_run(
        srt_segments(), MISSING, video_model="none", video_percent=100, intro_seconds=10,
        send=fake_send_factory(calls), distribution_path=dist)

    assert calls == []                      # Gemini не вызывался
    assert result["video"] == []
    assert result["photo"] == [2, 4]
    data = json.loads(dist.read_text(encoding="utf-8"))
    assert data["video_seconds"] == 0.0
    assert data["budget_seconds"] == 0.0
    assert [r["mode"] for r in data["segments"]] == ["photo", "photo"]


def test_ltx25_branch_writes_every_id_with_mode(tmp_path):
    calls: list = []
    dist = tmp_path / "distribution.json"
    # бюджет = 20 с (сумма missing) * 50 % = 10 с; вступления нет
    split_for_run(
        srt_segments(), MISSING, video_model="ltx25", video_percent=50, intro_seconds=0,
        send=fake_send_factory(calls), distribution_path=dist)

    assert len(calls) == 1                  # один запрос оценки
    data = json.loads(dist.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in data["segments"]}
    assert set(by_id) == {2, 4}             # каждый номер из missing
    assert all(r["mode"] in ("video", "photo") for r in data["segments"])
    assert by_id[2]["mode"] == "video" and by_id[2]["video_value"] == 90
    assert by_id[4]["mode"] == "photo" and by_id[4]["video_value"] == 10
    assert data["video_seconds"] == 10.0


def test_gemini_error_stops_run_with_russian_message_and_no_fallback(tmp_path):
    dist = tmp_path / "distribution.json"

    def broken_send(prompt: str) -> str:
        raise RuntimeError("сбой сети")

    with pytest.raises(gp.GeminiRuntimeError, match="Запуск остановлен"):
        split_for_run(
            srt_segments(), MISSING, video_model="ltx25", video_percent=50, intro_seconds=0,
            send=broken_send, distribution_path=dist)
    assert not dist.exists()                # распределение не записано, фолбэка нет
