"""Кадры по типу профиля (офлайн, без сети).

Фикстуры: граф и привязки из models/zimage_turbo (изображение) и models/ltx25 (видео).
Файлы только читаются. Тип задаётся переменной WORKER_PROFILE_TYPE.
"""
from __future__ import annotations

import json
import pathlib
import socket

import pytest

try:
    from Generate.worker import comfy_client as cc
except ImportError:  # pragma: no cover
    from worker import comfy_client as cc  # type: ignore

MODELS = pathlib.Path(__file__).resolve().parents[1] / "model_profiles" / "models"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _zimage():
    return _json(MODELS / "zimage_turbo" / "workflow.json"), _json(MODELS / "zimage_turbo" / "bindings.json")


def _ltx():
    return _json(MODELS / "ltx25" / "workflow.json"), _json(MODELS / "ltx25" / "bindings.json")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def _blocked(*a, **k):
        raise AssertionError("Сетевые обращения в тестах запрещены")
    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.mark.parametrize("env_value", ["video", None])
def test_video_without_num_frames_raises(monkeypatch, env_value):
    # Видео без num_frames: та же ошибка, что и раньше (None — тип не задан, по умолчанию видео)
    if env_value is None:
        monkeypatch.delenv("WORKER_PROFILE_TYPE", raising=False)
    else:
        monkeypatch.setenv("WORKER_PROFILE_TYPE", env_value)
    graph, bindings = _ltx()
    bindings = {k: v for k, v in bindings.items() if k != "num_frames"}
    with pytest.raises(cc.ComfyError, match="обязательных полей: num_frames"):
        cc.inject_workflow(graph, bindings, "текст", 25, 1, "clip")


def test_image_without_num_frames_injects(monkeypatch):
    # Изображение без num_frames: загружается и подставляется без ошибки, кадры не подставляются
    monkeypatch.setenv("WORKER_PROFILE_TYPE", "image")
    graph, bindings = _zimage()
    assert "num_frames" not in bindings
    out = cc.inject_workflow(graph, bindings, "кадр", 25, 7, "photo_1")
    assert out["57:27"]["inputs"]["text"] == "кадр"
    assert out["57:3"]["inputs"]["seed"] == 7
    assert out["9"]["inputs"]["filename_prefix"] == "photo_1"
    assert out["57:13"]["inputs"] == graph["57:13"]["inputs"]  # размер не тронут


@pytest.mark.parametrize("env_value", ["video", None])
def test_video_with_num_frames_unchanged(monkeypatch, env_value):
    # Видео с num_frames: число кадров подставляется, как и раньше
    if env_value is None:
        monkeypatch.delenv("WORKER_PROFILE_TYPE", raising=False)
    else:
        monkeypatch.setenv("WORKER_PROFILE_TYPE", env_value)
    graph, bindings = _ltx()
    out = cc.inject_workflow(graph, bindings, "текст", 49, 3, "clip")
    nf = bindings["num_frames"]
    assert out[str(nf["node_id"])]["inputs"][nf["field"]] == 49
