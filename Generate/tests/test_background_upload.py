"""A2: скачивание и ack сразу после готовности клипа, фоновая выгрузка в Release, пакетные ссылки и дренаж.

Тесты pytest (для GitHub Actions). Загрузчик, воркер и диск подменяются; сеть не используется.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Generate.core import gemini_prompts as gp  # noqa: E402
from Generate.core import orchestrator as orch  # noqa: E402


class _Resp:
    def __init__(self, data=None):
        self._data = data or {}

    def json(self):
        return self._data


class _Uploader:
    """Заглушка ReleaseUploader с параметром refresh_links: пишет вызовы; падает на заданных номерах."""

    def __init__(self, fail_on=(), gate=None):
        self.fail_on = set(fail_on)
        self.gate = gate            # threading.Event: загрузка ждёт сигнала (проверка порядка ack и выгрузки)
        self.clips = []             # (имя файла, значение refresh_links, имя потока)
        self.refresh_at = []        # число выгруженных клипов в момент каждого refresh_links()

    def upload_clip(self, path, refresh_links=True):
        if self.gate is not None:
            self.gate.wait(5.0)
        name = os.path.basename(path)
        num = int(name.split(".")[0])
        if num in self.fail_on:
            raise gp.GeminiRuntimeError(f"сбой загрузки {name}")
        self.clips.append((name, refresh_links, threading.current_thread().name))
        return True

    def refresh_links(self):
        self.refresh_at.append(len(self.clips))


class _SlowUploader(_Uploader):
    def upload_clip(self, path, refresh_links=True):
        time.sleep(0.005)
        return super().upload_clip(path, refresh_links)


class _OneArgUploader:
    """Как FakeUploader в существующих тестах: upload_clip принимает только путь, refresh_links нет."""

    def __init__(self):
        self.clips = []

    def upload_clip(self, path):
        self.clips.append(os.path.basename(path))


def _orc(tmp_path, uploader, total=40):
    """Оркестратор без конструктора: состояние и заглушки воркера (вызовы пишутся в events)."""
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    o._total_needed = total
    o._run_pos = {n: n for n in range(1, total + 1)}
    o._video_queue = deque()
    o._photo_queue = deque()
    o._prompts = {str(n): {"prompt": f"промпт {n}", "num_frames": 5} for n in range(1, total + 1)}
    o._state = {"prompts": o._prompts}
    o._attempts, o._done, o._failed, o._durations = {}, set(), set(), []
    o._slots = {}
    o._tmp = str(tmp_path)
    o._uploader = uploader
    o.width, o.template = 0, "{num}.mp4"
    o.seed_base = 0
    o.prompts_path = str(tmp_path / "prompts.json")
    o._save_state = lambda: None  # файл промптов в тестах не пишем
    o.events = []

    def call(slot, method, path, payload=None):
        o.events.append((method, path))
        return _Resp({"id": f"t{payload['num']}"}) if payload else _Resp()

    def download(slot, tid, path):
        o.events.append(("download", tid))
        with open(path, "wb") as fh:
            fh.write(b"clip")

    o._call = call
    o._download = download
    return o


def _slot(index=1):
    return orch._CardSlot(index=index, generation=1, instance_id=77, machine_id=5, price=0.1, created=0.0)


def _run_clip(o, slot, num, now=1.0):
    """Клип уходит на карту и приходит готовым: как _poll_task, без опроса статуса."""
    task = orch._Task(num, f"t{num}", 0.0)
    slot.tasks.append(task)
    o._finish_clip(slot, now, task=task)
    return task


def test_download_then_ack_before_upload_finishes(tmp_path):
    """Ack уходит сразу после скачивания; загрузка в Release ещё заблокирована и не мешает ack."""
    gate = threading.Event()
    up = _Uploader(gate=gate)
    o = _orc(tmp_path, up)
    slot = _slot()
    o._slots[1] = slot
    o._start_upload_thread()
    try:
        _run_clip(o, slot, 1)
        o._join_download_threads()
        assert o.events == [("download", "t1"), ("POST", "/ack/t1")]
        assert slot.tasks == []                 # задача снята с карты
        assert o._pending_uploads_count() == 1  # файл ждёт в очереди выгрузки
        assert up.clips == []                   # GitHub ещё не получил клип
    finally:
        gate.set()
        o._stop_upload_thread()
    assert up.clips == [("1.mp4", False, "release-uploader")]
    assert o._pending_uploads == 0
    assert 1 in o._done
    assert not os.path.exists(os.path.join(str(tmp_path), "pending", "t1", "1.mp4"))


def test_upload_runs_inline_without_background_thread(tmp_path):
    """Вне _work (прямой вызов _finish_clip, как в старых тестах) выгрузка идёт в этом же потоке."""
    up = _Uploader()
    o = _orc(tmp_path, up)
    slot = _slot()
    o._slots[1] = slot
    _run_clip(o, slot, 1)
    assert o._done == {1}
    assert up.clips == [("1.mp4", False, threading.current_thread().name)]
    assert o._pending_uploads == 0
    assert o.events == [("download", "t1"), ("POST", "/ack/t1")]


def test_links_refreshed_every_20_uploaded_clips(tmp_path):
    """refresh_links после 20-го клипа; клипы 21..25 выгружены, их обновление — на следующем пакете."""
    up = _Uploader()
    o = _orc(tmp_path, up, total=25)
    slot = _slot()
    o._slots[1] = slot
    o._start_upload_thread()
    try:
        for n in range(1, 26):
            _run_clip(o, slot, n)
    finally:
        o._stop_upload_thread()
    assert len(up.clips) == 25
    assert up.refresh_at == [20]
    assert o._uploaded_since_links == 5
    assert all(refresh is False for _, refresh, _ in up.clips)
    assert all(thread == "release-uploader" for _, _, thread in up.clips)
    assert o._done == set(range(1, 26))
    assert len(o._durations) == 25


def test_upload_error_routes_to_fail_clip_and_cleans_up(tmp_path):
    """Ошибка загрузки клипа 3 идёт в _fail_clip, файл удаляется, счётчик ожидающих не остаётся."""
    up = _Uploader(fail_on={3})
    o = _orc(tmp_path, up)
    slot = _slot()
    o._slots[1] = slot
    failed = []
    orig_fail = o._fail_clip

    def spy(num, reason):
        failed.append((num, reason))
        orig_fail(num, reason)

    o._fail_clip = spy
    o._start_upload_thread()
    try:
        for n in range(1, 6):
            _run_clip(o, slot, n)
    finally:
        o._stop_upload_thread()
    assert [num for num, _ in failed] == [3]
    assert "не удалось загрузить в Release" in failed[0][1]
    assert 3 in o._video_queue      # попыток ещё мало: клип возвращён в очередь на повтор
    assert 3 not in o._done
    assert o._done == {1, 2, 4, 5}
    assert o._pending_uploads == 0
    assert not os.path.exists(os.path.join(str(tmp_path), "pending", "t3", "3.mp4"))


def test_download_failure_sends_no_ack_and_fails_clip(tmp_path):
    """Нет файла — нет ack: воркер хранит клип, а клип уходит в _fail_clip."""
    up = _Uploader()
    o = _orc(tmp_path, up)

    def bad_download(slot, tid, path):
        raise orch._WorkerHttpError(503)

    o._download = bad_download
    slot = _slot()
    o._slots[1] = slot
    _run_clip(o, slot, 1)
    assert ("POST", "/ack/t1") not in o.events
    assert 1 in o._video_queue
    assert o._pending_uploads == 0
    assert up.clips == []


def test_ack_failure_does_not_discard_downloaded_clip(tmp_path):
    """Сбой ack не отменяет уже скачанный и выгруженный клип: только предупреждение в лог."""
    up = _Uploader()
    o = _orc(tmp_path, up)

    def call(slot, method, path, payload=None):
        o.events.append((method, path))
        if path.startswith("/ack/"):
            raise orch._WorkerHttpError(500)
        return _Resp({"id": f"t{payload['num']}"}) if payload else _Resp()

    o._call = call
    slot = _slot()
    o._slots[1] = slot
    _run_clip(o, slot, 1)
    assert up.clips == [("1.mp4", False, threading.current_thread().name)]
    assert 1 in o._done


def test_work_waits_for_all_queued_uploads_before_returning(tmp_path):
    """_work дозаливает всё из очереди до выхода: _tmp не удаляется раньше времени."""
    up = _SlowUploader()
    o = _orc(tmp_path, up, total=12)
    slot = _slot()
    o._slots[1] = slot

    def loop(ready):
        for n in range(1, 13):
            _run_clip(o, slot, n)  # все клипы уходят в очередь выгрузки; цикл возвращается сразу

    o._work_loop = loop
    o._work([])
    assert len(up.clips) == 12
    assert o._pending_uploads == 0
    assert o._upload_thread is None


def test_work_drains_uploads_even_if_loop_fails(tmp_path):
    """Сбой цикла не оставляет скачанные клипы без выгрузки: дренаж в finally."""
    up = _SlowUploader()
    o = _orc(tmp_path, up)
    slot = _slot()
    o._slots[1] = slot

    def loop(ready):
        for n in range(1, 4):
            _run_clip(o, slot, n)
        raise RuntimeError("сбой цикла")

    o._work_loop = loop
    with pytest.raises(RuntimeError, match="сбой цикла"):
        o._work([])
    assert len(up.clips) == 3
    assert o._upload_thread is None


def test_single_arg_uploader_works_in_background(tmp_path):
    """Загрузчик без refresh_links (как FakeUploader): выгрузка идёт, пакетное обновление не вызывается."""
    up = _OneArgUploader()
    o = _orc(tmp_path, up, total=21)
    slot = _slot()
    o._slots[1] = slot
    o._start_upload_thread()
    try:
        for n in range(1, 22):
            _run_clip(o, slot, n)
    finally:
        o._stop_upload_thread()
    assert len(up.clips) == 21
    assert o._done == set(range(1, 22))
    assert o._uploaded_since_links == 1  # счётчик сбросился на 20-м клипе
