"""Офлайн-тесты двухфазного оркестратора и Release-адаптера для Cartoon I2V.

Покрывают:
- разделение done_photos / done_videos, allow_upgrade=True и дедупликацию links_text (только N.mp4);
- двухфазный рендер (Phase 1 фото -> переключение POST /profile -> Phase 2 I2V с POST /input_image);
- буфер MAX_CARD_QUEUE_DEPTH = 4 (1 активная + 3 в очереди) на Phase 2 с предзагрузкой /input_image;
- resume между фазами (скачивание N.png из Release и пропуск Phase 1);
- fallback на N.png при исчерпании попыток I2V, запись video_render_status и отчёт.
"""
from __future__ import annotations

import json
import sys
import types
from collections import deque
from pathlib import Path

import httpx
import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from Generate.core import orchestrator as orch
    from Generate.core import prompt_pipeline as pp
    from Generate.core import release_adapter as ra
    from Generate.core import vast_client as vc
    from Generate.tests.test_orchestrator import (
        CARD_IP, CARD_PORT, FULL_HASH, FINAL, PART1, TOKEN, VALID_BUILD, FakeClock, FakeVast, make_offer,
    )
except ImportError:  # pragma: no cover
    from core import orchestrator as orch  # type: ignore
    from core import prompt_pipeline as pp  # type: ignore
    from core import release_adapter as ra  # type: ignore
    from core import vast_client as vc  # type: ignore
    from tests.test_orchestrator import (  # type: ignore
        CARD_IP, CARD_PORT, FULL_HASH, FINAL, PART1, TOKEN, VALID_BUILD, FakeClock, FakeVast, make_offer,
    )


class I2VFakeStore(orch.ReleaseStore):
    """In-memory ReleaseStore с поддержкой upload_asset и download_asset."""

    def __init__(self, events: list) -> None:
        self.events = events
        self.releases: dict[str, orch.ReleaseInfo] = {}
        self.created: list[str] = []
        self.uploads: list[tuple[str, str]] = []
        self.blobs: dict[tuple[str, str], bytes] = {}
        self.downloads: list[tuple[str, str]] = []

    def preload(self, assets, *, tag: str = PART1, blobs: dict | None = None) -> None:
        self.releases[tag] = orch.ReleaseInfo(
            tag=tag, title="t", body=f"hash: {FULL_HASH}\nstage: gen\nrole: part", assets=list(assets),
        )
        for name, data in (blobs or {}).items():
            key = name if isinstance(name, tuple) else (tag, name)
            self.blobs[key] = data

    def find_releases(self, tag: str) -> list:
        rel = self.releases.get(tag)
        return [rel] if rel else []

    def create_release(self, tag: str, title: str, body: str) -> None:
        self.events.append("release_create")
        self.created.append(tag)
        self.releases[tag] = orch.ReleaseInfo(tag=tag, title=title, body=body, assets=[])

    def upload_asset(self, tag: str, path: str) -> None:
        import os
        name = os.path.basename(path)
        self.events.append(f"upload:{name}")
        self.uploads.append((tag, name))
        with open(path, "rb") as fh:
            self.blobs[(tag, name)] = fh.read()
        assets = self.releases[tag].assets
        if name not in assets:
            assets.append(name)

    def download_asset(self, tag: str, asset_name: str, dest: str) -> bool:
        self.downloads.append((tag, asset_name))
        data = self.blobs.get((tag, asset_name))
        if data is None:
            return False
        Path(dest).write_bytes(data)
        return True


class FakeI2VWorker:
    """HTTP-воркер с поддержкой /profile, /input_image (Bearer) и раздельных фаз фото/I2V."""

    def __init__(self, clock, token: str, events: list, *, render_sec: float = 10.0) -> None:
        self.clock = clock
        self.token = token
        self.events = events
        self.render_sec = render_sec
        self.profile = "cartoon"
        self.switch_state = "idle"
        self.profile_switches: list[str] = []
        self.input_images: dict[str, bytes] = {}
        self.input_image_calls: list[tuple[str, str]] = []
        self.task_calls: list[dict] = []
        self.call_sequence: list[str] = []
        self.video_plan: dict[int, list[str]] = {}
        self.photo_plan: dict[int, list[str]] = {}
        self.tasks: dict[str, dict] = {}
        self._seq = 0
        self._gpu_free_at = 0.0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("X-Worker-Token") != self.token:
            return httpx.Response(401, json={"error": "токен"})
        path, method = request.url.path, request.method
        if method == "GET" and path == "/health":
            return httpx.Response(200, json={
                "ok": True, "gpu": "FakeGPU", "vram_free": 12345, "switch": self.switch_state,
            })
        if method == "GET" and path == "/logs":
            return httpx.Response(200, text="", headers={"X-Log-Next": "0"})
        if method == "POST" and path == "/profile":
            body = json.loads(request.content)
            target = body["profile"]
            self.profile = target
            self.switch_state = "ready"
            self.profile_switches.append(target)
            self.events.append(f"switch:{target}")
            return httpx.Response(202, json={"status": "switching", "profile": target})
        if method == "POST" and path == "/input_image":
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {self.token}":
                return httpx.Response(401, json={"error": "bearer"})
            name = request.url.params.get("name", "")
            self.input_images[name] = bytes(request.content)
            self.input_image_calls.append((name, auth))
            self.call_sequence.append(f"input_image:{name}")
            return httpx.Response(200, json={"name": name, "bytes": len(request.content)})
        if method == "POST" and path == "/task":
            body = json.loads(request.content)
            num = int(body["num"])
            input_image = body.get("input_image")
            is_i2v = input_image is not None
            self.task_calls.append({**body, "profile": self.profile})
            self.call_sequence.append(f"task:{num}:{'i2v' if is_i2v else 'photo'}")
            self._seq += 1
            tid = f"t{self._seq}"
            plan = (self.video_plan if is_i2v else self.photo_plan).get(num, [])
            outcome = plan.pop(0) if plan else "ok"
            render_start = max(self.clock(), self._gpu_free_at)
            self._gpu_free_at = render_start + self.render_sec
            self.tasks[tid] = {
                "num": num, "is_i2v": is_i2v, "render_start": render_start, "outcome": outcome,
            }
            return httpx.Response(200, json={"id": tid})
        if method == "GET" and path.startswith("/task/"):
            t = self.tasks[path.rsplit("/", 1)[1]]
            now = self.clock()
            if t["outcome"] == "error" and now >= t["render_start"]:
                return httpx.Response(200, json={"state": "error", "error": "сбой I2V"})
            if t["outcome"] == "ok" and now >= t["render_start"] + self.render_sec:
                return httpx.Response(200, json={"state": "done"})
            if now >= t["render_start"]:
                return httpx.Response(200, json={"state": "running"})
            return httpx.Response(200, json={"state": "queued"})
        if method == "GET" and path.startswith("/file/"):
            t = self.tasks[path.rsplit("/", 1)[1]]
            tag = "MP4" if t["is_i2v"] else "PNG"
            return httpx.Response(200, content=f"{tag}-{t['num']}".encode())
        if method == "POST" and path.startswith("/ack/"):
            return httpx.Response(200, json={})
        return httpx.Response(404)


def _segments(nums):
    return [
        types.SimpleNamespace(num=n, start_ms=(n - 1) * 5000, end_ms=n * 5000 - 500, text=f"seg {n}")
        for n in nums
    ]


def _make_prompts_state(entries: dict[int, dict]) -> dict:
    return {
        "srt_hash": FULL_HASH,
        "model": "fake",
        "profile": "cartoon",
        "video_profile": "cartoon_i2v",
        "i2v_mode": True,
        "run_seed": 42,
        "prompts": {str(k): dict(v) for k, v in entries.items()},
    }


def _make_i2v_rig(tmp_path, monkeypatch, *, missing=(1, 2, 3), entries: dict | None = None,
                  preload=None, preload_blobs=None):
    events: list = []
    clock = FakeClock()
    work_dir = tmp_path / "work"
    prompts_path = tmp_path / "prompts.json"
    worker = FakeI2VWorker(clock, TOKEN, events)
    store = I2VFakeStore(events)
    if preload is not None:
        store.preload(preload, blobs=preload_blobs)
    vast = FakeVast([make_offer(1, 11)], events)
    segs = _segments(range(1, max(missing) + 1))

    monkeypatch.setattr(orch, "parse_missing", lambda path, s: list(missing))
    monkeypatch.setattr(ra, "srt_hash", lambda s: FULL_HASH)
    monkeypatch.setattr(orch.go, "srt_hash", lambda s: FULL_HASH)

    build_calls: list[dict] = []
    default_entries = entries or {
        n: {
            "num": n, "status": "ready", "prompt": f"photo prompt {n}", "num_frames": 1,
            "i2v": n != 3,
            **({
                "video_prompt": f"video prompt {n}",
                "video_num_frames": 25,
                "video_status": "ready",
            } if n != 3 else {}),
        }
        for n in missing
    }

    def fake_build_i2v_prompts(needed, **kw):
        build_calls.append({"needed": list(needed), **kw})
        state = _make_prompts_state(default_entries)
        orch.go.atomic_write_json(kw["prompts_path"], state)
        ready = [n for n in needed if state["prompts"].get(str(n), {}).get("status") == "ready"]
        return pp.PromptRun(ready=ready, state=state, errors=[], abort=None)

    monkeypatch.setattr(orch, "build_i2v_prompts", fake_build_i2v_prompts)

    http = httpx.Client(transport=httpx.MockTransport(worker))
    orc = orch.Orchestrator(
        segs,
        tmp_path / "missing.txt",
        types.SimpleNamespace(preset="cartoon", niche_preset_file=None),
        types.SimpleNamespace(fps=24, frame_rule={"kind": "8k+1"}, clip_min_sec=2.0,
                              clip_max_sec=8.0, gpu_max_sec=10.0, tail_pad_sec=0.5),
        store,
        vast,
        object(),
        prompts_path,
        repo="owner/repo",
        run_id="i2v-test",
        profile="cartoon",
        docker_image="img:latest",
        worker_token=TOKEN,
        poll_interval_sec=5.0,
        work_dir=work_dir,
        worker_build=VALID_BUILD,
        http_client=http,
        clock=clock,
        sleep=clock.sleep,
        i2v_mode=True,
        intro_seconds=180.0,
        intro_video_percent=70.0,
        after_intro_video_percent=10.0,
        min_anim_sec=2.0,
        i2v_video_cfg=types.SimpleNamespace(preset="cartoon_i2v"),
        i2v_video_profile="cartoon_i2v",
        i2v_timing=types.SimpleNamespace(fps=24, frame_rule={"kind": "8k+1"}, clip_min_sec=2.0,
                                         clip_max_sec=8.0, gpu_max_sec=10.0, tail_pad_sec=0.5),
    )
    return types.SimpleNamespace(
        orc=orc, store=store, worker=worker, vast=vast, clock=clock,
        prompts_path=prompts_path, build_calls=build_calls, events=events,
    )


def test_release_uploader_done_split_allow_upgrade_and_links_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "srt_hash", lambda s: FULL_HASH)
    events: list = []
    store = I2VFakeStore(events)
    store.preload(["1.png", "2.png", "3.mp4", "5.png"], blobs={(PART1, "1.png"): b"PNG1"})
    u = ra.ReleaseUploader(store, _segments([1, 2, 3, 5]), "cartoon", log=lambda m: None, repo="owner/repo")

    assert u.prepare() == {1, 2, 3, 5}
    assert u.done_photos == {1, 2, 5}
    assert u.done_videos == {3}
    assert u.done == {1, 2, 3, 5}

    mp4 = tmp_path / "5.mp4"
    mp4.write_bytes(b"MP4-5")
    assert u.upload_clip(str(mp4), refresh_links=False, allow_upgrade=False) is False
    assert u.upload_clip(str(mp4), refresh_links=False, allow_upgrade=True) is True
    assert u.upload_clip(str(mp4), refresh_links=False, allow_upgrade=True) is False
    assert 5 in u.done_videos

    links = u.links_text("owner/repo").splitlines()
    assert links == [
        f"https://github.com/owner/repo/releases/download/{PART1}/1.png",
        f"https://github.com/owner/repo/releases/download/{PART1}/2.png",
        f"https://github.com/owner/repo/releases/download/{PART1}/3.mp4",
        f"https://github.com/owner/repo/releases/download/{PART1}/5.mp4",
    ]
    assert not any(line.endswith("/5.png") for line in links)

    dest = tmp_path / "cache" / "1.png"
    out = u.download_clip_asset("1.png", dest)
    assert out == dest
    assert dest.read_bytes() == b"PNG1"


def test_i2v_two_phase_render_switch_and_links_dedup(tmp_path, monkeypatch):
    rig = _make_i2v_rig(tmp_path, monkeypatch, missing=(1, 2, 3))
    summary = rig.orc.run()

    assert summary.exit_code == 0
    assert (summary.completed, summary.failed, summary.pending) == (3, 0, 0)
    assert len(rig.build_calls) == 1
    assert rig.build_calls[0]["photo_profile"] == "cartoon"
    assert rig.build_calls[0]["video_profile"] == "cartoon_i2v"

    # Карта стартовала с cartoon и переключилась на cartoon_i2v после фазы фото
    assert rig.vast.created[0]["env"]["MODEL_PROFILE"] == "cartoon"
    assert rig.worker.profile_switches == ["cartoon_i2v"]

    # В Phase 1 рендерились фото 1..3; в Phase 2 — только I2V клипы 1 и 2 с предзагрузкой /input_image
    photo_tasks = [c for c in rig.worker.task_calls if "input_image" not in c]
    i2v_tasks = [c for c in rig.worker.task_calls if "input_image" in c]
    assert [c["num"] for c in photo_tasks] == [1, 2, 3]
    assert all(c["num_frames"] == 1 for c in photo_tasks)
    assert [c["num"] for c in i2v_tasks] == [1, 2]
    assert all(c["input_image"] == f"clip_{c['num']}.png" for c in i2v_tasks)
    assert all(c["prompt"] == f"video prompt {c['num']}" and c["num_frames"] == 25 for c in i2v_tasks)
    for n in (1, 2):
        img_idx = rig.worker.call_sequence.index(f"input_image:clip_{n}.png")
        task_idx = rig.worker.call_sequence.index(f"task:{n}:i2v")
        assert img_idx < task_idx
        assert rig.worker.input_images[f"clip_{n}.png"] == f"PNG-{n}".encode()

    links = rig.store.blobs[(FINAL, "generated_links.txt")].decode("utf-8").splitlines()
    assert [line.rsplit("/", 1)[-1] for line in links] == ["1.mp4", "2.mp4", "3.png"]


def test_i2v_phase2_maintains_max_card_queue_depth_4(tmp_path, monkeypatch):
    rig = _make_i2v_rig(tmp_path, monkeypatch, missing=(1, 2, 3, 4, 5))
    orc = rig.orc
    orc._prompts = {
        str(n): {
            "num": n, "status": "ready", "prompt": f"p {n}", "num_frames": 1,
            "i2v": True, "video_prompt": f"vp {n}", "video_num_frames": 25, "video_status": "ready",
        }
        for n in range(1, 6)
    }
    cache = tmp_path / "photos_cache"
    cache.mkdir()
    for n in range(1, 6):
        (cache / f"{n}.png").write_bytes(f"PNG-{n}".encode())
    orc._photos_cache = cache
    orc._i2v_phase2_nums = {1, 2, 3, 4, 5}
    orc._video_queue = deque([1, 2, 3, 4, 5])
    orc._photo_queue = deque()

    slot = orch._CardSlot(index=1, generation=1, instance_id=101, machine_id=11,
                          price=0.3, created=rig.clock(), state="ready", role="video", profile="cartoon_i2v")
    slot.url = f"http://{CARD_IP}:{CARD_PORT}"
    slot.accepted = True
    orc._slots[1] = slot

    now = rig.clock()
    orc._dispatch(slot, now)
    orc._dispatch_levels(now, False)

    assert [t.num for t in slot.tasks] == [1, 2, 3, 4]
    assert list(orc._video_queue) == [5]
    assert [c[0] for c in rig.worker.input_image_calls] == ["clip_1.png", "clip_2.png", "clip_3.png", "clip_4.png"]
    for n in range(1, 5):
        assert rig.worker.call_sequence.index(f"input_image:clip_{n}.png") < rig.worker.call_sequence.index(f"task:{n}:i2v")


def test_i2v_resume_between_phases_downloads_png_and_skips_phase1(tmp_path, monkeypatch):
    entries = {
        1: {"num": 1, "status": "ready", "prompt": "p1", "num_frames": 1,
            "i2v": True, "video_prompt": "vp1", "video_num_frames": 25, "video_status": "ready"},
        2: {"num": 2, "status": "ready", "prompt": "p2", "num_frames": 1, "i2v": False},
        3: {"num": 3, "status": "ready", "prompt": "p3", "num_frames": 1,
            "i2v": True, "video_prompt": "vp3", "video_num_frames": 25, "video_status": "ready"},
        4: {"num": 4, "status": "ready", "prompt": "p4", "num_frames": 1,
            "i2v": True, "video_prompt": "vp4", "video_num_frames": 25, "video_status": "ready"},
    }
    rig = _make_i2v_rig(
        tmp_path,
        monkeypatch,
        missing=(1, 2, 3, 4),
        entries=entries,
        preload=["1.png", "1.mp4", "2.png", "3.png"],
        preload_blobs={(PART1, "3.png"): b"CACHED-PNG-3"},
    )
    orch.go.atomic_write_json(rig.prompts_path, _make_prompts_state(entries))

    summary = rig.orc.run()

    assert summary.exit_code == 0
    assert summary.skipped_done == 2  # клипы 1 (есть 1.mp4) и 2 (статичный 2.png) пропущены полностью
    assert (summary.total_needed, summary.completed, summary.failed) == (2, 2, 0)
    assert (PART1, "3.png") in rig.store.downloads

    photo_nums = [c["num"] for c in rig.worker.task_calls if "input_image" not in c]
    i2v_nums = [c["num"] for c in rig.worker.task_calls if "input_image" in c]
    assert photo_nums == [4]  # клип 3 пропустил Phase 1!
    assert sorted(i2v_nums) == [3, 4]
    assert rig.worker.input_images["clip_3.png"] == b"CACHED-PNG-3"

    links = rig.store.blobs[(FINAL, "generated_links.txt")].decode("utf-8").splitlines()
    assert [line.rsplit("/", 1)[-1] for line in links] == ["1.mp4", "2.png", "3.mp4", "4.mp4"]


def test_i2v_fallback_photo_when_i2v_fails_after_retries(tmp_path, monkeypatch):
    entries = {
        1: {"num": 1, "status": "ready", "prompt": "p1", "num_frames": 1,
            "i2v": True, "video_prompt": "vp1", "video_num_frames": 25, "video_status": "ready"},
        2: {"num": 2, "status": "ready", "prompt": "p2", "num_frames": 1,
            "i2v": True, "video_prompt": "vp2", "video_num_frames": 25, "video_status": "ready"},
    }
    rig = _make_i2v_rig(tmp_path, monkeypatch, missing=(1, 2), entries=entries)
    rig.worker.video_plan[2] = ["error"] * orch.MAX_ATTEMPTS

    summary = rig.orc.run()

    assert summary.exit_code == 0
    assert (summary.completed, summary.failed, summary.pending) == (2, 0, 0)
    assert summary.fallback_photos == [2]

    i2v_calls_for_2 = [c for c in rig.worker.task_calls if c.get("input_image") == "clip_2.png"]
    assert len(i2v_calls_for_2) == orch.MAX_ATTEMPTS

    saved = json.loads(rig.prompts_path.read_text(encoding="utf-8"))
    assert saved["prompts"]["2"]["video_render_status"] == "fallback_photo"
    assert saved["prompts"]["1"]["video_render_status"] == "done"

    links = rig.store.blobs[(FINAL, "generated_links.txt")].decode("utf-8").splitlines()
    assert [line.rsplit("/", 1)[-1] for line in links] == ["1.mp4", "2.png"]

    report = rig.store.blobs[(FINAL, "summary.md")].decode("utf-8")
    assert "fallback" in report.lower() and "2" in report
