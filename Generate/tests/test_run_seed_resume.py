"""Блок 6B: run_seed прогона (seed_mode: run) и восстановление prompts.json из релиза при перезапуске.

Тесты написаны, не запускались (прогоняет пользователь на GitHub). Сеть не используется: релиз подменён фейками,
объект Orchestrator создаётся без конструктора (как в соседних тестах), профили run/per_clip — фейковые имена.
Профили ltx25 и zimage_turbo читаются из реальных profile.yaml (локальный файл, без сети).
"""
import json
import logging
from collections import deque
from pathlib import Path

import pytest

from Generate.core import gemini_orchestrator as go
from Generate.core import orchestrator as orch
from Generate.core import previews
from Generate.core.gemini_prompts import GeminiInputError
from Generate.core.release_adapter import GhReleaseStore, ReleaseUploader
from Generate.core.srt_parser import Segment

RUN_PROFILE = "fake_run"      # фейковый профиль с seed_mode: run
CLIP_PROFILE = "fake_clip"    # фейковый профиль без seed_mode (per_clip)


def _segs():
    return [Segment(1, 0, 500, "первый"), Segment(2, 1000, 1500, "второй"), Segment(3, 2000, 2500, "третий")]


@pytest.fixture
def seed_modes(monkeypatch):
    # подмена чтения seed_mode: только фейковые имена, файлы профилей не открываются
    modes = {RUN_PROFILE: "run", CLIP_PROFILE: "per_clip"}
    monkeypatch.setattr(orch, "load_profile_seed_mode", lambda name: modes.get(name, "per_clip"))
    return modes


def _orc(folder, profile, *, photo_profile=None, segments=None, seed_base=0):
    # объект без __init__: только поля, которые читают выбор seed, восстановление и отправку клипа
    segments = segments or _segs()
    folder.mkdir(parents=True, exist_ok=True)
    o = orch.Orchestrator.__new__(orch.Orchestrator)
    o.segments = segments
    o.profile = profile
    o.photo_profile = photo_profile
    o.seed_base = seed_base
    o.prompts_path = folder / "prompts.json"
    o._state = go.load_state(o.prompts_path, segments, profile)
    o._prompts = o._state.setdefault("prompts", {})
    o._run_seed = None
    o._previews = {}
    o._done = set()
    o._slots = {}
    o._attempts = {}
    o._video_queue = deque()
    o._photo_queue = None
    o._photo_nums = frozenset()
    o._refs_enabled_for_role = lambda role: False  # референсы в этих тестах не нужны
    return o


def _fill_prompts(o, nums):
    for n in nums:
        o._prompts[str(n)] = {"prompt": f"промпт {n}", "num_frames": 5, "status": "ready"}


def _dispatch_each(o, role, nums):
    """Отправляет клипы по одному. Воркер отвечает 429: клип возвращается в очередь без попытки, payload сохраняется."""
    sent = []

    def fake_call(slot, method, path, payload):
        sent.append(payload)
        raise orch._WorkerHttpError(429)

    o._call = fake_call
    slot = orch._CardSlot(index=1, generation=1, instance_id=1, machine_id=1, price=0.1, created=0.0, role=role)
    for num in nums:
        queue = o._queue_for_role(role)
        queue.clear()
        queue.append(num)
        o._dispatch(slot, 0.0)
    return sent


class FakeUploader:
    """Подмена ReleaseUploader: download_service_file отдаёт байты из словаря, как служебный файл релиза."""

    def __init__(self, segments, assets=None):
        self.full_hash = go.srt_hash(segments)
        self.final_tag = "final-gen-test"
        self.assets = dict(assets or {})
        self.calls = []

    def download_service_file(self, filename, dest):
        self.calls.append(filename)
        if filename not in self.assets:
            return False
        Path(dest).write_bytes(self.assets[filename])
        return True


def _remote_state(segments, profile, *, run_seed=777, srt=None):
    state = {
        "srt_hash": srt or go.srt_hash(segments),
        "model": "",
        "profile": profile,
        "prompts": {"1": {"prompt": "из релиза", "num_frames": 5, "status": "ready"}},
    }
    if run_seed is not None:
        state["run_seed"] = run_seed
    return json.dumps(state, ensure_ascii=False).encode("utf-8")


def _no_new_seed(*args, **kwargs):
    raise AssertionError("новый run_seed не должен создаваться при перезапуске")


# --- seed прогона ---------------------------------------------------------------------------------

def test_run_mode_generates_one_run_seed_used_by_every_clip(tmp_path, seed_modes):
    o = _orc(tmp_path, RUN_PROFILE)
    _fill_prompts(o, [1, 2, 3])
    o._ensure_run_seed()
    first = o._run_seed
    o._ensure_run_seed()  # повторный вызов в том же прогоне не создаёт второй seed
    assert o._run_seed == first
    assert isinstance(first, int) and first >= 1
    sent = _dispatch_each(o, "video", [1, 2, 3])
    assert [p["seed"] for p in sent] == [first, first, first]


def test_run_seed_written_to_prompts_json_top_level_and_state(tmp_path, seed_modes):
    o = _orc(tmp_path, RUN_PROFILE)
    o._ensure_run_seed()
    on_disk = json.loads(o.prompts_path.read_text(encoding="utf-8"))
    assert on_disk["run_seed"] == o._run_seed == o._state["run_seed"]
    assert on_disk["srt_hash"] == go.srt_hash(o.segments)


def test_restart_with_release_prompts_reuses_same_run_seed(tmp_path, seed_modes, monkeypatch):
    # первый старт: seed создан и записан в prompts.json
    first = _orc(tmp_path / "first", RUN_PROFILE)
    first._ensure_run_seed()
    seed = first._run_seed
    uploaded = first.prompts_path.read_bytes()
    # перезапуск на чистом месте: локального prompts.json нет, файл приходит из релиза
    restart = _orc(tmp_path / "restart", RUN_PROFILE)
    uploader = FakeUploader(_segs(), {"prompts.json": uploaded})
    restart._restore_prompts_from_release(uploader)
    assert restart.prompts_path.exists()
    monkeypatch.setattr(orch.secrets, "randbelow", _no_new_seed)
    restart._state = go.load_state(restart.prompts_path, _segs(), RUN_PROFILE)
    restart._ensure_run_seed()
    assert restart._run_seed == seed


def test_release_prompts_for_other_profile_ignored_with_warning(tmp_path, seed_modes, caplog):
    segs = _segs()
    uploader = FakeUploader(segs, {"prompts.json": _remote_state(segs, "other_profile")})
    o = _orc(tmp_path, RUN_PROFILE, segments=segs)
    with caplog.at_level(logging.WARNING):
        o._restore_prompts_from_release(uploader)
    assert not o.prompts_path.exists()  # файл из релиза не используется
    assert any("другого профиля" in r.getMessage() for r in caplog.records)
    o._ensure_run_seed()  # генерация продолжается: seed создаётся локально
    assert isinstance(o._run_seed, int)


def test_broken_release_prompts_json_raises_input_error(tmp_path, seed_modes):
    segs = _segs()
    uploader = FakeUploader(segs, {"prompts.json": b"{not json"})
    o = _orc(tmp_path, RUN_PROFILE, segments=segs)
    with pytest.raises(GeminiInputError):
        o._restore_prompts_from_release(uploader)
    assert not o.prompts_path.exists()


def test_release_prompts_with_other_srt_hash_raises_input_error(tmp_path, seed_modes):
    segs = _segs()
    uploader = FakeUploader(segs, {"prompts.json": _remote_state(segs, RUN_PROFILE, srt="f" * 64)})
    o = _orc(tmp_path, RUN_PROFILE, segments=segs)
    with pytest.raises(GeminiInputError):
        o._restore_prompts_from_release(uploader)
    assert not o.prompts_path.exists()


def test_gh_download_asset_fake_runner_restores_service_file(tmp_path, seed_modes):
    # реальные GhReleaseStore.download_asset и ReleaseUploader.download_service_file; gh заменён фейковым запуском
    segs = _segs()
    payload = _remote_state(segs, RUN_PROFILE, run_seed=777)

    def fake_runner(args):
        assert args[:2] == ["release", "download"]
        Path(args[args.index("--output") + 1]).write_bytes(payload)
        return ""

    store = GhReleaseStore("owner/repo", runner=fake_runner)
    uploader = ReleaseUploader(store, segs, "model", log=lambda msg: None, repo="owner/repo")
    uploader.final_assets = ["prompts.json"]
    o = _orc(tmp_path, RUN_PROFILE, segments=segs)
    o._restore_prompts_from_release(uploader)
    assert json.loads(o.prompts_path.read_text(encoding="utf-8"))["run_seed"] == 777


# --- профили per_clip: без run_seed, без восстановления из релиза, seed = seed_base + num ----------

def test_ltx25_and_zimage_turbo_use_per_clip_seed_without_run_seed(tmp_path):
    segs = _segs()
    uploader = FakeUploader(segs, {"prompts.json": _remote_state(segs, "ltx25", run_seed=777)})
    o = _orc(tmp_path, "ltx25", photo_profile="zimage_turbo", segments=segs, seed_base=1000)
    o._restore_prompts_from_release(uploader)
    assert uploader.calls == []  # восстановления из релиза нет
    assert not o.prompts_path.exists()
    o._ensure_run_seed()
    assert o._run_seed is None
    assert "run_seed" not in o._state
    _fill_prompts(o, [1, 2])
    o._video_queue.append(1)
    video = _dispatch_each(o, "video", [1])
    assert video[0]["seed"] == 1000 + 1
    o._photo_queue = deque()
    o._photo_nums = frozenset({2})
    o._prompts["2"] = {"prompt": "фото", "num_frames": 1, "status": "ready"}
    photo = _dispatch_each(o, "photo", [2])
    assert photo[0]["seed"] == 1000 + 2


# --- превью: всегда preview.seed ---------------------------------------------------------------

@pytest.mark.parametrize("profile", [RUN_PROFILE, CLIP_PROFILE])
def test_previews_always_use_preview_seed(tmp_path, seed_modes, profile):
    o = _orc(tmp_path, profile)
    o._run_seed = 999  # для профиля run это значение не должно попасть в превью
    num = previews.PREVIEW_NUM_BASE + 1
    o._previews = {num: previews.PreviewItem(num=num, x=1, y=1, prompt="кадр", seed=123456,
                                             filename="preview_1_1.png")}
    o._photo_queue = deque()
    sent = _dispatch_each(o, "photo", [num])
    assert sent[0]["seed"] == 123456
