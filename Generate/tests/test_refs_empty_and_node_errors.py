"""Офлайн-тесты: поле refs из extra, явный refs=[] для профиля с референсами, латент 1920x1088, node_errors.

Сети нет: ComfyUI подменяется объектом с методом request; граф берётся из реальных файлов ниши
cartoon_photos (только чтение), профили ltx25 и zimage_turbo в тестах не меняются.
Сообщения на русском (SPEC 0.5). Тексты промптов и секреты в тестах не используются.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

try:
    from Generate.worker import comfy_client as cc
except ImportError:  # pragma: no cover
    from worker import comfy_client as cc  # type: ignore


def _orch():
    """Оркестратор импортируется лениво: тесты клиента ComfyUI не требуют его зависимостей."""
    try:
        from Generate.core import orchestrator as orch
    except ImportError:  # pragma: no cover
        from core import orchestrator as orch  # type: ignore
    return orch

_REPO = Path(__file__).resolve().parents[2]
_NICHE = _REPO / "Generate" / "Nishes" / "cartoon_photos"


def _real_graph() -> dict:
    return json.loads((_NICHE / "workflow.json").read_text(encoding="utf-8"))


def _real_bindings() -> dict:
    return json.loads((_NICHE / "bindings.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _reset_profile_type():
    yield
    cc.set_profile_type(None)


def _inject(refs, image=True):
    cc.set_profile_type("image" if image else "video")
    return cc.inject_workflow(_real_graph(), _real_bindings(), "промпт", 1, 7, "clip", refs=refs)


# ---------------------------------------------------------------- 1.1 _entry_refs

def test_entry_refs_top_level_wins():
    assert _orch()._entry_refs({"refs": [2], "extra": {"refs": [5]}}) == [2]


def test_entry_refs_falls_back_to_extra():
    assert _orch()._entry_refs({"extra": {"refs": [2]}}) == [2]


def test_entry_refs_missing_returns_none():
    orch = _orch()
    assert orch._entry_refs({}) is None
    assert orch._entry_refs({"extra": {}}) is None
    assert orch._entry_refs({"extra": "строка"}) is None
    assert orch._entry_refs(None) is None


def test_entry_refs_empty_list_is_kept():
    # пустой список — валидное значение (0 референсов), а не отсутствие поля
    orch = _orch()
    assert orch._entry_refs({"extra": {"refs": []}}) == []
    assert orch._valid_refs([])


# ---------------------------------------------------------------- 1.2 / 1.3: латент для refs=()

def test_zero_refs_strips_all_loadimage_nodes():
    out = _inject(())
    loads = [k for k, n in out.items() if n.get("class_type") == "LoadImage"]
    assert loads == []
    prompt_inputs = out["491"]["inputs"]
    assert not [k for k in prompt_inputs if k.startswith("images.image_")]


def test_zero_refs_replaces_ksampler_latent_with_1920x1088():
    out = _inject(())
    link = out["488"]["inputs"]["latent_image"]
    assert link[0] != "491"  # больше не латент узла промпта
    node = out[link[0]]
    assert node["class_type"] == "EmptyLatentImage"
    assert node["inputs"] == {"width": 1920, "height": 1088, "batch_size": 1}


def test_zero_refs_new_node_id_does_not_collide():
    out = _inject(())
    original = set(_real_graph())
    new_ids = set(out) - original
    assert len(new_ids) == 1


def test_zero_refs_keeps_future_ref_slots_in_bindings():
    # refs=[] не удаляет привязки ref_7..10 (они нужны задачам с этими номерами) и не меняет словарь привязок
    bindings = _real_bindings()
    snapshot = copy.deepcopy(bindings)
    cc.set_profile_type("image")
    cc.inject_workflow(_real_graph(), bindings, "промпт", 1, 7, "clip", refs=())
    assert bindings == snapshot
    assert all(f"ref_{n}" in bindings for n in range(1, 11))
    assert bindings["ref_7"]["node_id"] == "507"


def test_refs_none_leaves_latent_link_unchanged():
    out = _inject(None)
    assert out["488"]["inputs"]["latent_image"] == ["491", 2]


def test_refs_one_leaves_latent_link_unchanged():
    out = _inject((1,))
    assert out["488"]["inputs"]["latent_image"] == ["491", 2]


def test_zero_refs_video_profile_does_not_replace_latent():
    # профиль видео (ltx25 и прочие): латент не трогаем, даже при refs=()
    # у видео обязательна привязка num_frames; в image-нише её нет, поэтому в тесте она добавляется
    # на безвредное поле узла 488 (значение не проверяется, важен только латент)
    cc.set_profile_type("video")
    bindings = _real_bindings()
    bindings["num_frames"] = {"node_id": "488", "field": "denoise"}
    out = cc.inject_workflow(_real_graph(), bindings, "промпт", 1, 7, "clip", refs=())
    assert out["488"]["inputs"]["latent_image"] == ["491", 2]


def test_zero_refs_non_qwen_latent_source_untouched():
    graph = _real_graph()
    graph["488"]["inputs"]["latent_image"] = ["5", 0]
    graph["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}}
    cc.set_profile_type("image")
    out = cc.inject_workflow(graph, _real_bindings(), "p", 1, 7, "clip", refs=())
    assert out["488"]["inputs"]["latent_image"] == ["5", 0]


def test_target_size_rounds_height_up_to_32():
    assert cc._target_size(_real_graph()) == (1920, 1088)


def test_target_size_without_imagescale_is_error():
    graph = {"1": {"class_type": "KSampler", "inputs": {}}}
    with pytest.raises(cc.ComfyError):
        cc._target_size(graph)


# ---------------------------------------------------------------- 1.3: node_errors при 400

class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("не JSON")
        return self._body


class _FakeHttp:
    def __init__(self, resp):
        self._resp = resp

    def request(self, method, url, **kwargs):
        return self._resp


def _client(status, body):
    return cc.ComfyClient(client=_FakeHttp(_Resp(status, body)))


def test_400_node_errors_are_in_exception():
    body = {
        "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
        "node_errors": {
            "507": {"class_type": "LoadImage", "errors": [
                {"type": "value_not_in_list", "message": "Value not in list: image: 'ref_7.png'",
                 "details": "СЕКРЕТНЫЙ_ТЕКСТ_ПРОМПТА"}]},
        },
    }
    with pytest.raises(cc.ComfyExecutionError) as info:
        _client(400, body).queue_prompt({})
    text = str(info.value)
    assert "код 400" in text
    assert "узел 507 (LoadImage)" in text
    assert "Value not in list" in text
    assert "СЕКРЕТНЫЙ_ТЕКСТ_ПРОМПТА" not in text  # details не выводятся


def test_400_without_node_errors_keeps_old_message():
    with pytest.raises(cc.ComfyExecutionError) as info:
        _client(400, {"error": {"message": "плохо"}}).queue_prompt({})
    assert str(info.value) == "ComfyUI отклонил задачу (код 400): плохо"


def test_400_non_json_body_does_not_crash():
    with pytest.raises(cc.ComfyExecutionError) as info:
        _client(400, "не JSON").queue_prompt({})
    assert str(info.value) == "ComfyUI отклонил задачу (код 400)."


def test_node_error_text_is_capped():
    long = "x" * 1000
    body = {"node_errors": {"1": {"class_type": "A", "errors": [{"message": long}]}}}
    text = cc._node_errors_text(body)
    assert len(text) <= len("узел 1 (A): ") + cc.NODE_ERROR_TEXT_LIMIT


def test_success_200_still_returns_prompt_id():
    assert _client(200, {"prompt_id": "abc"}).queue_prompt({}) == "abc"


# ---------------------------------------------------------------- 1.2: payload POST /task (диспатч)

class _FakeResp:
    def json(self):
        return {"id": "task-1"}


class _DispatchSelf:
    """Минимальный объект для настоящих методов _dispatch: очереди, промпты, превью и запись POST /task."""

    def __init__(self, profile, prompts, role="video", photo_profile=None, previews=None):
        from collections import deque
        self.profile = profile
        self.photo_profile = photo_profile
        self._prompts = prompts
        self._previews = previews or {}
        self._done = set()
        self._failed = set()
        self._attempts = {}
        self.seed_base = 1000
        self._queues = {"video": deque(), "photo": deque()}
        self._queues[role].append(5)
        self.sent = []

    def _queue_for_role(self, role):
        return self._queues[role]

    def _assigned_slot(self, num):
        return None

    def _call(self, slot, method, path, payload=None):
        self.sent.append(payload)
        return _FakeResp()

    def _fail_clip(self, num, reason):  # pragma: no cover - в тестах не вызывается
        raise AssertionError(f"неожиданный отказ клипа: {reason}")


def _dispatch(profile, entry, role="video", photo_profile=None, previews=None):
    orch = _orch()
    slot = orch._CardSlot(index=1, generation=1, instance_id=1, machine_id=1, price=0.0, created=0.0, role=role)
    fake = _DispatchSelf(profile, {"5": entry}, role=role, photo_profile=photo_profile, previews=previews)
    fake._profile_for_role = lambda r: orch.Orchestrator._profile_for_role(fake, r)
    fake._refs_enabled_for_role = lambda r: orch.Orchestrator._refs_enabled_for_role(fake, r)
    fake._clip_label = lambda n: f"клип {n}"
    fake._json = orch.Orchestrator._json  # статический метод
    orch.Orchestrator._dispatch(fake, slot, 0.0)
    assert len(fake.sent) == 1
    return fake.sent[0]


_ENTRY = {"prompt": "текст", "num_frames": 1}


def test_dispatch_generated_refs_from_extra_are_sent():
    payload = _dispatch("cartoon_photos", {**_ENTRY, "extra": {"refs": [2]}})
    assert payload["refs"] == [2]


def test_dispatch_top_level_refs_still_sent():
    payload = _dispatch("cartoon_photos", {**_ENTRY, "refs": [1, 6]})
    assert payload["refs"] == [1, 6]


def test_dispatch_no_refs_on_refs_profile_sends_empty_list():
    payload = _dispatch("cartoon_photos", dict(_ENTRY))
    assert payload["refs"] == []


def test_dispatch_no_refs_on_ltx25_omits_field():
    payload = _dispatch("ltx25", dict(_ENTRY))
    assert "refs" not in payload


def test_dispatch_no_refs_on_zimage_omits_field():
    payload = _dispatch("zimage_turbo", dict(_ENTRY))
    assert "refs" not in payload


def test_dispatch_preview_without_refs_on_refs_profile_sends_empty_list():
    from types import SimpleNamespace
    preview = SimpleNamespace(x=1, y=1, prompt="p", seed=3, refs=None, card="", status="", filename="preview_1_1.png")
    payload = _dispatch("cartoon_photos", dict(_ENTRY), role="photo", photo_profile="cartoon_photos",
                        previews={5: preview})
    assert payload["refs"] == []


def test_profile_refs_enabled_reads_profile_yaml_generically():
    from Generate.core import config_loader as cl
    assert cl.profile_refs_enabled("cartoon_photos") is True
    assert cl.profile_refs_enabled("ltx25") is False
    assert cl.profile_refs_enabled("zimage_turbo") is False
    assert cl.profile_refs_enabled("") is False
    assert cl.profile_refs_enabled("нет_такого_профиля") is False
