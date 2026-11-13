"""Офлайн-тесты референсов на задачу: проверка поля refs, переподключение workflow, группировка по стекам.

Сети нет: клиент ComfyUI не вызывается, проверяются только чистые функции.
"""
from __future__ import annotations

import copy

import pytest

try:
    from Generate.worker import comfy_client as cc
    from Generate.worker import server as srv
    from Generate.core import orchestrator as orch
except ImportError:  # pragma: no cover
    from worker import comfy_client as cc  # type: ignore
    from worker import server as srv  # type: ignore
    from core import orchestrator as orch  # type: ignore


def _graph_10_slots() -> dict:
    """Граф с узлом промпта 491 (10 входов images.image_*) и узлами LoadImage ref_1…ref_10."""
    load_ids = {1: "470", 2: "475", 3: "503", 4: "504", 5: "505",
                6: "506", 7: "507", 8: "508", 9: "509", 10: "510"}
    graph = {
        "491": {"class_type": "TextEncodeQwenImage21",
                "inputs": {"prompt": "p", "clip": ["496", 0], "vae": ["497", 0]}},
        "488": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
        "5": {"class_type": "Latent", "inputs": {"length": 1}},
    }
    for n, node_id in load_ids.items():
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": f"ref_{n}.png"}}
        graph["491"]["inputs"][f"images.image_{n}"] = [node_id, 0]
    return graph


def _bindings_10() -> dict:
    """Привязки, как в Nishes/cartoon_photos/bindings.json (ref_N -> LoadImage)."""
    load_ids = {1: "470", 2: "475", 3: "503", 4: "504", 5: "505",
                6: "506", 7: "507", 8: "508", 9: "509", 10: "510"}
    bindings = {
        "prompt": {"node_id": "491", "field": "prompt"},
        "seed": {"node_id": "488", "field": "seed"},
        "filename_prefix": {"node_id": "461", "field": "filename_prefix"},
        "num_frames": {"node_id": "5", "field": "length"},
    }
    for n, node_id in load_ids.items():
        bindings[f"ref_{n}"] = {"node_id": node_id, "field": "image"}
    return bindings


def _base_graph() -> dict:
    graph = _graph_10_slots()
    graph["461"] = {"class_type": "SaveImageAdvanced", "inputs": {"filename_prefix": "x"}}
    return graph


def _payload(**extra) -> dict:
    data = {"num": 1, "prompt": "промпт", "num_frames": 97, "seed": 7}
    data.update(extra)
    return data


# --- проверка поля refs в _validate_payload ---

def test_refs_absent_gives_none():
    *_, refs = srv._validate_payload(_payload())
    assert refs is None


def test_refs_sorted_ascending():
    *_, refs = srv._validate_payload(_payload(refs=[5, 1, 3]))
    assert refs == (1, 3, 5)


@pytest.mark.parametrize("bad, text", [
    ("1,3", "списком"),
    (None, "списком"),
    ([1, "2"], "только целые"),
    ([True], "только целые"),
    ([0], "от 1 до 10"),
    ([11], "от 1 до 10"),
    ([2, 2], "не должны повторяться"),
])
def test_refs_bad_values_rejected_in_russian(bad, text):
    with pytest.raises(ValueError) as info:
        srv._validate_payload(_payload(refs=bad))
    assert text in str(info.value)


# --- переподключение workflow в inject_workflow ---

def test_no_refs_leaves_graph_unchanged():
    graph = _base_graph()
    before = copy.deepcopy(graph)
    out = cc.inject_workflow(graph, _bindings_10(), "p", 97, 7, "clip", refs=None)
    assert out["491"]["inputs"].keys() == before["491"]["inputs"].keys()
    assert all(f"images.image_{n}" in out["491"]["inputs"] for n in range(1, 11))


def test_refs_1_and_3_keeps_only_two_loadimage_nodes():
    out = cc.inject_workflow(_base_graph(), _bindings_10(), "p", 97, 7, "clip", refs=(1, 3))
    load_nodes = sorted(k for k, v in out.items() if v.get("class_type") == "LoadImage")
    # ref_1 -> 470, ref_3 -> 503 остаются; 475 (ref_2) и 504…510 (ref_4…ref_10) удалены
    assert load_nodes == ["470", "503"]


def test_refs_1_and_3_wires_image_1_and_2_only():
    out = cc.inject_workflow(_base_graph(), _bindings_10(), "p", 97, 7, "clip", refs=(1, 3))
    inputs = out["491"]["inputs"]
    image_keys = sorted(k for k in inputs if k.startswith("images.image_"))
    assert image_keys == ["images.image_1", "images.image_2"]
    assert inputs["images.image_1"] == ["470", 0]  # наименьший номер, ref_1
    assert inputs["images.image_2"] == ["503", 0]  # ref_3 идёт вторым, слот image_3 удалён


def test_refs_1_and_3_sets_loadimage_file_names():
    out = cc.inject_workflow(_base_graph(), _bindings_10(), "p", 97, 7, "clip", refs=(1, 3))
    assert out["470"]["inputs"]["image"] == "ref_1.png"
    assert out["503"]["inputs"]["image"] == "ref_3.png"


def test_other_slots_and_node_ids_unchanged():
    out = cc.inject_workflow(_base_graph(), _bindings_10(), "p", 97, 7, "clip", refs=(1, 3))
    assert out["491"]["inputs"]["clip"] == ["496", 0]
    assert out["491"]["inputs"]["vae"] == ["497", 0]
    assert out["491"]["inputs"]["prompt"] == "p"
    assert out["488"]["inputs"]["seed"] == 7


def test_refs_missing_binding_raises():
    bindings = _bindings_10()
    del bindings["ref_3"]
    with pytest.raises(cc.ComfyError):
        cc.inject_workflow(_base_graph(), bindings, "p", 97, 7, "clip", refs=(3,))


def test_ltx_style_bindings_without_ref_nodes_reject_refs():
    bindings = {"prompt": {"node_id": "491", "field": "prompt"},
                "seed": {"node_id": "488", "field": "seed"},
                "filename_prefix": {"node_id": "461", "field": "filename_prefix"},
                "num_frames": {"node_id": "5", "field": "length"}}
    with pytest.raises(cc.ComfyError):
        cc.inject_workflow(_base_graph(), bindings, "p", 97, 7, "clip", refs=(1,))


# --- группировка по стекам ---

def test_group_by_stack_keeps_order_when_no_refs():
    nums = [4, 1, 9, 2]
    assert orch.group_by_stack(nums, lambda n: ()) == nums


def test_group_by_stack_runs_stacks_consecutively_in_first_appearance_order():
    keys = {1: (1, 2), 2: (), 3: (1, 2), 4: (3,), 5: (), 6: (3,)}
    out = orch.group_by_stack([1, 2, 3, 4, 5, 6], keys.get)
    # стек (1,2) первым (встречен на 1), затем () (на 2), затем (3,)
    assert out == [1, 3, 2, 5, 4, 6]


def test_valid_refs_rules():
    assert orch._valid_refs([1, 10])
    assert not orch._valid_refs([0])
    assert not orch._valid_refs([11])
    assert not orch._valid_refs([1, 1])
    assert not orch._valid_refs([True])
    assert not orch._valid_refs("1,2")
