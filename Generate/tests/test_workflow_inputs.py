"""Лимит GitHub Actions: не больше 25 входов workflow_dispatch в каждом workflow-файле.

GitHub отклоняет workflow, у которого входов ручного запуска больше 25. Тест читает все
.yml и .yaml из .github/workflows через PyYAML (уже указан в requirements-dev.txt), без сети
и без запуска workflow. Дополнительно фиксирует, что из generate.yml удалены входы scan_sec
и verified: их значения берутся из Generate/config.default.yaml.
"""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
GENERATE_WORKFLOW = WORKFLOWS_DIR / "generate.yml"
MAX_DISPATCH_INPUTS = 25  # лимит GitHub Actions на число входов workflow_dispatch
REMOVED_INPUTS = ("scan_sec", "verified")


def _workflow_files():
    return sorted(
        p for p in WORKFLOWS_DIR.iterdir()
        if p.is_file() and p.suffix in {".yml", ".yaml"}
    )


def _dispatch_inputs(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # PyYAML читает ключ on как True (YAML 1.1), поэтому проверяем оба варианта
    on = data["on"] if "on" in data else data.get(True)
    if not isinstance(on, dict):
        return {}
    dispatch = on.get("workflow_dispatch") or {}
    return dispatch.get("inputs") or {}


def test_workflow_files_are_found():
    # защита от пустого списка: тест не должен проходить молча при неверном пути
    assert _workflow_files(), f"в {WORKFLOWS_DIR} нет файлов .yml или .yaml"


def test_every_workflow_has_at_most_25_dispatch_inputs():
    for path in _workflow_files():
        count = len(_dispatch_inputs(path))
        assert count <= MAX_DISPATCH_INPUTS, (
            f"{path.name}: {count} входов workflow_dispatch, "
            f"лимит GitHub {MAX_DISPATCH_INPUTS}"
        )


def test_generate_workflow_has_no_scan_sec_or_verified_input():
    # значения этих входов берутся из config.default.yaml; возврат входов не должен пройти незамеченным
    inputs = _dispatch_inputs(GENERATE_WORKFLOW)
    for name in REMOVED_INPUTS:
        assert name not in inputs, (
            f"{GENERATE_WORKFLOW.name}: вход {name} удалён, "
            "значение берётся из config.default.yaml"
        )
