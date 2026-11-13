"""Поиск папки режима по имени и выбор текстовой модели из profile.yaml.

Порядок папок: model_profiles/models/<имя>, затем Nishes/<имя>.
Приоритет текстовой модели: --text-model, затем text_model из profile.yaml, затем default_text_model.
Все пути подменяются на временные папки: настоящие профили и веса не читаются.
"""
from __future__ import annotations

import types

try:
    from Generate.core import config_loader as cl
    from Generate import cli
except ImportError:  # pragma: no cover
    from core import config_loader as cl  # type: ignore
    import cli  # type: ignore


def _cfg():
    # resolve_text_family читает только эти два поля конфига
    return types.SimpleNamespace(
        default_text_model="nemotron-super-vl",
        text_model_families={
            "gemini": {"provider": "gemini", "models": None},
            "gemma": {"provider": "openrouter", "models": ["gemma-x"]},
            "nemotron-super-vl": {"provider": "openrouter", "models": ["nemotron-x"]},
        },
    )


def test_profile_dir_prefers_models_then_nishes(tmp_path, monkeypatch):
    models = tmp_path / "model_profiles"
    nishes = tmp_path / "Nishes"
    (models / "models" / "both").mkdir(parents=True)
    (models / "models" / "only_models").mkdir(parents=True)
    (nishes / "both").mkdir(parents=True)
    (nishes / "only_nishes").mkdir(parents=True)
    monkeypatch.setattr(cl, "MODEL_PROFILES_DIR", models)
    monkeypatch.setattr(cl, "NISHES_DIR", nishes)

    # есть в обеих папках: побеждает model_profiles/models
    assert cl.profile_dir("both") == models / "models" / "both"
    # есть только в Nishes: берётся папка режима
    assert cl.profile_dir("only_nishes") == nishes / "only_nishes"
    # есть только в models
    assert cl.profile_dir("only_models") == models / "models" / "only_models"
    # нигде нет: путь в models, как раньше (сообщения об ошибке не меняются)
    assert cl.profile_dir("missing") == models / "models" / "missing"


def test_text_model_priority(tmp_path, monkeypatch):
    models = tmp_path / "model_profiles"
    nishes = tmp_path / "Nishes"
    (models / "models" / "withtext").mkdir(parents=True)
    (models / "models" / "withtext" / "profile.yaml").write_text(
        "type: image\ntext_model: gemma\n", encoding="utf-8")
    (models / "models" / "notext").mkdir(parents=True)
    (models / "models" / "notext" / "profile.yaml").write_text("type: video\n", encoding="utf-8")
    (models / "models" / "badtext").mkdir(parents=True)
    (models / "models" / "badtext" / "profile.yaml").write_text(
        "type: video\ntext_model: 123\n", encoding="utf-8")
    (models / "models" / "unknowntext").mkdir(parents=True)
    (models / "models" / "unknowntext" / "profile.yaml").write_text(
        "type: video\ntext_model: mistral\n", encoding="utf-8")
    monkeypatch.setattr(cl, "MODEL_PROFILES_DIR", models)
    monkeypatch.setattr(cl, "NISHES_DIR", nishes)

    cfg = _cfg()
    # 1. профиля нет: default_text_model
    assert cli.resolve_text_family(cfg, None)[0] == "nemotron-super-vl"
    assert cli.resolve_text_family(cfg, None, None)[0] == "nemotron-super-vl"
    # 2. профиль без ключа text_model: default_text_model
    assert cli.resolve_text_family(cfg, None, "notext")[0] == "nemotron-super-vl"
    # 3. профиль с text_model: берётся он (вместо default)
    assert cli.resolve_text_family(cfg, None, "withtext")[0] == "gemma"
    # 4. --text-model сильнее профиля
    assert cli.resolve_text_family(cfg, "gemini", "withtext")[0] == "gemini"
    # 5. профиль с text_model, которого нет в families: ошибка входа (код 2)
    try:
        cli.resolve_text_family(cfg, None, "unknowntext")
    except cli.GeminiInputError as e:
        assert "mistral" in str(e)
    else:
        raise AssertionError("ожидалась ошибка неизвестного семейства")
    # 6. не строка в text_model: ошибка входа
    try:
        cli.resolve_text_family(cfg, None, "badtext")
    except cli.GeminiInputError as e:
        assert "text_model" in str(e)
    else:
        raise AssertionError("ожидалась ошибка типа text_model")
    # 7. профиль из Nishes без profile.yaml: default_text_model, без ошибки
    (nishes / "cartoon").mkdir(parents=True)
    assert cli.resolve_text_family(cfg, None, "cartoon")[0] == "nemotron-super-vl"
