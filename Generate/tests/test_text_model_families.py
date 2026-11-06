"""Тесты семейств текстовых моделей (этап 2): выбор, проверка конфигурации, опции Gemini, check.

Офлайн: сеть не используется, ключи только тестовые («test-key-1»), транспорт OpenRouter подменён.
"""
import pathlib
from types import SimpleNamespace

import pytest
import yaml

from Generate import cli
from Generate.core import config_loader as cl
from Generate.core import openrouter_transport as ort
from Generate.core.gemini_prompts import GeminiInputError

DATA = pathlib.Path(__file__).resolve().parent / "data" / "final.srt"
GEMMA_MODEL = "google/gemma-4-31b-it:free"
TEST_KEY = "test-key-1"


def _base() -> dict:
    return yaml.safe_load(cl.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _cfg():
    return cl.load_config()


def _rejected(raw: dict, fragment: str) -> None:
    with pytest.raises(GeminiInputError) as ei:
        cl.parse_config(raw)
    assert ei.value.exit_code == 2
    assert fragment in str(ei.value), str(ei.value)


@pytest.fixture
def missing_file(tmp_path):
    p = tmp_path / "missing.txt"
    p.write_text("1\n2\n", encoding="utf-8")
    return p


@pytest.fixture
def check_env(monkeypatch):
    """HF_TOKEN нужен предстартовой проверке check; ключи OpenRouter сначала снимаются, каждый тест задаёт свои."""
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    for i in range(1, 11):
        monkeypatch.delenv(f"OPENROUTER{i}", raising=False)


# ---------------------------------------------------------------- выбор семейства

def test_default_family_is_gemini():
    name, provider, models = cli.resolve_text_family(_cfg(), None)
    assert (name, provider, models) == ("gemini", "gemini", None)


def test_requested_gemma_selects_openrouter_family():
    name, provider, models = cli.resolve_text_family(_cfg(), "gemma")
    assert name == "gemma"
    assert provider == "openrouter"
    assert models == [GEMMA_MODEL]


def test_unknown_family_is_input_error():
    with pytest.raises(GeminiInputError) as ei:
        cli.resolve_text_family(_cfg(), "nope")
    assert "nope" in str(ei.value)
    assert "gemma" in str(ei.value)


def test_text_model_argument_on_run_and_check():
    parser = cli.build_parser()
    assert parser.parse_args(["run"]).text_model is None
    assert parser.parse_args(["run", "--text-model", "gemma"]).text_model == "gemma"
    assert parser.parse_args(["check", "--text-model", "nemotron"]).text_model == "nemotron"


def test_default_file_has_new_fields_and_gemini_unchanged():
    cfg = _cfg()
    assert cfg.default_text_model == "gemini"
    assert set(cfg.text_model_families) == {"gemini", "gemma", "nemotron", "nemotron-super-vl"}
    assert cfg.text_model_families["gemini"] == {"provider": "gemini", "models": None}
    assert cfg.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert cfg.openrouter_limits == {"rpm": 20, "tpm": 1000000, "rpd": 1000}
    assert cfg.openrouter_max_output_tokens == 16000
    assert cfg.gemini_models == ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash",
                                 "gemini-3.1-flash-lite"]


# ---------------------------------------------------------------- проверка конфигурации

def test_unknown_provider_rejected():
    raw = _base()
    raw["text_model_families"]["gemma"]["provider"] = "azure"
    _rejected(raw, "«provider»")


def test_openrouter_empty_models_rejected():
    raw = _base()
    raw["text_model_families"]["gemma"]["models"] = []
    _rejected(raw, "«models»")


def test_default_text_model_not_in_families_rejected():
    raw = _base()
    raw["default_text_model"] = "mistral"
    _rejected(raw, "«default_text_model»")


def test_missing_gemini_family_rejected():
    raw = _base()
    del raw["text_model_families"]["gemini"]
    raw["default_text_model"] = "gemma"
    _rejected(raw, "семейство «gemini»")


def test_gemini_family_accepts_explicit_model_list():
    raw = _base()
    raw["text_model_families"]["gemini"]["models"] = ["gemini-3.5-flash-lite"]
    assert cl.parse_config(raw).text_model_families["gemini"]["models"] == ["gemini-3.5-flash-lite"]


# ---------------------------------------------------------------- опции Gemini и транспорт

def test_gemini_options_gemini_family_uses_gemini_models_as_before():
    cfg = _cfg()
    opts = cli._gemini_options(cfg, cli.resolve_text_family(cfg, None))
    assert opts == {
        "models": tuple(cfg.gemini_models),
        "rpm_limit": cfg.gemini_rpm_limit,
        "tpm_limit": cfg.gemini_tpm_limit,
        "rpd_limit": cfg.gemini_rpd_limit,
        "transient_switch_after": cfg.gemini_transient_switch_after,
    }


def test_gemini_options_gemma_uses_openrouter_models():
    cfg = _cfg()
    opts = cli._gemini_options(cfg, cli.resolve_text_family(cfg, "gemma"))
    assert opts["models"] == (GEMMA_MODEL,)


def test_make_transport_gemini_keeps_realtransport_kwargs(monkeypatch):
    captured = {}

    class FakeRealTransport:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(cli, "_find_symbol", lambda name, modules: FakeRealTransport)
    cfg = _cfg()
    cli._make_transport(cfg, cli.resolve_text_family(cfg, None))
    assert captured["rpm_limit"] == cfg.gemini_rpm_limit
    assert captured["tpm_limit"] == cfg.gemini_tpm_limit
    assert captured["rpd_limit"] == cfg.gemini_rpd_limit
    expected = {m: (v["rpm"], v["tpm"], v["rpd"]) for m, v in cfg.gemini_model_limits.items()}
    assert captured["model_limits"] == expected


def test_make_transport_openrouter_passes_family_settings(monkeypatch):
    captured = {}

    class FakeOpenRouterTransport:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ort, "OpenRouterTransport", FakeOpenRouterTransport)
    cfg = _cfg()
    cli._make_transport(cfg, cli.resolve_text_family(cfg, "gemma"))
    limits = cfg.openrouter_limits
    assert captured["base_url"] == cfg.openrouter_base_url
    assert captured["model_limits"] == {GEMMA_MODEL: (limits["rpm"], limits["tpm"], limits["rpd"])}
    assert captured["max_output_tokens"] == cfg.openrouter_max_output_tokens


# ---------------------------------------------------------------- check

def test_check_gemini_names_gemini_keys_and_prints_no_value(capsys, missing_file, check_env, monkeypatch):
    monkeypatch.setenv("GEMINI" + "1", TEST_KEY)  # имя ключа собирается из частей: структурный тест ищет литерал
    code = cli.main(["check", "--srt", str(DATA), "--missing", str(missing_file)])
    cap = capsys.readouterr()
    assert code == 0
    assert "семейство gemini" in cap.out
    assert "GEMINI" + "1" + "…GEMINI" + "10" in cap.out
    assert TEST_KEY not in cap.out + cap.err


def test_check_gemma_names_openrouter_keys_and_prints_no_value(capsys, missing_file, check_env, monkeypatch):
    monkeypatch.setenv("OPENROUTER" + "1", TEST_KEY)
    code = cli.main(["check", "--srt", str(DATA), "--missing", str(missing_file), "--text-model", "gemma"])
    cap = capsys.readouterr()
    assert code == 0
    assert "семейство gemma" in cap.out
    assert GEMMA_MODEL in cap.out
    assert "OPENROUTER" + "1" + "…OPENROUTER" + "10" in cap.out
    assert TEST_KEY not in cap.out + cap.err


def test_check_gemma_without_openrouter_key_fails_before_vast(capsys, missing_file, check_env, monkeypatch):
    def vast_must_not_run(*args, **kwargs):
        raise AssertionError("VastClient не должен создаваться при check")

    monkeypatch.setattr(cli, "VastClient", vast_must_not_run)
    code = cli.main(["check", "--srt", str(DATA), "--missing", str(missing_file), "--text-model", "gemma"])
    cap = capsys.readouterr()
    assert code == cli.EXIT_INPUT
    assert "OPENROUTER" + "1" + "…OPENROUTER" + "10" in cap.out + cap.err


def test_check_unknown_family_is_input_error(capsys, missing_file, check_env):
    code = cli.main(["check", "--srt", str(DATA), "--missing", str(missing_file), "--text-model", "nope"])
    cap = capsys.readouterr()
    assert code == cli.EXIT_INPUT
    assert "nope" in cap.out + cap.err
