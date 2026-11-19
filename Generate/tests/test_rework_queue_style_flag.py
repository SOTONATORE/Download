"""Очередь исправлений для флага стиля «отрицание» и загрузчики булевых ключей профиля.

Без сети: профиль подменяется через monkeypatch; pytest в этом репозитории запускает модуль как обычно.
"""
import pytest

from Generate.core import config_loader
from Generate.core import gemini_prompts as gp
from Generate.core.config_loader import GeminiInputError
from Generate.core.gemini_prompts import PromptConfig, PromptResult

NEG = gp._NEGATION_REASON


def _cfg(**kw):
    return PromptConfig(style_file="style.md", media="image", **kw)


def _result(reason, needs_review=True, prompt="a scene"):
    return PromptResult(1, prompt, needs_review, reason)


def _profile(monkeypatch, data):
    monkeypatch.setattr(config_loader, "profile_exists", lambda name: True)
    monkeypatch.setattr(config_loader, "_read_profile", lambda name: dict(data))


# --- конфиг и очередь ---------------------------------------------------------

def test_default_flag_is_off():
    assert PromptConfig(style_file="s", media="image").rework_style_flag_without_refs is False


def test_style_only_flag_without_refs_is_queued_when_flag_on():
    assert gp._rework_gate_open(_cfg(rework_style_flag_without_refs=True), _result(NEG)) is True


def test_style_only_flag_without_refs_not_queued_when_flag_off():
    assert gp._rework_gate_open(_cfg(), _result(NEG)) is False


def test_other_reason_queued_without_refs_with_flag_on():
    # Русский: флаг rework_style_flag_without_refs ведёт в очередь любой флаг разбора, не только «отрицание»
    reason = NEG + "; стиль: кавычки"
    assert gp._rework_gate_open(_cfg(rework_style_flag_without_refs=True), _result(reason)) is True


def test_refs_enabled_path_unchanged():
    assert gp._rework_gate_open(_cfg(refs_enabled=True), _result("стиль: кавычки")) is True
    assert gp._rework_gate_open(_cfg(refs_enabled=True), _result(NEG)) is True


def test_fail_fast_and_no_review_never_queue():
    assert gp._rework_gate_open(_cfg(rework_style_flag_without_refs=True, fail_fast=True), _result(NEG)) is False
    assert gp._rework_gate_open(_cfg(rework_style_flag_without_refs=True), _result(NEG, needs_review=False)) is False
    assert gp._rework_gate_open(_cfg(rework_style_flag_without_refs=True), _result("")) is False


def test_only_negation_flag_helper():
    assert gp._only_negation_flag(NEG) is True
    assert gp._only_negation_flag(NEG + "; " + NEG) is True
    assert gp._only_negation_flag(NEG + "; стиль: кавычки") is False


# --- загрузчики ключей профиля ------------------------------------------------

@pytest.mark.parametrize("loader", [
    config_loader.load_profile_rework_style_flag_without_refs,
    config_loader.load_profile_fail_on_needs_review,
    config_loader.load_profile_fail_on_duplicate_prompts,
])
def test_loaders_default_false_when_key_absent(monkeypatch, loader):
    _profile(monkeypatch, {"refs_enabled": False})
    assert loader("any") is False


@pytest.mark.parametrize("loader", [
    config_loader.load_profile_rework_style_flag_without_refs,
    config_loader.load_profile_fail_on_needs_review,
    config_loader.load_profile_fail_on_duplicate_prompts,
])
def test_loaders_read_true_and_false(monkeypatch, loader):
    _profile(monkeypatch, {"rework_style_flag_without_refs": True, "fail_on_needs_review": True,
                           "fail_on_duplicate_prompts": True})
    assert loader("any") is True


@pytest.mark.parametrize("bad", ["true", "false", 1, 0, None, []])
def test_loaders_reject_non_bool(monkeypatch, bad):
    _profile(monkeypatch, {"fail_on_needs_review": bad})
    with pytest.raises(GeminiInputError):
        config_loader.load_profile_fail_on_needs_review("any")


def test_loader_without_profile_is_false(monkeypatch):
    monkeypatch.setattr(config_loader, "profile_exists", lambda name: False)
    assert config_loader.load_profile_fail_on_duplicate_prompts("none-profile") is False
    assert config_loader.load_profile_fail_on_duplicate_prompts("") is False
