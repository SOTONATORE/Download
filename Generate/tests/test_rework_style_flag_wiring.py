"""Проводка ключа rework_style_flag_without_refs из profile.yaml в PromptConfig через cli._build_prompt_config.

Без сети и без подмены профилей: читаются настоящие profile.yaml. Ключа нет у ltx25 и zimage_turbo — там False.
"""
from Generate import cli  # noqa: F401


def test_cartoon_photos_flag_is_true():
    cfg = cli._build_prompt_config("cartoon_photos", 20, "none")
    assert cfg.rework_style_flag_without_refs is True


def test_ltx25_flag_is_false():
    cfg = cli._build_prompt_config("ltx25", 20, "none")
    assert cfg.rework_style_flag_without_refs is False


def test_zimage_turbo_flag_is_false():
    cfg = cli._build_prompt_config("zimage_turbo", 20, "none")
    assert cfg.rework_style_flag_without_refs is False
