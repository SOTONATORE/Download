"""Проверка: имена из gemini_prompts доступны на уровне модуля cli, а не только внутри функции.

Тест написан для pytest, но в этом шаге не запускается.
"""
from Generate import cli


def test_cli_has_prompt_helpers_at_module_level():
    # REF_MAX — целое число; остальные имена — функции
    assert hasattr(cli, "REF_MAX")
    assert isinstance(cli.REF_MAX, int)
    for name in ("count_ref_files", "detect_srt_language", "profile_relaxation"):
        assert hasattr(cli, name), f"в модуле cli нет имени: {name}"
        assert callable(getattr(cli, name)), f"имя в модуле cli не вызывается: {name}"
