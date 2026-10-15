"""Тесты загрузчика конфигурации (SPEC 3): схема, границы, связи полей, входы, ошибки.

Офлайн: только файлы во временной папке. Ошибки всегда GeminiInputError с кодом выхода 2.
"""
import dataclasses

import pytest
import yaml

try:
    from Generate.core import config_loader as cl
    from Generate.core.gemini_prompts import GeminiInputError
except ImportError:  # плоский импорт
    from core import config_loader as cl  # type: ignore
    from core.gemini_prompts import GeminiInputError  # type: ignore


def _base() -> dict:
    return yaml.safe_load(cl.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _assert_rejected(raw: dict, *fragments: str) -> GeminiInputError:
    with pytest.raises(GeminiInputError) as ei:
        cl.parse_config(raw)
    assert ei.value.exit_code == 2
    for fragment in fragments:
        assert fragment in str(ei.value), str(ei.value)
    return ei.value


# ---------------------------------------------------------------- файл по умолчанию

def test_default_file_loads_and_matches_spec_values():
    cfg = cl.load_config()
    assert cfg.gemini_models == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    assert cfg.gemini_batch_size == 100
    assert cfg.disk_gb == 100
    assert cfg.silent_host_timeout_min == 20
    assert cfg.card_max_lifetime_min == 90
    assert cfg.worker_idle_timeout_min == 5
    assert cfg.job_deadline_min == 330
    assert cfg.gpu_max_sec is None          # берётся из профиля модели
    assert cl.load_profile_number(cfg.model_profile, "gpu_max_sec") == 10.0


def test_schema_keys_match_default_file_exactly():
    raw = _base()
    names = {f.name for f in dataclasses.fields(cl.Config)}
    assert names == set(raw), "схема и config.default.yaml должны совпадать по ключам"


# ---------------------------------------------------------------- ключи

def test_unknown_key_is_error_with_suggestion():
    raw = _base()
    raw["disk_gbb"] = 3
    _assert_rejected(raw, "неизвестный ключ «disk_gbb»", "возможно, «disk_gb»")


def test_missing_key_is_error():
    raw = _base()
    del raw["fps"]
    _assert_rejected(raw, "отсутствует обязательный ключ «fps»")


# ---------------------------------------------------------------- verified (производственный фильтр Vast)

def test_verified_default_file_is_false():
    """Значение в config.default.yaml: 0 (как вход workflow по умолчанию)."""
    assert cl.load_config().verified is False


def test_verified_zero_from_file_is_passed_through_as_false():
    raw = _base()
    raw["verified"] = 0
    assert cl.parse_config(raw).verified is False
    raw["verified"] = 1
    assert cl.parse_config(raw).verified is True


@pytest.mark.parametrize("value", [True, False, "true", "false", "0", "1", 2, -1])
def test_verified_accepts_only_integers_zero_and_one(value):
    raw = _base()
    raw["verified"] = value
    with pytest.raises(GeminiInputError) as ei:
        cl.parse_config(raw)
    assert "«verified»: нужно 0 или 1" in str(ei.value)


def test_verified_missing_key_defaults_to_true():
    """Fail closed: без ключа фильтр проверенных хостов остаётся включённым."""
    raw = _base()
    del raw["verified"]
    assert cl.parse_config(raw).verified is True


def test_verified_missing_key_does_not_hide_other_missing_keys():
    raw = _base()
    del raw["verified"]
    del raw["fps"]
    err = _assert_rejected(raw, "отсутствует обязательный ключ «fps»")
    assert "«verified»" not in str(err)


def test_secret_like_key_is_refused_with_hint():
    raw = _base()
    raw["vast_api_key"] = "x"
    _assert_rejected(raw, "секреты не хранятся в конфигурации")


def test_top_level_must_be_mapping():
    _assert_rejected(["a", "b"], "верхний уровень")


def test_empty_file_reports_missing_keys():
    with pytest.raises(GeminiInputError) as ei:
        cl.parse_config(None)
    assert "отсутствует обязательный ключ" in str(ei.value)


# ---------------------------------------------------------------- типы

@pytest.mark.parametrize("key, value, fragment", [
    ("fps", 24.5, "целое число"),
    ("fps", "24", "целое число"),
    ("fps", True, "целое число"),                 # bool не считается целым
    ("clip_min_sec", "2.5", "число"),
    ("clip_min_sec", False, "число"),
    ("interruptible", "true", "true или false"),
    ("verified", "true", "0 или 1"),              # строка «true» не считается значением verified
    ("verified", True, "0 или 1"),                # bool не принимается даже в виде True/False
    ("verified", None, "не может быть пустым"),   # null не превращается молча в умолчание
    ("filename_template", 12, "строка"),
    ("gemini_models", "gemini-3.5-flash-lite", "список строк"),
    ("gemini_models", [1, 2], "список строк"),
    ("disk_gb", None, "не может быть пустым"),     # обязательные поля не допускают null
])
def test_type_errors(key, value, fragment):
    raw = _base()
    raw[key] = value
    _assert_rejected(raw, f"«{key}»", fragment)


def test_optional_gpu_max_accepts_null_and_number():
    raw = _base()
    raw["gpu_max_sec"] = None
    assert cl.parse_config(raw).gpu_max_sec is None
    raw["gpu_max_sec"] = 8
    assert cl.parse_config(raw).gpu_max_sec == 8.0


def test_int_field_accepts_int_float_field_accepts_int():
    raw = _base()
    raw["budget_limit_usd"] = 2          # число без дроби допустимо для float (кредит по умолчанию 2.0)
    raw["max_cards"] = 2
    cfg = cl.parse_config(raw)
    assert cfg.budget_limit_usd == 2.0 and isinstance(cfg.budget_limit_usd, float)
    assert cfg.max_cards == 2


# ---------------------------------------------------------------- границы

@pytest.mark.parametrize("key, value, fragment", [
    ("disk_gb", -1, "«disk_gb»: должно быть не меньше 1"),     # из задания: disk_gb < 0
    ("disk_gb", 0, "не меньше 1"),
    ("fps", 0, "«fps»: должно быть не меньше 1"),              # из задания: fps <= 0
    ("clip_min_sec", 0, "«clip_min_sec»: должно быть больше 0"),
    ("tail_pad_sec", -0.1, "не меньше 0"),
    ("min_price_per_hour", -0.01, "не меньше 0"),
    ("min_reliability", 1.5, "не больше 1"),
    ("min_reliability", -0.1, "не меньше 0"),
    ("gpu_max_sec", 0, "больше 0"),
    ("limit_clips", -1, "не меньше 0"),
    ("gemini_rpd_limit", 0, "не меньше 1"),
    ("silent_host_timeout_min", 0, "не меньше 1"),
])
def test_bounds(key, value, fragment):
    raw = _base()
    raw[key] = value
    _assert_rejected(raw, fragment)


@pytest.mark.parametrize("key, value, fragment", [
    ("filename_template", "clip.mp4", "должно содержать {num}"),
    ("filename_template", "", "не может быть пустой"),
    ("model_profile", "../secret", "не соответствует формату"),
    ("model_profile", "", "не соответствует формату"),
    ("target_resolution", "1920*1080", "ШИРИНАxВЫСОТА"),
    ("docker_image", "", "непустая строка без пробелов"),
    ("docker_image", "img latest", "непустая строка без пробелов"),
    ("gemini_models", [], "не может быть пустым"),
    ("gemini_models", ["ok", "  "], "пустые элементы"),
])
def test_string_rules(key, value, fragment):
    raw = _base()
    raw[key] = value
    _assert_rejected(raw, fragment)


def test_empty_gpu_name_means_any_and_is_allowed():
    raw = _base()
    raw["gpu_name"] = ""
    assert cl.parse_config(raw).gpu_name == ""


# ---------------------------------------------------------------- связи полей

def test_clip_max_not_less_than_min():
    raw = _base()
    raw["clip_min_sec"], raw["clip_max_sec"] = 5.0, 2.5
    _assert_rejected(raw, "«clip_max_sec» не может быть меньше «clip_min_sec»")


def test_max_price_must_be_strictly_greater_than_min():
    raw = _base()
    raw["min_price_per_hour"] = raw["max_price_per_hour"]
    _assert_rejected(raw, "строго больше «min_price_per_hour»")


def test_job_deadline_must_be_below_actions_limit():
    raw = _base()
    raw["job_deadline_min"] = cl.ACTIONS_JOB_LIMIT_MIN
    _assert_rejected(raw, "«job_deadline_min» должен быть меньше лимита")


def test_min_balance_default_equals_budget():
    cfg = cl.load_config()
    assert cfg.min_balance_usd == 2.0 == cfg.budget_limit_usd


def test_min_balance_below_budget_is_rejected_with_both_names():
    raw = _base()
    raw["min_balance_usd"] = 1.0
    _assert_rejected(raw, "«min_balance_usd» не может быть меньше «budget_limit_usd»")


def test_min_balance_equal_to_budget_is_allowed():
    raw = _base()
    raw["budget_limit_usd"] = 2.0
    raw["min_balance_usd"] = 2.0
    assert cl.parse_config(raw).min_balance_usd == 2.0


def test_min_balance_below_budget_via_overrides_is_error():
    with pytest.raises(GeminiInputError) as ei:
        cl.load_config(overrides={"min_balance_usd": 1.0})
    assert ei.value.exit_code == 2
    assert "«min_balance_usd» не может быть меньше «budget_limit_usd»" in str(ei.value)


@pytest.mark.parametrize("value, fragment", [
    (-1, "«min_balance_usd»: должно быть не меньше 0"),
    ("5.0", "«min_balance_usd»: нужно число"),
])
def test_min_balance_bounds_and_type(value, fragment):
    raw = _base()
    raw["min_balance_usd"] = value
    _assert_rejected(raw, fragment)


def test_all_problems_are_reported_together():
    raw = _base()
    raw["disk_gb"] = -1
    raw["fps"] = 0
    raw["extra_key"] = 1
    err = _assert_rejected(raw, "«disk_gb»", "«fps»", "«extra_key»")
    assert "проблем: 3" in str(err)


def test_error_message_does_not_echo_string_values():
    raw = _base()
    raw["disk_gb"] = "TOP-SECRET-VALUE"
    err = _assert_rejected(raw, "«disk_gb»: нужно целое число")
    assert "TOP-SECRET-VALUE" not in str(err)


# ---------------------------------------------------------------- файл и YAML

def test_missing_file_is_input_error(tmp_path):
    with pytest.raises(GeminiInputError) as ei:
        cl.load_config(tmp_path / "нет.yaml")
    assert ei.value.exit_code == 2 and "не найден" in str(ei.value)


def test_duplicate_key_in_file_is_error(tmp_path):
    p = tmp_path / "dup.yaml"
    text = cl.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8") + "\nfps: 25\n"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(GeminiInputError) as ei:
        cl.load_config(p)
    assert "повторяющийся ключ «fps»" in str(ei.value)


def test_yaml_syntax_error_is_input_error(tmp_path):
    p = tmp_path / "broken.yaml"
    p.write_text("fps: [24\n", encoding="utf-8")
    with pytest.raises(GeminiInputError) as ei:
        cl.load_config(p)
    assert ei.value.exit_code == 2 and "не разобран как YAML" in str(ei.value)


def test_file_values_pass_schema(tmp_path):
    p = tmp_path / "ok.yaml"
    p.write_text(cl.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
                 .replace("disk_gb: 100", "disk_gb: 80"), encoding="utf-8")
    assert cl.load_config(p).disk_gb == 80


def test_file_with_bad_value_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(cl.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
                 .replace("disk_gb: 100", "disk_gb: -5"), encoding="utf-8")
    with pytest.raises(GeminiInputError) as ei:
        cl.load_config(p)
    assert "«disk_gb»" in str(ei.value)


# ---------------------------------------------------------------- входы (overrides)

def test_overrides_replace_file_values_and_none_is_ignored():
    cfg = cl.load_config(overrides={"max_cards": 2, "style_brief": "стиль", "disk_gb": None})
    assert cfg.max_cards == 2
    assert cfg.style_brief == "стиль"
    assert cfg.disk_gb == 100               # None = «не задано», остаётся из файла


def test_overrides_are_validated_like_file():
    with pytest.raises(GeminiInputError) as ei:
        cl.load_config(overrides={"max_cards": 0})
    assert "«max_cards»" in str(ei.value)


def test_unknown_override_key_is_error():
    with pytest.raises(GeminiInputError) as ei:
        cl.load_config(overrides={"max_card": 2})
    assert "неизвестный ключ «max_card»" in str(ei.value)


# ---------------------------------------------------------------- профиль модели

def test_profile_number_reads_value_and_rejects_bad_names():
    assert cl.load_profile_number("ltx25", "gpu_max_sec") == 10.0
    with pytest.raises(GeminiInputError):
        cl.load_profile_number("../ltx25", "gpu_max_sec")
    with pytest.raises(GeminiInputError):
        cl.load_profile_number("no_such_profile", "gpu_max_sec")
    with pytest.raises(GeminiInputError):
        cl.load_profile_number("ltx25", "no_such_key")


def test_max_inet_price_per_tb_default_and_required():
    assert cl.load_config().max_inet_price_per_tb == 4.1
    raw = _base()
    del raw["max_inet_price_per_tb"]
    _assert_rejected(raw, "отсутствует обязательный ключ «max_inet_price_per_tb»")


@pytest.mark.parametrize("value, fragment", [
    (0, "«max_inet_price_per_tb»: должно быть больше 0"),
    (-1, "«max_inet_price_per_tb»: должно быть больше 0"),
    ("4.1", "«max_inet_price_per_tb»: нужно число"),
])
def test_max_inet_price_per_tb_bounds_and_type(value, fragment):
    raw = _base()
    raw["max_inet_price_per_tb"] = value
    _assert_rejected(raw, fragment)
