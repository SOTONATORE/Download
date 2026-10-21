"""Вход workflow scan_sec: переопределение конфига, пустое значение, отказ до аренды.

Всё через cli.main / build_parser и load_config; без сети и без Vast (подмена на падающую заглушку).
"""
import pytest

from Generate import cli
from Generate.core.config_loader import load_config


@pytest.fixture
def missing_file(tmp_path):
    p = tmp_path / "missing.txt"
    p.write_text("1\n2\n", encoding="utf-8")
    return p


def test_override_replaces_config_value():
    args = cli.build_parser().parse_args(["run", "--scan-sec", "120"])
    assert args.scan_sec == 120
    assert load_config(None, {"scan_sec": args.scan_sec}).scan_sec == 120


def test_empty_override_is_ignored_and_config_value_used():
    args = cli.build_parser().parse_args(["run", "--scan-sec", ""])
    assert args.scan_sec is None
    default = load_config().scan_sec
    assert load_config(None, {"scan_sec": args.scan_sec}).scan_sec == default


def test_value_below_five_times_step_is_rejected_before_rental(monkeypatch, missing_file):
    """scan_step_sec по умолчанию 20: порог 5 × 20 = 100. 99 отвергается до создания клиента Vast."""
    def must_not_run(*_a, **_k):
        raise AssertionError("вызов Vast до проверки конфигурации")

    monkeypatch.setattr(cli, "VastClient", must_not_run, raising=False)
    monkeypatch.setattr(cli, "_make_transport", must_not_run, raising=False)
    monkeypatch.setattr(cli, "_make_release_store", must_not_run, raising=False)

    code = cli.main(["run", "--scan-sec", "99", "--srt", "unused.srt",
                     "--missing", str(missing_file)])
    assert code == cli.EXIT_INPUT


def test_value_at_five_times_step_is_accepted():
    assert load_config(None, {"scan_sec": 100}).scan_sec == 100


@pytest.mark.parametrize("bad", ["abc", "1.5", "-5", "0", "12s"])
def test_non_numeric_or_non_positive_value_is_rejected(bad, monkeypatch, missing_file):
    def must_not_run(*_a, **_k):
        raise AssertionError("вызов Vast при неверном входе")

    monkeypatch.setattr(cli, "VastClient", must_not_run, raising=False)
    code = cli.main(["run", "--scan-sec", bad, "--srt", "unused.srt",
                     "--missing", str(missing_file)])
    assert code == cli.EXIT_INPUT


def test_bad_value_message_does_not_echo_input():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError) as info:
        cli._parse_scan_sec("abc-секрет")
    assert "abc-секрет" not in str(info.value)
