"""Тесты каркаса cli.py: всё через main(argv), без сети и ключей."""
import pathlib

import pytest

from Generate import cli
from Generate.core.srt_parser import parse_srt, srt_hash

DATA = pathlib.Path(__file__).resolve().parent / "data" / "final.srt"


@pytest.fixture
def missing_file(tmp_path):
    p = tmp_path / "missing.txt"
    p.write_text("1\n2\n", encoding="utf-8")
    return p


def test_check_ok(capsys, missing_file):
    code = cli.main(["check", "--srt", str(DATA), "--missing", str(missing_file)])
    out = capsys.readouterr().out
    segs = parse_srt(str(DATA))
    assert code == 0
    assert str(len(segs)) in out
    assert srt_hash(segs) in out


def test_check_does_not_parse_missing(capsys, tmp_path):
    """Содержимое missing.txt не разбирается: любой текст допустим."""
    m = tmp_path / "missing.txt"
    m.write_text("это вообще не список номеров\n", encoding="utf-8")
    assert cli.main(["check", "--srt", str(DATA), "--missing", str(m)]) == 0


def test_check_no_srt(capsys, missing_file, tmp_path):
    code = cli.main(["check", "--srt", str(tmp_path / "нет.srt"),
                     "--missing", str(missing_file)])
    err = capsys.readouterr().err
    assert code == 2
    assert "SRT" in err and "Traceback" not in err


def test_check_no_missing(capsys, tmp_path):
    code = cli.main(["check", "--srt", str(DATA),
                     "--missing", str(tmp_path / "нет.txt")])
    err = capsys.readouterr().err
    assert code == 2
    assert "missing" in err and "Traceback" not in err


def test_check_broken_srt(capsys, missing_file, tmp_path):
    bad = tmp_path / "bad.srt"
    bad.write_text("1\nэто не тайминг\nтекст\n", encoding="utf-8")
    code = cli.main(["check", "--srt", str(bad), "--missing", str(missing_file)])
    err = capsys.readouterr().err
    assert code == 2
    assert "Traceback" not in err


def test_check_empty_srt(capsys, missing_file, tmp_path):
    empty = tmp_path / "empty.srt"
    empty.write_text("", encoding="utf-8")
    assert cli.main(["check", "--srt", str(empty), "--missing", str(missing_file)]) == 2


def test_check_srt_is_directory(capsys, missing_file, tmp_path):
    assert cli.main(["check", "--srt", str(tmp_path), "--missing", str(missing_file)]) == 2


def test_check_not_utf8(capsys, missing_file, tmp_path):
    bad = tmp_path / "cp1251.srt"
    bad.write_bytes("1\n00:00:00,000 --> 00:00:01,000\nпривет\n".encode("cp1251"))
    code = cli.main(["check", "--srt", str(bad), "--missing", str(missing_file)])
    assert code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_defaults_are_in_cwd(capsys, tmp_path, monkeypatch):
    """Без параметров ищутся final.srt и missing.txt в текущей папке."""
    monkeypatch.chdir(tmp_path)
    assert cli.main(["check"]) == 2  # файлов нет
    (tmp_path / "final.srt").write_bytes(DATA.read_bytes())
    (tmp_path / "missing.txt").write_text("", encoding="utf-8")
    assert cli.main(["check"]) == 0


@pytest.mark.parametrize("cmd", ["run", "kill-cards"])
def test_stubs_return_3(cmd, capsys):
    code = cli.main([cmd])
    assert code == 3
    assert "ещё не реализована" in capsys.readouterr().err


def test_unknown_command(capsys):
    assert cli.main(["нет-такой"]) == 2


def test_no_command(capsys):
    assert cli.main([]) == 2


def test_unknown_option(capsys):
    assert cli.main(["check", "--неизвестно", "x"]) == 2


def test_help_is_zero(capsys):
    assert cli.main(["--help"]) == 0


def test_unexpected_exception_gives_3(capsys, monkeypatch):
    def boom(args):
        raise RuntimeError("СЕКРЕТНЫЙ-ТЕКСТ-ИСКЛЮЧЕНИЯ")

    monkeypatch.setattr(cli, "cmd_run", boom)
    code = cli.main(["run"])
    cap = capsys.readouterr()
    assert code == 3
    assert "Traceback" not in cap.err
    assert "RuntimeError" in cap.err
    # текст исключения не печатается (SPEC 0.1)
    assert "СЕКРЕТНЫЙ-ТЕКСТ-ИСКЛЮЧЕНИЯ" not in cap.err + cap.out
