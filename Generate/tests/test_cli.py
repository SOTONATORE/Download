"""Тесты cli.py: всё через main(argv), без сети и ключей (всё внешнее подменено)."""
import json
import logging
import pathlib
import re
import shutil
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from Generate import cli
from Generate.core import vast_client
from Generate.core.srt_parser import parse_srt, srt_hash
from Generate.core import config_loader as _config_loader
from Generate.core import gemini_prompts as _gemini_prompts

DATA = pathlib.Path(__file__).resolve().parent / "data" / "final.srt"
VAST_KEY_ENV = "GEN_" + "VAST_API_KEY"
SECRET_VALUE = "СЕКРЕТНОЕ-ЗНАЧЕНИЕ-КЛЮЧА"


@pytest.fixture
def missing_file(tmp_path):
    p = tmp_path / "missing.txt"
    p.write_text("1\n2\n", encoding="utf-8")
    return p


@pytest.fixture
def hf_token(monkeypatch):
    """Preflight-проверка (S1) требует HF_TOKEN в os.environ при команде check."""
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")


# ---------------------------------------------------------------- check

def test_check_ok(capsys, missing_file, hf_token):
    code = cli.main(["check", "--srt", str(DATA), "--missing", str(missing_file)])
    out = capsys.readouterr().out
    segs = parse_srt(str(DATA))
    assert code == 0
    assert str(len(segs)) in out
    assert srt_hash(segs) in out


def test_check_does_not_parse_missing(capsys, tmp_path, hf_token):
    """Содержимое missing.txt не разбирается: любой текст допустим."""
    m = tmp_path / "missing.txt"
    m.write_text("это вообще не список номеров\n", encoding="utf-8")
    assert cli.main(["check", "--srt", str(DATA), "--missing", str(m)]) == 0


def test_check_no_srt(capsys, missing_file, tmp_path, hf_token):
    code = cli.main(["check", "--srt", str(tmp_path / "нет.srt"),
                     "--missing", str(missing_file)])
    err = capsys.readouterr().err
    assert code == 2
    assert "SRT" in err and "Traceback" not in err


def test_check_no_missing(capsys, tmp_path, hf_token):
    code = cli.main(["check", "--srt", str(DATA),
                     "--missing", str(tmp_path / "нет.txt")])
    err = capsys.readouterr().err
    assert code == 2
    assert "missing" in err and "Traceback" not in err


def test_check_broken_srt(capsys, missing_file, tmp_path, hf_token):
    bad = tmp_path / "bad.srt"
    bad.write_text("1\nэто не тайминг\nтекст\n", encoding="utf-8")
    code = cli.main(["check", "--srt", str(bad), "--missing", str(missing_file)])
    err = capsys.readouterr().err
    assert code == 2
    assert "Traceback" not in err


def test_check_empty_srt(capsys, missing_file, tmp_path, hf_token):
    empty = tmp_path / "empty.srt"
    empty.write_text("", encoding="utf-8")
    assert cli.main(["check", "--srt", str(empty), "--missing", str(missing_file)]) == 2


def test_check_srt_is_directory(capsys, missing_file, tmp_path, hf_token):
    assert cli.main(["check", "--srt", str(tmp_path), "--missing", str(missing_file)]) == 2


def test_check_not_utf8(capsys, missing_file, tmp_path, hf_token):
    bad = tmp_path / "cp1251.srt"
    bad.write_bytes("1\n00:00:00,000 --> 00:00:01,000\nпривет\n".encode("cp1251"))
    code = cli.main(["check", "--srt", str(bad), "--missing", str(missing_file)])
    assert code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_defaults_are_in_cwd(capsys, tmp_path, monkeypatch, hf_token):
    """Без параметров ищутся final.srt и missing.txt в текущей папке."""
    monkeypatch.chdir(tmp_path)
    assert cli.main(["check"]) == 2  # файлов нет
    (tmp_path / "final.srt").write_bytes(DATA.read_bytes())
    (tmp_path / "missing.txt").write_text("", encoding="utf-8")
    assert cli.main(["check"]) == 0


# ---------------------------------------------------------------- общие

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


# ---------------------------------------------------------------- run

class FakeVast:
    """Подмена VastClient: ключ не читается, сети нет."""
    instances = []
    credit = 10.0        # что вернёт предстартовая проверка кредита (клиент читает поле credit)
    credit_error = None  # если задано, проверка кредита падает с этой ошибкой

    def __init__(self, *a, **k):
        self.closed = False
        FakeVast.instances.append(self)

    def get_balance_usd(self):
        if FakeVast.credit_error is not None:
            raise FakeVast.credit_error
        return FakeVast.credit

    def check_instances_endpoint(self):
        """Предстартовая проверка эндпоинта списка (по умолчанию успешна)."""
        return None

    def close(self):
        self.closed = True


@pytest.fixture
def run_env(monkeypatch):
    """Подменяет всё внешнее для команды run; возвращает список вызовов оркестратора."""
    calls = []
    FakeVast.instances = []
    FakeVast.credit = 10.0
    FakeVast.credit_error = None
    monkeypatch.setattr(cli, "_build_timing", lambda cfg, profile: "TIMING")
    monkeypatch.setattr(cli, "_build_prompt_config",
                        lambda profile, batch_size, preset="none": ("CFG", profile, batch_size))
    monkeypatch.setattr(cli, "_make_transport", lambda: "TRANSPORT")
    monkeypatch.setattr(cli, "_make_release_store", lambda repo: ("STORE", repo))
    monkeypatch.setattr(cli, "VastClient", FakeVast)

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return fake_run.summary

    fake_run.summary = SimpleNamespace(
        total_needed=2, completed=2, failed=0, pending=0, skipped_done=1,
        spent_usd=0.5, exit_code=0, deadline_reached=False,
        budget_exceeded=False, cleanup_ok=True, cards_rented=1)
    monkeypatch.setattr(cli, "run_generation", fake_run)
    fake_run.calls = calls
    return fake_run


def _run_args(srt, missing, *extra):
    return ["run", "--srt", str(srt), "--missing", str(missing), *extra]


def test_run_no_srt(capsys, run_env, missing_file, tmp_path):
    code = cli.main(_run_args(tmp_path / "нет.srt", missing_file))
    assert code == 2
    assert "SRT" in capsys.readouterr().err
    assert run_env.calls == []


def test_run_broken_srt(capsys, run_env, missing_file, tmp_path):
    bad = tmp_path / "bad.srt"
    bad.write_text("1\nэто не тайминг\nтекст\n", encoding="utf-8")
    assert cli.main(_run_args(bad, missing_file)) == 2
    assert run_env.calls == []


def test_run_no_missing(capsys, run_env, tmp_path):
    code = cli.main(_run_args(DATA, tmp_path / "нет.txt"))
    err = capsys.readouterr().err
    assert code == 2
    assert "missing" in err and "Traceback" not in err
    assert run_env.calls == []


def test_run_success(capsys, run_env, missing_file):
    code = cli.main(_run_args(DATA, missing_file, "--repo", "o/r", "--run-id", "77",
                              "--limit-clips", "3", "--max-cards", "2",
                              "--budget-limit", "1.5", "--release-tag", "v1",
                              "--model-profile", "ltx25",
                              "--prompts-path", "p.json"))
    out = capsys.readouterr().out
    assert code == 0
    assert "Итоги запуска" in out
    (args, kwargs), = run_env.calls
    assert args[1] == str(missing_file)
    assert args[2] == ("CFG", "ltx25", 50)   # gemini_batch_size из config.default.yaml
    assert args[3] == "TIMING"
    assert args[4] == ("STORE", "o/r")
    assert isinstance(args[5], FakeVast)
    assert args[6] == "TRANSPORT"
    assert args[7] == "p.json"
    assert kwargs["limit_clips"] == 3
    assert kwargs["max_cards"] == 2
    assert kwargs["budget_limit_usd"] == 1.5
    assert kwargs["release_tag"] == "v1"
    assert kwargs["run_id"] == "77"
    assert kwargs["repo"] == "o/r"
    assert FakeVast.instances[0].closed


def test_run_nothing_to_generate_is_ok(capsys, run_env, missing_file):
    run_env.summary = SimpleNamespace(exit_code=0)
    assert cli.main(_run_args(DATA, missing_file)) == 0


def test_run_partial_gives_3(capsys, run_env, missing_file):
    run_env.summary = SimpleNamespace(total_needed=3, completed=1, failed=2,
                                      exit_code=3, cleanup_ok=True)
    code = cli.main(_run_args(DATA, missing_file))
    assert code == 3
    assert FakeVast.instances[0].closed


def test_run_defaults_without_repo_and_run_id(capsys, run_env, missing_file, monkeypatch):
    """Без --repo и --run-id значения пустые; окружение cli не читает (SPEC 0.2)."""
    monkeypatch.setenv("GITHUB_REPOSITORY", "env/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "555")
    assert cli.main(_run_args(DATA, missing_file)) == 0
    _, kwargs = run_env.calls[0]
    assert kwargs["repo"] == ""
    assert kwargs["run_id"] == ""
    assert kwargs["limit_clips"] == 0
    assert kwargs["max_cards"] == 1


def test_run_default_offer_filter(capsys, run_env, missing_file):
    """Без флагов фильтр поиска карт получает значения по умолчанию."""
    assert cli.main(_run_args(DATA, missing_file)) == 0
    _, kwargs = run_env.calls[0]
    f = kwargs["offer_filter"]
    assert isinstance(f, cli.OfferFilter)
    assert f.gpu_name == "RTX 5090"
    assert f.min_price == 0.35
    assert f.max_price == 1.0
    assert f.min_reliability == 0.90
    assert f.min_inet_mbps == 800.0
    assert f.max_inet_price_per_tb == 4.1
    assert f.interruptible is False                 # rental_type по умолчанию on-demand


def test_run_custom_offer_filter(capsys, run_env, missing_file):
    """Флаги поиска карт доходят до run_generation через offer_filter."""
    code = cli.main(_run_args(DATA, missing_file,
                              "--gpu-name", "RTX 4090",
                              "--min-price-per-hour", "0.2",
                              "--max-price-per-hour", "0.6",
                              "--min-reliability", "0.9",
                              "--min-inet-mbps", "1500",
                              "--max-inet-price-per-tb", "3.0"))
    assert code == 0
    _, kwargs = run_env.calls[0]
    f = kwargs["offer_filter"]
    assert isinstance(f, cli.OfferFilter)
    assert f.gpu_name == "RTX 4090"
    assert f.min_price == 0.2
    assert f.max_price == 0.6
    assert f.min_reliability == 0.9
    assert f.min_inet_mbps == 1500.0
    assert f.max_inet_price_per_tb == 3.0


def test_run_default_offer_filter_verified_is_true(capsys, run_env, missing_file):
    """Производственный поиск берёт verified из config.default.yaml: по умолчанию 1 (True)."""
    assert cli.main(_run_args(DATA, missing_file)) == 0
    _, kwargs = run_env.calls[0]
    assert kwargs["offer_filter"].verified is True


def test_run_verified_zero_from_config_reaches_offer_filter(capsys, run_env, missing_file, tmp_path):
    """Значение verified: 0 из файла конфигурации доходит до OfferFilter как False."""
    text = cli.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert "\nverified: 1\n" in text
    text = text.replace("\nverified: 1\n", "\nverified: 0\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(text, encoding="utf-8")
    assert cli.main(_run_args(DATA, missing_file, "--config", str(cfg))) == 0
    _, kwargs = run_env.calls[0]
    assert kwargs["offer_filter"].verified is False


# ---------------------------------------------------------------- --preset (вход workflow)

def test_preset_defaults_to_none_and_reaches_config(loaded_cfgs, run_env, missing_file):
    assert cli.build_parser().parse_args(["run"]).preset == "none"
    cli.main(_run_args(DATA, missing_file))
    assert loaded_cfgs[-1].preset == "none"


# ---------------------------------------------------------------- --verified (вход workflow)

@pytest.fixture
def loaded_cfgs(monkeypatch):
    """Запоминает конфиг, который cmd_run получил от _load_cfg (уже с наложенными входами)."""
    seen = []
    real_load = cli._load_cfg

    def spy(path, overrides=None):
        cfg, code = real_load(path, overrides)
        seen.append(cfg)
        return cfg, code

    monkeypatch.setattr(cli, "_load_cfg", spy)
    return seen


@pytest.mark.parametrize("raw, expected", [
    ("1", True),
    ("0", False),
])
def test_verified_flag_parsing(raw, expected):
    assert cli.build_parser().parse_args(["run", "--verified", raw]).verified is expected


def test_verified_missing_argument_is_none():
    assert cli.build_parser().parse_args(["run"]).verified is None


@pytest.mark.parametrize("bad", ["true", "false", "TRUE", "False", "", "2", "maybe", "yes"])
def test_verified_invalid_value_is_input_error(capsys, run_env, missing_file, bad):
    """Допустимы только «1» и «0». true, false, пустая строка и любое другое значение: код 2, до Vast."""
    code = cli.main(_run_args(DATA, missing_file, "--verified", bad))
    err = capsys.readouterr().err
    assert code == 2
    assert "допустимы только значения 1 и 0" in err
    assert "Traceback" not in err
    assert run_env.calls == []


def test_verified_zero_flag_reaches_loaded_config(capsys, run_env, loaded_cfgs, missing_file):
    assert cli.main(_run_args(DATA, missing_file, "--verified", "0")) == 0
    assert loaded_cfgs[0].verified is False
    _, kwargs = run_env.calls[0]
    assert kwargs["offer_filter"].verified is False


def test_verified_absent_keeps_config_value(capsys, run_env, loaded_cfgs, missing_file):
    assert cli.main(_run_args(DATA, missing_file)) == 0
    assert loaded_cfgs[0].verified is True       # значение из config.default.yaml: 1


def test_verified_one_flag_overrides_config_zero(capsys, run_env, loaded_cfgs, missing_file, tmp_path):
    text = cli.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert "\nverified: 1\n" in text
    text = text.replace("\nverified: 1\n", "\nverified: 0\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(text, encoding="utf-8")
    assert cli.main(_run_args(DATA, missing_file, "--config", str(cfg), "--verified", "1")) == 0
    assert loaded_cfgs[0].verified is True
    _, kwargs = run_env.calls[0]
    assert kwargs["offer_filter"].verified is True


def test_run_default_card_params(capsys, run_env, missing_file):
    """Без флагов docker_image/disk_gb/silent_host_timeout_min получают значения по умолчанию."""
    assert cli.main(_run_args(DATA, missing_file)) == 0
    _, kwargs = run_env.calls[0]
    assert kwargs["docker_image"] == "vastai/comfy:v0.41.0-cuda-13.2-py312"
    assert kwargs["disk_gb"] == 100
    assert kwargs["silent_host_timeout_min"] == 20


def test_run_custom_card_params(capsys, run_env, missing_file):
    """Пользовательские значения флагов доходят до run_generation."""
    code = cli.main(_run_args(DATA, missing_file,
                              "--docker-image", "my-image:v1",
                              "--disk-gb", "60",
                              "--silent-host-timeout-min", "20"))
    assert code == 0
    _, kwargs = run_env.calls[0]
    assert kwargs["docker_image"] == "my-image:v1"
    assert kwargs["disk_gb"] == 60
    assert kwargs["silent_host_timeout_min"] == 20


def test_run_empty_gpu_name_means_any(capsys, run_env, missing_file):
    """Пустое имя карты превращается в None (любая модель)."""
    assert cli.main(_run_args(DATA, missing_file, "--gpu-name", "")) == 0
    _, kwargs = run_env.calls[0]
    assert kwargs["offer_filter"].gpu_name is None


def test_run_explicit_repo_and_run_id(capsys, run_env, missing_file):
    code = cli.main(_run_args(DATA, missing_file, "--repo", "arg/repo", "--run-id", "321"))
    assert code == 0
    (args, kwargs), = run_env.calls
    assert args[4] == ("STORE", "arg/repo")
    assert kwargs["repo"] == "arg/repo"
    assert kwargs["run_id"] == "321"


def test_run_vast_key_missing_gives_2(capsys, run_env, missing_file, monkeypatch):
    class NoKey(cli.VastError):
        exit_code = 2

    def boom(*a, **k):
        raise NoKey("Не задан ключ Vast")

    monkeypatch.setattr(cli, "VastClient", boom)
    code = cli.main(_run_args(DATA, missing_file))
    cap = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in cap.err
    assert run_env.calls == []


# ---------------------------------------------------------------- предстартовая проверка кредита

TESTKEY = "TESTKEY-12345"


def test_credit_below_minimum_gives_2(capsys, run_env, missing_file):
    FakeVast.credit = 1.5
    code = cli.main(_run_args(DATA, missing_file))
    err = capsys.readouterr().err
    assert code == 2
    assert "Недостаточно средств на Vast.ai: кредит 1.50 USD, нужно не менее 2.00 USD" in err
    assert "Пополните баланс и запустите снова" in err
    # до Gemini и до аренды карт дело не доходит: оркестратор не вызывается
    assert run_env.calls == []
    assert FakeVast.instances[0].closed


def test_credit_query_fails_gives_3(capsys, run_env, missing_file):
    FakeVast.credit_error = cli.VastRuntimeError("Vast API вернул HTTP 503: GET /users/current")
    code = cli.main(_run_args(DATA, missing_file))
    cap = capsys.readouterr()
    assert code == 3
    assert "Не удалось проверить кредит Vast.ai" in cap.err
    assert "карты не арендованы" in cap.err
    assert "Traceback" not in cap.err
    assert run_env.calls == []
    assert FakeVast.instances[0].closed


def test_credit_sufficient_continues_and_logs(capsys, run_env, missing_file, caplog):
    caplog.set_level(logging.INFO, logger="Generate.cli")
    FakeVast.credit = 12.5
    code = cli.main(_run_args(DATA, missing_file))
    assert code == 0
    assert len(run_env.calls) == 1
    assert "Кредит Vast.ai: 12.50 USD, минимум 2.00 USD." in caplog.text


def test_credit_equal_to_minimum_continues(capsys, run_env, missing_file):
    FakeVast.credit = 2.0
    assert cli.main(_run_args(DATA, missing_file)) == 0
    assert len(run_env.calls) == 1


def test_min_balance_flag_reaches_preflight(capsys, run_env, missing_file):
    """Вход --min-balance-usd переопределяет конфиг и действует на проверку."""
    FakeVast.credit = 8.0
    code = cli.main(_run_args(DATA, missing_file, "--min-balance-usd", "9.0"))
    assert code == 2
    assert "нужно не менее 9.00 USD" in capsys.readouterr().err
    assert run_env.calls == []


def test_credit_refusal_does_not_leak_key(capsys, run_env, missing_file, monkeypatch):
    monkeypatch.setenv(VAST_KEY_ENV, TESTKEY)
    FakeVast.credit = 1.0
    assert cli.main(_run_args(DATA, missing_file)) == 2
    cap = capsys.readouterr()
    assert TESTKEY not in cap.out + cap.err


def test_credit_http_error_does_not_leak_key(capsys, run_env, missing_file, monkeypatch):
    """Настоящий клиент: ответ 401, который эхом возвращает заголовок, не выводит ключ (SPEC 0.1)."""
    monkeypatch.setenv(VAST_KEY_ENV, TESTKEY)

    def echo(req):
        return httpx.Response(401, text=f"bad header {req.headers['Authorization']}")

    real = vast_client.VastClient(client=httpx.Client(transport=httpx.MockTransport(echo)))
    monkeypatch.setattr(cli, "VastClient", lambda *a, **k: real)
    code = cli.main(_run_args(DATA, missing_file))
    cap = capsys.readouterr()
    assert code == 3
    assert "HTTP 401" in cap.err
    assert TESTKEY not in cap.out + cap.err
    assert run_env.calls == []


def test_run_setup_error_hides_text(capsys, run_env, missing_file, monkeypatch):
    def boom():
        raise ValueError(SECRET_VALUE)

    monkeypatch.setattr(cli, "_make_transport", boom)
    code = cli.main(_run_args(DATA, missing_file))
    cap = capsys.readouterr()
    assert code == 3
    assert "ValueError" in cap.err
    assert SECRET_VALUE not in cap.err + cap.out


# ---------------------------------------------------------------- kill-cards

class KillVast:
    """Подмена VastClient для kill-cards."""
    result = [11, 22]
    error = None
    labels = []

    def __init__(self, *a, **k):
        self.closed = False

    def destroy_by_label(self, label):
        KillVast.labels.append(label)
        if KillVast.error is not None:
            raise KillVast.error
        return list(KillVast.result)

    def close(self):
        self.closed = True


@pytest.fixture
def kill_env(monkeypatch):
    KillVast.result = [11, 22]
    KillVast.error = None
    KillVast.labels = []
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.setattr(cli, "VastClient", KillVast)
    return KillVast


def test_kill_no_label_no_run_id(capsys, kill_env):
    code = cli.main(["kill-cards", "--run-id", ""])
    assert code == 2
    assert capsys.readouterr().err
    assert kill_env.labels == []


def test_kill_by_run_id(capsys, kill_env):
    code = cli.main(["kill-cards", "--run-id", "123"])
    out = capsys.readouterr().out
    assert code == 0
    assert kill_env.labels == ["gen-123"]
    assert "2" in out


def test_kill_by_label_wins(capsys, kill_env):
    code = cli.main(["kill-cards", "--label", "моя", "--run-id", "123"])
    assert code == 0
    assert kill_env.labels == ["моя"]


def test_kill_explicit_run_id(capsys, kill_env):
    assert cli.main(["kill-cards", "--run-id", "999"]) == 0
    assert kill_env.labels == ["gen-999"]


def test_kill_ignores_env_run_id(capsys, kill_env, monkeypatch):
    """Номер запуска из окружения не подхватывается: без аргументов метки нет (SPEC 0.2)."""
    monkeypatch.setenv("GITHUB_RUN_ID", "999")
    assert cli.main(["kill-cards"]) == 2
    assert kill_env.labels == []


def test_kill_nothing_to_destroy_is_ok(capsys, kill_env):
    kill_env.result = []
    assert cli.main(["kill-cards", "--label", "x"]) == 0
    assert "0" in capsys.readouterr().out


def test_kill_vast_failure_gives_3(capsys, kill_env):
    kill_env.error = cli.VastError("Не удалось уничтожить экземпляры Vast: [1]")
    code = cli.main(["kill-cards", "--label", "x"])
    cap = capsys.readouterr()
    assert code == 3
    assert "Traceback" not in cap.err


def test_kill_client_init_failure_gives_3(capsys, monkeypatch):
    def boom(*a, **k):
        raise cli.VastError("Не задан ключ Vast")

    monkeypatch.setattr(cli, "VastClient", boom)
    assert cli.main(["kill-cards", "--label", "x"]) == 3


def test_kill_does_not_leak_key(capsys, kill_env, monkeypatch):
    monkeypatch.setenv(VAST_KEY_ENV, SECRET_VALUE)
    cli.main(["kill-cards", "--label", "x"])
    cap = capsys.readouterr()
    assert SECRET_VALUE not in cap.out + cap.err


# ---------------------------------------------------------------- finalize-release

class FakeStore:
    """Подмена хранилища: find_releases возвращает заранее заданный список."""
    def __init__(self, releases):
        self.releases = releases
        self.tags = []

    def find_releases(self, tag):
        self.tags.append(tag)
        return list(self.releases)


class FakeUploader:
    """Подмена ReleaseUploader: запоминает загрузки и содержимое файлов."""
    instances = []
    done = {1, 2}
    error = None

    def __init__(self, store, segments, model, release_tag=None, **kw):
        self.store = store
        self.model = model
        self.release_tag = release_tag
        self.done = set(FakeUploader.done)
        self.prepared = False
        self.uploaded = {}  # имя файла -> содержимое
        FakeUploader.instances.append(self)

    def prepare(self):
        if FakeUploader.error is not None:
            raise FakeUploader.error
        self.prepared = True
        return set(self.done)

    def upload_service_file(self, path):
        with open(path, encoding="utf-8") as fh:
            self.uploaded[pathlib.Path(path).name] = fh.read()

    def links_text(self, repo):
        return f"https://github.com/{repo}/releases/download/t/1.mp4\n"


@pytest.fixture
def fin_env(monkeypatch):
    FakeUploader.instances = []
    FakeUploader.done = {1, 2}
    FakeUploader.error = None
    store = FakeStore([SimpleNamespace(tag="x")])
    monkeypatch.setattr(cli, "_make_release_store", lambda repo: store)
    monkeypatch.setattr(cli, "ReleaseUploader", FakeUploader)
    store.fake_uploader = FakeUploader
    return store


@pytest.fixture
def prompts_file(tmp_path):
    p = tmp_path / "prompts.json"
    p.write_text('{"1": "промпт"}', encoding="utf-8")
    return p


def _fin_args(*extra):
    return ["finalize-release", "--srt", str(DATA), *extra]


def test_finalize_bad_srt(capsys, fin_env, tmp_path):
    code = cli.main(["finalize-release", "--srt", str(tmp_path / "нет.srt")])
    assert code == 2
    assert FakeUploader.instances == []


def test_finalize_release_missing_is_ok(capsys, fin_env, prompts_file):
    fin_env.releases = []
    code = cli.main(_fin_args("--repo", "o/r", "--prompts-path", str(prompts_file)))
    out = capsys.readouterr().out
    assert code == 0
    assert "не создан" in out
    assert FakeUploader.instances == []


def test_finalize_uploads_prompts_and_links(capsys, fin_env, prompts_file):
    code = cli.main(_fin_args("--repo", "o/r", "--model-profile", "ltx25",
                              "--prompts-path", str(prompts_file)))
    out = capsys.readouterr().out
    assert code == 0
    (up,) = FakeUploader.instances
    assert up.prepared
    assert up.model == "ltx25"
    assert up.uploaded["prompts.json"] == '{"1": "промпт"}'
    assert up.uploaded["generated_links.txt"].startswith("https://github.com/o/r/")
    assert "Выдача клипов завершена" in out


def test_finalize_prompts_custom_name_is_renamed(capsys, fin_env, tmp_path):
    custom = tmp_path / "мои.json"
    custom.write_text("{}", encoding="utf-8")
    assert cli.main(_fin_args("--repo", "o/r", "--prompts-path", str(custom))) == 0
    assert "prompts.json" in FakeUploader.instances[0].uploaded


def test_finalize_without_prompts_file(capsys, fin_env, tmp_path):
    code = cli.main(_fin_args("--repo", "o/r", "--prompts-path", str(tmp_path / "нет.json")))
    assert code == 0
    (up,) = FakeUploader.instances
    assert "prompts.json" not in up.uploaded
    assert "generated_links.txt" in up.uploaded


def test_finalize_no_repo_skips_links(capsys, fin_env, prompts_file):
    assert cli.main(_fin_args("--prompts-path", str(prompts_file))) == 0
    (up,) = FakeUploader.instances
    assert "generated_links.txt" not in up.uploaded
    assert "prompts.json" in up.uploaded


def test_finalize_no_clips_skips_links(capsys, fin_env, prompts_file):
    FakeUploader.done = set()
    assert cli.main(_fin_args("--repo", "o/r", "--prompts-path", str(prompts_file))) == 0
    assert "generated_links.txt" not in FakeUploader.instances[0].uploaded


def test_finalize_custom_release_tag(capsys, fin_env, prompts_file):
    cli.main(_fin_args("--repo", "o/r", "--release-tag", "мой-тег",
                       "--prompts-path", str(prompts_file)))
    assert fin_env.tags == ["мой-тег"]
    assert FakeUploader.instances[0].release_tag == "мой-тег"


def test_finalize_failure_gives_3_and_hides_text(capsys, fin_env, prompts_file):
    FakeUploader.error = ValueError(SECRET_VALUE)
    code = cli.main(_fin_args("--repo", "o/r", "--prompts-path", str(prompts_file)))
    cap = capsys.readouterr()
    assert code == 3
    assert "ValueError" in cap.err
    assert SECRET_VALUE not in cap.err + cap.out
    assert "Traceback" not in cap.err


# ---------------------------------------------------------------- структура

CLI_SOURCE = pathlib.Path(cli.__file__).read_text(encoding="utf-8")


def test_no_api_key_names_in_code():
    for name in ("GEN_" + "VAST_API_KEY", "GEN_" + "GEMINI_API_KEY"):
        assert name not in CLI_SOURCE


def test_no_environment_reading_in_cli():
    # разрешено одно чтение: GEN_LOG_LEVEL (SPEC 10); всё остальное окружение запрещено
    assert CLI_SOURCE.count("os." + "environ") == 1
    assert "os.environ.get(LOG_LEVEL_ENV" in CLI_SOURCE
    assert "get" + "env" not in CLI_SOURCE


# ---------------------------------------------------------------- образ карты (стоковый, SPEC 8.4, вариант A)

SHA = "0123456789abcdef0123456789abcdef01234567"
STOCK_IMAGE = "vastai/comfy:v0.41.0-cuda-13.2-py312"


def test_run_without_image_arg_uses_config_image(capsys, run_env, missing_file):
    code = cli.main(_run_args(DATA, missing_file, "--worker-build", SHA))
    assert code == 0
    _, kwargs = run_env.calls[0]
    assert kwargs["docker_image"] == STOCK_IMAGE


def test_run_empty_image_is_input_error(capsys, run_env, missing_file):
    code = cli.main(_run_args(DATA, missing_file, "--docker-image", "", "--worker-build", SHA))
    assert code == 2
    assert run_env.calls == []


def test_run_image_with_space_is_input_error(capsys, run_env, missing_file):
    code = cli.main(_run_args(DATA, missing_file, "--docker-image", "a b:c", "--worker-build", SHA))
    assert code == 2
    assert run_env.calls == []


def test_explicit_image_is_used_as_is(capsys, run_env, missing_file):
    code = cli.main(_run_args(DATA, missing_file, "--docker-image", "my/img:1", "--worker-build", SHA))
    assert code == 0
    assert run_env.calls[0][1]["docker_image"] == "my/img:1"


def test_resolve_image_returns_value_as_is():
    assert cli._resolve_docker_image("a:b") == "a:b"
    assert cli._resolve_docker_image(STOCK_IMAGE) == STOCK_IMAGE
    with pytest.raises(ValueError):
        cli._resolve_docker_image("")
    with pytest.raises(ValueError):
        cli._resolve_docker_image("a b:c")


def test_no_ghcr_substitution_left_in_cli():
    code = CLI_SOURCE
    assert "ghcr.io" not in code
    assert "manifest" not in code
    assert "_check_worker_image" not in code


def _check_args(missing, *extra):
    return ["check", "--srt", str(DATA), "--missing", str(missing), *extra]


def test_check_without_image_args_passes(capsys, missing_file):
    assert cli.main(_check_args(missing_file)) == 0


def test_check_rejects_removed_image_args(capsys, missing_file):
    assert cli.main(_check_args(missing_file, "--docker-image", "x:y")) == 2


def test_run_passes_loaded_bad_hosts_to_orchestrator(capsys, run_env, missing_file, monkeypatch):
    """Список плохих машин загружается один раз и передаётся оркестратору (SPEC 3, 8.2)."""
    monkeypatch.setattr(cli, "load_bad_hosts", lambda *a, **k: frozenset({58482}))
    assert cli.main(_run_args(DATA, missing_file)) == 0
    _, kwargs = run_env.calls[0]
    assert kwargs["excluded_machines"] == frozenset({58482})


def test_run_bad_hosts_error_stops_before_any_spend(capsys, run_env, missing_file, monkeypatch):
    """Ошибка формата bad_hosts.txt: код 2, оркестратор не вызывается, в stderr назван номер строки."""
    def boom(*a, **k):
        raise cli.BadHostsError("неверный формат файла bad_hosts.txt: строка 2: test")
    monkeypatch.setattr(cli, "load_bad_hosts", boom)
    assert cli.main(_run_args(DATA, missing_file)) == 2
    assert run_env.calls == []
    assert "строка 2" in capsys.readouterr().err


def test_run_passes_worker_build_and_run_id_to_orchestrator(capsys, run_env, missing_file):
    """Этап 6а: номер сборки и номер запуска доходят до оркестратора (SPEC 8.4, 9.1)."""
    code = cli.main(_run_args(DATA, missing_file, "--run-id", "555", "--worker-build", SHA))
    assert code == 0
    (_, kwargs), = run_env.calls
    assert kwargs["worker_build"] == SHA
    assert kwargs["run_id"] == "555"
    assert "card_max_lifetime_min" not in kwargs


def test_orchestrator_input_error_is_exit_2_in_russian(capsys, run_env, missing_file, monkeypatch):
    """Ошибка входа из оркестратора (нет run_id) — код 2 и текст на русском, без трейсбека."""
    def refuse(*args, **kwargs):
        raise cli.GeminiInputError("Не задан номер запуска (run_id): без него карты нельзя пометить меткой.")
    monkeypatch.setattr(cli, "run_generation", refuse)
    code = cli.main(_run_args(DATA, missing_file, "--worker-build", SHA))
    err = capsys.readouterr().err
    assert code == 2
    assert "Ошибка входных данных" in err and "run_id" in err
    assert "Traceback" not in err


# ---------------------------------------------------------------- этап 6c: релиз запуска, артефакт, отмена

def test_release_start_error_is_exit_2_in_russian(capsys, run_env, missing_file, monkeypatch):
    """E1: служебный релиз не открыт или файл не загружен — код 2 до трат, текст на русском."""
    def refuse(*args, **kwargs):
        raise cli.ReleaseStartError("Не удалось загрузить missing.txt в служебный релиз: сбой. Запуск остановлен до трат.")
    monkeypatch.setattr(cli, "run_generation", refuse)
    code = cli.main(_run_args(DATA, missing_file, "--worker-build", SHA, "--run-id", "9"))
    err = capsys.readouterr().err
    assert code == 2
    assert "Ошибка выдачи в релиз" in err and "missing.txt" in err
    assert "Traceback" not in err


def test_artifact_dir_is_passed_to_orchestrator(capsys, run_env, missing_file, tmp_path):
    """E5: --artifact-dir доходит до оркестратора; без него артефакт не ведётся (None)."""
    art = tmp_path / "art"
    assert cli.main(_run_args(DATA, missing_file, "--run-id", "5", "--worker-build", SHA,
                              "--artifact-dir", str(art))) == 0
    assert run_env.calls[-1][1]["artifact_dir"] == str(art)
    assert cli.main(_run_args(DATA, missing_file, "--run-id", "5", "--worker-build", SHA)) == 0
    assert run_env.calls[-1][1]["artifact_dir"] is None


def test_stop_signal_sets_flag_and_restores_handlers(monkeypatch):
    """F1: SIGTERM и SIGINT только выставляют флаг остановки; прежние обработчики возвращаются."""
    import signal
    import threading
    stop = threading.Event()
    before = {s: signal.getsignal(s) for s in cli.STOP_SIGNALS}
    previous = cli._install_stop_handlers(stop)
    try:
        assert not stop.is_set()
        signal.raise_signal(signal.SIGTERM)
        assert stop.is_set()
    finally:
        cli._restore_handlers(previous)
    assert {s: signal.getsignal(s) for s in cli.STOP_SIGNALS} == before


# ---------------------------------------------------------------- наблюдатель heartbeat (SPEC 8.3 а)

WATCH_BASE = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc)


class FakeWatchClock:
    """Часы и сон для наблюдателя: сон двигает время, реального ожидания нет."""

    def __init__(self):
        self.t = 0.0
        self.sleeps = 0
        self.on_sleep = None

    def clock(self):
        return self.t

    def utc_now(self):
        return WATCH_BASE + timedelta(seconds=self.t)

    def sleep(self, sec):
        self.sleeps += 1
        if self.sleeps > 500:  # защита от зависания теста: наблюдатель должен был выйти раньше
            raise AssertionError("наблюдатель не завершился")
        self.t += sec
        if self.on_sleep is not None:
            self.on_sleep(self)


def _write_hb(path, at_sec):
    """Файл heartbeat с временем WATCH_BASE + at_sec (секунды относительно фейковых часов)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    when = WATCH_BASE + timedelta(seconds=at_sec)
    path.write_text(json.dumps({"utc": when.strftime(cli.HEARTBEAT_UTC_FORMAT), "label": "gen-123"}),
                    encoding="utf-8")


@pytest.fixture
def watch(tmp_path):
    kills = []
    clock = FakeWatchClock()

    def fake_kill(run_id):
        kills.append(run_id)
        return fake_kill.code

    fake_kill.code = cli.EXIT_OK
    return SimpleNamespace(hb=tmp_path / "generate-heartbeat" / "heartbeat.json",
                           clock=clock, kills=kills, kill=fake_kill)


def _run_watch(w, run_id="123"):
    return cli.watch_heartbeat(run_id, w.hb, max_silence_sec=120, check_every_sec=30,
                               kill=w.kill, clock=w.clock.clock, sleep=w.clock.sleep,
                               utc_now=w.clock.utc_now)


def test_watch_first_limit_constant_is_600():
    assert cli.WATCH_FIRST_HEARTBEAT_LIMIT_SEC == 600
    assert cli.WATCH_DEFAULT_SILENCE_SEC == 120
    assert cli.WATCH_DEFAULT_CHECK_SEC == 30


def test_watch_first_heartbeat_never_appears_gives_3_without_kill(watch):
    code = _run_watch(watch)
    assert code == 3
    assert watch.kills == []                 # до первой записи карт нет: уборка не запускается
    assert watch.clock.t == 600              # выход ровно по пределу, без реального ожидания


def test_watch_fresh_heartbeat_keeps_running_until_marker(watch):
    _write_hb(watch.hb, 0)

    def on_sleep(c):
        _write_hb(watch.hb, c.t)             # оркестратор пишет свежий файл
        if c.sleeps == 4:
            cli.finished_marker_path(watch.hb).write_text("finished\n", encoding="utf-8")

    watch.clock.on_sleep = on_sleep
    assert _run_watch(watch) == 0
    assert watch.kills == []
    assert watch.clock.sleeps == 4           # продолжал работать, пока не появилась метка


def test_watch_stale_heartbeat_kills_gen_run_and_keeps_running_until_marker(watch):
    _write_hb(watch.hb, -200)                # запись старше порога 120 с

    def on_sleep(c):                         # после уборки наблюдатель продолжает работать (F3)
        if c.sleeps == 3:
            cli.finished_marker_path(watch.hb).write_text("finished\n", encoding="utf-8")

    watch.clock.on_sleep = on_sleep
    assert _run_watch(watch) == 0            # выход только по метке конца запуска
    assert watch.kills == ["123"]            # одна уборка на эпизод тишины
    assert watch.clock.sleeps == 3


def test_watch_heartbeat_lost_after_first_write_kills(watch):
    _write_hb(watch.hb, 0)

    def on_sleep(c):
        if c.sleeps == 1:
            watch.hb.unlink()                # файл пропал после того, как появлялся
        if c.sleeps == 4:
            cli.finished_marker_path(watch.hb).write_text("finished\n", encoding="utf-8")

    watch.clock.on_sleep = on_sleep
    assert _run_watch(watch) == 0
    assert watch.kills == ["123"]            # пропажа файла: одна уборка, без повтора


def test_watch_kill_failure_keeps_observer_running_and_exits_0_at_marker(watch):
    """F3 + без G11: неудачная уборка не завершает наблюдателя; при выходе по метке код всегда 0."""
    watch.kill.code = cli.EXIT_RUNTIME
    _write_hb(watch.hb, -200)

    def on_sleep(c):
        if c.sleeps == 2:
            cli.finished_marker_path(watch.hb).write_text("finished\n", encoding="utf-8")

    watch.clock.on_sleep = on_sleep
    assert _run_watch(watch) == 0
    assert watch.kills == ["123"]
    assert watch.clock.sleeps == 2           # работал после уборки до метки


def test_watch_marker_exits_0_without_kill_even_if_stale(watch):
    _write_hb(watch.hb, -500)
    cli.finished_marker_path(watch.hb).write_text("finished\n", encoding="utf-8")
    assert _run_watch(watch) == 0
    assert watch.kills == []


def test_watch_marker_without_any_heartbeat_exits_0(watch):
    cli.finished_marker_path(watch.hb).parent.mkdir(parents=True, exist_ok=True)
    cli.finished_marker_path(watch.hb).write_text("finished\n", encoding="utf-8")
    assert _run_watch(watch) == 0
    assert watch.kills == []


def _write_real_hb(path, age_sec):
    path.parent.mkdir(parents=True, exist_ok=True)
    when = datetime.now(timezone.utc) - timedelta(seconds=age_sec)
    path.write_text(json.dumps({"utc": when.strftime(cli.HEARTBEAT_UTC_FORMAT)}), encoding="utf-8")


def _stop_after_kill(monkeypatch, hb):
    """F3: наблюдатель после уборки не выходит. Чтобы тест завершился, метку конца запуска ставит сама уборка."""
    real_kill = cli._watch_kill

    def kill_then_finish(run_id):
        code = real_kill(run_id)
        cli.finished_marker_path(hb).write_text("finished\n", encoding="utf-8")
        return code

    monkeypatch.setattr(cli, "_watch_kill", kill_then_finish)


# проверка каждую 1 с: реальный сон наблюдателя короткий (по умолчанию 30 с)
WATCH_ARGS_FAST = ["--check-every-sec", "1"]


def test_watch_main_stale_kills_through_kill_cards(capsys, kill_env, tmp_path, monkeypatch):
    hb = tmp_path / "heartbeat.json"
    _write_real_hb(hb, 300)
    _stop_after_kill(monkeypatch, hb)
    code = cli.main(["watch-heartbeat", "--heartbeat-path", str(hb), "--run-id", "123"] + WATCH_ARGS_FAST)
    assert code == 0                         # уборка прошла: выход по метке с кодом 0
    assert kill_env.labels == ["gen-123"]


def test_watch_main_real_kill_failure_gives_0_and_error_line(capsys, kill_env, tmp_path, monkeypatch):
    # без G11: неудачная уборка не меняет код выхода по метке (0); причина остаётся строкой ERROR в журнале
    kill_env.error = cli.VastError("Не удалось уничтожить экземпляры Vast: [11]")
    hb = tmp_path / "heartbeat.json"
    _write_real_hb(hb, 300)
    _stop_after_kill(monkeypatch, hb)
    code = cli.main(["watch-heartbeat", "--heartbeat-path", str(hb), "--run-id", "123"] + WATCH_ARGS_FAST)
    cap = capsys.readouterr()
    assert code == 0
    assert kill_env.labels == ["gen-123"]
    assert "Traceback" not in cap.err
    assert "не удалось" in cap.err.lower()


def test_watch_main_marker_exits_0_without_vast(capsys, kill_env, tmp_path):
    hb = tmp_path / "heartbeat.json"
    cli.finished_marker_path(hb).write_text("finished\n", encoding="utf-8")
    assert cli.main(["watch-heartbeat", "--heartbeat-path", str(hb), "--run-id", "123"]) == 0
    assert kill_env.labels == []


def test_watch_does_not_print_key_or_heartbeat_content(capsys, kill_env, tmp_path, monkeypatch):
    # канарейка: значение ключа в окружении и в содержимом файла heartbeat не должно попасть в вывод
    canary = "TESTKEY-12345"
    monkeypatch.setenv(VAST_KEY_ENV, canary)
    kill_env.error = cli.VastError("Не удалось уничтожить экземпляры Vast: [11]")
    hb = tmp_path / "heartbeat.json"
    when = datetime.now(timezone.utc) - timedelta(seconds=300)
    hb.write_text(json.dumps({"utc": when.strftime(cli.HEARTBEAT_UTC_FORMAT), "note": canary,
                              "cards": [{"public_ipaddr": "203.0.113.5"}]}), encoding="utf-8")
    _stop_after_kill(monkeypatch, hb)
    code = cli.main(["watch-heartbeat", "--heartbeat-path", str(hb), "--run-id", "123"] + WATCH_ARGS_FAST)
    cap = capsys.readouterr()
    assert code == 0                         # уборка не удалась, наблюдатель вышел по метке с кодом 0 (без G11)
    assert canary not in cap.out + cap.err
    assert "203.0.113.5" not in cap.out + cap.err


# ---------------------------------------------------------------- onstart пилота (этап 1.2)

PILOT_SHA = "0123456789abcdef0123456789abcdef01234567"


def test_pilot_onstart_fetches_script_at_sha_and_logs_on_rental():
    cmd = cli.pilot_onstart_cmd(PILOT_SHA, "SOTONATORE/Download")
    assert cmd.strip()
    assert f"https://raw.githubusercontent.com/SOTONATORE/Download/{PILOT_SHA}/Generate/worker/pilot_start.sh" in cmd
    assert "bash /opt/pilot/pilot_start.sh" in cmd
    assert f">> {cli.PILOT_ONSTART_LOG} 2>&1" in cmd
    assert cmd.count(cli.PILOT_ONSTART_LOG) == 2
    # COMFYUI_ARGS приходит через окружение карты, токенов в команде нет
    assert "COMFYUI_ARGS" not in cmd
    assert "token" not in cmd.lower()


@pytest.mark.parametrize("sha", ["", "main", "0" * 39, "0" * 41, "G" * 40, "A" * 40, "$GITHUB_SHA"])
def test_pilot_onstart_refuses_bad_sha(sha):
    with pytest.raises(ValueError):
        cli.pilot_onstart_cmd(sha, "SOTONATORE/Download")


@pytest.mark.parametrize("repo", ["", "Download", "a/b/c", "a/b;rm", "owner/repo name"])
def test_pilot_onstart_refuses_bad_repo(repo):
    with pytest.raises(ValueError):
        cli.pilot_onstart_cmd(PILOT_SHA, repo)


def test_cli_summary_average_label_says_what_is_measured(capsys):
    cli._print_summary(SimpleNamespace(avg_render_sec=12.34))
    out = capsys.readouterr().out
    assert "  среднее время клипа от отправки до выгрузки, с: 12.3" in out
    assert "рендера" not in out
    cli._print_summary(SimpleNamespace(avg_render_sec=0.0))
    out = capsys.readouterr().out
    assert "  среднее время клипа от отправки до выгрузки, с: нет данных" in out


# ---------------------------------------------------------------- проверка профиля и пресета до аренды (фаза E2)

@pytest.fixture
def tmp_profiles(tmp_path, monkeypatch):
    """Корень профилей и папка пресетов — временные; реальные папки не меняются."""
    root = tmp_path / "profiles"
    (root / "models").mkdir(parents=True)
    # профиль ltx25 копируется из репозитория (до подмены корня): проверка профиля проходит,
    # и тесты доходят до проверки пресета
    shutil.copytree(_config_loader.MODEL_PROFILES_DIR / "models" / "ltx25",
                    root / "models" / "ltx25")
    presets = tmp_path / "presets"
    presets.mkdir()
    monkeypatch.setattr(_config_loader, "MODEL_PROFILES_DIR", root)
    monkeypatch.setattr(_gemini_prompts, "PRESETS_DIR", presets)
    return SimpleNamespace(root=root, presets=presets)


def _complete_profile(root, name):
    folder = root / "models" / name
    (folder / "prompts").mkdir(parents=True)
    for rel in ("workflow.json", "bindings.json", "profile.yaml", "prompts/style.md"):
        (folder / rel).write_text("{}", encoding="utf-8")
    return folder


def test_check_builtin_profile_and_none_is_ok():
    assert cli._check_profile_and_preset("ltx25", "none") is None


def test_check_unknown_profile_is_input_error(capsys):
    assert cli._check_profile_and_preset("nope", "none") == cli.EXIT_INPUT
    err = capsys.readouterr().err
    assert "профиль модели «nope» не найден" in err and "Traceback" not in err


def test_check_incomplete_profile_is_input_error(capsys, tmp_profiles):
    folder = tmp_profiles.root / "models" / "half"
    (folder / "prompts").mkdir(parents=True)
    (folder / "profile.yaml").write_text("{}", encoding="utf-8")
    assert cli._check_profile_and_preset("half", "none") == cli.EXIT_INPUT
    err = capsys.readouterr().err
    assert "неполный" in err
    for name in ("workflow.json", "bindings.json", "prompts/style.md"):
        assert name in err


def test_check_complete_temp_profile_is_ok(tmp_profiles):
    _complete_profile(tmp_profiles.root, "full")
    assert cli._check_profile_and_preset("full", "none") is None


def test_check_preset_missing_is_input_error(capsys, tmp_profiles):
    assert cli._check_profile_and_preset("ltx25", "nope") == cli.EXIT_INPUT
    assert "Пресет «nope» не найден" in capsys.readouterr().err


def test_check_preset_empty_is_input_error(capsys, tmp_profiles):
    (tmp_profiles.presets / "empty.md").write_text("\n", encoding="utf-8")
    assert cli._check_profile_and_preset("ltx25", "empty") == cli.EXIT_INPUT
    assert "пуст" in capsys.readouterr().err


def test_check_preset_present_is_ok(tmp_profiles):
    (tmp_profiles.presets / "pz.md").write_text("текст пресета", encoding="utf-8")
    assert cli._check_profile_and_preset("ltx25", "pz") is None


def test_run_unknown_profile_stops_before_vast(capsys, run_env, missing_file):
    code = cli.main(_run_args(DATA, missing_file, "--model-profile", "nope"))
    cap = capsys.readouterr()
    assert code == cli.EXIT_INPUT
    assert "профиль модели «nope» не найден" in cap.err and "Traceback" not in cap.err
    assert run_env.calls == [] and FakeVast.instances == []


def test_run_missing_preset_stops_before_vast(capsys, run_env, missing_file):
    code = cli.main(_run_args(DATA, missing_file, "--preset", "nope"))
    cap = capsys.readouterr()
    assert code == cli.EXIT_INPUT
    assert "Пресет «nope» не найден" in cap.err and "Traceback" not in cap.err
    assert run_env.calls == [] and FakeVast.instances == []


def test_preset_reaches_prompt_config(capsys, run_env, missing_file, monkeypatch, tmp_profiles):
    (tmp_profiles.presets / "pz.md").write_text("текст пресета", encoding="utf-8")
    seen = []

    def fake_build(profile, batch_size, preset="none"):
        seen.append(preset)
        return ("CFG",)

    monkeypatch.setattr(cli, "_build_prompt_config", fake_build)
    code = cli.main(_run_args(DATA, missing_file, "--preset", "pz"))
    assert code == cli.EXIT_OK
    assert seen == ["pz"]


def test_build_prompt_config_carries_preset_field():
    cfg = cli._build_prompt_config("ltx25", 20, "none")
    assert cfg.preset == "none" and cfg.batch_size == 20


# --- фото (STEP 7, блок B2): правило кадров и проверка --photo-profile ---------------------------
def test_frame_rule_for_image_profile_is_none():
    assert cli._frame_rule_for("zimage_turbo") is None


def test_frame_rule_for_video_profile_keeps_ltx_rule():
    assert cli._frame_rule_for("ltx25") == {"kind": "8k+1"}


def test_photo_profile_validator_rejects_video_type(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "vidonly"
    folder.mkdir()
    (folder / "profile.yaml").write_text("type: video\n", encoding="utf-8")
    monkeypatch.setattr(cli, "profile_dir", lambda name: folder)
    monkeypatch.setattr(cli, "load_profile_type", lambda name: "video")
    assert cli._check_photo_profile("vidonly") == cli.EXIT_INPUT
    out = capsys.readouterr()
    assert "image" in out.out + out.err


# --- лимит проверки по типам: limit_by_type ------------------------------------------------------
VIDEO6 = [1, 2, 3, 4, 5, 6]
PHOTO6 = [7, 8, 9, 10, 11, 12]


def test_limit_by_type_even_limit_takes_half_of_each():
    assert cli.limit_by_type(VIDEO6, PHOTO6, 4) == [1, 2, 7, 8]


def test_limit_by_type_odd_limit_gives_extra_to_video():
    assert cli.limit_by_type(VIDEO6, PHOTO6, 5) == [1, 2, 3, 7, 8]


def test_limit_by_type_zero_gives_all():
    assert cli.limit_by_type(VIDEO6, PHOTO6, 0) == sorted(VIDEO6 + PHOTO6)


def test_limit_by_type_without_photo_takes_first_video():
    assert cli.limit_by_type(VIDEO6, [], 3) == [1, 2, 3]
