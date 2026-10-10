"""Тесты cli.py: всё через main(argv), без сети и ключей (всё внешнее подменено)."""
import pathlib
from types import SimpleNamespace

import pytest

from Generate import cli
from Generate.core.srt_parser import parse_srt, srt_hash

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

    def __init__(self, *a, **k):
        self.closed = False
        FakeVast.instances.append(self)

    def close(self):
        self.closed = True


@pytest.fixture
def run_env(monkeypatch):
    """Подменяет всё внешнее для команды run; возвращает список вызовов оркестратора."""
    calls = []
    FakeVast.instances = []
    monkeypatch.setattr(cli, "_build_timing", lambda: "TIMING")
    monkeypatch.setattr(cli, "_build_prompt_config", lambda profile, brief: ("CFG", profile, brief))
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
                              "--model-profile", "ltx25", "--style-brief", "стиль",
                              "--prompts-path", "p.json"))
    out = capsys.readouterr().out
    assert code == 0
    assert "Итоги запуска" in out
    (args, kwargs), = run_env.calls
    assert args[1] == str(missing_file)
    assert args[2] == ("CFG", "ltx25", "стиль")
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
    assert kwargs["max_cards"] == 4


def test_run_default_offer_filter(capsys, run_env, missing_file):
    """Без флагов фильтр поиска карт получает значения по умолчанию."""
    assert cli.main(_run_args(DATA, missing_file)) == 0
    _, kwargs = run_env.calls[0]
    f = kwargs["offer_filter"]
    assert isinstance(f, cli.OfferFilter)
    assert f.gpu_name == "RTX 5090"
    assert f.min_price == 0.35
    assert f.max_price == 0.90
    assert f.min_reliability == 0.95
    assert f.min_inet_mbps == 2000.0


def test_run_custom_offer_filter(capsys, run_env, missing_file):
    """Флаги поиска карт доходят до run_generation через offer_filter."""
    code = cli.main(_run_args(DATA, missing_file,
                              "--gpu-name", "RTX 4090",
                              "--min-price-per-hour", "0.2",
                              "--max-price-per-hour", "0.6",
                              "--min-reliability", "0.9",
                              "--min-inet-mbps", "1500"))
    assert code == 0
    _, kwargs = run_env.calls[0]
    f = kwargs["offer_filter"]
    assert isinstance(f, cli.OfferFilter)
    assert f.gpu_name == "RTX 4090"
    assert f.min_price == 0.2
    assert f.max_price == 0.6
    assert f.min_reliability == 0.9
    assert f.min_inet_mbps == 1500.0


def test_run_default_card_params(capsys, run_env, missing_file):
    """Без флагов docker_image/disk_gb/silent_host_timeout_min получают значения по умолчанию."""
    assert cli.main(_run_args(DATA, missing_file)) == 0
    _, kwargs = run_env.calls[0]
    assert kwargs["docker_image"] == "ghcr.io/sotonatore/download/videogen-worker:latest"
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
    assert "os." + "environ" not in CLI_SOURCE
    assert "get" + "env" not in CLI_SOURCE
