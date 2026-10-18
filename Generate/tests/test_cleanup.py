"""Тесты уборки карт (cli.py, команда cleanup-vast, SPEC 8.3 г).

Всё внешнее подменено: Vast — FakeCleanupVast, статус запуска — функция-заглушка, gh не вызывается.
Сети и реального ключа нет.
"""
import json
import time
from types import SimpleNamespace

import pytest

from Generate import cli
from Generate.core.gemini_prompts import GeminiRuntimeError
from Generate.core.vast_client import InstanceInfo, VastInputError, VastRuntimeError

NOW = 1_800_000_000.0
YOUNG = NOW - 3600            # карта работает час
OLD = NOW - 7 * 3600          # карта работает семь часов (больше 6 часов)
YOUNG_AGE = 3600              # возраст карты, сек (1 час)
OLD_AGE = 7 * 3600            # возраст карты, сек (больше 6 часов)
VAST_TOKEN_ENV = "GEN_" + "VAST_API_KEY"
CANARY = "TESTKEY-" + "12345"


def _inst(iid, label, start, **extra):
    raw = {"start_date": start} if start is not None else {}
    raw.update(extra)
    return InstanceInfo(instance_id=iid, actual_status="running", intended_status="running",
                        label=label, raw=raw)


# ---------------------------------------------------------------- разбор меток

@pytest.mark.parametrize("label, run_id", [
    ("gen-123", "123"),
    ("pilot-456", "456"),
    ("gen-abc", None),
    ("pilot-", None),
    ("random", None),
    ("gen-", None),
    ("gen-12a", None),
    ("", None),
    (None, None),
])
def test_parse_card_label(label, run_id):
    assert cli.parse_card_label(label) == run_id


def test_pilot_label_with_newline_is_unparsed():
    # fullmatch: перевод строки в метке не пропускается (как у ^...$ без хвоста)
    assert cli.parse_card_label("pilot-1\nx") is None


# ---------------------------------------------------------------- решение по карте

@pytest.mark.parametrize("label, status, age, destroy", [
    ("gen-1", "completed", YOUNG_AGE, True),             # завершённый запуск -> уничтожить
    ("gen-1", "completed", OLD_AGE, True),
    ("gen-1", "in_progress", YOUNG_AGE, False),          # идёт и карта молодая -> оставить
    ("gen-1", "in_progress", OLD_AGE, True),             # идёт, но карта старше 6 ч -> уничтожить
    ("gen-1", "queued", YOUNG_AGE, False),
    ("gen-1", "waiting", YOUNG_AGE, False),
    ("gen-1", "requested", YOUNG_AGE, False),
    ("gen-1", "pending", YOUNG_AGE, False),
    ("gen-1", None, YOUNG_AGE, False),                   # статус не прочитан, карта молодая -> оставить
    ("gen-1", None, OLD_AGE, True),                      # статус не прочитан, карта старая -> уничтожить
    ("gen-1", "weird", YOUNG_AGE, False),                # неизвестный статус трактуется как не прочитанный
    ("pilot-7", "completed", YOUNG_AGE, True),
    ("pilot-7", "in_progress", YOUNG_AGE, False),
    ("gen-1", "in_progress", None, False),           # возраст неизвестен: предел по возрасту не срабатывает
    ("gen-abc", "completed", YOUNG_AGE, True),           # метка не разобрана -> уничтожить при любом статусе
    ("random", None, YOUNG_AGE, True),
    (None, None, YOUNG_AGE, True),
])
def test_decide_card(label, status, age, destroy):
    got, reason = cli.decide_card(label, status, age)
    assert got is destroy
    assert reason  # причина печатается, пустой не бывает


def test_max_age_is_six_hours():
    assert cli.MAX_AGE_SEC == 6 * 3600


def test_decision_reason_does_not_repeat_label_text():
    # причина не печатает текст метки: метки приходят из Vast
    _, reason = cli.decide_card("random-secret-label", None, YOUNG_AGE)
    assert "random-secret-label" not in reason


def test_card_age_reads_start_date():
    assert cli.card_age_sec(_inst(1, "gen-1", NOW - 100), NOW) == pytest.approx(100.0)


def test_card_age_missing_or_bad_field_is_none():
    assert cli.card_age_sec(_inst(1, "gen-1", None), NOW) is None
    assert cli.card_age_sec(_inst(1, "gen-1", "не число"), NOW) is None


def test_card_age_future_start_is_zero():
    assert cli.card_age_sec(_inst(1, "gen-1", NOW + 50), NOW) == 0.0


# ---------------------------------------------------------------- статус запуска через gh (подмена)

def test_gh_status_completed():
    calls = []

    def runner(args):
        calls.append(args)
        return json.dumps({"status": "completed", "conclusion": "success"})

    assert cli.gh_run_status("owner/repo", "123", runner=runner) == "completed"
    assert calls == [["run", "view", "123", "--repo", "owner/repo", "--json", "status,conclusion"]]


def test_gh_status_in_progress():
    assert cli.gh_run_status("o/r", "1", runner=lambda a: '{"status": "in_progress"}') == "in_progress"


@pytest.mark.parametrize("answer", ["не json", "[]", '{"conclusion": "x"}', '{"status": 5}'])
def test_gh_status_unreadable_answer_is_none(answer):
    assert cli.gh_run_status("o/r", "1", runner=lambda a: answer) is None


def test_gh_status_runner_error_is_none():
    def boom(args):
        raise GeminiRuntimeError("Команда gh завершилась с кодом 1: секретный текст")

    assert cli.gh_run_status("o/r", "1", runner=boom) is None


def _gh_failing(stderr_text):
    """Раннер gh, который падает так же, как GhCommandRunner: stderr попадает в сообщение исключения."""
    def runner(args):
        raise GeminiRuntimeError(f"Команда gh завершилась с кодом 1: {stderr_text}")
    return runner


@pytest.mark.parametrize("stderr_text", ["HTTP 404: Not Found", "run not found", "Run NOT FOUND"])
def test_gh_missing_run_is_missing(stderr_text):
    """Запуска нет (HTTP 404 или not found, без учёта регистра) -> статус missing."""
    assert cli.gh_run_status("o/r", "1", runner=_gh_failing(stderr_text)) == "missing"


@pytest.mark.parametrize("age, destroy", [(YOUNG_AGE, True), (OLD_AGE, True)])
def test_missing_run_card_destroyed_regardless_of_age(age, destroy):
    status = cli.gh_run_status("o/r", "1", runner=_gh_failing("HTTP 404: Not Found"))
    got, reason = cli.decide_card("gen-1", status, age)
    assert got is destroy
    assert "не найден" in reason


def test_classify_missing_is_finished():
    assert cli.classify_run_status("missing") == "finished"
    assert cli.classify_run_status("completed") == "finished"


@pytest.mark.parametrize("age, destroy", [(YOUNG_AGE, False), (OLD_AGE, True)])
def test_network_error_is_unknown_kept_until_six_hours(age, destroy):
    status = cli.gh_run_status("o/r", "1", runner=_gh_failing("network timeout"))
    assert status is None                       # не 404: статус не прочитан, а не missing
    got, _ = cli.decide_card("gen-1", status, age)
    assert got is destroy


def test_in_progress_young_card_is_kept():
    status = cli.gh_run_status("o/r", "1", runner=lambda a: '{"status": "in_progress"}')
    assert cli.decide_card("gen-1", status, YOUNG_AGE) == (False, "запуск 1 ещё выполняется")


def test_completed_card_is_destroyed():
    status = cli.gh_run_status("o/r", "1", runner=lambda a: '{"status": "completed"}')
    got, _ = cli.decide_card("gen-1", status, YOUNG_AGE)
    assert got is True


def test_label_rule_foo_kept_and_unparsed_gen_destroyed(cleanup_env):
    """Правило меток (подтверждено владельцем): чужая метка не трогается, gen-без-номера уничтожается."""
    FakeCleanupVast.instances = [_inst(1, "foo-1", YOUNG), _inst(2, "gen-abc", YOUNG)]
    result = cli.cleanup_cards(FakeCleanupVast(), run_status_fn=_status_fn({}), now=NOW,
                               out=lambda text: None)
    assert 1 not in FakeCleanupVast.destroyed
    assert 2 in FakeCleanupVast.destroyed
    assert result.destroyed == 1 and result.kept == 0


# ---------------------------------------------------------------- команда cleanup-vast (fake клиент)

class FakeCleanupVast:
    """Подмена VastClient: список карт, учёт уничтожений, отказ на заданных id."""
    instances: list = []
    fail_ids: set = set()
    destroyed: list = []
    closed: bool = False

    def __init__(self, *a, **k):
        pass

    def list_instances(self, label=None):
        return list(FakeCleanupVast.instances)

    def destroy_instance(self, iid):
        if iid in FakeCleanupVast.fail_ids:
            raise VastRuntimeError("Vast отклонил удаление экземпляра")
        FakeCleanupVast.destroyed.append(iid)
        return True

    def close(self):
        FakeCleanupVast.closed = True


@pytest.fixture(autouse=True)
def _reset_fake_vast():
    # состояние подмены — атрибуты класса: сбрасываем перед каждым тестом
    FakeCleanupVast.instances = []
    FakeCleanupVast.fail_ids = set()
    FakeCleanupVast.destroyed = []
    FakeCleanupVast.closed = False
    yield


@pytest.fixture
def cleanup_env(monkeypatch):
    monkeypatch.setattr(cli, "VastClient", FakeCleanupVast)
    return FakeCleanupVast


def _status_fn(table):
    """Статусы по номеру запуска; номер вне таблицы = статус не прочитан (None)."""
    asked = []

    def fn(run_id):
        asked.append(run_id)
        return table.get(run_id)

    fn.asked = asked
    return fn


MIXED_CARDS = [
    _inst(1, "gen-10", YOUNG),        # завершён -> уничтожить
    _inst(2, "gen-11", YOUNG),        # идёт, молодая -> оставить
    _inst(3, "gen-12", OLD),          # идёт, старая -> уничтожить
    _inst(4, "gen-13", YOUNG),        # статус не прочитан, молодая -> оставить
    _inst(5, "gen-14", OLD),          # статус не прочитан, старая -> уничтожить
    _inst(6, "pilot-15", YOUNG),      # пилот, завершён -> уничтожить
    _inst(7, "gen-abc", YOUNG),       # метка не разобрана -> уничтожить
    _inst(8, "other-tool", YOUNG),    # не наша карта: не трогаем
    _inst(9, None, YOUNG),            # без метки: не трогаем
]
MIXED_STATUS = {"10": "completed", "11": "in_progress", "12": "in_progress",
                "15": "completed"}   # 13, 14 не в таблице = не прочитан


def test_cleanup_destroys_and_keeps_the_right_cards():
    FakeCleanupVast.instances = MIXED_CARDS
    fn = _status_fn(MIXED_STATUS)
    result = cli.cleanup_cards(FakeCleanupVast(), run_status_fn=fn, now=NOW, out=lambda s: None)
    assert FakeCleanupVast.destroyed == [1, 3, 5, 6, 7]
    assert (result.kept, result.destroyed, result.failed, result.unclassified) == (2, 5, 0, 0)
    assert result.exit_code == 0


def test_cleanup_does_not_ask_status_for_unparsed_or_foreign_cards():
    FakeCleanupVast.instances = MIXED_CARDS
    fn = _status_fn(MIXED_STATUS)
    cli.cleanup_cards(FakeCleanupVast(), run_status_fn=fn, now=NOW, out=lambda s: None)
    assert "abc" not in fn.asked
    assert sorted(fn.asked) == sorted(["10", "11", "12", "13", "14", "15"])


def test_cleanup_foreign_cards_are_never_destroyed():
    FakeCleanupVast.instances = [_inst(8, "other-tool", YOUNG), _inst(9, None, OLD)]
    result = cli.cleanup_cards(FakeCleanupVast(), run_status_fn=_status_fn({}), now=NOW,
                               out=lambda s: None)
    assert FakeCleanupVast.destroyed == []
    assert result.exit_code == 0


def test_cleanup_all_destroys_ok_gives_0():
    FakeCleanupVast.instances = [_inst(1, "gen-10", YOUNG), _inst(2, "gen-11", YOUNG)]
    result = cli.cleanup_cards(FakeCleanupVast(), run_status_fn=_status_fn({"10": "completed", "11": "completed"}),
                               now=NOW, out=lambda s: None)
    assert result.exit_code == 0
    assert result.destroyed == 2


def test_cleanup_nothing_to_do_gives_0(capsys):
    FakeCleanupVast.instances = []
    result = cli.cleanup_cards(FakeCleanupVast(), run_status_fn=_status_fn({}), now=NOW,
                               out=print)
    assert result.exit_code == 0
    assert "убирать нечего" in capsys.readouterr().out


def test_cleanup_one_destroy_fails_gives_3_and_continues(monkeypatch):
    FakeCleanupVast.instances = [_inst(1, "gen-10", YOUNG), _inst(2, "gen-11", YOUNG),
                                 _inst(3, "gen-12", YOUNG)]
    FakeCleanupVast.fail_ids = {2}
    table = {"10": "completed", "11": "completed", "12": "completed"}
    result = cli.cleanup_cards(FakeCleanupVast(), run_status_fn=_status_fn(table), now=NOW,
                               out=lambda s: None)
    assert result.exit_code == 3
    assert result.failed == 1
    # остальные карты уничтожены: сбой одной не останавливает уборку
    assert FakeCleanupVast.destroyed == [1, 3]


def test_cleanup_card_without_id_is_unclassified_gives_3():
    FakeCleanupVast.instances = [_inst(0, "gen-10", YOUNG)]
    result = cli.cleanup_cards(FakeCleanupVast(), run_status_fn=_status_fn({"10": "completed"}),
                               now=NOW, out=lambda s: None)
    assert result.unclassified == 1
    assert result.exit_code == 3
    assert FakeCleanupVast.destroyed == []


def test_cleanup_output_has_card_ids_not_addresses(capsys):
    card = _inst(42, "gen-10", YOUNG, public_ipaddr="203.0.113.7", ssh_host="ssh.example.test")
    FakeCleanupVast.instances = [card]
    cli.cleanup_cards(FakeCleanupVast(), run_status_fn=_status_fn({"10": "completed"}),
                      now=NOW, out=print)
    out = capsys.readouterr().out
    assert "42" in out
    assert "203.0.113.7" not in out
    assert "ssh.example.test" not in out


def test_cleanup_command_exit_0_when_all_ok(capsys, cleanup_env, monkeypatch):
    FakeCleanupVast.instances = [_inst(1, "gen-10", time.time() - 7 * 3600)]
    monkeypatch.setattr(cli, "gh_run_status", lambda repo, rid, runner=None: "in_progress")
    code = cli.main(["cleanup-vast", "--repo", "owner/repo"])
    assert code == 0
    assert FakeCleanupVast.destroyed == [1]          # старая карта уничтожена, хоть запуск и идёт
    assert FakeCleanupVast.closed


def test_cleanup_command_exit_3_when_destroy_fails(capsys, cleanup_env, monkeypatch):
    FakeCleanupVast.instances = [_inst(1, "gen-10", time.time() - 3600)]
    FakeCleanupVast.fail_ids = {1}
    monkeypatch.setattr(cli, "gh_run_status", lambda repo, rid, runner=None: "completed")
    code = cli.main(["cleanup-vast", "--repo", "owner/repo"])
    assert code == 3
    assert "Traceback" not in capsys.readouterr().err


def test_cleanup_command_passes_repo_to_gh(cleanup_env, monkeypatch):
    FakeCleanupVast.instances = [_inst(1, "gen-10", time.time() - 3600)]
    seen = []

    def fake_status(repo, rid, runner=None):
        seen.append((repo, rid))
        return "completed"

    monkeypatch.setattr(cli, "gh_run_status", fake_status)
    cli.main(["cleanup-vast", "--repo", "owner/repo"])
    assert seen == [("owner/repo", "10")]


def test_cleanup_command_requires_repo(capsys, cleanup_env):
    FakeCleanupVast.instances = [_inst(1, "gen-10", time.time() - 7 * 3600)]
    assert cli.main(["cleanup-vast"]) == 2
    assert FakeCleanupVast.destroyed == []


def test_cleanup_command_writes_summary_file(tmp_path, cleanup_env, monkeypatch):
    FakeCleanupVast.instances = [_inst(1, "gen-10", time.time() - 3600), _inst(2, "gen-11", time.time() - 3600)]
    monkeypatch.setattr(cli, "gh_run_status",
                        lambda repo, rid, runner=None: "completed" if rid == "10" else "in_progress")
    summary = tmp_path / "summary.md"
    code = cli.main(["cleanup-vast", "--repo", "o/r", "--summary-file", str(summary)])
    text = summary.read_text(encoding="utf-8")
    assert code == 0
    assert "## Уборка карт Vast" in text
    assert "Уничтожено карт: 1" in text
    assert "Оставлено (запуск идёт или статус не прочитан): 1" in text


def test_cleanup_command_vast_client_error_is_runtime(capsys, monkeypatch):
    def boom(*a, **k):
        raise cli.VastRuntimeError("Vast API вернул HTTP 503: GET /api/v1/instances/")

    monkeypatch.setattr(cli, "VastClient", boom)
    assert cli.main(["cleanup-vast", "--repo", "o/r"]) == 3


def test_cleanup_command_missing_key_is_input_error(capsys, monkeypatch):
    def no_key(*a, **k):
        raise VastInputError("Не задан ключ Vast")

    monkeypatch.setattr(cli, "VastClient", no_key)
    # неверная конфигурация (нет ключа) -> код 2, до любого запроса к Vast
    assert cli.main(["cleanup-vast", "--repo", "o/r"]) == 2


def test_cleanup_output_does_not_contain_key_canary(capsys, cleanup_env, monkeypatch):
    monkeypatch.setenv(VAST_TOKEN_ENV, CANARY)
    FakeCleanupVast.instances = [_inst(1, "gen-10", time.time() - 3600, ssh_host="203.0.113.9")]
    monkeypatch.setattr(cli, "gh_run_status", lambda repo, rid, runner=None: "completed")
    cli.main(["cleanup-vast", "--repo", "o/r"])
    cap = capsys.readouterr()
    assert CANARY not in cap.out + cap.err
    assert "203.0.113.9" not in cap.out + cap.err


def test_cleanup_adds_no_environment_read_to_cli():
    # cleanup-vast не читает окружение: ключ Vast читает VastClient, gh получает GhCommandRunner
    text = open(cli.__file__, encoding="utf-8").read()
    assert text.count("os.environ") == 1
