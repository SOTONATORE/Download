"""Группы GitHub Actions для строк лога воркера: офлайн-тесты.

GET /logs подменяется httpx.MockTransport, сеть не используется. Проверяем только печать:
строки карты уходят в stderr одной группой, журнал файла получает их с префиксом «[карта N]»,
а маркеры групп в debug.log не попадают.
"""
import logging

import httpx

from Generate.core import log_setup as ls
from Generate.core import orchestrator as orch

CARD_URL = "http://10.0.0.1:18000"
CARD_NO = 101


class _FakeOrc:
    """Минимум от Orchestrator, нужный _pull_card_logs при final=True: HTTP-клиент и заглушки."""

    def __init__(self, text: str, cursor: str = "5"):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=text, headers={"X-Log-Next": cursor})
        self._http = httpx.Client(transport=httpx.MockTransport(handler))
        self._headers = {}
        self._artifact = None
        self._mask_remote = lambda t: t          # маскирование здесь не проверяется
        self._warn_card = lambda *a, **k: None   # предупреждения о сбоях здесь не нужны


def _slot():
    return orch._CardSlot(index=1, generation=1, instance_id=CARD_NO, machine_id=11, price=0.3,
                          created=0.0, state="ready", url=CARD_URL)


def _pull(text: str) -> bool:
    """Одна выгрузка с финальным режимом: без монитора весов, печатается вся пачка."""
    fake = _FakeOrc(text)
    return orch.Orchestrator._pull_card_logs(fake, _slot(), final=True)


def test_batch_of_three_lines_is_one_group(capsys):
    """Пачка из трёх строк: один открывающий маркер, три строки без префикса карты, один закрывающий."""
    assert _pull("first line\nsecond line\nthird line\n") is True
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "::group::карта 101: лог воркера (строк: 3)\n"
        "first line\n"
        "second line\n"
        "third line\n"
        "::endgroup::\n"
    )
    assert "[карта" not in captured.err


def test_empty_batch_prints_nothing(capsys):
    """Пустая пачка: ни маркеров, ни строк. Курсор при этом двигается как обычно."""
    fake = _FakeOrc("", cursor="7")
    assert orch.Orchestrator._pull_card_logs(fake, _slot(), final=True) is True
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_worker_line_starting_with_double_colon_is_not_a_command(capsys):
    """Строка, начинающаяся с «::», пишется с пробелом впереди: GitHub не должен читать её как команду."""
    _pull("::set-env name=X::1\n")
    err = capsys.readouterr().err.splitlines()
    assert err[1] == " ::set-env name=X::1"
    assert err[0].startswith("::group::") and err[-1] == "::endgroup::"


def test_orchestrator_lines_are_outside_groups(capsys):
    """Строки оркестратора не попадают внутрь группы: предупреждение до пачки и ошибка после неё
    печатаются вне маркеров, а внутри группы только строки воркера."""
    root = logging.getLogger()
    saved_level = root.level
    handlers, _ = ls.setup_logging("INFO", None)   # консоль пишет в stderr, как в боевом запуске
    try:
        log = logging.getLogger("Generate.orchestrator")
        log.warning("Оркестратор: предупреждение до пачки")
        _pull("worker one\nworker two\n")
        log.error("Оркестратор: ошибка после пачки")
    finally:
        ls.teardown_logging(handlers)
        root.setLevel(saved_level)

    lines = capsys.readouterr().err.splitlines()
    open_idx = lines.index("::group::карта 101: лог воркера (строк: 2)")
    close_idx = lines.index("::endgroup::")
    inside = lines[open_idx + 1:close_idx]
    assert inside == ["worker one", "worker two"]
    before = [ln for ln in lines[:open_idx] if "предупреждение до пачки" in ln]
    after = [ln for ln in lines[close_idx + 1:] if "ошибка после пачки" in ln]
    assert len(before) == 1 and len(after) == 1
    assert not any("Оркестратор" in ln for ln in inside)


def test_debug_log_keeps_prefixed_lines_and_no_markers(tmp_path, capsys):
    """debug.log: строки карты с префиксом «[карта N]», маркеров групп там нет; консоль не дублирует строки."""
    root = logging.getLogger()
    saved_level = root.level
    handlers, note = ls.setup_logging("DEBUG", str(tmp_path))
    try:
        assert note is None
        _pull("alpha\nbeta\n")
    finally:
        ls.teardown_logging(handlers)
        root.setLevel(saved_level)

    debug_text = (tmp_path / "run" / "debug.log").read_text(encoding="utf-8")
    assert "[карта 101] alpha" in debug_text
    assert "[карта 101] beta" in debug_text
    assert "::group::" not in debug_text and "::endgroup::" not in debug_text
    # в консоли строки есть только внутри группы (без префикса), логгер для них консоль не печатает
    err = capsys.readouterr().err
    assert "[карта" not in err
    assert err.count("::group::") == 1 and err.count("::endgroup::") == 1


# ---------------------------------------------------------------------------
# Фильтр «Итоги генерации» в generate.yml: группы воркера в сводку не попадают.
# Скрипт фильтра берётся из самого workflow, запускается локально; сети нет.
# ---------------------------------------------------------------------------

def _summary_filter_source() -> str:
    import re
    import textwrap
    from pathlib import Path

    import yaml

    wf = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "generate.yml"
    doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
    for job in doc["jobs"].values():
        for step in job.get("steps", []):
            run = step.get("run") or ""
            if "RAW_RE = re.compile" in run and "summary_filter" in run:
                m = re.search(r"cat > \"\$summary_filter\" <<'PY'\n(.*?)\n[ \t]*PY\n", run, re.S)
                assert m, "не найден текст скрипта фильтра сводки"
                return textwrap.dedent(m.group(1))
    raise AssertionError("фильтр сводки не найден в generate.yml")


def _run_summary_filter(tmp_path, text: str):
    import subprocess
    import sys

    script = tmp_path / "summary_filter.py"
    script.write_text(_summary_filter_source(), encoding="utf-8")
    summary = tmp_path / "summary.md"
    summary.write_text("", encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script), str(summary)], input=text,
                          capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout, summary.read_text(encoding="utf-8")


def test_summary_filter_skips_worker_group_but_keeps_stdout(tmp_path):
    """Строки между ::group:: и ::endgroup:: (и сами маркеры) в сводку не идут; в stdout они остаются."""
    text = (
        "2026-11-04 10:00:00,000 [INFO] Старт прогона\n"
        "::group::карта 101: лог воркера (строк: 2)\n"
        "строка воркера один\n"
        "строка воркера два\n"
        "::endgroup::\n"
        "2026-11-04 10:00:05,000 [INFO] Готово\n"
    )
    stdout, summary = _run_summary_filter(tmp_path, text)
    assert stdout == text
    assert summary == (
        "2026-11-04 10:00:00,000 [INFO] Старт прогона\n"
        "2026-11-04 10:00:05,000 [INFO] Готово\n"
    )
    assert "::group::" not in summary and "::endgroup::" not in summary


def test_summary_filter_group_stays_open_for_escaped_lines_and_resumes_after(tmp_path):
    """Строка внутри группы, начинающаяся с «::» (с пробелом впереди), не закрывает группу.
    После ::endgroup:: обычные строки снова попадают в сводку; сырые строки Vast по-прежнему режутся."""
    text = (
        "::group::карта 7: лог воркера (строк: 2)\n"
        "первая строка\n"
        " ::endgroup::\n"
        "всё ещё внутри группы\n"
        "::endgroup::\n"
        "2026-11-04 10:01:00,000 [INFO] Запрос Vast #3 GET /x\n"
        "тело запроса\n"
        "2026-11-04 10:01:01,000 [INFO] Итог\n"
    )
    _, summary = _run_summary_filter(tmp_path, text)
    assert summary == "2026-11-04 10:01:01,000 [INFO] Итог\n"
