"""Офлайн-тесты логирования и диагностики (SPEC 0.1, 8.4, 10).

Проверяется: GET /logs воркера (токен, tail, буфер на 5000 строк, маскирование), строки о клипе
(номер, время, длительность, размер, без текста промпта), выгрузка логов карты до уничтожения
(в том числе при таймауте и ошибке), строки поиска с счётчиками, переходы состояний карты,
GEN_LOG_LEVEL (передача карте, код 2 при неверном значении).
Сеть не используется: MockTransport, фальшивые часы, ComfyUI подменён.
"""
from __future__ import annotations

import logging
import re
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from Generate import cli
    from Generate.core import gemini_prompts as gp
    from Generate.core import orchestrator as orch
    from Generate.core import vast_client as vc
    from Generate.tests import test_orchestrator as to
    from Generate.tests import test_worker_server as tws
    from Generate.tests.test_worker_server import (  # noqa: F401  (фикстура make_app)
        FakeComfy, TASK, TOKEN, WRONG, client_for, make_app, submit_done, wait_state)
    from Generate.worker import server
except ImportError:  # pragma: no cover - плоский импорт
    import cli  # type: ignore
    import orchestrator as orch  # type: ignore
    import vast_client as vc  # type: ignore
    import gemini_prompts as gp  # type: ignore
    import test_orchestrator as to  # type: ignore
    import test_worker_server as tws  # type: ignore
    from test_worker_server import (  # type: ignore  # noqa: F401
        FakeComfy, TASK, TOKEN, WRONG, client_for, make_app, submit_done, wait_state)
    import server  # type: ignore

PAYLOAD_LEN = len(tws.PAYLOAD_BYTES)
LOGS_TOKEN_MARK = "***"


# ---------------------------------------------------------------------------
# Воркер: GET /logs и буфер
# ---------------------------------------------------------------------------

def test_logs_refused_without_token_and_no_content(make_app):
    app, _ = make_app()
    with client_for(app, TOKEN) as good:
        good.get("/health")  # в буфере уже есть строки запуска
    with client_for(app, None) as none:
        resp = none.get("/logs")
        assert resp.status_code == 401
        assert "Воркер запущен" not in resp.text and TOKEN not in resp.text
    with client_for(app, WRONG) as bad:
        resp = bad.get("/logs")
        assert resp.status_code == 403
        assert "Воркер запущен" not in resp.text and WRONG not in resp.text


def test_logs_plain_text_tail_and_bad_tail(make_app):
    app, _ = make_app()
    with client_for(app) as c:
        resp = c.get("/logs")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert "Воркер запущен: уровень логов INFO" in resp.text
        lines = resp.text.splitlines()
        assert len(lines) >= 1 and all(line for line in lines)
        last_two = c.get("/logs?tail=2").text.splitlines()
        assert last_two == lines[-2:]
        assert c.get("/logs?tail=0").text == ""
        assert c.get("/logs?tail=-1").status_code == 400
        assert c.get("/logs?tail=abc").status_code == 400


def test_startup_and_shutdown_lines_in_buffer(make_app):
    app, _ = make_app()
    with client_for(app) as c:
        assert "Воркер запущен" in c.get("/logs").text
    app.close()
    assert any("Воркер останавливается" in line for line in app.logbuf.tail())


def test_deadman_trigger_is_logged(make_app):
    fired = threading.Event()
    app, _ = make_app(idle_timeout_seconds=0.2, check_interval=0.05, on_idle_timeout=fired.set)
    assert fired.wait(3)
    assert any("самоуничтожение" in line for line in app.logbuf.tail())


def test_http_request_lines_only_at_debug(make_app):
    app_info, _ = make_app()
    with client_for(app_info) as c:
        c.get("/health")
        assert "HTTP:" not in c.get("/logs").text
    app_debug, _ = make_app(log_level="DEBUG")
    with client_for(app_debug) as c:
        c.get("/health")
        assert "HTTP:" in c.get("/logs").text


def test_clip_lines_have_timing_size_and_no_prompt_text(make_app, tmp_path):
    app, _ = make_app()
    with client_for(app) as c:
        submit_done(c)
        body = c.get("/logs").text
    assert "Клип 7 принят в" in body
    assert "Клип 7: старт рендера" in body
    assert "Клип 7: конец рендера" in body
    assert re.search(r"длительность \d+\.\d с, размер %d байт" % PAYLOAD_LEN, body)
    assert re.search(r"sha256 [0-9a-f]{12}", body)
    assert f"промпт {len(TASK['prompt'])} симв." in body
    assert TASK["prompt"] not in body


def test_comfy_error_logged_fully_but_prompt_scrubbed(make_app, tmp_path):
    comfy = FakeComfy(tmp_path / "out", fail=f"сбой на тексте «{TASK['prompt']}» узел 6")
    app, _ = make_app(comfy)
    with client_for(app) as c:
        tid = c.post("/task", json=TASK).json()["task_id"]
        state = wait_state(c, tid, "error")
        body = c.get("/logs").text
    assert "узел 6" in body and "завершилась ошибкой" in body
    assert "[текст промпта скрыт]" in body
    assert TASK["prompt"] not in body and TASK["prompt"] not in state["error"]


def test_buffer_keeps_at_most_5000_and_drops_oldest():
    buf = server.LogBuffer()
    assert buf._lines.maxlen == 5000
    for i in range(5005):
        buf.add(f"строка {i}")
    assert len(buf) == 5000
    lines = buf.tail()
    assert lines[0] == "строка 5" and lines[-1] == "строка 5004"
    assert buf.tail(3) == ["строка 5002", "строка 5003", "строка 5004"]


def test_token_masked_in_worker_line_and_stdout_format():
    server.register_secret(TOKEN)
    record = logging.LogRecord("Generate.worker.server", logging.INFO, __file__, 1,
                               "утечка %s в строке", (TOKEN,), None)
    text = server.MaskingFormatter(server.LOG_FORMAT).format(record)
    assert TOKEN not in text and LOGS_TOKEN_MARK in text
    buf = server.LogBuffer()
    buf.add(f"Authorization: Bearer {TOKEN}")
    assert TOKEN not in buf.tail()[0]


@pytest.mark.parametrize("raw, expected", [
    (None, ("INFO", True)), ("", ("INFO", True)), ("  ", ("INFO", True)),
    ("INFO", ("INFO", True)), ("DEBUG", ("DEBUG", True)),
    ("debug", ("INFO", False)), ("TRACE", ("INFO", False)),
])
def test_worker_read_log_level(raw, expected):
    assert server.read_log_level(raw) == expected


@pytest.mark.parametrize("line, secret", [
    ("Authorization: Basic zzSECRETzz", "zzSECRETzz"),
    ("заголовок Bearer abcdefghijklmnop", "abcdefghijklmnop"),
    ("ключ sk-ABCDEFGH12345678 в строке", "sk-ABCDEFGH12345678"),
    ("gemini AIzaSyD1234567890abcdefghijk", "AIzaSyD1234567890abcdefghijk"),
    ("hf hf_abcdefghijklmnopqrstu", "hf_abcdefghijklmnopqrstu"),
    ("X-Worker-Token: qqSECRETqq", "qqSECRETqq"),
])
def test_mask_patterns_cover_key_shapes(line, secret):
    text = server.mask_secrets(line)
    assert secret not in text and LOGS_TOKEN_MARK in text


# ---------------------------------------------------------------------------
# Оркестратор: строки поиска, карт, выдачи; выгрузка логов до уничтожения
# ---------------------------------------------------------------------------

@pytest.fixture
def vast_key(monkeypatch):
    monkeypatch.setenv(vc.ENV_KEY_NAME, "TESTKEY-VAST-98765")


class LogsWorker(to.FakeWorker):
    """Воркер с GET /logs по контракту server.py: since=N и заголовок X-Log-Next.

    logs_mode: ok | timeout | error. fail_pulls: номера вызовов /logs (с 1), которые падают таймаутом.
    emit=True: каждый опрос задачи добавляет строку «L<n> рендер»; все такие строки в emitted.
    """

    def __init__(self, *args, logs_mode: str = "ok", emit: bool = False, fail_pulls=(),
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logs_mode = logs_mode
        self.logs_calls = 0
        self.since_calls = 0
        self.emit = emit
        self.fail_pulls = set(fail_pulls)
        self.buffer = ["[12:00:00] клип 1 готов", f"эхо токена {self.token} в строке карты"]
        self.emitted: list = []

    def _tick(self) -> None:
        if self.emit:
            line = f"L{len(self.emitted) + 1} рендер"
            self.emitted.append(line)
            self.buffer.append(line)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/logs":
            self.logs_calls += 1
            self.events.append("logs")
            if request.headers.get("X-Worker-Token") != self.token:
                return httpx.Response(401, text="")
            if self.logs_mode == "timeout" or self.logs_calls in self.fail_pulls:
                raise httpx.ReadTimeout(f"таймаут, токен {self.token}", request=request)
            if self.logs_mode == "error":
                return httpx.Response(500, text=f"внутренняя ошибка {self.token}")
            since = request.url.params.get("since")
            if since is None:
                body = self.buffer
            else:
                self.since_calls += 1
                body = self.buffer[int(since):]
            return httpx.Response(200, text="".join(line + "\n" for line in body),
                                  headers={"X-Log-Next": str(len(self.buffer))})
        if request.method == "GET" and request.url.path.startswith("/task/"):
            self._tick()
        return super().__call__(request)


@pytest.fixture
def logs_rig(tmp_path, monkeypatch, vast_key):
    monkeypatch.setattr(to, "FakeWorker", LogsWorker)

    def build(**kw):
        return to.Rig(tmp_path, monkeypatch, **kw)
    return build


def test_card_logs_fetched_masked_and_logged(logs_rig, caplog):
    caplog.set_level(logging.INFO)
    rig = logs_rig(missing=(1, 2))
    summary = rig.run()
    assert summary.exit_code == 0
    assert rig.worker.logs_calls >= 1
    assert "Логи карты" in caplog.text and "[карта " in caplog.text
    assert TOKEN not in caplog.text and "эхо токена ***" in caplog.text


def test_destroy_runs_after_logs_are_requested(logs_rig, caplog):
    caplog.set_level(logging.INFO)
    rig = logs_rig(missing=(1,))
    rig.orc._slots[1] = orch._CardSlot(
        index=1, generation=1, instance_id=101, machine_id=11, price=0.3,
        created=rig.clock(), state="ready", url="http://10.0.0.1:18000")
    rig.orc._cleanup()
    assert rig.events[-2:] == ["logs", "destroy_by_label"]
    assert rig.vast.label_destroyed == [to.LABEL]


@pytest.mark.parametrize("mode, marker", [("timeout", "ReadTimeout"), ("error", "_WorkerHttpError")])
def test_destroy_proceeds_when_logs_fail(logs_rig, caplog, mode, marker):
    caplog.set_level(logging.INFO)
    rig = logs_rig(missing=(1, 2), worker_kw={"logs_mode": mode})
    summary = rig.run()
    assert summary.exit_code == 0 and summary.cleanup_ok
    assert rig.vast.destroyed, "карта должна быть уничтожена, даже если логи не получены"
    assert "Не удалось получить логи карты" in caplog.text
    assert marker in caplog.text
    assert TOKEN not in caplog.text


def test_search_counts_add_up_and_no_per_offer_lines(monkeypatch, vast_key, caplog):
    caplog.set_level(logging.INFO)

    def bundle(i, **over):
        b = {"id": i, "dph_total": 0.5, "min_bid": None, "reliability2": 0.99,
             "inet_down": 3000, "gpu_ram": 32768, "gpu_name": "RTX 5090",
             "num_gpus": 1, "machine_id": 100 + i}
        b.update(over)
        return b

    bundles = [
        bundle(1),
        "мусор",
        {"dph_total": 0.5},                       # без id: не разобрано
        bundle(3, num_gpus=2),                    # не одиночная карта
        bundle(4, dph_total=0.2),                 # цена ниже минимума
        bundle(5, dph_total=1.2),                 # цена выше максимума
        bundle(6, reliability2=0.5),              # надёжность
        bundle(7, inet_down=100),                 # сеть
        bundle(8, gpu_ram=8192),                  # VRAM (8 ГБ < 24 ГБ)
        bundle(9, gpu_name="RTX 4090"),           # имя GPU
        bundle(2),
    ]
    flt = vc.OfferFilter(gpu_name="RTX 5090", min_price=0.35, max_price=0.9,
                         min_reliability=0.95, min_inet_mbps=2000, min_vram_gb=24)
    client = vc.VastClient(client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"offers": bundles}))))
    offers = client.search_offers(flt)
    assert sorted(o.offer_id for o in offers) == [1, 2]

    lines = [r.getMessage() for r in caplog.records if "Поиск предложений Vast" in r.getMessage()]
    assert len(lines) == 1, "одна строка на поиск"
    line = lines[0]
    nums = [int(n) for n in re.findall(r"\s(\d+)[,;]", line)]
    received, *rejected = nums
    accepted = int(re.search(r"подходит (\d+)\.", line).group(1))
    assert received == len(bundles) == 11
    assert sum(rejected) + accepted == received
    assert "не разобрано 2" in line and "не одиночная карта 1" in line
    assert "цена ниже минимума 1" in line and "цена выше максимума 1" in line
    assert "надёжность 1" in line and "сеть 1" in line and "VRAM 1" in line and "имя GPU 1" in line
    assert "Пропущено предложение" not in caplog.text  # номера предложений в логе не выводятся


def test_clip_duration_logged_per_clip_and_summary_fields(logs_rig, caplog):
    caplog.set_level(logging.INFO)
    rig = logs_rig(missing=(1, 2))
    summary = rig.run()
    assert len(re.findall(r"Клип \d выдан в Release \(рендер на карте \d+: \d+\.\d с\)",
                          caplog.text)) == 2
    assert summary.destroyed_cards >= 1
    assert summary.avg_render_sec > 0
    assert "уничтожено карт" in caplog.text and "среднее время клипа" in caplog.text


def test_card_state_transitions_logged_once(logs_rig, caplog):
    caplog.set_level(logging.INFO)
    rig = logs_rig(missing=(1,))
    rig.run()
    assert caplog.text.count("состояние Vast «running»") == 1
    assert caplog.text.count("healthcheck OK через") == 1
    assert caplog.text.count("создана, ждём загрузку образа") == 1
    assert "Арендована карта" in caplog.text


def test_card_gets_log_level_and_invalid_level_rejected(logs_rig):
    rig = logs_rig(missing=(1,), log_level="DEBUG")
    rig.run()
    assert rig.vast.created[0]["env"]["GEN_LOG_LEVEL"] == "DEBUG"
    with pytest.raises(gp.GeminiInputError):
        logs_rig(missing=(1,), log_level="TRACE")


def test_gen_log_level_invalid_gives_exit_2(monkeypatch, capsys):
    monkeypatch.setenv(cli.LOG_LEVEL_ENV, "TRACE")
    code = cli.main(["check"])
    assert code == 2
    err = capsys.readouterr().err
    assert "TRACE" not in err and "INFO и DEBUG" in err


def test_gen_log_level_debug_accepted_and_restored(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv(cli.LOG_LEVEL_ENV, "DEBUG")
    root = logging.getLogger()
    old = root.level
    try:
        code = cli.main(["check", "--srt", str(tmp_path / "нет.srt")])
        assert code == 2
        assert cli.LOG_LEVEL_ENV not in capsys.readouterr().err
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(old)


def test_logs_refused_when_port_unreachable_is_not_fatal_for_destroy(logs_rig, caplog):
    """Порт карты закрыт: соединение рвётся, уничтожение всё равно выполняется."""
    caplog.set_level(logging.INFO)
    rig = logs_rig(missing=(1,), worker_kw={"logs_mode": "timeout"})
    rig.orc._slots[2] = orch._CardSlot(
        index=2, generation=1, instance_id=202, machine_id=12, price=0.3,
        created=rig.clock(), state="ready", url="http://10.0.0.9:18001")
    rig.orc._cleanup()
    assert rig.vast.label_destroyed == [to.LABEL]
    assert "Не удалось получить логи карты 202: ReadTimeout." in caplog.text


# ---------------------------------------------------------------------------
# Периодическая выгрузка логов карт (SPEC 8.4): курсор since/X-Log-Next, без дублей
# ---------------------------------------------------------------------------

def _assert_each_emitted_line_printed_once(caplog, worker) -> None:
    msgs = [r.getMessage() for r in caplog.records if "[карта " in r.getMessage()]
    assert worker.emitted, "в тесте должны быть строки, появившиеся на карте"
    for line in worker.emitted:
        printed = [m for m in msgs if m.endswith(" " + line)]
        assert len(printed) == 1, f"строка «{line}» напечатана {len(printed)} раз(а)"


def test_periodic_pulls_print_each_line_once(logs_rig, caplog):
    caplog.set_level(logging.INFO)
    rig = logs_rig(missing=(1, 2, 3, 4, 5), worker_kw={"emit": True, "render_sec": 30.0})
    summary = rig.run()
    assert summary.exit_code == 0 and summary.cleanup_ok
    # несколько периодических выгрузок (раз в минуту) и финальная с курсором
    assert rig.worker.logs_calls >= 3 and rig.worker.since_calls >= 2
    _assert_each_emitted_line_printed_once(caplog, rig.worker)
    assert TOKEN not in caplog.text


def test_failed_pull_warns_and_run_continues(logs_rig, caplog):
    caplog.set_level(logging.INFO)
    rig = logs_rig(missing=(1, 2, 3, 4, 5),
                   worker_kw={"emit": True, "render_sec": 30.0, "fail_pulls": {1}})
    summary = rig.run()
    assert summary.exit_code == 0 and summary.completed == 5 and summary.cleanup_ok
    assert rig.vast.destroyed, "карта должна быть уничтожена, несмотря на сбой выгрузки"
    assert caplog.text.count("Не удалось получить логи карты") >= 1
    assert "ReadTimeout" in caplog.text
    # строки, не попавшие в первую выгрузку, напечатаны при следующей: без потерь и дублей
    _assert_each_emitted_line_printed_once(caplog, rig.worker)
    assert TOKEN not in caplog.text


def test_pull_without_cursor_header_prints_nothing(logs_rig, caplog):
    caplog.set_level(logging.INFO)
    rig = logs_rig(missing=(1,))
    rig.orc._slots[1] = orch._CardSlot(
        index=1, generation=1, instance_id=101, machine_id=11, price=0.3,
        created=rig.clock(), state="ready", url="http://10.0.0.1:18000")
    rig.worker.buffer.append("строка без курсора")
    original = rig.worker.__call__

    def no_header(request):
        resp = original(request)
        if request.url.path == "/logs":
            resp.headers.pop("X-Log-Next", None)
        return resp

    rig.http._transport = httpx.MockTransport(no_header)
    assert rig.orc._pull_card_logs(rig.orc._slots[1], final=False) is False
    assert rig.orc._slots[1].log_next is None
    assert "строка без курсора" not in caplog.text
    assert "_LogCursorError" in caplog.text


def test_periodic_pull_interval_and_stop_after_destroy(logs_rig):
    rig = logs_rig(missing=(1,), worker_kw={"emit": True})
    t0 = rig.clock()
    rig.orc._slots[1] = orch._CardSlot(
        index=1, generation=1, instance_id=101, machine_id=11, price=0.3,
        created=t0, state="ready", url="http://10.0.0.1:18000")
    rig.orc._pull_due_logs(t0)          # первая выгрузка
    rig.orc._pull_due_logs(t0 + 30)     # меньше минуты: не нужна
    assert rig.worker.logs_calls == 1
    rig.orc._pull_due_logs(t0 + 60)     # минута прошла: курсор передан в since
    assert rig.worker.logs_calls == 2 and rig.worker.since_calls == 1
    rig.orc._slots.pop(1)               # карта уничтожена: опросов больше нет
    rig.orc._pull_due_logs(t0 + 600)
    assert rig.worker.logs_calls == 2


def test_booting_card_is_not_polled_for_logs(logs_rig):
    rig = logs_rig(missing=(1,))
    rig.orc._slots[1] = orch._CardSlot(
        index=1, generation=1, instance_id=101, machine_id=11, price=0.3,
        created=rig.clock(), state="booting", url="http://10.0.0.1:18000")
    rig.orc._pull_due_logs(rig.clock())
    assert rig.worker.logs_calls == 0


# ---------------------------------------------------------------------------
# Воркер: первая строка «Сборка воркера: …», since и X-Log-Next в GET /logs
# ---------------------------------------------------------------------------

def test_build_line_is_first_in_logs(make_app):
    app, _ = make_app(build_id="abc1234")
    with client_for(app) as c:
        first = c.get("/logs").text.splitlines()[0]
    assert first == "Сборка воркера: abc1234"


def test_build_line_default_is_unknown_and_normalized(make_app):
    app, _ = make_app(build_id="  a\nb  ")
    with client_for(app) as c:
        assert c.get("/logs").text.splitlines()[0] == "Сборка воркера: a b"
    app2, _ = make_app()
    with client_for(app2) as c:
        assert c.get("/logs").text.splitlines()[0] == "Сборка воркера: unknown"


def test_logs_since_returns_only_new_lines_and_cursor(make_app):
    app, _ = make_app()
    with client_for(app) as c:
        full = c.get("/logs")
        n = int(full.headers["X-Log-Next"])
        assert n == len(full.text.splitlines())   # строка k имеет номер k
        submit_done(c)
        part = c.get(f"/logs?since={n}")
        new_next = int(part.headers["X-Log-Next"])
        lines = part.text.splitlines()
        assert len(lines) == new_next - n > 0
        assert any("Клип 7 принят" in line for line in lines)
        again = c.get(f"/logs?since={new_next}")
        assert again.text == "" and int(again.headers["X-Log-Next"]) == new_next


def test_logs_since_invalid_is_400(make_app):
    app, _ = make_app()
    with client_for(app) as c:
        assert c.get("/logs?since=-1").status_code == 400
        assert c.get("/logs?since=abc").status_code == 400


def test_logs_since_keeps_auth_behaviour(make_app):
    app, _ = make_app()
    with client_for(app, None) as none:
        resp = none.get("/logs?since=0")
        assert resp.status_code == 401 and "X-Log-Next" not in resp.headers
    with client_for(app, WRONG) as bad:
        resp = bad.get("/logs?since=0")
        assert resp.status_code == 403 and "X-Log-Next" not in resp.headers


def test_log_buffer_read_since_after_eviction():
    buf = server.LogBuffer(max_lines=3)
    for i in range(5):
        buf.add(f"строка {i}")
    assert buf.read(since=0) == (["строка 2", "строка 3", "строка 4"], 5)
    assert buf.read(since=4) == (["строка 4"], 5)
    assert buf.read(since=5) == ([], 5)
    assert buf.read(since=99) == ([], 5)
    assert buf.read(tail=2) == (["строка 3", "строка 4"], 5)
