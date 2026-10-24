"""Офлайн-тесты логирования и диагностики (SPEC 0.1, 8.4, 10).

Проверяется: GET /logs воркера (токен, tail, буфер на 5000 строк, маскирование), строки о клипе
(номер, время, длительность, размер, без текста промпта), выгрузка логов карты до уничтожения
(в том числе при таймауте и ошибке), строки поиска с счётчиками, переходы состояний карты,
GEN_LOG_LEVEL (передача карте, код 2 при неверном значении).
Сеть не используется: MockTransport, фальшивые часы, ComfyUI подменён.
"""
from __future__ import annotations

import json
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
    from Generate.core import log_setup as ls
    from Generate.tests import test_orchestrator as to
    from Generate.tests import test_worker_server as tws
    from Generate.tests.test_worker_server import (  # noqa: F401  (фикстура make_app)
        FakeComfy, TASK, TOKEN, WRONG, client_for, make_app, submit_done, wait_state)
    from Generate.worker import server
except ImportError:  # pragma: no cover - плоский импорт
    import cli  # type: ignore
    import orchestrator as orch  # type: ignore
    import vast_client as vc  # type: ignore
    import log_setup as ls  # type: ignore
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
        assert resp.status_code == 401
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
    assert any("остановка воркера" in line for line in app.logbuf.tail())


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


def test_buffer_writes_segments_and_keeps_no_line_list_in_memory(tmp_path):
    seg_dir = tmp_path / "logs"
    buf = server.LogBuffer(log_dir=str(seg_dir))
    total = server.SEGMENT_LINES * 2 + 5
    for i in range(total):
        buf.add(f"строка {i}")
    assert len(buf) == total                       # строки не вытесняются
    names = sorted(p.name for p in seg_dir.glob("seg-*.log"))
    assert names == ["seg-000001.log", "seg-000002.log", "seg-000003.log"]
    assert not hasattr(buf, "_lines")              # списка всех строк в памяти нет
    assert all(len(v) <= server.SEGMENT_LINES for v in vars(buf._store).values()
               if isinstance(v, (list, dict, set)))
    assert buf.tail(3) == [f"строка {total - 3}", f"строка {total - 2}", f"строка {total - 1}"]
    assert buf.read(since=server.SEGMENT_LINES * 2 - 1)[0] == [f"строка {i}" for i in range(19999, total)]


def test_start_removes_previous_segments_and_numbers_from_one(tmp_path):
    seg_dir = tmp_path / "logs"
    seg_dir.mkdir()
    (seg_dir / "seg-000001.log").write_text("старая строка 1\n", encoding="utf-8")
    (seg_dir / "seg-000002.log").write_text("старая строка 2\n", encoding="utf-8")
    buf = server.LogBuffer(log_dir=str(seg_dir))
    assert list(seg_dir.iterdir()) == []                   # прошлый запуск удалён, подкаталогов не создано
    buf.add("новая строка")
    assert len(buf) == 1 and buf.tail() == ["новая строка"]  # номера строк с 1
    assert (seg_dir / "seg-000001.log").read_text(encoding="utf-8") == "новая строка\n"
    assert sorted(p.name for p in seg_dir.iterdir()) == ["seg-000001.log"]


def test_logs_since_9999_across_segment_boundary(make_app, tmp_path):
    app, _ = make_app()
    app.logbuf = server.LogBuffer(log_dir=str(tmp_path / "seg"))  # чистый буфер: номера строк известны
    for i in range(1, server.SEGMENT_LINES + 1):
        app.logbuf.add(f"строка {i}")                             # строка k имеет номер k
    app.logbuf.add(f"строка {server.SEGMENT_LINES + 1}")
    with client_for(app) as c:
        resp = c.get("/logs?since=9999")
        assert resp.text.splitlines() == ["строка 10000", "строка 10001"]  # на границе сегментов, без пропуска и повтора
        assert resp.headers["X-Log-Next"] == "10001"
        everything = c.get("/logs").text.splitlines()
    assert len(everything) == 10001 and everything[9999] == "строка 10000"
    assert (tmp_path / "seg" / "seg-000002.log").exists()


def test_masking_happens_before_segment_file_write(tmp_path):
    server.register_secret(CANARY)                 # так же регистрируются токен воркера и HF_TOKEN
    seg_dir = tmp_path / "mask"
    buf = server.LogBuffer(log_dir=str(seg_dir))
    buf.add(f"токен воркера {CANARY}, HF_TOKEN={CANARY}")
    written = (seg_dir / "seg-000001.log").read_text(encoding="utf-8")
    assert CANARY not in written and "***" in written   # на диск попадает уже замаскированная строка
    assert CANARY not in buf.tail()[0]


def test_token_masked_in_worker_line_and_stdout_format(tmp_path):
    server.register_secret(TOKEN)
    record = logging.LogRecord("Generate.worker.server", logging.INFO, __file__, 1,
                               "утечка %s в строке", (TOKEN,), None)
    text = server.MaskingFormatter(server.LOG_FORMAT).format(record)
    assert TOKEN not in text and LOGS_TOKEN_MARK in text
    buf = server.LogBuffer(log_dir=str(tmp_path / "logs"))
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
             "inet_up_cost": 0.001, "inet_down_cost": 0.001,
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
    assert len(re.findall(r"Клип \d выдан в Release \(рендер: карта \d+ поколение \d+ \(экземпляр \d+\), \d+\.\d с\)",
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


def test_booting_card_with_url_is_polled_for_logs(logs_rig):
    """C1 (этап 6c): карта в booting с адресом воркера отдаёт GET /logs (строки весов нужны правилам)."""
    rig = logs_rig(missing=(1,))
    rig.orc._slots[1] = orch._CardSlot(
        index=1, generation=1, instance_id=101, machine_id=11, price=0.3,
        created=rig.clock(), state="booting", url="http://10.0.0.1:18000")
    rig.orc._pull_due_logs(rig.clock())
    assert rig.worker.logs_calls == 1


def test_booting_card_without_url_is_not_polled_for_logs(logs_rig):
    """C2: без адреса воркера GET /logs не выполняется."""
    rig = logs_rig(missing=(1,))
    rig.orc._slots[1] = orch._CardSlot(
        index=1, generation=1, instance_id=101, machine_id=11, price=0.3,
        created=rig.clock(), state="booting", url="")
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
        assert resp.status_code == 401 and "X-Log-Next" not in resp.headers


def test_log_buffer_read_ignores_max_lines_and_reads_since(tmp_path):
    buf = server.LogBuffer(max_lines=3, log_dir=str(tmp_path / "logs"))  # параметр принимается и не используется: строки не вытесняются (этап 9в2)
    for i in range(5):
        buf.add(f"строка {i}")
    assert buf.read(since=0) == (["строка 0", "строка 1", "строка 2", "строка 3", "строка 4"], 5)
    assert buf.read(since=4) == (["строка 4"], 5)
    assert buf.read(since=5) == ([], 5)
    assert buf.read(since=99) == ([], 5)
    assert buf.read(tail=2) == (["строка 3", "строка 4"], 5)


# ---------------------------------------------------------------------------
# Маскирование журнала: режимы A и B, белый список, шаблоны по форме (SPEC 0.1, п. 14; этап 7в)
# Проверка статическая по составу тестов; прогон владельцем на GitHub Actions.
# ---------------------------------------------------------------------------

CANARY = "TESTKEY-12345"
SHAPE_TOKEN = "hf_abcdefghijklmnopqrstu"
MASK_MODES = ("A", "B")
ALLOWED_LINE = "max_tokens=512 tokenizer: gpt2 token_count=7 num_tokens=9"


def test_mask_mode_default_is_a_and_mask_text_follows_it():
    assert ls.MASK_MODE == "A"
    assert ls.mask_text("status=running") == ls.mask_text_with_mode("status=running", ls.MASK_MODE)


def test_unknown_mask_mode_is_rejected():
    with pytest.raises(ValueError):
        ls.mask_text_with_mode("TOKEN=x", "C")


def test_allowlist_is_frozen():
    """Белый список зафиксирован (28 имён, решение владельца 18.10.2026): любое изменение требует правки этого теста."""
    assert isinstance(ls.ALLOWED_NAMES, frozenset)
    assert tuple(sorted(ls.ALLOWED_NAMES)) == (
        "address", "component", "connIndex", "device", "dph", "event", "gpu", "hash", "id", "identity",
        "ip", "is_bid", "limit", "location", "max_tokens", "msg", "name", "num_tokens", "port", "protocol",
        "rc", "run_id", "size", "status", "target", "threshold", "token_count", "tokenizer",
    )
    assert len(ls.ALLOWED_NAMES) == 28


def test_mode_a_masks_key_suffix_and_keeps_name():
    assert ls.mask_text_with_mode("MY_SERVICE_KEY=abcdef123456 done", "A") == "MY_SERVICE_KEY=*** done"


def test_mode_a_plain_pair_is_not_masked():
    assert ls.mask_text_with_mode("status=running", "A") == "status=running"


def test_mode_a_keeps_allowlisted_names_in_any_case():
    assert ls.mask_text_with_mode(ALLOWED_LINE, "A") == ALLOWED_LINE
    assert ls.mask_text_with_mode("MAX_TOKENS=512", "A") == "MAX_TOKENS=512"


def test_mode_b_masks_plain_pair_and_keeps_name():
    assert ls.mask_text_with_mode("color=running", "B") == "color=***"


def test_mode_b_keeps_allowlisted_names():
    assert ls.mask_text_with_mode(ALLOWED_LINE, "B") == ALLOWED_LINE


def test_mode_b_masks_key_suffix():
    assert ls.mask_text_with_mode("MY_SERVICE_KEY=abcdef123456", "B") == "MY_SERVICE_KEY=***"


# Имя собирается из частей: тест структуры запрещает литерал имени ключа вне комментариев.
VAST_KEY_NAME = "GEN_" + "VAST_API_KEY"


def test_mode_b_allowlisted_status_and_ip_not_masked():
    text = "status=running ip=46.224.213.21"
    assert ls.mask_text_with_mode(text, "B") == text


def test_mode_b_allowlisted_name_with_capital_letter_matches_any_case():
    """connIndex в списке с заглавной буквой: совпадение без учёта регистра (CONNINDEX, connindex)."""
    text = "connIndex=5 CONNINDEX=6 connindex=7"
    assert ls.mask_text_with_mode(text, "B") == text


def test_mode_b_masks_secret_pair_not_in_allowlist():
    assert ls.mask_text_with_mode(f"{VAST_KEY_NAME}={CANARY}", "B") == f"{VAST_KEY_NAME}=***"


def test_mode_a_plain_allowlisted_pair_kept_and_secret_pair_masked():
    assert ls.mask_text_with_mode("status=running", "A") == "status=running"
    out = ls.mask_text_with_mode(f"{VAST_KEY_NAME}={CANARY}", "A")
    assert CANARY not in out and out == f"{VAST_KEY_NAME}=***"


@pytest.mark.parametrize("mode", MASK_MODES)
def test_value_shape_token_is_masked_in_both_modes(mode):
    out = ls.mask_text_with_mode(f"ответ {SHAPE_TOKEN} конец", mode)
    assert SHAPE_TOKEN not in out and "***" in out


@pytest.mark.parametrize("mode", MASK_MODES)
def test_value_shape_inside_allowlisted_pair_is_still_masked(mode):
    out = ls.mask_text_with_mode(f"max_tokens={SHAPE_TOKEN}", mode)
    assert SHAPE_TOKEN not in out and out.startswith("max_tokens=")


@pytest.mark.parametrize("shape", [
    "github_pat_ABCDEFGHIJKL1234",
    "ghp_ABCDEFGHIJ12",
    "AIzaSyD1234567890abcdefghijk",
    "sk-ABCDEFGH12345678",
    "Bearer abcdefghijklmnop",
])
@pytest.mark.parametrize("mode", MASK_MODES)
def test_known_key_shapes_are_masked(mode, shape):
    out = ls.mask_text_with_mode(f"значение {shape} конец", mode)
    assert shape not in out


def test_bearer_keeps_word_and_masks_token():
    out = ls.mask_text_with_mode("заголовок Bearer abcdefghijklmnop", "B")
    assert "abcdefghijklmnop" not in out and "Bearer ***" in out


def test_json_pairs_by_mode():
    text = '{"HF_TOKEN": "abc123def", "stage": "running", "max_tokens": "5"}'
    out_a = ls.mask_text_with_mode(text, "A")
    out_b = ls.mask_text_with_mode(text, "B")
    assert "abc123def" not in out_a and "abc123def" not in out_b
    assert "running" in out_a and "running" not in out_b
    assert '"max_tokens": "5"' in out_a and '"max_tokens": "5"' in out_b


@pytest.mark.parametrize("mode", MASK_MODES)
@pytest.mark.parametrize("line", [
    f"HF_TOKEN={CANARY}",
    f'"HF_TOKEN": "{CANARY}"',
    f"vast api_key: {CANARY}",
])
def test_canary_after_secret_name_is_masked(mode, line):
    assert CANARY not in ls.mask_text_with_mode(line, mode)


def test_canary_after_plain_name_is_masked_in_mode_b_only():
    """Режим A по замыслу не маскирует пару с обычным именем; режим B маскирует любую пару."""
    assert CANARY not in ls.mask_text_with_mode(f"zone={CANARY}", "B")


def test_remote_text_and_log_share_one_masking_function(monkeypatch):
    """_mask_remote и журнал дают один результат: общая функция mask_remote_text (SPEC 0.1, п. 14)."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    sample = (f"ответ hf_abcdefghijklmnopqrstu\n"
              f"{VAST_KEY_NAME}={CANARY} status=running\n"
              f"ip=46.224.213.21 max_tokens=512")
    remote = orch.Orchestrator.__new__(orch.Orchestrator)  # без конструктора: нужны только _token и vast
    remote._token = ""
    remote._card_addresses = set()
    remote.vast = None
    out = remote._mask_remote(sample)
    assert out == ls.mask_remote_text(sample) == ls.mask_text(sample)
    assert "hf_abcdefghijklmnopqrstu" not in out and CANARY not in out
    assert "status=running" in out and "ip=46.224.213.21" in out and "max_tokens=512" in out


# ---------------------------------------------------------------------------
# Чёрный список имён (этап 7д): маскируется всегда, в обоих режимах, раньше белого списка
# ---------------------------------------------------------------------------

def test_blocked_names_are_frozen():
    assert isinstance(ls.BLOCKED_NAMES, frozenset)
    assert tuple(sorted(ls.BLOCKED_NAMES)) == ("crisp_hmac", "hmac")
    assert len(ls.BLOCKED_NAMES) == 2


def test_blocked_and_allowed_names_do_not_intersect():
    blocked = {n.lower() for n in ls.BLOCKED_NAMES}
    allowed = {n.lower() for n in ls.ALLOWED_NAMES}
    assert not (blocked & allowed)


@pytest.mark.parametrize("mode", MASK_MODES)
@pytest.mark.parametrize("line, expected", [
    (f"crisp_hmac={CANARY}", "crisp_hmac=***"),
    (f'"crisp_hmac": "{CANARY}"', '"crisp_hmac": "***"'),
    (f"hmac={CANARY}", "hmac=***"),
])
def test_blocked_pairs_are_masked_in_both_modes(mode, line, expected):
    out = ls.mask_text_with_mode(line, mode)
    assert out == expected
    assert CANARY not in out


def test_blocked_name_is_masked_by_remote_path():
    out = ls.mask_remote_text(f"crisp_hmac={CANARY}")
    assert out == "crisp_hmac=***" and CANARY not in out


# ---------------------------------------------------------------------------
# Этап 9а: поля экземпляра, значения токенов, _err
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_known_secrets_in_this_module():
    yield
    ls.set_known_secrets([])


def test_remove_instance_fields_removes_named_fields_at_any_depth():
    """9а п. 1: адрес, порты, ssh, env, extra_env, ключ экземпляра и любые ключи с ssh, ipaddr, host_ip, port
    удаляются на любой глубине; прочие поля остаются; исходный объект не меняется."""
    body = {
        "id": 7, "label": "gen-1", "actual_status": "running", "dph_total": 0.3,
        "public_ipaddr": "203.0.113.77", "direct_port_start": 47123, "direct_port_end": 47123,
        "direct_port_count": 1, "ports": {"18188/tcp": [{"HostPort": "47123"}]},
        "ssh_host": "ssh.example", "ssh_port": 39871, "env": {"A": "1"}, "extra_env": [["A", "1"]],
        "instance_api_key": "k", "local_ipaddrs": "10.0.0.2", "host_ip": "1.1.1.1", "SSH_KEY_PATH": "/x",
        "offers": [{"id": 1, "public_ipaddr": "1.2.3.4", "dph_total": 0.3}],
        "nested": {"deep": {"ports_x": 1, "status": "ok"}},
    }
    out = vc.remove_instance_fields(body)
    assert out == {
        "id": 7, "label": "gen-1", "actual_status": "running", "dph_total": 0.3,
        "offers": [{"id": 1, "dph_total": 0.3}], "nested": {"deep": {"status": "ok"}},
    }
    assert body["env"] == {"A": "1"} and body["public_ipaddr"] == "203.0.113.77"   # исходный объект цел


def test_debug_body_for_log_removes_instance_fields_and_keeps_the_rest():
    raw = json.dumps({"instances": [{
        "id": 7, "public_ipaddr": "203.0.113.77", "ssh_host": "ssh.example",
        "env": {"WORKER_TOKEN": CANARY}, "extra_env": [["WORKER_TOKEN", CANARY]],
        "status_msg": "ok", "dph_total": 0.3}]})
    out = vc.debug_body_for_log(raw)
    for needle in ("203.0.113.77", "ssh.example", CANARY, '"env"', "extra_env", "public_ipaddr"):
        assert needle not in out
    assert '"status_msg": "ok"' in out and '"dph_total": 0.3' in out and '"id": 7' in out


def test_known_secret_without_name_or_shape_is_masked_in_log_and_remote_text():
    """9а п. 4: значение, переданное оркестратором, маскируется в свободном тексте без имени и без известной формы."""
    ls.set_known_secrets([CANARY])
    line = f"в ответе карты стоит {CANARY} без имени"
    assert CANARY not in ls.mask_text(line) and "***" in ls.mask_text(line)
    assert CANARY not in ls.mask_remote_text(line)


def test_known_secrets_ignore_empty_none_and_short_values():
    """9а п. 4: пустые значения и значения короче 8 символов игнорируются; длиннее 8 маскируются."""
    ls.set_known_secrets(["", "   ", None, "abc1234"])          # 7 символов: не маскируется
    assert ls.mask_text("x abc1234 y") == "x abc1234 y"
    ls.set_known_secrets(["abc12345"])                         # 8 символов: маскируется
    assert ls.mask_text("x abc12345 y") == "x *** y"


def test_set_known_secrets_replaces_previous_values():
    ls.set_known_secrets([CANARY])
    ls.set_known_secrets(["OTHER-VALUE-99"])
    assert ls.mask_text(CANARY) == CANARY                      # прежнее значение больше не маскируется
    assert ls.mask_text("OTHER-VALUE-99") == "***"


def test_err_prints_masked_text_only(capsys):
    """9а п. 5: _err пропускает текст через маску; текст не обрезается."""
    ls.set_known_secrets([CANARY])
    cli._err(f"Ошибка: сервер вернул значение {CANARY} и HF_TOKEN={CANARY} в конце сообщения")
    err = capsys.readouterr().err
    assert CANARY not in err and "Ошибка: сервер вернул значение" in err and "в конце сообщения" in err


def test_canary_absent_from_debug_log_file_written_by_cli(tmp_path, monkeypatch):
    """9а п. 10 (e): run/debug.log, который пишет cli через файловый обработчик, не содержит канарейку ни
    с именем, ни без имени. Значения токенов передаёт маске оркестратор (set_known_secrets), как в __init__."""
    import os
    canary = "TESTKEY-12345"
    monkeypatch.setenv("WORKER_TOKEN", canary)
    monkeypatch.setenv("HF_TOKEN", canary)
    ls.set_known_secrets([os.environ["WORKER_TOKEN"], os.environ["HF_TOKEN"]])
    art = tmp_path / "art"
    handlers, note = ls.setup_logging("DEBUG", str(art))
    try:
        log = logging.getLogger("Generate.test_canary")
        log.debug("без имени: %s", canary)
        log.info("WORKER_TOKEN=%s HF_TOKEN=%s", canary, canary)
        log.info('{"WORKER_TOKEN": "%s"}', canary)
    finally:
        ls.teardown_logging(handlers)
        ls.set_known_secrets([])
    assert note is None
    text = (art / "run" / "debug.log").read_text(encoding="utf-8")
    assert text.strip(), "файл журнала пуст: проверка значима только при записанных строках"
    assert canary not in text
    assert "***" in text


def test_detail_line_stays_out_of_console_but_reaches_file(tmp_path, monkeypatch):
    """Этап 7в (F3): подробная строка («[setup] ...», «проверка /health») не попадает в консоль, но есть в файле;
    ключевое событие (extra=KEY_EVENT_EXTRA) и обычная ошибка попадают в консоль."""
    import io
    console = io.StringIO()
    monkeypatch.setattr(sys, "stderr", console)
    art = tmp_path / "art"
    handlers, note = ls.setup_logging("DEBUG", str(art))
    try:
        log = logging.getLogger("Generate.test_console_split")
        log.info("[карта 500] [setup] [1/5] Начинаю загрузку: m.safetensors")
        log.info("карта 500: проверка /health: код 200, ответ ok, за 0.1 с, готов.")
        log.warning("карта 500: проверка /health: запрос не дошёл (ConnectError).")
        log.info("Веса ltx25 начало загрузки", extra=ls.KEY_EVENT_EXTRA)
        log.info("карта 500: Веса ltx25 10%", extra=ls.KEY_EVENT_EXTRA)
        log.warning("Не удалось получить логи карты 500: ConnectError.")
    finally:
        ls.teardown_logging(handlers)
    out = console.getvalue()
    assert "[setup]" not in out and "проверка /health" not in out
    assert "Веса ltx25 начало загрузки" in out
    assert "карта 500: Веса ltx25 10%" in out
    assert "Не удалось получить логи карты 500" in out
    text = (art / "run" / "debug.log").read_text(encoding="utf-8")
    assert "[setup] [1/5] Начинаю загрузку" in text
    assert "проверка /health: код 200" in text and "проверка /health: запрос не дошёл" in text
    assert "Веса ltx25 начало загрузки" in text
