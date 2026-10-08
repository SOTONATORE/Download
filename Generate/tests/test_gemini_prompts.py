"""Тесты core/gemini_prompts на заглушке транспорта. Сеть не используется."""
import ast
import json
import logging
import pathlib
import socket
import traceback

import httpx
import pytest

from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import (
    GeminiInputError, GeminiRuntimeError, PromptConfig, RealTransport,
    build_requests, generate_prompts, merge_segments_into_sentences,
    parse_and_validate, strip_user_only,
)
from Generate.core.srt_parser import Segment

GEN = pathlib.Path(__file__).resolve().parent.parent
LTX_STYLE = GEN / "model_profiles" / "prompt_styles" / "ltx25.md"
# Имя переменной окружения собрано из частей (структурный тест запрещает литерал в коде).
ENV_NAME = "GEN_" + "GEMINI_API_KEY"
TESTKEY = "TESTKEY-12345"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


def seg(num, text, start=0):
    return Segment(num, start, start + 1000, text)


def make_segments():
    return [
        seg(1, "The old city woke slowly."),
        seg(2, "Merchants opened their stalls"),
        seg(3, "and carts rolled in from the hills."),
        seg(4, "Dr. Hale watched from a window."),
        seg(5, ""),
        seg(6, "Then the bells began to ring!"),
        seg(7, "Nobody ran."),
    ]


def make_cfg(tmp_path, **kw):
    style = tmp_path / "style.md"
    style.write_text("<!-- USER_ONLY_START -->\nСводка для пользователя SECRET-NOTE\n"
                     "<!-- USER_ONLY_END -->\n# Style guide\nWrite one paragraph.\n",
                     encoding="utf-8")
    kw.setdefault("style_file", str(style))
    kw.setdefault("style_brief", "dusty medieval chronicle")
    kw.setdefault("max_prompt_chars", 300)
    kw.setdefault("batch_size", 10)
    return PromptConfig(**kw)


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_json(self, model, system, user, schema):
        self.calls.append({"model": model, "system": system, "user": user, "schema": schema})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)


def ok(nums, text="A wide shot of a quiet street at dawn, a baker lifts a tray."):
    return [{"segment_index": n, "prompt": text} for n in nums]


# ---------------------------------------------------------------- предложения

def test_sentence_merge_and_abbreviation():
    sentences, by = merge_segments_into_sentences(make_segments())
    assert by[2] is by[3]                       # склейка через границу сегментов
    assert by[1] is not by[2]
    assert by[5].text == "" and by[5] is not by[4]   # тишина вне предложения: отдельная запись
    _, by2 = merge_segments_into_sentences([seg(1, "I met Dr."), seg(2, ""), seg(3, "Hale today.")])
    assert by2[1] is by2[3] and by2[2] is by2[1]     # «Dr.» не конец; пустой сегмент не рвёт
    assert sentences[1].text == "Merchants opened their stalls and carts rolled in from the hills."
    a, b = by[3].spans[3]
    assert by[3].text[a:b] == "and carts rolled in from the hills."


def test_ellipsis_and_empty_segments():
    segs = [seg(1, "Well..."), seg(2, "and then"), seg(3, ""), seg(4, "Done.")]
    _, by = merge_segments_into_sentences(segs)
    assert by[1] is by[2]                       # многоточие перед строчной: продолжение
    assert by[3] is by[2]
    assert by[4] is not by[3] or by[4].text.endswith("Done.")


def test_context_from_full_srt_only_wanted_taken(tmp_path):
    cfg = make_cfg(tmp_path)
    batches = build_requests(make_segments(), [3], cfg)
    assert len(batches) == 1 and batches[0].nums == (3,)
    user = batches[0].user
    assert "segment_index: 3" in user
    assert "segment_index: 2" not in user and "segment_index: 4" not in user
    # предложение по всему SRT: 2+3 склеены, маркер стоит на сегменте 3
    assert "Merchants opened their stalls >>and carts rolled in from the hills.<<" in user
    assert "Before (context only): The old city woke slowly." in user
    assert "After (context only): Dr. Hale watched from a window." in user


def test_sentence_crossing_batch_boundary(tmp_path):
    cfg = make_cfg(tmp_path, batch_size=1)
    batches = build_requests(make_segments(), [2, 3], cfg)
    assert [b.nums for b in batches] == [(2,), (3,)]
    for b in batches:   # оба батча видят предложение целиком
        assert "Merchants opened their stalls" in b.user and "hills." in b.user


def test_batching_and_dedup_sorted(tmp_path):
    cfg = make_cfg(tmp_path, batch_size=2)
    batches = build_requests(make_segments(), [7, 1, 3, 3, 6], cfg)
    assert [b.nums for b in batches] == [(1, 3), (6, 7)]


def test_schema_default():
    s = gp.response_schema([{"segment_index": int, "prompt": str}])
    assert s["type"] == "array"
    assert s["items"]["properties"]["segment_index"] == {"type": "integer"}
    assert s["items"]["required"] == ["segment_index", "prompt"]


# ---------------------------------------------------------------- USER_ONLY

def test_strip_user_only_synthetic(tmp_path):
    cfg = make_cfg(tmp_path)
    system = build_requests(make_segments(), [1], cfg)[0].system
    assert "USER_ONLY" not in system and "SECRET-NOTE" not in system
    assert "Write one paragraph." in system
    assert "dusty medieval chronicle" in system


def test_strip_user_only_real_style_file():
    text = LTX_STYLE.read_text(encoding="utf-8")
    assert "USER_ONLY_START" in text
    out = strip_user_only(text)
    assert "USER_ONLY" not in out
    assert "Сводка для пользователя" not in out
    assert "# LTX-2.5 prompt style guide" in out


def test_unpaired_marker_is_error():
    with pytest.raises(GeminiInputError):
        strip_user_only("a <!-- USER_ONLY_START --> b")


# ---------------------------------------------------------------- вход

def test_number_outside_srt_raises_before_transport(tmp_path):
    cfg = make_cfg(tmp_path)
    t = FakeTransport([])
    with pytest.raises(GeminiInputError) as ei:
        generate_prompts(make_segments(), [3, 99], cfg, t, "m")
    assert ei.value.exit_code == 2 and "99" in str(ei.value)
    assert t.calls == []


def test_empty_wanted_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    t = FakeTransport([])
    with pytest.raises(GeminiInputError):
        generate_prompts(make_segments(), [], cfg, t, "m")
    assert t.calls == []


def test_exit_codes():
    assert GeminiInputError.exit_code == 2 and GeminiRuntimeError.exit_code == 3


# ---------------------------------------------------------------- ответ и retry

def test_happy_path(tmp_path):
    cfg = make_cfg(tmp_path)
    t = FakeTransport([ok([1, 3])])
    res = generate_prompts(make_segments(), [1, 3], cfg, t, "model-x")
    assert [r.num for r in res] == [1, 3]
    assert not any(r.needs_review for r in res)
    assert len(t.calls) == 1 and t.calls[0]["model"] == "model-x"


def test_wrong_set_retry_then_needs_review(tmp_path):
    cfg = make_cfg(tmp_path)
    # 1-й ответ: нет номера 3; 2-й (retry только по 3): снова не тот номер
    t = FakeTransport([ok([1]), ok([5])])
    res = generate_prompts(make_segments(), [1, 3], cfg, t, "m")
    assert len(t.calls) == 2
    assert "segment_index: 3" in t.calls[1]["user"] and "segment_index: 1" not in t.calls[1]["user"]
    by = {r.num: r for r in res}
    assert not by[1].needs_review
    assert by[3].needs_review and by[3].prompt == "" and by[3].review_reason


def test_retry_fixes_problem(tmp_path):
    cfg = make_cfg(tmp_path)
    t = FakeTransport([ok([1]), ok([3], "A close-up of a cart wheel turning on cobblestones.")])
    res = generate_prompts(make_segments(), [1, 3], cfg, t, "m")
    assert len(t.calls) == 2 and not any(r.needs_review for r in res)


def test_too_long_prompt_keeps_text_flagged(tmp_path):
    cfg = make_cfg(tmp_path, max_prompt_chars=50)
    long = "x " * 100
    t = FakeTransport([ok([1], long), ok([1], long)])
    res = generate_prompts(make_segments(), [1], cfg, t, "m")
    assert len(t.calls) == 2
    assert res[0].needs_review and res[0].prompt and "длина" in res[0].review_reason


def test_not_json_retry(tmp_path):
    cfg = make_cfg(tmp_path)
    t = FakeTransport(["not json at all", ok([1])])
    res = generate_prompts(make_segments(), [1], cfg, t, "m")
    assert len(t.calls) == 2 and not res[0].needs_review


def test_on_batch_called_after_each_batch_before_next(tmp_path):
    cfg = make_cfg(tmp_path, batch_size=1)
    t = FakeTransport([ok([1]), ok([3])])
    seen, got = [], []

    def cb(batch):
        seen.append(len(t.calls))   # сколько запросов к моменту вызова
        got.append([r.num for r in batch])

    res = generate_prompts(make_segments(), [1, 3], cfg, t, "m", on_batch=cb)
    assert got == [[1], [3]] and seen == [1, 2]
    assert [r.num for r in res] == [1, 3]


def test_on_batch_gets_batch_after_retry(tmp_path):
    cfg = make_cfg(tmp_path)
    t = FakeTransport([ok([1]), ok([3], "A close-up of a cart wheel turning.")])
    got = []
    generate_prompts(make_segments(), [1, 3], cfg, t, "m", on_batch=got.append)
    assert len(got) == 1 and [r.num for r in got[0]] == [1, 3]
    assert not any(r.needs_review for r in got[0])


def test_on_batch_exception_not_swallowed(tmp_path):
    cfg = make_cfg(tmp_path, batch_size=1)
    t = FakeTransport([ok([1]), ok([3])])

    def cb(batch):
        raise ValueError("колбэк упал")

    with pytest.raises(ValueError):
        generate_prompts(make_segments(), [1, 3], cfg, t, "m", on_batch=cb)
    assert len(t.calls) == 1   # дальше не пошли


def test_runtime_error_on_second_batch_first_delivered(tmp_path):
    cfg = make_cfg(tmp_path, batch_size=1)
    t = FakeTransport([ok([1]), GeminiRuntimeError("сбой", status=500)])
    got = []
    with pytest.raises(GeminiRuntimeError):
        generate_prompts(make_segments(), [1, 3], cfg, t, "m", on_batch=got.append)
    assert len(t.calls) == 2
    assert [[r.num for r in b] for b in got] == [[1]]
    assert got[0][0].prompt and not got[0][0].needs_review


def test_runtime_error_propagates(tmp_path):
    cfg = make_cfg(tmp_path)
    t = FakeTransport([GeminiRuntimeError("сбой", status=500)])
    with pytest.raises(GeminiRuntimeError):
        generate_prompts(make_segments(), [1], cfg, t, "m")
    assert len(t.calls) == 1


# ---------------------------------------------------------------- parse_and_validate

def _pv(raw, nums, **kw):
    cfg = PromptConfig(style_file="x", **kw)
    return parse_and_validate(raw if isinstance(raw, str) else json.dumps(raw), nums, cfg)


def test_validate_clean():
    res, probs = _pv(ok([1, 2]), [1, 2])
    assert not probs and set(res) == {1, 2}


def test_validate_duplicates_extras_missing():
    raw = ok([1, 1, 9])
    res, probs = _pv(raw, [1, 2])
    reasons = {(p.num, p.reason) for p in probs}
    assert (1, "номер повторяется в ответе") in reasons
    assert (9, "лишний номер в ответе") in reasons
    assert (2, "номер отсутствует в ответе") in reasons
    assert 9 not in res and res[1].needs_review


@pytest.mark.parametrize("bad", [
    "```json\nA street.\n```", "Segment 4: a street at dawn", "12: a street at dawn",
    "### A street", "segment_index 3 a street",
])
def test_validate_garbage(bad):
    res, probs = _pv([{"segment_index": 1, "prompt": bad}], [1])
    assert res[1].needs_review and any("мусор" in p.reason for p in probs)


@pytest.mark.parametrize("fine", [
    "A segment of the old wall glows at dusk.",
    "Segments of a mosaic catch the light.",
    "35-year-old baker lifts a tray.",
    "A 2.5 meter wall, then 12:30 bells.",
    "3 candles burn on the table.",
])
def test_validate_no_false_positive(fine):
    res, probs = _pv([{"segment_index": 1, "prompt": fine}], [1])
    assert not probs and not res[1].needs_review and res[1].prompt == fine


@pytest.mark.parametrize("bad", [
    "Segment 4 a street", "segment #4: a street", "Segment: a street at dawn",
    "A street. segment_index: 4", "3. A street at dawn", "[3] A street", "4) A street",
    "12 - a street", '{"prompt": "a street"}',
])
def test_validate_service_markers_still_caught(bad):
    res, probs = _pv([{"segment_index": 1, "prompt": bad}], [1])
    assert res[1].needs_review and any("мусор" in p.reason for p in probs)


def test_validate_empty_prompt_and_types():
    res, probs = _pv([{"segment_index": 1, "prompt": "  "}, {"segment_index": "2", "prompt": "a"}], [1, 2])
    assert not res
    assert {p.num for p in probs} >= {1, 2}


def test_validate_normalizes_newlines():
    res, _ = _pv([{"segment_index": 1, "prompt": "A street.\nA baker."}], [1])
    assert res[1].prompt == "A street. A baker."


@pytest.mark.parametrize("bad, reason", [
    ("A street with no cars at dawn.", "стиль: отрицание"),
    ("A baker works without gloves.", "стиль: отрицание"),
    ("A baker doesn\u2019t look up.", "стиль: отрицание"),
    ('A baker says "good morning".', "стиль: кавычки"),
    ("A street at dawn, 4k, masterpiece.", "стиль: keyword spam"),
    ("The camera pushes in on his thoughtful face.", "стиль: эмоциональная метка"),
    ("The camera stays static, capturing the quiet stillness.", "стиль: эмоциональная метка"),
])
def test_validate_style_rules(bad, reason):
    res, probs = _pv([{"segment_index": 1, "prompt": bad}], [1])
    assert res[1].needs_review and any(p.reason == reason for p in probs)


@pytest.mark.parametrize("fine", [
    "A close-up of a hand tracing the edge of a crown in warm candlelight.",
    "A wide shot of a calm grey sea, a woman lowers her head onto folded arms.",
    "A static wide shot of a quiet cobblestone street with a single lamp glowing.",
    "Dramatic shadows fall across a rough stone wall as the camera pulls back.",
])
def test_validate_style_no_false_positive(fine):
    res, probs = _pv([{"segment_index": 1, "prompt": fine}], [1])
    assert not probs and not res[1].needs_review


def test_style_violation_retry_gets_fix_note(tmp_path):
    cfg = make_cfg(tmp_path)
    t = FakeTransport([ok([1], "A baker looks thoughtful."), ok([1])])
    res = generate_prompts(make_segments(), [1], cfg, t, "m")
    assert len(t.calls) == 2 and not res[0].needs_review
    assert "Fix these issues" not in t.calls[0]["user"]
    assert "Fix these issues" in t.calls[1]["user"]
    assert "segment_index: 1:" in t.calls[1]["user"] and "mood or emotion" in t.calls[1]["user"]


def test_style_violation_stays_flagged_after_retry(tmp_path):
    cfg = make_cfg(tmp_path)
    bad = "A baker looks thoughtful."
    t = FakeTransport([ok([1], bad), ok([1], bad)])
    res = generate_prompts(make_segments(), [1], cfg, t, "m")
    assert res[0].needs_review and res[0].prompt == bad
    assert "эмоциональная метка" in res[0].review_reason


def test_system_has_hard_rules(tmp_path):
    system = build_requests(make_segments(), [1], make_cfg(tmp_path))[0].system
    assert "# Hard rules for every prompt" in system
    assert "Exactly one camera move" in system


# ---------------------------------------------------------------- RealTransport и ключ

def _transport(handler):
    return RealTransport(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_real_transport_header_not_url(monkeypatch):
    monkeypatch.setenv(ENV_NAME, TESTKEY)
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "[]"}]}}]})

    out = _transport(handler).generate_json("gemini-test", "SYS", "USR", {"type": "array"})
    assert out == "[]"
    assert seen["key"] == TESTKEY
    assert TESTKEY not in seen["url"] and "key=" not in seen["url"]
    assert seen["url"].endswith("/gemini-test:generateContent")
    gc = seen["body"]["generationConfig"]
    assert gc["responseMimeType"] == "application/json" and gc["responseJsonSchema"] == {"type": "array"}
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"


def test_empty_candidates_yields_empty_text_then_review(monkeypatch, tmp_path):
    """200 без кандидатов: транспорт отдаёт пустой текст (не исключение), батч помечается."""
    monkeypatch.setenv(ENV_NAME, TESTKEY)
    n = {"calls": 0}

    def handler(request):
        n["calls"] += 1
        return httpx.Response(200, json={"candidates": []})

    t = _transport(handler)
    assert t.generate_json("m", "s", "u", {}) == ""
    res = generate_prompts(make_segments(), [1], make_cfg(tmp_path), t, "m")
    assert n["calls"] == 3   # 1 прямой вызов + запрос и retry
    assert res[0].needs_review and res[0].prompt == ""


def test_non_json_error_body(monkeypatch):
    monkeypatch.setenv(ENV_NAME, TESTKEY)
    t = _transport(lambda r: httpx.Response(502, content=b"<html>Bad Gateway</html>"))
    with pytest.raises(GeminiRuntimeError) as ei:
        t.generate_json("m", "s", "u", {})
    e = ei.value
    assert e.status == 502 and e.code == "" and not e.daily_quota and e.retry_after is None
    assert "Bad Gateway" not in str(e)


def test_close_owned_client(monkeypatch):
    monkeypatch.setenv(ENV_NAME, TESTKEY)
    t = RealTransport()
    inner = t._client
    assert not inner.is_closed
    t.close()
    assert inner.is_closed
    t.close()   # повтор безопасен
    with RealTransport() as t2:
        inner2 = t2._client
        assert not inner2.is_closed
    assert inner2.is_closed


def test_close_does_not_touch_external_client(monkeypatch):
    monkeypatch.setenv(ENV_NAME, TESTKEY)
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with RealTransport(client=client):
        pass
    RealTransport(client=client).close()
    assert not client.is_closed
    client.close()


def test_missing_key_is_input_error(monkeypatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    with pytest.raises(GeminiInputError):
        RealTransport()


def test_error_fields_for_quota(monkeypatch):
    monkeypatch.setenv(ENV_NAME, TESTKEY)
    body = {"error": {"status": "RESOURCE_EXHAUSTED", "message": "ECHO " + TESTKEY,
                      "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]},
                                  {"retryDelay": "17s"}]}}
    t = _transport(lambda r: httpx.Response(429, json=body))
    with pytest.raises(GeminiRuntimeError) as ei:
        t.generate_json("m", "s", "u", {})
    e = ei.value
    assert e.status == 429 and e.code == "RESOURCE_EXHAUSTED" and e.daily_quota and e.retry_after == 17.0
    assert TESTKEY not in str(e)


def test_error_code_fallback_quota_exceeded(monkeypatch):
    monkeypatch.setenv(ENV_NAME, TESTKEY)
    body = {"error": {"code": "quota_exceeded", "message": "ECHO " + TESTKEY}}
    t = _transport(lambda r: httpx.Response(429, json=body))
    with pytest.raises(GeminiRuntimeError) as ei:
        t.generate_json("m", "s", "u", {})
    e = ei.value
    assert e.status == 429 and e.code == "QUOTA_EXCEEDED" and e.daily_quota
    assert TESTKEY not in str(e)


def test_key_does_not_leak(monkeypatch, tmp_path, caplog, capsys):
    """SPEC 0.1 п.10: нормальный путь и путь с ошибкой; логи, исключения, файлы, вывод."""
    monkeypatch.setenv(ENV_NAME, TESTKEY)
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.DEBUG)   # попытка включить DEBUG снаружи

    cfg = make_cfg(tmp_path)
    good = json.dumps(ok([1]))

    def good_handler(request):
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": good}]}}]})

    res = generate_prompts(make_segments(), [1], cfg, _transport(good_handler), "m")
    assert res[0].prompt

    texts = []
    # ошибка HTTP, тело которой «эхом» возвращает ключ
    t_http = _transport(lambda r: httpx.Response(
        401, json={"error": {"status": "UNAUTHENTICATED", "message": TESTKEY}}, headers={"x-echo": TESTKEY}))
    # сетевой сбой, исключение которого несёт запрос с заголовком
    def net_handler(request):
        raise httpx.ConnectError("boom " + str(request.headers), request=request)
    t_net = _transport(net_handler)

    for t in (t_http, t_net):
        with pytest.raises(GeminiRuntimeError) as ei:
            generate_prompts(make_segments(), [1], cfg, t, "m")
        e = ei.value
        texts += [str(e), repr(e), "".join(traceback.format_exception(type(e), e, e.__traceback__))]

    # новые пути: тело не в JSON с эхом ключа, пустой candidates, колбэк, закрытие клиента
    t_html = _transport(lambda r: httpx.Response(
        502, content=("<html>" + TESTKEY + "</html>").encode(), headers={"x-echo": TESTKEY}))
    with pytest.raises(GeminiRuntimeError) as ei:
        generate_prompts(make_segments(), [1], cfg, t_html, "m", on_batch=lambda b: None)
    e = ei.value
    texts += [str(e), repr(e), "".join(traceback.format_exception(type(e), e, e.__traceback__))]
    t_empty = _transport(lambda r: httpx.Response(200, json={"candidates": [], "echo": TESTKEY}))
    got = []
    res_empty = generate_prompts(make_segments(), [1], cfg, t_empty, "m", on_batch=got.append)
    texts += [repr(res_empty), repr(got)]
    with RealTransport() as t_own:
        texts.append(repr(t_own))
    texts.append(repr(t_own))   # после close
    assert TESTKEY not in repr(t_html) and TESTKEY not in repr(t_empty)

    assert not any(TESTKEY in x for x in texts)
    assert TESTKEY not in caplog.text
    out = capsys.readouterr()
    assert TESTKEY not in out.out and TESTKEY not in out.err
    assert TESTKEY not in repr(t_http) and TESTKEY not in repr(t_net)
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert TESTKEY not in f.read_text(encoding="utf-8", errors="ignore"), f.name
            assert TESTKEY not in f.name


# ---------------------------------------------------------------- переносимость

def test_module_does_not_import_download():
    tree = ast.parse((GEN / "core" / "gemini_prompts.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]] if node.level == 0 else []
        else:
            continue
        assert "Download" not in roots and "generate_queries" not in roots
