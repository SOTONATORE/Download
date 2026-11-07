"""Журнал генерации промптов Gemini: офлайн-тесты на заглушке транспорта. Сеть не используется.

Проверяем только строки журнала: что они появляются и что в них нет текста промпта и ключа API.
Ответы и исключения генерации проверяются существующими тестами test_gemini_prompts.py.
"""
import json
import logging

import pytest

from Generate.core.gemini_prompts import GeminiRuntimeError, PromptConfig, generate_prompts
from Generate.core.srt_parser import Segment

LOGGER = "Generate.gemini_prompts"
# Условная строка «ключа»: в журнале её быть не должно. Ключ в тестах только как проверочная строка.
FAKE_KEY = "TESTKEY-LOGS-98765"
# Уникальный маркер внутри текста промпта: если он попадёт в журнал, тест упадёт.
PROMPT_MARKER = "MARKERSCENE7Q"


class StubTransport:
    """Заглушка GeminiTransport: отдаёт заранее заданные ответы по очереди; исключение поднимает."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate_json(self, model, system, user, schema):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)


def _segments():
    return [
        Segment(1, 0, 1000, "A woman walks along a wet street."),
        Segment(2, 1000, 2000, "Rain falls on a harbor filled with small boats."),
    ]


def _cfg(tmp_path, **kw):
    style = tmp_path / "style.md"
    style.write_text("Warm light.", encoding="utf-8")
    base = dict(style_file=str(style), media="video", batch_size=2)
    base.update(kw)
    return PromptConfig(**base)


def _clean_reply():
    """Чистый ответ на два сегмента: без отрицаний, кавычек и эмоциональных слов."""
    return [
        {"segment_index": 1, "prompt": f"A woman walks along a wet street, {PROMPT_MARKER}, camera holds steady."},
        {"segment_index": 2, "prompt": "Rain falls on a harbor filled with small wooden boats, camera holds steady."},
    ]


def _messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER]


def test_successful_batch_logs_start_success_and_each_prompt(tmp_path, caplog):
    """Успешный батч: начало прогона, начало батча, принятие каждого промпта, итог батча и прогона."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    transport = StubTransport([_clean_reply()])

    results = generate_prompts(_segments(), [1, 2], _cfg(tmp_path), transport, "model-x")

    assert [r.num for r in results] == [1, 2]
    assert transport.calls == 1
    msgs = _messages(caplog)
    # начало прогона: вид, число клипов, батчей и ключей
    assert any(m.startswith("=== Генерация ") and ": клипов 2, батчей 1, ключей " in m for m in msgs)
    # начало батча: номер и число элементов
    assert any("Батч 1 из 1: начало, клипов в батче: 2." in m for m in msgs)
    # каждый принятый промпт: номер клипа, длина в символах, короткий хеш из 12 символов
    accepted = [m for m in msgs if m.startswith("Промпт принят:")]
    assert len(accepted) == 2
    assert accepted[0].startswith("Промпт принят: клип 1, ")
    assert accepted[1].startswith("Промпт принят: клип 2, ")
    for m in accepted:
        assert " символов, sha256 " in m
        digest = m.rsplit("sha256 ", 1)[1].rstrip(".")
        assert len(digest) == 12 and all(c in "0123456789abcdef" for c in digest)
    # успех батча: число промптов с текстом и время
    assert any(m.startswith("Батч 1: получено промптов с текстом: 2 из 2, за ") and m.endswith(" с.")
               for m in msgs)
    # итог прогона: готово, без текста, требуют проверки, общее время
    assert any(m.startswith("=== Генерация ") and "-промптов готова: готово 2, без текста 0, за " in m
               for m in msgs)


def test_failed_attempt_logs_batch_attempt_and_reason_without_key(tmp_path, caplog):
    """Неудачный запрос: строка с номером батча, номером попытки, классом ошибки и безопасной причиной.
    Ключ API в журнале не появляется: заглушка его не содержит, а текст ошибки безопасен."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    err = GeminiRuntimeError("Gemini вернул ошибку: HTTP 400, код INVALID_ARGUMENT.", status=400,
                             code="INVALID_ARGUMENT")
    transport = StubTransport([err])

    # ошибка не-временная: генерация её пробрасывает, журнал должен успеть записать попытку
    with pytest.raises(GeminiRuntimeError):
        generate_prompts(_segments(), [1, 2], _cfg(tmp_path), transport, "model-x")

    msgs = _messages(caplog)
    attempt = [m for m in msgs if "попытка" in m and "не удалась" in m]
    assert attempt == [
        "Батч 1: попытка 1 не удалась, класс ошибки GeminiRuntimeError "
        "(Gemini вернул ошибку: HTTP 400, код INVALID_ARGUMENT.)."
    ]
    assert not any(FAKE_KEY in m for m in msgs)
    assert not any(FAKE_KEY in r.getMessage() for r in caplog.records)


def test_non_runtime_error_logs_class_name_only(tmp_path, caplog):
    """Прочие исключения транспорта: в журнал попадает только имя класса, не текст исключения."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    transport = StubTransport([RuntimeError("тело запроса с " + FAKE_KEY)])

    with pytest.raises(RuntimeError):
        generate_prompts(_segments(), [1, 2], _cfg(tmp_path), transport, "model-x")

    msgs = _messages(caplog)
    assert "Батч 1: попытка 1 не удалась, класс ошибки RuntimeError." in msgs
    assert not any(FAKE_KEY in m for m in msgs)


def test_prompt_text_never_appears_in_any_log_line(tmp_path, caplog):
    """Текст промпта (с уникальным маркером) не попадает ни в одну строку журнала этого логгера."""
    caplog.set_level(logging.DEBUG)  # все логгеры, включая корневой
    transport = StubTransport([_clean_reply()])

    results = generate_prompts(_segments(), [1, 2], _cfg(tmp_path), transport, "model-x")

    # маркер действительно есть в результате, то есть проверка не пустая
    assert PROMPT_MARKER in results[0].prompt
    for rec in caplog.records:
        assert PROMPT_MARKER not in rec.getMessage()
        assert "A woman walks" not in rec.getMessage()
        assert "Rain falls" not in rec.getMessage()


# ---------------------------------------------------------------------------
# Новые проверки: строки для консоли GitHub и журнал сохранения prompts.json.
# ---------------------------------------------------------------------------

def test_console_shows_batch_and_run_lines_but_not_per_clip(tmp_path, caplog):
    """Консоль: начало и итог прогона, начало и итог батча проходят фильтр; «Промпт принят» — нет."""
    from Generate.core import log_setup as ls

    caplog.set_level(logging.INFO, logger=LOGGER)
    transport = StubTransport([_clean_reply()])
    generate_prompts(_segments(), [1, 2], _cfg(tmp_path), transport, "model-x")

    records = [r for r in caplog.records if r.name == LOGGER]
    console = ls.ConsoleFilter()
    shown = [r.getMessage() for r in records if console.filter(r)]
    # 2 клипа в одном батче: начало прогона, начало батча, получено по батчу, итог прогона
    assert len(shown) == 4
    assert shown[0].startswith("=== Генерация ") and ": клипов 2, батчей 1, ключей " in shown[0]
    assert shown[1].startswith("Батч 1 из 1: начало")
    assert shown[2].startswith("Батч 1: получено промптов с текстом")
    assert shown[3].startswith("=== Генерация ") and "-промптов готова:" in shown[3]
    # построчные записи «Промпт принят» остаются только в файле журнала
    accepted = [r for r in records if r.getMessage().startswith("Промпт принят:")]
    assert len(accepted) == 2
    assert not any(console.filter(r) for r in accepted)


def test_save_logs_start_and_success_with_path_and_total(tmp_path, caplog, monkeypatch):
    """Сохранение батча: строка начала (путь и число промптов) и строка успеха (всего в файле), обе в консоль."""
    from types import SimpleNamespace

    from Generate.core import gemini_orchestrator as go
    from Generate.core import log_setup as ls

    caplog.set_level(logging.INFO, logger="Generate.gemini_orchestrator")

    def fake_generate(segments, wanted, cfg, transport, model, on_batch=None):
        on_batch([SimpleNamespace(num=n, prompt=f"промпт {n}", review_reason="", needs_review=False, extra={})
                  for n in wanted])
        return []

    monkeypatch.setattr(go.gp, "generate_prompts", fake_generate)
    path = tmp_path / "prompts.json"
    timing = go.TimingParams(24, {"kind": "8k+1"}, 2.5, 5.0, 10.0, 0.5)
    segs = [Segment(1, 0, 3000, "A woman walks."), Segment(2, 3000, 6000, "Rain falls.")]
    go.generate_and_store(segs, [1, 2], _cfg(tmp_path), StubTransport([]), path, profile="p", timing=timing)

    recs = [r for r in caplog.records if r.name == "Generate.gemini_orchestrator"]
    msgs = [r.getMessage() for r in recs]
    assert f"Сохранение промптов начато: файл {path}, промптов к сохранению: 2." in msgs
    assert f"Промпты сохранены: файл {path}, всего в файле: 2." in msgs
    console = ls.ConsoleFilter()
    for r in recs:
        if r.getMessage().startswith(("Сохранение промптов начато", "Промпты сохранены")):
            assert console.filter(r)


def test_save_failure_warns_with_path_and_reason_and_still_raises(tmp_path, caplog, monkeypatch):
    """Сбой записи prompts.json: предупреждение с путём и причиной; исключение уходит дальше как раньше."""
    from types import SimpleNamespace

    from Generate.core import gemini_orchestrator as go

    caplog.set_level(logging.INFO, logger="Generate.gemini_orchestrator")

    def fake_generate(segments, wanted, cfg, transport, model, on_batch=None):
        on_batch([SimpleNamespace(num=1, prompt="промпт 1", review_reason="", needs_review=False, extra={})])
        return []

    def failing_write(path, data):
        raise OSError("нет места на диске")

    monkeypatch.setattr(go.gp, "generate_prompts", fake_generate)
    monkeypatch.setattr(go, "atomic_write_json", failing_write)
    path = tmp_path / "prompts.json"
    timing = go.TimingParams(24, {"kind": "8k+1"}, 2.5, 5.0, 10.0, 0.5)
    segs = [Segment(1, 0, 3000, "A woman walks.")]

    with pytest.raises(OSError):
        go.generate_and_store(segs, [1], _cfg(tmp_path), StubTransport([]), path, profile="p", timing=timing)

    warns = [r for r in caplog.records if r.name == "Generate.gemini_orchestrator" and r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warns] == [
        f"Не удалось сохранить промпты в файл {path}: OSError: нет места на диске."
    ]


def test_orchestrator_save_state_failure_names_path_and_reason(tmp_path, caplog, monkeypatch):
    """_save_state (render_status): предупреждение о сбое содержит путь и причину; сохранение не меняется."""
    from Generate.core import gemini_orchestrator as go
    from Generate.core import orchestrator as orch

    caplog.set_level(logging.WARNING, logger="Generate.orchestrator")

    def failing_write(path, data):
        raise OSError("диск недоступен")

    monkeypatch.setattr(go, "atomic_write_json", failing_write)
    path = tmp_path / "prompts.json"

    class _Fake:
        prompts_path = path
        _state = {"prompts": {}}

    orch.Orchestrator._save_state(_Fake())  # исключение гасится, как и раньше

    msgs = [r.getMessage() for r in caplog.records if r.name == "Generate.orchestrator"]
    assert msgs == [f"Не удалось сохранить файл промптов {path}: OSError: диск недоступен."]
