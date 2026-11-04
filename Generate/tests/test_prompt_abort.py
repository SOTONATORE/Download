"""Остановка до аренды карт при жёсткой ошибке генерации промптов (офлайн, только заглушки).

Проверяется: жёсткая ошибка любой волны (видео или фото) останавливает прогон до аренды и до рендера,
готовые промпты остаются в prompts.json; временный сбой, который проходит на повторе, и прогон без ошибок
идут как прежде. Сети нет: Gemini, Vast, воркер и релиз — заглушки из test_orchestrator (Rig).
"""
from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from Generate.core import gemini_prompts as gp
    from Generate.core import orchestrator as orch
    from Generate.core import prompt_pipeline as pp
    from Generate.tests.test_orchestrator import Rig
except ImportError:  # pragma: no cover
    from core import gemini_prompts as gp  # type: ignore
    from core import orchestrator as orch  # type: ignore
    from core import prompt_pipeline as pp  # type: ignore
    from tests.test_orchestrator import Rig  # type: ignore

# строка об остановке (вид из ТЗ; номер сегмента подставляется из текста ошибки шага)
STOP_VIDEO = ("Прогон остановлен до аренды карт: ошибка генерации промптов (видео, сегмент 2). "
              "Готовые промпты сохранены, карты не арендованы.")
STOP_PHOTO = ("Прогон остановлен до аренды карт: ошибка генерации промптов (фото, сегмент 2). "
              "Готовые промпты сохранены, карты не арендованы.")
HARD_TEXT = ("Сегмент 2: в промпте остались запрещённые типы кадра или маркеры после 2 переписываний: "
             "запрещённый формат кадра: «framing». Прогон остановлен.")


class _Ok:
    """Внутренний транспорт, который всегда отвечает пустым списком (вызов удался)."""

    def generate_json(self, model, system, user, schema) -> str:
        return "[]"


def _managed(transport):
    """ManagedTransport с нулевыми паузами и большими лимитами: повторы и счётчики реальные, ожиданий нет."""
    return orch.go.ManagedTransport(transport, ("m",), rpm_limit=10 ** 6, tpm_limit=10 ** 12,
                                    rpd_limit=10 ** 6, sleep=lambda s: None)


def _video_hard(rig, seg=2):
    """Видео: один успешный вызов Gemini, часть промптов записана в файл, затем ошибка переписывания."""
    def fake(segments, needed, cfg, transport, prompts_path, *, profile, timing, **kw):
        _managed(transport).generate_json("m", "sys", "user", {})   # батч прошёл до ошибки
        rig.gemini(segments, [n for n in needed if n == 1], cfg, transport, prompts_path,
                   profile=profile, timing=timing, **kw)           # сегмент 1 — готов и сохранён
        raise gp.GeminiRuntimeError(HARD_TEXT.replace("Сегмент 2", f"Сегмент {seg}"))
    return fake


def _photo_hard(monkeypatch, seg=2):
    """Фото: настоящая волна фото (pp._photo_step), генерация подменена: сегмент 1 готов, затем жёсткая ошибка."""
    real = orch.build_prompts

    def as_photo(needed, **kw):  # все нужные номера идут в фото
        kw.update(photo_nums=set(needed), photo_profile="photo-x", photo_cfg=object())
        return real(needed, **kw)

    def fake_generate(segments, wanted, cfg, transport, model, on_batch=None):
        transport.generate_json(model, "sys", "user", {})           # успешный вызов до ошибки
        on_batch([types.SimpleNamespace(num=1, prompt="промпт 1", review_reason="", needs_review=False, extra={})])
        raise gp.GeminiRuntimeError(HARD_TEXT.replace("Сегмент 2", f"Сегмент {seg}"))

    monkeypatch.setattr(orch, "build_prompts", as_photo)
    monkeypatch.setattr(orch.gp, "generate_prompts", fake_generate)


def _spy_rent(monkeypatch):
    """Записывает обращения к аренде и к _work; сами они ничего не делают."""
    calls = {"maybe_rent": [], "desired": [], "rent_one": [], "work": []}
    monkeypatch.setattr(orch.Orchestrator, "_maybe_rent", lambda self, now: calls["maybe_rent"].append(now))
    monkeypatch.setattr(orch.Orchestrator, "_desired", lambda self, now: calls["desired"].append(now) or 0)
    monkeypatch.setattr(orch.Orchestrator, "_rent_one",
                        lambda self, now, offers=None, role="video": calls["rent_one"].append(now) or "exhausted")
    monkeypatch.setattr(orch.Orchestrator, "_work", lambda self, ready: calls["work"].append(list(ready)))
    return calls


def _messages(caplog) -> list:
    return [r.getMessage() for r in caplog.records]


def _assert_stopped_before_rent(rig, calls, caplog, stop_line):
    assert calls == {"maybe_rent": [], "desired": [], "rent_one": [], "work": []}
    assert rig.vast.search_calls == 0 and rig.vast.created == []
    assert rig.worker.posted == []
    msgs = _messages(caplog)
    assert msgs.count(stop_line) == 1                       # строка об остановке ровно один раз
    assert not any("Используем уже сохранённые" in m for m in msgs)  # старый путь возврата не срабатывает
    assert not any("Без готового промпта" in m for m in msgs)


# --- жёсткая ошибка -------------------------------------------------------------

def test_video_hard_error_stops_before_rent_and_render(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2))
    monkeypatch.setattr(orch.go, "generate_and_store", _video_hard(rig))
    calls = _spy_rent(monkeypatch)

    summary = rig.run()

    _assert_stopped_before_rent(rig, calls, caplog, STOP_VIDEO)
    assert summary.cards_rented == 0 and summary.completed == 0
    # готовый промпт, записанный до ошибки, остаётся в prompts.json
    state = json.loads(rig.prompts_path.read_text(encoding="utf-8"))
    assert state["prompts"]["1"]["status"] == "ready" and state["prompts"]["1"]["prompt"] == "промпт 1"


def test_photo_hard_error_stops_before_rent_and_render(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2))
    _photo_hard(monkeypatch)
    calls = _spy_rent(monkeypatch)

    summary = rig.run()

    _assert_stopped_before_rent(rig, calls, caplog, STOP_PHOTO)
    assert summary.cards_rented == 0 and summary.completed == 0
    state = json.loads(rig.prompts_path.read_text(encoding="utf-8"))
    assert state["prompts"]["1"]["status"] == "ready" and state["prompts"]["1"]["prompt"] == "промпт 1"


# --- без жёсткой ошибки: поведение прежнее ----------------------------------------

def test_no_hard_error_renders_as_before(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2))   # штатный GeminiSpy: все промпты готовы

    summary = rig.run()

    assert summary.completed == 2
    assert rig.worker.posted == [1, 2]
    assert rig.vast.created, "аренда карт должна была состояться"
    msgs = _messages(caplog)
    assert not any("Прогон остановлен до аренды карт" in m for m in msgs)
    assert not any("Без готового промпта" in m for m in msgs)


def test_transient_503_retry_success_renders_as_before(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    rig = Rig(tmp_path, monkeypatch, missing=(1, 2))

    class _Flaky:
        """Первый вызов — 503, второй — успех: повтор делает ManagedTransport, как в боевом коде."""

        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, model, system, user, schema) -> str:
            self.calls += 1
            if self.calls == 1:
                raise gp.GeminiRuntimeError("Сервис Gemini временно недоступен", status=503)
            return "[]"

    rig.orc.gemini_transport = _Flaky()

    def fake(segments, needed, cfg, transport, prompts_path, *, profile, timing, **kw):
        _managed(transport).generate_json("m", "sys", "user", {})   # 503, затем успех на повторе
        return rig.gemini(segments, needed, cfg, transport, prompts_path, profile=profile, timing=timing, **kw)

    monkeypatch.setattr(orch.go, "generate_and_store", fake)

    summary = rig.run()

    assert summary.completed == 2
    assert rig.worker.posted == [1, 2]
    msgs = _messages(caplog)
    assert not any("Прогон остановлен до аренды карт" in m for m in msgs)
    assert not any("Без готового промпта" in m for m in msgs)


# --- классификация в конвейере (реальный build_prompts) ------------------------------

def test_transient_exhausted_still_falls_back_to_saved_prompts(tmp_path, monkeypatch):
    """Временный сбой, исчерпавший повторы, — как прежде: abort не ставится, ошибка уходит в errors."""

    class _Down:
        def generate_json(self, model, system, user, schema) -> str:
            raise gp.GeminiRuntimeError("сбой сети", status=None)

    def fake(segments, needed, cfg, transport, prompts_path, *, profile, timing, **kw):
        return _managed(transport).generate_json("m", "sys", "user", {})   # 60 сбоев подряд → исчерпано

    monkeypatch.setattr(orch.go, "generate_and_store", fake)
    run = pp.build_prompts([1], segments=[], photo_nums=set(), cfg=None, profile="p", timing=None,
                           photo_profile=None, photo_cfg=None, transport=_Down(), gemini_options={},
                           prompts_path=tmp_path / "prompts.json")

    assert run.abort is None
    assert run.ready == []
    assert run.errors and "попыток подряд" in run.errors[0]


def test_hard_error_after_successful_call_sets_abort_with_segment(tmp_path, monkeypatch):
    """Ошибка переписывания после успешного вызова: abort с номером сегмента из текста ошибки."""

    def fake(segments, needed, cfg, transport, prompts_path, *, profile, timing, **kw):
        _managed(transport).generate_json("m", "s", "u", {})
        raise gp.GeminiRuntimeError(HARD_TEXT.replace("Сегмент 2", "Сегмент 271"))

    monkeypatch.setattr(orch.go, "generate_and_store", fake)
    run = pp.build_prompts([271], segments=[], photo_nums=set(), cfg=None, profile="p", timing=None,
                           photo_profile=None, photo_cfg=None, transport=_Ok(), gemini_options={},
                           prompts_path=tmp_path / "prompts.json")

    assert run.abort == "видео, сегмент 271"
