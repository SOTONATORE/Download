"""Дубликаты промптов уходят в очередь исправлений (ниши с ключом fail_on_duplicate_prompts).

Без сети: модели и файлы подменяются фейками через monkeypatch. Профили ltx25 и zimage_turbo ключа не задают:
для них этап дубликатов не должен ничего менять (проверяется здесь же через загрузчик с отсутствующим ключом).
"""
import threading

from Generate.core import config_loader
from Generate.core import duplicate_check as dc
from Generate.core import gemini_prompts as gp
from Generate.core import prompt_check as pc
from Generate.core import prompt_pipeline as pp
from Generate.core.gemini_prompts import PromptConfig

STYLE = "Soft light."


def _entry(text, styled=True, **extra):
    """Запись состояния клипа: промпт со стилем, как после шага сохранения."""
    prompt = (text.rstrip(".") + ". " + STYLE) if styled else text
    return {"num": 0, "prompt": prompt, "status": "ready", "num_frames": 1, **extra}


def _state(texts):
    return {"prompts": {str(n): _entry(t) for n, t in texts.items()}}


def _profile(monkeypatch, data):
    monkeypatch.setattr(config_loader, "profile_exists", lambda name: True)
    monkeypatch.setattr(config_loader, "_read_profile", lambda name: dict(data))


def _cfg():
    return PromptConfig(style_file="style.md", media="image")


# --- чистая функция постановки в очередь -------------------------------------

def test_group_of_three_gives_two_rework_entries():
    # три клипа с одинаковым текстом: в очередь уходят второй и третий, первый остаётся как есть
    prompts = _state({1: "A dog in a field", 2: "A dog in a field", 3: "A dog in a field", 4: "A cat on a roof"})
    queued = pp.mark_duplicates_for_rework(prompts["prompts"], [1, 2, 3, 4], STYLE)
    assert queued == [2, 3]
    assert "rework_pending" not in prompts["prompts"]["1"]
    assert "rework_pending" not in prompts["prompts"]["4"]
    for key in ("2", "3"):
        entry = prompts["prompts"][key]
        assert entry["rework_pending"] == [pc.make_problem(pc.PROBLEM_DUPLICATE_PROMPT, "1")]
        assert entry["prompt_raw"] == "A dog in a field"   # сырой текст без стиля
        assert entry["rework_round"] == 0


def test_27_clips_in_10_groups_give_17_rework_entries():
    # 7 групп по 3 клипа и 3 группы по 2 клипа: 27 клипов, 10 групп, в очередь group size - 1 на группу
    texts, num = {}, 1
    for g, size in enumerate([3, 3, 3, 3, 3, 3, 3, 2, 2, 2]):
        for _ in range(size):
            texts[num] = f"group text {g}"
            num += 1
    assert len(texts) == 27
    prompts = _state(texts)["prompts"]
    queued = pp.mark_duplicates_for_rework(prompts, list(texts), STYLE)
    assert len(queued) == 17


def test_clip_with_pending_queue_is_not_changed():
    # клип уже в очереди по содержанию: его причина не заменяется дубликатом
    prompts = _state({1: "same", 2: "same"})
    prompts["prompts"]["2"]["rework_pending"] = ["camera_motion: «pan»"]
    assert pp.mark_duplicates_for_rework(prompts["prompts"], [1, 2], STYLE) == []
    assert prompts["prompts"]["2"]["rework_pending"] == ["camera_motion: «pan»"]


def test_clip_at_round_limit_is_not_queued():
    # на пределе кругов очередь всё равно не возьмёт клип: запись не меняется
    prompts = _state({1: "same", 2: "same"})
    prompts["prompts"]["2"]["rework_round"] = gp.REPAIR_ROUNDS
    assert pp.mark_duplicates_for_rework(prompts["prompts"], [1, 2], STYLE) == []
    assert "rework_pending" not in prompts["prompts"]["2"]


def test_raw_prompt_strips_style_back():
    assert pp._raw_prompt("A dog in a field. " + STYLE, STYLE) == "A dog in a field"
    assert pp._raw_prompt("A dog in a field", "") == "A dog in a field"


# --- текст повтора и подпись причины -----------------------------------------

def test_duplicate_rework_text_is_english_and_asks_for_visible_change():
    text = gp._rework_text([pc.make_problem(pc.PROBLEM_DUPLICATE_PROMPT, "1")])
    assert text == gp.DUPLICATE_REWORK_TEXT
    assert "visibly different" in text
    assert "layout" in text and "focal object" in text and "setting" in text and "pose" in text and "action" in text


def test_duplicate_reason_label_is_russian():
    assert pc.problem_ru(pc.make_problem(pc.PROBLEM_DUPLICATE_PROMPT, "1")) == "дубликат промпта: «1»"


# --- этап в build_prompts: фейки вместо модели и файла --------------------

def test_stage_queues_group_and_runs_rework_with_distinct_result(monkeypatch):
    _profile(monkeypatch, {"fail_on_duplicate_prompts": True})
    state = _state({1: "same", 2: "same", 3: "same", 4: "other"})
    calls = {}

    def fake_photo_step(needed, **kw):
        # очередь исправлений: модель вернула разные тексты, очередь снимается
        calls["needed"] = list(needed)
        for n in needed:
            e = kw["state"]["prompts"][str(n)]
            e["prompt"] = f"Distinct scene {n}. {STYLE}"
            e["rework_pending"] = []
            e["rework_round"] = 1

    monkeypatch.setattr(pp, "_photo_step", fake_photo_step)
    monkeypatch.setattr(pp.go, "ManagedTransport", lambda *a, **k: object())
    monkeypatch.setattr(pp.go, "atomic_write_json", lambda path, data: None)
    monkeypatch.setattr(pp, "read_state", lambda path: state)

    out = pp._duplicate_rework([1, 2, 3, 4], photo_nums={1, 2, 3, 4}, segments=[], cfg=_cfg(), photo_cfg=_cfg(),
                               profile="fake_video", photo_profile="fake_photo", timing=None, watch=object(),
                               prompts_path="unused.json", gemini_options={}, errors=[], aborts=[],
                               state=state, lock=threading.Lock())
    assert calls["needed"] == [2, 3]
    # после исправления дубликатов нет: итоговая проверка verify_prompts не падает
    texts = {k: v["prompt"] for k, v in out["prompts"].items()}
    assert dc.find_duplicate_prompts(texts) == []


def test_niche_without_key_produces_no_rework_entries(monkeypatch):
    # ltx25 / zimage_turbo: ключа нет, этап ничего не читает и не меняет
    _profile(monkeypatch, {"refs_enabled": False})

    def forbidden(*a, **k):
        raise AssertionError("очередь дубликатов не должна запускаться без ключа")

    monkeypatch.setattr(pp, "_photo_step", forbidden)
    monkeypatch.setattr(pp, "_video_step", forbidden)
    state = _state({1: "same", 2: "same", 3: "same"})
    before = {k: dict(v) for k, v in state["prompts"].items()}
    out = pp._duplicate_rework([1, 2, 3], photo_nums={1, 2, 3}, segments=[], cfg=_cfg(), photo_cfg=_cfg(),
                               profile="ltx_like", photo_profile="zim_like", timing=None, watch=object(),
                               prompts_path="unused.json", gemini_options={}, errors=[], aborts=[],
                               state=state, lock=threading.Lock())
    assert out is state
    assert state["prompts"] == before
    assert all("rework_pending" not in v for v in state["prompts"].values())
