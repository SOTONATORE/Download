"""Серии соседних клипов с одинаковым фоном (backdrop.py и обвязка в duplicate_check / prompt_pipeline).

Без сети и без файлов: модели и конфиги подменяются через monkeypatch, как в test_duplicate_rework.py.
"""
import threading

from Generate.core import backdrop as bd
from Generate.core import config_loader
from Generate.core import duplicate_check as dc
from Generate.core import gemini_prompts as gp
from Generate.core import prompt_check as pc
from Generate.core import prompt_pipeline as pp
from Generate.core.gemini_prompts import PromptConfig

STYLE = "Soft light."
SKY = ("A wide outdoor scene under a pale dusty blue sky with two puffy white clouds and distant muted olive hills. "
       "Flat tan dirt covers the ground.")
CREAM_WALL = ("A wide indoor scene against a flat warm cream wall and a flat tan wooden table at the bottom edge. "
              "A child stands at the center.")
CAVE = "A dark cave entrance with a rocky floor. A small character stands at the center."


def sky(n):
    """Тот же фон, но текст другой: иначе сработает проверка дубликатов, а не фона."""
    return SKY + f" Figure number {n} stands at the center."


def _entry(text, styled=True):
    prompt = (text.rstrip(".") + ". " + STYLE) if styled else text
    return {"num": 0, "prompt": prompt, "status": "ready", "num_frames": 1}


def _state(texts):
    return {"prompts": {str(n): _entry(t) for n, t in texts.items()}}


def _profile(monkeypatch, data):
    monkeypatch.setattr(config_loader, "profile_exists", lambda name: True)
    monkeypatch.setattr(config_loader, "_read_profile", lambda name: dict(data))


def _cfg():
    return PromptConfig(style_file="style.md", media="image")


# --- классификация фона ------------------------------------------------------

def test_sky_clouds_hills_formula_is_classified():
    assert bd.backdrop_signature(SKY) == bd.SKY_CLOUDS_HILLS


def test_sky_formula_wins_over_forest_word():
    # лесной кадр с тем же небом повторяет фон, поэтому относится к небу с холмами
    assert bd.backdrop_signature("A forest scene under a pale dusty blue sky with two puffy white clouds.") \
        == bd.SKY_CLOUDS_HILLS


def test_cream_wall_and_table_is_classified():
    assert bd.backdrop_signature(CREAM_WALL) == bd.CREAM_WALL_TABLE


def test_other_backgrounds_and_empty_prompts():
    assert bd.backdrop_signature(CAVE) == bd.CAVE
    assert bd.backdrop_signature("A flat illustrated map over a pale sea.") == bd.MAP
    assert bd.backdrop_signature("A plain thing with nothing known.") == bd.OTHER
    assert bd.backdrop_signature("") == bd.OTHER
    assert bd.backdrop_signature(None) == bd.OTHER


def test_keywords_match_whole_words_only():
    # «map» внутри «mapping» и «lab» внутри «label» не считаются
    assert bd.backdrop_signature("A mapping of stars and a label.") == bd.OTHER


# --- серии по номерам клипов -------------------------------------------------

def test_three_consecutive_same_backdrop_are_flagged():
    prompts = {1: sky(1), 2: sky(2), 3: sky(3)}
    assert bd.find_backdrop_runs(prompts) == [[1, 2, 3]]


def test_two_consecutive_same_backdrop_are_not_flagged():
    assert bd.find_backdrop_runs({1: SKY, 2: SKY}) == []
    assert bd.find_backdrop_runs({1: SKY, 2: SKY, 3: CAVE, 4: SKY, 5: SKY}) == []


def test_numbering_gap_breaks_run():
    # 2 и 4 одинаковы по фону, но 3 отсутствует: серия из 1-2 и серия из 4 не образуют тройку
    assert bd.find_backdrop_runs({1: SKY, 2: SKY, 4: SKY}) == []
    # пропуск перед тройкой: серия начинается заново после разрыва
    assert bd.find_backdrop_runs({1: SKY, 2: SKY, 4: SKY, 5: SKY, 6: SKY}) == [[4, 5, 6]]


def test_unknown_backgrounds_do_not_form_runs():
    assert bd.find_backdrop_runs({1: "x y", 2: "p q", 3: "m n"}) == []


def test_wrapper_returns_key_groups_like_duplicates():
    prompts = {"1": sky(1), "2": sky(2), "3": sky(3), "4": CAVE, "9": sky(9), "x": sky(0)}
    assert dc.find_backdrop_groups(prompts) == [["1", "2", "3"]]
    assert dc.find_duplicate_prompts(prompts) == []


# --- очередь исправлений и флаг ниши -----------------------------------------

def test_run_is_queued_with_backdrop_problem_and_remark(monkeypatch):
    state = _state({1: sky(1), 2: sky(2), 3: sky(3), 4: CAVE})
    queued = pp.mark_duplicates_for_rework(state["prompts"], [1, 2, 3, 4], STYLE)
    assert queued == [2, 3]
    assert "rework_pending" not in state["prompts"]["1"]
    entry = state["prompts"]["2"]
    assert entry["rework_pending"] == [pc.make_problem(pc.PROBLEM_BACKDROP_RUN, "1")]
    text = gp._rework_text(entry["rework_pending"])
    assert "same backdrop as the neighbouring clips" in text
    assert pc.problem_ru(entry["rework_pending"][0]) == "одинаковый фон у соседних клипов: «1»"


def test_profile_without_flag_gets_no_backdrop_check(monkeypatch):
    # ниша без fail_on_duplicate_prompts: этап не читает состояние и не ставит ни дубликатов, ни фонов в очередь
    _profile(monkeypatch, {"refs_enabled": False})

    def forbidden(*a, **k):
        raise AssertionError("очередь проверки не должна запускаться без флага")

    monkeypatch.setattr(pp, "_photo_step", forbidden)
    monkeypatch.setattr(pp, "_video_step", forbidden)
    state = _state({1: SKY, 2: SKY, 3: SKY})
    out = pp._duplicate_rework([1, 2, 3], photo_nums={1, 2, 3}, segments=[], cfg=_cfg(), photo_cfg=_cfg(),
                               profile="ltx_like", photo_profile="zim_like", timing=None, watch=object(),
                               prompts_path="unused.json", gemini_options={}, errors=[], aborts=[],
                               state=state, lock=threading.Lock())
    assert out is state
    assert all("rework_pending" not in v for v in state["prompts"].values())


def test_flagged_profile_runs_backdrop_check_in_pipeline(monkeypatch):
    # с флагом тот же этап ставит клипы серии в очередь, пока фейковый шаг не снимет её
    _profile(monkeypatch, {"fail_on_duplicate_prompts": True})
    state = _state({1: sky(1), 2: sky(2), 3: sky(3), 4: CAVE})
    seen = {}

    def fake_photo_step(needed, **kw):
        seen["needed"] = list(needed)
        seen["problems"] = {n: list(kw["state"]["prompts"][str(n)]["rework_pending"]) for n in needed}
        for n in needed:
            e = kw["state"]["prompts"][str(n)]
            e["rework_pending"] = []
            e["rework_round"] = 1

    monkeypatch.setattr(pp, "_photo_step", fake_photo_step)
    monkeypatch.setattr(pp.go, "ManagedTransport", lambda *a, **k: object())
    monkeypatch.setattr(pp.go, "atomic_write_json", lambda path, data: None)
    monkeypatch.setattr(pp, "read_state", lambda path: state)
    pp._duplicate_rework([1, 2, 3, 4], photo_nums={1, 2, 3, 4}, segments=[], cfg=_cfg(), photo_cfg=_cfg(),
                         profile="fake_video", photo_profile="fake_photo", timing=None, watch=object(),
                         prompts_path="unused.json", gemini_options={}, errors=[], aborts=[],
                         state=state, lock=threading.Lock())
    assert seen["needed"] == [2, 3]
    assert seen["problems"][2] == [pc.make_problem(pc.PROBLEM_BACKDROP_RUN, "1")]
