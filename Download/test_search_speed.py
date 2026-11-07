"""Тесты скорости поиска (этап 3): лимит Pixabay, кэш запросов, ранний выход, неизменность выбора.

Все тесты офлайн: HTTP и отбор кандидатов подменены фейками, сети нет.
Тесты написаны под новый код; в этом этапе они не запускались (запуск - на GitHub).
"""
import asyncio
import dataclasses
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import download  # noqa: E402
import search  # noqa: E402


# ---------------------------------------------------------------------------
# Лимит Pixabay: по умолчанию 99, максимум 99
# ---------------------------------------------------------------------------

def test_pixabay_default_is_99():
    assert download.load_pixabay_rpm({}) == 99
    assert search.load_rate_config({}).pixabay_requests_per_minute == 99


def test_pixabay_99_is_accepted():
    env = {"PIXABAY_REQUESTS_PER_MINUTE": "99"}
    assert download.load_pixabay_rpm(env) == 99
    assert search.load_rate_config(env).pixabay_requests_per_minute == 99


def test_pixabay_100_raises_value_error_with_variable_name():
    env = {"PIXABAY_REQUESTS_PER_MINUTE": "100"}
    with pytest.raises(ValueError) as e_dl:
        download.load_pixabay_rpm(env)
    assert "PIXABAY_REQUESTS_PER_MINUTE" in str(e_dl.value) and "99" in str(e_dl.value)
    with pytest.raises(ValueError) as e_s:
        search.load_rate_config(env)
    assert "PIXABAY_REQUESTS_PER_MINUTE" in str(e_s.value) and "99" in str(e_s.value)


# ---------------------------------------------------------------------------
# Кэш запросов: один запрос на (нормализованный запрос, сайт, тип медиа)
# ---------------------------------------------------------------------------

def _make_ctx(**kw):
    return search.Context(
        session=None, pexels_api_key="", pixabay_api_key="k", site_semaphores={},
        global_semaphore=asyncio.Semaphore(1), clip_semaphore=asyncio.Semaphore(1),
        used_files_lock=asyncio.Lock(), **kw,
    )


def test_same_normalized_query_same_site_makes_one_request(monkeypatch):
    calls = []

    async def _fake_http(ctx_, site, url, headers=None, params=None, treat_429_as_exhaustion=False):
        calls.append((site, params["q"], url))
        return {"hits": []}

    monkeypatch.setattr(search, "http_get_json", _fake_http)

    async def _run():
        ctx = _make_ctx()
        a = await search.search_pixabay(ctx, "Dog  Run", "video")
        b = await search.search_pixabay(ctx, "dog run", "video")
        return a, b

    a, b = asyncio.run(_run())
    assert a == [] and b == []
    assert len(calls) == 1, calls


def test_cache_does_not_merge_media_types_or_different_queries(monkeypatch):
    calls = []

    async def _fake_http(ctx_, site, url, headers=None, params=None, treat_429_as_exhaustion=False):
        calls.append((site, params["q"], url))
        return {"hits": []}

    monkeypatch.setattr(search, "http_get_json", _fake_http)

    async def _run():
        ctx = _make_ctx()
        await search.search_pixabay(ctx, "dog run", "video")
        await search.search_pixabay(ctx, "dog run", "image")     # другой тип медиа - новый запрос
        await search.search_pixabay(ctx, "dog walk", "video")    # другой запрос - новый запрос

    asyncio.run(_run())
    assert len(calls) == 3, calls


# ---------------------------------------------------------------------------
# Ранний выход и неизменность выбора на фикстуре из 5 сегментов
# ---------------------------------------------------------------------------

def _cand(site, cid, *, rank, own, reason, accepted):
    return search.Candidate(
        site=site, cand_id=cid, text="", license_ok=True, preview_url=None,
        page_url=f"{site}/{cid}", own_sim=own, similarity=own, rank=rank,
        accepted=accepted, reject_reason=reason, variant=None, kind="video",
    )


STRONG = dict(rank=1, own=0.9, reason="abs", accepted=True)
WEAK = dict(rank=5, own=0.5, reason="rank", accepted=True)
REJECTED = dict(rank=9, own=0.2, reason="rank", accepted=False)

# (сайт, запрос варианта) -> кандидаты, которые вернёт поиск и оценка
FIXTURE = {
    ("pexels", "n1"): [_cand("pexels", "1", **STRONG)],
    ("pexels", "n2"): [_cand("pexels", "w2", **WEAK)],
    ("pixabay", "b2"): [_cand("pixabay", "b2", **STRONG)],
    ("pixabay", "n3"): [_cand("pixabay", "3", **STRONG)],
    ("pexels", "n4"): [_cand("pexels", "w4p", **dict(WEAK, own=0.4))],
    ("pixabay", "b4"): [_cand("pixabay", "w4", **dict(WEAK, own=0.6))],
    ("pexels", "n5"): [_cand("pexels", "r5", **REJECTED)],
    ("pixabay", "b5"): [_cand("pixabay", "r5b", **dict(REJECTED, own=0.1))],
}

# Ожидаемый выбор по сегментам (вычислен вручную по правилам каскада, до и после изменений)
EXPECTED_URLS = {1: "pexels/1", 2: "pixabay/b2", 3: "pixabay/3", 4: "pixabay/w4", 5: None}


def _run_five_segments(monkeypatch):
    fetched = []

    async def _fake_fetch(ctx_, site, seg_, query, variant_name, stats_key=None):
        fetched.append((seg_.index, site, query))
        return [dataclasses.replace(c) for c in FIXTURE.get((site, query), [])]

    async def _fake_score(ctx_, top, seg_index=None, stats_key=None, variant_name=None):
        return list(top)

    async def _fake_claim(ctx_, ordered, seg_index):
        return (ordered[0].page_url, None) if ordered else (None, None)

    monkeypatch.setattr(search, "fetch_and_filter", _fake_fetch)
    monkeypatch.setattr(search, "score_candidates", _fake_score)
    monkeypatch.setattr(search, "try_claim_pool", _fake_claim)
    monkeypatch.setitem(search.SITE_SEARCH_FUNCS, "pexels", lambda *a, **k: None)
    monkeypatch.setitem(search.SITE_SEARCH_FUNCS, "pixabay", lambda *a, **k: None)

    async def _run():
        out = {}
        for k in range(1, 6):
            seg = search.SegmentSpec(
                index=k, scene="s", sites=["pexels", "pixabay"],
                query_narrow=f"n{k}", query_medium=f"n{k}", query_broad=f"b{k}",
                type="video", is_entity=False, entity_keywords=[],
            )
            cascade = [
                search.Variant(name="narrow", query=f"n{k}", sites=["pexels", "pixabay"], apply_entity_filter=False),
                search.Variant(name="broad", query=f"b{k}", sites=["pixabay", "pexels"], apply_entity_filter=False),
            ]
            url, _backup, _rejected = await search._cascade_normal(_make_ctx(), seg, cascade)
            out[k] = url
        return out

    return asyncio.run(_run()), fetched


def test_selection_five_segments_is_unchanged(monkeypatch):
    urls, _ = _run_five_segments(monkeypatch)
    assert urls == EXPECTED_URLS


def test_early_stop_skips_second_site_after_strong_accept(monkeypatch):
    _, fetched = _run_five_segments(monkeypatch)
    # сегмент 1: pexels дал сильного принятого - pixabay и второй вариант не запрашиваются
    assert [f for f in fetched if f[0] == 1] == [(1, "pexels", "n1")]
    # сегмент 2: pexels дал только слабого - pixabay запрашивается; в broad сильный pixabay
    # стоит первым, поэтому pexels в broad не запрашивается
    seg2 = [f for f in fetched if f[0] == 2]
    assert (2, "pixabay", "b2") in seg2 and (2, "pexels", "b2") not in seg2
