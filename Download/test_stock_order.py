"""Тесты порядка поиска этапа 4: вступление, видео перед фото, правило архива.

Все тесты офлайн: сеть не используется. Вместо поиска подменены run_variant (что запрошено и
что вернулось) и finalize_candidate (ссылка на файл). Тесты написаны в этом этапе и НЕ запускались
(запуск - на GitHub).
"""
import asyncio
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import search  # noqa: E402
from search import (  # noqa: E402
    Candidate,
    Context,
    SegmentSpec,
    archive_allowed,
    backup_candidate_sites,
    build_cascade,
    is_intro,
    process_segment_inner,
)

STOCK = {"pexels", "pixabay"}
ARCHIVE = {"wikimedia", "loc", "nasa"}


@pytest.fixture(autouse=True)
def fixed_settings(monkeypatch):
    """Не зависим от окружения CI: окно 1200 с, режим источников - микс."""
    monkeypatch.setattr(search, "INTRO_SECONDS", 1200.0)
    monkeypatch.setattr(search, "SOURCES_MODE", search.SOURCES_MIX)


def _seg(index=1, start_s=1500.0, is_entity=False, sites=("pexels", "pixabay", "wikimedia"),
         entity=("Rome",)):
    return SegmentSpec(
        index=index, scene="s", sites=list(sites), query_narrow="Rome arch gate",
        query_medium="Rome doorway", query_broad="city", type="image", is_entity=is_entity,
        entity_keywords=list(entity) if is_entity else [], min_duration=5.0, start_s=start_s,
    )


def _ctx(mode=1):
    return Context(
        session=None, pexels_api_key="", pixabay_api_key="", site_semaphores={},
        global_semaphore=asyncio.Semaphore(1), clip_semaphore=asyncio.Semaphore(1),
        used_files_lock=asyncio.Lock(), media_mode=mode,
    )


def _cand(kind, cid, accepted, why="abs", sim=0.35):
    return Candidate(
        site="pexels", cand_id=cid, text="", license_ok=True, preview_url=None,
        page_url="p" + cid, own_sim=sim, similarity=sim, rank=1, accepted=accepted,
        reject_reason=why, variant="medium", stats_key="pexels", kind=kind,
    )


def _install(monkeypatch, video_found, photo_found, log):
    """Подмена поиска: video_found / photo_found - нужен ли принятый кандидат этого типа."""

    async def fake_run_variant(ctx, seg, variant):
        log.append((seg.type, variant.name, tuple(variant.sites)))
        if seg.type == "video":
            return [_cand("video", "v1", True)] if video_found else []
        return [_cand("photo", "i1", True)] if photo_found else []

    async def fake_finalize(ctx, cand):
        return f"https://cdn.example/{cand.site}/{cand.cand_id}"

    monkeypatch.setattr(search, "run_variant", fake_run_variant)
    monkeypatch.setattr(search, "finalize_candidate", fake_finalize)


def _run(seg, mode=1):
    return asyncio.run(process_segment_inner(_ctx(mode), seg))


def _sites_used(log):
    return {s for _t, _n, sites in log for s in sites}


# --- граница вступления -------------------------------------------------------

def test_intro_boundary_exact_is_not_intro():
    assert is_intro(_seg(start_s=1200.0)) is False


def test_intro_boundary_just_before_is_intro():
    assert is_intro(_seg(start_s=1199.999)) is True


def test_no_start_time_is_not_intro():
    assert is_intro(_seg(start_s=None)) is False


# --- вступление: без архива, только видео, пропуск ----------------------------

def test_intro_segment_never_gets_archive_site(monkeypatch):
    log: list = []
    _install(monkeypatch, video_found=False, photo_found=False, log=log)
    seg = _seg(start_s=100.0, is_entity=True, sites=("wikimedia", "loc", "pexels"))
    _run(seg)
    assert log, "поиск должен был выполниться"
    assert _sites_used(log) <= STOCK, log
    # то же для построения каскада и для резерва
    assert all(set(v.sites) <= STOCK for v in build_cascade(seg))
    assert not archive_allowed(seg)
    assert "wikimedia" not in backup_candidate_sites(seg.sites, "pexels", allow_archive=archive_allowed(seg))


def test_intro_segment_with_no_video_is_skipped_without_photo(monkeypatch):
    log: list = []
    _install(monkeypatch, video_found=False, photo_found=True, log=log)  # фото есть, но не должно искаться
    seg = _seg(start_s=100.0)
    assert _run(seg) == (None, None)  # пропуск: сегмент уходит в missing
    types = {t for t, _n, _s in log}
    assert types == {"video"}, log  # ни одного запроса фото


# --- обычный сегмент: видео сначала, фото только по необходимости -------------

def test_ordinary_segment_with_video_never_queries_image(monkeypatch):
    log: list = []
    _install(monkeypatch, video_found=True, photo_found=True, log=log)
    url, _backup = _run(_seg(start_s=1500.0))
    assert url and url.endswith("/v1"), url
    assert all(t == "video" for t, _n, _s in log), log


def test_ordinary_segment_without_video_queries_image(monkeypatch):
    log: list = []
    _install(monkeypatch, video_found=False, photo_found=True, log=log)
    url, _backup = _run(_seg(start_s=1500.0))
    assert url and url.endswith("/i1"), url
    types = [t for t, _n, _s in log]
    # видео прошло все проходы раньше фото
    assert types.index("image") > max(i for i, t in enumerate(types) if t == "video"), types


# --- правило архива (d) --------------------------------------------------------

def test_is_entity_outside_intro_gets_archive_site(monkeypatch):
    log: list = []
    _install(monkeypatch, video_found=False, photo_found=False, log=log)
    seg = _seg(start_s=1500.0, is_entity=True, sites=("pexels", "pixabay", "wikimedia"))
    _run(seg)
    assert archive_allowed(seg) is True
    assert "wikimedia" in _sites_used(log), log
    assert "wikimedia" in backup_candidate_sites(seg.sites, "pexels", allow_archive=archive_allowed(seg))


def test_is_entity_inside_intro_gets_no_archive_site(monkeypatch):
    log: list = []
    _install(monkeypatch, video_found=False, photo_found=False, log=log)
    seg = _seg(start_s=100.0, is_entity=True, sites=("pexels", "pixabay", "wikimedia"))
    _run(seg)
    assert archive_allowed(seg) is False
    assert "wikimedia" not in _sites_used(log), log
    assert all(not (set(v.sites) & ARCHIVE) for v in build_cascade(seg))
