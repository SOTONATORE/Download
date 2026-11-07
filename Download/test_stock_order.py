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


# ---------------------------------------------------------------------------
# Этап 5: флаг вступления для генератора, размер батча, имена переменных окружения.
# Тесты офлайн и НЕ запускались (запуск - на GitHub).
# ---------------------------------------------------------------------------

def test_generator_intro_flag_in_payload(monkeypatch):
    """Блок сегмента несёт is_intro: true для начала 100 с и is_intro: false для начала 1500 с (порог 1200)."""
    gq = pytest.importorskip("generate_queries")
    monkeypatch.setattr(gq, "INTRO_SECONDS_LIMIT", 1200.0)
    early = gq.Segment(index=1, start="00:01:40,000", end="00:01:45,000", text="A street in Rome.")
    late = gq.Segment(index=2, start="00:25:00,000", end="00:25:05,000", text="A street in Rome.")
    sentences, by_segment = gq.merge_segments_into_sentences([early, late])
    block_early = gq.build_segment_block(early, sentences, by_segment)
    block_late = gq.build_segment_block(late, sentences, by_segment)
    assert "is_intro: true" in block_early.splitlines()
    assert "is_intro: false" in block_late.splitlines()


def test_generator_intro_flag_absent_without_limit(monkeypatch):
    """Без порога (None, как в self-test) строки is_intro нет: старые блоки не меняются."""
    gq = pytest.importorskip("generate_queries")
    monkeypatch.setattr(gq, "INTRO_SECONDS_LIMIT", None)
    seg = gq.Segment(index=1, start="00:01:40,000", end="00:01:45,000", text="A street in Rome.")
    sentences, by_segment = gq.merge_segments_into_sentences([seg])
    assert "is_intro" not in gq.build_segment_block(seg, sentences, by_segment)


def test_generator_intro_rule_in_system_instruction():
    """Правило вступления есть в режимах 1 и 3 и отсутствует в режиме 2 (только архив)."""
    gq = pytest.importorskip("generate_queries")
    for mode in (gq.SOURCES_MIX, gq.SOURCES_STOCK):
        assert "is_intro: true" in gq.build_system_instruction(mode)
    assert "is_intro: true" not in gq.build_system_instruction(gq.SOURCES_ARCHIVE)


def test_generator_batch_size_default_25():
    """Размер батча генератора по умолчанию 25."""
    gq = pytest.importorskip("generate_queries")
    assert gq.DEFAULT_BATCH_SIZE == 25


def test_select_coverage_reads_coverage_always_env_only(monkeypatch):
    """select_coverage берёт COVERAGE_ALWAYS_SECONDS, а INTRO_SECONDS (окно поиска) не читает."""
    import select_coverage as sc

    captured = {}

    def fake_run(requests_path, srt_path, percent, coverage_always_seconds, window_seconds):
        captured["coverage_always"] = coverage_always_seconds
        return None

    monkeypatch.setattr(sc, "run", fake_run)
    monkeypatch.setattr(sc, "format_summary", lambda result: "")
    monkeypatch.setenv("INTRO_SECONDS", "777")
    monkeypatch.delenv("COVERAGE_ALWAYS_SECONDS", raising=False)
    sc.main(["r.json", "s.srt"])
    assert captured["coverage_always"] == 10.0  # INTRO_SECONDS=777 не влияет
    monkeypatch.setenv("COVERAGE_ALWAYS_SECONDS", "42")
    sc.main(["r.json", "s.srt"])
    assert captured["coverage_always"] == 42.0


def test_search_intro_seconds_env_parsing():
    """Окно поиска: INTRO_SECONDS пусто - 1200; число - число; мусор - 1200."""
    assert search.intro_seconds_from_env(None) == 1200.0
    assert search.intro_seconds_from_env("") == 1200.0
    assert search.intro_seconds_from_env("600") == 600.0
    assert search.intro_seconds_from_env("abc") == 1200.0


def test_download_workflow_inputs_and_env_names():
    """download.yml: вход intro_seconds по умолчанию 1200; coverage_always_seconds по умолчанию 10;
    поиск получает INTRO_SECONDS, выбор покрытия получает COVERAGE_ALWAYS_SECONDS (имена разные)."""
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "download.yml"
    lines = path.read_text(encoding="utf-8").splitlines()

    def input_block(name):
        start = lines.index(f"      {name}:")
        block = []
        for ln in lines[start + 1:]:
            if ln.startswith("      ") and not ln.startswith("        "):
                break
            block.append(ln)
        return block

    assert 'default: "1200"' in input_block("intro_seconds")
    assert 'default: "10"' in input_block("coverage_always_seconds")
    text = "\n".join(lines)
    assert "COVERAGE_ALWAYS_SECONDS: ${{ inputs.coverage_always_seconds || '10' }}" in text
    assert "INTRO_SECONDS: ${{ inputs.intro_seconds || '1200' }}" in text


def test_generate_yml_intro_default_1200():
    """Строковая проверка generate.yml: у входа intro_seconds default 1200."""
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "generate.yml"
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("      intro_seconds:")
    block = []
    for ln in lines[start + 1:]:
        if ln.startswith("      ") and not ln.startswith("        "):
            break
        block.append(ln)
    assert "        default: 1200" in block


def test_coverage_window_quota_no_carry():
    """select_coverage: квота окна = percent * сумма его сегментов, без переноса остатка между окнами.

    Тот же случай, что вторая самопроверка в select_coverage.py: 240 с, 40 сегментов по 6 с,
    первые 3 окна без текста, 70%. Последнее окно не получает секунд сверх своей квоты.
    """
    import select_coverage as sc

    texts = ["" if i <= 24 else "т" for i in range(1, 41)]
    reqs, srt = sc._synth(n=40, texts=texts)
    r = sc.select_segments(sc.build_items(reqs, srt), 70, 0)
    assert r.n_windows == 5, f"ожидалось 5 окон, получено {r.n_windows}"
    assert all(r.per_window[k][1] == 0 for k in range(3)), f"окна без текста не пустые: {r.per_window}"
    c_last, ms_last, quota_last = r.per_window[-1]
    assert ms_last <= quota_last + 1e-6, f"последнее окно: {ms_last} мс сверх квоты {quota_last} мс"
    assert c_last == 5 and ms_last == 30000, f"последнее окно должно взять 5 сегментов: {r.per_window}"
