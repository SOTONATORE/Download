#!/usr/bin/env python3
"""select_coverage.py - выбор сегментов для поиска по заданному проценту покрытия.

Запускается между generate_queries.py и поиском. Читает requests.json и .srt,
проставляет каждому сегменту булево поле "skip" (false - искать, true - не искать)
и атомарно переписывает requests.json. Остальные поля не меняются.

Сопоставление: ключ сегмента в requests.json ("1", "2", ...) == номер реплики в .srt.
Тайм-коды берутся только из .srt.

Алгоритм - см. функцию select_segments().

Примеры:
    python select_coverage.py requests.json result.srt --percent 40
    COVERAGE_PERCENT=40 INTRO_SECONDS=12 python select_coverage.py requests.json result.srt
    python select_coverage.py --selftest
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field

log = logging.getLogger("select_coverage")

DEFAULT_PERCENT = 100
DEFAULT_INTRO_SECONDS = 10.0
DEFAULT_WINDOW_SECONDS = 45.0  # целевая длина окна; число окон = round(total / это)
EPS_MS = 1e-6


# ---------------------------------------------------------------------------
# Разбор SRT (parse_srt из проекта, без изменения логики)
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    index: int
    start: str
    end: str
    text: str


SRT_BLOCK_RE = re.compile(
    r"(?P<index>\d+)\s*\n"
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})[^\n]*\n"
    r"(?P<text>.*?)(?=\n\s*\n\d+\s*\n|\Z)",
    re.DOTALL,
)


def parse_srt(path: str) -> list[Segment]:
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    raw = raw.replace("\r\n", "\n").strip() + "\n\n"

    segments: list[Segment] = []
    for m in SRT_BLOCK_RE.finditer(raw):
        idx = int(m.group("index"))
        text = m.group("text").strip()
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\{[^}]*\}", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        segments.append(
            Segment(index=idx, start=m.group("start"), end=m.group("end"), text=text)
        )

    if not segments:
        raise ValueError(f"Не удалось распарсить ни одного сегмента из {path}")

    segments.sort(key=lambda s: s.index)

    seen: set[int] = set()
    duplicates: set[int] = set()
    for s in segments:
        if s.index in seen:
            duplicates.add(s.index)
        seen.add(s.index)

    if duplicates:
        raise ValueError(f"В SRT обнаружены дублирующиеся номера сегментов: {sorted(duplicates)}.")

    expected = set(range(segments[0].index, segments[-1].index + 1))
    missing = expected - seen
    if missing:
        raise ValueError(
            f"В SRT отсутствуют номера сегментов: {sorted(missing)} (диапазон "
            f"{segments[0].index}..{segments[-1].index}, распарсено {len(segments)} из "
            f"{len(expected)})."
        )

    return segments


def tc_to_ms(tc: str) -> int:
    """HH:MM:SS,mmm (или с точкой) -> миллисекунды (целые, без накопления float-ошибок)."""
    h, m, rest = tc.split(":")
    s, ms = re.split(r"[,.]", rest)
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


# ---------------------------------------------------------------------------
# Ядро
# ---------------------------------------------------------------------------

class CoverageError(Exception):
    """Понятная ошибка входных данных: скрипт останавливается, файл не трогается."""


@dataclass
class Item:
    index: int
    start: int          # мс
    end: int            # мс
    has_text: bool
    visual_value: float
    is_entity: bool
    dur: int = 0
    window: int = 0
    intro: bool = False
    mandatory: bool = False
    selected: bool = False


@dataclass
class Result:
    skip: dict[int, bool]
    percent: int
    total_ms: int
    budget_ms: float
    selected_ms: int
    mandatory_ms: int
    n_windows: int
    n_selected: int
    n_unselected: int
    n_intro: int
    n_entity: int
    n_mandatory: int
    per_window: list[tuple[int, int, int]]  # (число выбранных, мс выбранных, мс доступного бюджета)
    warnings: list[str] = field(default_factory=list)


def build_items(requests: dict, srt: list[Segment]) -> list[Item]:
    """Сопоставление и валидация. Любая проблема - CoverageError."""
    try:
        req_ids = {int(k): k for k in requests}
    except (TypeError, ValueError):
        raise CoverageError("В requests.json есть ключи, не являющиеся номерами сегментов.")
    srt_ids = {s.index for s in srt}

    only_req = sorted(set(req_ids) - srt_ids)
    only_srt = sorted(srt_ids - set(req_ids))
    if only_req or only_srt:
        parts = []
        if only_req:
            parts.append(f"есть в requests.json, но нет в .srt: {_short(only_req)}")
        if only_srt:
            parts.append(f"есть в .srt, но нет в requests.json: {_short(only_srt)}")
        raise CoverageError("Индексы requests.json и .srt не совпали: " + "; ".join(parts))

    bad = []
    for i, key in sorted(req_ids.items()):
        seg = requests[key]
        vv = seg.get("visual_value") if isinstance(seg, dict) else None
        if isinstance(vv, bool) or not isinstance(vv, (int, float)):
            bad.append(i)
    if bad:
        raise CoverageError(f"У сегментов нет числового поля visual_value: {_short(bad)}")

    items = []
    for s in srt:
        seg = requests[req_ids[s.index]]
        a, b = tc_to_ms(s.start), tc_to_ms(s.end)
        if b < a:
            raise CoverageError(f"Сегмент {s.index}: конец ({s.end}) раньше начала ({s.start}).")
        items.append(Item(
            index=s.index, start=a, end=b, has_text=bool(s.text),
            visual_value=float(seg["visual_value"]),
            is_entity=seg.get("is_entity") is True, dur=b - a,
        ))
    return items


def _short(ids: list[int], limit: int = 15) -> str:
    return str(ids) if len(ids) <= limit else f"{ids[:limit]} ... (всего {len(ids)})"


def select_segments(items: list[Item], percent: int, intro_seconds: float,
                    window_seconds: float = DEFAULT_WINDOW_SECONDS) -> Result:
    """Выбор сегментов.

    1. total = конец последнего - начало первого; budget = percent/100 * total.
    2. Обязательные: вступление (подряд с начала, пока конец сегмента относительно начала
       ролика не достиг intro_seconds; последний берётся целиком) + все is_entity.
    3. Окон n = max(1, round(total / window_seconds)); окно k = [t0 + k*L, t0 + (k+1)*L),
       L = total/n. Сегмент относится к окну по своему началу (последнее окно включает
       правую границу). Бюджет окна = budget/n + остаток (или долг) от предыдущих окон.
    4. В окне: из доступного вычитаются обязательные сегменты окна, затем кандидаты
       идут по убыванию visual_value (при равенстве - раньше по времени); берётся тот,
       что целиком помещается, иначе пропускается и проверяется следующий. Неизрасходованный
       остаток уходит в следующее окно. Если обязательные окна превысили его бюджет,
       "долг" тоже переносится вперёд (уменьшает бюджет следующих окон), поэтому
       общий итог не превышает budget, пока обязательные сами в него помещаются.
    5. Кандидатом не бывают: нулевая длительность, пустой текст, visual_value <= 0.
    """
    items = sorted(items, key=lambda x: x.index)
    t0 = items[0].start
    total = items[-1].end - t0
    for it in items:
        it.selected = it.mandatory = it.intro = False

    # percent == 100: всё выбираем, ничего не вычисляем
    if percent == 100:
        for it in items:
            it.selected = True
        return _result(items, percent, total, float(total), 1, [(len(items), total, float(total))], [])

    budget = percent / 100 * total

    # --- обязательные ---
    intro_ms = intro_seconds * 1000
    if intro_ms > 0:
        for it in items:
            it.intro = it.mandatory = True
            if it.end - t0 >= intro_ms - EPS_MS:
                break
    for it in items:
        if it.is_entity:
            it.mandatory = True
    for it in items:
        it.selected = it.mandatory

    # --- окна ---
    n = max(1, round(total / (window_seconds * 1000))) if total > 0 else 1
    wlen = total / n if total > 0 else 1
    for it in items:
        it.window = min(n - 1, int((it.start - t0) / wlen)) if total > 0 else 0

    by_win: list[list[Item]] = [[] for _ in range(n)]
    for it in items:
        by_win[it.window].append(it)

    share = budget / n
    carry = 0.0
    per_window = []
    for w in range(n):
        avail = share + carry
        remaining = avail - sum(it.dur for it in by_win[w] if it.mandatory)
        cands = [it for it in by_win[w]
                 if not it.mandatory and it.dur > 0 and it.has_text and it.visual_value > 0]
        cands.sort(key=lambda it: (-it.visual_value, it.index))
        for it in cands:
            if it.dur <= remaining + EPS_MS:
                it.selected = True
                remaining -= it.dur
        carry = remaining  # >0 - остаток, <0 - долг от обязательных
        sel = [it for it in by_win[w] if it.selected]
        per_window.append((len(sel), sum(it.dur for it in sel), avail))

    warnings: list[str] = []
    mand_ms = sum(it.dur for it in items if it.mandatory)
    if mand_ms > budget + EPS_MS:
        over = (mand_ms - budget) / 1000
        pct = 100 * mand_ms / total if total else 0.0
        warnings.append(
            f"ПРЕДУПРЕЖДЕНИЕ: обязательные сегменты (вступление, is_entity) сами превышают бюджет: "
            f"фактический процент {pct:.1f}% при заданных {percent}%, сверх бюджета {over:.1f} с. "
            f"Все обязательные оставлены, ничего не урезано."
        )
    return _result(items, percent, total, budget, n, per_window, warnings)


def _result(items, percent, total, budget, n, per_window, warnings) -> Result:
    sel_ms = sum(it.dur for it in items if it.selected)
    r = Result(
        skip={it.index: not it.selected for it in items},
        percent=percent, total_ms=total, budget_ms=budget, selected_ms=sel_ms,
        mandatory_ms=sum(it.dur for it in items if it.mandatory),
        n_windows=n,
        n_selected=sum(it.selected for it in items),
        n_unselected=sum(not it.selected for it in items),
        n_intro=sum(it.intro for it in items),
        n_entity=sum(it.is_entity and it.mandatory for it in items),
        n_mandatory=sum(it.mandatory for it in items),
        per_window=per_window, warnings=warnings,
    )
    if not warnings and percent != 100 and total > 0:
        diff = (sel_ms - budget) / 1000
        if diff > EPS_MS:
            r.warnings.append(f"ПРЕДУПРЕЖДЕНИЕ: процент превышен на {diff:.1f} с (цельные сегменты).")
        elif -diff > 1e-3:
            r.warnings.append(f"Недобор {-diff:.1f} с относительно бюджета (дискретность сегментов "
                              f"и неподходящие по размеру/нулевые по ценности сегменты).")
    return r


def format_summary(r: Result) -> str:
    actual = 100 * r.selected_ms / r.total_ms if r.total_ms else 0.0
    lines = [
        "=== Сводка покрытия ===",
        f"Заданный процент: {r.percent}%; фактический (по секундам): {actual:.1f}% "
        f"({r.selected_ms / 1000:.1f} с из {r.total_ms / 1000:.1f} с; бюджет {r.budget_ms / 1000:.1f} с)",
        f"Сегментов выбрано: {r.n_selected}, не выбрано: {r.n_unselected}",
        f"Обязательных: {r.n_mandatory} (вступление: {r.n_intro}, is_entity: {r.n_entity}; "
        f"{r.mandatory_ms / 1000:.1f} с)",
        f"Окон: {r.n_windows}; выбрано по окнам (сегментов/секунд, доступный бюджет окна): "
        + ", ".join(f"[{k + 1}] {c}/{ms / 1000:.1f}с ({av / 1000:.1f}с)"
                    for k, (c, ms, av) in enumerate(r.per_window)),
    ]
    lines += r.warnings
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ввод/вывод
# ---------------------------------------------------------------------------

def atomic_write_json(path: str, data) -> None:
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".requests_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def run(requests_path: str, srt_path: str, percent: int, intro_seconds: float,
        window_seconds: float = DEFAULT_WINDOW_SECONDS) -> Result:
    with open(requests_path, "r", encoding="utf-8") as f:
        requests = json.load(f)
    if not isinstance(requests, dict):
        raise CoverageError("requests.json: ожидался объект {\"1\": {...}, ...}.")
    srt = parse_srt(srt_path)
    items = build_items(requests, srt)
    result = select_segments(items, percent, intro_seconds, window_seconds)
    for key, seg in requests.items():
        seg["skip"] = result.skip[int(key)]
    atomic_write_json(requests_path, requests)
    return result


# ---------------------------------------------------------------------------
# Самопроверки
# ---------------------------------------------------------------------------

def _synth(n=20, dur_ms=6000, vv=None, entities=(), texts=None):
    reqs, srt = {}, []
    for i in range(1, n + 1):
        a = (i - 1) * dur_ms
        tc = lambda ms: f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
        srt.append(Segment(i, tc(a), tc(a + dur_ms), "текст" if texts is None else texts[i - 1]))
        reqs[str(i)] = {"visual_value": (100 - i) if vv is None else vv[i - 1],
                        "is_entity": i in entities}
    return reqs, srt


def selftest() -> None:
    # 1. 100% выбирает всё
    reqs, srt = _synth()
    r = select_segments(build_items(reqs, srt), 100, 10)
    assert all(v is False for v in r.skip.values()), "100% должен выбирать всё"

    # 2. вступление: сегмент, пересекающий 10 с, берётся целиком (сегменты по 6 с: 0-6, 6-12)
    reqs, srt = _synth()
    r = select_segments(build_items(reqs, srt), 40, 10)
    assert r.n_intro == 2 and not r.skip[1] and not r.skip[2], r.n_intro

    # 3. обязательные сверх бюджета остаются + предупреждение
    reqs, srt = _synth(entities={5, 6, 7, 8, 9, 10})
    r = select_segments(build_items(reqs, srt), 10, 10)  # бюджет 12 с, обязательных >> 12 с
    assert all(not r.skip[i] for i in (1, 2, 5, 6, 7, 8, 9, 10))
    assert any("превышают бюджет" in w for w in r.warnings), r.warnings

    # 4. нет visual_value -> ошибка
    reqs, srt = _synth()
    del reqs["3"]["visual_value"]
    try:
        build_items(reqs, srt)
        raise AssertionError("ожидалась CoverageError")
    except CoverageError as e:
        assert "visual_value" in str(e)

    # 4b. индексы не совпали в обе стороны
    reqs, srt = _synth()
    reqs["99"] = reqs.pop("5")
    try:
        build_items(reqs, srt)
        raise AssertionError("ожидалась CoverageError")
    except CoverageError as e:
        assert "нет в .srt" in str(e) and "нет в requests.json" in str(e)

    # 5. детерминизм
    reqs, srt = _synth(vv=[50] * 20)
    a = select_segments(build_items(reqs, srt), 40, 10).skip
    b = select_segments(build_items(reqs, srt), 40, 10).skip
    assert a == b

    # 6. нет склейки в один кусок: 20 сегментов по 6 с (120 с), окно 30 с -> 4 окна,
    #    40% -> 48 с, по 12 с на окно; в каждом окне есть выбранные
    reqs, srt = _synth()
    items = build_items(reqs, srt)
    r = select_segments(items, 40, 10, window_seconds=30)
    assert r.n_windows == 4
    assert all(c > 0 for c, _, _ in r.per_window), r.per_window
    assert r.selected_ms <= r.budget_ms + 1e-6

    # 7. пустые/нулевые сегменты и vv=0 не выбираются
    reqs, srt = _synth(texts=["т"] * 10 + [""] + ["т"] * 9, vv=[10] * 10 + [99] + [0] + [10] * 8)
    r = select_segments(build_items(reqs, srt), 60, 0)
    assert r.skip[11] and r.skip[12]

    print("selftest: все проверки пройдены")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _percent_type(v: str) -> int:
    try:
        p = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"percent должен быть целым 1-100, получено {v!r}")
    if not 1 <= p <= 100:
        raise argparse.ArgumentTypeError(f"percent должен быть в диапазоне 1-100, получено {p}")
    return p


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Проставляет skip в requests.json по проценту покрытия.")
    p.add_argument("requests", nargs="?", default=os.environ.get("COVERAGE_REQUESTS"),
                   help="путь к requests.json (env COVERAGE_REQUESTS)")
    p.add_argument("srt", nargs="?", default=os.environ.get("COVERAGE_SRT"),
                   help="путь к .srt (env COVERAGE_SRT)")
    p.add_argument("--percent", type=_percent_type,
                   default=_percent_type(os.environ.get("COVERAGE_PERCENT", str(DEFAULT_PERCENT))))
    p.add_argument("--intro-seconds", type=float,
                   default=float(os.environ.get("INTRO_SECONDS", DEFAULT_INTRO_SECONDS)))
    p.add_argument("--window-seconds", type=float,
                   default=float(os.environ.get("WINDOW_SECONDS", DEFAULT_WINDOW_SECONDS)),
                   help="целевая длина окна, с (число окон = round(длительность / это), минимум 1)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.selftest:
        selftest()
        return 0
    if not args.requests or not args.srt:
        p.error("нужны пути к requests.json и .srt (аргументами или COVERAGE_REQUESTS/COVERAGE_SRT)")
    if args.intro_seconds < 0 or args.window_seconds <= 0:
        p.error("--intro-seconds >= 0, --window-seconds > 0")

    try:
        result = run(args.requests, args.srt, args.percent, args.intro_seconds, args.window_seconds)
    except (CoverageError, ValueError, OSError, json.JSONDecodeError) as e:
        log.error("Ошибка: %s", e)
        return 1
    log.info(format_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
