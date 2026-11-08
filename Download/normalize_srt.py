#!/usr/bin/env python3
"""normalize_srt.py - склейка коротких соседних реплик .srt в куски заданной длины.

Алгоритм:
  1. Текстовые реплики делятся на блоки: пустая реплика разделяет блоки и не склеивается.
  2. Для каждого блока динамическое программирование (C5) ищет разбиение с минимальной
     суммарной стоимостью среди допустимых кусков длиной [SEG_MIN, SEG_MAX] секунд.
  3. Локальное уточнение (C4): каждая внутренняя граница сдвигается на одну реплику,
     если суммарная стоимость двух соседних кусков уменьшается.
  Если блок нельзя разбить допустимо, выводится предупреждение и блок склеивается жадно.

Запуск:  python normalize_srt.py INPUT.srt OUTPUT.srt
         python normalize_srt.py --selftest
"""
from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

# --- формат SRT и закрывающие символы: скопировано из generate_queries.py ---
SRT_BLOCK_RE = re.compile(
    r"(?P<index>\d+)\s*\n"
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})[^\n]*\n"
    r"(?P<text>.*?)(?=\n\s*\n\d+\s*\n|\Z)",
    re.DOTALL,
)
SENTENCE_CLOSERS = "\"'\u00bb\u201d\u2019)]"
SENTENCE_END_CHARS = ".!?\u2026"

# --- параметры склейки (секунды) ---
SEG_MIN = 5.0            # нижняя граница длины склеенного куска
SEG_MAX = 8.0            # верхняя граница длины склеенного куска
SEG_TARGET = 8.0         # желаемая длина: стоимость растёт как квадрат отклонения от неё
SENTENCE_PENALTY = 2.0   # штраф (сек^2) за конец куска не в конце предложения
REFINE_PASSES = 20       # предел проходов локального уточнения (C4)
COST_EPS = 1e-9          # улучшение меньше этого не принимается (защита от шума float)

SEG_MIN_MS = round(SEG_MIN * 1000)
SEG_MAX_MS = round(SEG_MAX * 1000)


def _strip_closers(text: str) -> str:
    return text.rstrip(SENTENCE_CLOSERS + " ")


@dataclass
class Cue:
    index: int       # исходный номер
    start: int       # мс
    end: int         # мс
    text: str        # нормализованный текст ("" для пустой реплики)

    @property
    def dur(self) -> int:
        return self.end - self.start


def ts_to_ms(ts: str) -> int:
    h, m, rest = ts.split(":")
    s, ms = re.split(r"[,.]", rest)
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def ms_to_ts(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_cues(raw: str) -> list[Cue]:
    raw = raw.replace("\r\n", "\n").lstrip("\ufeff").strip() + "\n\n"
    cues = []
    for m in SRT_BLOCK_RE.finditer(raw):
        # текст нормализуется так же, как в parse_srt (теги не трогаем - см. ТЗ: только пробелы)
        text = re.sub(r"\s+", " ", m.group("text")).strip()
        cues.append(Cue(int(m.group("index")), ts_to_ms(m.group("start")),
                        ts_to_ms(m.group("end")), text))
    cues.sort(key=lambda c: c.index)
    return cues


def ends_sentence(text: str) -> bool:
    core = _strip_closers(text)
    return bool(core) and core[-1] in SENTENCE_END_CHARS


# ---------------------------------------------------------------------------
# Оптимизатор склейки: C5 (глобальное ДП) + C4 (локальное уточнение границ)

def _seg_cost(block: list[Cue], s: int, k: int) -> float:
    """Стоимость куска block[s:k] (полуинтервал по индексам); inf - кусок недопустим.

    dur - секунды от начала первой реплики куска до конца последней (паузы внутри входят).
    cost = (SEG_TARGET - dur)^2 + SENTENCE_PENALTY, если последняя реплика куска
    не заканчивает предложение; иначе cost = (SEG_TARGET - dur)^2.
    Допустимо только при SEG_MIN <= dur <= SEG_MAX.
    """
    dur_ms = block[k - 1].end - block[s].start
    if dur_ms < SEG_MIN_MS or dur_ms > SEG_MAX_MS:
        return math.inf
    dur = dur_ms / 1000.0
    cost = (SEG_TARGET - dur) ** 2
    if not ends_sentence(block[k - 1].text):
        cost += SENTENCE_PENALTY
    return cost


def _dp_partition(block: list[Cue]) -> tuple[Optional[list[tuple[int, int]]], int]:
    """C5: глобальное ДП по префиксам блока.

    Возвращает (границы кусков или None, последняя достижимая точка префикса).
    None - разбиения, где все куски допустимы, не существует; тогда вторая часть
    результата - индекс первой реплики, от которой дальше нельзя собрать кусок.
    Перебор концов ограничен: из начала s рассматриваются только концы k, пока
    span <= SEG_MAX (концы по времени не убывают, поэтому можно остановиться).
    При равной стоимости выигрывает меньший индекс начала (внешний цикл по s возрастает).
    """
    n = len(block)
    best = [math.inf] * (n + 1)
    back = [-1] * (n + 1)
    best[0] = 0.0
    for s in range(n):
        if best[s] == math.inf:
            continue
        k = s + 1
        while k <= n:
            if block[k - 1].end - block[s].start > SEG_MAX_MS:
                break
            c = best[s] + _seg_cost(block, s, k)
            if c < best[k]:
                best[k] = c
                back[k] = s
            k += 1
    reach = max(k for k in range(n + 1) if best[k] < math.inf)
    if best[n] == math.inf:
        return None, reach
    bounds: list[tuple[int, int]] = []
    k = n
    while k > 0:
        s = back[k]
        bounds.append((s, k))
        k = s
    bounds.reverse()
    return bounds, n


def _refine(block: list[Cue], bounds: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """C4: сдвиг каждой внутренней границы на одну реплику влево или вправо.

    Для границы b между кусками [a0, b) и [b, c) новая граница b' = b-1 или b+1.
    Изменение суммарной стоимости: delta = cost(a0,b') + cost(b',c) - cost(a0,b) - cost(b,c).
    Ход принимается, если delta < -COST_EPS; при равенстве сначала проверяется b-1.
    Проходы повторяются до отсутствия изменений или до REFINE_PASSES.
    """
    segs = list(bounds)
    for _ in range(REFINE_PASSES):
        changed = False
        for t in range(len(segs) - 1):
            a0, b = segs[t]
            _, c = segs[t + 1]
            base = _seg_cost(block, a0, b) + _seg_cost(block, b, c)
            best_val, best_b = base - COST_EPS, None
            for nb in (b - 1, b + 1):
                if nb <= a0 or nb >= c:
                    continue
                val = _seg_cost(block, a0, nb) + _seg_cost(block, nb, c)
                if val < best_val:
                    best_val, best_b = val, nb
            if best_b is not None:
                segs[t] = (a0, best_b)
                segs[t + 1] = (best_b, c)
                changed = True
        if not changed:
            break
    return segs


def _greedy_bounds(block: list[Cue]) -> list[tuple[int, int]]:
    """Запасной жадный вариант: закрыть кусок, как только он >= SEG_MIN и не переходить SEG_MAX.

    Одиночная реплика длиннее SEG_MAX остаётся отдельным куском (ошибка уже выведена).
    Хвост короче SEG_MIN приклеивается к предыдущему куску, если это не превышает SEG_MAX.
    """
    bounds: list[tuple[int, int]] = []
    s = 0
    n = len(block)
    for k in range(n):
        if k > s and block[k].end - block[s].start > SEG_MAX_MS:
            bounds.append((s, k))
            s = k
        if block[k].end - block[s].start >= SEG_MIN_MS:
            bounds.append((s, k + 1))
            s = k + 1
    if s < n:
        if bounds and block[n - 1].end - block[bounds[-1][0]].start <= SEG_MAX_MS:
            bounds[-1] = (bounds[-1][0], n)
        else:
            bounds.append((s, n))
    return bounds


def _build_segment(block: list[Cue], s: int, k: int) -> Cue:
    """Склеивает block[s:k]: начало первой, конец последней, тексты через один пробел."""
    return Cue(block[s].index, block[s].start, block[k - 1].end,
               " ".join(c.text for c in block[s:k]))


def _partition_block(block: list[Cue], warn) -> list[Cue]:
    """Разбивает блок текстовых реплик: сначала ДП + уточнение, при неудаче - жадно с предупреждением."""
    bounds, reach = _dp_partition(block)
    if bounds is None:
        bad = block[reach]
        if warn:
            warn(f"первая проблемная реплика №{bad.index}: от неё нельзя собрать кусок "
                 f"{SEG_MIN:g}-{SEG_MAX:g} с; блок склеен жадно, куски этого блока могут выйти за диапазон")
        return [_build_segment(block, s, k) for s, k in _greedy_bounds(block)]
    bounds = _refine(block, bounds)
    return [_build_segment(block, s, k) for s, k in bounds]


def merge_cues(cues: list[Cue], warn=None) -> list[Cue]:
    """Склейка реплик. Пустые реплики остаются отдельно на своих местах и не склеиваются с текстовыми."""
    out: list[Cue] = []
    block: list[Cue] = []
    for c in cues:
        if not c.text:
            if block:
                out.extend(_partition_block(block, warn))
                block = []
            out.append(Cue(c.index, c.start, c.end, ""))
        else:
            block.append(c)
    if block:
        out.extend(_partition_block(block, warn))
    return out


def render(segs: list[Cue]) -> str:
    return "".join(f"{i}\n{ms_to_ts(s.start)} --> {ms_to_ts(s.end)}\n{s.text}\n\n"
                   for i, s in enumerate(segs, 1))


def validate_params(min_s: float, target_s: float, max_s: float) -> None:
    if not (min_s > 0 and target_s > 0 and max_s > 0):
        raise ValueError("все параметры должны быть > 0")
    if not (min_s <= target_s <= max_s):
        raise ValueError(f"нужно min <= target <= max, получено {min_s} / {target_s} / {max_s}")


def summary(n_in, segs, n_warn) -> str:
    durs = [s.dur for s in segs if s.text]
    lines = [f"Параметры: MIN={SEG_MIN} TARGET={SEG_TARGET} MAX={SEG_MAX} (сек)",
             f"Реплик на входе: {n_in}; сегментов на выходе: {len(segs)}"]
    if durs:
        lines.append(f"Длительность, с: мин {min(durs)/1000:.3f} / медиана "
                     f"{statistics.median(durs)/1000:.3f} / макс {max(durs)/1000:.3f}")
        lines.append(f"Короче 2 с: {sum(d < 2000 for d in durs)}; ровно на потолке: "
                     f"{sum(d == SEG_MAX_MS for d in durs)}; вне [MIN, MAX]: "
                     f"{sum(d < SEG_MIN_MS or d > SEG_MAX_MS for d in durs)}")
    lines.append(f"Предупреждений о нарушении диапазона: {n_warn}")
    return "\n".join(lines)


def run(inp: str, outp: str) -> int:
    validate_params(SEG_MIN, SEG_TARGET, SEG_MAX)
    with open(inp, "r", encoding="utf-8-sig") as f:
        cues = parse_cues(f.read())
    if not cues:
        raise ValueError(f"Не удалось распарсить ни одного сегмента из {inp}")
    warns: list[str] = []

    def warn(msg):
        warns.append(msg)
        print("ВНИМАНИЕ:", msg, file=sys.stderr)

    segs = merge_cues(cues, warn)
    d = os.path.dirname(os.path.abspath(outp))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(render(segs))
        os.replace(tmp, outp)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    print(summary(len(cues), segs, len(warns)))
    return 0


# ---------------------------------------------------------------------------
def selftest() -> int:
    import random
    import time

    def C(i, s, e, t):
        return Cue(i, round(s * 1000), round(e * 1000), t)

    def m(cs, warn=None):
        return merge_cues(cs, warn)

    # 1. оптимум: 10 реплик по 1 с -> ровно два куска по 5 с (других допустимых разбиений нет)
    cs = [C(i, i - 1, i, f"w{i}") for i in range(1, 11)]
    r = m(cs)
    assert [s.dur for s in r] == [5000, 5000], r
    assert r[0].text == " ".join(f"w{i}" for i in range(1, 6)), r
    assert r[1].text == " ".join(f"w{i}" for i in range(6, 11)), r
    # 2. выбор границы по стоимости: 13 реплик по 1 с; (6 с, 7 с) с концом "w6." дают минимум 5
    cs = [C(i, i - 1, i, f"w{i}" + ("." if i in (6, 13) else "")) for i in range(1, 14)]
    r = m(cs)
    assert [s.dur for s in r] == [6000, 7000], r
    assert r[0].text.endswith("w6.") and r[1].text.startswith("w7"), r
    # 3. пустая реплика разделяет блоки и сохраняется на своём месте
    cs2 = cs + [C(14, 13, 13.5, "")] + [C(15 + k, 13.5 + k, 14.5 + k, f"x{k}") for k in range(10)]
    r = m(cs2)
    assert len(r) == 5 and r[2].text == "" and r[2].start == 13000 and r[2].end == 13500, r
    assert [s.dur for s in r if s.text] == [6000, 7000, 5000, 5000], r
    # 4. одиночная реплика длиннее потолка: предупреждение с №1, реплика остаётся как есть
    w = []
    r = m([C(1, 0, 9, "long"), C(2, 9, 10, "b")], w.append)
    assert r[0].text == "long" and r[0].dur == 9000 and len(w) == 1 and "№1" in w[0], (r, w)
    # 5. невозможное разбиение: блок 3 с (меньше SEG_MIN) -> жадно и предупреждение с №1
    w = []
    r = m([C(1, 0, 1, "a"), C(2, 1, 2, "b"), C(3, 2, 3, "c")], w.append)
    assert len(r) == 1 and r[0].text == "a b c" and r[0].dur == 3000 and len(w) == 1 and "№1" in w[0], (r, w)
    # 6. свойства на синтетике: детерминизм, текст и порядок, диапазон (или предупреждение)
    rnd = random.Random(1)
    cs, t = [], 0.0
    for i in range(1, 200):
        d = rnd.uniform(0.4, 3.5)
        txt = "" if i % 37 == 0 else f"w{i}" + ("." if i % 5 == 0 else "")
        cs.append(C(i, t, t + d, txt))
        t += d + rnd.choice([0, 0.06, 0.5])
    cs = [Cue(c.index, int(c.start), int(c.end), c.text) for c in cs]
    w1, w2 = [], []
    r1, r2 = m(cs, w1.append), m(cs, w2.append)
    assert r1 == r2 and w1 == w2
    assert render(r1) == render(r2)
    assert all(s.dur <= SEG_MAX_MS for s in r1 if s.text)
    assert all(SEG_MIN_MS <= s.dur <= SEG_MAX_MS for s in r1 if s.text) or w1, r1
    assert r1[0].start == cs[0].start and r1[-1].end == cs[-1].end
    assert " ".join(c.text for c in cs if c.text) == " ".join(s.text for s in r1 if s.text)
    out = render(r1)
    blocks = list(SRT_BLOCK_RE.finditer(out.replace("\r\n", "\n").strip() + "\n\n"))
    assert [int(b.group("index")) for b in blocks] == list(range(1, len(r1) + 1))
    assert len(blocks) == len(r1)
    back = parse_cues(out)
    assert [(c.start, c.end, c.text) for c in back] == [(s.start, s.end, s.text) for s in r1]
    # 7. 1500 реплик: время работы в пределах 5 с (ограничение на бесплатном GitHub runner)
    rnd = random.Random(2)
    big, t_ms = [], 0
    for i in range(1, 1501):
        d_ms = rnd.randint(300, 3000)
        big.append(Cue(i, t_ms, t_ms + d_ms, "" if i % 50 == 0 else f"s{i}"))
        t_ms += d_ms + rnd.choice([0, 60, 500])
    t0 = time.perf_counter()
    r3 = m(big, [].append)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"слишком медленно: {elapsed:.2f} с"
    assert " ".join(c.text for c in big if c.text) == " ".join(s.text for s in r3 if s.text)
    # 8. неверные параметры
    for bad in [(0, 1, 2), (6, 5, 8), (5, 8, 7), (-1, -1, 1)]:
        try:
            validate_params(*bad)
        except ValueError:
            continue
        raise AssertionError(f"параметры {bad} должны давать ошибку")
    validate_params(SEG_MIN, SEG_TARGET, SEG_MAX)
    print("selftest OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Склейка коротких реплик .srt в куски заданной длины")
    ap.add_argument("input", nargs="?")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.input or not a.output:
        ap.error("нужны INPUT.srt и OUTPUT.srt")
    try:
        return run(a.input, a.output)
    except (ValueError, OSError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
