#!/usr/bin/env python3
"""normalize_srt.py - склейка коротких соседних реплик .srt в более длинные куски.

Запуск:  python normalize_srt.py INPUT.srt OUTPUT.srt [--min-seconds 3.5]
             [--sentence-min-seconds 2.0] [--max-seconds 5.0]
         python normalize_srt.py --selftest
"""
from __future__ import annotations

import argparse
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

DEFAULT_MIN = 3.5
DEFAULT_SENTENCE_MIN = 2.0
DEFAULT_MAX = 5.0


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


def merge_cues(cues: list[Cue], min_ms: int, sent_min_ms: int, max_ms: int,
               warn=None) -> list[Cue]:
    out: list[Cue] = []
    cur: Optional[Cue] = None
    # Признак "последний элемент out - непустой сегмент, не отделённый пустым":
    # пустые сегменты лежат в out как Cue с text == "", проверяем out[-1].text.

    def close():
        nonlocal cur
        if cur is not None:
            out.append(cur)
            cur = None

    for c in cues:
        if not c.text:
            close()
            out.append(Cue(c.index, c.start, c.end, ""))
            continue
        if c.dur > max_ms and warn:
            warn(f"реплика №{c.index} длиннее потолка: {c.dur / 1000:.3f} с > {max_ms / 1000:.3f} с "
                 f"(остаётся как есть)")
        if cur is None:
            cur = Cue(c.index, c.start, c.end, c.text)
        elif c.end - cur.start > max_ms:
            close()
            cur = Cue(c.index, c.start, c.end, c.text)
        else:
            cur = Cue(cur.index, cur.start, c.end, cur.text + " " + c.text)
        d = cur.dur
        if d >= min_ms or (d >= sent_min_ms and ends_sentence(cur.text)):
            close()

    if cur is not None:
        prev = out[-1] if out else None
        # Берём полный охват (с паузой между ними), т.к. пауза поглощается - так потолок точно не превысится.
        if prev is not None and prev.text and cur.end - prev.start <= max_ms:
            out[-1] = Cue(prev.index, prev.start, cur.end, prev.text + " " + cur.text)
        else:
            out.append(cur)
        cur = None
    return out


def render(segs: list[Cue]) -> str:
    return "".join(f"{i}\n{ms_to_ts(s.start)} --> {ms_to_ts(s.end)}\n{s.text}\n\n"
                   for i, s in enumerate(segs, 1))


def validate_params(min_s: float, sent_s: float, max_s: float) -> None:
    if not (min_s > 0 and sent_s > 0 and max_s > 0):
        raise ValueError("все параметры должны быть > 0")
    if not (sent_s <= min_s <= max_s):
        raise ValueError(f"нужно sentence_min <= min <= max, получено {sent_s} / {min_s} / {max_s}")


def to_ms(sec: float) -> int:
    return int(round(sec * 1000))


def summary(n_in, segs, min_s, sent_s, max_s, n_warn) -> str:
    durs = [s.dur for s in segs if s.text]
    lines = [f"Параметры: MIN={min_s} SENTENCE_MIN={sent_s} MAX={max_s} (сек)",
             f"Реплик на входе: {n_in}; сегментов на выходе: {len(segs)}"]
    if durs:
        lines.append(f"Длительность, с: мин {min(durs)/1000:.3f} / медиана "
                     f"{statistics.median(durs)/1000:.3f} / макс {max(durs)/1000:.3f}")
        lines.append(f"Короче 2 с: {sum(d < 2000 for d in durs)}; ровно на потолке: "
                     f"{sum(d == to_ms(max_s) for d in durs)}")
    lines.append(f"Предупреждений о слишком длинных одиночных репликах: {n_warn}")
    return "\n".join(lines)


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"{name}={v!r} - не число")


def run(inp: str, outp: str, min_s: float, sent_s: float, max_s: float) -> int:
    validate_params(min_s, sent_s, max_s)
    with open(inp, "r", encoding="utf-8-sig") as f:
        cues = parse_cues(f.read())
    if not cues:
        raise ValueError(f"Не удалось распарсить ни одного сегмента из {inp}")
    warns: list[str] = []

    def warn(msg):
        warns.append(msg)
        print("ВНИМАНИЕ:", msg, file=sys.stderr)

    segs = merge_cues(cues, to_ms(min_s), to_ms(sent_s), to_ms(max_s), warn)
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
    print(summary(len(cues), segs, min_s, sent_s, max_s, len(warns)))
    return 0


# ---------------------------------------------------------------------------
def selftest() -> int:
    def C(i, s, e, t):
        return Cue(i, to_ms(s) if False else int(s * 1000), int(e * 1000), t)

    MIN, SM, MAX = 3500, 2000, 5000

    def m(cs, warn=None):
        return merge_cues(cs, MIN, SM, MAX, warn)

    # 1. склейка до MIN
    r = m([C(1, 0, 1, "a"), C(2, 1, 2, "b"), C(3, 2, 3.6, "c"), C(4, 3.6, 8, "d")])
    assert r[0].text == "a b c" and r[0].dur == 3600 and r[1].text == "d", r
    # 2. конец предложения при >= SENTENCE_MIN
    r = m([C(1, 0, 1, "a"), C(2, 1, 2.2, "end."), C(3, 2.2, 3, "x"), C(4, 3, 5.5, "y")])
    assert r[0].text == "a end." and r[1].text.startswith("x"), r
    r = m([C(1, 0, 1, 'say "hi."'), C(2, 1, 1.5, "ok")])  # d=1.0<2: не закрывать
    assert r[0].text == 'say "hi." ok', r
    r = m([C(1, 0, 2.5, 'He said "stop."'), C(2, 2.5, 9, "Next")])
    assert r[0].text == 'He said "stop."', r
    # 3. потолок
    r = m([C(1, 0, 3, "a"), C(2, 3, 6, "b"), C(3, 6, 7, "c")])
    assert all(s.dur <= MAX for s in r) and len(r) >= 2, r
    # 4. пустая реплика
    r = m([C(1, 0, 1, "a"), C(2, 1, 2, ""), C(3, 2, 3, "b")])
    assert [s.text for s in r] == ["a", "", "b"], r
    # 5. хвост: приклеивается, если помещается; не приклеивается к пустой; не помещается
    r = m([C(1, 0, 3.6, "a"), C(2, 3.6, 4.2, "tail")])
    assert len(r) == 1 and r[0].text == "a tail", r
    r = m([C(1, 0, 3.6, "a"), C(2, 3.6, 4.0, ""), C(3, 4.0, 4.5, "tail")])
    assert [s.text for s in r] == ["a", "", "tail"], r
    r = m([C(1, 0, 3.6, "a"), C(2, 3.6, 5.5, "tail")])
    assert len(r) == 2, r
    # 6. одиночная реплика > MAX
    w = []
    r = m([C(1, 0, 7, "long"), C(2, 7, 8, "b")], w.append)
    assert r[0].text == "long" and r[0].dur == 7000 and len(w) == 1 and "№1" in w[0], (r, w)
    # 7-10. свойства на синтетике
    import random
    rnd = random.Random(1)
    cs, t = [], 0.0
    for i in range(1, 200):
        d = rnd.uniform(0.4, 3.5)
        txt = "" if i % 37 == 0 else f"w{i}" + ("." if i % 5 == 0 else "")
        cs.append(C(i, t, t + d, txt))
        t += d + rnd.choice([0, 0.06, 0.5])
    cs = [Cue(c.index, round(c.start), round(c.end), c.text) for c in cs]
    cs = [Cue(c.index, int(c.start), int(c.end), c.text) for c in cs]
    r1, r2 = m(cs), m(cs)
    assert r1 == r2
    out = render(r1)
    assert out == render(r2)
    assert all(s.dur <= MAX for s in r1 if s.text)
    assert r1[0].start == cs[0].start and r1[-1].end == cs[-1].end
    assert " ".join(c.text for c in cs if c.text) == " ".join(s.text for s in r1 if s.text)
    blocks = list(SRT_BLOCK_RE.finditer(out.replace("\r\n", "\n").strip() + "\n\n"))
    assert [int(b.group("index")) for b in blocks] == list(range(1, len(r1) + 1))
    assert len(blocks) == len(r1)
    back = parse_cues(out)
    assert [(c.start, c.end, c.text) for c in back] == [(s.start, s.end, s.text) for s in r1]
    # неверные параметры
    for bad in [(0, 1, 2), (3, 2, 1), (3.5, 4.0, 5.0), (-1, -1, 1)]:
        try:
            validate_params(*bad)
        except ValueError:
            continue
        raise AssertionError(f"параметры {bad} должны давать ошибку")
    validate_params(3.5, 2.0, 5.0)
    print("selftest OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Склейка коротких реплик .srt в более длинные куски")
    ap.add_argument("input", nargs="?")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--min-seconds", type=float, default=None)
    ap.add_argument("--sentence-min-seconds", type=float, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.input or not a.output:
        ap.error("нужны INPUT.srt и OUTPUT.srt")
    try:
        min_s = a.min_seconds if a.min_seconds is not None else env_float("MIN_SECONDS", DEFAULT_MIN)
        sent_s = (a.sentence_min_seconds if a.sentence_min_seconds is not None
                  else env_float("SENTENCE_MIN_SECONDS", DEFAULT_SENTENCE_MIN))
        max_s = a.max_seconds if a.max_seconds is not None else env_float("MAX_SECONDS", DEFAULT_MAX)
        return run(a.input, a.output, min_s, sent_s, max_s)
    except (ValueError, OSError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
