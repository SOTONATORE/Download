"""Разбор SRT с строгой валидацией.

Чистый модуль: только стандартная библиотека, без сети и БД.
Время хранится целыми миллисекундами.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    num: int        # номер из SRT, он же номер файла
    start_ms: int
    end_ms: int
    text: str       # нормализованный текст (теги убраны, пробелы схлопнуты), может быть ""


class SrtError(ValueError):
    """Файл SRT повреждён или не соответствует формату."""


_T = r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
# строка таймингов; хвост после конечного времени (координаты X1:..) игнорируем
_TIMING_RE = re.compile(rf"^\s*{_T}\s*-->\s*{_T}(?:\s.*)?$")
_BLOCK_SPLIT_RE = re.compile(r"\n(?:[ \t]*\n)+")
_TAG_ANGLE_RE = re.compile(r"<[^>]+>")      # <i>, </b>, <font ...>
_TAG_BRACE_RE = re.compile(r"\{[^}]*\}")    # {\an8}
_SPACES_RE = re.compile(r"\s+")


def _to_ms(h: str, m: str, s: str, ms: str) -> int | None:
    if int(m) >= 60 or int(s) >= 60:
        return None
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def _normalize_text(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = _TAG_ANGLE_RE.sub("", text)
    text = _TAG_BRACE_RE.sub("", text)
    return _SPACES_RE.sub(" ", text).strip()


def _preview(block: str) -> str:
    one = _SPACES_RE.sub(" ", block).strip()
    return one[:50] + ("…" if len(one) > 50 else "")


def parse_srt_text(raw: str) -> list[Segment]:
    """Разбор SRT из строки. При любом нарушении формата бросает SrtError."""
    raw = raw.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        raise SrtError("Файл SRT пуст: нет ни одного сегмента.")

    segments: list[Segment] = []
    bad_blocks: list[str] = []
    reversed_nums: list[int] = []

    for k, block in enumerate(_BLOCK_SPLIT_RE.split(raw), start=1):
        lines = block.split("\n")
        first = lines[0].strip()
        m = _TIMING_RE.match(lines[1]) if len(lines) > 1 else None
        if not first.isdigit() or m is None:
            bad_blocks.append(f"блок №{k} по порядку в файле: «{_preview(block)}»")
            continue
        g = m.groups()
        start = _to_ms(*g[0:4])
        end = _to_ms(*g[4:8])
        if start is None or end is None:
            bad_blocks.append(f"блок №{k} (номер {int(first)}): минуты/секунды ≥ 60")
            continue
        num = int(first)
        if end < start:
            reversed_nums.append(num)
        segments.append(Segment(num, start, end, _normalize_text(lines[2:])))

    errors: list[str] = []
    if bad_blocks:
        shown = "; ".join(bad_blocks[:10])
        more = f" (и ещё {len(bad_blocks) - 10})" if len(bad_blocks) > 10 else ""
        errors.append(f"Не удалось разобрать блоков: {len(bad_blocks)}: {shown}{more}")

    if reversed_nums:
        errors.append(
            f"Конец раньше начала (end < start) в сегментах: {sorted(reversed_nums)}"
        )

    segments.sort(key=lambda s: s.num)

    seen: set[int] = set()
    dups: set[int] = set()
    for s in segments:
        (dups if s.num in seen else seen).add(s.num)
    if dups:
        errors.append(f"Дублирующиеся номера сегментов: {sorted(dups)}")

    # пропуски проверяем только если все блоки разобраны, иначе будет шум
    if segments and not bad_blocks:
        missing = sorted(set(range(segments[0].num, segments[-1].num + 1)) - seen)
        if missing:
            errors.append(
                f"Пропущены номера сегментов: {missing} "
                f"(диапазон {segments[0].num}..{segments[-1].num})"
            )

    if errors:
        raise SrtError("SRT повреждён. " + " | ".join(errors))
    return segments


def parse_srt(path: str) -> list[Segment]:
    """Читает файл (utf-8-sig) и разбирает его."""
    with open(path, "r", encoding="utf-8-sig", newline=None) as f:
        raw = f.read()
    return parse_srt_text(raw)


def srt_hash(segments: list[Segment]) -> str:
    """sha256 по num|start|end|text. Сегменты разделены «\\n» (в тексте его нет)."""
    h = hashlib.sha256()
    for s in segments:
        h.update(f"{s.num}|{s.start_ms}|{s.end_ms}|{s.text}\n".encode("utf-8"))
    return h.hexdigest()


def check_srt(segments: list[Segment]) -> list[str]:
    """Предупреждения (не ошибки): перекрытия, пустые тексты, нулевая длина."""
    warns: list[str] = []
    for i, s in enumerate(segments):
        if not s.text:
            warns.append(f"Сегмент {s.num}: пустой текст.")
        if s.end_ms == s.start_ms:
            warns.append(f"Сегмент {s.num}: нулевая длительность.")
        if i + 1 < len(segments):
            nxt = segments[i + 1]
            if nxt.start_ms < s.end_ms:
                warns.append(
                    f"Сегмент {nxt.num}: перекрывается с предыдущим ({s.num}) "
                    f"на {s.end_ms - nxt.start_ms} мс."
                )
    return warns
