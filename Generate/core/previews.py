"""Превью: генерация промптов из файла prompts.txt на фото-модели (без Gemini, без стиля и пресета).

Формат файла: каждый промпт занимает блок строк, блоки разделяются одной или несколькими пустыми строками.
Порядок блоков задаёт номер промпта x (с 1). Проверок содержимого нет: только наличие файла и хотя бы одного промпта.

Переключатель prompts_count: «выкл» = превью не делаются; 1..4 = число вариантов y на каждый промпт.
Каждый вариант y получает свой случайный сид. Итог: файлы preview_<x>_<y>.png в итоговом релизе и
служебный previews_index.txt со всеми деталями (промпт, y, реальный сид, карта, статус).
"""

from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

# синтетические номера задач превью: выше любого номера клипа из SRT, не пересекаются с ними
PREVIEW_NUM_BASE = 1_000_000_000
PREVIEW_COUNT_OFF = "выкл"
PREVIEW_COUNT_CHOICES = (PREVIEW_COUNT_OFF, "1", "2", "3", "4")
PREVIEW_INDEX_FILENAME = "previews_index.txt"
SEED_LIMIT = 2 ** 32  # сид генерации: 32-битное беззнаковое число
REF_SLOTS = 10        # номера референсов 1…10 (как REF_SLOTS воркера)
REFS_LINE_RX = re.compile(r"^refs:\s*(.*?)\s*$", re.IGNORECASE)  # первая строка блока: refs: 1,6
LABEL_RX = re.compile(r"<image(\d+)>")

STATUS_QUEUED = "в очереди"
STATUS_RUNNING = "идёт генерация"
STATUS_DONE = "готово"
STATUS_FAILED = "ошибка"


class PreviewInputError(ValueError):
    """Ошибка входа для превью: сообщение уже на русском, понятно пользователю workflow."""


@dataclass
class PreviewItem:
    num: int           # синтетический номер задачи (>= PREVIEW_NUM_BASE)
    x: int             # номер промпта в файле, с 1
    y: int             # номер варианта, с 1 (из переключателя, не сид)
    prompt: str
    seed: int          # случайный сид этой генерации
    filename: str      # preview_<x>_<y>.png
    status: str = STATUS_QUEUED
    card: str = ""     # карта, на которой генерировали (для индекса)
    refs: Optional[tuple] = None  # номера референсов блока (отсортированы); None = блок без refs: как раньше


def count_value(value: Optional[str]) -> int:
    """Переключатель в число вариантов: «выкл» и пустое значение = 0."""
    text = (value or "").strip()
    if text in ("", PREVIEW_COUNT_OFF):
        return 0
    if text not in PREVIEW_COUNT_CHOICES:
        raise PreviewInputError(f"Недопустимое значение переключателя превью: «{text}». "
                                f"Допустимо: {', '.join(PREVIEW_COUNT_CHOICES)}.")
    return int(text)


def parse_prompts(text: str) -> List[str]:
    """Разбирает текст файла промптов: блоки строк, разделённые пустыми строками."""
    normalized = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    prompts: List[str] = []
    block: List[str] = []
    for line in normalized.split("\n"):
        if line.strip():
            block.append(line.strip())
            continue
        if block:
            prompts.append(" ".join(" ".join(block).split()))
            block = []
    if block:
        prompts.append(" ".join(" ".join(block).split()))
    return prompts


def parse_refs_line(line: str, number: int) -> tuple:
    """Строка «refs: 1,6» → отсортированный кортеж номеров. Ошибка — русский текст с номером промпта."""
    raw = line.strip()
    if not raw:
        raise PreviewInputError(f"Промпт {number}: в строке refs: нет номеров.")
    parts = [p.strip() for p in raw.split(",")]
    nums = []
    for part in parts:
        if not part.isdigit():
            raise PreviewInputError(f"Промпт {number}: в строке refs: «{raw}» номера должны быть целыми через запятую.")
        n = int(part)
        if not 1 <= n <= REF_SLOTS:
            raise PreviewInputError(f"Промпт {number}: номер референса {n} вне 1…{REF_SLOTS}.")
        nums.append(n)
    if len(set(nums)) != len(nums):
        raise PreviewInputError(f"Промпт {number}: в строке refs: номера повторяются.")
    return tuple(sorted(nums))


def parse_prompt_specs(text: str) -> List[Tuple[str, Optional[tuple]]]:
    """Блоки файла промптов: (текст, refs). Первая строка блока «refs: ...» выносится из текста.

    Блок без такой строки даёт (текст, None) — старый формат, результат тот же, что у parse_prompts.
    """
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    blocks: List[List[str]] = []
    block: List[str] = []
    for line in normalized.split("\n"):
        if line.strip():
            block.append(line.strip())
            continue
        if block:
            blocks.append(block)
            block = []
    if block:
        blocks.append(block)
    specs: List[Tuple[str, Optional[tuple]]] = []
    for number, lines in enumerate(blocks, 1):
        refs = None
        m = REFS_LINE_RX.match(lines[0])
        if m:
            refs = parse_refs_line(m.group(1), number)
            lines = lines[1:]
            if not lines:
                raise PreviewInputError(f"Промпт {number}: после строки refs: нет текста промпта.")
        specs.append((" ".join(" ".join(lines).split()), refs))
    return specs


def load_prompt_specs(path: str) -> List[Tuple[str, Optional[tuple]]]:
    """Читает файл промптов: (текст, refs) для каждого блока. Ошибка, если файла нет, он не в UTF-8 или промптов нет."""
    if not path or not os.path.isfile(path):
        raise PreviewInputError(f"Файл промптов не найден: {path or '(путь не задан)'}.")
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            text = fh.read()
    except UnicodeDecodeError:
        raise PreviewInputError(f"Файл промптов {path} не в кодировке UTF-8.") from None
    except OSError:
        raise PreviewInputError(f"Файл промптов {path} не удалось прочитать.") from None
    specs = parse_prompt_specs(text)
    if not specs:
        raise PreviewInputError(f"В файле промптов {path} нет ни одного промпта.")
    return specs


def load_prompts(path: str) -> List[str]:
    """Читает файл промптов. Ошибка, если файла нет, он не в UTF-8 или промптов нет. Тексты без refs."""
    return [text for text, _refs in load_prompt_specs(path)]


def check_refs(specs: Sequence[Tuple[str, Optional[tuple]]], allowed) -> None:
    """Проверка блоков с refs: сочетание разрешено профилем (allowed — набор кортежей или None) и метки <imageN>
    совпадают с позициями 1…len(refs). Блоки без refs не трогаются. Ошибка — русский текст с номером промпта."""
    for number, (text, refs) in enumerate(specs, 1):
        if refs is None:
            continue
        if allowed is not None and refs not in allowed:
            raise PreviewInputError(f"Промпт {number}: сочетание refs {list(refs)} не разрешено профилем ниши.")
        found = sorted({int(m) for m in LABEL_RX.findall(text)})
        expected = list(range(1, len(refs) + 1))
        if found != expected:
            tags = ", ".join(f"<image{n}>" for n in found) or "нет"
            raise PreviewInputError(
                f"Промпт {number}: метки {tags} не совпадают с refs {list(refs)}: "
                f"нужны <image1>…<image{len(refs)}>, по одной на каждый референс.")


def random_seed() -> int:
    return secrets.randbelow(SEED_LIMIT)


def preview_filename(x: int, y: int) -> str:
    return f"preview_{x}_{y}.png"


def build_previews(prompts: Sequence[Union[str, Tuple[str, Optional[tuple]]]], count: int,
                   seed_fn: Callable[[], int] = random_seed) -> List[PreviewItem]:
    """Список задач превью: для каждого промпта x все варианты y от 1 до count, в порядке x, затем y.

    prompts: тексты (старый вид) или пары (текст, refs) из load_prompt_specs.
    """
    items: List[PreviewItem] = []
    if count <= 0:
        return items
    for x, prompt in enumerate(prompts, 1):
        text, refs = (prompt, None) if isinstance(prompt, str) else prompt
        for y in range(1, count + 1):
            items.append(PreviewItem(
                num=PREVIEW_NUM_BASE + (x - 1) * count + (y - 1),
                x=x, y=y, prompt=text, seed=seed_fn(), filename=preview_filename(x, y), refs=refs))
    return items


def is_preview_num(num: int) -> bool:
    return num >= PREVIEW_NUM_BASE


def index_text(items: List[PreviewItem]) -> str:
    """Содержимое previews_index.txt: по одной записи на превью, промпт в одну строку."""
    lines = [
        "previews_index: превью по промптам из файла prompts.txt",
        "y = номер варианта из переключателя (1..4); seed = реальный случайный сид генерации",
        "",
    ]
    for item in items:
        refs_part = f"refs: {','.join(map(str, item.refs))} | " if item.refs else ""
        lines.append(
            f"{item.filename} | промпт {item.x} | y {item.y} | seed {item.seed} | "
            f"карта: {item.card or '—'} | статус: {item.status} | {refs_part}промпт: {item.prompt}")
    if not items:
        lines.append("превью не запускались")
    return "\n".join(lines) + "\n"
