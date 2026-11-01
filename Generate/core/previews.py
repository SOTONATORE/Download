"""Превью: генерация промптов из файла prompts.txt на фото-модели (без Gemini, без стиля и пресета).

Формат файла: каждый промпт занимает блок строк, блоки разделяются одной или несколькими пустыми строками.
Порядок блоков задаёт номер промпта x (с 1). Проверок содержимого нет: только наличие файла и хотя бы одного промпта.

Переключатель prompts_count: «выкл» = превью не делаются; 1..4 = число вариантов y на каждый промпт.
Каждый вариант y получает свой случайный сид. Итог: файлы preview_<x>_<y>.png в итоговом релизе и
служебный previews_index.txt со всеми деталями (промпт, y, реальный сид, карта, статус).
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Callable, List, Optional

# синтетические номера задач превью: выше любого номера клипа из SRT, не пересекаются с ними
PREVIEW_NUM_BASE = 1_000_000_000
PREVIEW_COUNT_OFF = "выкл"
PREVIEW_COUNT_CHOICES = (PREVIEW_COUNT_OFF, "1", "2", "3", "4")
PREVIEW_INDEX_FILENAME = "previews_index.txt"
SEED_LIMIT = 2 ** 32  # сид генерации: 32-битное беззнаковое число

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


def load_prompts(path: str) -> List[str]:
    """Читает файл промптов. Ошибка, если файла нет, он не в UTF-8 или промптов нет."""
    if not path or not os.path.isfile(path):
        raise PreviewInputError(f"Файл промптов не найден: {path or '(путь не задан)'}.")
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            text = fh.read()
    except UnicodeDecodeError:
        raise PreviewInputError(f"Файл промптов {path} не в кодировке UTF-8.") from None
    except OSError:
        raise PreviewInputError(f"Файл промптов {path} не удалось прочитать.") from None
    prompts = parse_prompts(text)
    if not prompts:
        raise PreviewInputError(f"В файле промптов {path} нет ни одного промпта.")
    return prompts


def random_seed() -> int:
    return secrets.randbelow(SEED_LIMIT)


def preview_filename(x: int, y: int) -> str:
    return f"preview_{x}_{y}.png"


def build_previews(prompts: List[str], count: int,
                   seed_fn: Callable[[], int] = random_seed) -> List[PreviewItem]:
    """Список задач превью: для каждого промпта x все варианты y от 1 до count, в порядке x, затем y."""
    items: List[PreviewItem] = []
    if count <= 0:
        return items
    for x, prompt in enumerate(prompts, 1):
        for y in range(1, count + 1):
            items.append(PreviewItem(
                num=PREVIEW_NUM_BASE + (x - 1) * count + (y - 1),
                x=x, y=y, prompt=prompt, seed=seed_fn(), filename=preview_filename(x, y)))
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
        lines.append(
            f"{item.filename} | промпт {item.x} | y {item.y} | seed {item.seed} | "
            f"карта: {item.card or '—'} | статус: {item.status} | промпт: {item.prompt}")
    if not items:
        lines.append("превью не запускались")
    return "\n".join(lines) + "\n"
