"""
Утилиты именования клипов (Generate/core/naming.py).

Следует шаблону Фазы 1 (Download/download.py): файлы именуются как
"{num}.{ext}" без ведущих нулей по умолчанию. Здесь: шаблон "{num}.mp4",
ширина номера настраивается через clip_name_width (SPEC.md, раздел 3).
"""

from __future__ import annotations

import os
import re
from typing import Optional


def clip_filename(num: int, width: int = 0, template: str = "{num}.mp4") -> str:
    """
    Формирует имя файла клипа по номеру сегмента.

    Если width <= 0, номер форматируется без ведущих нулей (7.mp4, 42.mp4).
    Если width > 0, номер дополняется нулями слева до указанной ширины
    (0007.mp4 при width=4).

    Вызывает ValueError с сообщением на русском языке, если num отрицательный.
    """
    if num < 0:
        raise ValueError(f"Номер клипа не может быть отрицательным: {num}")

    if width > 0:
        num_str = str(num).zfill(width)
    else:
        num_str = str(num)

    return template.format(num=num_str)


def _template_to_regex(template: str) -> re.Pattern:
    """
    Преобразует шаблон имени файла (с плейсхолдером {num}) в регулярное
    выражение для разбора имени файла обратно в номер.
    """
    placeholder = "{num}"
    if placeholder not in template:
        raise ValueError(
            f"Шаблон имени файла должен содержать плейсхолдер {{num}}: {template!r}"
        )

    prefix, suffix = template.split(placeholder, 1)
    pattern = re.escape(prefix) + r"(\d+)" + re.escape(suffix)
    return re.compile(r"^" + pattern + r"$")


def parse_clip_num(filename: str, template: str = "{num}.mp4") -> Optional[int]:
    """
    Извлекает номер сегмента (целое число) из имени файла клипа.

    Принимает как голое имя файла, так и путь (используется только
    базовое имя файла). Возвращает None, если имя файла не соответствует
    шаблону клипа (например, служебные файлы prompts.json,
    generated_links.txt и т.п.).
    """
    basename = os.path.basename(filename)
    regex = _template_to_regex(template)
    match = regex.match(basename)
    if match is None:
        return None

    try:
        return int(match.group(1))
    except ValueError:
        return None
