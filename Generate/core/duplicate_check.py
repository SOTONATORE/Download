"""Поиск одинаковых промптов между клипами (чистая функция: без ввода-вывода и без состояния).

Используется только профилями с ключом fail_on_duplicate_prompts (см. verify_prompts.yml). Для остальных
профилей функция не вызывается.
"""
from __future__ import annotations

import re
from typing import Hashable, List, Mapping

_SPACES = re.compile(r"\s+")


def normalize_prompt(text: str) -> str:
    """Нормализация для сравнения: обрезка краёв, схлопывание пробелов и переводов строк, нижний регистр."""
    return _SPACES.sub(" ", text.strip()).lower()


def find_duplicate_prompts(prompts: Mapping[Hashable, str]) -> List[list]:
    """Группы ключей (номеров клипов), у которых нормализованный текст промпта совпадает.

    Возвращаются только группы из двух и более ключей; порядок групп и ключей внутри — по первому появлению.
    Пустые промпты (и не строки) не сравниваются: пустой промпт — отдельная проблема «промпт не готов».
    """
    groups: dict = {}
    for key, text in prompts.items():
        if not isinstance(text, str):
            continue
        norm = normalize_prompt(text)
        if not norm:
            continue
        groups.setdefault(norm, []).append(key)
    return [ids for ids in groups.values() if len(ids) >= 2]
