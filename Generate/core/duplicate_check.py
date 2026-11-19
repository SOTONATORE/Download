"""Поиск одинаковых промптов между клипами (чистая функция: без ввода-вывода и без состояния).

Используется только профилями с ключом fail_on_duplicate_prompts (см. verify_prompts.yml). Для остальных
профилей функция не вызывается.
"""
from __future__ import annotations

import re
from typing import Hashable, List, Mapping

try:  # серии одинакового фона (см. backdrop.py)
    from . import backdrop as bd
except ImportError:
    import backdrop as bd  # type: ignore

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


def find_backdrop_groups(prompts: Mapping[Hashable, str], max_run: int = bd.DEFAULT_MAX_RUN) -> List[list]:
    """Серии ключей клипов с одинаковым фоном, длиннее max_run (тот же вид результата, что у find_duplicate_prompts).

    Ключи — номера клипов в виде строк или чисел; нечисловой ключ пропускается. Пустые и не строковые промпты
    пропускаются, поэтому разрывают серию (номер не попадает в проверку). Возвращаются группы ключей по порядку номеров.
    """
    by_num: dict = {}
    key_of: dict = {}
    for key, text in prompts.items():
        try:
            num = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        by_num[num] = text
        key_of[num] = key
    return [[key_of[n] for n in run] for run in bd.find_backdrop_runs(by_num, max_run)]
