"""Фон кадра по тексту промпта и поиск серий соседних клипов с одинаковым фоном.

Чистые функции: без ввода-вывода и без состояния, только стандартная библиотека (re).
Фон определяется по ключевым словам: побеждает первое правило из BACKDROP_RULES, которое совпало.
Порядок важен: более конкретные фоны стоят раньше общих. Формула «небо с облаками и холмами»
стоит первой, поэтому лесной кадр с таким небом относится к sky_clouds_hills: повторяется именно фон.

Метка OTHER значит «фон не распознан». Такие клипы в серии не считаются, иначе несвязанные промпты
давали бы ложные срабатывания.
"""
from __future__ import annotations

import re
from typing import List, Mapping, Tuple

# Метки фонов. Строки — машинные коды; их меняют только вместе с BACKDROP_RULES.
SKY_CLOUDS_HILLS = "sky_clouds_hills"
CREAM_CARD = "cream_card"
CREAM_WALL_TABLE = "cream_wall_table"
UNDERWATER = "underwater"
FOREST = "forest"
SWAMP = "swamp"
CAVE = "cave"
MAP = "map"
INTERIOR_LAB = "interior_lab"
OTHER = "other"

# Правила: метка и ключевые слова (без учёта регистра, только целые слова). Первое совпавшее правило задаёт метку.
# Чтобы добавить фон, допишите строку в нужное место списка и добавьте метку выше.
BACKDROP_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (SKY_CLOUDS_HILLS, ("white clouds", "blue sky")),
    (CREAM_CARD, ("cream card",)),
    (CREAM_WALL_TABLE, ("cream wall",)),
    (MAP, ("map",)),
    (UNDERWATER, ("underwater", "under the sea", "seabed", "coral reef")),
    (FOREST, ("forest", "jungle", "woodland", "trees")),
    (SWAMP, ("swamp", "marsh", "bog", "wetland")),
    (CAVE, ("cave", "cavern", "grotto")),
    (INTERIOR_LAB, ("laboratory", "lab", "workshop")),
)

# Регулярные выражения собираются один раз при загрузке модуля из BACKDROP_RULES.
_RULE_RE: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple(
    (label, re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE))
    for label, keywords in BACKDROP_RULES
)

# Длина серии, которая допустима. Серия длиннее — нарушение (по умолчанию: три подряд уже плохо).
DEFAULT_MAX_RUN = 2


def backdrop_signature(prompt: str) -> str:
    """Метка фона промпта. Не строка или пустой текст — OTHER."""
    if not isinstance(prompt, str) or not prompt.strip():
        return OTHER
    for label, rx in _RULE_RE:
        if rx.search(prompt):
            return label
    return OTHER


def find_backdrop_runs(prompts: Mapping[int, str], max_run: int = DEFAULT_MAX_RUN) -> List[List[int]]:
    """Серии номеров клипов с одинаковым фоном, длиннее max_run.

    Считаются только подряд идущие номера: пропуск в нумерации обрывает серию. Клипы с меткой OTHER
    серий не образуют. Возвращаются списки номеров по возрастанию, серии идут по порядку номеров.
    """
    if max_run < 1:
        raise ValueError("max_run должен быть не меньше 1")
    runs: List[List[int]] = []
    current: List[int] = []
    current_sig = OTHER
    for num in sorted(prompts):
        sig = backdrop_signature(prompts[num])
        continues = bool(current) and num == current[-1] + 1 and sig == current_sig and sig != OTHER
        if not continues:
            if len(current) > max_run:
                runs.append(current)
            current = []
        if sig != OTHER:
            if not current:
                current_sig = sig
            current.append(num)
    if len(current) > max_run:
        runs.append(current)
    return runs
