"""Пути для тестов Generate.

Модули ядра и cli импортируются двумя способами: пакетно («Generate.core.x») и «плоско» («x»,
как из скриптов воркера и инструментов). Чтобы `python -m pytest Generate/tests` работал без
ручного PYTHONPATH (и в CI), здесь добавляются нужные папки в sys.path.
"""

import sys
from pathlib import Path

_GENERATE = Path(__file__).resolve().parents[1]   # .../Generate
_REPO = _GENERATE.parent                          # корень репозитория (пакет Generate)

for _p in (_REPO, _GENERATE, _GENERATE / "core", _GENERATE / "worker", _GENERATE / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
