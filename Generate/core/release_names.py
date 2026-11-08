"""Имена, теги и описания релизов выдачи (стадия 6).

Чистый модуль: только стандартная библиотека. Его используют и Download/publish_stock.py
(стоки), и Generate/core/release_adapter.py (генерация), поэтому для одного и того же
final.srt оба конвейера получают один хэш, одни теги и одни заголовки.

Шесть видов релизов:
  ФИНАЛ · СТОКИ            тег final-stock-<hash16>     текстовые файлы стоков и ссылки
  СТОКИ · ЧАСТЬ N          тег stock-part<N>-<hash16>   только медиа стоков, до 900 файлов
  ФИНАЛ · ГЕНЕРАЦИЯ        тег final-gen-<hash16>       служебные файлы, превью и ссылки
  ГЕНЕРАЦИЯ · ЧАСТЬ N      тег gen-part<N>-<hash16>     только клипы и фото, до 900 файлов

hash16 - первые 16 символов sha256 от final.srt текущего запуска (srt_hash из srt_parser.py).
Номер части пишется только у частей, у финалов его нет.
"""
from __future__ import annotations

from datetime import datetime, timezone

HASH16_LEN = 16
STAGES = ("stock", "gen")
# отображаемые названия этапов (в заголовках релизов)
STAGE_LABEL = {"stock": "СТОКИ", "gen": "ГЕНЕРАЦИЯ"}


def _check_stage(stage: str) -> str:
    if stage not in STAGE_LABEL:
        raise ValueError(f"Неизвестный этап релиза: {stage!r}")
    return stage


def hash16(full_hash: str) -> str:
    """Первые 16 символов полного хэша (его же использует тег и заголовок)."""
    if not full_hash:
        raise ValueError("Пустой хэш final.srt.")
    return full_hash[:HASH16_LEN]


def _utc_minute(when: datetime) -> str:
    """Время в UTC с точностью до минуты: YYYY-MM-DD HH:MM UTC."""
    return f"{when.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC"


def final_tag(stage: str, full_hash: str) -> str:
    """Тег итогового релиза: final-stock-<hash16> или final-gen-<hash16>."""
    return f"final-{_check_stage(stage)}-{hash16(full_hash)}"


def part_tag(stage: str, part: int, full_hash: str) -> str:
    """Тег части: stock-part<N>-<hash16> или gen-part<N>-<hash16>."""
    if part < 1:
        raise ValueError(f"Номер части должен быть не меньше 1: {part}")
    return f"{_check_stage(stage)}-part{part}-{hash16(full_hash)}"


def final_title(stage: str, full_hash: str, when: datetime) -> str:
    """Заголовок итогового релиза: ФИНАЛ · СТОКИ · <время> · <hash16> (номера части нет)."""
    return f"ФИНАЛ · {STAGE_LABEL[_check_stage(stage)]} · {_utc_minute(when)} · {hash16(full_hash)}"


def part_title(stage: str, part: int, full_hash: str, when: datetime) -> str:
    """Заголовок части: СТОКИ · ЧАСТЬ N · <время> · <hash16>."""
    if part < 1:
        raise ValueError(f"Номер части должен быть не меньше 1: {part}")
    return f"{STAGE_LABEL[_check_stage(stage)]} · ЧАСТЬ {part} · {_utc_minute(when)} · {hash16(full_hash)}"


def release_body(stage: str, role: str, full_hash: str) -> str:
    """Описание релиза. Полный хэш обязан быть в описании: по нему проверяется, что тег
    совпал с тем же файлом (проверка тела в release_adapter и publish_stock)."""
    if role not in ("final", "part"):
        raise ValueError(f"Неизвестная роль релиза: {role!r}")
    return f"hash: {full_hash}\nstage: {_check_stage(stage)}\nrole: {role}"
