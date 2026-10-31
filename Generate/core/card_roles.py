"""Стартовый план ролей карточек: фото, видео или фото, затем видео (STEP 7, PLAN_generate_photo_video).

Это только стартовый план ролей, пока не подключённый к оркестратору.
Чистый модуль: только стандартная библиотека, без сети, файлов и БД.
Переключение моделей внутри карточки и очереди здесь не делаются.
"""
from __future__ import annotations

PHOTO = "photo"
VIDEO = "video"


def plan_card_roles(n_cards: int, photo_on: bool, video_on: bool) -> list[tuple[str, ...]]:
    """Возвращает по одному кортежу ролей на каждую карточку.

    Правила:
    - только видео: каждая карточка ("video",);
    - только фото: каждая карточка ("photo",);
    - оба режима и одна карточка: ("photo", "video") — сначала фото, затем видео;
    - оба режима и две карточки и больше: фото-карточки идут первыми,
      нечётная карточка уходит в видео.

    Бросает ValueError, если n_cards не целое число не меньше 1 (bool не считается
    числом), или если оба режима выключены.
    """
    if isinstance(n_cards, bool) or not isinstance(n_cards, int):
        raise ValueError(f"n_cards должен быть целым числом, получено: {n_cards!r}")
    if n_cards < 1:
        raise ValueError(f"n_cards должен быть не меньше 1, получено: {n_cards!r}")
    if not photo_on and not video_on:
        raise ValueError("Оба режима выключены: нужно включить фото или видео.")

    if not photo_on:
        return [(VIDEO,)] * n_cards
    if not video_on:
        return [(PHOTO,)] * n_cards
    if n_cards == 1:
        return [(PHOTO, VIDEO)]

    photo_cards = n_cards // 2
    video_cards = n_cards - photo_cards
    return [(PHOTO,)] * photo_cards + [(VIDEO,)] * video_cards
