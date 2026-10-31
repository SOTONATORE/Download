"""Тесты стартового плана ролей карточек (STEP 7, PLAN_generate_photo_video).

Чистые тесты: без сети и без доступа к файлам.
"""
from __future__ import annotations

import pytest

try:
    from Generate.core import card_roles as cr
except ImportError:  # type: ignore
    from core import card_roles as cr  # type: ignore


def test_single_card_both_on_is_photo_then_video():
    assert cr.plan_card_roles(1, True, True) == [("photo", "video")]


def test_single_card_photo_only():
    assert cr.plan_card_roles(1, True, False) == [("photo",)]


def test_single_card_video_only():
    assert cr.plan_card_roles(1, False, True) == [("video",)]


def test_two_cards_both_on():
    assert cr.plan_card_roles(2, True, True) == [("photo",), ("video",)]


def test_three_cards_both_on_odd_goes_to_video():
    assert cr.plan_card_roles(3, True, True) == [("photo",), ("video",), ("video",)]


def test_four_cards_both_on_two_and_two_photo_first():
    result = cr.plan_card_roles(4, True, True)
    assert len(result) == 4
    assert result == [("photo",), ("photo",), ("video",), ("video",)]
    assert result.count(("photo",)) == 2
    assert result.count(("video",)) == 2


def test_five_cards_both_on_two_photo_three_video():
    result = cr.plan_card_roles(5, True, True)
    assert result.count(("photo",)) == 2
    assert result.count(("video",)) == 3
    assert result[:2] == [("photo",), ("photo",)]


def test_video_only_three_cards():
    assert cr.plan_card_roles(3, False, True) == [("video",)] * 3


def test_photo_only_three_cards():
    assert cr.plan_card_roles(3, True, False) == [("photo",)] * 3


def test_both_off_raises():
    with pytest.raises(ValueError):
        cr.plan_card_roles(2, False, False)


def test_zero_cards_raises():
    with pytest.raises(ValueError):
        cr.plan_card_roles(0, True, True)


def test_bool_count_raises():
    with pytest.raises(ValueError):
        cr.plan_card_roles(True, True, True)


def test_float_count_raises():
    with pytest.raises(ValueError):
        cr.plan_card_roles(2.0, True, True)


@pytest.mark.parametrize("n", list(range(1, 11)))
def test_result_length_equals_n_cards_both_on(n):
    assert len(cr.plan_card_roles(n, True, True)) == n
