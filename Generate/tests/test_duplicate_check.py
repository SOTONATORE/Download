"""Тесты поиска одинаковых промптов (duplicate_check.find_duplicate_prompts). Без сети и без файлов."""
import copy

from Generate.core.duplicate_check import find_duplicate_prompts, normalize_prompt


def test_identical_prompts_form_one_group():
    assert find_duplicate_prompts({"1": "a dog runs", "2": "a dog runs"}) == [["1", "2"]]


def test_normalization_ignores_case_and_whitespace():
    prompts = {"1": "  A  dog\n runs ", "2": "a DOG runs", "3": "a cat sits"}
    assert find_duplicate_prompts(prompts) == [["1", "2"]]
    assert normalize_prompt("  A  dog\n runs ") == "a dog runs"


def test_singletons_are_not_reported():
    assert find_duplicate_prompts({"1": "x", "2": "y", "3": "z"}) == []


def test_several_groups_keep_first_appearance_order():
    prompts = {"5": "cat", "1": "dog", "7": "cat", "2": "dog", "9": "bird"}
    assert find_duplicate_prompts(prompts) == [["5", "7"], ["1", "2"]]


def test_empty_and_non_string_prompts_are_skipped():
    prompts = {"1": "", "2": "   ", "3": None, "4": "", "5": "tree"}
    assert find_duplicate_prompts(prompts) == []


def test_function_is_pure():
    prompts = {"1": "a dog", "2": "A DOG"}
    snapshot = copy.deepcopy(prompts)
    result = find_duplicate_prompts(prompts)
    assert prompts == snapshot
    assert result == [["1", "2"]]
    assert find_duplicate_prompts(prompts) == result
