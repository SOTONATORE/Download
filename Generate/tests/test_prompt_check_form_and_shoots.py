"""Тесты ложных срабатываний на «form» и «shoots at» в prompt_check.py.

Запуск из корня репозитория:  python -m unittest discover -s Generate/tests -v
Модуль загружается по пути к файлу, поэтому пакет Generate/core не импортируется целиком.
"""

import importlib.util
import re
import unittest
from pathlib import Path

_CORE = Path(__file__).resolve().parents[1] / "core" / "prompt_check.py"
_spec = importlib.util.spec_from_file_location("prompt_check", _CORE)
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)


def _codes(text: str) -> list:
    return [pc.problem_code(p) for p in pc.find_problems(text)]


class FormFalsePositivesTest(unittest.TestCase):
    """Слово «form» в значении формы, а не бланка: ничего не должно срабатывать."""

    def test_base_verb_form_a(self):
        # клип 267: глагол «form» с множественным подлежащим
        self.assertEqual(pc.find_problems(
            "Above the plate two crossed bold red ink strokes form a red X"), [])

    def test_roots_form_an_arch(self):
        self.assertEqual(pc.find_problems("roots form an arch"), [])

    def test_idiom_in_the_form_of(self):
        # клипы 331, 333, 335, 337, 342
        self.assertEqual(pc.find_problems(
            "one mouth expression in the form of a closed-line smile"), [])

    def test_each_form_is_readable(self):
        # клип 607
        self.assertEqual(pc.find_problems("spaced loosely so each form is readable"), [])

    def test_its_and_their_form_is(self):
        self.assertEqual(pc.find_problems("the leaf keeps its form is simple"), [])
        self.assertEqual(pc.find_problems("the tubers keep their form looks round"), [])


class FormStillCaughtTest(unittest.TestCase):
    """Бланки и бумажные формы по-прежнему запрещены."""

    def assert_forbidden_form(self, text: str):
        self.assertIn(pc.PROBLEM_FORBIDDEN_SHOT_TYPE, _codes(text), msg=text)

    def test_blank_form(self):
        self.assert_forbidden_form("a blank form")

    def test_form_on_the_desk(self):
        self.assert_forbidden_form("a form on the desk")

    def test_stack_of_forms(self):
        self.assert_forbidden_form("a stack of forms")

    def test_tax_forms_on_the_table(self):
        self.assert_forbidden_form("tax forms on the table")

    def test_paper_forms_in_a_tray(self):
        self.assert_forbidden_form("paper forms in a tray")

    def test_printed_forms_on_a_shelf(self):
        self.assert_forbidden_form("printed forms on a shelf")

    def test_application_forms_in_a_box(self):
        self.assert_forbidden_form("application forms in a box")


class ShootsFalsePositivesTest(unittest.TestCase):
    """Ростки растения «shoots at» — не стрельба."""

    def test_clip_374_unwanted_shoots_at_lower_right(self):
        # клип 374
        self.assertEqual(pc.find_problems(
            "a small cluster of pale green unwanted shoots at the lower right"), [])

    def test_shoots_at_the_left_edge(self):
        self.assertEqual(pc.find_problems("shoots at the left edge"), [])

    def test_wild_shoots_at_the_far_corner(self):
        self.assertEqual(pc.find_problems("wild shoots at the far corner"), [])


class ShootsStillCaughtTest(unittest.TestCase):
    """Настоящая стрельба и бой по-прежнему запрещены."""

    def test_soldier_shoots_at_enemy(self):
        self.assertIn(pc.PROBLEM_CHAOS, _codes("a soldier shoots at the enemy"))

    def test_guard_shoots_back(self):
        self.assertIn(pc.PROBLEM_CHAOS, _codes("the guard shoots back"))


class RegexCompileTest(unittest.TestCase):
    """Все шаблоны компилируются стандартным re (без переменной ширины просмотра назад)."""

    def test_form_patterns_compile(self):
        for pattern in pc._STEM_EXCLUDE_VERB_RE["form"]:
            with self.subTest(pattern=pattern):
                re.compile(pattern, re.IGNORECASE)

    def test_forbidden_image_re_compiles(self):
        self.assertIsNotNone(pc.FORBIDDEN_IMAGE_RE)

    def test_shoot_rx_part_compiles(self):
        self.assertIsNotNone(re.compile(pc._SHOOT_RX_PART, re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
