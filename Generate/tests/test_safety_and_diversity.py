"""Тесты фильтра безопасности (кровь, травмы, боль, жестокость к животным) и совместимости style.md."""
from __future__ import annotations

from pathlib import Path

import pytest

from Generate.core import prompt_check as pc
from Generate.core.prompt_check import RulesValues

GEN = Path(__file__).resolve().parents[1]
COMIC = RulesValues(max_characters=4, animals_count=True, object_sets_forbidden=False, close_faces_allowed=False)


def _codes(prompt: str, media: str = "image", max_characters=None, rules=None, relaxed=frozenset()):
    return [
        pc.problem_code(p)
        for p in pc.find_problems(
            prompt, media, relaxed=relaxed, max_characters=max_characters, rules=rules
        )
    ]


@pytest.mark.parametrize("prompt", [
    "Red blood drips onto the cobblestones.",
    "A bloody bandage lies on the table.",
    "The farmer has a deep wound on his arm.",
    "An injured traveller sits beside the road.",
    "Visible bruises cover the man's shoulder.",
    "A figure writhing in pain on the wooden floor.",
    "A dead horse lies in the dry grass.",
    "A man beating a dog near the wooden fence.",
    "Cruelty to animals in a dark barn.",
    "A carcass rests on the dusty ground.",
])
def test_gore_injury_and_animal_cruelty_flagged(prompt):
    assert pc.PROBLEM_GORE_OR_CRUELTY in _codes(prompt, media="video"), prompt
    assert pc.PROBLEM_GORE_OR_CRUELTY in _codes(prompt, media="image", max_characters=4, rules=COMIC), prompt


@pytest.mark.parametrize("prompt", [
    "A painted wooden barn stands beside a sunlit field.",
    "A farmer wears a muted red scarf in the cool morning air.",
    "Enlarged biological cutaway of a curved blood vessel with pale blue sugar chains.",
    "Diagram of glucose entering the bloodstream on a warm cream backdrop.",
    "Chart card comparing blood sugar levels across two tubers.",
    "A clean rope wound around a wooden fence post.",
    "A healthy brown horse stands calmly beside a wooden cart.",
    "Low-angle view of a tall sunflower rising against a pale dusty-blue morning sky.",
    "Over-the-shoulder view past a farmer toward lined-up olive-green potato plants.",
])
def test_safe_educational_and_diverse_prompts_do_not_false_positive(prompt):
    assert pc.PROBLEM_GORE_OR_CRUELTY not in _codes(prompt, media="video"), prompt
    assert pc.PROBLEM_GORE_OR_CRUELTY not in _codes(prompt, media="image", max_characters=4, rules=COMIC), prompt


def test_gore_or_cruelty_cannot_be_relaxed():
    relaxed = pc.RELAXABLE_RULES
    assert pc.PROBLEM_GORE_OR_CRUELTY in _codes(
        "A bleeding wound on a wooden table.", media="image", relaxed=relaxed
    )


def test_cartoon_style_examples_stay_clean():
    style_text = (GEN / "Nishes" / "cartoon" / "prompts" / "style.md").read_text(encoding="utf-8")
    assert "# Visual Diversity and Shot Scale" in style_text
    assert "four independent axes" in style_text
    assert "cruelty to animals" in style_text
