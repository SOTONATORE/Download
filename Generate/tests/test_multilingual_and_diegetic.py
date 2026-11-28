"""Мультиязычные проверки (FR/DE/ES/RU/CJK) и политика диегетических вывесок vs плавающих плашек.

Сеть и GPU не используются.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Generate.core import frame_rules as fr  # noqa: E402
from Generate.core import gemini_prompts as gp  # noqa: E402
from Generate.core import prompt_check as pc  # noqa: E402
from Generate.core import srt_optimizer as so  # noqa: E402
from Generate.core.srt_parser import Segment  # noqa: E402


# ---------------------------------------------------------------------------
# 1. frame_rules: французские / DE / ES запрещённые типы кадра и «forme/formes»
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Une carte ancienne repose sur la table.",
    "Un drapeau flotte au-dessus du bâtiment.",
    "Il montre son passeport à la frontière.",
    "Un calendrier accroché au mur.",
    "Une pièce d'identité posée sur le bureau.",
    "Une feuille blanche et une feuille de papier.",
    "Le formulaire est rempli à la main.",
    "Eine alte Karte liegt auf dem Tisch.",
    "Ein Reisepass und eine Flagge im Büro.",
    "Un mapa y una bandera en la sala.",
    "El pasaporte y el periódico están aquí.",
])
def test_frame_rules_catches_french_and_european_forbidden_items(text):
    assert fr.needs_rephrase(text) is True
    assert fr.uncovered_hits(text) != []


@pytest.mark.parametrize("text", [
    "Cette forme arrondie protège la racine.",
    "Les formes naturelles changent avec le temps.",
    "Les nuages peuvent former un arc.",
    "Une croûte formée par la chaleur.",
    "Des pierres formées sous la mer.",
    "En formant un cercle autour du puits.",
])
def test_french_forme_words_are_not_false_positives(text):
    assert fr.needs_rephrase(text) is False
    assert pc.find_problems(text, "image") == []


def test_frame_rules_allowed_items_releases_french_map_and_paper_but_not_flag():
    allowed = frozenset({"map", "sheet of paper"})
    assert fr.needs_rephrase("Une carte ancienne sur la table.", allowed) is False
    assert fr.needs_rephrase("Une feuille de papier sur le bureau.", allowed) is False
    assert fr.needs_rephrase("Un drapeau sur la tour.", allowed) is True
    assert fr.needs_rephrase("Un passeport sur le bureau.", allowed) is True


# ---------------------------------------------------------------------------
# 2. prompt_check: французские элизии, лигатуры, служебные слова и отрицания
# ---------------------------------------------------------------------------

def test_french_elision_matches_uppercase_labels():
    prompt_h = "A wooden shop sign reading HISTOIRE in bold black uppercase lettering."
    assert pc.segment_lettering_mismatch(prompt_h, "C'est toute l'histoire du village.") == []
    assert pc.segment_lettering_mismatch(prompt_h, "C\u2019est toute l\u2019histoire du village.") == []

    prompt_e = "A wooden harbor sign reading EAU in bold black uppercase lettering."
    assert pc.segment_lettering_mismatch(prompt_e, "La pompe remonte l'eau du puits.") == []
    assert pc.segment_lettering_mismatch(prompt_e, "La pompe remonte l\u2019eau du puits.") == []


def test_french_and_german_ligatures_fold_cleanly():
    assert pc._fold("Cœur") == pc._fold("COEUR") == "coeur"
    assert pc._fold("Œuvre") == pc._fold("OEUVRE") == "oeuvre"
    assert pc._fold("Straße") == pc._fold("STRASSE") == "strasse"

    prompt_c = "A bakery sign reading COEUR in bold black uppercase lettering."
    assert pc.segment_lettering_mismatch(prompt_c, "Au cœur du marché d'automne.") == []
    prompt_o = "A wooden sign reading OEUVRE in bold black uppercase lettering."
    assert pc.segment_lettering_mismatch(prompt_o, "Une grande œuvre artisanale.") == []


def test_french_dangling_words_and_negations():
    for bad in ("PAIN DE", "LA MER LE", "PORT DU", "RUE DES", "EAU POUR"):
        prompt = f"A wooden shop sign reading {bad} in bold black uppercase lettering."
        assert pc.dangling_text_hits(prompt) == [bad]

    prompt_ok = "A wooden shop sign reading PAIN in bold black uppercase lettering."
    assert pc.dangling_text_hits(prompt_ok) == []
    assert pc.segment_lettering_mismatch(prompt_ok, "Il ne mange pas de pain.") != []
    assert pc.segment_lettering_mismatch(prompt_ok, "Sans pain sur la table.") != []
    assert pc.segment_lettering_mismatch(prompt_ok, "Il n'a jamais de pain.") != []
    assert pc.segment_lettering_mismatch(prompt_ok, "Le boulanger vend du pain.") == []


def test_french_ethnicity_roots_prevent_false_flag():
    prompt = "A wide market scene with an African man carrying baskets."
    assert pc.ethnicity_hits(prompt, "Un marchand venu d'Afrique.") == []
    prompt_cn = "A medium scene with a Chinese farmer in a field."
    assert pc.ethnicity_hits(prompt_cn, "Un paysan chinois cultive le riz.") == []


# ---------------------------------------------------------------------------
# 3. gemini_prompts: _READ_LABEL_RX, _needs_full_sentence, аббревиатуры
# ---------------------------------------------------------------------------

def test_read_label_rx_preserves_french_apostrophes_and_stops_at_accented_lowercase():
    m1 = gp._READ_LABEL_RX.search("A wooden sign reading L'EAU in bold black uppercase lettering.")
    assert m1 is not None and m1.group(1) == "L'EAU"
    m2 = gp._READ_LABEL_RX.search("A street plaque reading D'OR à droite du port.")
    assert m2 is not None and m2.group(1) == "D'OR"
    m3 = gp._READ_LABEL_RX.search("A shop sign reading C'EST BON.")
    assert m3 is not None and m3.group(1) == "C'EST BON"


def test_needs_full_sentence_strips_closing_quotes_and_brackets():
    assert gp._needs_full_sentence("« Le boulanger ouvre sa boutique chaque matin. »") is False
    assert gp._needs_full_sentence("« Le boulanger ouvre sa boutique chaque matin.\u00a0»") is False
    assert gp._needs_full_sentence("(Farmers harvest the ripe golden tubers in autumn.)") is False
    assert gp._needs_full_sentence("« Bonjour. »") is True  # слишком короткий фрагмент (<= 5 слов)
    assert gp._needs_full_sentence("« Le boulanger ouvre sa boutique chaque matin »") is True  # нет точки


def test_multilingual_abbreviations_do_not_break_sentences():
    segs_fr = [
        Segment(1, 0, 1000, "M."),
        Segment(2, 1000, 2000, "Dupont arrive au port."),
    ]
    sentences_fr, _ = gp.merge_segments_into_sentences(segs_fr)
    assert [s.text for s in sentences_fr] == ["M. Dupont arrive au port."]

    segs_mme = [
        Segment(1, 0, 1000, "Mme."),
        Segment(2, 1000, 2000, "Curie travaille au laboratoire."),
    ]
    sentences_mme, _ = gp.merge_segments_into_sentences(segs_mme)
    assert [s.text for s in sentences_mme] == ["Mme. Curie travaille au laboratoire."]

    segs_jc = [
        Segment(1, 0, 1000, "En 50 av. J.-C."),
        Segment(2, 1000, 2000, "les habitants cultivent la vallée."),
    ]
    sentences_jc, _ = gp.merge_segments_into_sentences(segs_jc)
    assert [s.text for s in sentences_jc] == ["En 50 av. J.-C. les habitants cultivent la vallée."]


# ---------------------------------------------------------------------------
# 4. srt_optimizer: французские NBSP/NNBSP и CJK-завершители
# ---------------------------------------------------------------------------

def test_srt_optimizer_handles_french_nbsp_and_cjk_enders():
    assert so.ends_sentence("« Bonjour.\u00a0»") is True
    assert so.ends_sentence("« Incroyable\u202f!\u00a0»") is True
    assert so.ends_sentence("Vraiment\u202f?") is True
    assert so.ends_sentence("今日は晴れです。") is True
    assert so.ends_sentence("本当！") is True
    assert so.ends_sentence("雨ですか？") is True
    assert so.ends_sentence("« Bonjour\u00a0»") is False


# ---------------------------------------------------------------------------
# 5. Диегетические вывески разрешены; плавающие плашки / карточки запрещены
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt,setting", [
    (
        "A wide street scene with a wooden storefront sign reading BOULANGERIE in bold black uppercase lettering.",
        "wide: street",
    ),
    (
        "A medium shot of a market stall with a painted shop sign reading PAIN in bold black uppercase lettering.",
        "medium: market",
    ),
    (
        "A wide view of the quay with a wooden harbor sign reading PORT in bold black uppercase lettering.",
        "wide: harbor",
    ),
])
def test_diegetic_in_world_signs_are_allowed_on_ordinary_scenes(prompt, setting):
    reasons = gp._batch_style_reasons(prompt, setting)
    assert gp._UNAUTH_TEXT_REASON not in reasons
    assert gp._OVERLAY_CARRIER_REASON not in reasons


@pytest.mark.parametrize("prompt,setting", [
    (
        "A wide field with a label card reading EARTH PEAR in bold black uppercase lettering.",
        "wide: field",
    ),
    (
        "A full-width pale cream header band reading HISTORY in bold black uppercase lettering.",
        "map: valley",
    ),
    (
        "A wide street with a title plaque reading 1842 in bold black uppercase lettering.",
        "wide: street",
    ),
    (
        "A macro view of a tuber beside an accent plaque reading TUBER in bold black uppercase lettering.",
        "macro: tuber",
    ),
    (
        "A number card reading 42 centered on a warm cream backdrop.",
        "spotlight: number",
    ),
])
def test_floating_overlay_text_carriers_are_rejected_even_with_reading(prompt, setting):
    reasons = gp._batch_style_reasons(prompt, setting)
    assert gp._OVERLAY_CARRIER_REASON in reasons
