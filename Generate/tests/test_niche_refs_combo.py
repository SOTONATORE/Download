"""Ниша cartoon_photos (N6): разрешённые сочетания refs (allowed_refs) и копия референсов в папку ComfyUI.

Тесты написаны, не запускались (прогоняет пользователь на GitHub). Копия проверяется на временной папке,
настоящий ComfyUI не нужен.
"""
import json
from pathlib import Path

import pytest

from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import GeminiInputError, PromptConfig
try:
    from Generate.worker import server as srv
except ImportError:  # pragma: no cover
    from worker import server as srv  # type: ignore

ALLOWED = {(1,), (1, 6), (1, 2), (1, 2, 6), (2,), (3,), (4,), (5,)}


def _cfg(tmp_path, allowed=frozenset(ALLOWED)):
    style = tmp_path / "style.md"
    style.write_text("# Style\nPlain scene.\n", encoding="utf-8")
    return PromptConfig(style_file=str(style), media="image", refs_enabled=True, ref_max=6,
                        allowed_refs=allowed)


def _refs_reasons(refs, tmp_path, allowed=frozenset(ALLOWED)):
    """Причины refs: для одного сегмента с меткой на каждый референс (метки совпадают, проверяется только сочетание)."""
    tags = " ".join(f"<image{i}>" for i in range(1, len(refs) + 1))
    raw = json.dumps([{"segment_index": 1, "prompt": f"A flat scene with two simple shapes, {tags}.", "refs": refs}])
    _, problems = gp.parse_and_validate(raw, [1], _cfg(tmp_path, allowed))
    return [p.reason for p in problems if p.reason.startswith("refs:")]


# --- таблица сочетаний -----------------------------------------------------------------------------

def test_allowed_list_parses_to_exactly_eight_combinations(tmp_path):
    # Русский: файл профиля создаётся в tmp_path (восемь сочетаний); profile.yaml ниши cartoon_photos не читается
    f = tmp_path / "profile.yaml"
    f.write_text('allowed_refs: ["1", "1,2", "1,2,6", "1,6", "2", "3", "4", "5"]\n', encoding="utf-8")
    got = gp.parse_allowed_refs(f.read_text(encoding="utf-8").splitlines(), "refs_allowed")
    assert got == frozenset(ALLOWED)
    assert len(got) == 8


@pytest.mark.parametrize("line", [
    'allowed_refs: ["1,11"]',        # номер вне 1…10
    'allowed_refs: ["1,1"]',         # повтор номера
    'allowed_refs: ["1, 2"]',        # пробел внутри записи
    'allowed_refs: ["1", 2]',        # запись без кавычек
    'allowed_refs: ["1",]',          # лишняя запятая
    'allowed_refs: []',              # пустой список
    'allowed_refs: "1"',             # не список
])
def test_invalid_allowed_entry_is_error_at_profile_load(line):
    with pytest.raises(GeminiInputError, match="allowed_refs"):
        gp.parse_allowed_refs([line], "x")


def test_missing_allowed_refs_key_means_no_check():
    assert gp.parse_allowed_refs(["type: image"], "x") is None


def test_refs_six_alone_is_a_refs_problem(tmp_path):
    assert _refs_reasons([6], tmp_path) == [gp.REF_REASON_COMBO]


@pytest.mark.parametrize("refs", [[1, 2, 6], [2], [1, 6], [1]])
def test_allowed_combinations_pass(tmp_path, refs):
    assert _refs_reasons(refs, tmp_path) == []


@pytest.mark.parametrize("refs", [[3, 4], [1, 2, 4]])
def test_disallowed_combinations_are_refs_problems(tmp_path, refs):
    assert _refs_reasons(refs, tmp_path) == [gp.REF_REASON_COMBO]


def test_profile_without_allowed_refs_is_unchanged(tmp_path):
    assert _refs_reasons([6], tmp_path, allowed=None) == []


def test_combination_repair_text_is_the_style_guide_hint():
    problems = gp._ref_fix_problems("text", None, [gp.REF_REASON_COMBO])
    assert problems == [f"refs_invalid: «{gp.REF_COMBO_HINT}»"]


# --- копия референсов в папку входа ComfyUI ---------------------------------------------------------

def test_copy_six_refs_into_input_folder_with_equal_sizes(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    for i in range(1, 7):
        (refs / f"ref_{i}.png").write_bytes(b"x" * (i * 100))
    inp = tmp_path / "input"
    names = srv.copy_niche_refs(refs, inp)
    assert names == [f"ref_{i}.png" for i in range(1, 7)]
    for i in range(1, 7):
        assert (inp / f"ref_{i}.png").stat().st_size == (refs / f"ref_{i}.png").stat().st_size


def test_copy_refuses_missing_source(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    for i in (1, 2, 4, 5, 6):
        (refs / f"ref_{i}.png").write_bytes(b"x" * 10)
    with pytest.raises(srv.NicheRefsError, match="подряд"):
        srv.copy_niche_refs(refs, tmp_path / "input")


def test_copy_of_niche_refs_matches_sizes(tmp_path):
    # Изолированная ниша в tmp_path: refs_enabled: true, ref_1.png…ref_6.png разного размера
    nishes = tmp_path / "Nishes"
    profile_dir = nishes / "refs_niche"
    (profile_dir / "refs").mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text("refs_enabled: true\n", encoding="utf-8")
    for i in range(1, 7):
        (profile_dir / "refs" / f"ref_{i}.png").write_bytes(b"x" * (i * 100))
    inp = tmp_path / "input"
    names = srv.prepare_niche_refs(profile_dir, nishes, inp)
    assert names == [f"ref_{i}.png" for i in range(1, 7)]
    for name in names:
        assert (inp / name).stat().st_size == (profile_dir / "refs" / name).stat().st_size


def test_non_niche_profile_is_not_copied(tmp_path):
    models_profile = tmp_path / "models" / "ltx25"
    models_profile.mkdir(parents=True)
    (models_profile / "profile.yaml").write_text("refs_enabled: true\n", encoding="utf-8")
    assert srv.prepare_niche_refs(models_profile, tmp_path / "Nishes", tmp_path / "input") == []
    assert not (tmp_path / "input").exists()


def test_niche_without_refs_enabled_is_not_copied(tmp_path):
    niche = tmp_path / "Nishes" / "plain"
    niche.mkdir(parents=True)
    (niche / "profile.yaml").write_text("refs_enabled: false\n", encoding="utf-8")
    assert srv.prepare_niche_refs(niche, tmp_path / "Nishes", tmp_path / "input") == []


def test_comfy_input_dir_default_is_comfy_dir_input(monkeypatch):
    monkeypatch.delenv("COMFY_INPUT_DIR", raising=False)
    monkeypatch.setenv("COMFY_DIR", "/some/comfy")
    assert srv.comfy_input_dir() == Path("/some/comfy") / "input"


def test_comfy_input_dir_env_overrides(monkeypatch):
    monkeypatch.setenv("COMFY_INPUT_DIR", "/elsewhere/in")
    assert srv.comfy_input_dir() == Path("/elsewhere/in")
