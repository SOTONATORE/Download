"""Превью с refs: блок «refs: 1,6» в prompts.txt, проверка до аренды, payload, индекс.

Старый формат (блоки без refs:) проверяется отдельно: результат тот же, что был до правок.
Сеть не используется. Настоящая модель и ComfyUI не нужны.
"""
import sys
from pathlib import Path

import pytest

_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

try:
    from Generate import cli
    from Generate.core import config_loader as cl
    from Generate.core import previews as pv
except ImportError:  # pragma: no cover
    import cli  # type: ignore
    from core import config_loader as cl  # type: ignore
    from core import previews as pv  # type: ignore

# Русский: реальный prompts_cartoon_refs.txt удалён в Блоке 0 и больше не читается.
# Файл с 8 блоками создаёт фикстура prompts_file в tmp_path; сочетания — из таблицы ниши.
COMBOS = [(1,), (1, 2), (1, 2, 6), (1, 6), (2,), (3,), (4,), (5,)]


def _write(tmp_path, text):
    f = tmp_path / "prompts.txt"
    f.write_text(text, encoding="utf-8")
    return str(f)


# --- файл с 8 блоками по таблице --------------------------------------------------------------

@pytest.fixture
def prompts_file(tmp_path):
    """Русский: файл промптов с 8 блоками, по одному на сочетание refs; создаётся в tmp_path."""
    blocks = []
    for refs in COMBOS:
        labels = " ".join(f"<image{n}>" for n in range(1, len(refs) + 1))
        blocks.append(f"refs: {','.join(str(n) for n in refs)}\nA flat scene with {labels}.")
    f = tmp_path / "prompts.txt"
    f.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return str(f)


def test_prompts_file_has_one_block_per_combination(prompts_file):
    specs = pv.load_prompt_specs(prompts_file)
    got = sorted(refs for _t, refs in specs)
    assert sorted(COMBOS) == sorted(got)
    assert len(specs) == 8


def test_prompts_file_passes_refs_check_for_cartoon(prompts_file):
    pv.check_refs(pv.load_prompt_specs(prompts_file), frozenset(COMBOS))


def test_prompts_file_passes_cli_check_for_cartoon(prompts_file, tmp_path, monkeypatch):
    # Русский: профиль cartoon_photos имеет refs_enabled: false, поэтому проверка блоков с refs идёт
    # на синтетической нише в tmp_path: refs_enabled: true и allowed_refs из COMBOS. Имя теста прежнее.
    folder = tmp_path / "Nishes" / "refs_niche"
    folder.mkdir(parents=True)
    allowed = ", ".join('"' + ",".join(str(n) for n in refs) + '"' for refs in COMBOS)
    (folder / "profile.yaml").write_text(f"refs_enabled: true\nallowed_refs: [{allowed}]\n", encoding="utf-8")
    monkeypatch.setattr(cl, "NISHES_DIR", tmp_path / "Nishes")
    assert cli._check_previews("1", prompts_file, "refs_niche") is None


# --- разбор блоков ----------------------------------------------------------------------------

def test_refs_line_is_taken_out_of_text_and_sorted(tmp_path):
    specs = pv.load_prompt_specs(_write(tmp_path, "refs: 6,1\nA scene <image1> and <image2>.\n"))
    assert specs == [("A scene <image1> and <image2>.", (1, 6))]


def test_refs_line_is_case_insensitive_and_spaces_allowed(tmp_path):
    specs = pv.load_prompt_specs(_write(tmp_path, "REFS:  2 ,  1\nText <image1> <image2>.\n"))
    assert specs[0][1] == (1, 2)


def test_old_blocks_without_refs_are_unchanged(tmp_path):
    text = "первый промпт\n\nвторой промпт <image6>\n"
    specs = pv.load_prompt_specs(_write(tmp_path, text))
    assert specs == [("первый промпт", None), ("второй промпт <image6>", None)]
    assert [t for t, _ in specs] == pv.parse_prompts(text)


def test_load_prompts_still_returns_texts_only(tmp_path):
    assert pv.load_prompts(_write(tmp_path, "refs: 1\nA flat scene <image1>.\n")) == ["A flat scene <image1>."]


@pytest.mark.parametrize("line, why", [
    ("refs: 1,11", "вне 1…10"),
    ("refs: 1,1", "повторяются"),
    ("refs: 1 6", "через запятую"),
    ("refs: ", "нет номеров"),
    ("refs: a,b", "целыми"),
])
def test_bad_refs_line_is_input_error(tmp_path, line, why):
    with pytest.raises(pv.PreviewInputError, match=why):
        pv.load_prompt_specs(_write(tmp_path, f"{line}\nText <image1>.\n"))


def test_refs_line_without_text_is_input_error(tmp_path):
    with pytest.raises(pv.PreviewInputError, match="нет текста"):
        pv.load_prompt_specs(_write(tmp_path, "refs: 1\n"))


# --- проверка сочетаний и меток ---------------------------------------------------------------

@pytest.mark.parametrize("text, refs, why", [
    ("Text <image1> <image2>.", "3,4", "не разрешено"),
    ("Text <image1>.", "6", "не разрешено"),
    ("Text <image1> only.", "1,6", "не совпадают"),
    ("Text <image1> <image2> <image3>.", "1,6", "не совпадают"),
])
def test_check_refs_rejects_bad_combo_or_labels(tmp_path, text, refs, why):
    specs = [(text, tuple(int(x) for x in refs.split(",")))]
    with pytest.raises(pv.PreviewInputError, match=why):
        pv.check_refs(specs, frozenset(COMBOS))


def test_check_refs_without_profile_list_checks_labels_only():
    pv.check_refs([("Text <image1> <image2>.", (3, 4))], None)  # сочетание не проверяется, метки — да
    with pytest.raises(pv.PreviewInputError, match="не совпадают"):
        pv.check_refs([("Text <image1>.", (3, 4))], None)


def test_old_blocks_are_not_checked(tmp_path):
    pv.check_refs([("Any text <image6>.", None)], frozenset(COMBOS))


# --- до аренды: cli ---------------------------------------------------------------------------

def test_cli_refs_block_on_profile_without_refs_is_input_error(tmp_path):
    f = _write(tmp_path, "refs: 1\nA flat scene <image1>.\n")
    assert cli._check_previews("1", f, "zimage_turbo") == cli.EXIT_INPUT


def test_cli_old_file_on_profile_without_refs_still_passes(tmp_path):
    f = _write(tmp_path, "A flat scene without tags.\n")
    assert cli._check_previews("1", f, "zimage_turbo") is None


def test_cli_switch_off_checks_nothing(tmp_path):
    assert cli._check_previews("выкл", str(tmp_path / "нет.txt"), "cartoon_photos") is None


# --- payload и индекс -------------------------------------------------------------------------

def test_build_previews_keeps_refs_and_default_is_none():
    items = pv.build_previews([("a <image1>.", (1,)), "b"], 1, seed_fn=lambda: 5)
    assert items[0].refs == (1,)
    assert items[1].refs is None


def test_index_line_has_refs_only_when_block_has_them():
    items = pv.build_previews([("Text <image1> <image2>.", (1, 6)), "кот у окна"], 1, seed_fn=lambda: 5)
    lines = pv.index_text(items).splitlines()
    assert "refs: 1,6 |" in lines[3]
    assert "refs:" not in lines[4]
    assert lines[4] == "preview_2_1.png | промпт 2 | y 1 | seed 5 | карта: — | статус: в очереди | промпт: кот у окна"
