"""Тесты запроса cartoon_photos: поле setting, батч одним текстом с маркерами сегментов, строка разнообразия
сеттингов, без финальной самопроверки. У прочих профилей прежний вид запроса. Сеть не используется."""
import pathlib

from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import PromptConfig, build_requests, parse_and_validate, response_schema
from Generate.core.srt_parser import Segment

GEN = pathlib.Path(__file__).resolve().parent.parent
CARTOON_DIR = GEN / "Nishes" / "cartoon_photos"
LTX_DIR = GEN / "model_profiles" / "models" / "ltx25"
SETTING_FORMAT = [{"segment_index": int, "setting": str, "prompt": str}]
SELF_CHECK_BULLET = "check each prompt against the self-check checklist"
VARIETY_TEXT = "Neighbouring clips in the same batch must use different settings"


def seg(num, text):
    return Segment(num, 0, 1000, text)


def make_segments():
    # предложения: 1 | 2+3 | 4 | 5 | 6 (пустой, вне предложения) | 7 | 8
    return [
        seg(1, "The old city woke slowly."),
        seg(2, "Merchants opened their stalls"),
        seg(3, "and carts rolled in from the hills."),
        seg(4, "Dr. Hale watched from a window."),
        seg(5, "Rain fell."),
        seg(6, ""),
        seg(7, "Then the bell rang."),
        seg(8, "Crowds gathered."),
    ]


def make_cfg(tmp_path, passage, batch_size=3, response_format=None):
    style = tmp_path / "style.md"
    style.write_text("Style guide for tests.", encoding="utf-8")
    kwargs = {}
    if passage:
        kwargs["batch_passage"] = True
        kwargs["response_format"] = response_format or SETTING_FORMAT
    return PromptConfig(style_file=str(style), media="image", batch_size=batch_size, **kwargs)


def batch_lines(user):
    return [ln for ln in user.split("\n") if ln.startswith("segment_index: ")]


# ---------------------------------------------------------------------------
# Схема ответа: setting перед prompt, segment_index и prompt сохранены
# ---------------------------------------------------------------------------

def test_setting_present_in_schema_before_prompt(tmp_path):
    s = response_schema(SETTING_FORMAT)
    props = s["items"]["properties"]
    assert "setting" in props and props["setting"] == {"type": "string"}
    assert "segment_index" in props and "prompt" in props
    assert s["items"]["required"] == ["segment_index", "setting", "prompt"]
    # порядок ключей: setting идёт до prompt
    assert list(props) == ["segment_index", "setting", "prompt"]
    builder = gp._Builder(make_segments(), [1], make_cfg(tmp_path, passage=True))
    assert "setting" in builder.schema["items"]["properties"]


def test_parser_keeps_setting_as_one_line(tmp_path):
    cfg = make_cfg(tmp_path, passage=True)
    raw = ('[{"segment_index": 1, "setting": "  muddy\\nriverbank   with reeds ", '
           '"prompt": "A small wooden boat rests on a muddy riverbank with reeds."}]')
    results, problems = parse_and_validate(raw, [1], cfg, texts={1: "The old city woke slowly."})
    assert 1 in results
    assert results[1].extra["setting"] == "muddy riverbank with reeds"
    assert not [p for p in problems if "setting" in p.reason]


# ---------------------------------------------------------------------------
# Системная инструкция: строка разнообразия вместо самопроверки
# ---------------------------------------------------------------------------

def test_final_check_line_absent_for_cartoon_and_variety_present(tmp_path):
    system = gp.build_system_instruction(make_cfg(tmp_path, passage=True))
    assert SELF_CHECK_BULLET not in system
    assert VARIETY_TEXT in system
    assert "\"setting\": <a short English phrase" in system


def test_final_check_line_kept_for_other_profiles(tmp_path):
    system = gp.build_system_instruction(make_cfg(tmp_path, passage=False))
    assert SELF_CHECK_BULLET in system
    assert VARIETY_TEXT not in system
    assert "\"setting\"" not in system


# ---------------------------------------------------------------------------
# Батч одним текстом
# ---------------------------------------------------------------------------

def test_passage_batch_has_each_segment_index_once(tmp_path):
    cfg = make_cfg(tmp_path, passage=True, batch_size=3)
    batches = build_requests(make_segments(), list(range(1, 9)), cfg)
    assert [b.nums for b in batches] == [(1, 2, 3), (4, 5, 6), (7, 8)]
    for b in batches:
        lines = batch_lines(b.user)
        assert len(lines) == len(b.nums)
        for n in b.nums:
            prefix = f"segment_index: {n} | "
            assert sum(1 for ln in lines if ln.startswith(prefix)) == 1


def test_passage_segment_line_has_text_and_marked_sentence(tmp_path):
    cfg = make_cfg(tmp_path, passage=True)
    user = build_requests(make_segments(), [3], cfg)[0].user
    assert ("segment_index: 3 | Text (what this segment says): and carts rolled in from the hills. | "
            "Sentence: Merchants opened their stalls >>and carts rolled in from the hills.<<") in user
    assert "segment_index: 2 " not in user and "segment_index: 4 " not in user


def test_passage_context_is_whole_sentences_and_not_repeated_in_batch(tmp_path):
    cfg = make_cfg(tmp_path, passage=True)
    batches = build_requests(make_segments(), list(range(1, 9)), cfg)
    for b in batches:
        lines = batch_lines(b.user)
        seg_part = "\n".join(lines)
        for ln in b.user.split("\n"):
            if ln.startswith("Before (context only): ") or ln.startswith("After (context only): "):
                ctx = ln.split(": ", 1)[1]
                assert ctx.endswith((".", "?", "!")), ctx        # целые предложения
                # каждое предложение контекста не повторяется среди строк сегментов батча
                for sentence in _split_sentences(ctx):
                    assert sentence not in seg_part, sentence


def test_passage_context_labels_and_content(tmp_path):
    cfg = make_cfg(tmp_path, passage=True)
    user = build_requests(make_segments(), [1, 2, 3], cfg)[0].user
    assert "Before (context only):" not in user            # первый батч: слева контекста нет
    assert "After (context only): Dr. Hale watched from a window. Rain fell." in user
    user2 = build_requests(make_segments(), [4, 5, 6], cfg)[0].user
    # контекст до батча 4..6: два целых предложения, сами они в строках сегментов не повторяются
    assert ("Before (context only): The old city woke slowly. "
            "Merchants opened their stalls and carts rolled in from the hills.") in user2
    assert "Sentence: >>Dr. Hale watched from a window.<<" in user2


def test_passage_empty_segment_marked_silence(tmp_path):
    cfg = make_cfg(tmp_path, passage=True)
    user = build_requests(make_segments(), [6], cfg)[0].user
    assert "segment_index: 6 | Text (what this segment says): (silence / no text)" in user


def _split_sentences(text):
    out, cur = [], ""
    for ch in text:
        cur += ch
        if ch in ".?!" and cur.strip().endswith(ch):
            out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


# ---------------------------------------------------------------------------
# Прочие профили: прежний вид блоков
# ---------------------------------------------------------------------------

def test_other_profiles_keep_block_form(tmp_path):
    cfg = make_cfg(tmp_path, passage=False)
    user = build_requests(make_segments(), [3], cfg)[0].user
    assert "segment_index: 3\nText (what this segment says): and carts rolled in from the hills.\n" in user
    assert "Before (context only): The old city woke slowly." in user
    assert " | Sentence: " not in user
    assert "segment_index: 3 | " not in user


def test_real_ltx25_profile_keeps_block_form_and_schema(tmp_path):
    rules = gp.load_rules(LTX_DIR / "prompts" / "rules.yaml")
    cfg = PromptConfig(style_file=str(LTX_DIR / "prompts" / "style.md"), media="video", batch_size=20, rules=rules)
    assert cfg.batch_passage is False
    assert rules.subject_source == "text_and_context"
    b = gp._Builder(make_segments(), [1, 2], cfg)
    assert "setting" not in b.schema["items"]["properties"]
    assert "segment_index: 2\nText (what this segment says)" in b.batches([1, 2])[0].user
    assert SELF_CHECK_BULLET in b.system


# ---------------------------------------------------------------------------
# Файлы профиля cartoon_photos (только этот профиль)
# ---------------------------------------------------------------------------

def test_cartoon_profile_files_enable_passage_and_text_and_context():
    profile = (CARTOON_DIR / "profile.yaml").read_text(encoding="utf-8").splitlines()
    assert "prompt_batch_form: passage" in profile
    rules = gp.load_rules(CARTOON_DIR / "prompts" / "rules.yaml")
    assert rules.subject_source == "text_and_context"
    assert rules.dangling_text_check is True
    ltx_profile = (LTX_DIR / "profile.yaml").read_text(encoding="utf-8")
    assert "prompt_batch_form" not in ltx_profile
