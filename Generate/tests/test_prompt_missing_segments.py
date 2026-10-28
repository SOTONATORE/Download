"""Тесты H3, часть B: недостающие промпты, батчи по 50, описания персонажей по батчу.

Сеть не используется: транспорт подменён, время подменено.
"""
import json
import pathlib
import socket

import pytest

from Generate.core import character_file as cf
from Generate.core import gemini_orchestrator as go
from Generate.core import gemini_prompts as gp
from Generate.core.gemini_prompts import PromptConfig
from Generate.core.srt_parser import Segment, srt_hash

TIMING = go.TimingParams(24, {"kind": "8k+1"}, 2.5, 5.0, 10.0, 0.5)
GOOD_PROMPT = "A wide shot of a quiet street at dawn."
ANNA = {"id": "CHAR_1", "name": "ANNA", "aliases": [], "keywords": ["Anna"], "appearance": "a woman in her thirties"}
BOB = {"id": "CHAR_2", "name": "BOB", "aliases": ["Robert"], "keywords": ["Bob"], "appearance": "a man in his forties"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")
    monkeypatch.setattr(socket.socket, "connect", boom)


class RecordingTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.users = []
        self.systems = []

    def generate_json(self, model, system, user, schema):
        self.users.append(user)
        self.systems.append(system)
        return self.replies.pop(0)


def cfg(tmp_path, **kw):
    style = tmp_path / "style.md"
    style.write_text("# Style\nWrite one paragraph.\n", encoding="utf-8")
    kw.setdefault("style_file", str(style))
    kw.setdefault("max_prompt_chars", 300)
    kw.setdefault("batch_size", 10)
    return PromptConfig(**kw)


def make_segs(n):
    return [Segment(i, i * 3000, i * 3000 + 1000, f"Sentence number {i}.") for i in range(1, n + 1)]


def _ready(n, text):
    return {"num": n, "start_ms": 0, "end_ms": 1, "clip_ms": 1, "num_frames": 1, "prompt": text,
            "status": "ready", "attempts": 1, "error": "", "needs_review": False, "extra": {}}


# ------------------------------------------------------------------ недостающие номера (B2)

def test_missing_segments_computed():
    prompts = {str(n): _ready(n, f"p{n}") for n in (1, 2, 4, 5)}
    assert go.missing_segments(prompts, [1, 2, 3, 4, 5]) == [3]


def test_failed_entry_counts_as_missing():
    prompts = {"1": {"status": "failed", "prompt": ""}}
    assert go.missing_segments(prompts, [1]) == [1]


def test_generate_only_missing_and_keep_others(tmp_path):
    segs = make_segs(5)
    path = tmp_path / "prompts.json"
    existing = {"srt_hash": srt_hash(segs), "model": "m", "profile": "ltx25",
                "prompts": {str(n): _ready(n, f"old {n}") for n in (1, 2, 4, 5)}}
    path.write_text(json.dumps(existing), encoding="utf-8")
    # первый ответ: записи персонажей (окно одно, персонажей нет); второй: промпт сегмента 3
    t = RecordingTransport([json.dumps({"characters": []}),
                            json.dumps([{"segment_index": 3, "prompt": GOOD_PROMPT}])])
    go.generate_and_store(segs, [1, 2, 3, 4, 5], cfg(tmp_path), t, path, profile="ltx25", timing=TIMING,
                          clock=lambda: 1000.0, sleep=lambda s: None)
    assert len(t.users) == 2
    assert "segment_index: 3\n" in t.users[1]
    assert "segment_index: 1\n" not in t.users[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data["prompts"]) == ["1", "2", "3", "4", "5"]
    for n in (1, 2, 4, 5):
        assert data["prompts"][str(n)]["prompt"] == f"old {n}"
    assert data["prompts"]["3"]["prompt"] == GOOD_PROMPT


# ------------------------------------------------------------------ батчи по 50 (B1)

def test_batches_of_fifty(tmp_path):
    segs = make_segs(120)
    reqs = gp.build_requests(segs, list(range(1, 121)), cfg(tmp_path, batch_size=50))
    assert [len(r.nums) for r in reqs] == [50, 50, 20]


def test_config_default_batch_size_is_fifty():
    text = (pathlib.Path(__file__).resolve().parents[1] / "config.default.yaml").read_text(encoding="utf-8")
    assert "gemini_batch_size: 50\n" in text


# ------------------------------------------------------------------ описания персонажей по батчу (B3, B4)

def test_batch_descriptions_only_characters_in_batch():
    chars = [ANNA, BOB, {"id": "CHAR_3", "name": "CARL", "aliases": [], "keywords": ["Carl"], "appearance": ""}]
    assert gp.batch_descriptions(chars, "Before: Anna waves. Text: nothing else") == {"CHAR_1": "a woman in her thirties"}
    assert gp.batch_descriptions(chars, "Robert sits down") == {"CHAR_2": "a man in his forties"}
    assert gp.batch_descriptions(chars, "Carl enters") == {}  # запись без внешности не передаётся


def test_batch_descriptions_normalizes_old_cached_appearance():
    # старая запись с точкой в конце: в батч уходит фраза без точки и со строчной буквы
    old = dict(ANNA, appearance="A woman in her thirties with short dark hair.")
    assert gp.batch_descriptions([old], "Anna waves.") == {"CHAR_1": "a woman in her thirties with short dark hair"}


def test_batch_descriptions_normalizer_matches_character_file():
    # копия в gemini_prompts должна давать тот же результат, что normalize_appearance в character_file
    samples = ["A woman in her thirties with short dark hair.", "Man in his fifties. Grey beard.",
               "NATO flag.", "  a brick house  ", "Two-storey brick house. White shutters.", "",
               # новая форма: одна фраза с «, and », без точки
               "a man in his fifties, and a grey beard", "A man in his fifties, and a grey beard",
               # старая форма с одной чертой и с «U.S.», где точка не делит фразу
               "Two-storey brick house.", "Man from the U.S. army. Grey beard.",
               # новая форма записи: одна фраза с предлогом и без точки, с точкой и с пробелами по краям
               "A woman in her thirties with short dark hair", "a two-storey brick house with white shutters",
               "A two-storey brick house with white shutters.", "  A two-storey brick house with white shutters  "]
    for text in samples:
        assert gp._normalize_appearance(text) == cf.normalize_appearance(text), text


def test_new_single_phrase_values_match_in_both_copies():
    # новая фраза возвращается строчной, без точки; артикль не добавляется
    cases = [("A woman in her thirties with short dark hair", "a woman in her thirties with short dark hair"),
             ("A two-storey brick house with white shutters.", "a two-storey brick house with white shutters"),
             ("Two-storey brick house with white shutters", "two-storey brick house with white shutters")]
    for raw, expected in cases:
        assert gp._normalize_appearance(raw) == expected, raw
        assert cf.normalize_appearance(raw) == expected, raw


def test_batch_descriptions_keys_characters_and_objects():
    house = {"id": "OBJ_1", "name": "HOUSE", "aliases": [], "keywords": ["house"],
             "appearance": "a brick house", "owner": "CHAR_1"}
    out = gp.batch_descriptions([ANNA, house], "Anna looks at the house.")
    assert out == {"CHAR_1": "a woman in her thirties", "OBJ_1": "a brick house"}


def test_marker_instruction_lists_ids_not_names():
    text = gp.marker_instruction(["CHAR_1", "OBJ_2"])
    assert "CHAR_1, OBJ_2" in text
    assert "[CHAR_1]" in text and "[OBJ_2]" in text
    assert "ANNA" not in text and "HOUSE" not in text


def test_marker_instruction_in_user_message_not_system(tmp_path):
    segs = [Segment(1, 0, 1000, "Anna walks in.")]
    t = RecordingTransport([json.dumps([{"segment_index": 1,
                                         "prompt": "[CHAR_1] walks in a quiet street at dawn."}])])
    res = gp.generate_prompts(segs, [1], cfg(tmp_path), t, "m", characters=[ANNA])
    assert "CHAR_1" in t.users[0]
    assert "ANNA" not in t.users[0]  # имя не входит в инструкцию о маркерах
    assert "a woman in her thirties" not in t.users[0]
    assert "[CHAR_1]" not in t.systems[0]
    assert res[0].prompt == "A woman in her thirties walks in a quiet street at dawn."
    assert res[0].needs_review is False


def test_old_cached_appearance_reads_as_one_sentence_in_prompt(tmp_path):
    # старые записи (точки в конце и между чертами) дают одно предложение без точки внутри
    old_anna = dict(ANNA, appearance="A woman in her thirties with short dark hair.")
    old_house = {"id": "OBJ_1", "name": "HOUSE", "aliases": [], "keywords": ["house"],
                 "appearance": "Two-storey brick house.", "owner": "CHAR_1"}
    segs = [Segment(1, 0, 1000, "Anna stands by the house.")]
    t = RecordingTransport([json.dumps([{"segment_index": 1,
                                         "prompt": "[CHAR_1] stands by [OBJ_1] in the quiet street."}])])
    # фикстура без артикля, поэтому в ожидаемом тексте тоже нет «a»; код артикль не добавляет
    assert gp._normalize_appearance(old_anna["appearance"]) == "a woman in her thirties with short dark hair"
    res = gp.generate_prompts(segs, [1], cfg(tmp_path), t, "m", characters=[old_anna, old_house])
    assert res[0].prompt == ("A woman in her thirties with short dark hair stands by two-storey brick house "
                             "in the quiet street.")


def test_generate_prompts_receives_object_ids_and_resolves_them(tmp_path):
    house = {"id": "OBJ_1", "name": "HOUSE", "aliases": [], "keywords": ["house"],
             "appearance": "a brick house", "owner": "CHAR_1"}
    segs = [Segment(1, 0, 1000, "Anna enters the house.")]
    t = RecordingTransport([json.dumps([{"segment_index": 1, "prompt": "[CHAR_1] enters [OBJ_1]."}])])
    res = gp.generate_prompts(segs, [1], cfg(tmp_path), t, "m", characters=[ANNA, house])
    assert "OBJ_1" in t.users[0]
    assert "a brick house" not in t.users[0]
    assert res[0].prompt == "A woman in her thirties enters a brick house."
