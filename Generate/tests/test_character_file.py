"""Тесты core/character_file: слияние записей о персонажах и кэш characters.json. Сеть не используется."""
import json
import logging

import pytest

from Generate.core import character_file as cf
from Generate.core.srt_parser import Segment, srt_hash

ANNA = {"name": "ANNA", "aliases": [], "keywords": ["Anna"],
        "appearance": "A woman in her thirties with short dark hair."}
BOB = {"name": "BOB", "aliases": [], "keywords": ["Bob"],
       "appearance": "A man in his forties with short grey hair."}


def _item(name, new_traits=(), conflicts=(), keywords=(), aliases=()):
    return {"name": name, "aliases": list(aliases), "keywords": list(keywords),
            "new_traits": list(new_traits), "conflicts": list(conflicts)}


def test_new_character_added():
    out = cf.merge_characters([], [_item("CARL", ["man in his fifties", "grey beard"], keywords=["Carl"])])
    assert len(out) == 1
    assert out[0]["name"] == "CARL"
    assert out[0]["keywords"] == ["Carl"]
    assert out[0]["appearance"] == "Man in his fifties. Grey beard."


def test_known_character_unchanged():
    before = json.dumps([ANNA], sort_keys=True)
    out = cf.merge_characters([ANNA], [_item("ANNA", new_traits=["woman in her thirties"], keywords=["Anna"])])
    assert len(out) == 1
    assert out[0]["appearance"] == ANNA["appearance"]
    assert json.dumps([ANNA], sort_keys=True) == before  # вход не меняется


def test_new_permanent_trait_appended():
    out = cf.merge_characters([BOB], [_item("BOB", new_traits=["grey beard"])])
    assert len(out) == 1
    assert out[0]["appearance"] == "A man in his forties with short grey hair. Grey beard."


def test_conflicting_trait_not_recorded_and_logged(caplog):
    caplog.set_level(logging.WARNING, logger="Generate.character_file")
    out = cf.merge_characters([ANNA], [_item("ANNA", conflicts=["blonde hair"])])
    assert out[0]["appearance"] == ANNA["appearance"]
    assert "conflict, not recorded: ANNA, blonde hair" in caplog.messages


def test_cached_file_with_matching_hash_returned_without_client(tmp_path):
    segs = [Segment(1, 0, 1000, "Anna walks in.")]
    record = {"srt_hash": srt_hash(segs), "characters": [ANNA]}
    (tmp_path / cf.CHARACTERS_FILENAME).write_text(json.dumps(record), encoding="utf-8")

    class NoCallClient:
        def generate_json(self, *args, **kwargs):
            raise AssertionError("клиент не должен вызываться при совпавшем srt_hash")

    out = cf.get_character_file(tmp_path, segs, NoCallClient())
    assert out == record


# ------------------------------------------------------------------ окна, частичный файл (часть A)

GOOD_ANNA = json.dumps({"characters": [{"name": "ANNA", "aliases": [], "keywords": ["Anna"],
                                        "new_traits": ["woman in her thirties"], "conflicts": []}]})


def _six_segments():
    texts = ["Anna walks in.", "Anna sits.", "Bob talks.", "Bob leaves.", "Anna smiles.", "Anna waves."]
    return [Segment(i + 1, i * 1000, i * 1000 + 900, t) for i, t in enumerate(texts)]


class ScriptClient:
    """Отдаёт ответы по порядку и запоминает тексты запросов. Сети нет."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_json(self, model, system, user, schema):
        self.calls.append(user)
        return self.replies.pop(0)


def test_failing_window_raises_and_writes_no_characters_json(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "WINDOW_SEGMENTS", 2)
    client = ScriptClient([GOOD_ANNA, "не json", "не json"])
    with pytest.raises(cf.CharacterWindowError) as exc:
        cf.get_character_file(tmp_path, _six_segments(), client)
    assert (exc.value.first, exc.value.last) == (3, 4)
    assert not (tmp_path / cf.CHARACTERS_FILENAME).exists()
    partial = json.loads((tmp_path / cf.PARTIAL_FILENAME).read_text(encoding="utf-8"))
    assert partial["finished"] == [{"first": 1, "last": 2}]
    assert [c["name"] for c in partial["characters"]] == ["ANNA"]
    assert len(client.calls) == 3  # окно 1, окно 2 дважды; окно 3 не запрашивалось


def test_partial_file_lets_next_run_skip_finished_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "WINDOW_SEGMENTS", 2)
    segs = _six_segments()
    with pytest.raises(cf.CharacterWindowError):
        cf.get_character_file(tmp_path, segs, ScriptClient([GOOD_ANNA, "x", "x"]))
    client = ScriptClient([GOOD_ANNA, GOOD_ANNA])
    cf.get_character_file(tmp_path, segs, client)
    assert len(client.calls) == 2  # только окна 2 и 3
    assert "Bob talks." in client.calls[0]
    assert "Anna smiles." in client.calls[1]
    assert (tmp_path / cf.CHARACTERS_FILENAME).is_file()
    assert not (tmp_path / cf.PARTIAL_FILENAME).exists()


def test_completed_run_writes_characters_json_and_removes_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "WINDOW_SEGMENTS", 2)
    segs = _six_segments()
    partial = {"srt_hash": srt_hash(segs), "characters": [], "finished": [{"first": 1, "last": 2}]}
    (tmp_path / cf.PARTIAL_FILENAME).write_text(json.dumps(partial), encoding="utf-8")
    client = ScriptClient([GOOD_ANNA, GOOD_ANNA])
    out = cf.get_character_file(tmp_path, segs, client)
    stored = json.loads((tmp_path / cf.CHARACTERS_FILENAME).read_text(encoding="utf-8"))
    assert stored == out
    assert set(stored) == {"srt_hash", "characters"}
    assert not (tmp_path / cf.PARTIAL_FILENAME).exists()
    assert len(client.calls) == 2
