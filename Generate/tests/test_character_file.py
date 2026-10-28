"""Тесты core/character_file: слияние записей о персонажах и кэш characters.json. Сеть не используется."""
import json
import logging

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
    record = {"srt_hash": srt_hash(segs), "characters": [ANNA], "skipped_windows": []}
    (tmp_path / cf.CHARACTERS_FILENAME).write_text(json.dumps(record), encoding="utf-8")

    class NoCallClient:
        def generate_json(self, *args, **kwargs):
            raise AssertionError("клиент не должен вызываться при совпавшем srt_hash")

    out = cf.get_character_file(tmp_path, segs, NoCallClient())
    assert out == record
