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
        self.systems = []

    def generate_json(self, model, system, user, schema):
        self.calls.append(user)
        self.systems.append(system)
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
    assert set(stored) == {"srt_hash", "characters", "objects"}
    assert not (tmp_path / cf.PARTIAL_FILENAME).exists()
    assert len(client.calls) == 2


# ------------------------------------------------------------------ текст запроса (без сети)

def test_request_text_asks_for_appearance_even_without_visual_details():
    text = "Anna walks in.\nBob talks."  # в окне нет описаний внешности
    instr = cf.build_character_instructions()
    assert "permanent appearance" in instr
    assert "Return only one JSON object, with no markdown, no code fences and no text before or after it." in instr
    user = cf._window_request(text, [])
    assert "always contains at least age range, build, hair, and facial features" in instr
    assert "do not add any identifier beyond the name" in instr
    assert "Do not add any uncertainty marker or extra field." in instr
    assert "always contains at least age range" not in user  # правила не дублируются в запросе окна
    assert "do not add any identifier beyond the name" not in user
    assert "Do not add any uncertainty marker or extra field." not in user


def test_instruction_covers_characters_objects_and_appearance():
    instr = cf.build_character_instructions()
    assert "Rules for characters:" in instr
    assert "Rules for objects:" in instr
    assert "Appearance rule:" in instr
    assert "owner:" in instr
    assert "Do not describe clothing or temporary state" in instr


def test_window_request_has_no_old_permanent_visible_wording():
    user = cf._window_request("Anna walks in.", [ANNA])
    assert "only permanent visible traits" not in user
    assert "CHAR_1 ANNA" not in user  # ANNA здесь без id: строка без id
    assert "ANNA: A woman in her thirties with short dark hair." in user


# ------------------------------------------------------------------ id, миграция, объекты (часть B)

def _obj(name, owner=None, new_traits=(), conflicts=(), keywords=(), aliases=()):
    return {"name": name, "aliases": list(aliases), "keywords": list(keywords),
            "new_traits": list(new_traits), "conflicts": list(conflicts), "owner": owner}


def test_character_ids_assigned_in_order():
    out = cf.merge_characters([], [_item("ANNA", ["woman in her thirties"]),
                                   _item("BOB", ["man in his forties"])])
    assert [c["id"] for c in out] == ["CHAR_1", "CHAR_2"]


def test_old_record_without_id_gets_id_on_migration():
    old = {"srt_hash": "h", "characters": [dict(ANNA), dict(BOB)]}
    before = json.dumps(old, sort_keys=True)
    new = cf.migrate_record(old)
    assert [c["id"] for c in new["characters"]] == ["CHAR_1", "CHAR_2"]
    assert new["objects"] == []
    assert json.dumps(old, sort_keys=True) == before  # вход не меняется


def test_migration_keeps_existing_ids_and_numbers_new_ones_after_them():
    old = {"srt_hash": "h", "characters": [dict(ANNA, id="CHAR_3"), dict(BOB)]}
    new = cf.migrate_record(old)
    assert [c["id"] for c in new["characters"]] == ["CHAR_3", "CHAR_4"]


def test_same_name_objects_with_different_owners_stay_separate():
    chars = [dict(ANNA, id="CHAR_1"), dict(BOB, id="CHAR_2")]
    out = cf.merge_objects([], [_obj("HOUSE", "ANNA", ["two-storey brick house"]),
                                _obj("HOUSE", "BOB", ["wooden cabin"])], chars)
    assert len(out) == 2
    assert [(o["id"], o["owner"]) for o in out] == [("OBJ_1", "CHAR_1"), ("OBJ_2", "CHAR_2")]


def test_same_name_objects_without_owner_merge_into_one():
    out = cf.merge_objects([], [_obj("KEY", None, ["small brass key"]),
                                _obj("KEY", None, ["brass key", "worn edges"])], [])
    assert len(out) == 1
    assert out[0]["owner"] is None
    assert out[0]["appearance"] == "Small brass key. Worn edges."


def test_unknown_owner_name_sets_owner_null_and_logs(caplog):
    caplog.set_level(logging.WARNING, logger="Generate.character_file")
    out = cf.merge_objects([], [_obj("HOUSE", "NOBODY", ["two-storey brick house"])],
                           [dict(ANNA, id="CHAR_1")])
    assert out[0]["owner"] is None
    assert "owner not found, set to null: HOUSE" in caplog.messages


def test_new_object_without_visual_details_gets_appearance_from_stub():
    segs = [Segment(1, 0, 1000, "The house stands on the hill.")]
    stub_reply = json.dumps({"characters": [], "objects": [
        {"name": "HOUSE", "aliases": [], "keywords": ["house"],
         "new_traits": ["two-storey brick house", "white shutters"], "conflicts": [], "owner": None}]})
    client = ScriptClient([stub_reply])
    record = cf.build_character_file(segs, client, {})
    assert len(record["objects"]) == 1
    assert record["objects"][0]["appearance"]
    assert "brick house" in record["objects"][0]["appearance"]
    assert "Rules for objects:" in client.systems[0]


def test_new_object_without_appearance_is_rejected_for_retry():
    segs = [Segment(1, 0, 1000, "The house stands on the hill.")]
    bad = json.dumps({"characters": [], "objects": [
        {"name": "HOUSE", "aliases": [], "keywords": [], "new_traits": [], "conflicts": [], "owner": None}]})
    with pytest.raises(cf.CharacterWindowError):
        cf.build_character_file(segs, ScriptClient([bad, bad]), {})


def test_marker_descriptions_keyed_by_id():
    record = {"characters": [dict(ANNA, id="CHAR_1")],
              "objects": [{"id": "OBJ_1", "name": "HOUSE", "owner": "CHAR_1", "aliases": [],
                           "keywords": [], "appearance": "Two-storey brick house."}]}
    assert cf.build_marker_descriptions(record) == {
        "CHAR_1": ANNA["appearance"],
        "OBJ_1": "Two-storey brick house.",
    }


def test_id_format_matches_marker_regex():
    gp = cf.gp
    for rid in ("CHAR_1", "CHAR_12", "OBJ_1", "OBJ_40"):
        assert gp._MARKER_RX.fullmatch(f"[{rid}]"), rid
