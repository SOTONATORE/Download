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
    assert out[0]["appearance"] == "Man in his fifties, and grey beard"


def test_known_character_unchanged():
    before = json.dumps([ANNA], sort_keys=True)
    out = cf.merge_characters([ANNA], [_item("ANNA", new_traits=["woman in her thirties"], keywords=["Anna"])])
    assert len(out) == 1
    assert out[0]["appearance"] == ANNA["appearance"]
    assert json.dumps([ANNA], sort_keys=True) == before  # вход не меняется


def test_new_permanent_trait_appended():
    out = cf.merge_characters([BOB], [_item("BOB", new_traits=["grey beard"])])
    assert len(out) == 1
    assert out[0]["appearance"] == "A man in his forties with short grey hair, and grey beard"


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
    # 5b: запись без id и без objects мигрирует при чтении; сравниваем с мигрированной записью, не с исходной
    expected = {"srt_hash": record["srt_hash"], "characters": [dict(ANNA, id="CHAR_1")], "objects": []}
    assert out == expected
    stored = json.loads((tmp_path / cf.CHARACTERS_FILENAME).read_text(encoding="utf-8"))
    assert stored == expected


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
    assert out[0]["appearance"] == "Small brass key, and worn edges"


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


def test_build_marker_descriptions_is_removed():
    # мёртвая функция удалена (3a); маркеры по id обслуживает gemini_prompts.batch_descriptions
    assert not hasattr(cf, "build_marker_descriptions")


def test_normalize_appearance_strips_old_trailing_period():
    # старая запись с точкой в конце (как её писал прежний _sentence) читается как фраза
    assert cf.normalize_appearance("A woman in her thirties with short dark hair.") == \
        "a woman in her thirties with short dark hair"


def test_normalize_appearance_turns_inner_sentences_into_one_phrase():
    # две черты, склеенные «. », дают одну фразу без точки внутри
    assert cf.normalize_appearance("Man in his fifties. Grey beard.") == "man in his fifties, and grey beard"


def test_normalize_appearance_keeps_leading_acronym():
    assert cf.normalize_appearance("NATO flag.") == "NATO flag"


def test_sentence_one_trait_has_no_period():
    assert cf._sentence("a grey beard") == "A grey beard"
    assert cf._sentence("a grey beard.") == "A grey beard"


def test_two_new_traits_stored_as_one_phrase_without_period():
    out = cf.merge_characters([], [_item("CARL", ["a man in his fifties", "a grey beard"], keywords=["Carl"])])
    assert out[0]["appearance"] == "A man in his fifties, and a grey beard"
    assert "." not in out[0]["appearance"]


def test_normalize_appearance_one_old_trait_has_no_period():
    assert cf.normalize_appearance("Two-storey brick house.") == "two-storey brick house"


def test_normalize_appearance_does_not_split_on_us_abbreviation():
    assert cf.normalize_appearance("Man from the U.S. army. Grey beard.") == "man from the U.S. army, and grey beard"


def test_appearance_rule_asks_for_noun_phrase_and_example_matches():
    instr = cf.build_character_instructions()
    assert "short noun phrase in lowercase" in instr
    assert "with no final period, at most 25 words" in instr
    assert '"new_traits": ["a woman in her thirties with short dark hair"]' in instr
    assert '"new_traits": ["a two-storey brick house with white shutters"]' in instr


def test_id_format_matches_marker_regex():
    gp = cf.gp
    for rid in ("CHAR_1", "CHAR_12", "OBJ_1", "OBJ_40"):
        assert gp._MARKER_RX.fullmatch(f"[{rid}]"), rid


# ------------------------------------------------------------------ часть C: владелец, пустая внешность, миграция

def test_owner_schema_is_plain_string():
    props = cf.WINDOW_SCHEMA["properties"]["objects"]["items"]["properties"]
    assert props["owner"] == {"type": "string"}  # без anyOf и без null (3a)
    assert "null" not in json.dumps(cf.WINDOW_SCHEMA)


def test_owner_rule_says_empty_string():
    instr = cf.build_character_instructions()
    assert "or an empty string when no listed character owns it." in instr
    assert '"owner": ""' in instr


def test_owner_empty_string_stored_as_null_without_log(caplog):
    caplog.set_level(logging.WARNING, logger="Generate.character_file")
    raw = json.dumps({"characters": [], "objects": [{"name": "KEY", "aliases": [], "keywords": [],
                                                     "new_traits": ["small brass key"], "conflicts": [],
                                                     "owner": ""}]})
    parsed = cf.parse_window_reply(raw, "A key lies here.")
    out = cf.merge_objects([], parsed["objects"], [])
    assert out[0]["owner"] is None
    assert not any("owner not found" in m for m in caplog.messages)  # пустая строка — без записи в лог


def test_unknown_owner_name_via_parse_sets_null_and_logs(caplog):
    caplog.set_level(logging.WARNING, logger="Generate.character_file")
    raw = json.dumps({"characters": [], "objects": [{"name": "HOUSE", "aliases": [], "keywords": [],
                                                     "new_traits": ["brick house"], "conflicts": [],
                                                     "owner": "NOBODY"}]})
    parsed = cf.parse_window_reply(raw, "The house stands.")
    out = cf.merge_objects([], parsed["objects"], [dict(ANNA, id="CHAR_1")])
    assert out[0]["owner"] is None
    assert "owner not found, set to null: HOUSE" in caplog.messages


def test_same_name_other_owner_is_new_object_after_retry(monkeypatch):
    monkeypatch.setattr(cf, "WINDOW_SEGMENTS", 1)
    segs = [Segment(1, 0, 1000, "Anna and Bob talk."), Segment(2, 1000, 2000, "Bob sees the house.")]
    w1 = json.dumps({"characters": [
        {"name": "ANNA", "aliases": [], "keywords": ["Anna"], "new_traits": ["woman in her thirties"], "conflicts": []},
        {"name": "BOB", "aliases": [], "keywords": ["Bob"], "new_traits": ["man in his forties"], "conflicts": []}],
        "objects": [{"name": "HOUSE", "aliases": [], "keywords": [], "new_traits": ["two-storey brick house"],
                     "conflicts": [], "owner": "ANNA"}]})
    # второе окно: тот же HOUSE, но владелец BOB — новый объект; первый ответ без внешности, повтор с внешностью
    w2_bad = json.dumps({"characters": [], "objects": [{"name": "HOUSE", "aliases": [], "keywords": [],
                                                        "new_traits": [], "conflicts": [], "owner": "BOB"}]})
    w2_good = json.dumps({"characters": [], "objects": [{"name": "HOUSE", "aliases": [], "keywords": [],
                                                         "new_traits": ["wooden cabin"], "conflicts": [],
                                                         "owner": "BOB"}]})
    client = ScriptClient([w1, w2_bad, w2_good])
    record = cf.build_character_file(segs, client, {})
    assert [(o["id"], o["owner"]) for o in record["objects"]] == [("OBJ_1", "CHAR_1"), ("OBJ_2", "CHAR_2")]
    assert record["objects"][1]["appearance"] == "Wooden cabin"
    assert len(client.calls) == 3  # окно 1, окно 2 и один повтор окна 2


def test_empty_new_object_twice_raises_named_error_and_stores_nothing(tmp_path):
    segs = [Segment(1, 0, 1000, "The key lies here.")]
    bad = json.dumps({"characters": [], "objects": [{"name": "KEY", "aliases": [], "keywords": ["key"],
                                                     "new_traits": [], "conflicts": [], "owner": ""}]})
    client = ScriptClient([bad, bad])
    with pytest.raises(cf.EmptyAppearanceError) as exc:
        cf.get_character_file(tmp_path, segs, client)
    assert exc.value.record_name == "KEY"
    assert "KEY" in str(exc.value)
    assert isinstance(exc.value, cf.CharacterWindowError)
    assert len(client.calls) == 2  # один повтор, не больше
    assert not (tmp_path / cf.CHARACTERS_FILENAME).exists()
    assert not (tmp_path / cf.PARTIAL_FILENAME).exists()


def test_empty_new_character_twice_raises_named_error(tmp_path):
    segs = [Segment(1, 0, 1000, "Carl enters.")]
    bad = json.dumps({"characters": [{"name": "CARL", "aliases": [], "keywords": ["Carl"],
                                      "new_traits": [], "conflicts": []}]})
    client = ScriptClient([bad, bad])
    with pytest.raises(cf.EmptyAppearanceError) as exc:
        cf.get_character_file(tmp_path, segs, client)
    assert exc.value.record_name == "CARL"
    assert len(client.calls) == 2


def test_cached_file_without_ids_migrated_and_written_back(tmp_path):
    segs = [Segment(1, 0, 1000, "Anna walks in.")]
    old = {"srt_hash": srt_hash(segs), "characters": [dict(ANNA)],
           "objects": [{"name": "HOUSE", "appearance": "Brick house."}]}
    (tmp_path / cf.CHARACTERS_FILENAME).write_text(json.dumps(old), encoding="utf-8")

    class NoCallClient:
        def generate_json(self, *args, **kwargs):
            raise AssertionError("клиент не должен вызываться при совпавшем srt_hash")

    out = cf.get_character_file(tmp_path, segs, NoCallClient())
    stored = json.loads((tmp_path / cf.CHARACTERS_FILENAME).read_text(encoding="utf-8"))
    assert stored == out
    assert set(stored) == {"srt_hash", "characters", "objects"}
    assert stored["srt_hash"] == srt_hash(segs)  # srt_hash не меняется
    assert [c["id"] for c in stored["characters"]] == ["CHAR_1"]
    assert stored["objects"] == [{"id": "OBJ_1", "name": "HOUSE", "owner": None, "aliases": [],
                                  "keywords": [], "appearance": "Brick house."}]


def test_loaded_character_keeps_its_id_new_one_gets_next():
    out = cf.merge_characters([dict(ANNA, id="CHAR_5")], [_item("BOB", ["man in his forties"])])
    assert [c["id"] for c in out] == ["CHAR_5", "CHAR_6"]
    assert out[0]["appearance"] == ANNA["appearance"]
