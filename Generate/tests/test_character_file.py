"""Тесты core/character_file: слияние записей о персонажах и кэш characters.json. Сеть не используется."""
import json
import logging

import pytest

from Generate.core import character_file as cf
from Generate.core import log_setup
from Generate.core.srt_parser import Segment, srt_hash

ANNA = {"name": "ANNA", "aliases": [], "keywords": ["Anna"],
        "appearance": "A woman in her thirties with short dark hair."}
BOB = {"name": "BOB", "aliases": [], "keywords": ["Bob"],
       "appearance": "A man in his forties with short grey hair."}


def _item(name, appearance="", rid="", keywords=(), aliases=()):
    return {"id": rid, "name": name, "aliases": list(aliases), "keywords": list(keywords),
            "appearance": appearance}


def test_new_character_added():
    out = cf.merge_characters([], [_item("CARL", "a man in his fifties with a grey beard", keywords=["Carl"])])
    assert len(out) == 1
    assert out[0]["name"] == "CARL"
    assert out[0]["keywords"] == ["Carl"]
    assert out[0]["appearance"] == "A man in his fifties with a grey beard"


def test_known_character_unchanged():
    before = json.dumps([ANNA], sort_keys=True)
    out = cf.merge_characters([dict(ANNA, id="CHAR_1")],
                              [_item("ANNA", ANNA["appearance"], rid="CHAR_1", keywords=["Anna"])])
    assert len(out) == 1
    assert out[0]["appearance"] == ANNA["appearance"]
    assert json.dumps([ANNA], sort_keys=True) == before  # вход не меняется


def test_known_record_takes_changed_appearance():
    # раньше новая черта не дописывалась; теперь модель возвращает полную внешность, и она записывается
    out = cf.merge_characters([dict(BOB, id="CHAR_2")],
                              [_item("BOB", "a man in his forties with short grey hair and a grey beard", rid="CHAR_2")])
    assert len(out) == 1
    assert out[0]["appearance"] == "a man in his forties with short grey hair and a grey beard"


def test_conflicts_are_not_logged_any_more(caplog):
    # поле conflicts удалено из схемы и из лога; прежняя строка «conflict, not recorded» не пишется
    caplog.set_level(logging.WARNING, logger="Generate.character_file")
    out = cf.merge_characters([dict(ANNA, id="CHAR_1")], [_item("ANNA", ANNA["appearance"], rid="CHAR_1")])
    assert out[0]["appearance"] == ANNA["appearance"]
    assert not any("conflict" in m for m in caplog.messages)


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

GOOD_ANNA = json.dumps({"characters": [{"id": "", "name": "ANNA", "aliases": [], "keywords": ["Anna"],
                                        "appearance": "woman in her thirties"}]})
# тот же ANNA в следующем окне: id уже известен (CHAR_1), повтор новой записи дал бы дубликат имени
GOOD_ANNA_KNOWN = json.dumps({"characters": [{"id": "CHAR_1", "name": "ANNA", "aliases": [], "keywords": ["Anna"],
                                              "appearance": "woman in her thirties"}]})


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
    client = ScriptClient([GOOD_ANNA_KNOWN, GOOD_ANNA_KNOWN])
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
    client = ScriptClient([GOOD_ANNA, GOOD_ANNA_KNOWN])  # окно 3 даёт ANNA новой записью, окно 5 — известной
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
    assert "all permanent traits" in instr
    assert "Return only one JSON object, with no markdown, no code fences and no text before or after it." in instr
    user = cf._window_request(text, [])
    assert "age range, build, hair, and facial features for a person" in instr
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
    user = cf._window_request("Anna walks in.", [dict(ANNA, id="CHAR_1")])
    assert "only permanent visible traits" not in user
    assert "id: CHAR_1 | name: ANNA | previous appearance: A woman in her thirties with short dark hair." in user


# ------------------------------------------------------------------ id, миграция, объекты (часть B)

def _obj(name, owner=None, appearance="", rid="", keywords=(), aliases=()):
    return {"id": rid, "name": name, "aliases": list(aliases), "keywords": list(keywords),
            "appearance": appearance, "owner": owner}


def test_character_ids_assigned_in_order():
    out = cf.merge_characters([], [_item("ANNA", "woman in her thirties"),
                                   _item("BOB", "man in his forties")])
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
    out = cf.merge_objects([], [_obj("HOUSE", "ANNA", "two-storey brick house"),
                                _obj("HOUSE", "BOB", "wooden cabin")], chars)
    assert len(out) == 2
    assert [(o["id"], o["owner"]) for o in out] == [("OBJ_1", "CHAR_1"), ("OBJ_2", "CHAR_2")]


def test_same_name_objects_without_owner_are_separate_new_records():
    # раньше два объекта с одним именем без владельца сливались по имени; теперь id решает, новая ли запись.
    # merge_objects сам повторы имён не проверяет; повтор имени внутри одного ответа ловит путь окна (_check_new_objects).
    out = cf.merge_objects([], [_obj("KEY", None, "small brass key"),
                                _obj("KEY", None, "brass key")], [])
    assert [o["id"] for o in out] == ["OBJ_1", "OBJ_2"]
    assert out[0]["owner"] is None
    assert out[0]["appearance"] == "Small brass key"


def test_unknown_owner_name_sets_owner_null_and_logs(caplog):
    caplog.set_level(logging.WARNING, logger="Generate.character_file")
    out = cf.merge_objects([], [_obj("HOUSE", "NOBODY", "two-storey brick house")],
                           [dict(ANNA, id="CHAR_1")])
    assert out[0]["owner"] is None
    assert "owner not found, set to null: HOUSE" in caplog.messages


def test_new_object_without_visual_details_gets_appearance_from_stub():
    segs = [Segment(1, 0, 1000, "The house stands on the hill.")]
    stub_reply = json.dumps({"characters": [], "objects": [
        {"id": "", "name": "HOUSE", "aliases": [], "keywords": ["house"],
         "appearance": "two-storey brick house with white shutters", "owner": ""}]})
    client = ScriptClient([stub_reply])
    record = cf.build_character_file(segs, client, {})
    assert len(record["objects"]) == 1
    assert record["objects"][0]["appearance"]
    assert "brick house" in record["objects"][0]["appearance"]
    assert "Rules for objects:" in client.systems[0]


def test_new_object_without_appearance_is_rejected_for_retry():
    segs = [Segment(1, 0, 1000, "The house stands on the hill.")]
    bad = json.dumps({"characters": [], "objects": [
        {"id": "", "name": "HOUSE", "aliases": [], "keywords": [], "appearance": "", "owner": ""}]})
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


def test_new_phrase_stored_capitalized_without_period():
    out = cf.merge_characters([], [_item("CARL", "a man in his fifties with a grey beard", keywords=["Carl"])])
    assert out[0]["appearance"] == "A man in his fifties with a grey beard"
    assert "." not in out[0]["appearance"]


def test_normalize_appearance_one_old_trait_has_no_period():
    assert cf.normalize_appearance("Two-storey brick house.") == "two-storey brick house"


def test_normalize_appearance_does_not_split_on_us_abbreviation():
    assert cf.normalize_appearance("Man from the U.S. army. Grey beard.") == "man from the U.S. army, and grey beard"


def test_appearance_rule_asks_for_noun_phrase_and_example_matches():
    instr = cf.build_character_instructions()
    assert "appearance is one complete phrase in lowercase" in instr
    assert "No final period. At most 30 words." in instr
    assert '"appearance": "a woman in her thirties with short dark hair"' in instr
    assert '"appearance": "a two-storey brick house with white shutters"' in instr


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
    raw = json.dumps({"characters": [], "objects": [{"id": "", "name": "KEY", "aliases": [], "keywords": [],
                                                     "appearance": "small brass key", "owner": ""}]})
    parsed = cf.parse_window_reply(raw, "A key lies here.")
    out = cf.merge_objects([], parsed["objects"], [])
    assert out[0]["owner"] is None
    assert not any("owner not found" in m for m in caplog.messages)  # пустая строка — без записи в лог


def test_unknown_owner_name_via_parse_sets_null_and_logs(caplog):
    caplog.set_level(logging.WARNING, logger="Generate.character_file")
    raw = json.dumps({"characters": [], "objects": [{"id": "", "name": "HOUSE", "aliases": [], "keywords": [],
                                                     "appearance": "brick house", "owner": "NOBODY"}]})
    parsed = cf.parse_window_reply(raw, "The house stands.")
    out = cf.merge_objects([], parsed["objects"], [dict(ANNA, id="CHAR_1")])
    assert out[0]["owner"] is None
    assert "owner not found, set to null: HOUSE" in caplog.messages


def test_same_name_other_owner_is_new_object_after_retry(monkeypatch):
    monkeypatch.setattr(cf, "WINDOW_SEGMENTS", 1)
    segs = [Segment(1, 0, 1000, "Anna and Bob talk."), Segment(2, 1000, 2000, "Bob sees the house.")]
    w1 = json.dumps({"characters": [
        {"id": "", "name": "ANNA", "aliases": [], "keywords": ["Anna"], "appearance": "woman in her thirties"},
        {"id": "", "name": "BOB", "aliases": [], "keywords": ["Bob"], "appearance": "man in his forties"}],
        "objects": [{"id": "", "name": "HOUSE", "aliases": [], "keywords": [],
                     "appearance": "two-storey brick house", "owner": "ANNA"}]})
    # второе окно: тот же HOUSE, но владелец BOB — новый объект; первый ответ без внешности, повтор с внешностью
    w2_bad = json.dumps({"characters": [], "objects": [{"id": "", "name": "HOUSE", "aliases": [], "keywords": [],
                                                        "appearance": "", "owner": "BOB"}]})
    w2_good = json.dumps({"characters": [], "objects": [{"id": "", "name": "HOUSE", "aliases": [], "keywords": [],
                                                         "appearance": "wooden cabin", "owner": "BOB"}]})
    client = ScriptClient([w1, w2_bad, w2_good])
    record = cf.build_character_file(segs, client, {})
    assert [(o["id"], o["owner"]) for o in record["objects"]] == [("OBJ_1", "CHAR_1"), ("OBJ_2", "CHAR_2")]
    assert record["objects"][1]["appearance"] == "Wooden cabin"
    assert len(client.calls) == 3  # окно 1, окно 2 и один повтор окна 2


def test_empty_new_object_twice_raises_named_error_and_stores_nothing(tmp_path):
    segs = [Segment(1, 0, 1000, "The key lies here.")]
    bad = json.dumps({"characters": [], "objects": [{"id": "", "name": "KEY", "aliases": [], "keywords": ["key"],
                                                     "appearance": "", "owner": ""}]})
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
    bad = json.dumps({"characters": [{"id": "", "name": "CARL", "aliases": [], "keywords": ["Carl"],
                                      "appearance": ""}]})
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
    out = cf.merge_characters([dict(ANNA, id="CHAR_5")], [_item("BOB", "man in his forties")])
    assert [c["id"] for c in out] == ["CHAR_5", "CHAR_6"]
    assert out[0]["appearance"] == ANNA["appearance"]


# ------------------------------------------------------------------ часть D: одна фраза на запись

def test_new_character_one_phrase_stored_capitalized_no_period():
    out = cf.merge_characters([], [_item("ANNA", "a woman in her thirties with short dark hair.")])
    assert out[0]["appearance"] == "A woman in her thirties with short dark hair"
    assert not out[0]["appearance"].endswith(".")


def test_new_object_one_phrase_stored_capitalized_no_period():
    out = cf.merge_objects([], [_obj("HOUSE", None, "a two-storey brick house with white shutters")], [])
    assert out[0]["appearance"] == "A two-storey brick house with white shutters"


def test_existing_record_changed_appearance_is_stored_and_logged(caplog):
    # раньше внешность известной записи не менялась; теперь новая возвращённая внешность записывается (6a)
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    new = "a woman in her thirties with short dark hair and a scar on her left cheek"
    pending: list = []
    out = cf.merge_characters([dict(ANNA, id="CHAR_1")], [_item("ANNA", new, rid="CHAR_1")], pending=pending)
    cf._flush_pending(pending)  # в окне строка пишется после принятия; здесь вызов имитирует принятие
    assert out[0]["appearance"] == new
    assert ("appearance updated: CHAR_1 ANNA: A woman in her thirties with short dark hair. -> " + new
            ) in caplog.messages


def test_existing_record_same_appearance_logs_nothing(caplog):
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    out = cf.merge_characters([dict(ANNA, id="CHAR_1")], [_item("ANNA", ANNA["appearance"], rid="CHAR_1")])
    assert out[0]["appearance"] == ANNA["appearance"]
    assert not any("appearance updated" in m for m in caplog.messages)


def test_empty_new_item_retried_once_and_good_retry_is_stored():
    segs = [Segment(1, 0, 1000, "Carl enters.")]
    bad = json.dumps({"characters": [{"id": "", "name": "CARL", "aliases": [], "keywords": ["Carl"],
                                      "appearance": ""}]})
    good = json.dumps({"characters": [{"id": "", "name": "CARL", "aliases": [], "keywords": ["Carl"],
                                       "appearance": "a man in his fifties with a grey beard"}]})
    client = ScriptClient([bad, good])
    record = cf.build_character_file(segs, client, {})
    assert len(client.calls) == 2
    assert record["characters"][0]["appearance"] == "A man in his fifties with a grey beard"


def test_empty_new_item_twice_raises_and_stores_nothing():
    segs = [Segment(1, 0, 1000, "Carl enters.")]
    bad = json.dumps({"characters": [{"id": "", "name": "CARL", "aliases": [], "keywords": ["Carl"],
                                      "appearance": ""}]})
    client = ScriptClient([bad, bad])
    written = []
    with pytest.raises(cf.EmptyAppearanceError) as exc:
        cf.build_character_file(segs, client, {}, on_window=written.append)
    assert exc.value.record_name == "CARL"
    assert len(client.calls) == 2
    assert written == []  # окно не сохранено ни в каком виде


def test_normalize_new_phrase_lowercased_no_period():
    assert cf.normalize_appearance("A woman in her thirties with short dark hair") == \
        "a woman in her thirties with short dark hair"


def test_normalize_legacy_three_sentences_keeps_comma_and_form():
    assert cf.normalize_appearance("Man in his fifties. Grey beard. Short hair.") == \
        "man in his fifties, and grey beard, and short hair"


def test_instruction_one_phrase_rule_and_no_joining_word():
    instr = cf.build_character_instructions()
    assert "appearance is one complete phrase in lowercase" in instr
    assert "No final period. At most 30 words." in instr
    plain = instr.lower().replace("no code fences", "")  # «code fences» — формат ответа, не про код
    for word in ("join", "joined", "joining", "storage", "stored", "code"):
        assert word not in plain, word


def test_add_trait_removed_no_joining_left():
    # _add_trait склеивал черты через «, and »; после отказа от склейки его нет и вызывать его некому
    assert not hasattr(cf, "_add_trait")


# ------------------------------------------------------------------ обновление записей, id, проверка ответа (часть E)

def _c(name, appearance, rid="", aliases=(), keywords=()):
    return {"id": rid, "name": name, "aliases": list(aliases), "keywords": list(keywords), "appearance": appearance}


def _reply(*characters, objects=()):
    return json.dumps({"characters": list(characters), "objects": list(objects)})


def test_schema_has_no_null_no_old_fields_and_string_appearance():
    text = json.dumps(cf.WINDOW_SCHEMA)
    assert "null" not in text
    assert "new_traits" not in text and "conflicts" not in text
    for section in ("characters", "objects"):
        props = cf.WINDOW_SCHEMA["properties"][section]["items"]["properties"]
        assert props["appearance"] == {"type": "string"}
        assert props["id"] == {"type": "string"}


def test_instruction_has_single_phrase_and_update_rule():
    instr = cf.build_character_instructions()
    assert "appearance is one complete phrase in lowercase" in instr
    assert "return its appearance in full, exactly as given" in instr
    assert "Change it only when a new permanent trait appears" in instr


def test_instruction_has_no_old_field_words():
    instr = cf.build_character_instructions()
    for word in ("new_traits", "conflicts", "visible traits", "only permanent visible traits"):
        assert word not in instr, word


def test_window_request_lists_each_known_record_with_id_name_appearance_and_owner():
    chars = [dict(ANNA, id="CHAR_1"), dict(BOB, id="CHAR_2")]
    houses = [{"id": "OBJ_1", "name": "HOUSE", "aliases": [], "keywords": [], "appearance": "a brick house",
               "owner": "CHAR_1"},
              {"id": "OBJ_2", "name": "KEY", "aliases": [], "keywords": [], "appearance": "a brass key",
               "owner": None}]
    user = cf._window_request("Anna and Bob.", chars, houses)
    assert "Known records (each one is separate; do not mix them):" in user
    assert "id: CHAR_1 | name: ANNA | previous appearance: A woman in her thirties with short dark hair." in user
    assert "id: CHAR_2 | name: BOB | previous appearance: A man in his forties with short grey hair." in user
    assert "id: OBJ_1 | name: HOUSE | previous appearance: a brick house | owner: CHAR_1" in user
    assert "id: OBJ_2 | name: KEY | previous appearance: a brass key | owner: none" in user


def test_new_records_with_empty_id_get_char_ids_in_order():
    out = cf.merge_characters([dict(ANNA, id="CHAR_1")],
                              [_c("BOB", "a man in his forties", rid=""), _c("CARL", "a man in his fifties", rid="")])
    assert [c["id"] for c in out] == ["CHAR_1", "CHAR_2", "CHAR_3"]


def test_changed_appearance_of_known_record_updated_through_window(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    monkeypatch.setattr(cf, "WINDOW_SEGMENTS", 1)
    segs = [Segment(1, 0, 1000, "Anna walks in."), Segment(2, 1000, 2000, "Anna sits.")]
    w1 = _reply(_c("ANNA", "a woman in her thirties", rid=""))
    w2 = _reply(_c("ANNA", "a woman in her thirties with a scar on her left cheek", rid="CHAR_1"))
    client = ScriptClient([w1, w2])
    record = cf.build_character_file(segs, client, {})
    assert record["characters"][0]["appearance"] == "a woman in her thirties with a scar on her left cheek"
    assert any(m.startswith("appearance updated: CHAR_1 ANNA: A woman in her thirties -> ") for m in caplog.messages)


def test_one_changed_record_does_not_touch_other_known_record(caplog):
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    chars = [dict(ANNA, id="CHAR_1"), dict(BOB, id="CHAR_2")]
    pending: list = []
    out = cf.merge_characters(chars, [_c("ANNA", "a woman in her thirties with short dark hair and a scar", rid="CHAR_1"),
                                      _c("BOB", BOB["appearance"], rid="CHAR_2")], pending=pending)
    cf._flush_pending(pending)  # в окне строка пишется после принятия; здесь вызов имитирует принятие
    assert out[0]["appearance"] == "a woman in her thirties with short dark hair and a scar"
    assert out[1]["appearance"] == BOB["appearance"]
    assert [m for m in caplog.messages if "appearance updated" in m] == [
        "appearance updated: CHAR_1 ANNA: A woman in her thirties with short dark hair. -> "
        "a woman in her thirties with short dark hair and a scar"]


def test_unknown_id_is_dropped_and_logged(caplog):
    caplog.set_level(logging.WARNING, logger="Generate.character_file")
    segs = [Segment(1, 0, 1000, "Carl enters.")]
    client = ScriptClient([_reply(_c("CARL", "a man in his fifties", rid="CHAR_9"))])
    record = cf.build_character_file(segs, client, {})
    assert record["characters"] == []
    assert len(client.calls) == 1  # выбрасывание не считается ошибкой ответа: повтора нет
    assert "id not in window, dropped: CHAR_9 CARL" in caplog.messages


def test_list_instead_of_appearance_triggers_one_retry_and_valid_second_is_stored():
    segs = [Segment(1, 0, 1000, "Carl enters.")]
    bad = json.dumps({"characters": [{"id": "", "name": "CARL", "aliases": [], "keywords": ["Carl"],
                                      "appearance": ["a man in his fifties"]}]})
    good = _reply(_c("CARL", "a man in his fifties with a grey beard"))
    client = ScriptClient([bad, good])
    record = cf.build_character_file(segs, client, {})
    assert len(client.calls) == 2
    assert record["characters"][0]["appearance"] == "A man in his fifties with a grey beard"


def test_list_appearance_twice_raises_plain_error_and_stores_nothing():
    segs = [Segment(1, 0, 1000, "Carl enters.")]
    bad = json.dumps({"characters": [{"id": "", "name": "CARL", "aliases": [], "keywords": ["Carl"],
                                      "appearance": ["a man in his fifties"]}]})
    client = ScriptClient([bad, bad])
    written = []
    with pytest.raises(cf.CharacterWindowError) as exc:
        cf.build_character_file(segs, client, {}, on_window=written.append)
    assert type(exc.value) is cf.CharacterWindowError  # не EmptyAppearanceError: провал не пустой внешностью
    assert len(client.calls) == 2
    assert written == []


def test_blank_appearance_of_new_record_triggers_one_retry():
    segs = [Segment(1, 0, 1000, "Carl enters.")]
    bad = _reply(_c("CARL", "   ", rid=""))
    good = _reply(_c("CARL", "a man in his fifties with a grey beard", rid=""))
    client = ScriptClient([bad, good])
    record = cf.build_character_file(segs, client, {})
    assert len(client.calls) == 2
    assert record["characters"][0]["appearance"] == "A man in his fifties with a grey beard"


def test_new_record_with_existing_name_triggers_retry_then_known_id_is_stored(monkeypatch):
    monkeypatch.setattr(cf, "WINDOW_SEGMENTS", 1)
    segs = [Segment(1, 0, 1000, "Anna walks in."), Segment(2, 1000, 2000, "Anna sits.")]
    w1 = _reply(_c("ANNA", "a woman in her thirties", rid=""))
    dup = _reply(_c("ANNA", "a woman in her thirties", rid=""))  # имя уже есть, id пустой: ошибка ответа
    good = _reply(_c("ANNA", "a woman in her thirties with short dark hair", rid="CHAR_1"))
    client = ScriptClient([w1, dup, good])
    record = cf.build_character_file(segs, client, {})
    assert len(client.calls) == 3
    assert [c["id"] for c in record["characters"]] == ["CHAR_1"]
    assert record["characters"][0]["appearance"] == "a woman in her thirties with short dark hair"


def test_new_record_with_existing_name_twice_raises_and_stores_nothing(monkeypatch):
    monkeypatch.setattr(cf, "WINDOW_SEGMENTS", 1)
    segs = [Segment(1, 0, 1000, "Anna walks in."), Segment(2, 1000, 2000, "Anna sits.")]
    w1 = _reply(_c("ANNA", "a woman in her thirties", rid=""))
    dup = _reply(_c("ANNA", "a woman in her thirties", rid=""))
    client = ScriptClient([w1, dup, dup])
    written = []
    with pytest.raises(cf.CharacterWindowError) as exc:
        cf.build_character_file(segs, client, {}, on_window=written.append)
    assert type(exc.value) is cf.CharacterWindowError
    assert len(client.calls) == 3
    assert len(written) == 1  # окно 1 записано, окно 2 — нет
    assert [c["name"] for c in written[0]["characters"]] == ["ANNA"]


def test_legacy_record_returned_unchanged_is_not_updated(caplog):
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    legacy = dict(ANNA, id="CHAR_1")  # «A woman in her thirties with short dark hair.» — со старой точкой
    out = cf.merge_characters([legacy], [_c("ANNA", "a woman in her thirties with short dark hair", rid="CHAR_1")])
    assert out[0]["appearance"] == legacy["appearance"]
    assert not any("appearance updated" in m for m in caplog.messages)


def test_legacy_record_returned_with_change_is_updated():
    legacy = {"id": "CHAR_2", "name": "BOB", "aliases": [], "keywords": [], "appearance": "Man in his fifties. Grey beard."}
    out = cf.merge_characters([legacy], [_c("BOB", "a man in his fifties with a grey beard", rid="CHAR_2")])
    assert out[0]["appearance"] == "a man in his fifties with a grey beard"


def test_owner_empty_string_null_without_log_and_unknown_name_null_with_log(caplog):
    caplog.set_level(logging.WARNING, logger="Generate.character_file")
    objects = [_obj("KEY", "", "small brass key", rid=""), _obj("HOUSE", "NOBODY", "brick house", rid="")]
    out = cf.merge_objects([], objects, [dict(ANNA, id="CHAR_1")])
    assert [o["owner"] for o in out] == [None, None]
    assert "owner not found, set to null: HOUSE" in caplog.messages
    assert not any("owner not found, set to null: KEY" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# Проверка дубликатов объектов после слияния персонажей (2a–2e) и события журнала (3)
# ---------------------------------------------------------------------------

def _existing(characters=(), objects=(), segs=()):
    """Частичная запись прошлого запуска с тем же srt_hash, что у сегментов: окна уже не ждут очереди."""
    return {"srt_hash": srt_hash(list(segs)), "characters": list(characters), "objects": list(objects),
            "finished": []}


def test_reproduction_unowned_house_and_new_owner_is_not_duplicate():
    # 2e: известный HOUSE без владельца; ответ: новый персонаж ANNA и HOUSE с владельцем ANNA.
    # Раньше владелец сравнивался по старому списку персонажей (None) и давал ложный дубль.
    segs = [Segment(1, 0, 1000, "Anna walks to the house.")]
    existing = _existing(objects=[_obj("HOUSE", None, "a brick house", rid="OBJ_1")], segs=segs)
    reply = _reply(_c("ANNA", "a woman in her thirties"),
                   objects=[_obj("HOUSE", "ANNA", "a two-storey house with white shutters")])
    record, client = _run_window(segs, [reply], existing)
    assert len(client.calls) == 1  # повтора нет
    assert [(o["id"], o["owner"]) for o in record["objects"]] == [("OBJ_1", None), ("OBJ_2", "CHAR_1")]


def test_two_new_objects_same_name_same_owner_in_one_reply_retry_then_raise():
    # 2b, 2c: два новых HOUSE с владельцем ANNA в одном ответе — дубль; повтор такой же — ошибка окна
    segs = [Segment(1, 0, 1000, "Anna walks to the house.")]
    dup = _reply(_c("ANNA", "a woman in her thirties"),
                 objects=[_obj("HOUSE", "ANNA", "two-storey house"), _obj("HOUSE", "ANNA", "wooden cabin")])
    written = []
    client = ScriptClient([dup, dup])
    with pytest.raises(cf.CharacterWindowError) as exc:
        cf.build_character_file(segs, client, {}, on_window=written.append)
    assert type(exc.value) is cf.CharacterWindowError
    assert len(client.calls) == 2  # один повтор
    assert written == []  # окно не сохранено


def test_two_new_objects_same_name_same_owner_one_retry_then_good_reply_stored():
    segs = [Segment(1, 0, 1000, "Anna walks to the house.")]
    dup = _reply(_c("ANNA", "a woman in her thirties"),
                 objects=[_obj("HOUSE", "ANNA", "two-storey house"), _obj("HOUSE", "ANNA", "wooden cabin")])
    good = _reply(_c("ANNA", "a woman in her thirties"), objects=[_obj("HOUSE", "ANNA", "two-storey house")])
    record, client = _run_window(segs, [dup, good], {})
    assert len(client.calls) == 2
    assert [(o["id"], o["owner"]) for o in record["objects"]] == [("OBJ_1", "CHAR_1")]


def test_new_object_same_name_same_owner_id_as_existing_retry_then_raise():
    # 2b, 2c: новый HOUSE с владельцем ANNA; ANNA известен как CHAR_1, у существующего HOUSE тот же владелец CHAR_1
    segs = [Segment(1, 0, 1000, "Anna sits by the house.")]
    existing = _existing(characters=[dict(ANNA, id="CHAR_1")],
                         objects=[_obj("HOUSE", "CHAR_1", "a brick house", rid="OBJ_1")], segs=segs)
    dup = _reply(_c("ANNA", ANNA["appearance"], rid="CHAR_1"),
                 objects=[_obj("HOUSE", "ANNA", "two-storey house")])
    client = ScriptClient([dup, dup])
    with pytest.raises(cf.CharacterWindowError):
        cf.build_character_file(segs, client, existing)
    assert len(client.calls) == 2


def test_new_object_same_name_same_owner_id_as_existing_one_retry_then_known_id_stored():
    segs = [Segment(1, 0, 1000, "Anna sits by the house.")]
    existing = _existing(characters=[dict(ANNA, id="CHAR_1")],
                         objects=[_obj("HOUSE", "CHAR_1", "a brick house", rid="OBJ_1")], segs=segs)
    dup = _reply(_c("ANNA", ANNA["appearance"], rid="CHAR_1"), objects=[_obj("HOUSE", "ANNA", "two-storey house")])
    good = _reply(_c("ANNA", ANNA["appearance"], rid="CHAR_1"),
                  objects=[_obj("HOUSE", "ANNA", "two-storey house", rid="OBJ_1")])
    record, client = _run_window(segs, [dup, good], existing)
    assert len(client.calls) == 2
    assert [(o["id"], o["owner"]) for o in record["objects"]] == [("OBJ_1", "CHAR_1")]


def test_new_object_same_name_other_owner_id_is_separate_record():
    # 2b: то же имя, но другой владелец по id — не дубль, одна запись на одно окно без повтора
    segs = [Segment(1, 0, 1000, "Anna and Bob at the house.")]
    existing = _existing(characters=[dict(ANNA, id="CHAR_1")],
                         objects=[_obj("HOUSE", "CHAR_1", "a brick house", rid="OBJ_1")], segs=segs)
    reply = _reply(_c("ANNA", ANNA["appearance"], rid="CHAR_1"), _c("BOB", BOB["appearance"]),
                   objects=[_obj("HOUSE", "BOB", "wooden cabin")])
    record, client = _run_window(segs, [reply], existing)
    assert len(client.calls) == 1
    assert [(o["id"], o["owner"]) for o in record["objects"]] == [("OBJ_1", "CHAR_1"), ("OBJ_2", "CHAR_2")]


def test_two_new_characters_same_name_in_one_reply_retry_then_raise():
    # 2b: два новых персонажа с одним именем в одном ответе — дубль (владельца у персонажей нет)
    segs = [Segment(1, 0, 1000, "Anna waves.")]
    dup = _reply(_c("ANNA", "a woman in her thirties"), _c("ANNA", "a woman in her forties"))
    client = ScriptClient([dup, dup])
    with pytest.raises(cf.CharacterWindowError):
        cf.build_character_file(segs, client, {})
    assert len(client.calls) == 2


def _run_window(segs, replies, existing):
    client = ScriptClient(replies)
    record = cf.build_character_file(segs, client, existing)
    return record, client


def test_three_record_events_reach_run_debug_log_file(tmp_path):
    # 3a, 3d: при уровне INFO три события пишутся в <каталог>/run/debug.log с прежним текстом
    prev_level = logging.getLogger().level
    handlers, note = log_setup.setup_logging("INFO", str(tmp_path))
    try:
        assert note is None
        pending: list = []
        cf.merge_characters([dict(ANNA, id="CHAR_1")],
                            [_item("ANNA", "a woman in her forties with grey hair", rid="CHAR_1")], pending=pending)
        cf.merge_objects([], [_obj("HOUSE", "NOBODY", "two-storey brick house")], [dict(ANNA, id="CHAR_1")])
        cf._check_reply({"characters": [_item("CARL", "a man in his fifties", rid="CHAR_9")], "objects": []},
                        [dict(ANNA, id="CHAR_1")], [], pending)
        cf._flush_pending(pending)  # строки пишутся после принятия окна; здесь вызов имитирует принятие
    finally:
        log_setup.teardown_logging(handlers)
        logging.getLogger().setLevel(prev_level)
    lines = (tmp_path / "run" / "debug.log").read_text(encoding="utf-8").splitlines()
    # строка лога: «дата время [УРОВЕНЬ] текст», поэтому ищем текст внутри строки
    def _has(text):
        return any(text in line for line in lines)
    assert _has("[INFO] appearance updated: CHAR_1 ANNA: A woman in her thirties with short dark hair. -> "
                "a woman in her forties with grey hair")
    assert _has("[WARNING] owner not found, set to null: HOUSE")
    assert _has("[WARNING] id not in window, dropped: CHAR_9 CARL")


def test_record_events_are_not_console_key_events(caplog):
    # 3b: события не помечены как ключевые, а INFO-строка «appearance updated» консоль не пропускает
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    pending: list = []
    cf.merge_characters([dict(ANNA, id="CHAR_1")], [_item("ANNA", "a woman in her forties", rid="CHAR_1")], pending=pending)
    cf.merge_objects([], [_obj("HOUSE", "NOBODY", "two-storey brick house")], [dict(ANNA, id="CHAR_1")])
    cf._check_reply({"characters": [_item("CARL", "a man in his fifties", rid="CHAR_9")], "objects": []},
                    [dict(ANNA, id="CHAR_1")], [], pending)
    cf._flush_pending(pending)  # строки пишутся после принятия окна; здесь вызов имитирует принятие
    events = [r for r in caplog.records if r.name == "Generate.character_file"]
    assert len(events) == 3
    for r in events:
        assert not getattr(r, log_setup.KEY_EVENT_ATTR, False)
    info = [r for r in events if r.levelno == logging.INFO]
    assert len(info) == 1 and info[0].getMessage().startswith("appearance updated: ")
    assert log_setup.ConsoleFilter().filter(info[0]) is False


# ---------------------------------------------------------------------------
# Строки «appearance updated» пишутся только за принятое окно (2b)
# ---------------------------------------------------------------------------

_OLD_ANNA_LINE = "appearance updated: CHAR_1 ANNA: A woman in her thirties with short dark hair. -> "


def _appearance_lines(caplog):
    return [m for m in caplog.messages if m.startswith("appearance updated: ")]


def test_accepted_window_writes_appearance_updated_once(caplog):
    # принятое окно меняет внешность известной записи: строка в логе ровно одна
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    segs = [Segment(1, 0, 1000, "Anna walks to the house.")]
    existing = _existing(characters=[dict(ANNA, id="CHAR_1")], segs=segs)
    reply = _reply(_c("ANNA", "a woman in her thirties with a scar", rid="CHAR_1"))
    record, client = _run_window(segs, [reply], existing)
    assert len(client.calls) == 1
    assert record["characters"][0]["appearance"] == "a woman in her thirties with a scar"
    assert _appearance_lines(caplog) == [_OLD_ANNA_LINE + "a woman in her thirties with a scar"]


def test_rejected_first_attempt_logs_only_accepted_attempt_change(caplog):
    # первая попытка меняет внешность ANNA, затем падает на дубле новых объектов (_check_new_objects);
    # вторая попытка меняет её на другое значение. В логе должна остаться только вторая смена.
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    segs = [Segment(1, 0, 1000, "Anna walks to the house.")]
    existing = _existing(characters=[dict(ANNA, id="CHAR_1")], segs=segs)
    dup_objects = [_obj("HOUSE", "ANNA", "two-storey house"), _obj("HOUSE", "ANNA", "wooden cabin")]
    bad = _reply(_c("ANNA", "a woman in her thirties with a scar", rid="CHAR_1"), objects=dup_objects)
    good = _reply(_c("ANNA", "a woman in her thirties with grey eyes", rid="CHAR_1"))
    record, client = _run_window(segs, [bad, good], existing)
    assert len(client.calls) == 2
    assert record["characters"][0]["appearance"] == "a woman in her thirties with grey eyes"
    assert _appearance_lines(caplog) == [_OLD_ANNA_LINE + "a woman in her thirties with grey eyes"]


def test_window_rejected_twice_writes_no_appearance_updated_and_stores_nothing(caplog):
    # оба ответа окна меняют внешность и падают на дубле: строк «appearance updated» нет, окно не сохранено
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    segs = [Segment(1, 0, 1000, "Anna walks to the house.")]
    existing = _existing(characters=[dict(ANNA, id="CHAR_1")], segs=segs)
    dup_objects = [_obj("HOUSE", "ANNA", "two-storey house"), _obj("HOUSE", "ANNA", "wooden cabin")]
    bad = _reply(_c("ANNA", "a woman in her thirties with a scar", rid="CHAR_1"), objects=dup_objects)
    written = []
    client = ScriptClient([bad, bad])
    with pytest.raises(cf.CharacterWindowError):
        cf.build_character_file(segs, client, existing, on_window=written.append)
    assert len(client.calls) == 2
    assert written == []
    assert _appearance_lines(caplog) == []


def test_merge_characters_without_pending_list_writes_no_log_and_same_data(caplog):
    # прямой вызов без списка отложенных строк: строки лога нет, результат тот же, что был раньше
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    new = "a woman in her thirties with a scar"
    chars = [dict(ANNA, id="CHAR_1")]
    extracted = [_item("ANNA", new, rid="CHAR_1")]
    out = cf.merge_characters(chars, extracted)
    assert out == [dict(ANNA, id="CHAR_1", appearance=new)]
    assert not any("appearance updated" in m for m in caplog.messages)
    assert cf.merge_characters(chars, extracted, pending=[]) == out


# ---------------------------------------------------------------------------
# Сводка принятого ответа окна и имя в предупреждении о формате (часть 2, 3)
# ---------------------------------------------------------------------------


def _count_lines(caplog):
    return [m for m in caplog.messages if "в ответе персонажей" in m]


def test_accepted_window_count_line_has_four_numbers(caplog):
    # два персонажа (ANNA новая, BOB известна) и один новый объект: строка сводки с четырьмя числами
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    segs = [Segment(1, 0, 1000, "Anna and Bob walk to the house.")]
    existing = _existing(characters=[dict(BOB, id="CHAR_2")], segs=segs)
    reply = _reply(_c("ANNA", "woman in her thirties"),
                   _c("BOB", "man in his forties", rid="CHAR_2"),
                   objects=[_obj("HOUSE", "ANNA", "a brick house")])
    _run_window(segs, [reply], existing)
    assert _count_lines(caplog) == [
        "Окно сегментов 1..1: в ответе персонажей 2, объектов 1, новых персонажей 1, новых объектов 1."
    ]


def test_reply_without_objects_key_counts_zero_objects(caplog):
    # ответ без ключа objects: в строке сводки объектов 0
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    segs = [Segment(1, 0, 1000, "Anna walks in.")]
    existing = _existing(segs=segs)
    reply = json.dumps({"characters": [_c("ANNA", "woman in her thirties")]})
    _run_window(segs, [reply], existing)
    assert _count_lines(caplog) == [
        "Окно сегментов 1..1: в ответе персонажей 1, объектов 0, новых персонажей 1, новых объектов 0."
    ]


def test_rejected_first_attempt_count_line_is_for_accepted_attempt_only(caplog):
    # первая попытка (1 персонаж, 2 одинаковых новых объекта) отвергнута; вторая принята (1 персонаж, 0 объектов).
    # Строка должна быть одна и с числами принятой попытки, а не отвергнутой.
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    segs = [Segment(1, 0, 1000, "Anna walks to the house.")]
    existing = _existing(characters=[dict(ANNA, id="CHAR_1")], segs=segs)
    dup_objects = [_obj("HOUSE", "ANNA", "two-storey house"), _obj("HOUSE", "ANNA", "wooden cabin")]
    bad = _reply(_c("ANNA", "a woman in her thirties", rid="CHAR_1"), objects=dup_objects)
    good = _reply(_c("ANNA", "a woman in her thirties", rid="CHAR_1"))
    _, client = _run_window(segs, [bad, good], existing)
    assert len(client.calls) == 2
    assert _count_lines(caplog) == [
        "Окно сегментов 1..1: в ответе персонажей 1, объектов 0, новых персонажей 0, новых объектов 0."
    ]


def test_window_rejected_twice_writes_no_count_line(caplog):
    # оба ответа отвергнуты: строки сводки окна нет
    caplog.set_level(logging.INFO, logger="Generate.character_file")
    segs = [Segment(1, 0, 1000, "Anna walks to the house.")]
    existing = _existing(characters=[dict(ANNA, id="CHAR_1")], segs=segs)
    dup_objects = [_obj("HOUSE", "ANNA", "two-storey house"), _obj("HOUSE", "ANNA", "wooden cabin")]
    bad = _reply(_c("ANNA", "a woman in her thirties", rid="CHAR_1"), objects=dup_objects)
    client = ScriptClient([bad, bad])
    with pytest.raises(cf.CharacterWindowError):
        cf.build_character_file(segs, client, existing)
    assert _count_lines(caplog) == []


def test_invalid_name_format_warning_contains_the_name(caplog):
    # имя «Bob Smith!» не проходит формат маркера: предупреждение содержит само имя
    caplog.set_level(logging.WARNING, logger="Generate.character_file")
    cf.parse_window_reply(_reply(_c("Bob Smith!", "a man in his forties")), "Bob Smith! walks.")
    assert "Имя «Bob Smith!» (персонаж) не подходит по формату маркера, запись пропущена." in caplog.messages
