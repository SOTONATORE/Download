"""Офлайн-тесты release_adapter: без сети и без настоящего gh."""
import subprocess
import traceback
from datetime import datetime, timezone

import pytest

try:
    from Generate.core import release_adapter as ra
    from Generate.core.gemini_prompts import GeminiInputError, GeminiRuntimeError
    from Generate.core.srt_parser import Segment, srt_hash
except ImportError:
    from core import release_adapter as ra
    from core.gemini_prompts import GeminiInputError, GeminiRuntimeError
    from core.srt_parser import Segment, srt_hash

SECRET = "ghp_SECRETTOKEN1234567890abcdef"
NOW = datetime(2026, 10, 5, 12, 30, tzinfo=timezone.utc)
SEGS = [Segment(1, 0, 1000, "a"), Segment(2, 1000, 2000, "b")]
FULL = srt_hash(SEGS)
TAG = f"clips-{FULL[:12]}"
RUN = "42"
SERVICE = f"run-{RUN}-service"
FINAL = f"run-{RUN}-final"


class FakeStore(ra.ReleaseStore):
    def __init__(self):
        self.releases = {}   # tag -> list[ReleaseInfo]
        self.uploads = []    # (tag, path)
        self.created = []

    def add(self, tag, body, assets=(), draft=False):
        self.releases.setdefault(tag, []).append(
            ra.ReleaseInfo(tag, "t", body, draft, list(assets)))

    def find_releases(self, tag):
        return list(self.releases.get(tag, []))

    def create_release(self, tag, title, body):
        self.created.append((tag, title, body))
        self.add(tag, body)

    def upload_asset(self, tag, path):
        self.uploads.append((tag, path))


def make(store, limit=900):
    return ra.ReleaseUploader(store, SEGS, "ltx25", release_asset_limit=limit,
                              now=lambda: NOW, log=lambda m: None)


def make_run(store, limit=900):
    return ra.ReleaseUploader(store, SEGS, "ltx25", release_asset_limit=limit,
                              now=lambda: NOW, log=lambda m: None, run_id=RUN)


def body():
    return ra.format_body(FULL, "ltx25")


def test_tag_title_body():
    assert ra.make_tag(SEGS) == TAG
    assert TAG.startswith("clips-") and not TAG.startswith("gen-")
    assert ra.make_tag(SEGS, "my-tag") == "my-tag"
    assert ra.format_title(FULL, NOW) == f"{TAG} · часть 1 · 2026-10-05 12:30 UTC"
    assert ra.format_title(FULL, NOW, 3) == f"{TAG} · часть 3 · 2026-10-05 12:30 UTC"
    assert ra.format_body(FULL, "ltx25") == f"hash: {FULL}\nmodel: ltx25"
    assert len(FULL) == 64


def test_check1_multiple_releases():
    s = FakeStore()
    s.add(TAG, body())
    s.add(TAG, body(), draft=True)
    with pytest.raises(GeminiInputError) as e:
        make(s).prepare()
    assert e.value.exit_code == 2
    assert str(e.value) == f"Найдено несколько релизов с тегом {TAG}, удалите лишние вручную."


@pytest.mark.parametrize("b", ["", "hash: " + "0" * 64, "hash: " + FULL[:12]])
def test_check2_hash_mismatch(b):
    s = FakeStore()
    s.add(TAG, b)
    with pytest.raises(GeminiInputError) as e:
        make(s).prepare()
    assert "Тег совпал, содержимое нет" in str(e.value)


def test_check3_new_release_created():
    s = FakeStore()
    assert make(s).prepare() == set()
    tag, title, b = s.created[0]
    assert tag == TAG and title.startswith(TAG) and b == body()


def test_check3_resume_collects_done_from_all_parts():
    s = FakeStore()
    s.add(TAG, body(), ["1.mp4", "2.mp4", "prompts.json", "generated_links.txt"])
    s.add(TAG + "-part2", body(), ["3.mp4", "x.txt"])
    u = make(s)
    assert u.prepare() == {1, 2, 3}
    assert s.created == []
    assert u.upload_clip("/tmp/2.mp4") is False
    assert u.upload_clip("/tmp/4.mp4") is True
    assert s.uploads == [(TAG + "-part2", "/tmp/4.mp4")]


def test_part_validation_hash():
    s = FakeStore()
    s.add(TAG, body())
    s.add(TAG + "-part2", "hash: other")
    with pytest.raises(GeminiInputError):
        make(s).prepare()


def test_multipart_rollover():
    s = FakeStore()
    u = make(s, limit=2)
    u.prepare()
    for n in range(1, 6):
        u.upload_clip(f"/tmp/{n}.mp4")
    assert [t for t, _ in s.uploads] == [TAG, TAG, TAG + "-part2", TAG + "-part2", TAG + "-part3"]
    assert [c[0] for c in s.created] == [TAG, TAG + "-part2", TAG + "-part3"]
    assert "часть 2" in s.created[1][1]
    assert FULL in s.created[2][2]


def test_service_files_only_main_without_run_id_refused():
    # без run_id служебным файлам негде быть: ошибка, а не тихая загрузка в клип-релиз
    s = FakeStore()
    u = make(s, limit=1)
    u.prepare()
    u.upload_clip("/tmp/1.mp4")
    u.upload_clip("/tmp/2.mp4")
    with pytest.raises(GeminiInputError):
        u.upload_service_file("/tmp/prompts.json")
    with pytest.raises(GeminiInputError):
        u.upload_service_file("/tmp/generated_links.txt")
    assert [t for t, _ in s.uploads] == [TAG, TAG + "-part2"]
    with pytest.raises(GeminiInputError):
        u.upload_service_file("/tmp/3.mp4")


def test_upload_before_prepare_and_bad_name():
    u = make(FakeStore())
    with pytest.raises(GeminiInputError):
        u.upload_clip("/tmp/1.mp4")
    u.prepare()
    with pytest.raises(GeminiInputError):
        u.upload_clip("/tmp/notes.txt")


def test_generated_links_format():
    txt = ra.create_generated_links("o/r", "gen-abc",
                                    ["1.mp4", "prompts.json", "2.mp4", "generated_links.txt"],
                                    {"2.mp4": "gen-abc-part2"})
    assert txt == ("https://github.com/o/r/releases/download/gen-abc/1.mp4\n"
                   "https://github.com/o/r/releases/download/gen-abc-part2/2.mp4\n")
    assert ra.create_generated_links("o/r", "t", ["prompts.json"]) == ""


def test_links_text_sorted_with_parts():
    s = FakeStore()
    u = make(s, limit=1)
    u.prepare()
    u.upload_clip("/tmp/10.mp4")
    u.upload_clip("/tmp/2.mp4")
    lines = u.links_text("o/r").splitlines()
    assert lines[0].endswith(f"{TAG}-part2/2.mp4") and lines[1].endswith(f"{TAG}/10.mp4")


def test_no_clips_empty_links_file_not_uploaded(tmp_path):
    s = FakeStore()
    logs = []
    u = ra.ReleaseUploader(s, SEGS, "ltx25", release_asset_limit=900, now=lambda: NOW,
                           log=logs.append)
    u.prepare()
    logs.clear()                                              # строка «Создан релиз» не относится к делу
    links = tmp_path / "generated_links.txt"
    links.write_text(u.links_text("o/r"), encoding="utf-8")   # клипов нет: текст пустой
    assert links.stat().st_size == 0
    u.upload_service_file(str(links))
    assert s.uploads == []
    assert logs == ["Нет готовых клипов: generated_links.txt не выкладывается."]


def test_links_file_with_clips_is_uploaded_to_final(tmp_path):
    s = FakeStore()
    s.add(TAG, body(), ["1.mp4"])
    u = make_run(s)
    u.prepare()
    links = tmp_path / "generated_links.txt"
    links.write_text(u.links_text("o/r"), encoding="utf-8")
    assert links.stat().st_size > 0
    u.upload_service_file(str(links))
    assert s.uploads == [(FINAL, str(links))]
    assert not any(t == TAG for t, p in s.uploads if p.endswith("generated_links.txt"))


def test_empty_prompts_file_goes_to_service_release(tmp_path):
    s = FakeStore()
    logs = []
    u = ra.ReleaseUploader(s, SEGS, "ltx25", release_asset_limit=900, now=lambda: NOW,
                           log=logs.append, run_id=RUN)
    u.prepare()
    logs.clear()
    prompts = tmp_path / "prompts.json"
    prompts.write_text("", encoding="utf-8")
    u.upload_service_file(str(prompts))
    assert s.uploads == [(SERVICE, str(prompts))] and logs == []


# --- gh и безопасность токена -------------------------------------------------

class FakeProc:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_runner_token_in_env_not_args():
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return FakeProc(out="ok")

    r = ra.GhCommandRunner(environ={"GITHUB_TOKEN": SECRET, "PATH": "/bin"}, run=run)
    assert r(["release", "list"]) == "ok"
    cmd, kw = calls[0]
    assert SECRET not in " ".join(cmd)
    assert kw["env"]["GH_TOKEN"] == SECRET


def test_runner_failure_scrubs_token(capsys):
    def run(cmd, **kw):
        return FakeProc(rc=1, err=f"auth failed for {SECRET} and ghp_ABCDEFGHIJKLMNOP123")

    r = ra.GhCommandRunner(environ={"GH_TOKEN": SECRET}, run=run)
    with pytest.raises(GeminiRuntimeError) as e:
        r(["release", "create", "x"])
    msg = str(e.value)
    assert e.value.exit_code == 3
    assert SECRET not in msg and "ghp_ABCDEF" not in msg and "***" in msg
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err


def test_runner_missing_token_and_gh():
    with pytest.raises(GeminiInputError) as e:
        ra.GhCommandRunner(environ={})(["x"])
    assert SECRET not in str(e.value)

    def run(cmd, **kw):
        raise FileNotFoundError()

    with pytest.raises(GeminiRuntimeError):
        ra.GhCommandRunner(environ={"GH_TOKEN": SECRET}, run=run)(["x"])

    def run2(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    with pytest.raises(GeminiRuntimeError) as e2:
        ra.GhCommandRunner(environ={"GH_TOKEN": SECRET}, run=run2)(["x"])
    assert SECRET not in str(e2.value)


def test_gh_store_find_create_upload():
    import json
    calls = []
    rows = [{"tag_name": TAG, "name": "n", "body": body(), "draft": True, "assets": ["1.mp4"]},
            {"tag_name": "other", "name": "o", "body": "", "draft": False, "assets": []}]

    def runner(args):
        calls.append(list(args))
        if args[0] == "api":
            return "\n".join(json.dumps(r) for r in rows) + "\n"
        return ""

    st = ra.GhReleaseStore("o/r", runner)
    found = st.find_releases(TAG)
    assert len(found) == 1 and found[0].is_draft and found[0].assets == ["1.mp4"]
    st.create_release("t", "Title", "B")
    st.upload_asset("t", "/tmp/1.mp4")
    assert calls[1][:3] == ["release", "create", "t"] and "--repo" in calls[1]
    assert calls[2][:4] == ["release", "upload", "t", "/tmp/1.mp4"]


def test_gh_store_bad_json():
    st = ra.GhReleaseStore("o/r", lambda a: "not json\n")
    with pytest.raises(GeminiRuntimeError):
        st.find_releases("t")


# --- три типа релизов (SPEC 9.1) -------------------------------------------

def test_three_release_types_and_tags():
    assert ra.make_tag(SEGS) == f"clips-{FULL[:12]}"
    assert ra.part_tag(TAG, 2) == TAG + "-part2"
    assert ra.service_tag(RUN) == SERVICE == "run-42-service"
    assert ra.final_tag(RUN) == FINAL == "run-42-final"
    assert len({TAG, SERVICE, FINAL}) == 3


@pytest.mark.parametrize("bad", ["", "a/b", "x y", "a\\b", "../x", "с-кириллицей", "a" * 65])
def test_run_id_validation(bad):
    with pytest.raises(GeminiInputError):
        ra.service_tag(bad)
    with pytest.raises(GeminiInputError):
        ra.ReleaseUploader(FakeStore(), SEGS, "ltx25", run_id=bad)


def test_run_titles_and_bodies_have_no_hash():
    t = ra.format_run_title("служебный", RUN, NOW)
    assert t == "run-42 · служебный · 2026-10-05 12:30 UTC"
    b = ra.format_run_body("service", RUN)
    assert b == "run: 42\nkind: service" and FULL not in b


def test_no_run_id_creates_only_clip_release():
    s = FakeStore()
    make(s).prepare()
    assert [c[0] for c in s.created] == [TAG]


def test_run_id_creates_service_at_start_not_final():
    s = FakeStore()
    make_run(s).prepare()
    assert [c[0] for c in s.created] == [TAG, SERVICE]
    assert FINAL not in s.releases


# --- проверки 9.2 только для клип-релизов ----------------------------------

def test_service_and_final_not_hash_checked():
    s = FakeStore()
    s.add(SERVICE, "совсем другой текст")
    s.add(FINAL, "")
    u = make_run(s)
    assert u.prepare() == set()          # служебный с чужим описанием не останавливает запуск
    assert [c[0] for c in s.created] == [TAG]   # существующий служебный переиспользован
    assert u.open_final_release() == FINAL       # итоговый с пустым описанием тоже


def test_clip_hash_check_still_applies_with_run_id():
    s = FakeStore()
    s.add(TAG, "hash: other")
    with pytest.raises(GeminiInputError) as e:
        make_run(s).prepare()
    assert "Тег совпал, содержимое нет" in str(e.value)


def test_service_release_duplicates_refused():
    s = FakeStore()
    s.add(SERVICE, "x")
    s.add(SERVICE, "y")
    with pytest.raises(GeminiInputError) as e:
        make_run(s).prepare()
    assert str(e.value) == f"Найдено несколько релизов с тегом {SERVICE}, удалите лишние вручную."


# --- маршрутизация файлов и финальный релиз ---------------------------------

def test_prompts_and_missing_to_service_links_to_final(tmp_path):
    s = FakeStore()
    u = make_run(s)
    u.prepare()
    u.upload_clip("/tmp/1.mp4")
    prompts = tmp_path / "prompts.json"
    prompts.write_text("{}", encoding="utf-8")
    missing = tmp_path / "missing.txt"
    missing.write_text("1\n", encoding="utf-8")
    u.upload_service_file(str(prompts))
    u.upload_run_file(str(missing))
    links = tmp_path / "generated_links.txt"
    links.write_text(u.links_text("o/r"), encoding="utf-8")
    u.upload_service_file(str(links))
    assert s.uploads == [(TAG, "/tmp/1.mp4"), (SERVICE, str(prompts)),
                         (SERVICE, str(missing)), (FINAL, str(links))]
    assert [c[0] for c in s.created] == [TAG, SERVICE, FINAL]


def test_final_created_once_even_if_links_uploaded_twice(tmp_path):
    s = FakeStore()
    u = make_run(s)
    u.prepare()
    u.upload_clip("/tmp/1.mp4")
    links = tmp_path / "generated_links.txt"
    links.write_text(u.links_text("o/r"), encoding="utf-8")
    u.upload_service_file(str(links))
    u.upload_service_file(str(links))
    assert [c[0] for c in s.created].count(FINAL) == 1
    assert [t for t, _ in s.uploads].count(FINAL) == 2


def test_run_files_refused_when_wrong_kind(tmp_path):
    u = make_run(FakeStore())
    u.prepare()
    with pytest.raises(GeminiInputError):
        u.upload_run_file("/tmp/3.mp4")                   # клип не в служебный релиз
    with pytest.raises(GeminiInputError):
        u.upload_run_file("/tmp/generated_links.txt")     # ссылки не в служебный релиз
    with pytest.raises(GeminiInputError):
        u.upload_service_file("/tmp/notes.txt")           # не служебный файл


def test_run_files_need_run_id(tmp_path):
    u = make(FakeStore())
    u.prepare()
    with pytest.raises(GeminiInputError):
        u.upload_run_file(str(tmp_path / "missing.txt"))
    with pytest.raises(GeminiInputError):
        u.open_final_release()


def test_service_files_do_not_count_against_clip_limit(tmp_path):
    s = FakeStore()
    u = make_run(s, limit=2)
    u.prepare()
    for name in ("prompts.json", "missing.txt", "generated_links.txt"):
        f = tmp_path / name
        f.write_text("1\n" if name != "prompts.json" else "{}", encoding="utf-8")
        if name == "missing.txt":
            u.upload_run_file(str(f))
        else:
            u.upload_service_file(str(f))
    u.upload_clip("/tmp/1.mp4")
    u.upload_clip("/tmp/2.mp4")
    u.upload_clip("/tmp/3.mp4")
    assert [t for t, _ in s.uploads if t.startswith("clips-")] == [TAG, TAG, TAG + "-part2"]


def test_part_switch_at_limit_with_run_id():
    s = FakeStore()
    s.add(TAG, body(), ["1.mp4", "2.mp4"])
    u = make_run(s, limit=2)
    assert u.prepare() == {1, 2}
    u.upload_clip("/tmp/3.mp4")
    assert [c[0] for c in s.created] == [SERVICE, TAG + "-part2"]
    assert s.uploads == [(TAG + "-part2", "/tmp/3.mp4")]


def test_links_built_from_all_clip_parts_and_uploaded_to_final(tmp_path):
    s = FakeStore()
    s.add(TAG, body(), ["1.mp4", "2.mp4", "prompts.json"])
    s.add(TAG + "-part2", body(), ["3.mp4"])
    u = make_run(s, limit=2)
    u.prepare()
    u.upload_clip("/tmp/4.mp4")
    links = tmp_path / "generated_links.txt"
    links.write_text(u.links_text("o/r"), encoding="utf-8")
    u.upload_service_file(str(links))
    lines = links.read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"https://github.com/o/r/releases/download/{TAG}/1.mp4",
        f"https://github.com/o/r/releases/download/{TAG}/2.mp4",
        f"https://github.com/o/r/releases/download/{TAG}-part2/3.mp4",
        f"https://github.com/o/r/releases/download/{TAG}-part2/4.mp4",
    ]
    assert s.uploads[-1] == (FINAL, str(links))


def test_empty_clips_with_run_id_no_links_no_final(tmp_path):
    s = FakeStore()
    u = make_run(s)
    u.prepare()
    links = tmp_path / "generated_links.txt"
    links.write_text(u.links_text("o/r"), encoding="utf-8")
    assert links.stat().st_size == 0
    u.upload_service_file(str(links))
    assert s.uploads == [] and FINAL not in s.releases


# --- канарейка: токен не попадает в вывод, логи и исключения -----------------

CANARY = "TESTKEY-12345"


def test_canary_normal_path_not_in_logs_or_result(capsys):
    logs = []

    import json

    row = json.dumps({"tag_name": "other", "name": CANARY, "body": f"token {CANARY}",
                      "draft": False, "assets": []})

    def run(cmd, **kw):
        return FakeProc(out=row + "\n")       # gh отдаёт токен внутри описания релиза

    r = ra.GhCommandRunner(environ={"GH_TOKEN": CANARY}, run=run)
    store = ra.GhReleaseStore("o/r", r)
    u = ra.ReleaseUploader(store, SEGS, "ltx25", now=lambda: NOW, log=logs.append,
                           run_id=RUN)
    assert CANARY not in r(["release", "list"]) and "***" in r(["release", "list"])
    u.prepare()
    out = capsys.readouterr()
    assert CANARY not in out.out + out.err
    assert all(CANARY not in m for m in logs)


def test_canary_error_path_not_in_exception_or_traceback(capsys):
    def run(cmd, **kw):
        return FakeProc(rc=1, err=f"HTTP 401 for token {CANARY}")

    logs = []
    store = ra.GhReleaseStore("o/r", ra.GhCommandRunner(environ={"GH_TOKEN": CANARY}, run=run))
    u = ra.ReleaseUploader(store, SEGS, "ltx25", now=lambda: NOW, log=logs.append, run_id=RUN)
    with pytest.raises(GeminiRuntimeError) as e:
        u.prepare()
    text = "".join(traceback.format_exception(e.value))
    assert CANARY not in text and CANARY not in str(e.value) and "***" in str(e.value)
    out = capsys.readouterr()
    assert CANARY not in out.out + out.err
    assert all(CANARY not in m for m in logs)


def test_canary_exception_inside_runner_not_leaked():
    def run(cmd, **kw):
        raise RuntimeError(f"boom {CANARY}")

    r = ra.GhCommandRunner(environ={"GH_TOKEN": CANARY}, run=run)
    with pytest.raises(GeminiRuntimeError) as e:
        r(["release", "list"])
    text = "".join(traceback.format_exception(e.value))
    assert CANARY not in text and CANARY not in repr(e.value)
    assert e.value.__cause__ is None


GEN_VAST = "GEN_" + "VAST_API_KEY"
GEN_GEMINI = "GEN_" + "GEMINI_API_KEY"
HF_TOKEN_NAME = "HF_" + "TOKEN"
PROJECT_SECRET_NAMES = (GEN_VAST, GEN_GEMINI, HF_TOKEN_NAME, "WORKER_TOKEN", "GITHUB_TOKEN")


def test_gh_env_names_are_exactly_the_constant_and_no_project_secret_name():
    """9а п. 6: набор имён окружения, которые получает gh, равен GH_ENV_NAMES; имён секретов проекта в нём нет."""
    assert set(ra.GH_ENV_NAMES) == {"PATH", "HOME", "GH_TOKEN"}
    for name in PROJECT_SECRET_NAMES:
        assert name not in ra.GH_ENV_NAMES


def test_gh_process_gets_exactly_gh_env_names_and_no_project_secrets():
    """9а п. 6: процессу gh передаётся ровно набор GH_ENV_NAMES; ключи проекта и лишние имена не попадают."""
    calls = []

    def run(cmd, **kw):
        calls.append(kw)
        return FakeProc(out="ok")

    environ = {
        "PATH": "/bin", "HOME": "/home/runner", "GH_TOKEN": SECRET, "GITHUB_TOKEN": "ghs-other-XYZ-000",
        GEN_VAST: "vast-canary-111", GEN_GEMINI: "gemini-canary-222",
        HF_TOKEN_NAME: "hf-canary-333", "WORKER_TOKEN": "worker-canary-444", "LANG": "C.UTF-8",
    }
    ra.GhCommandRunner(environ=environ, run=run)(["release", "list"])
    env = calls[0]["env"]
    assert set(env) == set(ra.GH_ENV_NAMES)               # точный набор имён, без лишних
    assert env["GH_TOKEN"] == SECRET
    for name in PROJECT_SECRET_NAMES:
        assert name not in env
    for value in ("vast-canary-111", "gemini-canary-222", "hf-canary-333", "worker-canary-444", "ghs-other-XYZ-000"):
        assert value not in env.values()
