"""Офлайн-тесты release_adapter: без сети и без настоящего gh.

Стадия 6: выдача генерации - ФИНАЛ · ГЕНЕРАЦИЯ (final-gen-<hash16>) и части ГЕНЕРАЦИЯ · ЧАСТЬ N
(gen-part<N>-<hash16>). Служебного релиза run-<id>-service больше нет."""
import subprocess
import traceback
from datetime import datetime, timezone

import pytest

try:
    from Generate.core import release_adapter as ra
    from Generate.core import release_names as rn
    from Generate.core.gemini_prompts import GeminiInputError, GeminiRuntimeError
    from Generate.core.srt_parser import Segment, srt_hash
except ImportError:
    from core import release_adapter as ra
    from core import release_names as rn
    from core.gemini_prompts import GeminiInputError, GeminiRuntimeError
    from core.srt_parser import Segment, srt_hash

SECRET = "ghp_SECRETTOKEN1234567890abcdef"
NOW = datetime(2026, 10, 5, 12, 30, tzinfo=timezone.utc)
SEGS = [Segment(1, 0, 1000, "a"), Segment(2, 1000, 2000, "b")]
FULL = srt_hash(SEGS)
H16 = FULL[:16]
FINAL = f"final-gen-{H16}"
PART1 = f"gen-part1-{H16}"
PART2 = f"gen-part2-{H16}"
PART3 = f"gen-part3-{H16}"
TAG = PART1   # тег для блока gh-хранилища ниже: там важен только сам тег строкой


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


def make(store, limit=900, repo=""):
    return ra.ReleaseUploader(store, SEGS, "ltx25", release_asset_limit=limit,
                              now=lambda: NOW, log=lambda m: None, repo=repo)


def body():
    return rn.release_body("gen", "final", FULL)


def test_names_six_kinds_tags_and_titles():
    # финалы и части: теги, заголовки; номер части - только у частей
    assert rn.final_tag("stock", FULL) == f"final-stock-{H16}"
    assert rn.final_tag("gen", FULL) == FINAL
    assert rn.part_tag("stock", 2, FULL) == f"stock-part2-{H16}"
    assert rn.part_tag("gen", 1, FULL) == PART1
    assert rn.final_title("gen", FULL, NOW) == f"ФИНАЛ · ГЕНЕРАЦИЯ · 2026-10-05 12:30 UTC · {H16}"
    assert rn.part_title("gen", 3, FULL, NOW) == f"ГЕНЕРАЦИЯ · ЧАСТЬ 3 · 2026-10-05 12:30 UTC · {H16}"
    assert "ЧАСТЬ" not in rn.final_title("gen", FULL, NOW)
    assert rn.release_body("gen", "final", FULL) == f"hash: {FULL}\nstage: gen\nrole: final"


def test_check1_multiple_releases():
    s = FakeStore()
    s.add(FINAL, body())
    s.add(FINAL, body(), draft=True)
    with pytest.raises(GeminiInputError) as e:
        make(s).prepare()
    assert e.value.exit_code == 2
    assert str(e.value) == f"Найдено несколько релизов с тегом {FINAL}, удалите лишние вручную."


@pytest.mark.parametrize("b", ["", "hash: " + "0" * 64, "hash: " + FULL[:12]])
def test_check2_hash_mismatch(b):
    s = FakeStore()
    s.add(FINAL, b)
    with pytest.raises(GeminiInputError) as e:
        make(s).prepare()
    assert "Тег совпал, содержимое нет" in str(e.value)


def test_check3_new_final_created_no_parts():
    s = FakeStore()
    assert make(s).prepare() == set()
    tag, title, b = s.created[0]
    assert tag == FINAL and title == rn.final_title("gen", FULL, NOW) and b == body()
    assert [c[0] for c in s.created] == [FINAL]


def test_check3_resume_collects_done_from_all_parts():
    s = FakeStore()
    s.add(FINAL, body(), ["prompts.json", "generated_links.txt", "missing.txt"])
    s.add(PART1, body(), ["1.mp4", "2.mp4"])
    s.add(PART2, body(), ["3.mp4", "x.txt"])
    u = make(s)
    assert u.prepare() == {1, 2, 3}
    assert s.created == []
    assert u.upload_clip("/tmp/2.mp4") is False
    assert u.upload_clip("/tmp/4.mp4") is True
    assert s.uploads == [(PART2, "/tmp/4.mp4")]


def test_part_validation_hash():
    s = FakeStore()
    s.add(FINAL, body())
    s.add(PART1, "hash: other")
    with pytest.raises(GeminiInputError):
        make(s).prepare()


def test_multipart_rollover():
    s = FakeStore()
    u = make(s, limit=2)
    u.prepare()
    for n in range(1, 6):
        u.upload_clip(f"/tmp/{n}.mp4")
    assert [t for t, _ in s.uploads] == [PART1, PART1, PART2, PART2, PART3]
    assert [c[0] for c in s.created] == [FINAL, PART1, PART2, PART3]
    assert "ЧАСТЬ 2" in s.created[2][1]
    assert FULL in s.created[3][2]


def test_final_takes_text_and_previews_without_limit():
    # ФИНАЛ · ГЕНЕРАЦИЯ: служебные файлы и превью; превью не расходуют лимит частей
    s = FakeStore()
    u = make(s, limit=1)
    u.prepare()
    u.upload_service_file("/tmp/prompts.json")
    u.upload_run_file("/tmp/missing.txt")
    for n in (1, 2, 3):
        u.upload_preview(f"/tmp/preview_{n}_1.png")
    assert [t for t, _ in s.uploads] == [FINAL] * 5
    assert [c[0] for c in s.created] == [FINAL]


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
    assert lines[0].endswith(f"{PART2}/2.mp4") and lines[1].endswith(f"{PART1}/10.mp4")


def test_links_refreshed_in_final_after_each_clip():
    # стадия 6: generated_links.txt в финале обновляется после каждой загрузки клипа
    s = FakeStore()
    u = make(s, repo="o/r")
    u.prepare()
    u.upload_clip("/tmp/1.mp4")
    u.upload_clip("/tmp/2.mp4")
    assert [t for t, _ in s.uploads] == [PART1, FINAL, PART1, FINAL]
    assert s.uploads[1][1].endswith("generated_links.txt")


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
    s.add(PART1, body(), ["1.mp4"])
    u = make(s, repo="o/r")
    u.prepare()
    links = tmp_path / "generated_links.txt"
    links.write_text(u.links_text("o/r"), encoding="utf-8")
    assert links.stat().st_size > 0
    u.upload_service_file(str(links))
    assert s.uploads == [(FINAL, str(links))]
    assert not any(t == PART1 for t, p in s.uploads if p.endswith("generated_links.txt"))


def test_empty_prompts_file_goes_to_final(tmp_path):
    s = FakeStore()
    logs = []
    u = ra.ReleaseUploader(s, SEGS, "ltx25", release_asset_limit=900, now=lambda: NOW,
                           log=logs.append)
    u.prepare()
    logs.clear()
    prompts = tmp_path / "prompts.json"
    prompts.write_text("", encoding="utf-8")
    u.upload_service_file(str(prompts))
    assert s.uploads == [(FINAL, str(prompts))] and logs == []


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


def test_final_created_once_even_if_links_uploaded_twice(tmp_path):
    s = FakeStore()
    s.add(PART1, body(), ["1.mp4"])
    u = make(s, repo="o/r")
    u.prepare()
    links = tmp_path / "generated_links.txt"
    links.write_text(u.links_text("o/r"), encoding="utf-8")
    u.upload_service_file(str(links))
    u.upload_service_file(str(links))
    assert [c[0] for c in s.created] == [FINAL]                 # финал создан один раз, при prepare
    assert [t for t, _ in s.uploads] == [FINAL, FINAL]


def test_run_files_refused_when_wrong_kind(tmp_path):
    s = FakeStore()
    u = make(s)
    u.prepare()
    with pytest.raises(GeminiInputError):
        u.upload_run_file("/tmp/3.mp4")                   # клип не в финал через текстовый путь
    with pytest.raises(GeminiInputError):
        u.upload_run_file("/tmp/generated_links.txt")     # ссылки ведутся отдельно
    with pytest.raises(GeminiInputError):
        u.upload_service_file("/tmp/notes.txt")           # не служебный файл


def test_service_files_do_not_count_against_clip_limit(tmp_path):
    s = FakeStore()
    u = make(s, limit=1)
    u.prepare()
    for name in ("prompts.json", "missing.txt"):
        f = tmp_path / name
        f.write_text("x", encoding="utf-8")
        if name == "prompts.json":
            u.upload_service_file(str(f))
        else:
            u.upload_run_file(str(f))
    u.upload_clip("/tmp/1.mp4")
    u.upload_clip("/tmp/2.mp4")
    assert [t for t, _ in s.uploads if t.startswith("gen-part")] == [PART1, PART2]


def test_links_built_from_all_clip_parts_and_uploaded_to_final(tmp_path):
    s = FakeStore()
    s.add(PART1, body(), ["1.mp4", "2.mp4"])
    s.add(PART2, body(), ["3.mp4"])
    u = make(s, repo="o/r")
    u.prepare()
    links = tmp_path / "generated_links.txt"
    links.write_text(u.links_text("o/r"), encoding="utf-8")
    u.upload_service_file(str(links))
    assert links.read_text(encoding="utf-8").splitlines() == [
        f"https://github.com/o/r/releases/download/{PART1}/1.mp4",
        f"https://github.com/o/r/releases/download/{PART1}/2.mp4",
        f"https://github.com/o/r/releases/download/{PART2}/3.mp4",
    ]
    assert s.uploads == [(FINAL, str(links))]


def test_canary_normal_path_not_in_logs_or_result(capsys):
    logs = []

    import json

    row = json.dumps({"tag_name": "other", "name": CANARY, "body": f"token {CANARY}",
                      "draft": False, "assets": []})

    def run(cmd, **kw):
        return FakeProc(out=row + "\n")       # gh отдаёт токен внутри описания релиза

    r = ra.GhCommandRunner(environ={"GH_TOKEN": CANARY}, run=run)
    store = ra.GhReleaseStore("o/r", r)
    u = ra.ReleaseUploader(store, SEGS, "ltx25", now=lambda: NOW, log=logs.append)
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
    u = ra.ReleaseUploader(store, SEGS, "ltx25", now=lambda: NOW, log=logs.append)
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


# --- фото (STEP 7, блок B2): имена файлов фото распознаются наряду с клипами ---------------------
def test_clip_num_any_accepts_photo_and_video_names():
    assert ra._clip_num_any("12.png", "{num}.mp4") == 12
    assert ra._clip_num_any("12.mp4", "{num}.mp4") == 12
    assert ra._clip_num_any("prompts.json", "{num}.mp4") is None
    assert ra.PHOTO_FILENAME_TEMPLATE == "{num}.png"
