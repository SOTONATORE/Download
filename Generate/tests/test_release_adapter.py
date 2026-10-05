"""Офлайн-тесты release_adapter: без сети и без настоящего gh."""
import subprocess
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
TAG = f"gen-{FULL[:12]}"


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


def body():
    return ra.format_body(FULL, "ltx25")


def test_tag_title_body():
    assert ra.make_tag(SEGS) == TAG
    assert ra.make_tag(SEGS, "my-tag") == "my-tag"
    assert ra.format_title(FULL, NOW) == f"{TAG} · 2026-10-05 12:30 UTC"
    assert ra.format_title(FULL, NOW, 3).endswith("UTC · часть 3")
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
    assert s.created[1][1].endswith("часть 2")
    assert FULL in s.created[2][2]


def test_service_files_only_main():
    s = FakeStore()
    u = make(s, limit=1)
    u.prepare()
    u.upload_clip("/tmp/1.mp4")      # основной релиз заполнен
    u.upload_clip("/tmp/2.mp4")      # уходит в part2
    u.upload_service_file("/tmp/prompts.json")
    u.upload_service_file("/tmp/generated_links.txt")
    assert s.uploads[-2:] == [(TAG, "/tmp/prompts.json"), (TAG, "/tmp/generated_links.txt")]
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
