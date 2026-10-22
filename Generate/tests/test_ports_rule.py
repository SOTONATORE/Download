"""Офлайн-тесты правила портов (SPEC 0.1, новое правило портов).

Внутренние порты процессов карты (8000 воркер, 18188 ComfyUI) видны в журнале и артефактах, если стоят
после адреса локального прослушивания. Внешние адреса Vast (публичный IP, HostPort, ssh-адрес и порт)
маскируются везде. Диагностика карт (ssh_diag.log, vast_logs.txt, worker_log.txt, instance.json) в релиз
не попадает; в каталоге артефакта она остаётся.
Настоящего Vast, сети и ssh нет. Адреса взяты из документационных диапазонов, ключ заведомо фальшивый.
"""
import json
import zipfile

import pytest

try:
    from Generate.core import log_setup as ls
    from Generate.core import orchestrator as orch
    from Generate.core import release_adapter as ra
    from Generate.core.gemini_prompts import GeminiInputError
    from Generate.core.srt_parser import Segment
except ImportError:  # pragma: no cover
    from core import log_setup as ls  # type: ignore
    from core import orchestrator as orch  # type: ignore
    from core import release_adapter as ra  # type: ignore
    from core.gemini_prompts import GeminiInputError  # type: ignore
    from core.srt_parser import Segment  # type: ignore

EXT_IP = "198.51.100.44"     # внешний адрес карты (документационный диапазон)
EXT_PORT = "47123"           # внешний HostPort
SSH_HOST = "203.0.113.7"     # ssh-адрес карты
SSH_PORT = 39871             # ssh-порт карты
INSTKEY = "INSTKEY-98765"    # ключ экземпляра (заглушка)
ENV_TOKEN = "WTOKEN-VALUE-24680"  # значение из env карты (заглушка)


def _slot():
    slot = orch._CardSlot(index=1, generation=1, instance_id=77, machine_id=5, price=0.1, created=0.0)
    slot.state = "booting"
    return slot


# ---------------------------------------------------------------------------
# Маска: порты 8000 и 18188 у адреса локального прослушивания
# ---------------------------------------------------------------------------

def test_listener_output_keeps_internal_ports_and_masks_other_ports():
    text = ("State Recv-Q Local Address:Port\n"
            "LISTEN 0 4096 0.0.0.0:8000 0.0.0.0:*\n"
            "LISTEN 0 4096 127.0.0.1:18188 0.0.0.0:*\n"
            "LISTEN 0 4096 0.0.0.0:41234 0.0.0.0:*\n")
    out = ls.mask_fetched_text(text, keep_internal_ports=True)
    assert "0.0.0.0:8000" in out
    assert "127.0.0.1:18188" in out
    assert "41234" not in out


def test_plain_internal_port_numbers_are_not_masked():
    out = ls.mask_fetched_text("worker port 8000, comfy port 18188", keep_internal_ports=True)
    assert "port 8000" in out and "port 18188" in out


def test_default_mode_is_unchanged_and_masks_internal_ports_too():
    # без опции поведение прежнее: адрес с портом маскируется, как для всех остальных вызывающих
    out = ls.mask_fetched_text("LISTEN 0.0.0.0:8000")
    assert "0.0.0.0:8000" not in out


def test_internal_port_with_external_ip_is_still_masked():
    out = ls.mask_fetched_text(f"{SSH_HOST}:8000 ok", keep_internal_ports=True)
    assert SSH_HOST not in out and "8000" not in out


# ---------------------------------------------------------------------------
# mask_external: литеральные внешние значения
# ---------------------------------------------------------------------------

def test_mask_external_replaces_literal_ip_and_port_only_as_whole_values():
    text = "ip 198.51.100.44 port 47123 and 147123 and 198.51.100.440"
    out = ls.mask_external(text, [EXT_IP, int(EXT_PORT)])
    assert out == "ip *** port *** and 147123 and 198.51.100.440"


def test_mask_external_ignores_empty_and_none_values():
    assert ls.mask_external("plain text 8000", [None, "", "   "]) == "plain text 8000"


# ---------------------------------------------------------------------------
# instance.json: статус и внутренние порты, остальное удалено
# ---------------------------------------------------------------------------

def test_instance_json_keeps_status_and_internal_ports_only(tmp_path):
    raw = {
        "id": 77,
        "label": "gen-test",
        "status": "running",
        "actual_status": "running",
        "intended_status": "running",
        "status_msg": "ok",
        "public_ipaddr": EXT_IP,
        "ssh_host": SSH_HOST,
        "ssh_port": SSH_PORT,
        "direct_port_start": 47100,
        "direct_port_end": 47110,
        "direct_port_count": 2,
        "env": {"WORKER_TOKEN": ENV_TOKEN},
        "extra_env": [["WORKER_TOKEN", ENV_TOKEN]],
        "instance_api_key": INSTKEY,
        "ports": {
            "8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": EXT_PORT}],
            "18188/tcp": [{"HostIp": "0.0.0.0", "HostPort": "31888"}],
            "22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "50022"}],
        },
    }
    art = orch.RunArtifact(tmp_path / "art", lambda text: text)
    art.card_instance(_slot(), raw)
    text = (tmp_path / "art" / "cards" / "slot1_generation1" / "instance.json").read_text(encoding="utf-8")
    data = json.loads(text)

    for key in ("status", "actual_status", "intended_status", "status_msg"):
        assert key in data
    assert data["ports"] == {"8000/tcp": "***", "18188/tcp": "***"}
    for name in ("public_ipaddr", "ssh_host", "ssh_port", "direct_port_start", "direct_port_end",
                 "direct_port_count", "env", "extra_env", "instance_api_key"):
        assert f'"{name}"' not in text
    for value in (EXT_IP, EXT_PORT, "31888", "50022", str(SSH_PORT), SSH_HOST, INSTKEY, ENV_TOKEN):
        assert value not in text


def test_instance_json_without_ports_field_has_no_ports_key(tmp_path):
    art = orch.RunArtifact(tmp_path / "art", lambda text: text)
    art.card_instance(_slot(), {"id": 77, "status": "running"})
    data = json.loads((tmp_path / "art" / "cards" / "slot1_generation1" / "instance.json").read_text(encoding="utf-8"))
    assert "ports" not in data and data["status"] == "running"


# ---------------------------------------------------------------------------
# Релиз: диагностика карт не выкладывается; артефакт не меняется
# ---------------------------------------------------------------------------

CARD_FILES = (
    "cards/slot1_generation1/ssh_diag.log",
    "cards/slot1_generation1/vast_logs.txt",
    "cards/slot1_generation1/worker_log.txt",
    "cards/slot1_generation1/instance.json",
)


def _filled_artifact(tmp_path):
    art = orch.RunArtifact(tmp_path / "art", lambda text: text)
    slot = _slot()
    art.card_ssh_diag(slot, "=== 1 ===\n--- ping (ok) ---\nok")
    art.card_vast_logs(slot, "vast log line")
    art.card_line(slot, "worker log line")
    art.card_instance(slot, {"id": 77, "status": "running"})
    art.card_status(slot, "status line")
    art.card_meta(slot, {"slot": 1})
    art.write_report("итог запуска")
    art.write_files_list()
    return art


def test_release_bundle_excludes_card_diagnostics_only(tmp_path):
    art = _filled_artifact(tmp_path)
    dest = tmp_path / "release-logs.zip"
    art.make_bundle(str(dest), for_release=True)
    names = set(zipfile.ZipFile(dest).namelist())

    assert not (names & set(CARD_FILES))
    # файлы, которые были в релизе раньше, остаются
    for kept in ("run/summary.md", "FILES.txt", "cards/slot1_generation1/status.txt",
                 "cards/slot1_generation1/meta.json"):
        assert kept in names
    # в артефакте диагностика остаётся на диске
    for name in CARD_FILES:
        assert (art.root / name).is_file()


def test_bundle_without_release_flag_still_contains_card_files(tmp_path):
    art = _filled_artifact(tmp_path)
    dest = tmp_path / "all.zip"
    art.make_bundle(str(dest))
    assert set(CARD_FILES) <= set(zipfile.ZipFile(dest).namelist())


class _Store(ra.ReleaseStore):
    """Хранилище-заглушка: как в test_release_adapter.py, без сети."""

    def __init__(self):
        self.releases = {}
        self.uploads = []

    def add(self, tag, body, assets=(), draft=False):
        self.releases.setdefault(tag, []).append(ra.ReleaseInfo(tag, "t", body, draft, list(assets)))

    def find_releases(self, tag):
        return list(self.releases.get(tag, []))

    def create_release(self, tag, title, body):
        self.add(tag, body)

    def upload_asset(self, tag, path):
        self.uploads.append((tag, path))


def test_release_upload_refuses_card_diagnostics():
    assert ra.CARD_DIAGNOSTIC_FILES == frozenset({"ssh_diag.log", "vast_logs.txt", "worker_log.txt", "instance.json"})
    from datetime import datetime, timezone
    segs = [Segment(1, 0, 1000, "a"), Segment(2, 1000, 2000, "b")]
    store = _Store()
    up = ra.ReleaseUploader(store, segs, "ltx25", release_asset_limit=900,
                            now=lambda: datetime(2026, 10, 5, 12, 30, tzinfo=timezone.utc),
                            log=lambda m: None, run_id="42")
    up.prepare()
    for name in sorted(ra.CARD_DIAGNOSTIC_FILES):
        with pytest.raises(GeminiInputError):
            up.upload_run_file(f"/tmp/{name}")
    assert store.uploads == []
