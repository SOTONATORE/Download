"""Регрессия: setup.sh запускает сервер воркера из REPO_DIR, а не из жёсткого /opt/worker/Generate.

Onstart распаковывает код в /opt/worker/src/Generate (WORKER_SRC_DIR), поэтому путь /opt/worker/Generate
на карте не существует и сервер воркера не стартует.
"""
from pathlib import Path

SETUP = Path(__file__).resolve().parents[1] / "worker" / "setup.sh"


def _server_launch_lines() -> list[str]:
    # Строка запуска начинается с команды python; комментарии с упоминанием server.py не учитываем.
    return [line for line in SETUP.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("python") and "server.py" in line]


def test_setup_has_single_server_launch_line():
    assert len(_server_launch_lines()) == 1


def test_setup_launches_server_from_repo_dir():
    line = _server_launch_lines()[0]
    assert '"${REPO_DIR}/worker/server.py"' in line


def test_setup_has_no_hardcoded_workspace_generate_path():
    assert "/opt/worker/Generate" not in SETUP.read_text(encoding="utf-8")
