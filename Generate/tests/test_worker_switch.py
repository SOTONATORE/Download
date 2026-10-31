"""Офлайн-тесты переключения профиля (шаги W3–W5). Без сети и без запуска shell-скриптов.

Скрипты (switch_profile.sh, weights_download.sh, setup.sh) только читаются как текст.
Решения POST /profile проверяются напрямую функциями и фейковым менеджером.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    from Generate.worker import server as srv
except ImportError:  # pragma: no cover
    from worker import server as srv  # type: ignore

GEN = Path(__file__).resolve().parents[1]
SERVER = GEN / "worker" / "server.py"
MODEL_CONSTANTS = GEN / "core" / "model_constants.py"
SETUP = GEN / "worker" / "setup.sh"
WEIGHTS_DOWNLOAD = GEN / "worker" / "weights_download.sh"
SWITCH = GEN / "worker" / "switch_profile.sh"

_PATTERN_LINE = re.compile(r'^PROFILE_NAME_PATTERN = r"(.*)"\s*$', re.MULTILINE)


def _pattern_in(path: Path) -> str:
    match = _PATTERN_LINE.search(path.read_text(encoding="utf-8"))
    assert match is not None, f"нет строки PROFILE_NAME_PATTERN в {path.name}"
    return match.group(1)


# --- шаблон имени профиля: копия в server.py равна core/model_constants.py ---

def test_profile_name_pattern_in_server_equals_model_constants():
    assert _pattern_in(SERVER) == _pattern_in(MODEL_CONSTANTS)
    assert srv.PROFILE_NAME_PATTERN == _pattern_in(MODEL_CONSTANTS)


# --- проверка имени (POST /profile с неверным именем -> 400) ---

@pytest.mark.parametrize("name", ["ltx25", "zimage_turbo", "my-profile-2"])
def test_validate_profile_name_accepts_good_names(name):
    assert srv.validate_profile_name(name) is None


@pytest.mark.parametrize("name", ["", "../etc", "a/b", "a b", "ltx25.json", "тест", None, 5, "a\n"])
def test_validate_profile_name_rejects_bad_names_in_russian(name):
    error = srv.validate_profile_name(name)
    assert error is not None and "Имя профиля" in error


# --- решение о переключении при занятом воркере (POST /profile -> 409) ---

class FakeManager:
    def __init__(self, busy: bool = False, unacked: int = 0, switching: bool = False) -> None:
        self._busy = busy
        self._unacked = unacked
        self._switching = switching

    def busy(self) -> bool:
        return self._busy

    def unacked_count(self) -> int:
        return self._unacked

    def is_switching(self) -> bool:
        return self._switching


def test_switch_refused_when_busy():
    assert srv.switch_refusal(FakeManager(busy=True)) == srv.SWITCH_BUSY_TEXT


def test_switch_refused_while_unacked_clip_waits():
    assert srv.switch_refusal(FakeManager(unacked=1)) == srv.SWITCH_BUSY_TEXT


def test_switch_refused_when_switch_already_running():
    assert srv.switch_refusal(FakeManager(switching=True)) == srv.SWITCH_RUNNING_TEXT


def test_switch_allowed_when_idle():
    assert srv.switch_refusal(FakeManager()) is None


def test_switch_states_include_refused_and_exit_code_3():
    """Отказ до изменений (код 3) — отдельное состояние: карта остаётся в работе."""
    assert "refused" in srv.SWITCH_STATES
    assert "failed" in srv.SWITCH_STATES
    assert srv.SWITCH_EXIT_REFUSED == 3


def test_begin_switch_is_atomic_and_submit_refused_during_switch():
    manager = srv.TaskManager(comfy=None)
    assert manager.begin_switch() is None
    assert manager.is_switching()
    assert manager.begin_switch() == srv.SWITCH_RUNNING_TEXT
    with pytest.raises(srv.SwitchingError):
        manager.submit(1, "промпт", 9, 1)
    manager.abort_switch()
    assert not manager.is_switching()


# --- switch_profile.sh: только текстовая проверка, скрипт не запускается ---

def test_switch_script_text_has_required_env_check_and_exit_2():
    """Текстовая проверка, не запуск: в файле есть проверка переменных и «exit 2»."""
    text = SWITCH.read_text(encoding="utf-8")
    for name in ("COMFY_DIR", "COMFY_MODELS_DIR", "COMFY_PYTHON", "COMFY_LOG_FILE",
                 "COMFY_PID_FILE", "COMFYUI_ARGS", "HF_BASE_URL"):
        assert name in text, name
    assert "for var_name in" in text
    assert "exit 2" in text


def test_switch_script_preflight_section_exists_and_precedes_removal():
    """S1: раздел проверок есть, и первое удаление (rm -f) идёт после него."""
    text = SWITCH.read_text(encoding="utf-8")
    assert "preflight" in text
    assert text.index("# --- S1 preflight") < text.index("rm -f")


def test_switch_script_order_preflight_removal_download_comfy_start():
    """Порядок по тексту: проверки, удаление, первый start_download, запуск ComfyUI."""
    text = SWITCH.read_text(encoding="utf-8")
    preflight = text.index("# --- S1 preflight")
    removal = text.index("rm -f")
    download = text.index('start_download "')
    comfy_start = text.index('nohup "${COMFY_PYTHON}"')
    assert preflight < removal < download < comfy_start


def test_switch_script_exit_codes_0_1_3_exist():
    text = SWITCH.read_text(encoding="utf-8")
    for code in ("exit 0", "exit 1", "exit 3"):
        assert code in text, code


def test_switch_script_refusal_message_and_refuse_before_change():
    text = SWITCH.read_text(encoding="utf-8")
    assert "переключение отменено до изменений" in text
    assert "refuse()" in text


def test_switch_script_three_pass_download_loop_and_comfy_attempts():
    """S3: три прохода загрузки с паузой; S4: три попытки запуска ComfyUI."""
    text = SWITCH.read_text(encoding="utf-8")
    assert "DL_PASS_MAX=3" in text
    assert "DL_PASS=" in text
    assert "DL_PASS_PAUSE_SEC" in text
    assert "COMFY_START_ATTEMPTS_MAX=3" in text


def test_switch_script_token_printed_only_in_authorization_header():
    """Строки с HF_TOKEN, которые выводят текст (echo, printf, log), — только заголовок Authorization."""
    text = SWITCH.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "HF_TOKEN" not in line or line.lstrip().startswith("#"):
            continue
        if re.search(r"\b(echo|printf|log)\b", line):
            assert "Authorization" in line, line


# --- weights_download.sh: при загрузке не выполняется ни одна команда ---

_FUNC_OPEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{\s*$")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _top_level_problems(text: str) -> tuple:
    """Разбор по строкам: вне тела функции допустимы пустые строки, комментарии, присваивания и
    строки «имя() {». Глубина считается по фигурным скобкам строк кода (комментарии не считаются)."""
    depth = 0
    problems = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if depth == 0 and not (_FUNC_OPEN.match(line) or _ASSIGN.match(line)):
            problems.append((number, raw))
            continue
        depth += line.count("{") - line.count("}")
    return problems, depth


def test_weights_download_has_no_top_level_command():
    text = WEIGHTS_DOWNLOAD.read_text(encoding="utf-8")
    problems, depth = _top_level_problems(text)
    assert problems == [], problems
    assert depth == 0, "скобки функций не сходятся"
    for fn in ("start_download", "download_model", "wait_downloads", "comfy_answers"):
        assert re.search(rf"^{fn}\(\) \{{$", text, re.MULTILINE), fn


# --- setup.sh: подключает общий файл и не содержит вынесенных функций ---

def test_setup_sources_shared_download_and_no_longer_defines_moved_functions():
    text = SETUP.read_text(encoding="utf-8")
    source_line = 'source "${REPO_DIR}/worker/weights_download.sh"'
    assert source_line in text
    for definition in ("start_download()", "download_model()", "wait_downloads()", "comfy_answers()"):
        assert definition not in text, definition
    source_at = text.index(source_line)
    assert source_at < text.index("if comfy_answers"), "source должен идти до первого вызова comfy_answers"
    assert source_at < text.index('start_download "'), "source должен идти до первого вызова start_download"
