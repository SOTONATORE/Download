"""Правила переносимости (SPEC 0.1, 0.2): статическая проверка исходников."""
import ast
import pathlib
import re

GEN = pathlib.Path(__file__).resolve().parent.parent
CORE = GEN / "core"

# Запрещённые литералы собраны из частей, чтобы этот файл сам их не содержал.
KEYS = ("GEN_" + "GEMINI_API_KEY", "GEN_" + "VAST_API_KEY")


def _code_without_comments(path: pathlib.Path) -> str:
    """Текст файла без комментариев (всё после «#» в строке)."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        lines.append(line.split("#", 1)[0])
    return "\n".join(lines)


def _core_files():
    files = sorted(CORE.glob("*.py"))
    assert files, "в Generate/core нет .py файлов"
    return files


def test_core_does_not_import_app():
    for f in _core_files():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]] if node.level == 0 else []
            else:
                continue
            assert "app" not in roots, f"{f.name}: импорт из app"
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+app\b", text, re.M), f.name


def test_core_has_no_github_specifics():
    for f in _core_files():
        text = f.read_text(encoding="utf-8")
        assert "GITHUB_" not in text, f"{f.name}: GITHUB_*"
        assert "gh release" not in text, f"{f.name}: gh release"
        for line in text.splitlines():
            if "subprocess" in line and re.search(r"\bgh\b", line):
                raise AssertionError(f"{f.name}: subprocess с gh: {line.strip()}")
        assert not re.search(r"""["']gh["']""", text), f"{f.name}: вызов gh"


def test_no_api_key_names_in_code():
    """Имена секретов допустимы только в комментариях и README: на этом этапе ключи не читаются."""
    checked = 0
    for f in sorted(GEN.rglob("*")):
        if not f.is_file() or f.suffix not in {".py", ".yaml", ".yml", ".json"}:
            continue
        if "__pycache__" in f.parts:
            continue
        code = _code_without_comments(f)
        for key in KEYS:
            assert key not in code, f"{f.relative_to(GEN)}: упоминание {key} вне комментария"
        checked += 1
    assert checked > 0


def test_no_environment_reading_in_cli():
    code = _code_without_comments(GEN / "cli.py")
    assert "os.environ" not in code
    assert "getenv" not in code
