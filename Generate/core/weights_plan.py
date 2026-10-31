"""Единственный источник списка весов и типа профиля воркера для setup.sh и bootstrap.sh.

Список файлов берётся из <папка профиля>/weights.json, тип профиля (image или video) — из строки
type: в <папка профиля>/profile.yaml. Нет строки type — профиль считается video (как раньше для ltx25).

Запуск как обычного скрипта, без импорта репозитория:
    python3 weights_plan.py files <папка профиля>   — строки: подкаталог, имя, путь в репо HF, базовый URL (через TAB)
    python3 weights_plan.py meta <папка профиля>    — строки: type=..., needs_token=0|1, files_total=N

Модуль не импортирует yaml: в образе воркера PyYAML не установлен. Чтение type делается регулярным выражением.
"""
import json
import os
import re
import sys

_TYPE_RE = re.compile(r"^type:\s*(\w+)\s*$", re.MULTILINE)
_ALLOWED_TYPES = ("image", "video")
_USAGE = "использование: weights_plan.py files|meta <папка профиля>"


def load_weights(profile_dir: str) -> dict:
    """Читает weights.json профиля и проверяет, что в нём есть непустой список files."""
    path = os.path.join(profile_dir, "weights.json")
    if not os.path.isfile(path):
        raise ValueError(f"нет файла весов: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"файл весов не читается как JSON: {path} ({exc})")
    if not isinstance(data, dict) or not isinstance(data.get("files"), list):
        raise ValueError(f"в файле весов нет списка files: {path}")
    if not data["files"]:
        raise ValueError(f"список files пуст: {path}")
    return data


def profile_type(profile_dir: str) -> str:
    """Возвращает тип профиля для воркера: image или video."""
    path = os.path.join(profile_dir, "profile.yaml")
    if not os.path.isfile(path):
        raise ValueError(f"нет файла профиля: {path}")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    match = _TYPE_RE.search(text)
    if match is None:
        return "video"
    value = match.group(1)
    if value not in _ALLOWED_TYPES:
        raise ValueError(f"недопустимый type в профиле: {value} (ожидается image или video): {path}")
    return value


def file_rows(profile_dir: str) -> list:
    """Строки по одной на файл: (подкаталог, имя, путь в репо HF или пусто, базовый URL)."""
    data = load_weights(profile_dir)
    default_base = data.get("base_url", "")
    rows = []
    for idx, entry in enumerate(data["files"], 1):
        if not isinstance(entry, dict) or not entry.get("subdir") or not entry.get("name"):
            raise ValueError(f"у файла №{idx} в weights.json нет subdir или name")
        base = entry.get("base_url", default_base)
        if not isinstance(base, str) or not base:
            raise ValueError(f"у файла №{idx} в weights.json не задан base_url")
        rows.append((entry["subdir"], entry["name"], entry.get("repo_path", "") or "", base))
    return rows


def download_url(row) -> str:
    """URL загрузки по тому же правилу, что download_model в setup.sh."""
    subdir, name, repo_path, base = row
    if repo_path:
        return base + "/" + repo_path
    return base + "/" + subdir + "/" + name


def meta(profile_dir: str) -> dict:
    """Сводка для setup.sh: тип профиля, нужен ли токен, число файлов."""
    data = load_weights(profile_dir)
    needs = data.get("needs_token")
    if not isinstance(needs, bool):
        raise ValueError(f"needs_token в weights.json должен быть true или false: {profile_dir}")
    return {
        "type": profile_type(profile_dir),
        "needs_token": needs,
        "files_total": len(data["files"]),
    }


def main(argv) -> int:
    if len(argv) != 3 or argv[1] not in ("files", "meta"):
        print(_USAGE, file=sys.stderr)
        return 2
    command, profile_dir = argv[1], argv[2]
    try:
        if command == "files":
            for row in file_rows(profile_dir):
                print("\t".join(row))
        else:
            info = meta(profile_dir)
            print(f"type={info['type']}")
            print(f"needs_token={1 if info['needs_token'] else 0}")
            print(f"files_total={info['files_total']}")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
