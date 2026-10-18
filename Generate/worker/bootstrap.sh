#!/bin/bash
# Generate/worker/bootstrap.sh
#
# Точка входа onstart для стокового образа vastai/comfy (SPEC 8.1, вариант A).
# Код репозитория приходит не из образа, а отсюда: архив нужного коммита, затем setup.sh
# (загрузка весов, ComfyUI, сервер воркера).
#
# Правила (SPEC 0.1): значения WORKER_TOKEN, HF_TOKEN и прочих переменных только читаются
# из окружения и никогда не печатаются. Дамп всех переменных и вывод команд шелла (трассировка)
# в этом файле запрещены: они печатают секреты.
# Все сообщения на русском (SPEC 0.5). Каждый шаг пишется строкой «[bootstrap] шаг N: ...»
# с кодом возврата команды.
#
# Коды выхода: 2 нет переменных или неверный WORKER_BUILD; 3 архив не получен;
# 4 нет инструментов и их не удалось установить; 5 нет папки ComfyUI; 6 не запустился setup.sh.

set -u

REPO_SLUG="${REPO_SLUG:-SOTONATORE/Download}"
REPO_DIR="${REPO_DIR:-/workspace/Generate}"
COMFY_DIR="${COMFY_DIR:-/workspace/ComfyUI}"

say() {
    printf '[bootstrap] %s\n' "$*"
}

say_err() {
    printf '[bootstrap] %s\n' "$*" >&2
}

# --- шаг 1: переменные окружения (значения не печатаются) ---
say "шаг 1: проверка переменных окружения (значения не печатаются)"
if [ -z "${WORKER_TOKEN:-}" ]; then
    say_err "шаг 1: ОШИБКА: не задана переменная WORKER_TOKEN, запуск невозможен (код 2)."
    exit 2
fi
if ! printf '%s' "${WORKER_BUILD:-}" | grep -Eq '^[0-9a-f]{7,64}$'; then
    say_err "шаг 1: ОШИБКА: WORKER_BUILD не задан или не похож на номер коммита (нужны 7-64 символа 0-9a-f) (код 2)."
    exit 2
fi
say "шаг 1: переменные на месте (код 0)"

# --- шаг 2: код репозитория в REPO_DIR ---
say "шаг 2: код репозитория в ${REPO_DIR}"
if [ -f "${REPO_DIR}/worker/setup.sh" ]; then
    say "шаг 2: каталог уже на месте, загрузка архива пропущена (код 0)"
elif [ -e "${REPO_DIR}" ]; then
    say_err "шаг 2: ОШИБКА: ${REPO_DIR} существует, но в нём нет worker/setup.sh. Каталог не трогаю (код 3)."
    exit 3
else
    tmp_dir="$(mktemp -d /tmp/bootstrap-repo.XXXXXX)" || {
        say_err "шаг 2: ОШИБКА: не удалось создать временную папку (код 3)."
        exit 3
    }
    tgz="${tmp_dir}/repo.tar.gz"
    t0=$(date +%s)
    # архив коммита напрямую с codeload.github.com, без git clone; адрес с номером коммита не печатается
    curl -fsSL --retry 3 -o "${tgz}" "https://codeload.github.com/${REPO_SLUG}/tar.gz/${WORKER_BUILD}"
    rc=$?
    t1=$(date +%s)
    if [ "${rc}" -ne 0 ]; then
        say_err "шаг 2: ОШИБКА: не удалось скачать архив коммита (код ${rc}, ${t1} - ${t0} с)."
        rm -rf "${tmp_dir}"
        exit 3
    fi
    size=$(stat -c %s "${tgz}" 2>/dev/null || echo "?")
    say "шаг 2: архив скачан: ${size} байт, $((t1 - t0)) с (код ${rc})"
    tar -xzf "${tgz}" -C "${tmp_dir}"
    rc=$?
    src=""
    for d in "${tmp_dir}"/*/Generate; do
        [ -d "${d}" ] && src="${d}" && break
    done
    if [ "${rc}" -ne 0 ] || [ -z "${src}" ]; then
        say_err "шаг 2: ОШИБКА: архив не распаковался или в нём нет папки Generate (код ${rc})."
        rm -rf "${tmp_dir}"
        exit 3
    fi
    mkdir -p "$(dirname "${REPO_DIR}")"
    mv "${src}" "${REPO_DIR}"
    rc=$?
    rm -rf "${tmp_dir}"
    if [ "${rc}" -ne 0 ] || [ ! -f "${REPO_DIR}/worker/setup.sh" ]; then
        say_err "шаг 2: ОШИБКА: не удалось разложить код в ${REPO_DIR} (код ${rc})."
        exit 3
    fi
    say "шаг 2: код распакован в ${REPO_DIR} (код 0)"
fi

# --- шаг 3: инструменты (UNVERIFIED против стокового образа vastai/comfy) ---
# Список проверен только по Dockerfile (apt: aria2, ffmpeg; pip: httpx). На стоковом образе
# не проверен: при первом прогоне сверить по выводу шага 3.
say "шаг 3: проверка инструментов: aria2c, ffmpeg, python3, модуль httpx"
missing_pkgs=()
for tool in aria2c ffmpeg python3; do
    if command -v "${tool}" >/dev/null 2>&1; then
        say "шаг 3: ${tool} найден (код 0)"
    else
        say "шаг 3: ${tool} не найден (код 1)"
        case "${tool}" in
            aria2c) missing_pkgs+=(aria2) ;;
            ffmpeg) missing_pkgs+=(ffmpeg) ;;
            python3) missing_pkgs+=(python3) ;;
        esac
    fi
done
need_httpx=0
if command -v python3 >/dev/null 2>&1; then
    python3 -c "import httpx" >/dev/null 2>&1
    rc=$?
    if [ "${rc}" -eq 0 ]; then
        say "шаг 3: модуль httpx найден (код 0)"
    else
        say "шаг 3: модуль httpx не найден (код ${rc})"
        need_httpx=1
    fi
else
    need_httpx=1
fi

if [ "${#missing_pkgs[@]}" -gt 0 ]; then
    say "шаг 3: установка через apt-get: ${missing_pkgs[*]}"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1
    rc=$?
    say "шаг 3: apt-get update (код ${rc})"
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing_pkgs[@]}" >/dev/null 2>&1
    rc=$?
    say "шаг 3: apt-get install (код ${rc})"
fi
if [ "${need_httpx}" -eq 1 ] && command -v python3 >/dev/null 2>&1; then
    python3 -m pip install -q httpx >/dev/null 2>&1
    rc=$?
    if [ "${rc}" -ne 0 ]; then
        python3 -m pip install -q --break-system-packages httpx >/dev/null 2>&1
        rc=$?
    fi
    say "шаг 3: установка httpx через pip (код ${rc})"
fi

still=""
for tool in aria2c ffmpeg python3; do
    command -v "${tool}" >/dev/null 2>&1 || still="${still} ${tool}"
done
if [ -n "${still}" ] || ! python3 -c "import httpx" >/dev/null 2>&1; then
    say_err "шаг 3: ОШИБКА: не хватает инструментов после установки:${still:- httpx} (код 4)."
    exit 4
fi
say "шаг 3: инструменты на месте (код 0)"

# --- шаг 4: папка ComfyUI (путь не угадываю: только COMFY_DIR или значение по умолчанию) ---
say "шаг 4: папка ComfyUI: ${COMFY_DIR}"
if [ ! -f "${COMFY_DIR}/main.py" ]; then
    say_err "шаг 4: ОШИБКА: в папке ComfyUI нет main.py: ${COMFY_DIR}. Задайте COMFY_DIR правильным путём (код 5)."
    exit 5
fi
say "шаг 4: ComfyUI найден (код 0)"

# --- шаг 5: окружение для setup.sh и передача управления ---
# Значения по умолчанию совпадают с Dockerfile (ветка option B, в резерве). Переменные из onstart имеют приоритет.
export COMFY_DIR
export COMFY_OUTPUT_DIR="${COMFY_OUTPUT_DIR:-${COMFY_DIR}/output}"
export COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"
export COMFY_LOG_PATH="${COMFY_LOG_PATH:-/workspace/comfyui.log}"
export WORKER_HTTP_PORT="${WORKER_HTTP_PORT:-8000}"
export WORKER_WORKFLOW_PATH="${WORKER_WORKFLOW_PATH:-${REPO_DIR}/model_profiles/workflows/ltx25.json}"
export WORKER_BINDINGS_PATH="${WORKER_BINDINGS_PATH:-${REPO_DIR}/model_profiles/workflows/ltx25_bindings.json}"
export GEN_LOG_LEVEL="${GEN_LOG_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1
# server.py импортирует пакет Generate.worker: корень пакета (родитель REPO_DIR) должен быть в PYTHONPATH
export PYTHONPATH="$(dirname "${REPO_DIR}")${PYTHONPATH:+:${PYTHONPATH}}"

setup_sh="${REPO_DIR}/worker/setup.sh"
say "шаг 5: передача управления setup.sh (веса, ComfyUI, сервер воркера) (код 0)"
exec bash "${setup_sh}"
say_err "шаг 6: ОШИБКА: не удалось запустить ${setup_sh} (код 6)."
exit 6
