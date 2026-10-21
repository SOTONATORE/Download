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
# с кодом возврата команды. Перед первым шагом и в каждой ветке ошибки пишется диагностика
# (функция diag): имена и размеры файлов в /workspace, пути программ, PID процессов, признаки переменных.
#
# Коды выхода: 2 нет переменных или неверный WORKER_BUILD; 3 архив не получен;
# 4 нет инструментов и их не удалось установить; 5 нет папки ComfyUI (ни по ожидаемому пути, ни поиском);
# 6 не запустился setup.sh.

set -u

REPO_SLUG="${REPO_SLUG:-SOTONATORE/Download}"
REPO_DIR="${REPO_DIR:-/workspace/Generate}"
# Ожидаемая папка ComfyUI. Если main.py там нет, шаг 4 ищет его в COMFY_SEARCH_DIRS.
COMFY_DIR="${COMFY_DIR:-/workspace/ComfyUI}"
COMFY_SEARCH_DIRS="${COMFY_SEARCH_DIRS:-/workspace /opt /root /home}"
COMFY_SEARCH_DEPTH="${COMFY_SEARCH_DEPTH:-6}"
COMFY_SEARCH_TIMEOUT="${COMFY_SEARCH_TIMEOUT:-60}"
BOOTSTRAP_LOG="${WORKER_BOOTSTRAP_LOG:-/workspace/bootstrap.log}"

# Одна функция записи для всех сообщений шагов. Строка уходит в stdout и в BOOTSTRAP_LOG.
# onstart запускает скрипт с «>> лог 2>&1», то есть stdout уже и есть файл лога: тогда второй записи нет,
# иначе каждая строка попала бы в файл дважды. Аргументы — только тексты шагов, значения переменных окружения
# сюда не передаются (WORKER_TOKEN и HF_TOKEN не печатаются).
log() {
    local line="[bootstrap] $*"
    if [ -n "${BOOTSTRAP_LOG}" ] && ! [ /proc/self/fd/1 -ef "${BOOTSTRAP_LOG}" ]; then
        printf '%s\n' "${line}" | tee -a "${BOOTSTRAP_LOG}"
    else
        printf '%s\n' "${line}"
    fi
}

say() {
    log "$@"
}

# Ошибки тоже идут в stdout (раньше в stderr): иначе при разборе stdout контейнера они терялись бы.
say_err() {
    log "$@"
}

# Диагностика в лог. Вызывается с аргументом «start» перед первым шагом и с причиной в каждой ветке ошибки.
# Выводятся только имена и размеры файлов, пути программ, имена процессов с PID и признаки переменных.
# Значения переменных и аргументы процессов не печатаются. Функция не завершает скрипт.
diag() {
    local reason="$1"
    say "диагностика (${reason}):"

    say "  содержимое /workspace (имя, размер в байтах):"
    if [ -d /workspace ]; then
        find /workspace -mindepth 1 -maxdepth 1 -printf '%s %f\n' 2>/dev/null | while read -r size name; do
            say "    ${name} ${size}"
        done
    else
        say "    папки /workspace нет"
    fi

    # Хвост лога только при повторном запуске: в логе уже есть строки прошлого bootstrap
    if [ "${2:-}" = "start" ] && [ -f "${BOOTSTRAP_LOG}" ] && grep -q '^\[bootstrap\] ' "${BOOTSTRAP_LOG}" 2>/dev/null; then
        say "  хвост ${BOOTSTRAP_LOG}, последние 50 строк (лог уже был, повторный запуск):"
        tail -n 50 "${BOOTSTRAP_LOG}" 2>/dev/null | while IFS= read -r tail_line; do
            say "    ${tail_line}"
        done
    fi

    say "  программы (путь или «нет»):"
    local prog p
    for prog in curl tar python python3 aria2c; do
        p="$(command -v "${prog}" 2>/dev/null)" || true
        say "    ${prog}: ${p:-нет}"
    done

    say "  процессы server.py и bootstrap (PID и имя скрипта, без аргументов):"
    if command -v ps >/dev/null 2>&1; then
        local procs pid pname
        procs="$(ps -eo pid=,args= 2>/dev/null | grep -E '[s]erver\.py|[b]ootstrap\.sh' \
            | sed -E 's/^[[:space:]]*([0-9]+) .*(server\.py|bootstrap\.sh).*$/\1 \2/')"
        if [ -n "${procs}" ]; then
            while read -r pid pname; do
                say "    PID ${pid}: ${pname}"
            done <<< "${procs}"
        else
            say "    нет"
        fi
    else
        say "    утилита ps не найдена"
    fi

    local var
    for var in HF_TOKEN WORKER_TOKEN; do
        if [ -n "${!var:-}" ]; then
            say "  переменная ${var}: задана"
        else
            say "  переменная ${var}: не задана"
        fi
    done
}

diag "старт" start

# --- шаг 1: переменные окружения (значения не печатаются) ---
say "шаг 1: проверка переменных окружения (значения не печатаются)"
if [ -z "${WORKER_TOKEN:-}" ]; then
    say_err "шаг 1: ОШИБКА: не задана переменная WORKER_TOKEN, запуск невозможен (код 2)."
    diag "шаг 1: нет WORKER_TOKEN"
    exit 2
fi
if ! printf '%s' "${WORKER_BUILD:-}" | grep -Eq '^[0-9a-f]{7,64}$'; then
    say_err "шаг 1: ОШИБКА: WORKER_BUILD не задан или не похож на номер коммита (нужны 7-64 символа 0-9a-f) (код 2)."
    diag "шаг 1: неверный WORKER_BUILD"
    exit 2
fi
say "шаг 1: переменные на месте (код 0)"

# --- шаг 2: код репозитория в REPO_DIR ---
say "шаг 2: код репозитория в ${REPO_DIR}"
if [ -f "${REPO_DIR}/worker/setup.sh" ]; then
    say "шаг 2: каталог уже на месте, загрузка архива пропущена (код 0)"
elif [ -e "${REPO_DIR}" ]; then
    say_err "шаг 2: ОШИБКА: ${REPO_DIR} существует, но в нём нет worker/setup.sh. Каталог не трогаю (код 3)."
    diag "шаг 2: в REPO_DIR нет setup.sh"
    exit 3
else
    tmp_dir="$(mktemp -d /tmp/bootstrap-repo.XXXXXX)" || {
        say_err "шаг 2: ОШИБКА: не удалось создать временную папку (код 3)."
        diag "шаг 2: mktemp"
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
        diag "шаг 2: curl ${rc}"
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
        diag "шаг 2: tar ${rc}"
        exit 3
    fi
    mkdir -p "$(dirname "${REPO_DIR}")"
    mv "${src}" "${REPO_DIR}"
    rc=$?
    rm -rf "${tmp_dir}"
    if [ "${rc}" -ne 0 ] || [ ! -f "${REPO_DIR}/worker/setup.sh" ]; then
        say_err "шаг 2: ОШИБКА: не удалось разложить код в ${REPO_DIR} (код ${rc})."
        diag "шаг 2: mv ${rc}"
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
    diag "шаг 3: нет инструментов"
    exit 4
fi
say "шаг 3: инструменты на месте (код 0)"

# --- шаг 4: папка ComfyUI: сначала ожидаемый путь, затем поиск main.py с ComfyUI в пути ---
say "шаг 4: ожидаемая папка ComfyUI: ${COMFY_DIR}"
if [ -f "${COMFY_DIR}/main.py" ]; then
    say "шаг 4: ComfyUI на ожидаемом месте, main.py есть (код 0)"
else
    say "шаг 4: в ожидаемой папке нет main.py, ищу (корни: ${COMFY_SEARCH_DIRS}; глубина до ${COMFY_SEARCH_DEPTH}; не дольше ${COMFY_SEARCH_TIMEOUT} с)"
    search_file="$(mktemp /tmp/bootstrap-find.XXXXXX 2>/dev/null)" || search_file=/dev/null
    # COMFY_SEARCH_DIRS намеренно без кавычек: это список папок через пробел. Ближайший по глубине main.py выигрывает.
    timeout "${COMFY_SEARCH_TIMEOUT}" find ${COMFY_SEARCH_DIRS} -maxdepth "${COMFY_SEARCH_DEPTH}" \
        -type f -name main.py -path '*ComfyUI*' -not -path '*/custom_nodes/*' \
        -printf '%d %p\n' > "${search_file}" 2>/dev/null
    find_rc=$?
    found="$(sort -n "${search_file}" 2>/dev/null | head -n 1 | cut -d' ' -f2-)"
    rm -f "${search_file}"
    say "шаг 4: поиск завершён (код ${find_rc}; 124 означает таймаут)"
    if [ -n "${found}" ] && [ -f "${found}" ]; then
        COMFY_DIR="$(dirname "${found}")"
        say "шаг 4: ComfyUI найден поиском: ${COMFY_DIR} (код 0)"
    else
        say_err "шаг 4: ОШИБКА: ComfyUI не найден. Ожидаемая папка ${COMFY_DIR} без main.py; поиск main.py с ComfyUI в пути (корни ${COMFY_SEARCH_DIRS}) ничего не дал. Задайте COMFY_DIR правильным путём (код 5)."
        diag "шаг 4: ComfyUI не найден"
        exit 5
    fi
fi

# --- шаг 5: окружение для setup.sh и передача управления ---
# Значения по умолчанию совпадают с Dockerfile (ветка option B, в резерве). Переменные из onstart имеют приоритет.
# COMFY_DIR здесь уже найденный путь: setup.sh и сервер берут его отсюда и сами не ищут.
export COMFY_DIR
export COMFY_OUTPUT_DIR="${COMFY_OUTPUT_DIR:-${COMFY_DIR}/output}"
export COMFY_URL="${COMFY_URL:-http://127.0.0.1:18188}"  # ComfyUI образа, порт из COMFYUI_ARGS (SPEC 8.5)
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
diag "шаг 6: exec setup.sh"
exit 6
