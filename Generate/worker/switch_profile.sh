#!/usr/bin/env bash
# Generate/worker/switch_profile.sh
# Переключение профиля модели на простое карте (шаг W3).
#
# Использование: switch_profile.sh <папка старого профиля> <папка нового профиля>
# (обе — абсолютные пути; внутри лежат weights.json и profile.yaml).
#
# Порядок шагов (при первом сбое остановка):
#   S1. проверки нового профиля и окружения до любых изменений: файлы, описание весов, токен,
#       HTTP 200 и Content-Length у каждого файла, место на диске. Отказ — выход 3, ComfyUI не трогается.
#   S2. удаление файлов старого профиля (только перечисленных в его weights.json и существующих),
#       затем остановка ComfyUI (TERM, до 20 с ожидания, затем KILL). Сбой — выход 1.
#   S3. загрузка файлов нового профиля через общий weights_download.sh, проверка размеров;
#       повторные проходы для недостающих и неверных файлов (всего до 3 проходов). Сбой — выход 1.
#   S4. запуск ComfyUI той же командой, что bootstrap.sh шаг 4б: до 3 попыток, ожидание ответа
#       на COMFY_READY_URL; неудачная попытка останавливается по PID. Сбой — выход 1.
#   S5. «готово», выход 0.
#
# Коды выхода: 0 — успех; 1 — сбой после удаления (карта не готова); 2 — неверные аргументы или
# не заданы переменные окружения; 3 — отказ до каких-либо изменений.
# Токен HF_TOKEN никогда не печатается и не попадает в аргументы процесса: он уходит только в
# заголовок Authorization (через process substitution, как в weights_download.sh).

set -u

log() {
    echo "[switch] $*"
}

# Отказ до изменений (код 3). Последняя строка «[switch]» — причина для сервера.
refuse() {
    echo "[switch] переключение отменено до изменений: $*"
    exit 3
}

# Процесс жив: есть PID и он не зомби
pid_alive() {
    local pid="$1"
    kill -0 "${pid}" 2>/dev/null || return 1
    if [ -r "/proc/${pid}/status" ] && grep -q '^State:[[:space:]]*Z' "/proc/${pid}/status"; then
        return 1
    fi
    return 0
}

# Остановка процесса по PID: TERM, до 20 с ожидания, затем KILL. Возврат 0 — процесса уже нет.
stop_comfy_pid() {
    local pid="$1"
    local waited=0
    if ! pid_alive "${pid}"; then
        return 0
    fi
    kill -TERM "${pid}" 2>/dev/null || true
    while pid_alive "${pid}" && [ "${waited}" -lt 20 ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if pid_alive "${pid}"; then
        log "Шаг 2: PID ${pid} не завершился за 20 сек, посылаю kill -9."
        kill -9 "${pid}" 2>/dev/null || true
        sleep 1
    fi
    ! pid_alive "${pid}"
}

if [ "$#" -ne 2 ] || [ -z "$1" ] || [ -z "$2" ]; then
    log "ОШИБКА: использование: switch_profile.sh <папка старого профиля> <папка нового профиля> (код 2)."
    exit 2
fi
OLD_PROFILE_DIR="$1"
NEW_PROFILE_DIR="$2"

# Обязательные переменные окружения (значения не печатаются)
MISSING=""
for var_name in COMFY_DIR COMFY_MODELS_DIR COMFY_PYTHON COMFY_LOG_FILE COMFY_PID_FILE COMFYUI_ARGS HF_BASE_URL; do
    if [ -z "${!var_name:-}" ]; then
        MISSING="${MISSING} ${var_name}"
    fi
done
if [ -n "${MISSING}" ]; then
    log "ОШИБКА: не заданы переменные окружения:${MISSING} (код 2)."
    exit 2
fi

if [ ! -d "${OLD_PROFILE_DIR}" ] || [ ! -d "${NEW_PROFILE_DIR}" ]; then
    log "ОШИБКА: папка профиля не найдена (код 2)."
    exit 2
fi

# REPO_DIR — родитель папки worker/, где лежит этот скрипт
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_PLAN="${REPO_DIR}/core/weights_plan.py"
WEIGHTS_DOWNLOAD="${REPO_DIR}/worker/weights_download.sh"

COMFY_READY_URL="http://127.0.0.1:18188/system_stats"
COMFY_TIMEOUT_SEC="${COMFY_TIMEOUT_SEC:-180}"
COMFY_POLL_SEC="${COMFY_POLL_SEC:-2}"
WORKER_BOOTSTRAP_LOG="${WORKER_BOOTSTRAP_LOG:-/opt/worker/bootstrap.log}"
ARIA2_MODE="aria2"
if ! command -v aria2c >/dev/null 2>&1; then
    ARIA2_MODE="curl"
fi
FREE_RESERVE_BYTES=1073741824   # запас места: 1 ГиБ
DL_PASS_MAX=3                   # проходов загрузки всего (первый и повторы)
DL_PASS_PAUSE_SEC=10
COMFY_START_ATTEMPTS_MAX=3

# --- S1 preflight: все проверки до любых изменений (ничего не удаляется, ComfyUI работает) ---

# окружение ComfyUI: нужно и для проверок, и для запуска в S4
[ -d "${COMFY_DIR}" ] || refuse "папка ComfyUI не найдена."
[ -f "${COMFY_DIR}/main.py" ] || refuse "в папке ComfyUI нет main.py."
read -r -a ARGS <<< "${COMFYUI_ARGS}"
if [ "${#ARGS[@]}" -eq 0 ]; then
    refuse "COMFYUI_ARGS пуст, запускать ComfyUI нечем."
fi

# a. файлы нового профиля, описание весов, число строк весов
for fname in profile.yaml workflow.json bindings.json weights.json; do
    test -f "${NEW_PROFILE_DIR}/${fname}" || refuse "в папке нового профиля нет файла ${fname}."
done
META="$(python3 "${WEIGHTS_PLAN}" meta "${NEW_PROFILE_DIR}")" || refuse "описание весов нового профиля не читается."
META_OK=1
if [ "$(printf '%s\n' "${META}" | wc -l)" -ne 3 ]; then META_OK=0; fi
if [ "${META_OK}" = 1 ]; then
    [[ "$(printf '%s\n' "${META}" | sed -n 1p)" =~ ^type=(video|image)$ ]] || META_OK=0
    [[ "$(printf '%s\n' "${META}" | sed -n 2p)" =~ ^needs_token=[01]$ ]] || META_OK=0
    [[ "$(printf '%s\n' "${META}" | sed -n 3p)" =~ ^files_total=[1-9][0-9]*$ ]] || META_OK=0
fi
if [ "${META_OK}" != 1 ]; then
    refuse "описание весов нового профиля некорректно."
fi
NEED_TOKEN="$(printf '%s\n' "${META}" | sed -n 's/^needs_token=//p')"
MODEL_FILES_TOTAL="$(printf '%s\n' "${META}" | sed -n 's/^files_total=//p')"

OLD_ROWS="$(python3 "${WEIGHTS_PLAN}" files "${OLD_PROFILE_DIR}")" || refuse "список весов старого профиля не читается."
NEW_ROWS="$(python3 "${WEIGHTS_PLAN}" files "${NEW_PROFILE_DIR}")" || refuse "список весов нового профиля не читается."
NEW_ROW_COUNT="$(printf '%s\n' "${NEW_ROWS}" | awk 'NF { n++ } END { print n + 0 }')"
if [ "${NEW_ROW_COUNT}" -ne "${MODEL_FILES_TOTAL}" ]; then
    refuse "число строк весов нового профиля не совпадает с files_total."
fi

# b. токен
HF_TOKEN="${HF_TOKEN:-}"
if [ "${NEED_TOKEN}" = "1" ] && [ -z "${HF_TOKEN}" ]; then
    refuse "для нового профиля нужен HF_TOKEN, а он не задан."
fi

# c. доступность каждого файла: HEAD с тем же заголовком Authorization, что в weights_download.sh
ROW_SUBS=()
ROW_NAMES=()
ROW_RPS=()
ROW_BASES=()
ROW_SIZES=()
NEW_TOTAL_BYTES=0
while IFS= read -r line; do
    [ -z "${line}" ] && continue
    sub="${line%%$'\t'*}"; rest="${line#*$'\t'}"
    name="${rest%%$'\t'*}"; rest="${rest#*$'\t'}"
    rp="${rest%%$'\t'*}"; base="${rest#*$'\t'}"
    base="${base:-${HF_BASE_URL}}"
    if [ -n "${rp}" ]; then
        url="${base}/${rp}"
    else
        url="${base}/${sub}/${name}"
    fi
    HDRS="$(curl -sIL -m 20 -H @<(printf 'Authorization: Bearer %s\n' "${HF_TOKEN}") "${url}" 2>/dev/null | tr -d '\r')"
    code="$(printf '%s\n' "${HDRS}" | awk '$1 ~ /^HTTP\// { code = $2 } END { print code }')"
    clen="$(printf '%s\n' "${HDRS}" | awk 'tolower($1) == "content-length:" { v = $2 } END { print v }')"
    if [ "${code}" != "200" ]; then
        refuse "файл ${sub}/${name} недоступен на сервере загрузки (HTTP ${code:-нет ответа})."
    fi
    if ! [[ "${clen}" =~ ^[1-9][0-9]*$ ]]; then
        refuse "у файла ${sub}/${name} нет корректного Content-Length."
    fi
    ROW_SUBS+=("${sub}")
    ROW_NAMES+=("${name}")
    ROW_RPS+=("${rp}")
    ROW_BASES+=("${base}")
    ROW_SIZES+=("${clen}")
    NEW_TOTAL_BYTES=$((NEW_TOTAL_BYTES + clen))
done <<< "${NEW_ROWS}"
ROW_COUNT="${#ROW_SUBS[@]}"

# d. место: свободное на диске моделей + старые файлы, которые будут удалены, >= новые + 1 ГиБ
FREE_BYTES="$(df -P -B1 "${COMFY_MODELS_DIR}" 2>/dev/null | awk 'NR == 2 { print $4 }')"
if ! [[ "${FREE_BYTES}" =~ ^[0-9]+$ ]]; then
    refuse "не удалось узнать свободное место в папке моделей."
fi
OLD_EXISTING_BYTES=0
while IFS= read -r line; do
    [ -z "${line}" ] && continue
    sub="${line%%$'\t'*}"; rest="${line#*$'\t'}"
    name="${rest%%$'\t'*}"
    path="${COMFY_MODELS_DIR}/${sub}/${name}"
    if [ -e "${path}" ]; then
        size="$(stat -c %s "${path}" 2>/dev/null || echo 0)"
        OLD_EXISTING_BYTES=$((OLD_EXISTING_BYTES + size))
    fi
done <<< "${OLD_ROWS}"
NEED_BYTES=$((NEW_TOTAL_BYTES + FREE_RESERVE_BYTES))
AVAIL_BYTES=$((FREE_BYTES + OLD_EXISTING_BYTES))
if [ "${AVAIL_BYTES}" -lt "${NEED_BYTES}" ]; then
    refuse "мало места на диске моделей: нужно ${NEED_BYTES} байт, доступно ${AVAIL_BYTES} байт."
fi
log "Проверка пройдена: новый профиль, ${ROW_COUNT} файлов, ${NEW_TOTAL_BYTES} байт, все доступны."

# --- S2 удаление файлов старого профиля, затем остановка ComfyUI ---
REMOVED=0
while IFS= read -r line; do
    [ -z "${line}" ] && continue
    sub="${line%%$'\t'*}"; rest="${line#*$'\t'}"; name="${rest%%$'\t'*}"
    path="${COMFY_MODELS_DIR}/${sub}/${name}"
    if [ -e "${path}" ]; then
        if rm -f -- "${path}"; then
            REMOVED=$((REMOVED + 1))
        else
            log "ОШИБКА: не удалось удалить файл старого профиля ${sub}/${name} (код 1)."
            exit 1
        fi
    fi
done <<< "${OLD_ROWS}"
log "Шаг 2: удалено файлов старого профиля: ${REMOVED}."

OLD_PID="$(cat "${COMFY_PID_FILE}" 2>/dev/null || true)"
if [ -n "${OLD_PID}" ] && pid_alive "${OLD_PID}"; then
    log "Шаг 2: останавливаю ComfyUI, PID ${OLD_PID}."
    if ! stop_comfy_pid "${OLD_PID}"; then
        log "ОШИБКА: не удалось остановить ComfyUI, PID ${OLD_PID} (код 1)."
        exit 1
    fi
else
    log "Шаг 2: ComfyUI не запущен (PID не найден), останавливать нечего."
fi

# --- S3 загрузка файлов нового профиля через общий weights_download.sh ---
MODEL_FILE_INDEX=0
MODEL_PATHS=()
DL_PIDS=()
COMFY_UP_EARLY=0
# Общие функции загрузки: start_download, download_model, wait_downloads, comfy_answers
source "${WEIGHTS_DOWNLOAD}"

# файл на месте и его размер совпадает с Content-Length из S1; незавершённая загрузка (.aria2) не считается готовой
file_ok() {
    local path="${COMFY_MODELS_DIR}/${ROW_SUBS[$1]}/${ROW_NAMES[$1]}"
    [ -f "${path}" ] && [ ! -f "${path}.aria2" ] && [ "$(stat -c %s "${path}" 2>/dev/null)" = "${ROW_SIZES[$1]}" ]
}

# один проход загрузки: all — все строки, bad — только недоступные или неверные файлы
run_download_pass() {
    local i path
    MODEL_FILE_INDEX=0
    MODEL_PATHS=()
    DL_PIDS=()
    for ((i = 0; i < ROW_COUNT; i++)); do
        if [ "$1" = "all" ] || ! file_ok "${i}"; then
            if [ "$1" != "all" ]; then
                # неверный файл удаляется, иначе download_model сочтёт его готовым и пропустит
                path="${COMFY_MODELS_DIR}/${ROW_SUBS[i]}/${ROW_NAMES[i]}"
                rm -f -- "${path}" "${path}.aria2"
            fi
            start_download "${ROW_SUBS[i]}" "${ROW_NAMES[i]}" "${ROW_RPS[i]}" "${ROW_BASES[i]}"
        fi
    done
    wait_downloads
    for dl_pid in "${DL_PIDS[@]}"; do
        wait "${dl_pid}" || true
    done
}

log "Шаг 3: загрузка весов нового профиля: ${ROW_COUNT} файлов."
run_download_pass all
DL_PASS=1
while :; do
    BAD_COUNT=0
    BAD_LIST=""
    for ((i = 0; i < ROW_COUNT; i++)); do
        if ! file_ok "${i}"; then
            BAD_COUNT=$((BAD_COUNT + 1))
            BAD_LIST="${BAD_LIST} ${ROW_SUBS[i]}/${ROW_NAMES[i]}"
        fi
    done
    if [ "${BAD_COUNT}" -eq 0 ]; then
        break
    fi
    if [ "${DL_PASS}" -ge "${DL_PASS_MAX}" ]; then
        log "ОШИБКА: после ${DL_PASS_MAX} проходов не загружены или неверного размера файлы:${BAD_LIST} (код 1), ComfyUI не запущен."
        exit 1
    fi
    DL_PASS=$((DL_PASS + 1))
    log "Шаг 3: повторная загрузка ${BAD_COUNT} файлов, проход ${DL_PASS} из ${DL_PASS_MAX}, через ${DL_PASS_PAUSE_SEC} сек."
    sleep "${DL_PASS_PAUSE_SEC}"
    run_download_pass bad
done
log "Шаг 3: все файлы нового профиля на месте, размеры совпадают."

# --- S4 запуск ComfyUI с повторами (та же команда, что bootstrap.sh шаг 4б) ---
READY=0
ATTEMPT=0
while [ "${ATTEMPT}" -lt "${COMFY_START_ATTEMPTS_MAX}" ]; do
    ATTEMPT=$((ATTEMPT + 1))
    log "Шаг 4: запуск ComfyUI, попытка ${ATTEMPT} из ${COMFY_START_ATTEMPTS_MAX}."
    (
        cd "${COMFY_DIR}" || exit 1
        nohup "${COMFY_PYTHON}" main.py "${ARGS[@]}" >> "${COMFY_LOG_FILE}" 2>&1 &
        printf '%s\n' "$!" > "${COMFY_PID_FILE}"
    )
    rc=$?
    NEW_PID=""
    if [ "${rc}" -eq 0 ]; then
        NEW_PID="$(cat "${COMFY_PID_FILE}" 2>/dev/null || true)"
    fi
    READY=0
    if [ -n "${NEW_PID}" ] && [[ "${NEW_PID}" =~ ^[0-9]+$ ]]; then
        ELAPSED=0
        while [ "${ELAPSED}" -lt "${COMFY_TIMEOUT_SEC}" ]; do
            if curl -sf -m 5 -o /dev/null "${COMFY_READY_URL}"; then
                READY=1
                break
            fi
            if ! pid_alive "${NEW_PID}"; then
                break
            fi
            sleep "${COMFY_POLL_SEC}"
            ELAPSED=$((ELAPSED + COMFY_POLL_SEC))
        done
    fi
    if [ "${READY}" -eq 1 ]; then
        break
    fi
    log "Шаг 4: попытка ${ATTEMPT} не удалась, ComfyUI не ответил за ${COMFY_TIMEOUT_SEC} сек."
    if [ -n "${NEW_PID}" ] && [[ "${NEW_PID}" =~ ^[0-9]+$ ]] && pid_alive "${NEW_PID}"; then
        if ! stop_comfy_pid "${NEW_PID}"; then
            log "ОШИБКА: не удалось остановить ComfyUI после неудачной попытки, PID ${NEW_PID} (код 1)."
            exit 1
        fi
    fi
done
if [ "${READY}" -ne 1 ]; then
    log "ОШИБКА: ComfyUI не запустился после ${COMFY_START_ATTEMPTS_MAX} попыток (код 1)."
    exit 1
fi

# --- S5 готово ---
log "готово"
exit 0
