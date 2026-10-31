#!/usr/bin/env bash
# Generate/worker/switch_profile.sh
# Переключение профиля модели на простое карте (шаг W3).
#
# Использование: switch_profile.sh <папка старого профиля> <папка нового профиля>
# (обе — абсолютные пути; внутри лежат weights.json и profile.yaml).
#
# Порядок шагов; при первом сбое остановка:
#   0. проверка нового профиля (meta) и переменных окружения, до любых изменений файлов;
#   1. удаление файлов старого профиля (только те, что перечислены в его weights.json и существуют);
#   2. остановка ComfyUI по COMFY_PID_FILE: TERM, до 20 с ожидания, затем KILL;
#   3. загрузка файлов нового профиля через общий weights_download.sh, ожидание загрузок;
#   4. запуск ComfyUI той же командой, что bootstrap.sh шаг 4б, ожидание ответа на COMFY_READY_URL;
#   5. «готово».
#
# Коды выхода: 0 — успех; 1 — сбой; 2 — неверные аргументы или не заданы переменные окружения.
# Токен HF_TOKEN никогда не печатается: он уходит только в заголовок Authorization (weights_download.sh).

set -u

log() {
    echo "[switch] $*"
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

# --- 0. Проверка нового профиля до любых изменений ---
META="$(python3 "${WEIGHTS_PLAN}" meta "${NEW_PROFILE_DIR}")" || { log "ОШИБКА: описание весов нового профиля не читается (код 1)."; exit 1; }
META_OK=1
if [ "$(printf '%s\n' "${META}" | wc -l)" -ne 3 ]; then META_OK=0; fi
if [ "${META_OK}" = 1 ]; then
    [[ "$(printf '%s\n' "${META}" | sed -n 1p)" =~ ^type=(video|image)$ ]] || META_OK=0
    [[ "$(printf '%s\n' "${META}" | sed -n 2p)" =~ ^needs_token=[01]$ ]] || META_OK=0
    [[ "$(printf '%s\n' "${META}" | sed -n 3p)" =~ ^files_total=[1-9][0-9]*$ ]] || META_OK=0
fi
if [ "${META_OK}" != 1 ]; then
    log "ОШИБКА: описание весов нового профиля некорректно (код 1)."
    exit 1
fi
NEED_TOKEN="$(printf '%s\n' "${META}" | sed -n 's/^needs_token=//p')"
MODEL_FILES_TOTAL="$(printf '%s\n' "${META}" | sed -n 's/^files_total=//p')"

HF_TOKEN="${HF_TOKEN:-}"
if [ "${NEED_TOKEN}" = "1" ] && [ -z "${HF_TOKEN}" ]; then
    log "ОШИБКА: для нового профиля нужен HF_TOKEN, а он не задан (код 2)."
    exit 2
fi

OLD_ROWS="$(python3 "${WEIGHTS_PLAN}" files "${OLD_PROFILE_DIR}")" || { log "ОШИБКА: список весов старого профиля не читается (код 1)."; exit 1; }
NEW_ROWS="$(python3 "${WEIGHTS_PLAN}" files "${NEW_PROFILE_DIR}")" || { log "ОШИБКА: список весов нового профиля не читается (код 1)."; exit 1; }

log "Переключение: старый профиль ${OLD_PROFILE_DIR}, новый профиль ${NEW_PROFILE_DIR}."

# --- 1. Удаление файлов старого профиля ---
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
log "Шаг 1: удалено файлов старого профиля: ${REMOVED}."

# --- 2. Остановка ComfyUI (как restart_comfy в setup.sh) ---
OLD_PID="$(cat "${COMFY_PID_FILE}" 2>/dev/null || true)"
if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
    log "Шаг 2: останавливаю ComfyUI, PID ${OLD_PID}."
    kill -TERM "${OLD_PID}" 2>/dev/null || true
    waited=0
    while kill -0 "${OLD_PID}" 2>/dev/null && [ "${waited}" -lt 20 ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        log "Шаг 2: PID ${OLD_PID} не завершился за 20 сек, посылаю kill -9."
        kill -9 "${OLD_PID}" 2>/dev/null || true
    fi
else
    log "Шаг 2: ComfyUI не запущен (PID не найден), останавливать нечего."
fi

# --- 3. Загрузка файлов нового профиля (общий код с setup.sh) ---
ARIA2_MODE="aria2"
if ! command -v aria2c >/dev/null 2>&1; then
    ARIA2_MODE="curl"
fi
WORKER_BOOTSTRAP_LOG="${WORKER_BOOTSTRAP_LOG:-/opt/worker/bootstrap.log}"
COMFY_READY_URL="http://127.0.0.1:18188/system_stats"
COMFY_TIMEOUT_SEC="${COMFY_TIMEOUT_SEC:-180}"
COMFY_POLL_SEC="${COMFY_POLL_SEC:-2}"
MODEL_FILE_INDEX=0
MODEL_PATHS=()
DL_PIDS=()
COMFY_UP_EARLY=0
# Общие функции загрузки: start_download, download_model, wait_downloads, comfy_answers
source "${WEIGHTS_DOWNLOAD}"

log "Шаг 3: загрузка весов нового профиля: ${MODEL_FILES_TOTAL} файлов."
while IFS= read -r line; do
    [ -z "${line}" ] && continue
    sub="${line%%$'\t'*}"; rest="${line#*$'\t'}"
    name="${rest%%$'\t'*}"; rest="${rest#*$'\t'}"
    rp="${rest%%$'\t'*}"; base="${rest#*$'\t'}"
    start_download "${sub}" "${name}" "${rp}" "${base}"
done <<< "${NEW_ROWS}"
wait_downloads
DL_FAILED=0
for dl_pid in "${DL_PIDS[@]}"; do
    if ! wait "${dl_pid}"; then
        DL_FAILED=1
    fi
done
if [ "${DL_FAILED}" -ne 0 ]; then
    log "ОШИБКА: не все веса нового профиля загружены (код 1), ComfyUI не запущен."
    exit 1
fi
log "Шаг 3: все файлы нового профиля на месте."

# --- 4. Запуск ComfyUI (та же команда, что bootstrap.sh шаг 4б) ---
read -r -a ARGS <<< "${COMFYUI_ARGS}"
if [ "${#ARGS[@]}" -eq 0 ]; then
    log "ОШИБКА: COMFYUI_ARGS пуст, ComfyUI не запущен (код 1)."
    exit 1
fi
(
    cd "${COMFY_DIR}" || exit 1
    nohup "${COMFY_PYTHON}" main.py "${ARGS[@]}" >> "${COMFY_LOG_FILE}" 2>&1 &
    printf '%s\n' "$!" > "${COMFY_PID_FILE}"
)
rc=$?
if [ "${rc}" -ne 0 ]; then
    log "ОШИБКА: не удалось перейти в ${COMFY_DIR} или запустить ComfyUI (код 1)."
    exit 1
fi
log "Шаг 4: ComfyUI запущен заново, ожидаю ответа на порту 18188 (до ${COMFY_TIMEOUT_SEC} сек)."

ELAPSED=0
READY=0
while [ "${ELAPSED}" -lt "${COMFY_TIMEOUT_SEC}" ]; do
    if curl -sf -m 5 -o /dev/null "${COMFY_READY_URL}"; then
        READY=1
        break
    fi
    sleep "${COMFY_POLL_SEC}"
    ELAPSED=$((ELAPSED + COMFY_POLL_SEC))
done
if [ "${READY}" -ne 1 ]; then
    log "ОШИБКА: ComfyUI не стал готов за ${COMFY_TIMEOUT_SEC} сек (код 1)."
    exit 1
fi

# --- 5. Готово ---
log "готово"
exit 0
