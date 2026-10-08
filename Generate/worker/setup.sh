#!/usr/bin/env bash
# Generate/worker/setup.sh

set -u

CHECKPOINT_DIR="/workspace/ComfyUI/models/checkpoints"
MODEL_NAME="ltx-video-2b-v0.9.1.safetensors"
MODEL_URL="https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.1.safetensors"
MODEL_PATH="${CHECKPOINT_DIR}/${MODEL_NAME}"

COMFY_READY_URL="http://127.0.0.1:8188/system_stats"
COMFY_TIMEOUT_SEC=180
COMFY_POLL_SEC=2

COMFY_PID=""
SERVER_PID=""

log() {
    echo "[setup] $*"
}

stop_processes() {
    log "Получен сигнал завершения, останавливаю процессы..."
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
    if [ -n "${COMFY_PID}" ] && kill -0 "${COMFY_PID}" 2>/dev/null; then
        kill -TERM "${COMFY_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    log "Все процессы остановлены."
    exit 0
}

trap stop_processes SIGTERM SIGINT

# --- 1. Проверка и загрузка чекпоинта модели ---
mkdir -p "${CHECKPOINT_DIR}"

if [ -s "${MODEL_PATH}" ]; then
    log "Чекпоинт модели уже существует: ${MODEL_NAME}"
else
    log "Чекпоинт модели не найден, начинаю загрузку: ${MODEL_NAME}"
    rm -f "${MODEL_PATH}" "${MODEL_PATH}.aria2" 2>/dev/null || true
    if command -v aria2c >/dev/null 2>&1; then
        log "Загрузка через aria2c (8 соединений)..."
        aria2c -x 8 -s 8 -k 1M \
            -d "${CHECKPOINT_DIR}" -o "${MODEL_NAME}" \
            --console-log-level=warn --summary-interval=0 \
            "${MODEL_URL}"
        DL_RC=$?
    else
        log "aria2c недоступен, загрузка через curl..."
        curl -L --fail -o "${MODEL_PATH}" "${MODEL_URL}"
        DL_RC=$?
    fi
    if [ "${DL_RC}" -ne 0 ] || [ ! -s "${MODEL_PATH}" ]; then
        log "ОШИБКА: не удалось загрузить чекпоинт модели (код ${DL_RC})."
        rm -f "${MODEL_PATH}" 2>/dev/null || true
        exit 1
    fi
    log "Чекпоинт модели успешно загружен."
fi

# --- 2. Запуск ComfyUI в headless-режиме ---
log "Запускаю ComfyUI (127.0.0.1:8188)..."
python /workspace/ComfyUI/main.py --listen 127.0.0.1 --port 8188 &
COMFY_PID=$!
log "ComfyUI запущен, PID: ${COMFY_PID}"

# --- 3. Ожидание готовности ComfyUI ---
log "Ожидаю готовности ComfyUI (до ${COMFY_TIMEOUT_SEC} сек)..."
ELAPSED=0
READY=0
while [ "${ELAPSED}" -lt "${COMFY_TIMEOUT_SEC}" ]; do
    if curl -sf -o /dev/null "${COMFY_READY_URL}"; then
        READY=1
        break
    fi
    if ! kill -0 "${COMFY_PID}" 2>/dev/null; then
        log "ОШИБКА: процесс ComfyUI неожиданно завершился."
        exit 1
    fi
    sleep "${COMFY_POLL_SEC}"
    ELAPSED=$((ELAPSED + COMFY_POLL_SEC))
done

if [ "${READY}" -ne 1 ]; then
    log "ОШИБКА: ComfyUI не стал готов за ${COMFY_TIMEOUT_SEC} сек."
    kill -TERM "${COMFY_PID}" 2>/dev/null || true
    exit 1
fi
log "ComfyUI готов к работе (прошло ${ELAPSED} сек)."

# --- 4. Запуск сервера воркера ---
log "Запускаю сервер воркера..."
python /workspace/Generate/worker/server.py &
SERVER_PID=$!
log "Сервер воркера запущен, PID: ${SERVER_PID}"

# --- 5. Ожидание завершения процессов ---
wait -n "${COMFY_PID}" "${SERVER_PID}"
EXIT_CODE=$?
log "Один из процессов завершился (код ${EXIT_CODE}), останавливаю остальные..."
kill -TERM "${SERVER_PID}" "${COMFY_PID}" 2>/dev/null || true
wait 2>/dev/null || true
exit "${EXIT_CODE}"
