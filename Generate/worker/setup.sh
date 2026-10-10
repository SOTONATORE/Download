#!/usr/bin/env bash
# Generate/worker/setup.sh

set -u

COMFY_MODELS_DIR="/workspace/ComfyUI/models"
HF_BASE_URL="https://huggingface.co/Lightricks/LTX-2.5/resolve/main"

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

# Загрузка одного файла модели: download_model <подкаталог в репозитории и в models/> <имя файла>
download_model() {
    local subdir="$1"
    local name="$2"
    local target_dir="${COMFY_MODELS_DIR}/${subdir}"
    local path="${target_dir}/${name}"
    local url="${HF_BASE_URL}/${subdir}/${name}"
    local rc

    mkdir -p "${target_dir}"

    # Файл считается готовым, только если он непустой и нет контрольного файла незавершённой загрузки aria2
    if [ -s "${path}" ] && [ ! -f "${path}.aria2" ]; then
        log "Файл уже существует, пропускаю: ${subdir}/${name}"
        return 0
    fi

    log "Начинаю загрузку: ${subdir}/${name}"
    aria2c \
        --header="Authorization: Bearer ${HF_TOKEN}" \
        -x 16 -s 16 -k 1M -c --console-log-level=warn \
        -d "${target_dir}" -o "${name}" \
        "${url}"
    rc=$?

    if [ "${rc}" -ne 0 ] || [ ! -s "${path}" ] || [ -f "${path}.aria2" ]; then
        log "ОШИБКА: не удалось загрузить файл ${subdir}/${name} (код ${rc})."
        exit 1
    fi
    log "Файл успешно загружен: ${subdir}/${name}"
}

# --- 1. Проверка токена и загрузка модульных файлов модели LTX-2.5 ---
if [ -z "${HF_TOKEN:-}" ]; then
    log "ОШИБКА: Не задана переменная окружения HF_TOKEN. Репозиторий Lightricks/LTX-2.5 закрыт (gated), скачивание весов невозможно."
    exit 1
fi

if ! command -v aria2c >/dev/null 2>&1; then
    log "ОШИБКА: aria2c недоступен, загрузка весов модели невозможна."
    exit 1
fi

download_model "diffusion_models" "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
download_model "text_encoders" "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
download_model "vae" "ltx-2.5-video-vae-bf16.safetensors"
download_model "vae" "ltx-2.5-audio-vae-bf16.safetensors"
download_model "latent_upscale_models" "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
log "Все файлы модели LTX-2.5 на месте."

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
