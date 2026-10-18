#!/usr/bin/env bash
# Generate/worker/setup.sh

set -u

# Папка ComfyUI задаётся bootstrap.sh (COMFY_DIR); по умолчанию /workspace/ComfyUI.
COMFY_DIR="${COMFY_DIR:-/workspace/ComfyUI}"
COMFY_MODELS_DIR="${COMFY_DIR}/models"
HF_BASE_URL="https://huggingface.co/Lightricks/LTX-2.5/resolve/main"

COMFY_READY_URL="http://127.0.0.1:8188/system_stats"
COMFY_TIMEOUT_SEC=180
COMFY_POLL_SEC=2

COMFY_PID=""
SERVER_PID=""

# Счётчик файлов модели для строк лога: «[1/5] ...»
MODEL_FILES_TOTAL=5
MODEL_FILE_INDEX=0

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

    MODEL_FILE_INDEX=$((MODEL_FILE_INDEX + 1))
    local tag="[${MODEL_FILE_INDEX}/${MODEL_FILES_TOTAL}]"

    mkdir -p "${target_dir}"

    # Файл считается готовым, только если он непустой и нет контрольного файла незавершённой загрузки aria2
    if [ -s "${path}" ] && [ ! -f "${path}.aria2" ]; then
        log "${tag} Файл уже существует, пропускаю: ${subdir}/${name}"
        return 0
    fi

    log "${tag} Начинаю загрузку: ${subdir}/${name}"
    # --summary-interval=10: строка прогресса (процент, скорость) каждые 10 с.
    # --console-log-level=warn снят: прогресс виден в логе контейнера. Служебные строки aria2
    # (в т.ч. адреса перенаправлений) тоже попадают туда; в GET /logs воркера они не идут.
    # Заголовок Authorization идёт во входном файле aria2 (process substitution): токен не
    # попадает в аргументы процесса и не пишется на диск (SPEC 0.1, п. 3). Проверено локально
    # aria2c 1.37 на заглушке сервера; на живой карте UNVERIFIED.
    aria2c \
        -x 16 -s 16 -k 1M -c --summary-interval=10 \
        -i <(printf '%s\n' "${url}" "  header=Authorization: Bearer ${HF_TOKEN}" \
             "  dir=${target_dir}" "  out=${name}")
    rc=$?

    if [ "${rc}" -ne 0 ] || [ ! -s "${path}" ] || [ -f "${path}.aria2" ]; then
        log "${tag} ОШИБКА: не удалось загрузить файл ${subdir}/${name} (код ${rc})."
        exit 1
    fi
    local size
    size=$(stat -c %s "${path}" 2>/dev/null || echo "?")
    log "${tag} Файл успешно загружен: ${subdir}/${name} (${size} байт)"
}

log "Запуск контейнера воркера. Уровень логов: ${GEN_LOG_LEVEL:-INFO}."

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
# Вывод ComfyUI (stdout и stderr) идёт и в лог контейнера, и в файл, который читает воркер (GET /logs).
# ComfyUI по умолчанию пишет логи в stderr; --verbose DEBUG включает уровень DEBUG (в нём сообщения
# «Unloading ...» и «freed ...»). Включается только при GEN_LOG_LEVEL=DEBUG.
COMFY_LOG_PATH="${COMFY_LOG_PATH:-/workspace/comfyui.log}"
COMFY_VERBOSE_ARGS=()
if [ "${GEN_LOG_LEVEL:-INFO}" = "DEBUG" ]; then
    COMFY_VERBOSE_ARGS=(--verbose DEBUG)
fi
log "Вывод ComfyUI пишется в файл ${COMFY_LOG_PATH} и в лог контейнера."
python "${COMFY_DIR}/main.py" --listen 127.0.0.1 --port 8188 "${COMFY_VERBOSE_ARGS[@]}" \
    > >(tee -a "${COMFY_LOG_PATH}") 2>&1 &
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
    log "Проверка готовности ComfyUI (${COMFY_READY_URL}): НЕ ПРОЙДЕНА за ${COMFY_TIMEOUT_SEC} сек."
    log "ОШИБКА: ComfyUI не стал готов за ${COMFY_TIMEOUT_SEC} сек."
    kill -TERM "${COMFY_PID}" 2>/dev/null || true
    exit 1
fi
log "Проверка готовности ComfyUI (${COMFY_READY_URL}): пройдена, прошло ${ELAPSED} сек."
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
