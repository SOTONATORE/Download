#!/usr/bin/env bash
# Generate/worker/setup.sh

set -u

# Папка ComfyUI приходит от bootstrap.sh (COMFY_DIR: найденный там путь, шаг 4). Жёсткого пути и поиска здесь нет.
COMFY_DIR="${COMFY_DIR:-}"
if [ -z "${COMFY_DIR}" ]; then
    echo "[setup] ОШИБКА: COMFY_DIR не задан. Путь ComfyUI выдаёт bootstrap.sh (шаг 4)."
    exit 1
fi
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$(dirname "${REPO_DIR}")${PYTHONPATH:+:${PYTHONPATH}}"
COMFY_MODELS_DIR="${COMFY_DIR}/models"
HF_BASE_URL="https://huggingface.co/Lightricks/LTX-2.5/resolve/main"

# ComfyUI запускает bootstrap.sh (шаг 4б), порт 18188 задаётся COMFYUI_ARGS в env карты (SPEC 8.5). setup.sh только ждёт его на порту 18188 (стоковый путь образа через supervisor на Vast в ssh-режиме не работает).
COMFY_READY_URL="http://127.0.0.1:18188/system_stats"
COMFY_TIMEOUT_SEC=180
COMFY_POLL_SEC=2

# Файл готовности для GET /health сервера воркера (SPEC 8.1): создаётся после ответа ComfyUI.
WORKER_READY_FLAG="${WORKER_READY_FLAG:-/opt/worker/worker_ready}"
# Файл причины для GET /health: создаётся, если веса не загрузились (строка «weights_failed»). Имя = флаг + «.failed».
WORKER_FAILED_FLAG="${WORKER_READY_FLAG}.failed"

# Лог setup.sh и aria2 (SPEC 8.1): stderr сервера воркера (трассировки падения) пишется сюда.
WORKER_BOOTSTRAP_LOG="${WORKER_BOOTSTRAP_LOG:-/opt/worker/bootstrap.log}"

# COMFY_TEMPLATE_LOG_PATH (SPEC 8.5): значение приходит из окружения карты и здесь не задаётся.
# Если его нет, сервер пишет одну строку «не задан» и ComfyUI не читает; путь не угадывается.
# Этот скрипт не пишет вывод ComfyUI через tee. ComfyUI запускает bootstrap.sh (шаг 4б), setup.sh только ждёт его на порту 18188.

SERVER_PID=""

# Счётчик файлов модели для строк лога: «[1/5] ...»
MODEL_FILES_TOTAL=5
MODEL_FILE_INDEX=0
MODEL_PATHS=()
# 1, если какая-либо загрузка весов не удалась: сервер воркера остаётся запущен, ComfyUI не ждём
WEIGHTS_FAILED=0

log() {
    echo "[setup] $*"
}

stop_processes() {
    log "Получен сигнал завершения, останавливаю процессы..."
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    log "Все процессы остановлены."
    exit 0
}

trap stop_processes SIGTERM SIGINT

# Любой выход из скрипта (ComfyUI не готов, сбой запуска): сервер воркера не остаётся без setup.
# При отказе весов скрипт сюда не выходит: сервер продолжает отвечать на /health (см. ниже).
stop_server_on_exit() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
}

trap stop_server_on_exit EXIT

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

    # после первого отказа остальные файлы не качаем: карта всё равно будет снята
    if [ "${WEIGHTS_FAILED}" -eq 1 ]; then
        log "${tag} Пропущено, предыдущая загрузка весов не удалась: ${subdir}/${name}"
        return 1
    fi

    MODEL_PATHS+=("${path}")
    mkdir -p "${target_dir}"

    # Файл считается готовым, только если он непустой и нет контрольного файла незавершённой загрузки aria2
    if [ -s "${path}" ] && [ ! -f "${path}.aria2" ]; then
        log "${tag} Файл уже существует, пропускаю: ${subdir}/${name}"
        return 0
    fi

    log "${tag} Начинаю загрузку: ${subdir}/${name}"
    # --summary-interval=10: строка прогресса (процент, скорость) каждые 10 с.
    # --console-log-level=warn снят: прогресс виден в логе. Служебные строки aria2 (в т.ч. адреса
    # перенаправлений) идут в bootstrap.log и через его хвост в GET /logs; query-часть адресов
    # там заменяется на «?***» (server.py, strip_url_queries).
    # Заголовок Authorization идёт во входном файле aria2 (process substitution): токен не
    # попадает в аргументы процесса и не пишется на диск (SPEC 0.1, п. 3). Проверено локально
    # aria2c 1.37 на заглушке сервера; на живой карте UNVERIFIED.
    aria2c \
        -x 16 -s 16 -k 1M -c --summary-interval=10 \
        -i <(printf '%s\n' "${url}" "  header=Authorization: Bearer ${HF_TOKEN}" \
             "  dir=${target_dir}" "  out=${name}")
    rc=$?

    if [ "${rc}" -ne 0 ] || [ ! -s "${path}" ] || [ -f "${path}.aria2" ]; then
        log "${tag} ОШИБКА: не удалось загрузить файл ${subdir}/${name} (код aria2 ${rc})."
        WEIGHTS_FAILED=1
        return 1
    fi
    local size
    size=$(stat -c %s "${path}" 2>/dev/null || echo "?")
    log "${tag} Файл успешно загружен: ${subdir}/${name} (${size} байт)"
}

log "Запуск контейнера воркера. Уровень логов: ${GEN_LOG_LEVEL:-INFO}."

# Флаг от прошлого запуска не должен открыть /health до готовности (SPEC 8.1)
rm -f "${WORKER_READY_FLAG}" "${WORKER_FAILED_FLAG}"
log "Флаг готовности сброшен; до создания флага GET /health отвечает ok=false."

# --- 1. Проверка токена и инструментов ---
if [ -z "${HF_TOKEN:-}" ]; then
    log "ОШИБКА: Не задана переменная окружения HF_TOKEN. Репозиторий Lightricks/LTX-2.5 закрыт (gated), скачивание весов невозможно."
    exit 1
fi

if ! command -v aria2c >/dev/null 2>&1; then
    log "ОШИБКА: aria2c недоступен, загрузка весов модели невозможна."
    exit 1
fi

# --- 2. Сервер воркера: стартует до загрузки весов (SPEC 8.1). Пока нет файла готовности, /health отвечает ok=false ---
log "Запускаю сервер воркера (до загрузки весов)..."
# stdout сервера не направляем в bootstrap.log: его строки уже идут в буфер GET /logs, и файл дал бы дубли.
# stderr (трассировки падения) пишется в bootstrap.log (D1), его хвост тоже попадает в GET /logs.
python3 "${REPO_DIR}/worker/server.py" >/dev/null 2>>"${WORKER_BOOTSTRAP_LOG}" &
SERVER_PID=$!
log "Сервер воркера запущен, PID: ${SERVER_PID}"

# --- 3. Загрузка модульных файлов модели LTX-2.5 ---
download_model "diffusion_models" "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
download_model "text_encoders" "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
download_model "vae" "ltx-2.5-video-vae-bf16.safetensors"
download_model "vae" "ltx-2.5-audio-vae-bf16.safetensors"
download_model "latent_upscale_models" "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
# --- 3б. Отказ загрузки весов: сервер остаётся, GET /health отвечает ok=false и reason=weights_failed ---
if [ "${WEIGHTS_FAILED}" -eq 1 ]; then
    if ! printf 'weights_failed\n' > "${WORKER_FAILED_FLAG}"; then
        log "ОШИБКА: не удалось записать файл причины отказа."
    fi
    log "ОШИБКА: веса модели не загружены. Сервер воркера остаётся запущен, файл готовности не создан, ожидание ComfyUI пропущено."
    wait "${SERVER_PID}"
    EXIT_CODE=$?
    log "Сервер воркера завершился (код ${EXIT_CODE})."
    exit "${EXIT_CODE}"
fi
log "Все файлы модели LTX-2.5 на месте."

# --- 4. Ожидание ComfyUI на порту 18188 ---
# Строка для разбора времени: все ли веса на месте в момент, когда ComfyUI уже может быть запущен.
# Открытый вопрос (SPEC 8.5): поднимается ли ComfyUI раньше весов и видит ли он их. Первый прогон покажет.
PRESENT=0
for model_path in "${MODEL_PATHS[@]}"; do
    if [ -s "${model_path}" ] && [ ! -f "${model_path}.aria2" ]; then
        PRESENT=$((PRESENT + 1))
    fi
done
ALL_PRESENT="нет"
if [ "${PRESENT}" -eq "${MODEL_FILES_TOTAL}" ]; then
    ALL_PRESENT="да"
fi
log "Веса перед ожиданием ComfyUI: все файлы на месте: ${ALL_PRESENT} (${PRESENT} из ${MODEL_FILES_TOTAL})."

log "Ожидаю готовности ComfyUI на порту 18188 (до ${COMFY_TIMEOUT_SEC} сек)..."
ELAPSED=0
READY=0
while [ "${ELAPSED}" -lt "${COMFY_TIMEOUT_SEC}" ]; do
    if curl -sf -o /dev/null "${COMFY_READY_URL}"; then
        READY=1
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "ОШИБКА: процесс сервера воркера неожиданно завершился."
        exit 1
    fi
    sleep "${COMFY_POLL_SEC}"
    ELAPSED=$((ELAPSED + COMFY_POLL_SEC))
done

if [ "${READY}" -ne 1 ]; then
    log "Проверка готовности ComfyUI (${COMFY_READY_URL}): НЕ ПРОЙДЕНА за ${COMFY_TIMEOUT_SEC} сек."
    log "ОШИБКА: ComfyUI на порту 18188 не стал готов за ${COMFY_TIMEOUT_SEC} сек."
    exit 1
fi
log "Проверка готовности ComfyUI (${COMFY_READY_URL}): пройдена, прошло ${ELAPSED} сек."
log "ComfyUI готов к работе (прошло ${ELAPSED} сек)."

# Веса и ComfyUI готовы: создаём файл, по которому GET /health вернёт ok=true (SPEC 8.1)
if ! touch "${WORKER_READY_FLAG}"; then
    log "ОШИБКА: не удалось создать файл готовности воркера (${WORKER_READY_FLAG})."
    exit 1
fi
log "Воркер готов: файл готовности создан, GET /health может вернуть ok=true."

# --- 5. Работа до завершения сервера воркера ---
wait "${SERVER_PID}"
EXIT_CODE=$?
log "Сервер воркера завершился (код ${EXIT_CODE})."
exit "${EXIT_CODE}"
