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
# Профиль модели и его список весов лежат в model_profiles/models/<MODEL_PROFILE>/ (weights.json, profile.yaml).
# Читает их core/weights_plan.py: новая модель = новая папка, без правок этого файла. По умолчанию ltx25.
PROFILE_DIR="${REPO_DIR}/model_profiles/models/${MODEL_PROFILE:-ltx25}"
META="$(python3 "${REPO_DIR}/core/weights_plan.py" meta "${PROFILE_DIR}")" || { echo "[setup] ОШИБКА: профиль модели не читается: ${PROFILE_DIR} (код 1)."; exit 1; }
# Проверка описания весов профиля (META из weights_plan.py meta): ровно три строки, значения по форме.
# Пустой или битый вывод — остановка с понятной ошибкой, а не тихий пропуск загрузки.
META_OK=1
if [ "$(printf '%s\n' "${META}" | wc -l)" -ne 3 ]; then META_OK=0; fi
if [ "${META_OK}" = 1 ]; then
    META_L1="$(printf '%s\n' "${META}" | sed -n 1p)"
    META_L2="$(printf '%s\n' "${META}" | sed -n 2p)"
    META_L3="$(printf '%s\n' "${META}" | sed -n 3p)"
    [[ "${META_L1}" =~ ^type=(video|image)$ ]] || META_OK=0
    [[ "${META_L2}" =~ ^needs_token=[01]$ ]] || META_OK=0
    [[ "${META_L3}" =~ ^files_total=[1-9][0-9]*$ ]] || META_OK=0
fi
if [ "${META_OK}" != 1 ]; then
    echo "[setup] ОШИБКА: описание весов профиля некорректно: ${PROFILE_DIR} (код 1)."
    exit 1
fi
MODEL_TYPE="$(printf '%s\n' "${META}" | sed -n 's/^type=//p')"
NEED_TOKEN="$(printf '%s\n' "${META}" | sed -n 's/^needs_token=//p')"
MODEL_FILES_TOTAL="$(printf '%s\n' "${META}" | sed -n 's/^files_total=//p')"
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

# Счётчик файлов модели для строк лога «[1/N] ...»: MODEL_FILES_TOTAL взят из weights.json выше.
MODEL_FILE_INDEX=0
MODEL_PATHS=()
# 1, если какая-либо загрузка весов не удалась: сервер воркера остаётся запущен, ComfyUI не ждём
WEIGHTS_FAILED=0
# Загрузки идут параллельно: PID фоновых процессов (по порядку вызовов start_download)
DL_PIDS=()
# Загрузчик: aria2 (aria2c есть или установлен по запросу) или curl (запасной путь)
ARIA2_MODE="aria2"
# 1, если ComfyUI отвечал 200 до того, как закончилась последняя загрузка: тогда он мог не увидеть все файлы
COMFY_UP_EARLY=0

log() {
    echo "[setup] $*"
}

# Переменные, которые нужны общему загрузчику весов (weights_download.sh), задаются до его source.
# Остальные (COMFY_MODELS_DIR, HF_BASE_URL, MODEL_FILES_TOTAL и т.д.) уже заданы выше.
HF_TOKEN="${HF_TOKEN:-}"

# Загрузка весов (start_download, download_model, wait_downloads, comfy_answers) вынесена в общий файл (шаг W2).
# Тот же файл использует switch_profile.sh при переключении профиля. Файл только объявляет функции.
source "${REPO_DIR}/worker/weights_download.sh"

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

# aria2c: если его нет, ставим через apt-get (каждая команда до 120 с, вывод в bootstrap.log).
# Если и это не помогло, загрузка пойдёт через curl (строка «aria2c недоступен» пишется один раз).
ensure_downloader() {
    if command -v aria2c >/dev/null 2>&1; then
        ARIA2_MODE="aria2"
        log "Загрузчик весов: aria2c (уже установлен)."
        return 0
    fi
    DEBIAN_FRONTEND=noninteractive timeout 120 apt-get update >>"${WORKER_BOOTSTRAP_LOG}" 2>&1 || true
    DEBIAN_FRONTEND=noninteractive timeout 120 apt-get install -y aria2 >>"${WORKER_BOOTSTRAP_LOG}" 2>&1 || true
    if command -v aria2c >/dev/null 2>&1; then
        ARIA2_MODE="aria2"
        log "Загрузчик весов: aria2c установлен по запросу."
    else
        ARIA2_MODE="curl"
        log "aria2c недоступен, загрузка через curl"
    fi
}

# Перезапуск ComfyUI один раз, как шаг 4б bootstrap.sh: останавливаем PID из COMFY_PID_FILE, запускаем заново.
restart_comfy() {
    local old_pid new_pid waited rc
    local -a ARGS=()

    if [ -z "${COMFY_PID_FILE:-}" ] || [ -z "${COMFY_LOG_FILE:-}" ] || [ -z "${COMFY_PYTHON:-}" ]; then
        log "Перезапуск ComfyUI: ОШИБКА: не заданы COMFY_PID_FILE, COMFY_LOG_FILE или COMFY_PYTHON, ComfyUI не перезапущен."
        return 1
    fi
    old_pid="$(cat "${COMFY_PID_FILE}" 2>/dev/null || true)"
    log "Перезапуск ComfyUI: ComfyUI отвечал до конца загрузки весов, останавливаю PID ${old_pid:-неизвестен} и запускаю заново."
    if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
        kill "${old_pid}" 2>/dev/null || true
        waited=0
        while kill -0 "${old_pid}" 2>/dev/null && [ "${waited}" -lt 20 ]; do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "${old_pid}" 2>/dev/null; then
            log "Перезапуск ComfyUI: PID ${old_pid} не завершился за 20 сек, посылаю kill -9."
            kill -9 "${old_pid}" 2>/dev/null || true
        fi
    fi

    # COMFYUI_ARGS делится по пробелам без eval; значения не печатаются
    read -r -a ARGS <<< "${COMFYUI_ARGS:-}"
    if [ "${#ARGS[@]}" -eq 0 ]; then
        log "Перезапуск ComfyUI: ОШИБКА: COMFYUI_ARGS пуст, ComfyUI не запущен заново."
        return 1
    fi
    # Тот же способ, что в bootstrap.sh шаг 4б: cd в отдельном процессе шелла, вывод дописывается в лог
    (
        cd "${COMFY_DIR}" || exit 1
        nohup "${COMFY_PYTHON}" main.py "${ARGS[@]}" >> "${COMFY_LOG_FILE}" 2>&1 &
        printf '%s\n' "$!" > "${COMFY_PID_FILE}"
    )
    rc=$?
    if [ "${rc}" -ne 0 ]; then
        log "Перезапуск ComfyUI: ОШИБКА: не удалось перейти в ${COMFY_DIR} или запустить ComfyUI (код ${rc})."
        return 1
    fi
    new_pid="$(cat "${COMFY_PID_FILE}" 2>/dev/null || echo '?')"
    log "Перезапуск ComfyUI выполнен: новый PID ${new_pid}, вывод в ${COMFY_LOG_FILE}."
}

log "Запуск контейнера воркера. Уровень логов: ${GEN_LOG_LEVEL:-INFO}."

# Флаг от прошлого запуска не должен открыть /health до готовности (SPEC 8.1)
rm -f "${WORKER_READY_FLAG}" "${WORKER_FAILED_FLAG}"
log "Флаг готовности сброшен; до создания флага GET /health отвечает ok=false."

# --- 1. Проверка токена и инструментов ---
# Токен нужен, если в weights.json профиля стоит needs_token: true (LTX: репозиторий Lightricks закрыт). Z-Image открыт.
if [ "${NEED_TOKEN}" = "1" ] && [ -z "${HF_TOKEN:-}" ]; then
    log "ОШИБКА: Не задана переменная окружения HF_TOKEN. Репозиторий Lightricks/LTX-2.5 закрыт (gated), скачивание весов невозможно."
    exit 1
fi

ensure_downloader

# --- 2. Сервер воркера: стартует до загрузки весов (SPEC 8.1). Пока нет файла готовности, /health отвечает ok=false ---
log "Запускаю сервер воркера (до загрузки весов)..."
# stdout сервера не направляем в bootstrap.log: его строки уже идут в буфер GET /logs, и файл дал бы дубли.
# stderr (трассировки падения) пишется в bootstrap.log (D1), его хвост тоже попадает в GET /logs.
# Тип профиля для воркера (comfy_client.py читает WORKER_PROFILE_TYPE) берётся из profile.yaml через weights_plan.py meta.
WORKER_PROFILE_TYPE="${MODEL_TYPE}"
export WORKER_PROFILE_TYPE
log "Тип профиля для воркера: ${WORKER_PROFILE_TYPE}."
python3 "${REPO_DIR}/worker/server.py" >/dev/null 2>>"${WORKER_BOOTSTRAP_LOG}" &
SERVER_PID=$!
log "Сервер воркера запущен, PID: ${SERVER_PID}"

# ComfyUI из шага 4б bootstrap.sh может уже отвечать. Если нет, ничего не делаем: он грузится параллельно с весами.
if comfy_answers; then
    COMFY_UP_EARLY=1
    log "ComfyUI уже отвечает до загрузки весов."
else
    log "ComfyUI пока не отвечает: он запущен на шаге 4б bootstrap и грузится параллельно с весами."
fi

# --- 3. Загрузка файлов модели: список из weights.json профиля (через weights_plan.py files), файлы параллельно ---
log "Запускаю параллельную загрузку весов: ${MODEL_FILES_TOTAL} файлов."
# Строки: подкаталог, имя, путь в репо HF (пусто для LTX), базовый URL, разделитель TAB.
# Ошибка чтения списка — выход с кодом 1. Поля разбираются по TAB вручную: read с IFS=TAB схлопнул бы пустое поле.
ROWS="$(python3 "${REPO_DIR}/core/weights_plan.py" files "${PROFILE_DIR}")" || exit 1
# Проверка списка весов (ROWS из weights_plan.py files): не пуст, число строк = MODEL_FILES_TOTAL,
# в каждой строке ровно четыре поля через TAB (три табуляции), подкаталог и имя не пустые.
ROWS_OK=1
if [ -z "${ROWS}" ]; then ROWS_OK=0; fi
if [ "${ROWS_OK}" = 1 ] && [ "$(printf '%s\n' "${ROWS}" | wc -l)" -ne "${MODEL_FILES_TOTAL}" ]; then ROWS_OK=0; fi
if [ "${ROWS_OK}" = 1 ]; then
    while IFS= read -r chk; do
        chk_tabs="$(printf '%s' "${chk}" | tr -cd '\t' | wc -c)"
        if [ "${chk_tabs}" -ne 3 ]; then ROWS_OK=0; break; fi
        chk_sub="${chk%%$'\t'*}"; chk_rest="${chk#*$'\t'}"; chk_name="${chk_rest%%$'\t'*}"
        if [ -z "${chk_sub}" ] || [ -z "${chk_name}" ]; then ROWS_OK=0; break; fi
    done <<< "${ROWS}"
fi
if [ "${ROWS_OK}" != 1 ]; then
    echo "[setup] ОШИБКА: список весов профиля некорректен: ${PROFILE_DIR} (код 1)."
    exit 1
fi
while IFS= read -r line; do
    sub="${line%%$'\t'*}"; rest="${line#*$'\t'}"
    name="${rest%%$'\t'*}"; rest="${rest#*$'\t'}"
    rp="${rest%%$'\t'*}"; base="${rest#*$'\t'}"
    start_download "${sub}" "${name}" "${rp}" "${base}"
done <<< "${ROWS}"
wait_downloads
for dl_pid in "${DL_PIDS[@]}"; do
    if ! wait "${dl_pid}"; then
        WEIGHTS_FAILED=1
    fi
done
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
if [ -z "${MODEL_PROFILE:-}" ] || [ "${MODEL_PROFILE}" = "ltx25" ]; then
    log "Все файлы модели LTX-2.5 на месте."
else
    log "Все файлы модели на месте (профиль ${MODEL_PROFILE})."
fi

# Перезапуск ComfyUI не более одного раза: только если он отвечал до конца последней загрузки (мог не увидеть все файлы)
if [ "${COMFY_UP_EARLY}" -eq 1 ]; then
    restart_comfy
else
    log "Перезапуск ComfyUI не нужен: ComfyUI не отвечал до окончания загрузки весов."
fi

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
