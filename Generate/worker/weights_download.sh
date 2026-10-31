#!/usr/bin/env bash
# Generate/worker/weights_download.sh
# Общая загрузка весов модели (шаг W2): используется setup.sh (первый запуск) и switch_profile.sh (переключение).
# Файл только объявляет функции: при загрузке (source) он не выполняет ни одной команды.
#
# Переменные, которые вызывающий скрипт задаёт ДО source этого файла:
#   COMFY_MODELS_DIR      папка models/ ComfyUI (локальные пути файлов)
#   HF_BASE_URL           базовый адрес HF по умолчанию (используется, если строка списка не задала свой base)
#   HF_TOKEN              токен HF (пустой допустим; токен уходит только в заголовок Authorization)
#   WORKER_BOOTSTRAP_LOG  файл лога, куда пишутся строки aria2 и curl
#   ARIA2_MODE            aria2 или curl (выбирает загрузчик)
#   COMFY_READY_URL       адрес проверки ComfyUI (читает comfy_answers)
#   COMFY_UP_EARLY        0 или 1 (wait_downloads ставит 1, если ComfyUI ответил до конца загрузок)
#   MODEL_FILES_TOTAL     число файлов для префикса [N/M]
#   MODEL_FILE_INDEX      счётчик файлов (0 в начале)
#   MODEL_PATHS           массив путей файлов (пустой массив в начале)
#   DL_PIDS               массив PID фоновых загрузок (пустой массив в начале)
# Также нужна функция log() вызывающего скрипта.
#
# COMFY_TIMEOUT_SEC и COMFY_POLL_SEC здесь не читаются: ожидание ComfyUI после загрузки остаётся в setup.sh.
# Перезапуск ComfyUI (restart_comfy) тоже остаётся в setup.sh.

# Загрузка одного файла модели в фоне:
# start_download <локальный подкаталог в models/> <имя файла> [путь в репозитории HF] [базовый URL HF]
# Без третьего и четвёртого аргумента URL строится как раньше: HF_BASE_URL/<подкаталог>/<имя файла> (LTX).
start_download() {
    MODEL_FILE_INDEX=$((MODEL_FILE_INDEX + 1))
    MODEL_PATHS+=("${COMFY_MODELS_DIR}/$1/$2")
    download_model "${MODEL_FILE_INDEX}" "$1" "$2" "${3:-}" "${4:-}" &
    DL_PIDS+=("$!")
}

# Тело загрузки одного файла (выполняется в фоновом процессе):
# download_model <номер> <локальный подкаталог> <имя файла> [путь в репозитории HF] [базовый URL HF]
# Локальная папка всегда ${COMFY_MODELS_DIR}/<локальный подкаталог>. Если задан путь в репозитории,
# URL = <базовый URL>/<путь в репозитории>; иначе URL = <базовый URL>/<подкаталог>/<имя файла>.
# Базовый URL по умолчанию — HF_BASE_URL (LTX), поэтому вызовы LTX без пятого аргумента не меняются.
download_model() {
    local idx="$1"
    local subdir="$2"
    local name="$3"
    local repo_path="${4:-}"
    local base="${5:-${HF_BASE_URL}}"
    local target_dir="${COMFY_MODELS_DIR}/${subdir}"
    local path="${target_dir}/${name}"
    local url
    if [ -n "${repo_path}" ]; then
        url="${base}/${repo_path}"
    else
        url="${base}/${subdir}/${name}"
    fi
    local tag="[${idx}/${MODEL_FILES_TOTAL}]"
    local tool="aria2"
    local rc
    local size

    mkdir -p "${target_dir}"

    # Файл считается готовым, только если он непустой и нет контрольного файла незавершённой загрузки aria2
    if [ -s "${path}" ] && [ ! -f "${path}.aria2" ]; then
        log "${tag} Файл уже существует, пропускаю: ${subdir}/${name}"
        return 0
    fi

    log "${tag} Начинаю загрузку: ${subdir}/${name}"
    if [ "${ARIA2_MODE}" = "curl" ]; then
        tool="curl"
        # У curl нет контрольного файла aria2: на время загрузки ставим свой «.aria2», чтобы прерванный файл
        # не принимался за готовый. Заголовок Authorization читается из process substitution (токен не в argv).
        : > "${path}.aria2"
        # Вывод curl (ошибки) идёт в bootstrap.log с префиксом «[file=<имя>] » (sed -u: построчно).
        curl -sS -L --fail --retry 3 -C - -o "${path}" \
            -H @<(printf 'Authorization: Bearer %s\n' "${HF_TOKEN}") "${url}" 2>&1 \
            | sed -u "s|^|[file=${name}] |" >>"${WORKER_BOOTSTRAP_LOG}"
        rc="${PIPESTATUS[0]}"
        if [ "${rc}" -eq 0 ]; then
            rm -f "${path}.aria2"
        fi
    else
        # --summary-interval=10: строка прогресса (процент, скорость) каждые 10 с.
        # --console-log-level=warn снят: прогресс виден в логе. Служебные строки aria2 (в т.ч. адреса
        # перенаправлений) идут в bootstrap.log и через его хвост в GET /logs; query-часть адресов
        # там заменяется на «?***» (server.py, strip_url_queries).
        # Заголовок Authorization идёт во входном файле aria2 (process substitution): токен не
        # попадает в аргументы процесса и не пишется на диск (SPEC 0.1, п. 3). Проверено локально
        # aria2c 1.37 на заглушке сервера; на живой карте UNVERIFIED.
        # Каждая строка вывода aria2c получает префикс «[file=<имя>] » (sed -u: построчно), строки идут
        # в bootstrap.log. Так параллельные загрузки различимы в мониторе (core/weights_log.py).
        aria2c \
            -x 16 -s 16 -k 1M -c --summary-interval=10 \
            -i <(printf '%s\n' "${url}" "  header=Authorization: Bearer ${HF_TOKEN}" \
                 "  dir=${target_dir}" "  out=${name}") 2>&1 \
            | sed -u "s|^|[file=${name}] |" >>"${WORKER_BOOTSTRAP_LOG}"
        rc="${PIPESTATUS[0]}"
    fi

    if [ "${rc}" -ne 0 ] || [ ! -s "${path}" ] || [ -f "${path}.aria2" ]; then
        log "${tag} ОШИБКА: не удалось загрузить файл ${subdir}/${name} (код ${tool} ${rc})."
        return 1
    fi
    size=$(stat -c %s "${path}" 2>/dev/null || echo "?")
    log "${tag} Файл успешно загружен: ${subdir}/${name} (${size} байт)"
}

# ComfyUI отвечает 200 на 18188 (короткая проверка, без ожидания)
comfy_answers() {
    curl -sf -m 5 -o /dev/null "${COMFY_READY_URL}"
}

# Ждёт все фоновые загрузки. Пока они идут, раз в секунду проверяет 18188. COMFY_UP_EARLY=1 ставится,
# только если ответ 200 пришёл, когда последняя загрузка ещё не завершилась.
wait_downloads() {
    local seen pid alive
    while :; do
        seen=0
        if [ "${COMFY_UP_EARLY}" -eq 0 ] && comfy_answers; then
            seen=1
        fi
        alive=0
        for pid in "${DL_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                alive=1
                break
            fi
        done
        if [ "${alive}" -eq 0 ]; then
            break
        fi
        if [ "${seen}" -eq 1 ]; then
            COMFY_UP_EARLY=1
        fi
        sleep 1
    done
}

