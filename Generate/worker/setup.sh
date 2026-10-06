#!/usr/bin/env bash
# Точка входа контейнера Vast.ai: ComfyUI (headless) -> проверка готовности -> HTTP-воркер.
set -eo pipefail

export PYTHONPATH="/workspace${PYTHONPATH:+:$PYTHONPATH}"
COMFY_DIR="${COMFY_DIR:-/workspace/ComfyUI}"
COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"
HEALTH_TIMEOUT="${COMFY_HEALTH_TIMEOUT:-180}"

COMFY_PID=""
WORKER_PID=""

log() { echo "[$(date +%H:%M:%S)] setup.sh: $*"; }

stop_pid() {
    local pid="$1" name="$2" i
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 0
    log "Останавливаю $name (PID $pid)."
    kill -TERM "$pid" 2>/dev/null || true
    for i in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.5
    done
    kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
    trap - TERM INT EXIT
    stop_pid "$WORKER_PID" "воркер"
    stop_pid "$COMFY_PID" "ComfyUI"
}

on_signal() {
    log "Получен сигнал завершения."
    cleanup
    exit 0
}
trap on_signal TERM INT
trap cleanup EXIT

log "Запуск ComfyUI."
python3 "$COMFY_DIR/main.py" --listen 127.0.0.1 --port 8188 --headless &
COMFY_PID=$!
log "ComfyUI запущен, PID $COMFY_PID."

log "Ожидание готовности ComfyUI (до ${HEALTH_TIMEOUT} с)."
start=$SECONDS
until curl -s -f -o /dev/null "$COMFY_URL/system_stats"; do
    if ! kill -0 "$COMFY_PID" 2>/dev/null; then
        log "ОШИБКА: ComfyUI завершился до готовности."
        exit 1
    fi
    if [ $((SECONDS - start)) -ge "$HEALTH_TIMEOUT" ]; then
        log "ОШИБКА: ComfyUI не ответил за ${HEALTH_TIMEOUT} с."
        exit 1
    fi
    sleep 2
done
log "ComfyUI готов ($((SECONDS - start)) с)."

log "Запуск HTTP-воркера на порту ${WORKER_HTTP_PORT:-8000}."
python3 -m Generate.worker.server &
WORKER_PID=$!
log "Воркер запущен, PID $WORKER_PID."

# Ждём завершения любого из процессов (воркер сам выходит по таймауту простоя).
rc=0
wait -n "$COMFY_PID" "$WORKER_PID" || rc=$?
log "Один из процессов завершился (код $rc), останавливаю остальные."
exit "$rc"
