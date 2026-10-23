#!/usr/bin/env bash
# Pilot start for ComfyUI on a Vast instance (ssh mode).
# Starts ComfyUI directly with /venv/main; does not use the stock supervisor.
# Prints READY on success (exit 0) or FAILED: <step> on failure (exit 1).
# Does not install packages and does not print the environment.
set -u

LOG=/opt/pilot/comfyui.log
PIDFILE=/opt/pilot/comfyui.pid
PORT_URL=http://127.0.0.1:18188/system_stats
VENV_PY=/venv/main/bin/python
INTERNAL_DIR=/opt/workspace-internal/ComfyUI
PILOT_DIR=/opt/pilot/ComfyUI
COMFY_REPO=https://github.com/comfyanonymous/ComfyUI.git
COMFY_COMMIT=98cc0bf84b7cb1383654c5b402e7612d875d6d22
WAIT_TOTAL=300
WAIT_STEP=5

mkdir -p /opt/pilot 2>/dev/null
touch "$LOG" 2>/dev/null

# say STEP MESSAGE: print to stdout and append to the log
say() {
    local line="[$1] $2"
    printf '%s\n' "$line"
    printf '%s\n' "$line" >> "$LOG" 2>/dev/null
}

# tail_log: last 20 lines of the log, to stdout
tail_log() {
    tail -n 20 "$LOG" 2>/dev/null
}

# pid_alive PID: true if the process exists and is not a zombie
pid_alive() {
    local state
    [ -n "${1:-}" ] || return 1
    kill -0 "$1" 2>/dev/null || return 1
    state=$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null)
    [ -n "$state" ] && [ "$state" != "Z" ]
}

# ---- Step 1: venv ----
if [ ! -x "$VENV_PY" ]; then
    say venv "FAILED: venv (missing $VENV_PY)"
    exit 1
fi
say venv "ok"

# ---- Step 2: already_up ----
if curl -fsS --max-time 5 "$PORT_URL" >/dev/null 2>&1; then
    say already_up "ComfyUI already answers on 18188"
    say already_up "READY"
    exit 0
fi
say already_up "not running"

# ---- Step 3: source ----
if [ -f "$INTERNAL_DIR/main.py" ]; then
    DIR="$INTERNAL_DIR"
    say source "using $DIR"
else
    DIR="$PILOT_DIR"
    if [ ! -f "$DIR/main.py" ]; then
        say source "cloning ComfyUI into $DIR"
        export GIT_TERMINAL_PROMPT=0
        if ! git clone "$COMFY_REPO" "$DIR" >>"$LOG" 2>&1 \
           || ! git -C "$DIR" checkout "$COMFY_COMMIT" >>"$LOG" 2>&1; then
            say source "FAILED: clone"
            tail_log
            exit 1
        fi
    fi
    say source "using $DIR"
fi

# ---- Step 4: start ----
read -r -a ARGS <<< "${COMFYUI_ARGS:-}"
if [ "${#ARGS[@]}" -eq 0 ]; then
    say start "FAILED: args (COMFYUI_ARGS is empty)"
    exit 1
fi
if ! cd "$DIR"; then
    say start "FAILED: args (cannot enter $DIR)"
    exit 1
fi
nohup "$VENV_PY" main.py "${ARGS[@]}" >> "$LOG" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PIDFILE" 2>/dev/null
say start "started pid $PID"

# ---- Step 5: wait ----
elapsed=0
while [ "$elapsed" -lt "$WAIT_TOTAL" ]; do
    if curl -fsS --max-time 4 "$PORT_URL" >/dev/null 2>&1; then
        say wait "READY"
        exit 0
    fi
    if ! pid_alive "$PID"; then
        say wait "FAILED: exited (pid $PID is gone)"
        tail_log
        exit 1
    fi
    sleep "$WAIT_STEP"
    elapsed=$((elapsed + WAIT_STEP))
done
say wait "FAILED: timeout after ${WAIT_TOTAL}s"
tail_log
exit 1
