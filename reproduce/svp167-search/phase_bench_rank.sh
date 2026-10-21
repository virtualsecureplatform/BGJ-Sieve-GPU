#!/bin/bash
# Run inside each allocated compute node; never launch diagnostics from login.
set -euo pipefail
: "${BENCH_NATIVE_WRAPPER:?}" "${MPI_BINARY:?}"
diagnostics="${3:?}.rank${SLURM_PROCID:-0}"
"${BENCH_NATIVE_WRAPPER}" "$@" &
worker=$!
trap 'kill -TERM "${worker}" 2>/dev/null || true' TERM INT

find_native() {
    local pending="${worker}" next pid depth executable
    for depth in 0 1 2 3 4; do
        next=""
        for pid in ${pending}; do
            executable="$(readlink "/proc/${pid}/exe" 2>/dev/null || true)"
            if [[ "${executable}" == "${MPI_BINARY}" ]] ||
               { [[ -n "${MPI_HOST_RUNTIME:-}" && "${executable}" == "${MPI_HOST_RUNTIME}/ld-linux-x86-64.so.2" ]] &&
                 grep -zFxq -- "${MPI_BINARY}" "/proc/${pid}/cmdline" 2>/dev/null; }; then
                echo "${pid}"
                return 0
            fi
            next+=" $(ps -o pid= --ppid "${pid}" 2>/dev/null || true)"
        done
        pending="${next}"
    done
    return 1
}

monitor() {
    local native sleeper="" next_stack=1200
    trap 'exit 0' TERM INT
    trap 'if [[ -n "${sleeper}" ]]; then kill "${sleeper}" 2>/dev/null || true; fi' EXIT
    SECONDS=0
    while kill -0 "${worker}" 2>/dev/null; do
        {
            date --iso-8601=seconds
            timeout 10 nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used \
                --format=csv,noheader || true
            native="$(find_native || true)"
            if [[ -n "${native}" ]]; then
                ps -p "${native}" -o pid,etime,time,pcpu,rss,stat,comm || true
                ps -L -p "${native}" -o tid,pcpu,stat,wchan:32,comm || true
                rg '^(VmRSS|VmHWM|Threads):' "/proc/${native}/status" || true
                # Counters allow successive samples to distinguish disk I/O
                # from busy waiting; never dump the process environment.
                if [[ -r "/proc/${native}/io" ]]; then
                    rg '^(rchar|wchar|read_bytes|write_bytes):' "/proc/${native}/io" || true
                fi
                if (( SECONDS >= next_stack )) && command -v gdb >/dev/null; then
                    # A short attach perturbs timing slightly; these runs are
                    # diagnostic, not a clean performance claim.
                    timeout --signal=INT --kill-after=5 20 gdb -q -n -batch \
                        -ex 'set pagination off' -ex "set sysroot /proc/${native}/root" \
                        -ex "attach ${native}" -ex 'thread apply all bt 8' -ex detach \
                        > "${diagnostics}.stacks-${SECONDS}.log" 2>&1 || true
                    kill -CONT "${native}" 2>/dev/null || true
                    next_stack=$((SECONDS + 1200))
                fi
            else
                echo "Native benchmark PID not found yet; launcher=${worker}"
            fi
        } >> "${diagnostics}.monitor.log" 2>&1
        sleep 60 &
        sleeper=$!
        wait "${sleeper}" || true
        sleeper=""
    done
}
monitor &
monitor_pid=$!
rc=0
wait "${worker}" || rc=$?
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
exit "${rc}"
