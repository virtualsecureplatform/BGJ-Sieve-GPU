#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
FILTER_TASK_VECS="${FILTER_TASK_VECS:-524288}"
BGJ2_REDUCER_THREADS="${BGJ2_REDUCER_THREADS:-48}"

if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to submit tracked, uncommitted source changes" >&2
    exit 2
fi
if [[ ! "$FILTER_TASK_VECS" =~ ^[0-9]+$ ]] || (( FILTER_TASK_VECS < 65536 || (FILTER_TASK_VECS & (FILTER_TASK_VECS - 1)) != 0 )); then
    echo "FILTER_TASK_VECS must be a power of two >= 65536" >&2
    exit 2
fi
if [[ ! "$BGJ2_REDUCER_THREADS" =~ ^[0-9]+$ ]] || (( BGJ2_REDUCER_THREADS < 4 || BGJ2_REDUCER_THREADS > 64 || BGJ2_REDUCER_THREADS % 4 != 0 )); then
    echo "BGJ2_REDUCER_THREADS must be a multiple of 4 in [4, 64]" >&2
    exit 2
fi

cd "$SOURCE"
build_job="$({ sbatch --parsable \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT},FILTER_TASK_VECS=${FILTER_TASK_VECS}" \
    build_svp138_a100x4.sbatch; } | cut -d';' -f1)"
run_job="$({ sbatch --parsable \
    --dependency="afterok:${build_job}" \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT},FILTER_TASK_VECS=${FILTER_TASK_VECS},BGJ2_REDUCER_THREADS=${BGJ2_REDUCER_THREADS},BUILD=0" \
    run_svp138_a100x4.sbatch; } | cut -d';' -f1)"

echo "build_job=${build_job}"
echo "run_job=${run_job} dependency=afterok:${build_job}"
