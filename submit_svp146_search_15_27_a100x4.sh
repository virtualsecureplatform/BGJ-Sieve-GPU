#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
LOG_ROOT="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/logs/svp146-search"

if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to submit tracked, uncommitted source changes" >&2
    exit 2
fi

mkdir -p "$LOG_ROOT"
cd "$SOURCE"
prep="$(sbatch --parsable \
    --output="${LOG_ROOT}/prep_%A_%a.log" \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT}" \
    prepare_svp146_search_a100x4.sbatch | cut -d';' -f1)"
run="$(sbatch --parsable \
    --output="${LOG_ROOT}/run_%A_%a.log" \
    --dependency="aftercorr:${prep}" \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT}" \
    run_svp146_search_a100x4.sbatch | cut -d';' -f1)"

echo "prepare_array=${prep} seeds=15-27 concurrency=2"
echo "run_array=${run} seeds=15-27 concurrency=2 dependency=aftercorr:${prep}"
echo "preprocess=LLL-heuristic-mpfr256-then-PotLLLBKZ64x8 MLD=120 TSD=136 target_norm2=9142369"
echo "logs=${LOG_ROOT}"
