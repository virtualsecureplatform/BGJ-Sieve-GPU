#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
LOG_ROOT="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/logs/svp158-search-bkz70"

if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to submit tracked, uncommitted source changes" >&2
    exit 2
fi

mkdir -p "$LOG_ROOT"
cd "$SOURCE"
prep="$(sbatch --parsable \
    --output="${LOG_ROOT}/prep_%j.log" \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT}" \
    prepare_svp158_search_bkz70_a100x4.sbatch | cut -d';' -f1)"
run="$(sbatch --parsable \
    --output="${LOG_ROOT}/run_%A_%a.log" \
    --dependency="afterok:${prep}" \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT}" \
    run_svp158_search_bkz70_a100x4.sbatch | cut -d';' -f1)"

echo "prepare_job=${prep} seeds=1-8 concurrent_processes=8 allocation=cpu:8"
echo "run_array=${run} seeds=1-8 concurrency=2 dependency=afterok:${prep}"
echo "preprocess=PotLLL-heuristic-mpfr256-then-BKZ70x8 MLD=120 TSD=139 strict_target_norm2=9799277"
echo "logs=${LOG_ROOT}"
