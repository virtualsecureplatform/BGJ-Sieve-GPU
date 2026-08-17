#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
LOG_ROOT="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/logs/svp158-seed2-bucketer-workers"
WAIT_FOR_JOB="${WAIT_FOR_JOB:-292983}"
common="SOURCE_COMMIT=${SOURCE_COMMIT},SVP_DIMENSION=158,SVP142_LATTICE_SEED=2,SVP_INSTANCE=svp158-seed2,SVP142_REPRO=${SOURCE}/reproduce/svp158-search/reproduce.py,SVP_PREPROCESS_MODE=potlll-bkz,SVP_BKZ_BETA=70,SVP_BKZ_LOOPS=8,BGJ_ENABLE_PROFILING=0"

if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to submit tracked, uncommitted source changes" >&2
    exit 2
fi
if [[ ! "$WAIT_FOR_JOB" =~ ^[0-9]+(_[0-9]+)?$ ]]; then
    echo "WAIT_FOR_JOB must be a Slurm job ID" >&2
    exit 2
fi

mkdir -p "$LOG_ROOT"
cd "$SOURCE"
build="$(sbatch --parsable \
    --job-name=svp158-buc-build \
    --output="${LOG_ROOT}/build_%j.log" \
    --dependency="afterany:${WAIT_FOR_JOB}" \
    --export="ALL,${common}" \
    prepare_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"
run="$(sbatch --parsable \
    --output="${LOG_ROOT}/run_%A_%a.log" \
    --dependency="afterok:${build}" \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT}" \
    run_svp158_seed2_bucketer_workers_a100x4.sbatch | cut -d';' -f1)"

echo "build_job=${build} dependency=afterany:${WAIT_FOR_JOB}"
echo "run_array=${run} workers=20,24 concurrency=2 dependency=afterok:${build}"
echo "variant=profiling-off-only MLD=120 TSD=139"
echo "logs=${LOG_ROOT}"
