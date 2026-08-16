#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
LOG_ROOT="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/logs/svp158-seed0"
common="SOURCE_COMMIT=${SOURCE_COMMIT},SVP_DIMENSION=158,SVP142_LATTICE_SEED=0,SVP_INSTANCE=svp158-seed0,SVP142_REPRO=${SOURCE}/reproduce/svp158-seed0/reproduce.py,SVP_PREPROCESS_MODE=potlll-bkz,SVP_BKZ_BETA=64,SVP_BKZ_LOOPS=8"

if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to submit tracked, uncommitted source changes" >&2
    exit 2
fi

mkdir -p "$LOG_ROOT"
cd "$SOURCE"
prep="$(sbatch --parsable --job-name=svp158-s0-potprep --time=02:00:00 \
    --output="${LOG_ROOT}/prep_%j.log" \
    --export="ALL,${common}" prepare_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"
run="$(sbatch --parsable --job-name=svp158-s0-potbkz64 --time=06:00:00 \
    --output="${LOG_ROOT}/run_%j.log" \
    --dependency="afterok:${prep}" \
    --export="ALL,${common},SVP_SIEVE_SEED=0,TARGET_SIEVING_DIM=139,MIN_LIFTING_DIM=120,SVP_TIMEOUT_SECONDS=21300" \
    run_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"

echo "prepare_job=${prep} run_job=${run} dependency=afterok:${prep}"
echo "preprocess=PotLLL-heuristic-mpfr256-then-BKZ64x8 MLD=120 TSD=139 target_norm2=9799278"
echo "logs=${LOG_ROOT}"
