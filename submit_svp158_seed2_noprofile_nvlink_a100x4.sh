#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
LOG_ROOT="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/logs/svp158-seed2-noprofile-nvlink"
WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-292925_3:292925_4}"
common="SOURCE_COMMIT=${SOURCE_COMMIT},SVP_DIMENSION=158,SVP142_LATTICE_SEED=2,SVP_INSTANCE=svp158-seed2,SVP142_REPRO=${SOURCE}/reproduce/svp158-search/reproduce.py,SVP_PREPROCESS_MODE=potlll-bkz,SVP_BKZ_BETA=70,SVP_BKZ_LOOPS=8,BGJ_ENABLE_PROFILING=0"

if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to submit tracked, uncommitted source changes" >&2
    exit 2
fi
if [[ -n "$WAIT_FOR_JOBS" && ! "$WAIT_FOR_JOBS" =~ ^[0-9]+(_[0-9]+)?(:[0-9]+(_[0-9]+)?)*$ ]]; then
    echo "WAIT_FOR_JOBS must be colon-separated Slurm job IDs" >&2
    exit 2
fi

mkdir -p "$LOG_ROOT"
cd "$SOURCE"
prep="$(sbatch --parsable \
    --job-name=svp158-s2-np-build \
    --output="${LOG_ROOT}/build_%j.log" \
    --export="ALL,${common}" \
    prepare_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"

dependency="afterok:${prep}"
if [[ -n "$WAIT_FOR_JOBS" ]]; then
    dependency+=",afterany:${WAIT_FOR_JOBS}"
fi
run="$(sbatch --parsable \
    --job-name=svp158-s2-np-nvlink \
    --time=06:00:00 \
    --output="${LOG_ROOT}/run_%j.log" \
    --dependency="$dependency" \
    --export="ALL,${common},SVP_SIEVE_SEED=0,TARGET_SIEVING_DIM=139,MIN_LIFTING_DIM=120,SVP_TIMEOUT_SECONDS=21300,APPTAINERENV_HD_GPU_NATIVE_BWC=1,APPTAINERENV_HD_GPU_NATIVE_BWC_P2P=1,APPTAINERENV_HD_GPU_NATIVE_BWC_VERIFY=0" \
    run_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"

echo "build_job=${prep} run_job=${run} dependency=${dependency}"
echo "seed=2 preprocess=PotLLL-heuristic-mpfr256-then-BKZ70x8 MLD=120 TSD=139"
echo "variant=profiling-off+exact-GPU-BWC+direct-CUDA-P2P strict_target_norm2=9799277"
echo "baseline=job-292925_2 elapsed=04:25:39 best_norm2=10651290 CSD=137"
echo "logs=${LOG_ROOT}"
