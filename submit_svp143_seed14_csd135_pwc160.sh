#!/bin/bash

set -euo pipefail
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
common="SOURCE_COMMIT=${SOURCE_COMMIT},SVP_DIMENSION=143,SVP142_LATTICE_SEED=14,SVP_INSTANCE=svp143-seed14,SVP142_REPRO=${SOURCE}/reproduce/svp143-seed14/reproduce.py,SVP_PREPROCESS_MODE=lll-deeplll-bkz,SVP_DEEPLLL_DEPTH=4,SVP_BKZ_BETA=64,SVP_BKZ_LOOPS=8,SVP_PREPROCESS_IMAGE=/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/bgj-sieve-gpu-cuda13.3.pre-b0c81ac.sif"

prep="$(sbatch --parsable --job-name=svp143-p64-c160 \
    --export="ALL,${common}" prepare_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"
run="$(sbatch --parsable --job-name=svp143-csd135-c160 \
    --dependency="afterok:${prep}" \
    --export="ALL,${common},SVP_SIEVE_SEED=0,TARGET_SIEVING_DIM=135,MIN_LIFTING_DIM=118,GPU_UID_SHARED_LOAD_PCT=60,CONTINUE_AFTER_TARGET=1" \
    run_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"

echo "prepare_job=${prep} run_job=${run} BKZ=64 MLD=118 TSD=135 PWC=160GiB continue=1"
