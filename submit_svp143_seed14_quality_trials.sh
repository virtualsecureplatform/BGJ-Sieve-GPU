#!/bin/bash

set -euo pipefail
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
common="SOURCE_COMMIT=${SOURCE_COMMIT},SVP_DIMENSION=143,SVP142_LATTICE_SEED=14,SVP_INSTANCE=svp143-seed14,SVP142_REPRO=${SOURCE}/reproduce/svp143-seed14/reproduce.py,SVP_PREPROCESS_MODE=lll-deeplll-bkz,SVP_DEEPLLL_DEPTH=4,SVP_PREPROCESS_IMAGE=/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/bgj-sieve-gpu-cuda13.3.pre-b0c81ac.sif"

base_prep="$(sbatch --parsable --job-name=svp143-d4-p60 \
    --export="ALL,${common},SVP_BKZ_BETA=60,SVP_BKZ_LOOPS=8" \
    prepare_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"
quality_run="$(sbatch --parsable --job-name=svp143-tsd135 \
    --dependency="afterok:${base_prep}" \
    --export="ALL,${common},SVP_BKZ_BETA=60,SVP_BKZ_LOOPS=8,SVP_SIEVE_SEED=0,TARGET_SIEVING_DIM=135,MIN_LIFTING_DIM=118,GPU_UID_SHARED_LOAD_PCT=60,CONTINUE_AFTER_TARGET=1" \
    run_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"

strong_prep="$(sbatch --parsable --job-name=svp143-d4-p64 \
    --dependency="afterok:${base_prep}" \
    --export="ALL,${common},SVP_BKZ_BETA=64,SVP_BKZ_LOOPS=8" \
    prepare_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"
strong_run="$(sbatch --parsable --job-name=svp143-p64 \
    --dependency="afterok:${strong_prep}" \
    --export="ALL,${common},SVP_BKZ_BETA=64,SVP_BKZ_LOOPS=8,SVP_SIEVE_SEED=0,TARGET_SIEVING_DIM=134,MIN_LIFTING_DIM=118,GPU_UID_SHARED_LOAD_PCT=60,CONTINUE_AFTER_TARGET=0" \
    run_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"

echo "current_basis_prepare=${base_prep} tsd135_run=${quality_run} MLD=118 continue=1"
echo "bkz64_prepare=${strong_prep} tsd134_run=${strong_run} MLD=118"
