#!/bin/bash

set -euo pipefail
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
common="SOURCE_COMMIT=${SOURCE_COMMIT},SVP_DIMENSION=143,SVP142_LATTICE_SEED=14,SVP_INSTANCE=svp143-seed14,SVP142_REPRO=${SOURCE}/reproduce/svp143-seed14/reproduce.py,SVP_PREPROCESS_MODE=lll-deeplll-bkz,SVP_DEEPLLL_DEPTH=4,SVP_BKZ_BETA=60,SVP_BKZ_LOOPS=8,SVP_PREPROCESS_IMAGE=/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/bgj-sieve-gpu-cuda13.3.pre-b0c81ac.sif"
prep="$(sbatch --parsable --job-name=svp143-d4-cache \
    --export="ALL,${common}" prepare_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"
echo "prepare_job=${prep}"
for rng in 0 1; do
    job="$(sbatch --parsable --job-name="svp143-d4-r${rng}" \
        --dependency="afterok:${prep}" \
        --export="ALL,${common},SVP_SIEVE_SEED=${rng},TARGET_SIEVING_DIM=134,MIN_LIFTING_DIM=119,GPU_UID_SHARED_LOAD_PCT=60" \
        run_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"
    echo "rng=${rng} run_job=${job} TSD=134 MLD=119"
done
