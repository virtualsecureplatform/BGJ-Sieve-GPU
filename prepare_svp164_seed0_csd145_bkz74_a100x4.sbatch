#!/bin/bash
#SBATCH --job-name=svp164-s0-bkz74-prep
#SBATCH --partition=gr20100g
#SBATCH --rsc=p=1:c=16:t=1:m=64G
#SBATCH --time=02:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp163-164/slurm-%x-%j.out

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
export SVP_DIMENSION=164
export SVP142_LATTICE_SEED=0
export SVP_INSTANCE=svp164-seed0
export SVP142_INPUT_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/${SVP_INSTANCE}"
export SVP142_REPRO="${SOURCE}/reproduce/svp164-seed0/reproduce.py"
export SVP142_TARGET_NORM2=11009124
export SVP142_RAW_SHA256=GENERATE
export SVP142_KNOWN_VECTOR="${SOURCE}/reproduce/svp158-search/no-known-vector.txt"
export SVP_PREPROCESS_MODE=potlll-bkz
export SVP_BKZ_BETA=74
export SVP_BKZ_LOOPS=8
export BGJ_ENABLE_PROFILING=0
export POOL_HOST_VEC_MAX_DIM=145

exec bash "${SOURCE}/prepare_svp142_seed0_a100x4.sbatch"
