#!/bin/bash
#SBATCH --job-name=svp143-search-prep
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=1
#SBATCH --array=15-19%1
#SBATCH --time=00:45:00
#SBATCH --output=svp143-search-prep_%A_%a.log

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
export SVP_DIMENSION=143
export SVP142_LATTICE_SEED="${SLURM_ARRAY_TASK_ID}"
export SVP_INSTANCE="svp143-seed${SLURM_ARRAY_TASK_ID}"
export SVP142_INPUT_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/${SVP_INSTANCE}"
export SVP142_REPRO="${SOURCE}/reproduce/svp143-search/reproduce.py"
export SVP_PREPROCESS_MODE=lll-deeplll-bkz
export SVP_DEEPLLL_DEPTH=4
export SVP_BKZ_BETA=64
export SVP_BKZ_LOOPS=8
export SVP_PREPROCESS_IMAGE="${SVP_PREPROCESS_IMAGE:-/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/bgj-sieve-gpu-cuda13.3.pre-b0c81ac.sif}"

exec bash "${SOURCE}/prepare_svp142_seed0_a100x4.sbatch"
