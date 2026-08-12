#!/bin/bash
#SBATCH --job-name=svp144-search-prep
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=1
#SBATCH --array=8-20%2
#SBATCH --time=00:45:00
#SBATCH --output=svp144-search-prep_%A_%a.log

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
export SVP_DIMENSION=144
export SVP142_LATTICE_SEED="${SLURM_ARRAY_TASK_ID}"
export SVP_INSTANCE="svp144-seed${SLURM_ARRAY_TASK_ID}"
export SVP142_INPUT_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/${SVP_INSTANCE}"
export SVP142_REPRO="${SOURCE}/reproduce/svp144-search/reproduce.py"
export SVP_PREPROCESS_MODE=lll-potlllbkz
export SVP_BKZ_BETA=64
export SVP_BKZ_LOOPS=8

exec bash "${SOURCE}/prepare_svp142_seed0_a100x4.sbatch"
