#!/bin/bash
#SBATCH --job-name=svp146-search-prep
#SBATCH --partition=gr20100g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=15-27
#SBATCH --time=01:15:00

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
export SVP_DIMENSION=146
export SVP142_LATTICE_SEED="${SLURM_ARRAY_TASK_ID}"
export SVP_INSTANCE="svp146-seed${SLURM_ARRAY_TASK_ID}"
export SVP142_INPUT_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/${SVP_INSTANCE}"
export SVP142_REPRO="${SOURCE}/reproduce/svp146-search/reproduce.py"
export SVP_PREPROCESS_MODE=lll-potlllbkz
export SVP_BKZ_BETA=64
export SVP_BKZ_LOOPS=8

exec bash "${SOURCE}/prepare_svp142_seed0_a100x4.sbatch"
