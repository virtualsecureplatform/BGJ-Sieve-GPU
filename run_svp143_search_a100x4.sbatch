#!/bin/bash
#SBATCH --job-name=svp143-search
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --array=15-19%2
#SBATCH --time=03:00:00
#SBATCH --output=svp143-search_%A_%a.log

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
export SVP_DIMENSION=143
export SVP142_LATTICE_SEED="${SLURM_ARRAY_TASK_ID}"
export SVP_INSTANCE="svp143-seed${SLURM_ARRAY_TASK_ID}"
export SVP142_INPUT_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/${SVP_INSTANCE}"
export SVP142_RUN_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/runs/${SVP_INSTANCE}/job-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export SVP142_REPRO="${SOURCE}/reproduce/svp143-search/reproduce.py"
export SVP_PREPROCESS_MODE=lll-deeplll-bkz
export SVP_DEEPLLL_DEPTH=4
export SVP_BKZ_BETA=64
export SVP_BKZ_LOOPS=8
export TARGET_SIEVING_DIM=134
export MIN_LIFTING_DIM=118

exec bash "${SOURCE}/run_svp142_seed0_a100x4.sbatch"
