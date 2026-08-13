#!/bin/bash
#SBATCH --job-name=svp150-search
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --array=1-8%2
#SBATCH --time=05:00:00

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
export SVP_DIMENSION=150
export SVP142_LATTICE_SEED="${SLURM_ARRAY_TASK_ID}"
export SVP_INSTANCE="svp150-seed${SLURM_ARRAY_TASK_ID}"
export SVP142_INPUT_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/${SVP_INSTANCE}"
export SVP142_RUN_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/runs/${SVP_INSTANCE}/job-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export SVP142_REPRO="${SOURCE}/reproduce/svp150-search/reproduce.py"
export SVP_PREPROCESS_MODE=lll-potlllbkz
export SVP_BKZ_BETA=64
export SVP_BKZ_LOOPS=8
export TARGET_SIEVING_DIM=137
export MIN_LIFTING_DIM=120
export SVP_TIMEOUT_SECONDS=17700

exec bash "${SOURCE}/run_svp142_seed0_a100x4.sbatch"
