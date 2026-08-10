#!/bin/bash
#SBATCH --job-name=svp142-search
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --array=43-52%2
#SBATCH --time=03:00:00
#SBATCH --output=svp142-search_%A_%a.log

set -euo pipefail
export SVP142_LATTICE_SEED="${SLURM_ARRAY_TASK_ID}"
export SVP142_INPUT_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/svp142-seed${SLURM_ARRAY_TASK_ID}"
export SVP142_RUN_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/runs/svp142-seed${SLURM_ARRAY_TASK_ID}/job-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SOURCE="${SLURM_SUBMIT_DIR}"
exec env SVP142_REPRO="${SOURCE}/reproduce/svp142-search/reproduce.py" \
    bash "${SOURCE}/run_svp142_seed0_a100x4.sbatch"
