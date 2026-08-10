#!/bin/bash
#SBATCH --job-name=svp142-search-prep
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=1
#SBATCH --array=43-52%1
#SBATCH --time=00:20:00
#SBATCH --output=svp142-search-prep_%A_%a.log

set -euo pipefail
export SVP142_LATTICE_SEED="${SLURM_ARRAY_TASK_ID}"
export SVP142_INPUT_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/svp142-seed${SLURM_ARRAY_TASK_ID}"
SOURCE="${SLURM_SUBMIT_DIR}"

# Reuse the generic preparation workflow, selecting the exploratory verifier.
exec env SVP142_REPRO="${SOURCE}/reproduce/svp142-search/reproduce.py" \
    bash "${SOURCE}/prepare_svp142_seed0_a100x4.sbatch"
