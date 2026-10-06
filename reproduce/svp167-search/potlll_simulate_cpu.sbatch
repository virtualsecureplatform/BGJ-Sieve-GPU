#!/bin/bash
#SBATCH --job-name=svp167-cpu-search
#SBATCH --partition=gr20100a
#SBATCH --rsc=p=1:c=8:t=1:m=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out
set -euo pipefail
export FIRST_SEED="${FIRST_SEED:-0}" LAST_SEED="${LAST_SEED:-63}" SEARCH_ONLY=1 SCAN_ALL=1
export REUSE_RUN="${REUSE_RUN:-/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/runs/svp167-search/potlll-simulation-300481}"
export SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${SLURM_SUBMIT_DIR}" rev-parse HEAD)}"
# Execute the common CPU-only workflow; its SBATCH GPU directives are comments
# here and do not apply to this allocation. No --nv or CUDA compilation is used.
WORKFLOW_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/builds/svp167-cpu-workflow-${SLURM_JOB_ID}"
mkdir -p "${WORKFLOW_DIR}"
git -C "${SLURM_SUBMIT_DIR}" archive "${SOURCE_COMMIT}" \
    reproduce/svp167-search/potlll_simulate_a100x1.sbatch | tar -x -C "${WORKFLOW_DIR}"
bash "${WORKFLOW_DIR}/reproduce/svp167-search/potlll_simulate_a100x1.sbatch"
