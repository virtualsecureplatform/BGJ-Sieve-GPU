#!/bin/bash
# Generic template: adjust account/partition/resources for the local cluster.
#SBATCH --job-name=bgj-pbkz
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300

set -euo pipefail

: "${BGJ_INPUT:?set BGJ_INPUT to the input basis}"
: "${BGJ_OUTPUT:?set BGJ_OUTPUT to the final checkpoint path}"

SOURCE="${SLURM_SUBMIT_DIR}"
RUN_ROOT="${BGJ_RUN_ROOT:-${SLURM_TMPDIR:-/tmp}/bgj-bkz-${SLURM_JOB_ID}}"
HD_SIEVE="${HD_SIEVE:-${SOURCE}/app/hd_sieve}"
SCHEDULE="${BGJ_SCHEDULE:-${SOURCE}/example/bkz-schedule.json}"
GPU_COUNT="${SLURM_GPUS_ON_NODE:-4}"

mkdir -p "${RUN_ROOT}"
for kind in pool bucket sol uid; do
    for ((device=0; device<GPU_COUNT; device++)); do
        mkdir -p "${RUN_ROOT}/.${kind}/${device}"
    done
done

srun python3 "${SOURCE}/tools/bkz_scheduler.py" \
    --input "${BGJ_INPUT}" \
    --schedule "${SCHEDULE}" \
    --binary "${HD_SIEVE}" \
    --work-dir "${RUN_ROOT}" \
    --output "${BGJ_OUTPUT}"
