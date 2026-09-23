#!/bin/bash
#SBATCH --job-name=svp164-dh-snapshot
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=1
#SBATCH --time=06:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out

set -euo pipefail

# The source pool must be quiescent for the entire copy; an active pump-down
# removes chunks faster than a full CSD138 copy can finish.

SOURCE_CODE="${SLURM_SUBMIT_DIR}"
LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SOURCE="${LARGE_BASE}/runs/svp164-seed0/bgj-bkz138"
SNAPSHOT="${LARGE_BASE}/diagnostics/svp164-dh-csd138-${SLURM_JOB_ID}"
mkdir -p "${SNAPSHOT}"

echo "snapshot=${SNAPSHOT} source=${SOURCE}"
cd "${SOURCE}"
find .pool -type f -print0 | xargs -0 -r -n 64 -P 32 \
    cp -a --reflink=auto --parents --target-directory="${SNAPSHOT}"
cp -a "${SOURCE}/.bkz-scheduler-input.basis_0" "${SNAPSHOT}/"

python3 "${SOURCE_CODE}/tools/validate_hd_pool_snapshot.py" "${SNAPSHOT}" \
    --left 26 --right 164 --hash 0xf9750b7f3b244bfe \
    --chunks 163130 --vectors 1336357958
