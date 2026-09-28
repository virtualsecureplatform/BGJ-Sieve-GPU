#!/bin/bash
#SBATCH --job-name=svp164-s0-csd144-hbm-pwc
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out

set -euo pipefail

LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SOURCE_RUN="${LARGE_BASE}/runs/svp164-seed0/b138-sieve-csd143-maxram"
RUN_ROOT="${LARGE_BASE}/runs/svp164-seed0/b138-sieve-csd144-hbm-pwc"
SOURCE_CHUNK="${SOURCE_RUN}/.pool/0/.MtWWo5zK_000000"
if [[ ! -f "${SOURCE_CHUNK}" || "$(od -An -tu1 -j10 -N2 "${SOURCE_CHUNK}" | xargs)" != "21 164" ]]; then
    echo "Source pool is not a completed CSD-143 checkpoint" >&2
    exit 1
fi

# Copy the completed checkpoint to an independent run root; an OOM must not
# damage the source needed for a revised memory split.
STAGING="${RUN_ROOT}.init-${SLURM_JOB_ID}"
if [[ -e "${RUN_ROOT}" || -e "${STAGING}" ]]; then
    echo "Refusing to overwrite an existing run directory: ${RUN_ROOT}" >&2
    exit 1
fi
mkdir -p "${STAGING}/.pool"
echo "Copying completed CSD-143 pool from ${SOURCE_RUN} to ${RUN_ROOT}"
pids=()
for device in 0 1 2 3; do
    cp -a --reflink=auto "${SOURCE_RUN}/.pool/${device}" "${STAGING}/.pool/${device}" &
    pids+=("$!")
done
copy_failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then copy_failed=1; fi
done
if (( copy_failed )); then
    echo "Pool copy failed; leaving staging directory for inspection" >&2
    exit 1
fi
mv "${STAGING}" "${RUN_ROOT}"

export RUN_ROOT PWC_GB=408 SIEVE_TIMEOUT_SECONDS=258600
export HD_RESUME_COMPLETED_CSD=143
export APPTAINERENV_HD_HBM_PWC_GB=16
export APPTAINERENV_HD_HBM_PWC_MIN_CSD=144
echo "CSD-144 cache: host PWC ${PWC_GB} GiB, HBM PWC 16 GiB/GPU"
TSD=144 bash "${SLURM_SUBMIT_DIR}/reproduce/svp164-seed0/sieve_a100x4.sbatch"
