#!/bin/bash
#SBATCH --job-name=svp164-s0-sieve-maxram
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out

set -euo pipefail

: "${TSD:?submit with TSD=143 or TSD=144}"
LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
case "${TSD}" in
    143)
        SOURCE_RUN="${LARGE_BASE}/runs/svp164-seed0/b138-sieve"
        RUN_ROOT="${LARGE_BASE}/runs/svp164-seed0/b138-sieve-csd143-maxram"
        EXPECTED_CONTEXT="22 164"
        PWC_GB=405
        ;;
    144)
        SOURCE_RUN="${LARGE_BASE}/runs/svp164-seed0/b138-sieve-csd143-maxram"
        RUN_ROOT="${LARGE_BASE}/runs/svp164-seed0/b138-sieve-csd144-maxram"
        EXPECTED_CONTEXT="21 164"
        PWC_GB=468
        export HD_RESUME_COMPLETED_CSD=143
        ;;
    *)
        echo "Unsupported TSD ${TSD}" >&2
        exit 2
        ;;
esac

SOURCE_CHUNK="${SOURCE_RUN}/.pool/0/.MtWWo5zK_000000"
if [[ ! -f "${SOURCE_CHUNK}" || "$(od -An -tu1 -j10 -N2 "${SOURCE_CHUNK}" | xargs)" != "${EXPECTED_CONTEXT}" ]]; then
    echo "Source pool is not the expected completed context [${EXPECTED_CONTEXT}]" >&2
    exit 1
fi

# Keep each completed source pool intact so an OOM can be retried safely.
STAGING="${RUN_ROOT}.init-${SLURM_JOB_ID}"
if [[ -e "${RUN_ROOT}" || -e "${STAGING}" ]]; then
    echo "Refusing to overwrite an existing run directory: ${RUN_ROOT}" >&2
    exit 1
fi
mkdir -p "${STAGING}/.pool"
echo "Copying completed pool from ${SOURCE_RUN} to ${RUN_ROOT}"
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

export RUN_ROOT PWC_GB HD_HBM_BWC_GB=24 SIEVE_TIMEOUT_SECONDS=258600
bash "${SLURM_SUBMIT_DIR}/reproduce/svp164-seed0/sieve_a100x4.sbatch"
