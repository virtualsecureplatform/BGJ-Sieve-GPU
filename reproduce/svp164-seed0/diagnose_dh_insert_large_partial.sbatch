#!/bin/bash
#SBATCH --job-name=svp164-dh-large-replay
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=03:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out

set -euo pipefail
ulimit -c 0

SOURCE_CODE="${SLURM_SUBMIT_DIR}"
LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
LIVE="${LARGE_BASE}/runs/svp164-seed0/bgj-bkz138"
STAGE="${LARGE_BASE}/diagnostics/svp164-dh-large-stage-${STAGE_JOB:-${SLURM_JOB_ID}}"
REPLAY="${LARGE_BASE}/diagnostics/svp164-dh-large-replay-${SLURM_JOB_ID}"
BINARY="${LARGE_BASE}/builds/svp164-dh-partial-replay-297758/hd_sieve"
SIF="${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif"

mkdir -p "${REPLAY}"
for device in 0 1 2 3; do
    mkdir -p "${REPLAY}/.pool/${device}"
done
if [[ -z "${STAGE_JOB:-}" ]]; then
    mkdir -p "${STAGE}"
    # Link first, while the live job may unlink old chunk names.  The replay
    # does not use these links: it receives its own full byte copies below.
    cp -al "${LIVE}/.pool" "${STAGE}/"
    cp -a "${LIVE}/.bkz-scheduler-input.basis_0" "${STAGE}/"
fi
test -f "${STAGE}/.bkz-scheduler-input.basis_0"

cd "${STAGE}"
find .pool -type f -print0 | xargs -0 -r -n 64 -P 32 \
    cp -a --reflink=auto --parents --target-directory="${REPLAY}"
cp -a .bkz-scheduler-input.basis_0 "${REPLAY}/"
python3 "${SOURCE_CODE}/tools/validate_hd_pool_snapshot.py" "${REPLAY}" \
    --left 26 --right 164 --hash 0xf9750b7f3b244bfe --chunks 141275

for kind in bucket sol uid; do
    for device in 0 1 2 3; do
        mkdir -p "${REPLAY}/.${kind}/${device}"
    done
done

export APPTAINERENV_HD_DH_REPLAY_INSERT=1
export APPTAINERENV_HD_DH_REPLAY_MAXTIME=448.06
export APPTAINERENV_HD_PUMP_RECOVERY_TRACE=1
export APPTAINERENV_HD_BUCKET_THREADS=20
export APPTAINERENV_HD_GPU_NATIVE_BWC=1
export APPTAINERENV_HD_GPU_NATIVE_BWC_P2P=1
export APPTAINERENV_HD_HBM_BWC_GB=24
export APPTAINERENV_HD_HBM_BWC_MIN_CSD=100

echo "large_replay=${REPLAY} binary=${BINARY} node=${SLURMD_NODENAME}"
timeout --signal=TERM --kill-after=60s 9000s \
    apptainer exec --nv --bind "${REPLAY}","$(dirname "${BINARY}")" \
    --pwd "${REPLAY}" "${SIF}" \
    "${BINARY}" --input .bkz-scheduler-input.basis_0 --task dh --POS 0
