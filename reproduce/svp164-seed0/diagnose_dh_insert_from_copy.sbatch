#!/bin/bash
#SBATCH --job-name=svp164-dh-copy-replay
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=02:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out

set -euo pipefail
ulimit -c 0

: "${REPLAY_JOB:?submit with REPLAY_JOB=<copy job ID>}"
LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
REPLAY="${LARGE_BASE}/diagnostics/svp164-dh-large-replay-${REPLAY_JOB}"
BINARY="${LARGE_BASE}/builds/svp164-dh-partial-replay-297758/hd_sieve"
SIF="${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif"

test -f "${REPLAY}/.bkz-scheduler-input.basis_0"
for device in 0 1 2 3; do
    test -d "${REPLAY}/.pool/${device}"
done
count=$(find "${REPLAY}/.pool" -type f | wc -l)
test "${count}" -eq 141275
bad=$(find "${REPLAY}/.pool" -type f ! -size 1249280c -print -quit)
test -z "${bad}"
echo "copy_guard_ok chunks=${count} root=${REPLAY} node=${SLURMD_NODENAME}"

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

timeout --signal=TERM --kill-after=60s 6900s \
    apptainer exec --nv --bind "${REPLAY}","$(dirname "${BINARY}")" \
    --pwd "${REPLAY}" "${SIF}" \
    "${BINARY}" --input .bkz-scheduler-input.basis_0 --task dh --POS 0
