#!/bin/bash
#SBATCH --job-name=svp164-dh-budget-fix
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=02:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out

set -euo pipefail
ulimit -c 0

SOURCE="${SLURM_SUBMIT_DIR}"
LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
REPLAY="${LARGE_BASE}/diagnostics/svp164-dh-large-replay-297763"
BUILD_ROOT="${LARGE_BASE}/builds/svp164-dh-budget-fix-${SLURM_JOB_ID}"
SIF="${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif"
CPPFLAGS="-DHD_A100X4_500G_CACHE_PROFILE=1 -DPOOL_HOST_VEC_MAX_DIM=144 -DENABLE_PROFILING=0"

test -f "${REPLAY}/.bkz-scheduler-input.basis_0"
count=$(find "${REPLAY}/.pool" -type f | wc -l)
test "${count}" -eq 141275
bad=$(find "${REPLAY}/.pool" -type f ! -size 1249280c -print -quit)
test -z "${bad}"

# The prior failing diagnostic left only disposable bucket/reducer scratch.
for kind in bucket sol uid; do
    find "${REPLAY}/.${kind}" -type f -delete
done
mkdir -p "${BUILD_ROOT}"

apptainer exec --nv --bind "${SOURCE}","${BUILD_ROOT}" "${SIF}" bash -lc \
    "make -B -C '${SOURCE}/src' -j 16 CPPFLAGS='${CPPFLAGS}' &&
     make -B -C '${SOURCE}/app' -j 4 hd_sieve CPPFLAGS='${CPPFLAGS}' &&
     cp '${SOURCE}/app/hd_sieve' '${BUILD_ROOT}/hd_sieve'"

export APPTAINERENV_HD_DH_REPLAY_INSERT=1
export APPTAINERENV_HD_DH_REPLAY_MAXTIME=448.06
export APPTAINERENV_HD_PUMP_RECOVERY_TRACE=1
export APPTAINERENV_HD_BUCKET_THREADS=20
export APPTAINERENV_HD_GPU_NATIVE_BWC=1
export APPTAINERENV_HD_GPU_NATIVE_BWC_P2P=1
export APPTAINERENV_HD_HBM_BWC_GB=24
export APPTAINERENV_HD_HBM_BWC_MIN_CSD=100

echo "fix_replay=${REPLAY} binary=${BUILD_ROOT}/hd_sieve node=${SLURMD_NODENAME}"
timeout --signal=TERM --kill-after=60s 6900s \
    apptainer exec --nv --bind "${REPLAY}","${BUILD_ROOT}" \
    --pwd "${REPLAY}" "${SIF}" \
    "${BUILD_ROOT}/hd_sieve" --input .bkz-scheduler-input.basis_0 --task dh --POS 0
