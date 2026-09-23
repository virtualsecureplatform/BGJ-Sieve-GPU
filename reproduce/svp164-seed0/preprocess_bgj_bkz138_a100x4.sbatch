#!/bin/bash
#SBATCH --job-name=svp164-s0-bgj-b138
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out

set -euo pipefail
ulimit -c 0

SOURCE="${SLURM_SUBMIT_DIR}"
LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SIF="${SIF:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
INPUT="${INPUT:-${LARGE_BASE}/inputs/svp164-seed0/L_164_0.hdbkz-b128-j8-f32.pre}"
OUTPUT="${OUTPUT:-${LARGE_BASE}/inputs/svp164-seed0/L_164_0.hdbkz-b138-j9-f26.pre}"
RAW="${RAW:-${LARGE_BASE}/inputs/svp164-seed0/L_164_0.raw}"
RUN_ROOT="${RUN_ROOT:-${LARGE_BASE}/runs/svp164-seed0/bgj-bkz138}"
BUILD_ROOT="${LARGE_BASE}/builds/svp164-bgj-bkz138-${SLURM_JOB_ID}"
LOG_ROOT="${LARGE_BASE}/logs/svp164-bgj"

mkdir -p "${RUN_ROOT}" "${BUILD_ROOT}" "${LOG_ROOT}" "$(dirname "${OUTPUT}")"
RESUME_CSD="${RESUME_CSD:-}"
for kind in pool bucket sol uid; do
    for device in 0 1 2 3; do
        mkdir -p "${RUN_ROOT}/.${kind}/${device}"
    done
    # These files are per-pump scratch, not scheduler checkpoints.  A killed
    # pump can leave tens of thousands of stale files which must not be loaded
    # by the restarted tour.
    if [[ "${kind}" != pool || -z "${RESUME_CSD}" ]]; then
        find "${RUN_ROOT}/.${kind}" -type f -delete
    fi
done
find "${RUN_ROOT}" -maxdepth 1 -type f \
    \( -name '.bkz-scheduler-input.basis_*' -o -name '.bkz.tmp_basis' \) -delete

BUILD_CPPFLAGS="-DHD_A100X4_500G_CACHE_PROFILE=1 -DPOOL_HOST_VEC_MAX_DIM=144 -DENABLE_PROFILING=0"
apptainer exec --nv --bind "${SOURCE}","${BUILD_ROOT}" "${SIF}" bash -lc \
    "make -B -C '${SOURCE}/src' -j 16 CPPFLAGS='${BUILD_CPPFLAGS}' && make -B -C '${SOURCE}/app' -j 4 hd_sieve CPPFLAGS='${BUILD_CPPFLAGS}' && cp '${SOURCE}/app/hd_sieve' '${BUILD_ROOT}/'"

export APPTAINERENV_HD_RSS_TELEMETRY=1
export APPTAINERENV_HD_HBM_TELEMETRY=1
export APPTAINERENV_HD_PROFILE_BUCKET_STATS=0
export APPTAINERENV_HD_GPU_NATIVE_BWC=1
export APPTAINERENV_HD_GPU_NATIVE_BWC_P2P=1
export APPTAINERENV_HD_HBM_BWC_GB=24
export APPTAINERENV_HD_HBM_BWC_MIN_CSD=100
export APPTAINERENV_HD_BUCKET_THREADS=20
if [[ -n "${RESUME_CSD}" ]]; then
    export APPTAINERENV_HD_PUMP_RESUME_CSD="${RESUME_CSD}"
    export APPTAINERENV_HD_PUMP_RECOVERY_TRACE=1
fi

timeout --signal=TERM --kill-after=60s 258930s \
    apptainer exec --nv \
        --bind "${SOURCE}","${BUILD_ROOT}","${RUN_ROOT}","$(dirname "${INPUT}")","$(dirname "${OUTPUT}")" \
        --pwd "${SOURCE}" \
        "${SIF}" python3 tools/bkz_scheduler.py \
        --input "${INPUT}" \
        --schedule reproduce/svp164-seed0/bgj-bkz138.json \
        --binary "${BUILD_ROOT}/hd_sieve" \
        --work-dir "${RUN_ROOT}" \
        --output "${OUTPUT}"

apptainer exec --bind "${SOURCE}","$(dirname "${RAW}")","$(dirname "${OUTPUT}")" \
    --pwd "${SOURCE}" "${SIF}" python3 tools/check_basis_det_mod.py "${RAW}" "${OUTPUT}"
