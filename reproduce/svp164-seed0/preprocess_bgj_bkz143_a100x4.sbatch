#!/bin/bash
#SBATCH --job-name=svp164-s0-bgj-b143
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out

set -euo pipefail
ulimit -c 0

SOURCE="${SLURM_SUBMIT_DIR}"
LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SIF="${SIF:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
INPUT="${LARGE_BASE}/inputs/svp164-seed0/L_164_0.hdbkz-b138-j9-f26.pre"
OUTPUT="${LARGE_BASE}/inputs/svp164-seed0/L_164_0.hdbkz-b143-j9-f21.pre"
RAW="${LARGE_BASE}/inputs/svp164-seed0/L_164_0.raw"
RUN_ROOT="${LARGE_BASE}/runs/svp164-seed0/bgj-bkz143-pump-checkpoints"
BUILD_ROOT="${LARGE_BASE}/builds/svp164-bgj-bkz143-${SLURM_JOB_ID}"
LOG_ROOT="${LARGE_BASE}/logs/svp164-bgj"

if [[ ! -s "${INPUT}" ]]; then
    echo "Missing completed BKZ-138 basis: ${INPUT}" >&2
    exit 1
fi
mkdir -p "${RUN_ROOT}" "${BUILD_ROOT}" "${LOG_ROOT}" "$(dirname "${OUTPUT}")"
for kind in pool bucket sol uid; do
    for device in 0 1 2 3; do
        mkdir -p "${RUN_ROOT}/.${kind}/${device}"
    done
    # Scratch from an interrupted pump must not contaminate the next attempt.
    find "${RUN_ROOT}/.${kind}" -type f -delete
done
find "${RUN_ROOT}" -maxdepth 1 -type f \
    \( -name '.bkz-scheduler-input.basis_*' -o -name '.bkz.tmp_basis' \) -delete

BUILD_CPPFLAGS="-DHD_A100X4_500G_CACHE_PROFILE=1 -DHD_A100X4_PWC_GB=405 -DHD_A100X4_BWC_GB=1 -DHD_A100X4_SWC_GB=4 -DPOOL_HOST_VEC_MAX_DIM=144 -DENABLE_PROFILING=0"
apptainer exec --nv --bind "${SOURCE}","${BUILD_ROOT}" "${SIF}" bash -lc \
    "make -B -C '${SOURCE}/src' -j 16 CPPFLAGS='${BUILD_CPPFLAGS}' && make -B -C '${SOURCE}/app' -j 4 hd_sieve CPPFLAGS='${BUILD_CPPFLAGS}' && cp '${SOURCE}/app/hd_sieve' '${BUILD_ROOT}/'"

export APPTAINERENV_HD_RSS_TELEMETRY=1
export APPTAINERENV_HD_HBM_TELEMETRY=1
export APPTAINERENV_HD_HOST_ARENA_INITIAL_GB=64
export APPTAINERENV_HD_PWC_NO_GROW=1
export APPTAINERENV_HD_LAZY_SYNC=1
export APPTAINERENV_HD_PROFILE_BUCKET_STATS=0
export APPTAINERENV_HD_GPU_NATIVE_BWC=1
export APPTAINERENV_HD_GPU_NATIVE_BWC_P2P=1
export APPTAINERENV_HD_HBM_BWC_GB=24
export APPTAINERENV_HD_HBM_BWC_MIN_CSD=100
export APPTAINERENV_HD_BUCKET_THREADS=20

timeout --signal=TERM --kill-after=60s 604200s \
    apptainer exec --nv \
        --bind "${SOURCE}","${BUILD_ROOT}","${RUN_ROOT}","$(dirname "${INPUT}")" \
        --pwd "${SOURCE}" \
        "${SIF}" python3 tools/bkz_scheduler.py \
        --input "${INPUT}" \
        --schedule reproduce/svp164-seed0/bgj-bkz143-pump-checkpoints.json \
        --binary "${BUILD_ROOT}/hd_sieve" \
        --work-dir "${RUN_ROOT}" \
        --output "${OUTPUT}"

apptainer exec --bind "${SOURCE}","$(dirname "${RAW}")" \
    --pwd "${SOURCE}" "${SIF}" python3 tools/check_basis_det_mod.py "${RAW}" "${OUTPUT}"
