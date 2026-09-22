#!/bin/bash
#SBATCH --job-name=svp164-s0-sieve
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out

set -euo pipefail

: "${TSD:?set TSD to the phase target}"
SOURCE="${SLURM_SUBMIT_DIR}"
LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SIF="${SIF:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
BASIS="${BASIS:-${LARGE_BASE}/inputs/svp164-seed0/L_164_0.hdbkz-b138-j9-f26.pre}"
# Both sieve jobs intentionally share this directory. The second invocation
# detects and reloads the pool produced by the first.
RUN_ROOT="${RUN_ROOT:-${LARGE_BASE}/runs/svp164-seed0/b138-sieve}"
BUILD_ROOT="${LARGE_BASE}/builds/svp164-b138-csd${TSD}-${SLURM_JOB_ID}"
LOG_ROOT="${LARGE_BASE}/logs/svp164-bgj"

mkdir -p "${BUILD_ROOT}" "${RUN_ROOT}" "${LOG_ROOT}"
for kind in pool bucket sol uid; do
    for device in 0 1 2 3; do
        mkdir -p "${RUN_ROOT}/.${kind}/${device}"
    done
done

BUILD_CPPFLAGS="-DHD_A100X4_500G_CACHE_PROFILE=1 -DHD_A100X4_PWC_GB=405 -DHD_A100X4_BWC_GB=1 -DHD_A100X4_SWC_GB=4 -DPOOL_HOST_VEC_MAX_DIM=144 -DENABLE_PROFILING=0"
apptainer exec --nv --bind "${SOURCE}","${BUILD_ROOT}" "${SIF}" bash -lc \
    "make -B -C '${SOURCE}/src' -j 16 CPPFLAGS='${BUILD_CPPFLAGS}' && make -B -C '${SOURCE}/app' -j 4 hd_sieve CPPFLAGS='${BUILD_CPPFLAGS}' && cp '${SOURCE}/app/hd_sieve' '${BUILD_ROOT}/'"

export APPTAINERENV_HD_RSS_TELEMETRY=1
export APPTAINERENV_HD_HBM_TELEMETRY=1
export APPTAINERENV_HD_HOST_ARENA_INITIAL_GB=64
export APPTAINERENV_HD_PWC_NO_GROW=1
export APPTAINERENV_HD_LAZY_SYNC=1
export APPTAINERENV_HD_SYNC_EVERY=1
export APPTAINERENV_HD_PROFILE_BUCKET_STATS=0
export APPTAINERENV_HD_GPU_NATIVE_BWC=1
export APPTAINERENV_HD_GPU_NATIVE_BWC_P2P=1
export APPTAINERENV_HD_HBM_BWC_GB=24
export APPTAINERENV_HD_HBM_BWC_MIN_CSD=100
export APPTAINERENV_HD_BUCKET_THREADS=16

echo "job=${SLURM_JOB_ID} basis=${BASIS} target_csd=${TSD} persistent_run_root=${RUN_ROOT}"
timeout --signal=TERM --kill-after=60s 86130s \
    apptainer exec --nv \
        --bind "${BUILD_ROOT}","${RUN_ROOT}","$(dirname "${BASIS}")" \
        --pwd "${RUN_ROOT}" \
        "${SIF}" "${BUILD_ROOT}/hd_sieve" \
        --input "${BASIS}" --task sieve --TSD "${TSD}" --MLD 129
