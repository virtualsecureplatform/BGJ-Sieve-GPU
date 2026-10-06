#!/bin/bash
#SBATCH --job-name=svp167-bkz-candidate
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
: "${SEARCH_RUN:?}" "${SELECTED_SEED:?}" "${SOURCE_COMMIT:?}"
BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SIF="${BASE}/bgj-sieve-gpu-cuda13.3.sif"
DIR="${SEARCH_RUN}/seed-${SELECTED_SEED}"
RUN="${DIR}/scheduled-bkz"
BUILD="${BASE}/builds/svp167-seed${SELECTED_SEED}-${SLURM_JOB_ID}"
mkdir -p "${BUILD}/source" "${RUN}"
git -C "${SLURM_SUBMIT_DIR}" archive "${SOURCE_COMMIT}" | tar -x -C "${BUILD}/source"
SOURCE="${BUILD}/source"
SCHEDULE="${SCHEDULE_FILE:-${DIR}/schedule.json}"
FLAGS="-DHD_A100X4_500G_CACHE_PROFILE=1 -DHD_A100X4_PWC_GB=380 -DHD_A100X4_BWC_GB=8 -DHD_A100X4_SWC_GB=4 -DPOOL_HOST_VEC_MAX_DIM=144 -DENABLE_PROFILING=0"
apptainer exec --nv --bind "${BASE}" "${SIF}" bash -lc \
    "make -B -C '${SOURCE}/src' -j 16 CPPFLAGS='${FLAGS}' && make -B -C '${SOURCE}/app' -j 4 hd_sieve CPPFLAGS='${FLAGS}'"
ARCHIVE="${RUN}/interrupted-scratch-before-${SLURM_JOB_ID}"
mkdir -p "${ARCHIVE}"
for kind in pool bucket sol uid; do
    if [[ -d "${RUN}/.${kind}" ]]; then mv "${RUN}/.${kind}" "${ARCHIVE}/.${kind}"; fi
    for device in 0 1 2 3; do mkdir -p "${RUN}/.${kind}/${device}"; done
done
export APPTAINERENV_HD_RSS_TELEMETRY=1 APPTAINERENV_HD_HBM_TELEMETRY=1
export APPTAINERENV_HD_HOST_ARENA_INITIAL_GB=64 APPTAINERENV_HD_PWC_NO_GROW=1
export APPTAINERENV_HD_LAZY_SYNC=1 APPTAINERENV_HD_BWC_LAZY_SYNC=0
export APPTAINERENV_HD_HBM_PWC_WRITEBACK=1 APPTAINERENV_HD_CACHE_WAIT_TIMEOUT_S=600
export APPTAINERENV_HD_PROFILE_BUCKET_STATS=0 APPTAINERENV_HD_POOL_BUFFER_CACHE=0
export APPTAINERENV_HD_GPU_NATIVE_BWC=1 APPTAINERENV_HD_GPU_NATIVE_BWC_P2P=1
export APPTAINERENV_HD_HBM_PWC_GB=16 APPTAINERENV_HD_HBM_PWC_MIN_CSD=143
export APPTAINERENV_HD_HBM_BWC_GB=12 APPTAINERENV_HD_HBM_BWC_MIN_CSD=100
export APPTAINERENV_HD_BGJ2_REDUCER_THREADS=32 APPTAINERENV_HD_BUCKET_THREADS=20
echo "SVP167 seed ${SELECTED_SEED}; source ${SOURCE_COMMIT}; target length <=3405"
cat "${SCHEDULE}"
timeout --signal=TERM --kill-after=60s 604200s apptainer exec --nv --bind "${BASE}" "${SIF}" \
    python3 "${SOURCE}/tools/bkz_scheduler.py" --input "${DIR}/potlll.basis" \
    --schedule "${SCHEDULE}" --binary "${SOURCE}/app/hd_sieve" \
    --work-dir "${RUN}" --output "${DIR}/result.basis"
apptainer exec --bind "${BASE}" "${SIF}" python3 "${SOURCE}/tools/check_basis_det_mod.py" "${DIR}/input.raw" "${DIR}/result.basis"
