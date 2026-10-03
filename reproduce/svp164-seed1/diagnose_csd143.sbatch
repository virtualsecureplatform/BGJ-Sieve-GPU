#!/bin/bash
#SBATCH --job-name=svp164-s1-csd143-replay
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=1-00:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SIF="${BASE}/bgj-sieve-gpu-cuda13.3.sif"
BINARY="${BASE}/builds/svp164-seed1-298832/source/app/hd_sieve"
CHECKPOINT_ROOT="${BASE}/runs/svp164-seed1/potlll-scheduled-bkz/bkz-checkpoints"
mapfile -t INPUTS < <(find "${CHECKPOINT_ROOT}" -maxdepth 2 -name stage-054-tour-01-b164-j09-f26.basis)
if [[ "${#INPUTS[@]}" != 1 ]]; then
    echo "Expected exactly one completed BKZ138 checkpoint, found ${#INPUTS[@]}" >&2
    exit 1
fi
RUN="${BASE}/diagnostics/csd143-seed1-${SLURM_JOB_ID}"
mkdir -p "${RUN}/source"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${SLURM_SUBMIT_DIR}" rev-parse HEAD)}"
git -C "${SLURM_SUBMIT_DIR}" archive "${SOURCE_COMMIT}" tools/diagnose_sieve_stall.py | tar -x -C "${RUN}/source"
cp "${INPUTS[0]}" "${RUN}/input.basis"
for kind in pool bucket sol uid; do
    for device in 0 1 2 3; do mkdir -p "${RUN}/.${kind}/${device}"; done
done
echo "Diagnostic source=${SOURCE_COMMIT}; original solver binary=${BINARY}"
sha256sum "${BINARY}" "${RUN}/input.basis"
# Match the original run exactly. In particular, do not activate the new
# cache timeout or change memory sizes while establishing the failing stack.
export APPTAINERENV_HD_RSS_TELEMETRY=1 APPTAINERENV_HD_HBM_TELEMETRY=1
export APPTAINERENV_HD_HOST_ARENA_INITIAL_GB=64 APPTAINERENV_HD_PWC_NO_GROW=1
export APPTAINERENV_HD_LAZY_SYNC=1 APPTAINERENV_HD_BWC_LAZY_SYNC=0
export APPTAINERENV_HD_PROFILE_BUCKET_STATS=0 APPTAINERENV_HD_CACHE_WAIT_TIMEOUT_S=0
export APPTAINERENV_HD_GPU_NATIVE_BWC=1 APPTAINERENV_HD_GPU_NATIVE_BWC_P2P=1
export APPTAINERENV_HD_HBM_PWC_GB=16 APPTAINERENV_HD_HBM_PWC_MIN_CSD=143
export APPTAINERENV_HD_HBM_BWC_GB=12 APPTAINERENV_HD_HBM_BWC_MIN_CSD=100
export APPTAINERENV_HD_BGJ2_REDUCER_THREADS=32 APPTAINERENV_HD_POOL_BUFFER_CACHE=0
export APPTAINERENV_HD_BUCKET_THREADS=20
python3 "${RUN}/source/tools/diagnose_sieve_stall.py" \
    --binary "${BINARY}" --output-dir "${RUN}" --idle-minutes 20 -- \
    apptainer exec --nv --bind "${BASE}" --pwd "${RUN}" "${SIF}" \
    "${BINARY}" --input input.basis --task bkz --BSD 143 --D4F 21 \
    --JUMP 9 --BDH 0 --STI 0 --STOP_INDEX 9 --output diagnostic-output.basis
