#!/bin/bash
#SBATCH --job-name=svp164-s1-potlll-bkz
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SIF="${BASE}/bgj-sieve-gpu-cuda13.3.sif"
INPUTS="${BASE}/inputs/svp164-seed1"
RUN="${BASE}/runs/svp164-seed1/potlll-scheduled-bkz"
BUILD="${BASE}/builds/svp164-seed1-${SLURM_JOB_ID}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${SLURM_SUBMIT_DIR}" rev-parse HEAD)}"
mkdir -p "${INPUTS}" "${RUN}" "${BUILD}/source"
git -C "${SLURM_SUBMIT_DIR}" archive "${SOURCE_COMMIT}" | tar -x -C "${BUILD}/source"
SOURCE="${BUILD}/source"
SIMULATOR="${SLURM_SUBMIT_DIR}/../pro-pnj-bkz/PnJBKZ_Simulator/Fig2_Fig3_Fig4_Verification_Experiments_of_PnjBKZ_Simulator_2025_LLL_practice.py"
cp "${SIMULATOR}" "${BUILD}/reference-simulator.py"
RAW="${INPUTS}/L_164_1.raw"
POT="${INPUTS}/L_164_1.potlll.pre"
OUTPUT="${INPUTS}/L_164_1.scheduled-bkz.pre"
GENERATOR="${BASE}/deps/ntl-9.3.0-svp-generator/bin/svp_challenge_generator"
FPLLL_COMMIT="b0e38e31085e1ef3f115594142675012fd5f7224"
FPLLL="${BASE}/deps/fplll-${FPLLL_COMMIT:0:12}"
echo "Source commit: ${SOURCE_COMMIT}; lattice dimension=164 seed=1"
# Optional read-only inspection of an existing allocation, from a compute node.
if [[ -n "${DIAG_JOB_ID:-}" ]]; then
    timeout 45s srun --jobid="${DIAG_JOB_ID}" --overlap -N1 -n1 bash -lc '
        ps -u "$USER" -L -o pid,tid,pcpu,stat,wchan:32,comm --sort=-pcpu | head -35
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
        for pid in $(pgrep -u "$USER" -x hd_sieve); do
            if command -v gdb >/dev/null; then
                timeout 20s gdb -batch -ex "set pagination off" -ex "thread apply all bt 5" -p "$pid"
            fi
        done
    ' > "${BUILD}/existing-job-diagnostic.log" 2>&1 || true
    cat "${BUILD}/existing-job-diagnostic.log"
fi
apptainer exec --bind "${BASE}","${SLURM_SUBMIT_DIR}" "${SIF}" \
    bash "${SOURCE}/tools/build_fplll.sh" "${FPLLL}" \
    "${SLURM_SUBMIT_DIR}/dep/fplll" "${FPLLL_COMMIT}"
# Fail before preprocessing if a cached executable lacks this extension.
apptainer exec --bind "${BASE}" "${SIF}" env LD_LIBRARY_PATH="${FPLLL}/lib" \
    "${FPLLL}/bin/fplll" --help | grep -q 'potlll'
if [[ ! -s "${RAW}" ]]; then
    apptainer exec --bind "${BASE}" "${SIF}" "${GENERATOR}" 164 1 > "${RAW}.tmp"
    mv "${RAW}.tmp" "${RAW}"
fi
if [[ ! -s "${POT}" ]]; then
    apptainer exec --bind "${BASE}" "${SIF}" env LD_LIBRARY_PATH="${FPLLL}/lib" \
        "${FPLLL}/bin/fplll" -a potlll -m heuristic -f mpfr -p 256 "${RAW}" > "${POT}.tmp"
    mv "${POT}.tmp" "${POT}"
fi
apptainer exec --bind "${BASE}" "${SIF}" python3 \
    "${SOURCE}/tools/check_basis_det_mod.py" "${RAW}" "${POT}"
sha256sum "${RAW}" "${POT}" "${GENERATOR}" "${FPLLL}/bin/fplll" "${BUILD}/reference-simulator.py"
FLAGS="-DHD_A100X4_500G_CACHE_PROFILE=1 -DHD_A100X4_PWC_GB=380 -DHD_A100X4_BWC_GB=1 -DHD_A100X4_SWC_GB=4 -DPOOL_HOST_VEC_MAX_DIM=144 -DENABLE_PROFILING=0"
apptainer exec --nv --bind "${BASE}" "${SIF}" bash -lc \
    "make -B -C '${SOURCE}/src' -j 16 CPPFLAGS='${FLAGS}' && make -B -C '${SOURCE}/app' -j 4 hd_sieve CPPFLAGS='${FLAGS}' && g++ -O2 '${SOURCE}/tools/dump_gso.cpp' '${SOURCE}/src/libllib.a' -lntl -lgmp -lm -fopenmp -o '${BUILD}/dump_gso'"
apptainer exec --bind "${BASE}" "${SIF}" "${BUILD}/dump_gso" "${POT}" > "${RUN}/potlll.gso"
if [[ ! -s "${RUN}/schedule.json" ]]; then
    apptainer exec --bind "${BASE}" "${SIF}" python3 "${SOURCE}/tools/plan_svp164_chain.py" \
        "${RUN}/potlll.gso" "${BUILD}/reference-simulator.py" "${RUN}/schedule.json.tmp"
    mv "${RUN}/schedule.json.tmp" "${RUN}/schedule.json"
fi
cat "${RUN}/schedule.json"
for kind in pool bucket sol uid; do
    for device in 0 1 2 3; do mkdir -p "${RUN}/.${kind}/${device}"; done
    find "${RUN}/.${kind}" -type f -delete
done
find "${RUN}" -maxdepth 1 -type f \( -name '.bkz-scheduler-input.basis_*' -o -name '.bkz.tmp_basis' \) -delete
export APPTAINERENV_HD_RSS_TELEMETRY=1 APPTAINERENV_HD_HBM_TELEMETRY=1
export APPTAINERENV_HD_HOST_ARENA_INITIAL_GB=64 APPTAINERENV_HD_PWC_NO_GROW=1
export APPTAINERENV_HD_LAZY_SYNC=1 APPTAINERENV_HD_PROFILE_BUCKET_STATS=0
export APPTAINERENV_HD_GPU_NATIVE_BWC=1 APPTAINERENV_HD_GPU_NATIVE_BWC_P2P=1
export APPTAINERENV_HD_HBM_PWC_GB=16 APPTAINERENV_HD_HBM_PWC_MIN_CSD=143
export APPTAINERENV_HD_HBM_BWC_GB=12 APPTAINERENV_HD_HBM_BWC_MIN_CSD=100
export APPTAINERENV_HD_BGJ2_REDUCER_THREADS=32 APPTAINERENV_HD_POOL_BUFFER_CACHE=0
export APPTAINERENV_HD_BUCKET_THREADS=20
timeout --signal=TERM --kill-after=60s 604200s apptainer exec --nv --bind "${BASE}" \
    "${SIF}" python3 "${SOURCE}/tools/bkz_scheduler.py" --input "${POT}" \
    --schedule "${RUN}/schedule.json" --binary "${SOURCE}/app/hd_sieve" \
    --work-dir "${RUN}" --output "${OUTPUT}"
apptainer exec --bind "${BASE}" "${SIF}" python3 "${SOURCE}/tools/check_basis_det_mod.py" "${RAW}" "${OUTPUT}"
