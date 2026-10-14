#!/bin/bash
#SBATCH --job-name=bgj-hbm-writeback-test
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=01:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
RUN="${BASE}/diagnostics/hbm-writeback-${SLURM_JOB_ID}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${SLURM_SUBMIT_DIR}" rev-parse HEAD)}"
mkdir -p "${RUN}/source"
git -C "${SLURM_SUBMIT_DIR}" archive "${SOURCE_COMMIT}" | tar -x -C "${RUN}/source"
FLAGS="-DHD_A100X4_500G_CACHE_PROFILE=1 -DHD_A100X4_PWC_GB=1 -DHD_A100X4_BWC_GB=1 -DHD_A100X4_SWC_GB=1 -DPOOL_HOST_VEC_MAX_DIM=144 -DENABLE_PROFILING=0"
SIF="${BASE}/bgj-sieve-gpu-cuda13.3.sif"
echo "Testing source ${SOURCE_COMMIT}"
apptainer exec --nv --bind "${BASE}" "${SIF}" bash -lc \
    "make -B -C '${RUN}/source/src' -j 16 CPPFLAGS='${FLAGS}' && cd '${RUN}/source' && nvcc -std=c++17 -Xcompiler '-O2 -g -fopenmp -pthread -march=native' ${FLAGS} tools/test_hbm_writeback.cpp -Lsrc -lllib -lntl -lgmp -lm -lcudart -lnuma -o '${RUN}/test'"
mapfile -t INPUTS < <(find "${BASE}/runs/svp164-seed1/potlll-scheduled-bkz/bkz-checkpoints" -maxdepth 2 -name stage-054-tour-01-b164-j09-f26.basis)
[[ "${#INPUTS[@]}" == 1 ]]
export APPTAINERENV_HD_LAZY_SYNC=1 APPTAINERENV_HD_HBM_PWC_WRITEBACK=1
export APPTAINERENV_HD_HBM_PWC_MIN_CSD=64 APPTAINERENV_HD_CACHE_WAIT_TIMEOUT_S=120
export APPTAINERENV_HD_HOST_ARENA_INITIAL_GB=24 APPTAINERENV_HD_POOL_BUFFER_CACHE=0
for mode in generous spill guarded-spill; do
    mkdir -p "${RUN}/${mode}"
    for kind in pool bucket sol uid; do
        for device in 0 1 2 3; do mkdir -p "${RUN}/${mode}/.${kind}/${device}"; done
    done
    if [[ "${mode}" == generous ]]; then
        export APPTAINERENV_HD_HBM_PWC_GB=0.125
        export APPTAINERENV_HD_PWC_FAIL_ON_SPILL=1
    else
        export APPTAINERENV_HD_HBM_PWC_GB=0.01
        export APPTAINERENV_HD_PWC_FAIL_ON_SPILL=0
    fi
    if [[ "${mode}" == guarded-spill ]]; then
        export APPTAINERENV_HD_PWC_FAIL_ON_SPILL=1
        set +e
        timeout 600s apptainer exec --nv --bind "${BASE}" --pwd "${RUN}/${mode}" \
            "${SIF}" "${RUN}/test" "${INPUTS[0]}" spill > "${RUN}/${mode}/test.log" 2>&1
        status=$?
        set -e
        [[ "${status}" == 134 ]]
        rg -q '\[PWC-SPILL-FORBIDDEN\]' "${RUN}/${mode}/test.log"
        [[ -z "$(find "${RUN}/${mode}/.pool" -type f -print -quit)" ]]
        echo "PASS: spill guard aborted before writing pool files"
        continue
    fi
    timeout 600s apptainer exec --nv --bind "${BASE}" --pwd "${RUN}/${mode}" \
        "${SIF}" "${RUN}/test" "${INPUTS[0]}" "${mode}"
done
echo "PASS: all HBM write-back tests"
