#!/bin/bash
#SBATCH --job-name=bgj-memory-safety
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=1
#SBATCH --time=02:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/mpi/slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}"
BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
RUN="${BASE}/diagnostics/mpi-safety-${SLURM_JOB_ID}"
mkdir -p "${RUN}/source"
git -C "${SLURM_SUBMIT_DIR}" archive "${SOURCE_COMMIT}" | tar -x -C "${RUN}/source"
apptainer exec --nv --bind /opt/system,"${BASE}" "${BASE}/bgj-sieve-gpu-cuda13.3.sif" \
    bash -s -- "${RUN}" <<'SH'
set -euo pipefail
run="$1"
source_dir="${run}/source"
flags='-DHD_A100X4_500G_CACHE_PROFILE=1 -DHD_A100X4_PWC_GB=260 -DHD_A100X4_BWC_GB=128 -DHD_A100X4_SWC_GB=16 -DPOOL_HOST_VEC_MAX_DIM=144 -DENABLE_PROFILING=0'
make -B -C "${source_dir}/src" -j 16 CPPFLAGS="${flags}"
nvcc -O3 -gencode arch=compute_80,code=sm_80 -static-global-template-stub=false \
    ${flags} "${source_dir}/tests/test_bucket_bounds.cu" "${source_dir}/src/libllib.a" \
    -Xcompiler -fopenmp -lntl -lgmp -o "${run}/bucket-bounds"
g++ -std=c++17 -O1 -g -pthread -fsanitize=address,undefined \
    "${source_dir}/tests/test_bounded_task_queue.cpp" -o "${run}/queue-asan"
timeout 180 "${run}/queue-asan"
g++ -std=c++17 -O1 -g -pthread -fsanitize=thread \
    "${source_dir}/tests/test_bounded_task_queue.cpp" -o "${run}/queue-tsan"
timeout 180 "${run}/queue-tsan"
for tool in memcheck initcheck racecheck synccheck; do
    timeout 1200 compute-sanitizer --tool "${tool}" --error-exitcode 86 \
        "${run}/bucket-bounds" > "${run}/${tool}.log" 2>&1
    tail -8 "${run}/${tool}.log"
done
SH
echo "PASS: memory-safety validation; source=${SOURCE_COMMIT} run=${RUN}"
