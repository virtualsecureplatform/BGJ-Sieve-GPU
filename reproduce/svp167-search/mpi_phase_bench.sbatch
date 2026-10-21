#!/bin/bash
#SBATCH --job-name=bgj-mpi-phase-bench
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=04:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/mpi/slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${VALIDATED_BUILD:?}"
TEST_TIMEOUT_SECONDS="${BENCH_TIMEOUT_SECONDS:-10800}"
if ! [[ "${TEST_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || (( TEST_TIMEOUT_SECONDS <= 0 )); then
    echo "BENCH_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
fi
case "${BENCH_BACKEND:-serial}" in
    serial) BACKEND=serial; NODES=1; GPUS=4; BINARY=hd_phase_bench ;;
    mpi) BACKEND=mpi; NODES=2; GPUS=8; BINARY=hd_phase_bench_mpi ;;
    *) echo "BENCH_BACKEND must be serial or mpi" >&2; exit 2 ;;
esac
# Each job runs exactly one backend. Never reserve a second idle node for
# the serial baseline. The submission helper supplies --rsc=g=8 for MPI.
if [[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" != "${NODES}" ]]; then
    echo "${BACKEND} benchmark requires exactly ${NODES} allocated node(s)" >&2
    exit 2
fi
export MPI_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
export MPI_SIF="${MPI_BASE}/bgj-sieve-gpu-cuda13.3.sif"
SOURCE="${VALIDATED_BUILD}/source"
# Reuse only binaries from the exact source that passed validation.
git -C "${SLURM_SUBMIT_DIR}" show "${SOURCE_COMMIT}:app/bin_hd_phase_bench.cu" | cmp - "${SOURCE}/app/bin_hd_phase_bench.cu"
git -C "${SLURM_SUBMIT_DIR}" show "${SOURCE_COMMIT}:app/mpi_sieve_runtime.cpp" | cmp - "${SOURCE}/app/mpi_sieve_runtime.cpp"
test -x "${SOURCE}/app/${BINARY}"
export MPI_LIBDIR="$(mpicxx --showme:link | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx OMPI_MCA_pml_ucx_tls=any OMPI_MCA_pml_ucx_devices=any PMIX_MCA_gds=hash
export APPTAINERENV_HD_FAIL_ON_SPILL=1 APPTAINERENV_HD_LAZY_SYNC=1 APPTAINERENV_HD_BWC_LAZY_SYNC=1
export APPTAINERENV_HD_EAGER_HOST_ARENA=1 APPTAINERENV_HD_PWC_NO_GROW=1
# No checkpoint/context transition occurs in this benchmark. Do not flush
# disposable scratch at exit; production checkpoint durability is unchanged.
export APPTAINERENV_HD_HBM_PWC_WRITEBACK=0 APPTAINERENV_HD_HBM_PWC_GB=0
export APPTAINERENV_HD_HBM_BWC_GB=12 APPTAINERENV_HD_POOL_BUFFER_CACHE=0
export APPTAINERENV_HD_CACHE_WAIT_TIMEOUT_S=600
export APPTAINERENV_HD_BGJ2_REDUCER_THREADS=16 APPTAINERENV_HD_BUCKET_THREADS=16
export APPTAINERENV_HD_BGJ3_MIN_CSD=177 APPTAINERENV_HD_SIEVE_SEED=1234
BENCH="${MPI_BASE}/diagnostics/mpi-phase-bench-${SLURM_JOB_ID}"
mkdir -p "${BENCH}"
LAUNCH_COMMIT="${BENCH_LAUNCH_COMMIT:-$(git -C "${SLURM_SUBMIT_DIR}" rev-parse HEAD)}"
mkdir -p "${BENCH}/launcher"
git -C "${SLURM_SUBMIT_DIR}" archive "${LAUNCH_COMMIT}" \
    reproduce/svp167-search/phase_bench_rank.sh | tar -x -C "${BENCH}/launcher"
RANK_WRAPPER="${BENCH}/launcher/reproduce/svp167-search/phase_bench_rank.sh"
export BENCH_NATIVE_WRAPPER="${SOURCE}/reproduce/svp167-search/mpi_rank.sh"
echo "Diagnostic launcher=${LAUNCH_COMMIT}; validated binary source=${SOURCE_COMMIT}"
BASIS="${MPI_BASE}/runs/svp167-search/potlll-simulation-8744618/seed-38/result.basis"
# Keep four GPUs and 128 CPUs per participating node. Override inherited
# GPU requests explicitly and fail promptly if step creation cannot proceed.
for dimension in ${BENCH_DIMENSIONS:-128}; do
    [[ "${dimension}" =~ ^[0-9]+$ ]] && (( dimension >= 40 && dimension <= 143 ))
    export MPI_BINARY="${SOURCE}/app/${BINARY}"
    echo "Launching ${BACKEND} CSD=${dimension}: nodes=${NODES}, GPUs=${GPUS}, timeout=${TEST_TIMEOUT_SECONDS}s"
    timeout --signal=TERM --kill-after=60 "${TEST_TIMEOUT_SECONDS}" srun --mpi=pmix_v2 --immediate=60 \
        --nodes="${NODES}" --ntasks="${NODES}" --gpus="${GPUS}" --gres=gpu:4 \
        --ntasks-per-node=1 --cpus-per-task=128 --cpu-bind=none --kill-on-bad-exit=1 \
        bash "${RANK_WRAPPER}" "${BASIS}" "${dimension}" "${BENCH}/${BACKEND}-${dimension}" \
        > "${BENCH}/${BACKEND}-${dimension}.log" 2>&1
    rg '\[PHASE-BENCH\]|\[MPI\] traffic' "${BENCH}/${BACKEND}-${dimension}.log"
done
echo "PASS: ${BACKEND} phase benchmark; source=${SOURCE_COMMIT} logs=${BENCH}"
