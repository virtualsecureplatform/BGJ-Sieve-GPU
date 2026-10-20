#!/bin/bash
#SBATCH --job-name=bgj-mpi-phase-bench
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=8
#SBATCH --time=04:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/mpi/slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${VALIDATED_BUILD:?}"
export MPI_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
export MPI_SIF="${MPI_BASE}/bgj-sieve-gpu-cuda13.3.sif"
SOURCE="${VALIDATED_BUILD}/source"
# Reuse only binaries from the exact source that passed validation.
git -C "${SLURM_SUBMIT_DIR}" show "${SOURCE_COMMIT}:app/bin_hd_phase_bench.cu" | cmp - "${SOURCE}/app/bin_hd_phase_bench.cu"
git -C "${SLURM_SUBMIT_DIR}" show "${SOURCE_COMMIT}:app/mpi_sieve_runtime.cpp" | cmp - "${SOURCE}/app/mpi_sieve_runtime.cpp"
for name in hd_phase_bench hd_phase_bench_mpi; do test -x "${SOURCE}/app/${name}"; done
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
BASIS="${MPI_BASE}/runs/svp167-search/potlll-simulation-8744618/seed-38/result.basis"
FIRST_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
# --rsc=g=8 exports a total eight-GPU request. Override it for the
# one-node serial step; inheriting eight GPUs there makes the step unschedulable.
# Keep all four GPUs and 128 CPUs per participating node in both tests.
# Fail promptly if step resources cannot be granted inside our allocation.
for dimension in ${BENCH_DIMENSIONS:-128}; do
    [[ "${dimension}" =~ ^[0-9]+$ ]] && (( dimension >= 40 && dimension <= 143 ))
    export MPI_BINARY="${SOURCE}/app/hd_phase_bench"
    echo "Launching serial CSD=${dimension}: node=${FIRST_NODE}, GPUs=4, CPUs=128"
    timeout --signal=TERM --kill-after=60 5400 srun --mpi=pmix_v2 --immediate=60 \
        --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=4 --gres=gpu:4 \
        --nodelist="${FIRST_NODE}" --cpus-per-task=128 --cpu-bind=none --kill-on-bad-exit=1 \
        "${SOURCE}/reproduce/svp167-search/mpi_rank.sh" "${BASIS}" "${dimension}" "${BENCH}/serial-${dimension}" \
        > "${BENCH}/serial-${dimension}.log" 2>&1
    export MPI_BINARY="${SOURCE}/app/hd_phase_bench_mpi"
    echo "Launching MPI CSD=${dimension}: nodes=2, GPUs=8, CPUs=256"
    timeout --signal=TERM --kill-after=60 5400 srun --mpi=pmix_v2 --immediate=60 \
        --nodes=2 --ntasks=2 --gpus=8 --gres=gpu:4 \
        --ntasks-per-node=1 --cpus-per-task=128 --cpu-bind=none --kill-on-bad-exit=1 \
        "${SOURCE}/reproduce/svp167-search/mpi_rank.sh" "${BASIS}" "${dimension}" "${BENCH}/mpi-${dimension}" \
        > "${BENCH}/mpi-${dimension}.log" 2>&1
    rg '\[PHASE-BENCH\]|\[MPI\] traffic' "${BENCH}/serial-${dimension}.log" "${BENCH}/mpi-${dimension}.log"
done
echo "PASS: one-phase serial/MPI comparison; source=${SOURCE_COMMIT} logs=${BENCH}"
