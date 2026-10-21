#!/bin/bash
#SBATCH --job-name=bgj-mpi-pipeline-replay
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=8
#SBATCH --time=02:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/mpi/slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${VALIDATED_BUILD:?}" "${MPI_HOST_RUNTIME:?}"
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" == 2 ]]
export MPI_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
export MPI_SIF="${MPI_BASE}/bgj-sieve-gpu-cuda13.3.sif"
SOURCE="${VALIDATED_BUILD}/source"
for path in app/mpi_sieve_runtime.cpp app/mpi_sieve_pipeline.inc; do
    git -C "${SLURM_SUBMIT_DIR}" show "${SOURCE_COMMIT}:${path}" | cmp - "${SOURCE}/${path}"
done
export MPI_BINARY="${SOURCE}/app/hd_sieve_mpi"
export MPI_LIBDIR="$(mpicxx --showme:link | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx OMPI_MCA_pml_ucx_tls=any OMPI_MCA_pml_ucx_devices=any PMIX_MCA_gds=hash
export APPTAINERENV_UCX_TLS=rc,sm,self APPTAINERENV_UCX_LOG_LEVEL=info
export APPTAINERENV_HD_FAIL_ON_SPILL=1 APPTAINERENV_HD_LAZY_SYNC=1 APPTAINERENV_HD_BWC_LAZY_SYNC=1
export APPTAINERENV_HD_HOST_ARENA_INITIAL_GB=4 APPTAINERENV_HD_PWC_NO_GROW=1
export APPTAINERENV_HD_CACHE_WAIT_TIMEOUT_S=120 APPTAINERENV_HD_POOL_BUFFER_CACHE=0
export APPTAINERENV_HD_SOURCE_COMMIT="${SOURCE_COMMIT}"
python3 "${SOURCE}/tools/test_mpi_pipeline_replay.py" \
    --wrapper "${SOURCE}/reproduce/svp167-search/mpi_rank.sh" \
    --basis "${VALIDATED_BUILD}/tests/input.basis" \
    --root "${MPI_BASE}/diagnostics/mpi-pipeline-replay-${SLURM_JOB_ID}"
