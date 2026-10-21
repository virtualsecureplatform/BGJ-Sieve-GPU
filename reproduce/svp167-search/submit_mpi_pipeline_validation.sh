#!/bin/bash
# Submit a strictly sequential chain: never more than two compute nodes.
set -euo pipefail
: "${SOURCE_COMMIT:?}" "${MPI_HOST_RUNTIME:?}"
export SOURCE_COMMIT MPI_HOST_RUNTIME
export APPTAINERENV_UCX_TLS=rc,sm,self APPTAINERENV_UCX_LOG_LEVEL=info
export APPTAINERENV_HD_MPI_PIPELINE=async
export VALIDATE_PIPELINES='async legacy'
export BENCH_DIMENSIONS=128 BENCH_STACK_INSPECTION=0 BENCH_TIMEOUT_SECONDS=10800
base="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build="$(sbatch --parsable --rsc=g=1 --time=02:00:00 --job-name=bgj-async-build \
    --export=ALL,BUILD_ONLY=1 "${script_dir}/mpi_bkz_validate.sbatch")"
build="${build%%;*}"
printf 'Build (one GPU): %s\n' "${build}"
validation="$(sbatch --parsable --rsc=g=8 --time=04:00:00 \
    --dependency="afterok:${build}" --kill-on-invalid-dep=yes \
    --export="ALL,BUILD_ONLY=0,REUSE_BUILD=${base}/diagnostics/mpi-bkz-${build}" \
    "${script_dir}/mpi_bkz_validate.sbatch")"
validation="${validation%%;*}"
printf 'Async + legacy integration (two nodes): %s\n' "${validation}"
export VALIDATED_BUILD="${base}/diagnostics/mpi-bkz-${validation}"
replay="$(sbatch --parsable --dependency="afterok:${validation}" --kill-on-invalid-dep=yes \
    "${script_dir}/mpi_pipeline_replay.sbatch")"
replay="${replay%%;*}"
printf 'Frozen replay and performance gate (two nodes): %s\n' "${replay}"
benchmark="$(sbatch --parsable --rsc=g=8 --job-name=bgj-async-csd128 \
    --dependency="afterok:${replay}" --kill-on-invalid-dep=yes \
    --export=ALL,BENCH_BACKEND=mpi "${script_dir}/mpi_phase_bench.sbatch")"
benchmark="${benchmark%%;*}"
printf 'Cold CSD128, gated on replay success (two nodes): %s\n' "${benchmark}"
