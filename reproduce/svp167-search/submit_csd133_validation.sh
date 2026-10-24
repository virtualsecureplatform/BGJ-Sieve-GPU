#!/bin/bash
# Validate a source revision against the original CSD132 checkpoint. This
# deliberately stops after CSD133; it never submits production continuation.
set -euo pipefail
: "${SOURCE_COMMIT:?}" "${MPI_HOST_RUNTIME:?}"
export SOURCE_COMMIT MPI_HOST_RUNTIME
for name in ${!APPTAINERENV_HD_@}; do unset "${name}"; done
export PHASE_BENCH_ONLY=0
export APPTAINERENV_UCX_TLS=rc,sm,self APPTAINERENV_UCX_LOG_LEVEL=info
export APPTAINERENV_HD_MPI_PIPELINE=async APPTAINERENV_HD_MPI_ASYNC_PROFILE=baseline
export APPTAINERENV_HD_MPI_CHECKPOINT_WRITER=legacy
export VALIDATE_PIPELINES='async legacy' VALIDATE_PROFILES=baseline VALIDATE_WRITERS='legacy direct'
base="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "${scripts}" rev-parse --show-toplevel)"
cd "${repo}"
[[ "$(git rev-parse "${SOURCE_COMMIT}^{commit}")" == "${SOURCE_COMMIT}" ]]
test -x "${MPI_HOST_RUNTIME}/ld-linux-x86-64.so.2"
prior="$(squeue -h -u "${USER}" -p gr20100g -o '%A' | sort -u | paste -sd: -)"
dependencies=()
if [[ -n "${prior}" ]]; then dependencies+=(--dependency="afterany:${prior}"); fi
submit() {
    local id
    id="$(sbatch --parsable "$@")" || return $?
    [[ "${id%%;*}" =~ ^[0-9]+$ ]] || return 2
    echo "${id%%;*}"
}
safety="$(submit "${dependencies[@]}" "${scripts}/mpi_safety_validate.sbatch")"
echo "Safety: ${safety}"
build="$(submit --dependency="afterok:${safety}" --kill-on-invalid-dep=yes --rsc=g=1 \
    --time=02:00:00 --job-name=bgj-barriers-build \
    --export="ALL,BUILD_ONLY=1,REUSE_BUILD=${base}/diagnostics/mpi-safety-${safety}" \
    "${scripts}/mpi_bkz_validate.sbatch")"
echo "Build: ${build}"
integration="$(submit --dependency="afterok:${build}" --kill-on-invalid-dep=yes --time=04:00:00 \
    --export="ALL,BUILD_ONLY=0,REUSE_BUILD=${base}/diagnostics/mpi-bkz-${build}" \
    "${scripts}/mpi_bkz_validate.sbatch")"
echo "MPI integration: ${integration}"
export VALIDATED_BUILD="${base}/diagnostics/mpi-bkz-${integration}"
root="${base}/diagnostics/csd133-barriers-${safety}/checkpoint"
phase="$(submit --dependency="afterok:${integration}" --kill-on-invalid-dep=yes \
    --job-name=bgj-csd133-barriers-phase --time=06:00:00 \
    --export="ALL,RUN_MODE=phase,RUN_ROOT=${root},IMPORT_FROM=${base}/mpi-runs/seed38-bkz144-async-c970044-tour1,IMPORT_GENERATION=34,IMPORT_SOURCE_COMMIT=c970044a5bb715bc6ac69c76e47d49fa75a27c25" \
    "${scripts}/mpi_bkz144_async.sbatch")"
echo "CSD132 import + one CSD133 phase: ${phase}"
echo "Checkpoint root: ${root}"
echo "Source: ${SOURCE_COMMIT}"
echo "No production continuation submitted. Any failed gate cancels dependent jobs."
