#!/bin/bash
# Strictly sequential compute jobs. Production cannot start without every
# correctness gate and a source-bound report selecting measured-safe options.
set -euo pipefail
: "${SOURCE_COMMIT:?}" "${MPI_HOST_RUNTIME:?}"
export SOURCE_COMMIT MPI_HOST_RUNTIME
for name in ${!APPTAINERENV_HD_MPI_@}; do unset "${name}"; done
unset APPTAINERENV_HD_SIZE_RATIO APPTAINERENV_HD_INT4_BUCKETS
export PHASE_BENCH_ONLY=0
export APPTAINERENV_UCX_TLS=rc,sm,self APPTAINERENV_UCX_LOG_LEVEL=info
export APPTAINERENV_HD_MPI_PIPELINE=async APPTAINERENV_HD_MPI_ASYNC_PROFILE=baseline
export APPTAINERENV_HD_MPI_CHECKPOINT_WRITER=legacy
export VALIDATE_PIPELINES='async legacy' VALIDATE_PROFILES='baseline optimized' VALIDATE_WRITERS='legacy direct'
base="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
prior="$(squeue -h -u "${USER}" -p gr20100g -o '%A' | sort -u | paste -sd: -)"
dependencies=()
if [[ -n "${prior}" ]]; then dependencies+=(--dependency="afterany:${prior}"); fi
submit() {
    local id
    id="$(sbatch --parsable "$@")" || return $?
    [[ "${id%%;*}" =~ ^[0-9]+$ ]] || return 2
    echo "${id%%;*}"
}
SAFETY_JOB="$(submit "${dependencies[@]}" "${scripts}/mpi_safety_validate.sbatch")"
export SAFETY_JOB
echo "Safety: ${SAFETY_JOB}"
build="$(submit --dependency="afterok:${SAFETY_JOB}" --kill-on-invalid-dep=yes --rsc=g=1 \
    --time=02:00:00 --job-name=bgj-rollout-build \
    --export="ALL,BUILD_ONLY=1,REUSE_BUILD=${base}/diagnostics/mpi-safety-${SAFETY_JOB}" \
    "${scripts}/mpi_bkz_validate.sbatch")"
echo "Build: ${build}"
INTEGRATION_JOB="$(submit --dependency="afterok:${build}" --kill-on-invalid-dep=yes --time=04:00:00 \
    --export="ALL,BUILD_ONLY=0,REUSE_BUILD=${base}/diagnostics/mpi-bkz-${build}" \
    "${scripts}/mpi_bkz_validate.sbatch")"
export INTEGRATION_JOB
echo "Full integration: ${INTEGRATION_JOB}"
export VALIDATED_BUILD="${base}/diagnostics/mpi-bkz-${INTEGRATION_JOB}"
export REPLAY_ASYNC_PROFILES=1 REPLAY_PERF_SOFT_GATE=1
REPLAY_JOB="$(submit --dependency="afterok:${INTEGRATION_JOB}" --kill-on-invalid-dep=yes \
    "${scripts}/mpi_pipeline_replay.sbatch")"
export REPLAY_JOB
echo "Paired async replay: ${REPLAY_JOB}"
release_root="${base}/diagnostics/two-node-release-${SAFETY_JOB}"
export PHASE_ROOT="${release_root}/csd133"
original="${base}/mpi-runs/seed38-bkz144-async-c970044-tour1"
original_commit=c970044a5bb715bc6ac69c76e47d49fa75a27c25
PHASE_JOB="$(submit --dependency="afterok:${REPLAY_JOB}" --kill-on-invalid-dep=yes \
    --job-name=bgj-csd133-verified --time=06:00:00 \
    --export="ALL,RUN_MODE=phase,RUN_ROOT=${PHASE_ROOT},IMPORT_FROM=${original},IMPORT_GENERATION=34,IMPORT_SOURCE_COMMIT=${original_commit}" \
    "${scripts}/mpi_bkz144_async.sbatch")"
export PHASE_JOB
echo "Verified CSD132 import + CSD133 phase: ${PHASE_JOB}"
CHECKPOINT_JOB="$(submit --dependency="afterok:${PHASE_JOB}" --kill-on-invalid-dep=yes \
    --job-name=bgj-checkpoint-compare --time=04:00:00 \
    --export="ALL,RUN_MODE=checkpoint-bench,RUN_ROOT=${release_root}/checkpoints,IMPORT_FROM=${original},IMPORT_GENERATION=34,IMPORT_SOURCE_COMMIT=${original_commit}" \
    "${scripts}/mpi_bkz144_async.sbatch")"
export CHECKPOINT_JOB
echo "Paired checkpoint benchmark: ${CHECKPOINT_JOB}"
export RELEASE_REPORT="${release_root}/report.json"
cold="$(submit --dependency="afterok:${CHECKPOINT_JOB}" --kill-on-invalid-dep=yes \
    "${scripts}/mpi_cold_compare.sbatch")"
echo "Matched cold CSD128 + release report: ${cold}"
production="$(submit --dependency="afterok:${cold}" --kill-on-invalid-dep=yes \
    --export="ALL,RUN_MODE=tour,RUN_ROOT=${base}/mpi-runs/seed38-bkz144-release-${SAFETY_JOB}" \
    "${scripts}/mpi_bkz144_async.sbatch")"
echo "Gated one-tour continuation: ${production}"
echo "Release report: ${RELEASE_REPORT}"
