#!/bin/bash
#SBATCH --job-name=bgj-async-cold-compare
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=8
#SBATCH --time=08:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/mpi/slurm-%x-%j.out
set -euo pipefail
: "${SOURCE_COMMIT:?}" "${VALIDATED_BUILD:?}" "${MPI_HOST_RUNTIME:?}"
export BENCH_BACKEND=mpi BENCH_DIMENSIONS=128 BENCH_STACK_INSPECTION=0
export BENCH_TIMEOUT_SECONDS=10800 BENCH_LAUNCH_COMMIT="${SOURCE_COMMIT}"
export APPTAINERENV_HD_MPI_PIPELINE=async
for profile in baseline optimized; do
    export APPTAINERENV_HD_MPI_ASYNC_PROFILE="${profile}" BENCH_LABEL="${profile}"
    SECONDS=0
    bash "${VALIDATED_BUILD}/source/reproduce/svp167-search/mpi_phase_bench.sbatch"
    echo "[MPI-COLD-COMPARE] profile=${profile} elapsed_seconds=${SECONDS}"
done
if [[ -n "${RELEASE_REPORT:-}" ]]; then
    : "${SAFETY_JOB:?}" "${INTEGRATION_JOB:?}" "${PHASE_JOB:?}" "${PHASE_ROOT:?}" "${REPLAY_JOB:?}" "${CHECKPOINT_JOB:?}"
    base="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
    python3 "${VALIDATED_BUILD}/source/tools/two_node_release.py" \
        --source-commit "${SOURCE_COMMIT}" \
        --safety-log "${base}/logs/mpi/slurm-bgj-memory-safety-${SAFETY_JOB}.out" \
        --integration-log "${base}/logs/mpi/slurm-bgj-mpi-bkz-validate-${INTEGRATION_JOB}.out" \
        --phase-log "${base}/logs/mpi/slurm-bgj-csd133-verified-${PHASE_JOB}.out" \
        --phase-root "${PHASE_ROOT}" \
        --replay-summary "${base}/diagnostics/mpi-pipeline-replay-${REPLAY_JOB}/summary.json" \
        --checkpoint-log "${base}/logs/mpi/slurm-bgj-checkpoint-compare-${CHECKPOINT_JOB}.out" \
        --cold-log "${base}/logs/mpi/slurm-bgj-async-cold-compare-${SLURM_JOB_ID}.out" \
        --baseline-root "${base}/diagnostics/mpi-phase-bench-${SLURM_JOB_ID}-baseline" \
        --optimized-root "${base}/diagnostics/mpi-phase-bench-${SLURM_JOB_ID}-optimized" \
        --output "${RELEASE_REPORT}"
fi
