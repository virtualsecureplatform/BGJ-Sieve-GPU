#!/bin/bash
# Submit separate allocations: one-node serial, then two-node MPI.
set -euo pipefail
: "${SOURCE_COMMIT:?}" "${VALIDATED_BUILD:?}"
export SOURCE_COMMIT VALIDATED_BUILD
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
export BENCH_LAUNCH_COMMIT="${BENCH_LAUNCH_COMMIT:-$(git rev-parse HEAD)}"
SCRIPT="${SCRIPT_DIR}/mpi_phase_bench.sbatch"
serial_dependency=()
mpi_prerequisite=""
if [[ -n "${BENCH_VALIDATION_JOB:-}" ]]; then
    [[ "${BENCH_VALIDATION_JOB}" =~ ^[0-9]+$ ]] || { echo "Invalid BENCH_VALIDATION_JOB" >&2; exit 2; }
    serial_dependency=(--dependency="afterok:${BENCH_VALIDATION_JOB}")
    mpi_prerequisite="afterok:${BENCH_VALIDATION_JOB},"
fi
serial="$(sbatch --parsable --rsc=g=4 --job-name=bgj-serial-phase-bench \
    "${serial_dependency[@]}" --export=ALL,BENCH_BACKEND=serial "${SCRIPT}")"
serial="${serial%%;*}"
echo "Serial benchmark: ${serial} (one node)"
# Serial failure must not suppress the independent MPI measurement. Waiting
# for completion still guarantees that this pair uses at most two nodes.
if ! mpi="$(sbatch --parsable --rsc=g=8 --job-name=bgj-mpi-phase-bench \
    --dependency="${mpi_prerequisite}afterany:${serial}" --export=ALL,BENCH_BACKEND=mpi "${SCRIPT}")"; then
    echo "MPI submission failed; serial job ${serial} remains queued" >&2
    exit 1
fi
echo "MPI benchmark: ${mpi%%;*} (two nodes, after serial completion)"
