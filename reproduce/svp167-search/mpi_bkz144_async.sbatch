#!/bin/bash
# One tour only; resubmit with the same RUN_ROOT/source to resume a saved phase.
#SBATCH --job-name=svp167-s38-async-bkz144
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=8
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@3600
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/mpi/slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${VALIDATED_BUILD:?}" "${MPI_HOST_RUNTIME:?}" "${RUN_ROOT:?}"
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" == 2 ]]
export MPI_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
export MPI_SIF="${MPI_BASE}/bgj-sieve-gpu-cuda13.3.sif"
SOURCE="${VALIDATED_BUILD}/source"
BASIS="${MPI_BASE}/runs/svp167-search/potlll-simulation-8744618/seed-38/result.basis"
# Fail closed if any executable input differs from the validated revision.
while IFS= read -r path; do
    git -C "${SLURM_SUBMIT_DIR}" show "${SOURCE_COMMIT}:${path}" | cmp - "${SOURCE}/${path}"
done < <(git -C "${SLURM_SUBMIT_DIR}" ls-tree -r --name-only "${SOURCE_COMMIT}" src include app)
test -x "${SOURCE}/app/hd_sieve_mpi"
test -x "${MPI_HOST_RUNTIME}/ld-linux-x86-64.so.2"
test -s "${BASIS}"
export MPI_BINARY="${SOURCE}/app/hd_sieve_mpi"
export MPI_LIBDIR="$(mpicxx --showme:link | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx OMPI_MCA_pml_ucx_tls=any
export OMPI_MCA_pml_ucx_devices=any PMIX_MCA_gds=hash
export APPTAINERENV_UCX_TLS=rc,sm,self APPTAINERENV_UCX_LOG_LEVEL=info
export APPTAINERENV_HD_SOURCE_COMMIT="${SOURCE_COMMIT}"
export APPTAINERENV_HD_MPI_PIPELINE=async APPTAINERENV_HD_MPI_CRASH_TRACE=1
export APPTAINERENV_HD_FAIL_ON_SPILL=1
export APPTAINERENV_HD_LAZY_SYNC=1 APPTAINERENV_HD_BWC_LAZY_SYNC=1
export APPTAINERENV_HD_HBM_PWC_WRITEBACK=1 APPTAINERENV_HD_HBM_PWC_GB=0
export APPTAINERENV_HD_HBM_BWC_GB=12
export APPTAINERENV_HD_HOST_ARENA_INITIAL_GB=64 APPTAINERENV_HD_PWC_NO_GROW=1
export APPTAINERENV_HD_EAGER_HOST_ARENA=1 APPTAINERENV_HD_POOL_BUFFER_CACHE=0
export APPTAINERENV_HD_CACHE_WAIT_TIMEOUT_S=600
export APPTAINERENV_HD_POOL_THREADS=8
export APPTAINERENV_HD_BGJ2_REDUCER_THREADS=16 APPTAINERENV_HD_BUCKET_THREADS=16
export APPTAINERENV_HD_BGJ3_MIN_CSD=177
export APPTAINERENV_HD_RSS_TELEMETRY=1 APPTAINERENV_HD_HBM_TELEMETRY=1
# Never inherit diagnostic stops, fault injection, or reduced-quality pool sizes.
unset APPTAINERENV_HD_MPI_BKZ_MAX_PHASES APPTAINERENV_HD_MPI_BKZ_STOP_AFTER
unset APPTAINERENV_HD_MPI_BKZ_FAULT APPTAINERENV_HD_MPI_SPILL_TEST
unset APPTAINERENV_HD_MPI_CONTROL_TEST APPTAINERENV_HD_MPI_PIPELINE_TEST_CORRUPT
unset APPTAINERENV_HD_MPI_CANDIDATE_QUEUE_KIB APPTAINERENV_HD_SIZE_RATIO
unset APPTAINERENV_HD_MPI_CHECKPOINT_BENCHMARK
unset APPTAINERENV_HD_MPI_CHECKPOINT_IO_FAULT
unset APPTAINERENV_HD_MPI_PIPELINE_REPLAY APPTAINERENV_HD_MPI_PIPELINE_CACHE_TEST
unset APPTAINERENV_HD_MPI_PIPELINE_TEST_DELAY_US APPTAINERENV_HD_INT4_BUCKETS
mode="${RUN_MODE:-tour}"
case "${mode}" in
    import|phase|checkpoint-bench) ;;
    tour)
        : "${RELEASE_REPORT:?Production requires a passed release report}"
        mapfile -t release < <(python3 - "${RELEASE_REPORT}" "${SOURCE_COMMIT}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report['source_commit'] == sys.argv[2] and report['passed'] is True
assert report['async_profile'] in ('baseline','optimized')
assert report['checkpoint_writer'] in ('legacy','direct')
for field in ('async_profile','checkpoint_writer','checkpoint_root','checkpoint_generation','source_commit'):
    print(report[field])
PY
        )
        [[ "${#release[@]}" == 5 ]]
        export APPTAINERENV_HD_MPI_ASYNC_PROFILE="${release[0]}"
        export APPTAINERENV_HD_MPI_CHECKPOINT_WRITER="${release[1]}"
        unset IMPORT_FROM IMPORT_GENERATION IMPORT_SOURCE_COMMIT
        if [[ ! -f "${RUN_ROOT}/LATEST-BKZ" ]]; then
            export IMPORT_FROM="${release[2]}" IMPORT_GENERATION="${release[3]}" IMPORT_SOURCE_COMMIT="${release[4]}"
        fi
        ;;
    *) echo "Unknown RUN_MODE=${mode}" >&2; exit 2 ;;
esac
echo "source=${SOURCE_COMMIT}; validated_build=${VALIDATED_BUILD}; runtime=${MPI_HOST_RUNTIME}"
echo "run=${RUN_ROOT}; input=${BASIS}; output=${RUN_ROOT}/tour1.basis"
echo "One BKZ tour: BSD=144 D4F=23 jump=9 starts=0,9,18,27; dual hash disabled."
echo "Per node: PWC=260GiB BWC=128GiB SWC=16GiB; HBM BWC=12GiB/GPU."
echo "Async pinned-host RDMA; spills forbidden, durable phase checkpoint writes allowed."
sha256sum "${BASIS}" "${MPI_BINARY}"
STEP_PID=''
trap 'if [[ -n "${STEP_PID}" ]]; then kill -USR1 "${STEP_PID}" || true; fi' USR1 TERM
run_step() {
set +e
srun --mpi=pmix_v2 --immediate=60 --nodes=2 --ntasks=2 --gpus=8 --gres=gpu:4 \
    --ntasks-per-node=1 --cpus-per-task=128 --cpu-bind=none --kill-on-bad-exit=1 \
    "${SOURCE}/reproduce/svp167-search/mpi_rank.sh" \
    --input "${BASIS}" --task bkz --BSD 144 --D4F 23 --JUMP 9 --BDH 0 --STI 0 \
    --output "${RUN_ROOT}/tour1.basis" --checkpoint-root "${RUN_ROOT}" \
    --resume auto --mpi-frame-mib 256 "$@" &
STEP_PID=$!
wait "${STEP_PID}"
solver_rc=$?
if kill -0 "${STEP_PID}" 2>/dev/null; then wait "${STEP_PID}"; solver_rc=$?; fi
set -e
}
if [[ -n "${IMPORT_FROM:-}" ]]; then
    : "${IMPORT_GENERATION:?}" "${IMPORT_SOURCE_COMMIT:?}"
    [[ ! -e "${RUN_ROOT}" ]]
    run_step --import-checkpoint "${IMPORT_FROM}" --import-generation "${IMPORT_GENERATION}" \
        --import-source-commit "${IMPORT_SOURCE_COMMIT}"
    [[ "${solver_rc}" == 0 ]] || exit "${solver_rc}"
    if [[ "${mode}" == import ]]; then echo "PASS: import source=${SOURCE_COMMIT}"; exit 0; fi
elif [[ "${mode}" == import ]]; then
    echo "Import mode requires IMPORT_FROM" >&2; exit 2
fi
if [[ "${mode}" == phase ]]; then
    export APPTAINERENV_HD_MPI_BKZ_MAX_PHASES=1 APPTAINERENV_HD_MPI_BKZ_CHECK_PHASES=1
elif [[ "${mode}" == checkpoint-bench ]]; then
    export APPTAINERENV_HD_MPI_CHECKPOINT_BENCHMARK=1
fi
run_step
if [[ "${mode}" == phase || "${mode}" == checkpoint-bench ]]; then
    [[ "${solver_rc}" == 75 && -s "${RUN_ROOT}/LATEST-BKZ" ]]
    if [[ "${mode}" == phase ]]; then
        python3 - "${RUN_ROOT}" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]); sequence = int((root/'LATEST-BKZ').read_text())
states = [(root/('generation-%d/rank%d/STATE' % (sequence,rank))).read_text().split()
          for rank in (0,1)]
assert all(int(state[9]) == 133 and int(state[11]) == 1 for state in states)
assert states[0][6:] == states[1][6:]
print('PASS: verified CSD133 phase checkpoint')
PY
        echo "PASS: CSD133 source=${SOURCE_COMMIT} root=${RUN_ROOT}"
    else
        echo "PASS: paired checkpoint measurements source=${SOURCE_COMMIT} root=${RUN_ROOT}"
    fi
    exit 0
fi
if [[ "${solver_rc}" == 75 && -s "${RUN_ROOT}/LATEST-BKZ" ]]; then
    echo "PAUSED: resume the same root/source; only completed phases are committed."
    exit 0
fi
[[ "${solver_rc}" == 0 ]] || exit "${solver_rc}"
test -s "${RUN_ROOT}/tour1.basis"
apptainer exec --bind "${MPI_BASE}" "${MPI_SIF}" python3 \
    "${SOURCE}/tools/check_basis_det_mod.py" "${BASIS}" "${RUN_ROOT}/tour1.basis"
echo "PASS: one BKZ-144 tour completed; no second tour automatically launched."
