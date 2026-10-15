#!/bin/bash
#SBATCH --job-name=svp167-seed38-mpi-bkz144-pilot
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=8
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@3600
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/mpi/slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${PILOT_ROOT:?}"
export MPI_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
export MPI_SIF="${MPI_BASE}/bgj-sieve-gpu-cuda13.3.sif"
BASIS="${MPI_BASE}/runs/svp167-search/potlll-simulation-8744618/seed-38/result.basis"
BUILD="${MPI_BASE}/builds/mpi-bkz144-pilot-${SLURM_JOB_ID}"
mkdir -p "${BUILD}/source" "${PILOT_ROOT}"
git -C "${SLURM_SUBMIT_DIR}" archive "${SOURCE_COMMIT}" | tar -x -C "${BUILD}/source"
SOURCE="${BUILD}/source"
MPI_CPPFLAGS="$(mpicxx --showme:compile)"
export MPI_LIBDIR="$(mpicxx --showme:link | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
FLAGS="-DHD_A100X4_500G_CACHE_PROFILE=1 -DHD_A100X4_PWC_GB=260 -DHD_A100X4_BWC_GB=128 -DHD_A100X4_SWC_GB=16 -DPOOL_HOST_VEC_MAX_DIM=144 -DENABLE_PROFILING=0"
apptainer exec --nv --bind /opt/system,"${MPI_BASE}" "${MPI_SIF}" bash -lc \
    "make -B -C '${SOURCE}/src' -j 16 CPPFLAGS='${FLAGS}' && make -B -C '${SOURCE}/app' -j 4 hd_sieve_mpi CPPFLAGS='${FLAGS}' MPI_CPPFLAGS='${MPI_CPPFLAGS}' MPI_LIBDIR='${MPI_LIBDIR}'"
export MPI_BINARY="${SOURCE}/app/hd_sieve_mpi"
export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx OMPI_MCA_pml_ucx_tls=any OMPI_MCA_pml_ucx_devices=any PMIX_MCA_gds=hash
export APPTAINERENV_HD_SOURCE_COMMIT="${SOURCE_COMMIT}"
export APPTAINERENV_HD_FAIL_ON_SPILL=1 APPTAINERENV_HD_LAZY_SYNC=1 APPTAINERENV_HD_BWC_LAZY_SYNC=1
export APPTAINERENV_HD_HBM_PWC_WRITEBACK=1 APPTAINERENV_HD_HBM_PWC_GB=0
export APPTAINERENV_HD_HBM_BWC_GB=12
export APPTAINERENV_HD_HOST_ARENA_INITIAL_GB=64 APPTAINERENV_HD_PWC_NO_GROW=1
export APPTAINERENV_HD_POOL_BUFFER_CACHE=0 APPTAINERENV_HD_CACHE_WAIT_TIMEOUT_S=600
export APPTAINERENV_HD_BGJ2_REDUCER_THREADS=16 APPTAINERENV_HD_BUCKET_THREADS=16
# Start with the BGJ2 path already exercised at CSD143, rather than the
# disk-oriented BGJ3L crossover. Pool size and saturation goals are unchanged.
export APPTAINERENV_HD_BGJ3_MIN_CSD=177
if [[ "${PILOT_RESUME:-0}" == 1 ]]; then
    [[ -f "${PILOT_ROOT}/LATEST-BKZ" ]]
    export APPTAINERENV_HD_MPI_BKZ_MAX_PHASES=1
    unset APPTAINERENV_HD_MPI_BKZ_STOP_AFTER || true
else
    [[ ! -f "${PILOT_ROOT}/LATEST-BKZ" ]]
    export APPTAINERENV_HD_MPI_BKZ_STOP_AFTER=first-down
    unset APPTAINERENV_HD_MPI_BKZ_MAX_PHASES || true
fi
echo "pilot root=${PILOT_ROOT} source=${SOURCE_COMMIT} resume=${PILOT_RESUME:-0} basis=${BASIS}"
STEP_PID=''
trap 'if [[ -n "${STEP_PID}" ]]; then kill -USR1 "${STEP_PID}" || true; fi' USR1
set +e
srun --mpi=pmix_v2 --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=128 \
    --cpu-bind=none --kill-on-bad-exit=1 \
    "${SOURCE}/reproduce/svp167-search/mpi_rank.sh" --input "${BASIS}" \
    --task pump --ind_l 0 --ind_r 167 --SSD 100 --MSD 144 --DS \
    --output "${PILOT_ROOT}/pilot.basis" --checkpoint-root "${PILOT_ROOT}" --resume auto \
    --mpi-frame-mib 256 &
STEP_PID=$!
wait "${STEP_PID}"
solver_rc=$?
# A signal interrupts bash wait without necessarily finishing the MPI step.
if kill -0 "${STEP_PID}" 2>/dev/null; then wait "${STEP_PID}"; solver_rc=$?; fi
set -e
if [[ "${solver_rc}" == 75 && -f "${PILOT_ROOT}/LATEST-BKZ" ]]; then
    echo "PASS: pilot paused at a committed phase; no full production chain launched"
    exit 0
fi
exit "${solver_rc}"
