#!/bin/bash
#SBATCH --job-name=bgj-mpi-bkz-validate
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=8
#SBATCH --time=02:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/mpi/slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}"
export MPI_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
export MPI_SIF="${MPI_BASE}/bgj-sieve-gpu-cuda13.3.sif"
RUN="${MPI_BASE}/diagnostics/mpi-bkz-${SLURM_JOB_ID}"
mkdir -p "${RUN}/source" "${RUN}/tests"
git -C "${SLURM_SUBMIT_DIR}" archive "${SOURCE_COMMIT}" | tar -x -C "${RUN}/source"
SOURCE="${RUN}/source"
MPI_CPPFLAGS="$(mpicxx --showme:compile)"
export MPI_LIBDIR="$(mpicxx --showme:link | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
FLAGS="-DHD_A100X4_500G_CACHE_PROFILE=1 -DHD_A100X4_PWC_GB=260 -DHD_A100X4_BWC_GB=128 -DHD_A100X4_SWC_GB=16 -DPOOL_HOST_VEC_MAX_DIM=144 -DENABLE_PROFILING=0"
REUSED=0
if [[ -n "${REUSE_BUILD:-}" && -f "${REUSE_BUILD}/source/src/libllib.a" ]]; then
    # Reuse only an identical library source/configuration, never merely newer mtimes.
    [[ "$(sed -n '/^FLAGS=/p' "${SOURCE}/reproduce/svp167-search/mpi_bkz_validate.sbatch")" == \
       "$(sed -n '/^FLAGS=/p' "${REUSE_BUILD}/source/reproduce/svp167-search/mpi_bkz_validate.sbatch")" ]]
    while IFS= read -r path; do
        cmp "${SOURCE}/${path}" "${REUSE_BUILD}/source/${path}"
    done < <(git -C "${SLURM_SUBMIT_DIR}" ls-tree -r --name-only "${SOURCE_COMMIT}" src include)
    cp "${REUSE_BUILD}/source/src/libllib.a" "${SOURCE}/src/libllib.a"
    REUSED=1
fi
if [[ "${REUSED}" == 0 ]]; then
    apptainer exec --nv --bind /opt/system,"${MPI_BASE}" "${MPI_SIF}" bash -lc \
        "make -B -C '${SOURCE}/src' -j 16 CPPFLAGS='${FLAGS}'"
fi
apptainer exec --nv --bind /opt/system,"${MPI_BASE}" "${MPI_SIF}" bash -lc \
    "make -B -C '${SOURCE}/app' -j 4 hd_sieve hd_sieve_mpi CPPFLAGS='${FLAGS}' MPI_CPPFLAGS='${MPI_CPPFLAGS}' MPI_LIBDIR='${MPI_LIBDIR}'"
if [[ "${BUILD_ONLY:-0}" == 1 ]]; then
    echo "PASS: MPI and serial binaries compiled; source=${SOURCE_COMMIT} run=${RUN}"
    exit 0
fi
export MPI_BINARY="${SOURCE}/app/hd_sieve_mpi"
export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx OMPI_MCA_pml_ucx_tls=any OMPI_MCA_pml_ucx_devices=any PMIX_MCA_gds=hash
export APPTAINERENV_HD_SOURCE_COMMIT="${SOURCE_COMMIT}"
export APPTAINERENV_HD_MPI_CRASH_TRACE=1
export APPTAINERENV_HD_MPI_BKZ_CHECK_PHASES=1
export APPTAINERENV_HD_FAIL_ON_SPILL=1 APPTAINERENV_HD_LAZY_SYNC=1 APPTAINERENV_HD_BWC_LAZY_SYNC=1
export APPTAINERENV_HD_HBM_PWC_WRITEBACK=1 APPTAINERENV_HD_HBM_PWC_GB=0
export APPTAINERENV_HD_HOST_ARENA_INITIAL_GB=4 APPTAINERENV_HD_PWC_NO_GROW=1
export APPTAINERENV_HD_POOL_BUFFER_CACHE=0 APPTAINERENV_HD_CACHE_WAIT_TIMEOUT_S=120
export APPTAINERENV_HD_BGJ2_REDUCER_THREADS=16 APPTAINERENV_HD_BUCKET_THREADS=16
export APPTAINERENV_HD_BGJ3_MIN_CSD=177
GENERATOR="${MPI_BASE}/deps/ntl-9.3.0-svp-generator/bin/svp_challenge_generator"
apptainer exec --bind "${MPI_BASE}" "${MPI_SIF}" "${GENERATOR}" 80 0 > "${RUN}/tests/input.raw"
FPLLL="${MPI_BASE}/deps/fplll-b0e38e31085e"
apptainer exec --bind "${MPI_BASE}" "${MPI_SIF}" env OMP_NUM_THREADS=1 LD_LIBRARY_PATH="${FPLLL}/lib" \
    "${FPLLL}/bin/fplll" -a potlll -m heuristic -f mpfr -p 256 "${RUN}/tests/input.raw" > "${RUN}/tests/input.basis"
python3 "${SOURCE}/tools/test_mpi_bkz.py" --wrapper "${SOURCE}/reproduce/svp167-search/mpi_rank.sh" \
    --basis "${RUN}/tests/input.basis" --root "${RUN}/tests"
echo "PASS: distributed BKZ validation source=${SOURCE_COMMIT} run=${RUN}"
