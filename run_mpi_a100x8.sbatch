#!/bin/bash
#SBATCH --job-name=bgj-mpi-a100x8
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=8
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/mpi/slurm-%x-%j.out

set -euo pipefail

: "${BASIS:?export BASIS=/absolute/path/to/preprocessed-basis}"
TSD="${TSD:-149}"
MLD="${MLD:-120}"
POOL_HOST_VEC_MAX_DIM="${POOL_HOST_VEC_MAX_DIM:-${TSD}}"
MPI_FRAME_MIB="${MPI_FRAME_MIB:-256}"
RESUME="${RESUME:-auto}"
SOURCE="${SLURM_SUBMIT_DIR}"
SIF="${SIF:-/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/bgj-sieve-gpu-cuda13.3.sif}"
RUN_ROOT="${RUN_ROOT:-/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/mpi-runs/${SLURM_JOB_NAME}}"
BUILD_ROOT="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/mpi-build/${SLURM_JOB_ID}"
LOG_ROOT="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/logs/mpi"

mkdir -p "${BUILD_ROOT}" "${RUN_ROOT}" "${LOG_ROOT}"

if [[ ! "${POOL_HOST_VEC_MAX_DIM}" =~ ^[0-9]+$ ]] ||
   (( POOL_HOST_VEC_MAX_DIM < TSD || POOL_HOST_VEC_MAX_DIM > 176 )); then
    echo "POOL_HOST_VEC_MAX_DIM must be an integer in [TSD, 176]" >&2
    exit 2
fi
BUILD_CPPFLAGS="-DPOOL_HOST_VEC_MAX_DIM=${POOL_HOST_VEC_MAX_DIM}"

# Use the container's CUDA/NTL toolchain but the cluster MPI headers and
# libraries. No second MPI implementation is installed in the image.
MPI_CPPFLAGS="$(mpicxx --showme:compile)"
MPI_LIBDIR="$(mpicxx --showme:link | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
apptainer exec --nv \
    --bind /opt/system,"${SOURCE}","${BUILD_ROOT}" \
    "${SIF}" bash -lc \
    "make -B -C '${SOURCE}/src' -j 16 CPPFLAGS='${BUILD_CPPFLAGS}' && make -B -C '${SOURCE}/app' -j 4 hd_sieve_mpi CPPFLAGS='${BUILD_CPPFLAGS}' MPI_CPPFLAGS='${MPI_CPPFLAGS}' MPI_LIBDIR='${MPI_LIBDIR}' && cp '${SOURCE}/app/hd_sieve_mpi' '${BUILD_ROOT}/'"

: "${UCX_LOG_LEVEL:=warn}"
export UCX_LOG_LEVEL
export OMPI_MCA_pml="${MPI_PML:-ucx}"
export OMPI_MCA_osc="${MPI_OSC:-ucx}"
export OMPI_MCA_pml_ucx_tls="${MPI_UCX_TLS:-any}"
export OMPI_MCA_pml_ucx_devices="${MPI_UCX_DEVICES:-any}"
if [[ "${OMPI_MCA_pml}" == "ob1" ]]; then
    export OMPI_MCA_btl=self,vader,tcp
    export OMPI_MCA_osc=pt2pt
fi
export OMPI_MCA_mca_base_component_show_load_errors=1
export PMIX_MCA_gds=hash
export APPTAINERENV_LD_LIBRARY_PATH="${MPI_LIBDIR}:/opt/system/app/nvhpc/23.9/Linux_x86_64/23.9/compilers/lib:/usr/local/cuda/lib64"
export HD_PWC_NO_GROW=1
export HD_LAZY_SYNC=1

echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} basis=${BASIS} TSD=${TSD} MLD=${MLD} host_vec_max=${POOL_HOST_VEC_MAX_DIM}"
set +e
srun --mpi=pmix_v2 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    apptainer exec --nv \
    --bind /opt/system,/dev/infiniband,"${BUILD_ROOT}","${RUN_ROOT}","$(dirname "${BASIS}")" \
    "${SIF}" "${BUILD_ROOT}/hd_sieve_mpi" \
    --input "${BASIS}" --task sieve --TSD "${TSD}" --MLD "${MLD}" \
    --checkpoint-root "${RUN_ROOT}" --resume "${RESUME}" \
    --mpi-frame-mib "${MPI_FRAME_MIB}"
solver_rc=$?
set -e

# Rank zero owns the main Slurm stream. Surface rank one's lift/vector lines as
# well so the global result can be read without opening rank-local logs.
if [[ "${RESUME}" == "none" ]]; then
    EFFECTIVE_RUN_ROOT="${RUN_ROOT}/run-${SLURM_JOB_ID}"
else
    EFFECTIVE_RUN_ROOT="${RUN_ROOT}"
fi
if [[ -f "${EFFECTIVE_RUN_ROOT}/rank1/rank.log" ]]; then
    grep -E '\[pos |min_lift|lift.*norm|shorter vector' \
        "${EFFECTIVE_RUN_ROOT}/rank1/rank.log" | sed 's/^/[MPI rank 1] /' || true
fi
exit "${solver_rc}"
