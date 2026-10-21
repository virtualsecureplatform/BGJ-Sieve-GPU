#!/bin/bash
set -euo pipefail
: "${MPI_BINARY:?}" "${MPI_SIF:?}" "${MPI_BASE:?}" "${MPI_LIBDIR:?}"
export APPTAINERENV_LD_LIBRARY_PATH="${MPI_LIBDIR}:/opt/system/app/nvhpc/23.9/Linux_x86_64/23.9/compilers/lib:/usr/local/cuda/lib64"
rdma_bind=()
if [[ "${HD_MPI_HOST_RDMA:-0}" == 1 ]]; then
    # Import a consistent host verbs/provider stack, not host glibc or the
    # entire /usr/lib64. UCX can otherwise silently fall back to TCP when its
    # libuct_ib plugin cannot resolve the Mellanox provider in the container.
    for soname in libibverbs.so.1 librdmacm.so.1 libmlx5.so.1 libnl-3.so.200 libnl-route-3.so.200; do
        library="$(readlink -e "/usr/lib64/${soname}")"
        rdma_bind+=(--bind "${library}:/opt/bgj-rdma/${soname}:ro")
    done
    rdma_bind+=(--bind /usr/lib64/libibverbs:/usr/lib64/libibverbs:ro
                --bind /etc/libibverbs.d:/etc/libibverbs.d:ro)
    export APPTAINERENV_LD_LIBRARY_PATH="/opt/bgj-rdma:${APPTAINERENV_LD_LIBRARY_PATH}"
fi
exec apptainer exec --nv --bind /opt/system,/dev/infiniband,"${MPI_BASE}" "${rdma_bind[@]}" \
    "${MPI_SIF}" "${MPI_BINARY}" "$@"
