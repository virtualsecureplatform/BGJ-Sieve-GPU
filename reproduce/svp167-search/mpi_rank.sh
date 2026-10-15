#!/bin/bash
set -euo pipefail
: "${MPI_BINARY:?}" "${MPI_SIF:?}" "${MPI_BASE:?}" "${MPI_LIBDIR:?}"
export APPTAINERENV_LD_LIBRARY_PATH="${MPI_LIBDIR}:/opt/system/app/nvhpc/23.9/Linux_x86_64/23.9/compilers/lib:/usr/local/cuda/lib64"
exec apptainer exec --nv --bind /opt/system,/dev/infiniband,"${MPI_BASE}" \
    "${MPI_SIF}" "${MPI_BINARY}" "$@"
