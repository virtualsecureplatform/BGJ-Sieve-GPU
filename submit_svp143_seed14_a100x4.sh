#!/bin/bash

set -euo pipefail
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SVP_DIMENSION=143
export SVP142_LATTICE_SEED=14
export SVP_INSTANCE=svp143-seed14
export SVP142_REPRO="${SOURCE}/reproduce/svp143-seed14/reproduce.py"
export TARGET_SIEVING_DIM="${TARGET_SIEVING_DIM:-133}"
export MIN_LIFTING_DIM="${MIN_LIFTING_DIM:-119}"
exec "${SOURCE}/submit_svp142_seed0_a100x4.sh"
