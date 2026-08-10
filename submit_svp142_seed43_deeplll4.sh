#!/bin/bash

set -euo pipefail
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SVP142_LATTICE_SEED=43
export SVP_INSTANCE=svp142-seed43
export SVP142_REPRO="${SOURCE}/reproduce/svp142-search/reproduce.py"
export SVP_PREPROCESS_MODE=lll-deeplll-bkz
export SVP_DEEPLLL_DEPTH=4
export SVP_PREPROCESS_IMAGE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/bgj-sieve-gpu-cuda13.3.pre-b0c81ac.sif"
export GPU_UID_SHARED_LOAD_PCT=60
exec "${SOURCE}/submit_svp142_seed0_a100x4.sh"
