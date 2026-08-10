#!/bin/bash
#SBATCH --job-name=svp142-s42-gen
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=1
#SBATCH --time=00:20:00
#SBATCH --output=svp142-s42-gen_%j.log

set -euo pipefail

LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SOURCE="${SLURM_SUBMIT_DIR}"
IMAGE="${BGJ_IMAGE:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
PREFIX="${LARGE_BASE}/deps/ntl-9.3.0-svp-generator"
INPUT_DIR="${LARGE_BASE}/inputs/svp142-seed42"
ARCHIVE="${SOURCE}/dep/dist/ntl-9.3.0.tar.gz"
BUILDER="${SOURCE}/tools/build_legacy_ntl_generator.sh"
GENERATOR_SOURCE="${SOURCE}/tools/svp_challenge_generator.cpp"

mkdir -p "$INPUT_DIR"
apptainer exec --bind "$LARGE_BASE:$LARGE_BASE" --pwd "$SOURCE" "$IMAGE" \
    "$BUILDER" "$PREFIX" "$ARCHIVE" "$GENERATOR_SOURCE"
RAW_TMP="${INPUT_DIR}/L_142_42.raw.tmp-${SLURM_JOB_ID}"
apptainer exec --bind "$LARGE_BASE:$LARGE_BASE" "$IMAGE" \
    "${PREFIX}/bin/svp_challenge_generator" 142 42 > "$RAW_TMP"
mv -f "$RAW_TMP" "${INPUT_DIR}/L_142_42.raw"
sha256sum "${INPUT_DIR}/L_142_42.raw"
