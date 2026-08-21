#!/bin/bash

set -euo pipefail

if (( $# < 3 || $# > 5 )); then
    echo "usage: $0 SOURCE SOURCE_COMMIT FILTER_TASK_VECS [ENABLE_PROFILING] [POOL_HOST_VEC_MAX_DIM]" >&2
    exit 2
fi

SOURCE="$1"
SOURCE_COMMIT="$2"
FILTER_TASK_VECS="$3"
ENABLE_PROFILING="${4:-1}"
POOL_HOST_VEC_MAX_DIM="${5:-144}"
if [[ "$ENABLE_PROFILING" != 0 && "$ENABLE_PROFILING" != 1 ]]; then
    echo "ENABLE_PROFILING must be 0 or 1" >&2
    exit 2
fi
if [[ ! "$POOL_HOST_VEC_MAX_DIM" =~ ^[0-9]+$ ]] ||
   (( POOL_HOST_VEC_MAX_DIM < 1 || POOL_HOST_VEC_MAX_DIM > 176 )); then
    echo "POOL_HOST_VEC_MAX_DIM must be an integer in [1, 176]" >&2
    exit 2
fi
for tree in app include src; do
    git -C "$SOURCE" rev-parse "${SOURCE_COMMIT}:${tree}"
done
printf '%s\n' \
    "cache-schema=2" \
    "cuda-profile=cuda13.3-sm80" \
    "defines=HD_A100X4_500G_CACHE_PROFILE=1,HD_FILTER_TASK_VECS=${FILTER_TASK_VECS},ENABLE_PROFILING=${ENABLE_PROFILING}"
if (( POOL_HOST_VEC_MAX_DIM != 144 )); then
    printf 'pool-host-vec-max-dim=%s\n' "$POOL_HOST_VEC_MAX_DIM"
fi
