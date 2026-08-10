#!/bin/bash

set -euo pipefail

if (( $# != 3 )); then
    echo "usage: $0 SOURCE SOURCE_COMMIT FILTER_TASK_VECS" >&2
    exit 2
fi

SOURCE="$1"
SOURCE_COMMIT="$2"
FILTER_TASK_VECS="$3"
for tree in app include src; do
    git -C "$SOURCE" rev-parse "${SOURCE_COMMIT}:${tree}"
done
printf '%s\n' \
    "cache-schema=1" \
    "cuda-profile=cuda13.3-sm80" \
    "defines=HD_A100X4_500G_CACHE_PROFILE=1,HD_FILTER_TASK_VECS=${FILTER_TASK_VECS}"
