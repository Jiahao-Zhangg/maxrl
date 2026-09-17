#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "error: expected a CUDA compiler command" >&2
    exit 2
fi

compiler=$1
shift

# CUDA 12.6 ptxas segfaults at its default optimization level for some vLLM
# FlashAttention-3 FP8 specializations on Linux aarch64. Keep the workaround
# limited to that source family so every other CUDA object retains its
# upstream flags.
for argument in "$@"; do
    if [[ "${argument}" == */hopper/instantiations/flash_fwd_*e4m3*_sm90.cu ]]; then
        exec "${compiler}" "$@" -Xptxas=-O1
    fi
done

exec "${compiler}" "$@"
