#!/usr/bin/env bash
# Acquire the same allocation lock after the predecessor's audited completion.
set -euo pipefail
umask 077
[[ $# -eq 4 ]] || { echo "Usage: $0 PYTHON OUTPUT_ROOT PREDECESSOR_ROOT HOLDER_LOCK" >&2; exit 2; }
REFERENCE_PYTHON=$1
REFERENCE_ROOT=$2
PREDECESSOR_ROOT=$3
REFERENCE_HOLDER_LOCK=$4
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
exec 9>"${REFERENCE_HOLDER_LOCK}"
flock 9
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${busy} ]] || { echo "GPUs remain occupied; refusing to overlap." >&2; exit 1; }
unset PYTHONHOME RAY_ADDRESS ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH=${REPO_ROOT}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_V1=0
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR=/tmp/qwen3-reference-eval-${SLURM_JOB_ID:-local}
export TRITON_CACHE_DIR=${TMPDIR}/triton
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}"
cd "${REPO_ROOT}"
exec "${REFERENCE_PYTHON}" -u "${SCRIPT_DIR}/eval_qwen3_reference.py" run \
    --output-root "${REFERENCE_ROOT}" --predecessor-root "${PREDECESSOR_ROOT}"
