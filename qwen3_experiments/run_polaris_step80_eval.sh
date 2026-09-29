#!/usr/bin/env bash
# Run on eight GPUs; completed responses survive preemption/restarts.
set -euo pipefail
umask 077
[[ $# -eq 2 ]] || { echo "Usage: $0 PYTHON OUTPUT_ROOT" >&2; exit 2; }
EVAL_PYTHON=$1
EVAL_ROOT=$2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
unset PYTHONHOME RAY_ADDRESS ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH=${REPO_ROOT}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_V1=0
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR=/tmp/step80-eval-${SLURM_JOB_ID:-local}
export TRITON_CACHE_DIR=${TMPDIR}/triton
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${EVAL_ROOT}"
if [[ -n ${STEP80_HOLDER_LOCK:-} ]]; then
    exec 9>"${STEP80_HOLDER_LOCK}"
    flock -n 9 || { echo "Another workload owns this holder." >&2; exit 1; }
fi
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${busy} ]] || { echo "GPUs are occupied; waiting work must finish first." >&2; exit 1; }
cd "${REPO_ROOT}"
"${EVAL_PYTHON}" -u "${SCRIPT_DIR}/eval_polaris_step80.py" prepare-model --output-root "${EVAL_ROOT}"
exec "${EVAL_PYTHON}" -u "${SCRIPT_DIR}/eval_polaris_step80.py" run --output-root "${EVAL_ROOT}"
