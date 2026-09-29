#!/usr/bin/env bash
set -euo pipefail
umask 077
EVAL_PYTHON=$1
EVAL_ROOT=$2
cd "${EVAL_ROOT}"
mapfile -t DIFFICULTY_HOLDER_LOCKS < <("${EVAL_PYTHON}" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["holder_locks"]))' "${EVAL_ROOT}/plan.json")
declare -a DIFFICULTY_LOCK_FDS=()
for DIFFICULTY_HOLDER_LOCK in "${DIFFICULTY_HOLDER_LOCKS[@]}"; do
    exec {DIFFICULTY_LOCK_FD}>"${DIFFICULTY_HOLDER_LOCK}"
    flock -n "${DIFFICULTY_LOCK_FD}" || exit 75
    DIFFICULTY_LOCK_FDS+=("${DIFFICULTY_LOCK_FD}")
done
DIFFICULTY_BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${DIFFICULTY_BUSY} ]] || exit 75
unset PYTHONHOME RAY_ADDRESS ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH
PYTHONPATH=$("${EVAL_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repository"])' "${EVAL_ROOT}/plan.json")
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_V1=0
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR=/tmp/training-difficulty-${SLURM_JOB_ID:-local}-${SLURM_STEP_ID:-local}
export TRITON_CACHE_DIR=${TMPDIR}/triton
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}"
DIFFICULTY_EXIT=0
"${EVAL_PYTHON}" -u "${EVAL_ROOT}/provenance/eval_training_difficulty.py" run --output-root "${EVAL_ROOT}" || DIFFICULTY_EXIT=$?
printf '%s\n' "${DIFFICULTY_EXIT}" >"${EVAL_ROOT}/exit_status"
exit "${DIFFICULTY_EXIT}"
