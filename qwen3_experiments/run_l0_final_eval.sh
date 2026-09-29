#!/usr/bin/env bash
set -euo pipefail
umask 077
EVAL_PYTHON=$1
EVAL_ROOT=$2
cd "${EVAL_ROOT}"

# Honor each existing launcher's lock, without stopping any occupied GPU.
mapfile -t L0_HOLDER_LOCKS < <("${EVAL_PYTHON}" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["holder_locks"]))' "${EVAL_ROOT}/plan.json")
declare -a L0_LOCK_FDS=()
for L0_HOLDER_LOCK in "${L0_HOLDER_LOCKS[@]}"; do
    mkdir -p "$(dirname -- "${L0_HOLDER_LOCK}")"
    exec {L0_LOCK_FD}>"${L0_HOLDER_LOCK}"
    flock -n "${L0_LOCK_FD}" || exit 75
    L0_LOCK_FDS+=("${L0_LOCK_FD}")
done
L0_BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${L0_BUSY} ]] || exit 75

unset PYTHONHOME RAY_ADDRESS ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH
PYTHONPATH=$("${EVAL_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repository"])' "${EVAL_ROOT}/plan.json")
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_V1=0
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR=/tmp/l0-final-eval-${SLURM_JOB_ID:-local}-${SLURM_STEP_ID:-local}
export TRITON_CACHE_DIR=${TMPDIR}/triton
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}"
L0_EXIT=0
"${EVAL_PYTHON}" -u "${EVAL_ROOT}/provenance/eval_l0_final.py" run --output-root "${EVAL_ROOT}" || L0_EXIT=$?
printf '%s\n' "${L0_EXIT}" >"${EVAL_ROOT}/exit_status"
exit "${L0_EXIT}"
