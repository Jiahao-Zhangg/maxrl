#!/usr/bin/env bash
# Login only starts this compute-node supervisor; it waits for all Minerva points.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
AFTER_EVAL=${ER_AFTER_EVAL:-"${REPO_ROOT}/outputs/per_context_rb_l0_0_qwen3_1_7b_compression_bs32_32k_146103/minerva_individual_budget_seed0_five_models"}
JOB_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "${AFTER_EVAL}/plan.json")
REFERENCE=${ER_REFERENCE_ROOT:-"${REPO_ROOT}/../efficient-reasoning-official-hybrid"}
CONDA_BASE=${CONDA_BASE:-/project/flame/jiahaoz4/miniconda3}
PYTHON_BIN=${ER_PYTHON_BIN:-"${CONDA_BASE}/envs/efficient_reasoning_official_hybrid/bin/python"}
RUN_DIR=${ER_RUN_DIR:-"${REPO_ROOT}/outputs/er_qwen3_1_7b_compression_hybrid8_bs32_n8_b128_32k_${JOB_ID}"}
HF_PREFIX=${ER_HF_PREFIX:-"hi-todayis-jh/rloo-qwen3-1.7b-compression-official-hybrid8-bs32-n8-b128-32k-${JOB_ID}"}
[[ ${JOB_ID} =~ ^[0-9]+$ ]] || { echo "Invalid allocation" >&2; exit 2; }
ACTION=launch
if [[ ${1:-} == --prepare-only ]]; then ACTION=prepare; shift; fi
[[ $# -eq 0 ]] || { echo "Usage: $0 [--prepare-only]" >&2; exit 2; }
NODE=$(squeue --noheader --jobs="${JOB_ID}" --format='%N')
[[ ${NODE} =~ ^[a-zA-Z0-9-]+$ ]] || { echo "Expected one allocated compute node" >&2; exit 1; }
REMOTE_ARGS=(env "PYTHONPATH=${REPO_ROOT}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false)
for VARIABLE in HF_HOME HF_TOKEN_PATH HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE; do
    if [[ -n ${!VARIABLE:-} ]]; then REMOTE_ARGS+=("${VARIABLE}=${!VARIABLE}"); fi
done
REMOTE_ARGS+=("${PYTHON_BIN}" -u -m qwen3_experiments.er_compression_compute "${ACTION}"
    --after-eval "${AFTER_EVAL}" --output-root "${RUN_DIR}" --reference-root "${REFERENCE}" --hf-prefix "${HF_PREFIX}")
printf -v REMOTE_COMMAND '%q ' "${REMOTE_ARGS[@]}"
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=15)
if [[ -n ${ER_SSH_KNOWN_HOSTS:-} ]]; then
    SSH_OPTIONS+=(-o "UserKnownHostsFile=\"${ER_SSH_KNOWN_HOSTS}\"" -o StrictHostKeyChecking=yes)
fi
exec ssh "${SSH_OPTIONS[@]}" "${NODE}" "${REMOTE_COMMAND}"
