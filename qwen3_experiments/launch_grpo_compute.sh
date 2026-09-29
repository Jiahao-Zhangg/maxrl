#!/usr/bin/env bash
# The login node only submits this short SSH launch. All persistent work is remote.
# Prepare/check configuration: bash launch_grpo_compute.sh --prepare-only
# Start training on allocation 146102: bash launch_grpo_compute.sh
# Optional environment: GRPO_JOB_ID, GRPO_RUN_DIR, GRPO_PYTHON_BIN, GRPO_HF_PREFIX.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
JOB_ID=${GRPO_JOB_ID:-146102}
PYTHON_BIN=${GRPO_PYTHON_BIN:-/project/flame/jiahaoz4/miniconda3/envs/maxrl/bin/python}
RUN_DIR=${GRPO_RUN_DIR:-"${REPO_ROOT}/outputs/grpo_qwen3_1_7b_polaris_1_8_3200_bs32_32k_${JOB_ID}"}
HF_PREFIX=${GRPO_HF_PREFIX:-"hi-todayis-jh/grpo-qwen3-1.7b-polaris-1-8-3200-bs32-32k-${JOB_ID}"}
[[ ${JOB_ID} =~ ^[0-9]+$ ]] || { echo "Invalid allocation ID" >&2; exit 2; }
ACTION=launch
if [[ ${1:-} == --prepare-only ]]; then
    ACTION=prepare
    shift
fi
[[ $# -eq 0 ]] || { echo "Usage: $0 [--prepare-only]" >&2; exit 2; }
NODE=$(squeue --noheader --jobs="${JOB_ID}" --format='%N')
[[ ${NODE} =~ ^[a-zA-Z0-9-]+$ ]] || { echo "Expected one allocated compute node" >&2; exit 1; }
REMOTE_ARGS=(env)
# SSH does not inherit the project cache location. Forward paths, never token values.
for CACHE_VARIABLE in HF_HOME HF_TOKEN_PATH HF_HUB_CACHE HUGGINGFACE_HUB_CACHE; do
    if [[ -n ${!CACHE_VARIABLE:-} ]]; then
        REMOTE_ARGS+=("${CACHE_VARIABLE}=${!CACHE_VARIABLE}")
    fi
done
REMOTE_ARGS+=("${PYTHON_BIN}" -u "${SCRIPT_DIR}/grpo_compute_control.py" "${ACTION}"
    --job-id "${JOB_ID}" --output-root "${RUN_DIR}" --hf-prefix "${HF_PREFIX}")
printf -v REMOTE_COMMAND '%q ' "${REMOTE_ARGS[@]}"
exec ssh -o BatchMode=yes -o ConnectTimeout=15 "${NODE}" "${REMOTE_COMMAND}"
