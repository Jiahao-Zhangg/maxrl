#!/usr/bin/env bash
# Start one detached compute-node supervisor after the GRPO final evaluation.
# Optional: L0_JOB_ID, L0_PYTHON_BIN, L0_RUN_DIR, L0_HF_PREFIX,
# L0_ROLLOUT_HF_REPO, L0_PREDECESSOR_EVALUATION. --prepare-only does not launch.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
JOB_ID=${L0_JOB_ID:-146102}
PYTHON_BIN=${L0_PYTHON_BIN:-$(command -v python)}
RUN_DIR=${L0_RUN_DIR:-"${REPO_ROOT}/outputs/per_context_rb_l0_4096_no_eos_qwen3_1_7b_compression_bs32_32k_${JOB_ID}"}
HF_PREFIX=${L0_HF_PREFIX:-"hi-todayis-jh/per-context-rb-l0-4096-no-eos-qwen3-1.7b-compression-bs32-32k-${JOB_ID}"}
PREDECESSOR=${L0_PREDECESSOR_EVALUATION:-"${REPO_ROOT}/outputs/grpo_final_eval_${JOB_ID}"}
[[ ${JOB_ID} =~ ^[0-9]+$ ]] || { echo "Invalid allocation ID" >&2; exit 2; }
ACTION=launch
if [[ ${1:-} == --prepare-only ]]; then
    ACTION=prepare
    shift
fi
[[ $# -eq 0 ]] || { echo "Usage: $0 [--prepare-only]" >&2; exit 2; }
NODE=$(squeue --noheader --jobs="${JOB_ID}" --format='%N')
[[ ${NODE} =~ ^[a-zA-Z0-9-]+$ ]] || { echo "Expected one allocated compute node" >&2; exit 1; }
REMOTE_ARGS=(env "PYTHONPATH=${REPO_ROOT}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1)
for CACHE_VARIABLE in HF_HOME HF_TOKEN_PATH HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE; do
    if [[ -n ${!CACHE_VARIABLE:-} ]]; then
        REMOTE_ARGS+=("${CACHE_VARIABLE}=${!CACHE_VARIABLE}")
    fi
done
REMOTE_ARGS+=("${PYTHON_BIN}" -u -m qwen3_experiments.compression_l0_compute_control "${ACTION}"
    --job-id "${JOB_ID}" --output-root "${RUN_DIR}" --hf-prefix "${HF_PREFIX}"
    --predecessor-evaluation "${PREDECESSOR}" --cost-offset-tokens 4096 --no-check-eos --skip-final-evaluation)
if [[ -n ${L0_ROLLOUT_HF_REPO:-} ]]; then
    REMOTE_ARGS+=(--rollout-hf-repo "${L0_ROLLOUT_HF_REPO}")
fi
printf -v REMOTE_COMMAND '%q ' "${REMOTE_ARGS[@]}"
exec ssh -o BatchMode=yes -o ConnectTimeout=15 "${NODE}" "${REMOTE_COMMAND}"
