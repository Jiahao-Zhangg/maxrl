#!/usr/bin/env bash
# Login-node entry point: one SSH launch; all persistent processes live on the allocation.
# Set L0_PYTHON_BIN to the maxrl environment's Python when that environment is not active.
# Optional: L0_JOB_ID, L0_RUN_DIR, L0_HF_PREFIX, L0_ROLLOUT_HF_REPO,
# L0_PREDECESSOR_CONFIG, L0_EVAL_TEMPLATE, L0_SSH_KNOWN_HOSTS.
# --prepare-only validates without launching.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
JOB_ID=${L0_JOB_ID:-146103}
PYTHON_BIN=${L0_PYTHON_BIN:-$(command -v python)}
RUN_DIR=${L0_RUN_DIR:-"${REPO_ROOT}/outputs/per_context_rb_l0_0_qwen3_1_7b_compression_bs32_32k_${JOB_ID}"}
HF_PREFIX=${L0_HF_PREFIX:-"hi-todayis-jh/per-context-rb-l0-0-qwen3-1.7b-compression-bs32-32k-${JOB_ID}"}
PREDECESSOR=${L0_PREDECESSOR_CONFIG:-"${REPO_ROOT}/outputs/l0_recovery_pipeline_20260921/config.json"}
EVAL_TEMPLATE=${L0_EVAL_TEMPLATE:-"${REPO_ROOT}/outputs/l0_final_eval_20260921"}
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
# Forward cache/token paths, never credentials in the remote command line.
for CACHE_VARIABLE in HF_HOME HF_TOKEN_PATH HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE; do
    if [[ -n ${!CACHE_VARIABLE:-} ]]; then
        REMOTE_ARGS+=("${CACHE_VARIABLE}=${!CACHE_VARIABLE}")
    fi
done
REMOTE_ARGS+=("${PYTHON_BIN}" -u -m qwen3_experiments.compression_l0_compute_control "${ACTION}"
    --job-id "${JOB_ID}" --output-root "${RUN_DIR}" --hf-prefix "${HF_PREFIX}"
    --predecessor-config "${PREDECESSOR}" --eval-template "${EVAL_TEMPLATE}")
if [[ -n ${L0_ROLLOUT_HF_REPO:-} ]]; then
    REMOTE_ARGS+=(--rollout-hf-repo "${L0_ROLLOUT_HF_REPO}")
fi
printf -v REMOTE_COMMAND '%q ' "${REMOTE_ARGS[@]}"
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=15)
if [[ -n ${L0_SSH_KNOWN_HOSTS:-} ]]; then
    SSH_OPTIONS+=(-o "UserKnownHostsFile=\"${L0_SSH_KNOWN_HOSTS}\"" -o StrictHostKeyChecking=yes)
fi
exec ssh "${SSH_OPTIONS[@]}" "${NODE}" "${REMOTE_COMMAND}"
