#!/usr/bin/env bash
# Queue f_cov after this allocation's compression MaxRL training and uploads.
# Login only dispatches; persistent supervision and all work run on compute.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
JOB_ID=${FCOV_JOB_ID:-146103}
PYTHON_BIN=${FCOV_PYTHON_BIN:-$(command -v python)}
RUN_DIR=${FCOV_RUN_DIR:-"${REPO_ROOT}/outputs/f_cov_l0_0_no_eos_qwen3_1_7b_compression_bs32_n16_32k_${JOB_ID}"}
HF_PREFIX=${FCOV_HF_PREFIX:-"hi-todayis-jh/f-cov-l0-0-no-eos-qwen3-1.7b-compression-bs32-n16-32k-${JOB_ID}"}
PREDECESSOR=${FCOV_PREDECESSOR_TRAINING:-"${REPO_ROOT}/outputs/maxrl_no_eos_qwen3_1_7b_compression_bs32_n16_32k_${JOB_ID}"}
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
    --predecessor-training "${PREDECESSOR}" --adv-estimator f_cov
    --cost-offset-tokens 0 --no-check-eos --skip-final-evaluation)
if [[ -n ${FCOV_ROLLOUT_HF_REPO:-} ]]; then
    REMOTE_ARGS+=(--rollout-hf-repo "${FCOV_ROLLOUT_HF_REPO}")
fi
printf -v REMOTE_COMMAND '%q ' "${REMOTE_ARGS[@]}"
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=15)
if [[ -n ${FCOV_SSH_KNOWN_HOSTS:-} ]]; then
    SSH_OPTIONS+=(-o "UserKnownHostsFile=\"${FCOV_SSH_KNOWN_HOSTS}\"" -o StrictHostKeyChecking=yes)
fi
exec ssh "${SSH_OPTIONS[@]}" "${NODE}" "${REMOTE_COMMAND}"
