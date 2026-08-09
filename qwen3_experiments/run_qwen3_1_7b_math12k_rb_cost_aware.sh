#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export MAXRL_SKIP_ENV_SETUP=0
export MAXRL_REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_qwen3_maxrl_cu124.txt"
export MAXRL_ADVANTAGE_ESTIMATOR=rb_cost_aware_maxrl
export MAXRL_LOSS_AGG_MODE=seq-mean-token-sum
export MAXRL_RB_COST_MAX_TOKENS=${MAXRL_RB_COST_MAX_TOKENS:-4096}
export MAXRL_EXPERIMENT_NAME=${MAXRL_EXPERIMENT_NAME:-rb_cost_aware_maxrl_Qwen3-1.7B-Base_math12k_success_gated}

exec "${SCRIPT_DIR}/run_qwen3_1_7b_math12k.sh" "$@"
