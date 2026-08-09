#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export MAXRL_ADVANTAGE_ESTIMATOR=cost_aware_maxrl
export MAXRL_LOSS_AGG_MODE=token-mean
export MAXRL_COST_REFERENCE_TOKENS=${MAXRL_COST_REFERENCE_TOKENS:-2048}
export MAXRL_MAX_INVERSE_COST=${MAXRL_MAX_INVERSE_COST:-4.0}
export MAXRL_EXPERIMENT_NAME=${MAXRL_EXPERIMENT_NAME:-cost_aware_maxrl_Qwen3-1.7B-Base_math12k_cap4}

exec "${SCRIPT_DIR}/run_qwen3_1_7b_math12k.sh" "$@"
