#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Prepare and validate the pinned environment by default. Set
# MAXRL_SKIP_ENV_SETUP=1 only when this snapshot is already installed.
export MAXRL_SKIP_ENV_SETUP=${MAXRL_SKIP_ENV_SETUP:-0}
case "$(uname -m)" in
    aarch64|arm64)
        DEFAULT_REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_qwen3_maxrl_cu126_aarch64.txt"
        DEFAULT_PYTORCH_REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_qwen3_maxrl_cu126_aarch64_pytorch.txt"
        DEFAULT_BOOTSTRAP_REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_qwen3_maxrl_cu126_aarch64_bootstrap.txt"
        DEFAULT_SOURCE_REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_qwen3_maxrl_cu126_aarch64_sources.txt"
        export MAXRL_REQUIREMENTS_FILE="${MAXRL_REQUIREMENTS_FILE:-${DEFAULT_REQUIREMENTS_FILE}}"
        export MAXRL_PYTORCH_REQUIREMENTS_FILE="${MAXRL_PYTORCH_REQUIREMENTS_FILE:-${DEFAULT_PYTORCH_REQUIREMENTS_FILE}}"
        export MAXRL_BOOTSTRAP_REQUIREMENTS_FILE="${MAXRL_BOOTSTRAP_REQUIREMENTS_FILE:-${DEFAULT_BOOTSTRAP_REQUIREMENTS_FILE}}"
        export MAXRL_SOURCE_REQUIREMENTS_FILE="${MAXRL_SOURCE_REQUIREMENTS_FILE:-${DEFAULT_SOURCE_REQUIREMENTS_FILE}}"
        export MAXRL_EXPECTED_TORCH_VERSION=${MAXRL_EXPECTED_TORCH_VERSION:-2.6.0+cu126}
        export MAXRL_EXPECTED_TORCH_CUDA=${MAXRL_EXPECTED_TORCH_CUDA:-12.6}
        export MAXRL_CUDA_ARCH_LIST=${MAXRL_CUDA_ARCH_LIST:-9.0}
        export MAXRL_FLASH_ATTN_CUDA_ARCHS=${MAXRL_FLASH_ATTN_CUDA_ARCHS:-90}
        export MAXRL_CUDA_HOME=${MAXRL_CUDA_HOME:-/sw/user/cudatoolkits/installs/cuda-12.6.1}
        export MAXRL_INSTALL_JOBS=${MAXRL_INSTALL_JOBS:-1}
        export MAXRL_NVCC_THREADS=${MAXRL_NVCC_THREADS:-1}
        ;;
    x86_64)
        export MAXRL_REQUIREMENTS_FILE="${MAXRL_REQUIREMENTS_FILE:-${SCRIPT_DIR}/requirements_qwen3_maxrl_cu124.txt}"
        export MAXRL_EXPECTED_TORCH_VERSION=${MAXRL_EXPECTED_TORCH_VERSION:-2.6.0+cu124}
        export MAXRL_EXPECTED_TORCH_CUDA=${MAXRL_EXPECTED_TORCH_CUDA:-12.4}
        ;;
    *)
        echo "error: unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

export MAXRL_ADVANTAGE_ESTIMATOR=fixed_n_rb_efficient_reasoning_cost_marginrl
export MAXRL_LOSS_AGG_MODE=${MAXRL_LOSS_AGG_MODE:-token-mean}
LOSS_AGG_TAG=${MAXRL_LOSS_AGG_MODE//-/_}
export MAXRL_EFFICIENT_REASONING_EPSILON=${MAXRL_EFFICIENT_REASONING_EPSILON:-1e-7}
export MAXRL_EXPERIMENT_NAME=${MAXRL_EXPERIMENT_NAME:-fixed_n_rb_er_cost_marginrl_Qwen3-1.7B-Base_math12k_${LOSS_AGG_TAG}}

exec "${SCRIPT_DIR}/run_qwen3_1_7b_math12k.sh" "$@"
