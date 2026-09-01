#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

# Prepare and validate the same pinned environment used by the Qwen3 MaxRL
# experiments. Set MAXRL_ENV_ONLY=1 to stop after environment preparation.
export MAXRL_SKIP_ENV_SETUP=${MAXRL_SKIP_ENV_SETUP:-0}
case "$(uname -m)" in
    aarch64|arm64)
        export MAXRL_REQUIREMENTS_FILE="${MAXRL_REQUIREMENTS_FILE:-${SCRIPT_DIR}/requirements_qwen3_maxrl_cu126_aarch64.txt}"
        export MAXRL_PYTORCH_REQUIREMENTS_FILE="${MAXRL_PYTORCH_REQUIREMENTS_FILE:-${SCRIPT_DIR}/requirements_qwen3_maxrl_cu126_aarch64_pytorch.txt}"
        export MAXRL_BOOTSTRAP_REQUIREMENTS_FILE="${MAXRL_BOOTSTRAP_REQUIREMENTS_FILE:-${SCRIPT_DIR}/requirements_qwen3_maxrl_cu126_aarch64_bootstrap.txt}"
        export MAXRL_SOURCE_REQUIREMENTS_FILE="${MAXRL_SOURCE_REQUIREMENTS_FILE:-${SCRIPT_DIR}/requirements_qwen3_maxrl_cu126_aarch64_sources.txt}"
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

DATA_ROOT=${MAXRL_DATA_DIR:-${REPO_ROOT}/data}
export MAXRL_TRAIN_DATASET_NAME="edbeeching/Polaris-Dataset-53K-4-8@ad1509a81e835614274acdb444c223164da0212c"
export MAXRL_TRAIN_DATASET_DIR=${MAXRL_TRAIN_DATASET_DIR:-${DATA_ROOT}/polaris_4_8}
export MAXRL_TRAIN_DATASET_CONVERTER="${REPO_ROOT}/examples/maxrl_data_preprocess/polaris_4_8.py"

export MAXRL_MODEL_PATH=${MAXRL_MODEL_PATH:-Qwen/Qwen3-0.6B}
export MAXRL_MODEL_NAME=${MAXRL_MODEL_NAME:-Qwen3-0.6B}
export MAXRL_ADVANTAGE_ESTIMATOR=maxrl
export MAXRL_LOSS_AGG_MODE=${MAXRL_LOSS_AGG_MODE:-token-mean}
export MAXRL_TOTAL_EPOCHS=${MAXRL_TOTAL_EPOCHS:-2}
export MAXRL_PROJECT_NAME=${MAXRL_PROJECT_NAME:-Qwen3_MaxRL_Experiments}
export MAXRL_EXPERIMENT_NAME=${MAXRL_EXPERIMENT_NAME:-maxrl_Qwen3-0.6B_polaris_4_8_2epochs}

# The shared launcher keeps the established MaxRL settings: four GPUs, N=16,
# a 4,096-token response cap, MathVerify rewards, and AIME25/MATH-500 validation.
exec "${SCRIPT_DIR}/run_qwen3_1_7b_math12k.sh" "$@"
