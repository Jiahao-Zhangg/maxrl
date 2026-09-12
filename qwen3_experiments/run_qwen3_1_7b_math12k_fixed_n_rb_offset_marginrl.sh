#!/usr/bin/env bash

# Qwen3-1.7B-Base on Math12K with fixed-N RB MarginRL and cost=L+256.
# Reuse the hard-clip launcher's model, data, prompt, grader, and training
# defaults through the same run_qwen3_1_7b_math12k.sh entry point.
# Stop at step 150, save every 50 steps, and upload each checkpoint to HF.
# Delete local checkpoints only after verified upload. Set
# MAXRL_UPLOAD_CHECKPOINTS=0 to keep checkpoints local without uploading.
# Save all training rollouts and upload their dataset at training exit.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Set MAXRL_ENV_ONLY=1 to install and validate dependencies without requiring GPUs.
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

export MAXRL_ADVANTAGE_ESTIMATOR=fixed_n_rb_offset_cost_aware_marginrl
export MAXRL_LOSS_AGG_MODE=${MAXRL_LOSS_AGG_MODE:-token-mean}
LOSS_AGG_TAG=${MAXRL_LOSS_AGG_MODE//-/_}
export MAXRL_COST_OFFSET_TOKENS=${MAXRL_COST_OFFSET_TOKENS:-256}
export MAXRL_EXPERIMENT_NAME=${MAXRL_EXPERIMENT_NAME:-fixed_n_rb_offset_cost_aware_marginrl_Qwen3-1.7B-Base_math12k_offset${MAXRL_COST_OFFSET_TOKENS}_${LOSS_AGG_TAG}}
export MAXRL_TOTAL_TRAINING_STEPS=${MAXRL_TOTAL_TRAINING_STEPS:-150}
export MAXRL_SAVE_FREQ=${MAXRL_SAVE_FREQ:-50}
export MAXRL_TEST_FREQ=${MAXRL_TEST_FREQ:-50}
export MAXRL_UPLOAD_CHECKPOINTS=${MAXRL_UPLOAD_CHECKPOINTS:-1}
HF_RUN_NAME=${MAXRL_EXPERIMENT_NAME//_/-}
export MAXRL_CHECKPOINT_HF_REPO_PREFIX=${MAXRL_CHECKPOINT_HF_REPO_PREFIX:-zjhhhh/${HF_RUN_NAME,,}}
export MAXRL_SAVE_ROLLOUT_DATASET=${MAXRL_SAVE_ROLLOUT_DATASET:-1}
export MAXRL_ROLLOUT_DATASET_HF_REPO=${MAXRL_ROLLOUT_DATASET_HF_REPO:-${MAXRL_CHECKPOINT_HF_REPO_PREFIX}-rollouts}

exec "${SCRIPT_DIR}/run_qwen3_1_7b_math12k.sh" "$@"
