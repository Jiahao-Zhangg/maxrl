#!/usr/bin/env bash

# Qwen3-4B-Base on the full Polaris53K training split, using original MaxRL.
# Only model, training dataset, epoch count, save frequency, and archival
# defaults differ from run_qwen3_1_7b_math12k.sh. In particular, reuse its
# grader unchanged: multi_thread MathVerify, 1s item / 10s batch outer
# deadlines, no EOS requirement, and no reward gate for length-capped outputs.
# Validation remains AIME25/MATH-500, initially and every 50 steps.
# Save every 60 steps and at the end of the epoch; upload, verify, then delete
# each local checkpoint. Upload failures retain local files for retry.
# MAXRL_UPLOAD_CHECKPOINTS=0 disables checkpoint archival.
# MAXRL_ENV_ONLY=1 only prepares the shared environment (no GPU training).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_ROOT=${MAXRL_DATA_DIR:-${REPO_ROOT}/data}

# This is the full original dataset, not the previously used 4/8--7/8 subset.
export MAXRL_TRAIN_DATASET_NAME=POLARIS-Project/Polaris-Dataset-53K
export MAXRL_TRAIN_DATASET_DIR=${MAXRL_TRAIN_DATASET_DIR:-${DATA_ROOT}/polaris53k}
export MAXRL_TRAIN_DATASET_CONVERTER="${REPO_ROOT}/examples/maxrl_data_preprocess/polaris.py"
export MAXRL_MODEL_PATH=${MAXRL_MODEL_PATH:-Qwen/Qwen3-4B-Base}
export MAXRL_MODEL_NAME=${MAXRL_MODEL_NAME:-Qwen3-4B-Base}
export MAXRL_ADVANTAGE_ESTIMATOR=maxrl
export MAXRL_TOTAL_EPOCHS=${MAXRL_TOTAL_EPOCHS:-1}
export MAXRL_SAVE_FREQ=${MAXRL_SAVE_FREQ:-60}
export MAXRL_EXPERIMENT_NAME=${MAXRL_EXPERIMENT_NAME:-maxrl_Qwen3-4B-Base_polaris53k_1epoch}
export MAXRL_UPLOAD_CHECKPOINTS=${MAXRL_UPLOAD_CHECKPOINTS:-1}
HF_RUN_NAME=${MAXRL_EXPERIMENT_NAME//_/-}
export MAXRL_CHECKPOINT_HF_REPO_PREFIX=${MAXRL_CHECKPOINT_HF_REPO_PREFIX:-zjhhhh/${HF_RUN_NAME,,}}

# Do not impose a guessed step cap: the trainer determines epoch length after
# its normal prompt-length filtering and drop_last handling. The upload
# supervisor reads that exact count from the trainer's startup log.
# All other settings, including rollout saving (off by default), are inherited.
exec "${SCRIPT_DIR}/run_qwen3_1_7b_math12k.sh" "$@"
