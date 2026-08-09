#!/usr/bin/env bash

set -euo pipefail

# Keep the Conda environment isolated from ~/.local Python packages.
export PYTHONNOUSERSITE=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

MAXRL_ENV_NAME=${MAXRL_ENV_NAME:-maxrl}
MAXRL_DATA_DIR=${MAXRL_DATA_DIR:-${REPO_ROOT}/data}
MAXRL_OUTPUT_DIR=${MAXRL_OUTPUT_DIR:-${REPO_ROOT}/outputs}
MAXRL_RAY_DIR=${MAXRL_RAY_DIR:-${MAXRL_OUTPUT_DIR}/ray}
MAXRL_SKIP_ENV_SETUP=${MAXRL_SKIP_ENV_SETUP:-0}
MAXRL_REFRESH_DATA=${MAXRL_REFRESH_DATA:-0}
MAXRL_INSTALL_JOBS=${MAXRL_INSTALL_JOBS:-4}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

die() {
    echo "error: $*" >&2
    exit 1
}

activate_environment() {
    command -v conda >/dev/null 2>&1 || die "conda is required; load Conda and rerun this script"
    eval "$(conda shell.bash hook)"

    if ! conda run -n "${MAXRL_ENV_NAME}" python --version >/dev/null 2>&1; then
        conda create -y -n "${MAXRL_ENV_NAME}" python=3.10
    fi
    conda activate "${MAXRL_ENV_NAME}"
    python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))' || \
        die "${MAXRL_ENV_NAME} exists but does not use Python 3.10; choose another MAXRL_ENV_NAME"
}

install_environment() {
    python -m pip install \
        pip==25.0.1 setuptools==75.8.0 wheel==0.45.1
    python -m pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        numpy==1.26.4 fsspec==2024.12.0 \
        --index-url https://download.pytorch.org/whl/cu124
    python -m pip install --no-cache-dir \
        vllm==0.8.4 tensordict==0.6.2 torchdata==0.11.0 \
        transformers==4.51.3 huggingface-hub==0.30.2 \
        "ray[cgraph]==2.43.0" cupy-cuda12x==13.3.0 \
        opencv-python-headless==4.11.0.86 "fastapi[standard]==0.115.12" \
        pydantic==2.11.3 protobuf==4.25.6 \
        opentelemetry-api==1.26.0 opentelemetry-sdk==1.26.0 \
        opentelemetry-proto==1.26.0 \
        opentelemetry-semantic-conventions==0.47b0
    python -m pip install --no-cache-dir \
        "transformers[hf_xet]==4.51.3" accelerate==1.6.0 datasets==3.5.0 \
        peft==0.17.0 hf-transfer==0.1.9 pyarrow==19.0.1 pandas==2.2.3 \
        codetiming==1.4.0 hydra-core==1.3.2 pylatexenc==2.10 \
        wandb==0.19.10 dill==0.3.8 pybind11==2.13.6 math-verify==0.9.0 \
        ninja==1.11.1.3 packaging==24.2 psutil==6.1.1 \
        nvidia-ml-py==12.575.51 optree==0.15.0

    if [[ "$(uname -m)" == "x86_64" ]]; then
        python -m pip install --no-cache-dir \
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
        python -m pip install --no-cache-dir \
            "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
    else
        MAX_JOBS="${MAXRL_INSTALL_JOBS}" python -m pip install --no-cache-dir \
            flash-attn==2.7.4.post1 --no-build-isolation
    fi
    python -m pip install --no-deps -e "${REPO_ROOT}"
}

validate_environment() {
    python -m pip check
    python - <<'PY'
from importlib.metadata import version

import flash_attn
import flashinfer
import pkg_resources
import ray
import torch
import transformers
import vllm

expected = {
    "ray": "2.43.0",
    "torch": "2.6.0+cu124",
    "transformers": "4.51.3",
    "vllm": "0.8.4",
}
actual = {package: version(package) for package in expected}
if actual != expected:
    raise SystemExit(f"Unexpected core package versions: {actual}")
print("Validated core packages:", actual)
PY
}

prepare_dataset() {
    local converter=$1
    local destination=$2

    mkdir -p "${destination}"
    if [[ "${MAXRL_REFRESH_DATA}" == "1" || ! -s "${destination}/train.parquet" || ! -s "${destination}/test.parquet" ]]; then
        python "${converter}" --local_dir "${destination}"
    else
        echo "Reusing prepared dataset in ${destination}"
    fi
}

if [[ "${MAXRL_SKIP_ENV_SETUP}" == "1" ]]; then
    echo "Skipping environment creation; using $(command -v python)"
else
    activate_environment
    install_environment
    validate_environment
fi

cd "${REPO_ROOT}"

MATH12K_DIR=${MAXRL_DATA_DIR}/math12k
AIME25_DIR=${MAXRL_DATA_DIR}/aime25
MATH500_DIR=${MAXRL_DATA_DIR}/math500

prepare_dataset \
    "${REPO_ROOT}/examples/maxrl_data_preprocess/math12k.py" \
    "${MATH12K_DIR}"
prepare_dataset \
    "${REPO_ROOT}/examples/maxrl_data_preprocess/aime25.py" \
    "${AIME25_DIR}"
prepare_dataset \
    "${REPO_ROOT}/examples/maxrl_data_preprocess/math_500.py" \
    "${MATH500_DIR}"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the active Python environment")
if torch.cuda.device_count() != 4:
    raise SystemExit(
        f"Expected four visible GPUs from CUDA_VISIBLE_DEVICES, found {torch.cuda.device_count()}"
    )
print("Visible GPUs:", [torch.cuda.get_device_name(i) for i in range(4)])
PY

mkdir -p "${MAXRL_OUTPUT_DIR}" "${MAXRL_RAY_DIR}"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export SEED=79
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export NCCL_ALGO=RING
export NCCL_IGNORE_CPU_AFFINITY=1

FULL_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=256
NUM_PER_PROMPT_ROLLOUTS=16
MAX_RESPONSE_LENGTH=4096
MAX_PROMPT_LENGTH=1024
LEARNING_RATE=1e-6
REWARD_MANAGER=multi_thread
PER_GPU_MINI_BATCH_SIZE=4
NUM_PER_PROMPT_ROLLOUTS_VALIDATION=32
MAX_MODEL_LEN=32000
MAX_NUM_BATCHED_TOKENS=32000
TENSOR_MODEL_PARALLEL_SIZE=1
PPO_EPOCHS=1
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2
GRAD_CLIP=0.3
KL_COEFF=0.0
TOTAL_EPOCHS=5

MODEL_PATH=Qwen/Qwen3-1.7B-Base
MODEL_NAME=Qwen3-1.7B-Base
ADVANTAGE_ESTIMATOR=maxrl
PROJECT_NAME=Qwen3_MaxRL_Experiments
EXPERIMENT_NAME=${ADVANTAGE_ESTIMATOR}_${MODEL_NAME}_math12k
CHECKPOINT_SAVE_PATH=${MAXRL_OUTPUT_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
TEST_DATASET_PATH="['${AIME25_DIR}/test.parquet','${MATH500_DIR}/test.parquet']"

echo "Training ${MODEL_NAME} with hiyouga/math12k for ${TOTAL_EPOCHS} epochs"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Checkpoints: ${CHECKPOINT_SAVE_PATH}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "Note: the original logger configuration uses Weights & Biases; ensure 'wandb login' has been run."
fi

python -W ignore -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${ADVANTAGE_ESTIMATOR}" \
    data.train_files="${MATH12K_DIR}/train.parquet" \
    data.val_files="${TEST_DATASET_PATH}" \
    data.train_batch_size="${FULL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PER_GPU_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEFF}" \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    actor_rollout_ref.actor.grad_clip="${GRAD_CLIP}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PER_GPU_MINI_BATCH_SIZE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n="${NUM_PER_PROMPT_ROLLOUTS}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PER_GPU_MINI_BATCH_SIZE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n="${NUM_PER_PROMPT_ROLLOUTS_VALIDATION}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef="${KL_COEFF}" \
    reward_model.reward_manager="${REWARD_MANAGER}" \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.val_on_last_step=True \
    "trainer.logger=['console','wandb']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_SAVE_PATH}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=400 \
    trainer.max_critic_ckpt_to_keep=400 \
    trainer.test_freq=50 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    ray_init.ray_dir="${MAXRL_RAY_DIR}" \
    "$@"
