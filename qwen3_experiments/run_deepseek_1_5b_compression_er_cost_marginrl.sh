#!/usr/bin/env bash

# DeepSeek-R1-Distill-Qwen-1.5B on the ER compression training subset, using
# our fixed-N Rao--Blackwellized ER-cost MarginRL (token-mean loss).
# For each prompt, c_i = sigmoid((tokens_i - mean(tokens)) / (std(tokens) + 1e-7)),
# using all 16 responses and population std; q_hat = successes / sum(c_i).
# This is the original ER-cost variant, including the failure-cost update.
# Match run_rloo_deepseek_1.5B_compression.sh's model, prompt, batch sizes,
# output budget, optimizer settings, and checkpoint frequency.
#
# Preview without Conda, downloads, services, or GPUs:
#   DRY_RUN=1 bash qwen3_experiments/run_deepseek_1_5b_compression_er_cost_marginrl.sh
# Prepare the 3,200 training rows without launching training:
#   PREPARE_ONLY=1 bash qwen3_experiments/run_deepseek_1_5b_compression_er_cost_marginrl.sh
# Launch in the existing maxrl environment on four GPUs:
#   GPU_IDS=0,1,2,3 bash qwen3_experiments/run_deepseek_1_5b_compression_er_cost_marginrl.sh
# Use PYTHON_BIN to select an existing interpreter instead of activating Conda.
# Checkpoints stay under OUTPUT_ROOT/RUN_NAME/checkpoints; RESUME=1 resumes them.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# Training settings are explicit so exports from earlier experiments cannot
# silently change the model, data, or the ER cost estimator.
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
DATASET_REPO=zjhhhh/compression_dataset
DATASET_REVISION=bfdd7af1633ecc6db191a9f28f76449165a4ee06
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=16
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=32000
MAX_MODEL_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
EFFICIENT_REASONING_EPSILON=1e-7

CONDA_ENV=${CONDA_ENV:-maxrl}
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-0,1,2,3}
RUN_NAME=${RUN_NAME:-er_cost_marginrl_r1_distill_1.5b_compression_n16_b512_32k_lr1e-6_kl0_seed42}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs}
RUN_DIR=${OUTPUT_ROOT}/${RUN_NAME}
CKPT_PATH=${RUN_DIR}/checkpoints
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data/compression_dataset}
TRAIN_DATA=${DATA_DIR}/train.parquet
DRY_RUN=${DRY_RUN:-0}
PREPARE_ONLY=${PREPARE_ONLY:-0}
RESUME=${RESUME:-0}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-maxrl_compression}
VERIFIER_WORKERS=${VERIFIER_WORKERS:-16}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/maxrl_er_cost_${UID}}

for option in DRY_RUN PREPARE_ONLY RESUME USE_WANDB; do
    [[ "${!option}" == "0" || "${!option}" == "1" ]] || {
        echo "${option} must be 0 or 1." >&2
        exit 2
    }
done
[[ "${GPU_IDS}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || {
    echo "GPU_IDS must list four distinct GPU IDs, for example 0,1,2,3." >&2
    exit 2
}
[[ "${VERIFIER_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "VERIFIER_WORKERS must be a positive integer." >&2
    exit 2
}
if (( $# > 0 )); then
    echo "This launcher fixes the experiment settings; edit the script for a different experiment." >&2
    exit 2
fi

PREPARE_CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/examples/maxrl_data_preprocess/compression.py"
    --local_dir "${DATA_DIR}"
    --dataset_repo "${DATASET_REPO}"
    --revision "${DATASET_REVISION}"
)
TRAIN_CMD=(
    "${PYTHON_BIN}" -u -m verl.trainer.main_ppo
    algorithm.adv_estimator=fixed_n_rb_efficient_reasoning_cost_marginrl
    "algorithm.efficient_reasoning_epsilon=${EFFICIENT_REASONING_EPSILON}"
    algorithm.efficient_reasoning_fixed_q_hat=null
    algorithm.use_kl_in_reward=false
    algorithm.kl_ctrl.kl_coef=0
    "data.train_files='${TRAIN_DATA}'"
    # The trainer requires a nonempty validation loader even when evaluation
    # is disabled. This train-only dataset is never used to report validation.
    "data.val_files='${TRAIN_DATA}'"
    data.val_batch_size=32
    "data.train_batch_size=${ROLLOUT_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    data.apply_chat_template=false
    data.filter_overlong_prompts=false
    data.truncation=right
    data.shuffle=true
    +data.seed=42
    "actor_rollout_ref.model.path=${MODEL}"
    actor_rollout_ref.model.use_remove_padding=true
    actor_rollout_ref.model.enable_gradient_checkpointing=true
    actor_rollout_ref.actor.dtype=bfloat16
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03
    actor_rollout_ref.actor.optim.warmup_style=constant
    actor_rollout_ref.actor.optim.weight_decay=0
    +actor_rollout_ref.actor.optim.betas=[0.9,0.95]
    # verl counts prompts here, then multiplies by rollout.n: 32 * 16 = 512
    # responses per optimizer update, exactly one update per outer step.
    "actor_rollout_ref.actor.ppo_mini_batch_size=${ROLLOUT_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_MODEL_LENGTH}"
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.2
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_kl_loss=false
    actor_rollout_ref.actor.kl_loss_coef=0
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LENGTH}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LENGTH}"
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    "actor_rollout_ref.rollout.n=${N_SAMPLES_PER_PROMPT}"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    +actor_rollout_ref.rollout.min_p=0.0
    +actor_rollout_ref.rollout.seed=42
    actor_rollout_ref.rollout.ignore_eos=false
    actor_rollout_ref.rollout.multi_turn.enable=false
    reward_model.reward_manager=multi_thread
    "+reward_model.reward_kwargs.num_reward_actors=${VERIFIER_WORKERS}"
    # Require EOS in the generated response, matching the ER reward server.
    +reward_model.reward_kwargs.check_eos=true
    +reward_model.reward_kwargs.zero_reward_on_max_response_length=false
    trainer.balance_batch=true
    trainer.critic_warmup=0
    trainer.val_before_train=false
    trainer.val_on_last_step=false
    trainer.eval_on_last_step=false
    trainer.test_freq=-1
    trainer.total_epochs=1
    trainer.total_training_steps=100
    trainer.save_freq=20
    trainer.max_actor_ckpt_to_keep=10
    trainer.n_gpus_per_node=4
    trainer.nnodes=1
    "trainer.project_name='${WANDB_PROJECT}'"
    "trainer.experiment_name='${RUN_NAME}'"
    "trainer.default_local_dir='${CKPT_PATH}'"
    "ray_init.ray_dir='${RAY_TMPDIR}'"
)
if [[ "${USE_WANDB}" == "1" ]]; then
    TRAIN_CMD+=("trainer.logger=['console','wandb']")
else
    TRAIN_CMD+=("trainer.logger=['console']")
fi
if [[ "${RESUME}" == "1" ]]; then
    TRAIN_CMD+=(trainer.resume_mode=auto)
else
    TRAIN_CMD+=(trainer.resume_mode=disable)
fi

echo "Model: ${MODEL}; dataset: ${DATASET_REPO}@${DATASET_REVISION} (3200 rows)."
echo "ER cost: sigmoid((tokens-group_mean)/(group_std+${EFFICIENT_REASONING_EPSILON})); all 16 responses per prompt."
echo "Fixed-N RB MarginRL: q_hat=M/sum(cost); failure advantage=-q_hat*cost/(M+1); token-mean loss."
echo "32 prompts x 16 responses = 512 responses/update; 1 update/step; 100 steps in 1 epoch."
echo "Prompt cap: ${MAX_PROMPT_LENGTH}; output cap: ${MAX_RESPONSE_LENGTH}; context: ${MAX_MODEL_LENGTH}."
echo "LR: 1e-6; warmup: 3 steps; KL: 0; checkpoints at steps 20, 40, 60, 80, 100."
echo "Reward completion check: responses must contain EOS; missing EOS receives zero reward."
echo "Checkpoints: ${CKPT_PATH}"
if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'Prepare command:\n'
    printf '  %q' "${PREPARE_CMD[@]}"
    printf '\nTraining command:\n'
    printf '  %q' "${TRAIN_CMD[@]}"
    printf '\n'
    exit 0
fi

if [[ "${PREPARE_ONLY}" != "1" ]]; then
    if [[ "${RESUME}" == "0" && -e "${RUN_DIR}" ]]; then
        echo "Run directory exists: ${RUN_DIR}. Choose another RUN_NAME or set RESUME=1." >&2
        exit 1
    fi
    if [[ "${RESUME}" == "1" && ! -s "${CKPT_PATH}/latest_checkpointed_iteration.txt" ]]; then
        echo "No local checkpoint to resume under ${CKPT_PATH}." >&2
        exit 1
    fi
fi

if [[ "${PYTHON_BIN}" == "python" ]]; then
    CONDA_BASE=$(conda info --base)
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_ADDRESS=local
export RAY_TMPDIR
export SEED=42
unset TRANSFORMERS_CACHE

"${PREPARE_CMD[@]}"
if [[ "${PREPARE_ONLY}" == "1" ]]; then
    echo "Dataset ready: ${TRAIN_DATA}"
    exit 0
fi

"${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version

import torch

if version("math-verify") != "0.9.0":
    raise SystemExit("This experiment requires math-verify==0.9.0")
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit("Expected four distinct, visible CUDA GPUs")
print("Visible GPUs:", [torch.cuda.get_device_name(i) for i in range(4)])
PY
IFS=',' read -r -a selected_gpus <<<"${GPU_IDS}"
for gpu_id in "${selected_gpus[@]}"; do
    busy=$(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader)
    if [[ -n "${busy}" ]]; then
        echo "GPU ${gpu_id} is busy (PIDs: ${busy}); select four idle GPUs with GPU_IDS." >&2
        exit 1
    fi
done

mkdir -p "${RUN_DIR}/logs" "${CKPT_PATH}" "${RAY_TMPDIR}"
"${TRAIN_CMD[@]}" 2>&1 | tee -a "${RUN_DIR}/logs/training.log"
