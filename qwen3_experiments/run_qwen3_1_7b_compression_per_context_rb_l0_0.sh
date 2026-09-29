#!/usr/bin/env bash
# Qwen3-1.7B on the exact 3,200-row DeepSeek ER compression subset, with L+0.
# Inherit the current Polaris GRPO hyperparameters and EOS/after-thinking grading.
# Raise the prompt cap from 1280 to 1536: the longest compression prompt is 1459
# Qwen3 tokens. Keep all 3200 rows and the original 32768-token response budget.
# Use this primary checkout and an activated maxrl environment on eight GPUs.
#
# Preview commands without downloads, services, or GPUs:
#   DRY_RUN=1 bash "$0"
# Resolve the trainer configuration without preparing inputs or training:
#   bash "$0" --cfg job --resolve
# Prepare and audit all data/tokenizer inputs without downloading model weights:
#   PREPARE_ONLY=1 bash "$0"
# Train on the eight GPUs assigned to this process:
#   bash "$0"
#
# Optional environment: PYTHON_BIN, L0_RUN_DIR, L0_DATA_DIR, L0_CHECKPOINT_DIR,
# L0_RAY_DIR, L0_TRAIN_LOGGER, L0_MAX_PROMPT_LENGTH, L0_ROLLOUT_DIR,
# L0_ROLLOUT_HF_REPO, L0_ROLLOUT_PRIVATE, L0_COST_OFFSET_TOKENS,
# L0_CHECK_EOS, L0_EXPERIMENT_NAME, L0_ADV_ESTIMATOR. Defaults remain L+0 with EOS required.
# Extra arguments are Hydra
# overrides. Model weights are resolved to the pinned initial Hub revision.
# Checkpoints stay in L0_CHECKPOINT_DIR (default: L0_RUN_DIR/checkpoints).
# All 512 rollouts per step are saved atomically as compressed JSONL shards;
# the trainer uploads and verifies the complete dataset before successful exit.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

MODEL=Qwen/Qwen3-1.7B
MODEL_REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
DATASET_REPO=zjhhhh/compression_dataset
DATASET_REVISION=bfdd7af1633ecc6db191a9f28f76449165a4ee06
PYTHON_BIN=${PYTHON_BIN:-python}
DRY_RUN=${DRY_RUN:-0}
PREPARE_ONLY=${PREPARE_ONLY:-0}
MAX_PROMPT_LENGTH=${L0_MAX_PROMPT_LENGTH:-1536}
COST_OFFSET_TOKENS=${L0_COST_OFFSET_TOKENS:-0}
CHECK_EOS=${L0_CHECK_EOS:-true}
ADV_ESTIMATOR=${L0_ADV_ESTIMATOR:-fixed_n_rb_offset_cost_aware_marginrl}
RUN_DIR=${L0_RUN_DIR:-"${REPO_ROOT}/outputs/per_context_rb_l0_${COST_OFFSET_TOKENS}_qwen3_1_7b_compression_bs32_32k_${SLURM_JOB_ID:-local}"}
DATA_DIR=${L0_DATA_DIR:-"${REPO_ROOT}/data/compression_dataset_qwen3"}
CHECKPOINT_DIR=${L0_CHECKPOINT_DIR:-"${RUN_DIR}/checkpoints"}
RAY_DIR=${L0_RAY_DIR:-"/tmp/qwen3compressionl0${SLURM_JOB_ID:-$$}/ray"}
ROLLOUT_DIR=${L0_ROLLOUT_DIR:-"${RUN_DIR}/rollout_dataset"}
ROLLOUT_HF_REPO=${L0_ROLLOUT_HF_REPO:-"hi-todayis-jh/per-context-rb-l0-${COST_OFFSET_TOKENS}-qwen3-1.7b-compression-bs32-32k-${SLURM_JOB_ID:-local}-rollouts"}
ROLLOUT_PRIVATE=${L0_ROLLOUT_PRIVATE:-false}

for option in DRY_RUN PREPARE_ONLY; do
    [[ ${!option} == 0 || ${!option} == 1 ]] || { echo "${option} must be 0 or 1." >&2; exit 2; }
done
[[ ${MAX_PROMPT_LENGTH} =~ ^[1-9][0-9]*$ ]] || { echo "L0_MAX_PROMPT_LENGTH must be positive." >&2; exit 2; }
[[ ${COST_OFFSET_TOKENS} =~ ^(0|[1-9][0-9]*)$ ]] || { echo "L0_COST_OFFSET_TOKENS must be a nonnegative integer." >&2; exit 2; }
[[ ${CHECK_EOS} == true || ${CHECK_EOS} == false ]] || { echo "L0_CHECK_EOS must be true or false." >&2; exit 2; }
[[ ${ADV_ESTIMATOR} == fixed_n_rb_offset_cost_aware_marginrl || ${ADV_ESTIMATOR} == maxrl || ${ADV_ESTIMATOR} == f_cov ]] || { echo "Unsupported compression advantage estimator." >&2; exit 2; }
[[ ${ROLLOUT_HF_REPO} =~ ^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$ ]] || { echo "L0_ROLLOUT_HF_REPO must be owner/name." >&2; exit 2; }
[[ ${ROLLOUT_PRIVATE} == true || ${ROLLOUT_PRIVATE} == false ]] || { echo "L0_ROLLOUT_PRIVATE must be true or false." >&2; exit 2; }
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + 32768))
CONFIG_ONLY=false
for argument in "$@"; do
    case ${argument} in
        --cfg|--cfg=*|-c|--help|-h|--hydra-help|--info|--info=*) CONFIG_ONLY=true ;;
    esac
done

unset PYTHONHOME RAY_ADDRESS ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH=${REPO_ROOT}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SEED=79
export WANDB_INIT_TIMEOUT=60

PREPARE_CMD=(
    "${PYTHON_BIN}" -m examples.maxrl_data_preprocess.compression
    --local_dir "${DATA_DIR}"
    --dataset_repo "${DATASET_REPO}" --revision "${DATASET_REVISION}"
    --prompt_format qwen3 --tokenizer "${MODEL}" --tokenizer_revision "${MODEL_REVISION}"
    --max_prompt_length "${MAX_PROMPT_LENGTH}"
)
MODEL_CMD=(
    "${PYTHON_BIN}" -c
    'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"]))'
    "${MODEL}" "${MODEL_REVISION}"
)
# Configuration previews use the source ID; an actual run substitutes the exact
# snapshot directory returned by the pinned download, never the moving Hub head.
MODEL_PATH=${MODEL}
if [[ ${DRY_RUN} == 0 && ${CONFIG_ONLY} == false ]]; then
    "${PREPARE_CMD[@]}"
    if [[ ${PREPARE_ONLY} == 1 ]]; then
        echo "Prepared all 3200 compression rows and audited Qwen3 thinking prompts: ${DATA_DIR}"
        exit 0
    fi
    "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 8, "Expected eight allocated, visible CUDA GPUs"'
    MODEL_PATH=$("${MODEL_CMD[@]}")
    [[ -f ${MODEL_PATH}/config.json ]] || { echo "Pinned model snapshot is incomplete." >&2; exit 1; }
    export TMPDIR=${RAY_DIR%/*}/tmp
    export TRITON_CACHE_DIR=${RAY_DIR%/*}/triton
    export WANDB_DIR=${RUN_DIR}/wandb
    mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${RAY_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}" "${WANDB_DIR}"
fi

TRAIN_CMD=(
    "${PYTHON_BIN}" -u -W ignore -m verl.trainer.main_ppo
    "algorithm.adv_estimator=${ADV_ESTIMATOR}"
    "algorithm.cost_offset_tokens=${COST_OFFSET_TOKENS}"
    "data.train_files='${DATA_DIR}/train.parquet'"
    "data.val_files='${DATA_DIR}/unused_validation.parquet'"
    data.train_batch_size=32
    data.val_batch_size=1
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    data.max_response_length=32768
    data.filter_overlong_prompts=True
    data.truncation=error
    data.apply_chat_template=True
    "actor_rollout_ref.model.path='${MODEL_PATH}'"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_MODEL_LEN}"
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.2
    actor_rollout_ref.actor.grad_clip=0.3
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.entropy_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=16
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.top_k=20
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.max_num_seqs=16
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.force_eos=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0
    reward_model.reward_manager=multi_thread
    "++reward_model.reward_kwargs.check_eos=${CHECK_EOS}"
    ++reward_model.reward_kwargs.score_after_thinking=True
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.val_before_train=False
    trainer.val_only=False
    trainer.val_on_last_step=False
    trainer.eval_on_last_step=False
    trainer.test_freq=-1
    trainer.save_freq=10
    trainer.total_epochs=1
    trainer.total_training_steps=100
    trainer.rollout_dataset.enabled=true
    "trainer.rollout_dataset.local_dir='${ROLLOUT_DIR}'"
    "trainer.rollout_dataset.hub_repo_id=${ROLLOUT_HF_REPO}"
    "trainer.rollout_dataset.private=${ROLLOUT_PRIVATE}"
    trainer.rollout_dataset.upload_num_workers=4
    trainer.resume_mode=disable
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    "trainer.logger=${L0_TRAIN_LOGGER:-['console','wandb']}"
    trainer.project_name=Qwen3_MaxRL_Experiments
    "trainer.experiment_name=${L0_EXPERIMENT_NAME:-per_context_rb_l0_${COST_OFFSET_TOKENS}_Qwen3-1.7B_compression_bs32_n16_32k_1epoch}"
    "trainer.default_local_dir='${CHECKPOINT_DIR}'"
    "ray_init.ray_dir='${RAY_DIR}'"
    ray_init.num_cpus=96
)

if [[ ${ADV_ESTIMATOR} == f_cov ]]; then
    # Global statistics cover the full prompt batch before GPU microbatching.
    # Keep this tied to Hydra's effective batch size, including caller overrides.
    TRAIN_CMD+=('algorithm.f_cov_num_prompts=${data.train_batch_size}')
fi

if [[ ${DRY_RUN} == 1 ]]; then
    echo "Model: ${MODEL}@${MODEL_REVISION}; data: ${DATASET_REPO}@${DATASET_REVISION} (3200 rows)."
    if [[ ${ADV_ESTIMATOR} == maxrl ]]; then
        echo "MaxRL: binary after-thinking correctness reward; check_eos=${CHECK_EOS}; no length cost."
    elif [[ ${ADV_ESTIMATOR} == f_cov ]]; then
        echo "Cross-context f_cov: cost=full response token count + ${COST_OFFSET_TOKENS}; full prompt-batch statistics; after-thinking grading; check_eos=${CHECK_EOS}."
    else
        echo "L+${COST_OFFSET_TOKENS}: cost=full response token count + ${COST_OFFSET_TOKENS}; per-context RB; after-thinking grading; check_eos=${CHECK_EOS}."
    fi
    echo "All training rollouts: ${ROLLOUT_DIR}; HF dataset: ${ROLLOUT_HF_REPO} (private=${ROLLOUT_PRIVATE})."
    printf 'Prepare command:\n'
    printf ' %q' "${PREPARE_CMD[@]}"
    printf '\n'
    printf 'Pinned model command:\n'
    printf ' %q' "${MODEL_CMD[@]}"
    printf '\n'
    printf 'Training command (source model ID is replaced by the pinned snapshot):\n'
    printf ' %q' "${TRAIN_CMD[@]}" "$@"
    printf '\n'
    exit 0
fi
exec "${TRAIN_CMD[@]}" "$@"
