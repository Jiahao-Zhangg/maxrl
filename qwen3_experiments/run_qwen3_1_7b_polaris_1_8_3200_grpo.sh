#!/usr/bin/env bash
# GRPO with the settings and prepared inputs from the L+0 run on allocation 146103.
# Uses a frozen L+0 runtime with EOS and after-thinking grading enabled.
# Run in the maxrl conda environment on one node with eight allocated GPUs.
# Preview without training: bash "$0" --cfg job --resolve
# Optional environment overrides: GRPO_RUN_DIR, GRPO_MODEL_PATH, GRPO_DATA_DIR,
# GRPO_L0_REPO_ROOT, GRPO_L0_RUN_DIR, GRPO_RAY_DIR, GRPO_TRAIN_LOGGER,
# GRPO_RUNTIME_REPO, GRPO_CHECKPOINT_DIR.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
L0_REPO_ROOT=${GRPO_L0_REPO_ROOT:-"${REPO_ROOT}/../maxrl-eval-145514"}
L0_REPO_ROOT=$(cd -- "${L0_REPO_ROOT}" && pwd)
L0_RUN_DIR=${GRPO_L0_RUN_DIR:-"${L0_REPO_ROOT}/outputs/per_context_rb_l0_0_qwen3_1_7b_polaris_1_8_3200_bs32_32k_146103"}

GRPO_RUN_DIR=${GRPO_RUN_DIR:-"${REPO_ROOT}/outputs/grpo_qwen3_1_7b_polaris_1_8_3200_bs32_32k_${SLURM_JOB_ID:-local}"}
GRPO_DATA_DIR=${GRPO_DATA_DIR:-"${L0_RUN_DIR}/data"}
# Shared copy of the exact Qwen/Qwen3-1.7B revision used to initialize L+0:
# 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e (not an L+0 training checkpoint).
GRPO_MODEL_PATH=${GRPO_MODEL_PATH:-"${L0_REPO_ROOT}/outputs/queued_per_context_rb_l0_0_after_maxrl_145514/model"}

for required_file in \
    "${GRPO_DATA_DIR}/train.parquet" \
    "${GRPO_DATA_DIR}/unused_validation.parquet" \
    "${GRPO_MODEL_PATH}/config.json"; do
    [[ -f ${required_file} ]] || { echo "Missing L+0 input: ${required_file}" >&2; exit 1; }
done

# Resolve paths before the inherited launcher changes the working directory.
GRPO_RUN_DIR=$(realpath -m -- "${GRPO_RUN_DIR}")
GRPO_DATA_DIR=$(cd -- "${GRPO_DATA_DIR}" && pwd)
GRPO_MODEL_PATH=$(cd -- "${GRPO_MODEL_PATH}" && pwd)

# Keep the original L+0 checkout unchanged while freezing this run's reward code.
GRPO_RUNTIME_REPO=${GRPO_RUNTIME_REPO:-"${GRPO_RUN_DIR}/runtime"}
python "${SCRIPT_DIR}/grpo_compute_control.py" snapshot \
    --source-repo "${L0_REPO_ROOT}" --overlay-repo "${REPO_ROOT}" \
    --runtime-dir "${GRPO_RUNTIME_REPO}"
GRPO_RUNTIME_REPO=$(cd -- "${GRPO_RUNTIME_REPO}" && pwd)
L0_LAUNCHER=${GRPO_RUNTIME_REPO}/qwen3_experiments/run_qwen3_1_7b_polaris_1_8_3200_per_context_rb_l0_0.sh

unset PYTHONHOME RAY_ADDRESS ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH=${GRPO_RUNTIME_REPO}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SEED=79

export MAXRL_TRAIN_RUN_DIR=${GRPO_RUN_DIR}
export MAXRL_MODEL_PATH=${GRPO_MODEL_PATH}
# L+0's prepared inputs require 1280 prompt tokens; the generic launcher defaults to 1024.
export MAXRL_MAX_PROMPT_LENGTH=1280
export MAXRL_RAY_DIR=${GRPO_RAY_DIR:-"/tmp/grpo${SLURM_JOB_ID:-$$}/ray"}
export MAXRL_TRAIN_LOGGER=${GRPO_TRAIN_LOGGER:-"['console','wandb']"}
export TMPDIR=${MAXRL_RAY_DIR%/*}/tmp
export TRITON_CACHE_DIR=${MAXRL_RAY_DIR%/*}/triton
export WANDB_DIR=${GRPO_RUN_DIR}/wandb
export WANDB_INIT_TIMEOUT=60

# Hydra's configuration preview needs no output or runtime directories.
CONFIG_ONLY=false
for argument in "$@"; do
    case ${argument} in
        --cfg|--cfg=*|-c|--help|-h|--hydra-help|--info|--info=*) CONFIG_ONLY=true ;;
    esac
done
if [[ ${CONFIG_ONLY} == false ]]; then
    mkdir -p "${GRPO_RUN_DIR}" "${MAXRL_RAY_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}" "${WANDB_DIR}"
fi

# Inherit batch 32, n=16, response 32768, lr=1e-6, one epoch / 100 steps,
# eight GPUs, token-mean loss, zero KL, checkpoint every 10 steps, no evaluation.
# cost_offset_tokens=0 is inherited from L+0 but is unused by the GRPO estimator.
exec bash "${L0_LAUNCHER}" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    ++reward_model.reward_kwargs.check_eos=True \
    ++reward_model.reward_kwargs.score_after_thinking=True \
    actor_rollout_ref.rollout.force_eos=False \
    "data.train_files=${GRPO_DATA_DIR}/train.parquet" \
    "data.val_files=${GRPO_DATA_DIR}/unused_validation.parquet" \
    trainer.experiment_name=grpo_Qwen3-1.7B_Polaris-1-8-3200_bs32_n16_32k_1epoch \
    trainer.save_freq=10 \
    "trainer.default_local_dir=${GRPO_CHECKPOINT_DIR:-${GRPO_RUN_DIR}/checkpoints}" \
    "$@"
