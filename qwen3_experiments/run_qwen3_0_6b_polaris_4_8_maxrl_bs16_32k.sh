#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Qwen3-0.6B on the pinned Polaris 4/8--7/8 subset, using the existing MaxRL
# environment, four GPUs, one epoch, and AIME25/MATH-500 validation.
# Each training batch contains 16 prompts x 16 rollouts = 256 responses.
# Save every 250 steps; the trainer also saves the final step.
export MAXRL_TOTAL_EPOCHS=${MAXRL_TOTAL_EPOCHS:-1}
export MAXRL_SAVE_FREQ=${MAXRL_SAVE_FREQ:-250}
export MAXRL_EXPERIMENT_NAME=${MAXRL_EXPERIMENT_NAME:-maxrl_Qwen3-0.6B_polaris_4_8_bs16_n16_32k_1epoch}

# The 32,768-token output cap includes thinking and the final answer. Reserve
# another 1,024 tokens for prompts. This rollout backend requires its token
# batching limit to cover the full context when chunked prefill is enabled.
# Use one response per GPU micro-batch for long-sequence updates/log-probs.
# Additional Hydra overrides supplied by the caller take precedence.
exec "${SCRIPT_DIR}/run_qwen3_0_6b_polaris_4_8_maxrl.sh" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=32768 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=33792 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_model_len=33792 \
    actor_rollout_ref.rollout.max_num_batched_tokens=33792 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    "$@"
