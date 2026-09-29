#!/usr/bin/env bash
# Qwen3-1.7B MaxRL on the same pinned compression subset, with no EOS reward gate.
# Reuse batch 32 x 16, 32k responses, lr=1e-6, KL=0, seed 79 and 100 steps.
# All 51,200 training rollouts are saved and uploaded to a public HF dataset.
# Preview: DRY_RUN=1 bash "$0"; Hydra preview: bash "$0" --cfg job --resolve.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
export L0_ADV_ESTIMATOR=maxrl L0_COST_OFFSET_TOKENS=0 L0_CHECK_EOS=false
export L0_RUN_DIR=${L0_RUN_DIR:-"${REPO_ROOT}/outputs/maxrl_no_eos_qwen3_1_7b_compression_bs32_n16_32k_${SLURM_JOB_ID:-local}"}
export L0_ROLLOUT_HF_REPO=${L0_ROLLOUT_HF_REPO:-"hi-todayis-jh/maxrl-no-eos-qwen3-1.7b-compression-bs32-n16-32k-${SLURM_JOB_ID:-local}-rollouts"}
export L0_EXPERIMENT_NAME=${L0_EXPERIMENT_NAME:-maxrl_no_eos_Qwen3-1.7B_compression_bs32_n16_32k_1epoch}
exec bash "${SCRIPT_DIR}/run_qwen3_1_7b_compression_per_context_rb_l0_0.sh" "$@"
