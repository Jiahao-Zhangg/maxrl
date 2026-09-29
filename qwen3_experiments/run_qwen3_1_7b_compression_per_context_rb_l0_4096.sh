#!/usr/bin/env bash
# Same compression recipe as L+0, with cost L+4096 and after-thinking-only grading.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
export L0_COST_OFFSET_TOKENS=4096 L0_CHECK_EOS=false
export L0_RUN_DIR=${L0_RUN_DIR:-"${REPO_ROOT}/outputs/per_context_rb_l0_4096_no_eos_qwen3_1_7b_compression_bs32_32k_${SLURM_JOB_ID:-local}"}
export L0_ROLLOUT_HF_REPO=${L0_ROLLOUT_HF_REPO:-"hi-todayis-jh/per-context-rb-l0-4096-no-eos-qwen3-1.7b-compression-bs32-32k-${SLURM_JOB_ID:-local}-rollouts"}
export L0_EXPERIMENT_NAME=${L0_EXPERIMENT_NAME:-per_context_rb_l0_4096_no_eos_Qwen3-1.7B_compression_bs32_n16_32k_1epoch}
exec bash "${SCRIPT_DIR}/run_qwen3_1_7b_compression_per_context_rb_l0_0.sh" "$@"
