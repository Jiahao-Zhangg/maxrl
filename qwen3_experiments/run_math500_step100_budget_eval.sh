#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
EVAL_ROOT=${MAXRL_EVAL_ROOT:-${REPO_ROOT}/outputs/eval_math500_step100}
CONDA_ENV=${MAXRL_EVAL_CONDA_ENV:-maxrl}
DATASET=${MAXRL_EVAL_DATASET:-${REPO_ROOT}/data/math500/test.parquet}
PLOT_PYTHON=${MAXRL_EVAL_PLOT_PYTHON:-$(conda info --base)/bin/python}

read -r -a MODELS <<< "${MAXRL_EVAL_MODELS:-maxrl cost_aware rb_cost_aware fixed_n_rb_step50 fixed_n_rb_step100 fixed_n_rb_capped_step100 fixed_n_rb_capped_step150}"
BUDGETS=(256 512 1024 2048 4096)
read -r -a GPU_IDS <<< "${MAXRL_EVAL_GPU_IDS:-0 1 2 3}"

declare -A MODEL_PATHS=(
    [maxrl]="${EVAL_ROOT}/merged/maxrl"
    [cost_aware]="${EVAL_ROOT}/merged/cost_aware"
    [rb_cost_aware]="${EVAL_ROOT}/merged/rb_cost_aware"
    [fixed_n_rb_step50]="${EVAL_ROOT}/merged/fixed_n_rb_step50"
    [fixed_n_rb_step100]="${EVAL_ROOT}/merged/fixed_n_rb_step100"
    [fixed_n_rb_capped_step100]="${EVAL_ROOT}/merged/fixed_n_rb_capped_step100"
    [fixed_n_rb_capped_step150]="${EVAL_ROOT}/merged/fixed_n_rb_capped_step150"
)
declare -A CHECKPOINT_REPOS=(
    [maxrl]="zjhhhh/maxrl-qwen3-1.7b-base-math12k-step_100"
    [cost_aware]="zjhhhh/cost-aware-maxrl-qwen3-1.7b-base-math12k-cap4-step_100"
    [rb_cost_aware]="zjhhhh/rb-cost-aware-maxrl-qwen3-1.7b-base-math12k-step_100"
    [fixed_n_rb_step50]="zjhhhh/fixed-n-rb-cost-aware-marginrl-qwen3-1.7b-base-math12k-step_50"
    [fixed_n_rb_step100]="zjhhhh/fixed-n-rb-cost-aware-marginrl-qwen3-1.7b-base-math12k-step_100"
    [fixed_n_rb_capped_step100]="zjhhhh/fixed-n-rb-capped-cost-aware-marginrl-qwen3-1.7b-base-math12k-cap4-step_100"
    [fixed_n_rb_capped_step150]="zjhhhh/fixed-n-rb-capped-cost-aware-marginrl-qwen3-1.7b-base-math12k-cap4-step_150"
)

mkdir -p "${EVAL_ROOT}/results" "${EVAL_ROOT}/logs"

run_wave() {
    local model=$1
    local status=0
    local batch_start

    for ((
        batch_start = 0;
        batch_start < ${#BUDGETS[@]};
        batch_start += ${#GPU_IDS[@]}
    )); do
        local pids=()
        local labels=()
        local slot

        for slot in "${!GPU_IDS[@]}"; do
            local index=$((batch_start + slot))
            if ((index >= ${#BUDGETS[@]})); then
                break
            fi
            local budget=${BUDGETS[$index]}
            local gpu=${GPU_IDS[$slot]}
            local label="${model}_max_tokens_${budget}"
            if [[
                -s "${EVAL_ROOT}/results/${label}_summary.json"
                && -s "${EVAL_ROOT}/results/${label}_samples.jsonl"
            ]]; then
                printf 'Skipping completed evaluation: %s\n' "${label}"
                continue
            fi
            labels+=("${label}")
            (
                cd "${REPO_ROOT}"
                exec conda run --no-capture-output -n "${CONDA_ENV}" \
                    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="${gpu}" \
                    CUDA_DEVICE_ORDER=PCI_BUS_ID \
                    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
                    python "${SCRIPT_DIR}/eval_math500_output_budget.py" \
                        --model-path "${MODEL_PATHS[$model]}" \
                        --model-label "${model}" \
                        --checkpoint-repo "${CHECKPOINT_REPOS[$model]}" \
                        --dataset "${DATASET}" \
                        --output-dir "${EVAL_ROOT}/results" \
                        --max-output-len "${budget}" \
                        --num-samples 4 \
                        --grader-workers 8
            ) >"${EVAL_ROOT}/logs/${label}.log" 2>&1 &
            pids+=("$!")
            printf 'Launched %s on GPU %s as PID %s\n' \
                "${label}" "${gpu}" "$!"
        done

        for index in "${!pids[@]}"; do
            if ! wait "${pids[$index]}"; then
                printf 'Evaluation failed: %s (see %s)\n' \
                    "${labels[$index]}" \
                    "${EVAL_ROOT}/logs/${labels[$index]}.log" >&2
                status=1
            fi
        done
    done
    return "${status}"
}

for model in "${MODELS[@]}"; do
    run_wave "${model}"
done

if [[ "${MAXRL_EVAL_SKIP_PLOT:-0}" != 1 ]]; then
    "${PLOT_PYTHON}" "${SCRIPT_DIR}/plot_math500_output_budget.py" \
        --results-dir "${EVAL_ROOT}/results" \
        --output "${EVAL_ROOT}/math500_mean_at_4_by_output_budget.png"
fi
