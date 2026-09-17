# MATH-500: Eval1, Eval2 and Eval3

These are the evaluators and plotting configurations used for the Qwen3-1.7B
MATH-500 comparisons. Run commands from the repository root.

## Protocols and entry points

| Evaluation | Entry point | Budget and metric |
|---|---|---|
| Eval1: single-response mean@4 | [eval_math500_output_budget.py](eval_math500_output_budget.py) | Four responses per question, each with the selected output cap. Accuracy is the mean binary score over all 2,000 responses. |
| Eval2: individual budget | [eval_math500_total_token_budget.py](eval_math500_total_token_budget.py) | Each question exhausts its own cumulative output-token budget, continuing after success. A question is solved if any response is correct. |
| Eval3: shared budget, skip solved | [eval_math500_cross_context_budget.py](eval_math500_cross_context_budget.py), `--question-selection sweep` | All 500 questions share `500 * b` output tokens. Seeded shuffled sweeps skip questions once solved. All 500 questions, including unvisited ones, remain in the denominator. |

Eval2 and Eval3 cap each response at 4,096 tokens, also respecting the remaining
budget. Eval3 may stop early if every question is solved. Costs count generated
output tokens; prompt tokens are not charged.

The shared-budget evaluator also supports `--question-selection random-with-replacement`:
IID question draws include already solved questions. This is a separate comparison
from the skip-solved Eval3 figures.

**Historical naming:** the shared-budget evaluator retains `Eval4` in some logs and
`eval4_cross_context_global_budget` in result metadata. This is the protocol called
**Eval3** in the current figures; metadata remains compatible with saved results.

## Inputs and environment

- Python 3.10 or later; install this repository with `pip install -e .`.
- A merged Hugging Face checkpoint, tokenizer and matching chat template.
- MATH-500 parquet at `data/math500/test.parquet`, or an explicit `--dataset` path:
  all 500 questions, chat messages in `prompt`, `reward_model.ground_truth`,
  `extra_info`, and `data_source=DigitalLearningGmbH/MATH-lighteval`.
- Historical packages: `torch==2.6.0+cu124`, `vllm==0.8.4`, `transformers==4.51.3`,
  `datasets==3.5.0`, `math-verify==0.9.0`. Plotting uses NumPy and Matplotlib;
  parquet reading uses pandas and a parquet engine.
- Temperature 0.6, top-p 0.95, top-k -1, seed 0, maximum prompt length 1,024,
  and the repository's `MathVerifyScorer` with a one-second scoring timeout.

Historical supervisors verify these package versions and dataset SHA256
`7e674a2fb0e85770931fed4b467c7236a674adb4af58f6c426fa9fd60d736af3`.
Re-exporting the same questions can change the parquet hash; preserve the original
snapshot for historical reproduction. Data, raw responses, weights and local run
manifests are stored separately from this source code.

## Run the evaluators directly

Activate the evaluation environment and set your checkpoint paths:

```bash
export CUDA_VISIBLE_DEVICES=4
export PYTHONNOUSERSITE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
EVAL_MODEL_PATH=/path/to/merged-model
EVAL_CHECKPOINT_REPO=owner/checkpoint
EVAL_MODEL_LABEL=my_model_step150
```

Eval1, one output cap (repeat for 256, 512, 1024, 2048, 4096):

```bash
python qwen3_experiments/eval_math500_output_budget.py \
  --model-path "$EVAL_MODEL_PATH" --model-label "$EVAL_MODEL_LABEL" \
  --checkpoint-repo "$EVAL_CHECKPOINT_REPO" \
  --output-dir outputs/my_math500/eval1/results \
  --max-output-len 4096 --num-samples 4
```

Eval2, the 512–12K grid:

```bash
for budget in 512 1024 2048 4096 8192 12288; do
  python qwen3_experiments/eval_math500_total_token_budget.py \
    --model-path "$EVAL_MODEL_PATH" --model-label "$EVAL_MODEL_LABEL" \
    --checkpoint-repo "$EVAL_CHECKPOINT_REPO" \
    --output-dir outputs/my_math500/eval2/results \
    --total-output-budget "$budget" --per-rollout-cap 4096
done
```

Eval3, shared budget with solved questions skipped:

```bash
python qwen3_experiments/eval_math500_cross_context_budget.py \
  --model-path "$EVAL_MODEL_PATH" --model-label "$EVAL_MODEL_LABEL" \
  --checkpoint-repo "$EVAL_CHECKPOINT_REPO" \
  --output-dir outputs/my_math500/eval3_skip_solved/results \
  --budget-per-prompt 256 512 1024 2048 4096 \
  --question-selection sweep --per-rollout-cap 4096
```

Each Eval1 point saves samples and a summary. Eval2/Eval3 additionally save per-question
records and budget accounting. Use a fresh result directory for a new run; the
historical resume behavior differs between entry points.

## Historical supervisors and checkpoint preparation

These scripts require their referenced baseline results or prepared-model manifests.
For a new model without those artifacts, use the direct entry points above.

| Script | Purpose |
|---|---|
| [run_math500_eval1_eval2.py](run_math500_eval1_eval2.py) | Resume Eval1 or Eval2 on one specified GPU and audit completed points. |
| [run_math500_eval2_aligned.py](run_math500_eval2_aligned.py) | Read a JSON run configuration, reuse audited points and evaluate missing individual-budget points. |
| [run_math500_two_eval3.py](run_math500_two_eval3.py) | Run `eval3_skip_solved`, `eval3_iid`, or both, with isolated status files and raw-result checks. |
| [run_math500_offset256_comparison.py](run_math500_offset256_comparison.py) | Historical L+256 comparison; shared protocol constants and helpers. |
| [run_math500_cost_comparison.py](run_math500_cost_comparison.py) | Coordinate the four historical protocol variants and compare saved cost-aware checkpoints. |
| [summarize_math500_offset256_comparison.py](summarize_math500_offset256_comparison.py) | Audit and summarize the corresponding artifacts. |
| [prepare_math500_fsdp_actor.py](prepare_math500_fsdp_actor.py) | Verify and merge a pinned four-rank FSDP actor on CPU using a reference tokenizer manifest. |
| [record_prepared_model_source.py](record_prepared_model_source.py) | Record checkpoint provenance for a merged model. |

Run each script with `--help` for required paths and checkpoint identifiers.

## Plot and audit saved results

Plotting runs on CPU. JSON configurations contain repository-relative result paths;
adjust `models[].result_dir` and `model_key` for your results. A fresh source checkout
does not include raw experimental artifacts.

```bash
python qwen3_experiments/plot_math500_eval1_figure2.py --audit
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --config qwen3_experiments/plot_math500_eval2_figure2_cap8_512_12k.json --audit
python qwen3_experiments/plot_math500_eval3_figure2.py --audit
```

`--audit` checks original responses, question identities, scores and token accounting.
Omit it for cosmetic rerenders. Outputs include PDF, PNG, SVG, point/fit tables and
the exact configuration. All three plots share the Eval3 renderer.

- [Eval1 plotting guide](plot_math500_eval1_figure2.md): mean@4 and actual mean response length.
- [Eval2 plotting guide](plot_math500_eval2_figure2.md): cap8/L+256/L+512 and 512–12K variants.
- [Eval3 plotting guide](plot_math500_eval3_figure2.md): shared budget and skip-solved validation.

## CPU checks

```bash
python -m pytest tests/utils/test_math500_eval_series_on_cpu.py -q
```

These tests check supervisor resume/isolation behavior and preserve zero accuracy
when normalizing results; they do not launch generation or load weights.
