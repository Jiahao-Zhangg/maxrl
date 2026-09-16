# Model × evaluation × dataset matrix

`math_eval_matrix.json` specifies five pinned checkpoints, four pinned benchmark
snapshots, the complete budget grid, and decoding/grading settings. A full run has
60 model/dataset/protocol groups and 700 individual budget/seed points.

The [2026-09-16 result snapshot](../docs/evaluation_results/math_eval_matrix_20260916/README.md)
contains complete Eval1/Eval2 results and partial Eval3 results, stopped by the
holder's time limit. It includes tables, figures, per-seed metrics, and input/source
provenance, but no raw rollouts, benchmark data, model weights, or local paths.

- Eval1: caps 256, 512, 1024, 2048, 4096; seed 0; four independent responses per
  question. Report mean correctness across all `4 × number_of_questions` responses,
  **not pass@4**.
- Eval2: budgets 512, 1024, 2048, 4096, 8192; seeds 0, 1, 2. Every question has
  its own cumulative output-token allowance. With `stop_on_first_success: true`,
  stop that question immediately after its first correct response; unsuccessful
  questions exhaust their allowance. Unused tokens are **not transferred** to other
  questions. A question is solved if any attempt within its allowance is correct.
  This preserves the solved-within-budget metric, not total token/rollout counts.
- Eval3: the same budget grid and seeds. All `M` questions share `M × B` output
  tokens. Seeded shuffled sweeps skip solved questions. Unvisited questions remain
  in the denominator. Stop when the budget is exhausted or all questions are solved.

Every failed attempt and generated EOS token counts toward cost; prompt tokens do
not. Eval2/3 cap each response at 4096 tokens or the remaining allowance. They do not
count sample numbers as budget, truncate previously scored responses, or assume
that partial answers are correct.

## Common prompt and grader

All five models, **including L1-Exact**, get exactly the same ordinary math prompt
and the pinned Qwen3-1.7B-Base tokenizer/chat template. No `Think for N tokens.`
instruction is added. The suffix is the one used in this repository's historical
MATH-500 evaluator. There is no prompt truncation or silent question filtering.

Decoding: temperature 0.6, top-p 0.95, top-k -1, BF16, one GPU per inference engine.
Every prompt/attempt slot has a deterministic seed shared across models. The full
`(repeat seed, question, attempt)` tuple is hashed so adjacent repeats do not reuse
shifted attempt streams. Grading
uses `verl.workers.reward_manager.multi_thread_naive.MathVerifyScorer` with the
historical one-second per-item deadline. This is a common MathVerify comparison,
not each benchmark's potentially different official grader/tolerance.

Input sources and full revisions are in the JSON configuration:

- `HuggingFaceH4/MATH-500`: 500 questions.
- `math-ai/minervamath`: 272 questions.
- `math-ai/olympiadbench`: 674 English text-only mathematics questions.
- `watermelonhjg/AMC_22_23`: the 83-question L1/DeepScaleR-style numerical-answer
  version of AMC22+23. This is distinct from the 89-question original-wording
  `scottgeng00/amc_22-24` selection. Do not mix results between these versions.

## Launch on an existing idle holder

```bash
bash qwen3_experiments/run_math_eval_matrix_on_holder.sh JOB_ID /path/to/persistent/results
```

An optional third argument reuses a node-local preparation directory. The launcher
does not submit or cancel Slurm jobs, changes no training watchers, acquires the
shared holder lock, and refuses to overlap occupied GPUs. Configuration, Conda
environment and plotting interpreter can be overridden with `MAXRL_EVAL_CONFIG`,
`MAXRL_EVAL_CONDA_ENV`, and `MAXRL_EVAL_PLOT_PYTHON`.
On GH200 the launcher explicitly selects CUDA 12.6, GCC 13, and CUDA architecture
9.0, avoiding the environment's `90a` auto-detection issue in FlashInfer.
`MAXRL_EVAL_RETRY_FAILED=1` retries previously failed groups after a diagnosed fix.

Models and actor-only training shards stay on node-local scratch; optimizer states
are never downloaded. All source files are size/hash checked against pinned Hub
metadata. Conversion checks tensor names/shapes/dtypes, tied embedding equality,
token IDs, and chat templates. Persistent inputs and receipts allow reconstruction
on another holder without retaining full model weights in the shared filesystem.

## Output and resume

Each point saves gzipped raw responses, actual output token IDs, scores, attempt
seeds, question IDs, budget balances, per-question counters, and a summary. An
independent pass over the saved ledger must reproduce the counters before a point
is marked complete. Completed artifacts have SHA256 receipts. Incomplete attempts
are kept separately, never treated as completed results or deleted automatically.

Re-running the launcher with the same output root resumes only unfinished points.
Inputs, package versions, config and evaluator source hashes must match. A changed
protocol requires a fresh result root. Failed groups are left for inspection;
`run_math_eval_matrix.py --retry-failed` explicitly retries them while still skipping
verified completed points. Creating `PAUSE` in the output root stops new points
after the current point finishes; remove it only when intentionally resuming.

`status.json`, `progress/gpu_*.json` and `logs/` report current progress. Reports are
regenerated every five minutes and on completion:

- `reports/results.md` and `reports/eval1_mean_at_4.csv`: Eval1 tables.
- `reports/eval2_questions_solved.{png,pdf,svg,csv}`: Individual Budget on x.
- `reports/eval3_questions_solved.{png,pdf,svg,csv}`: Averaged Shared Budget on x.
- `reports/all_points.csv`: all completed budget/seed measurements.

Both plots have one panel per dataset, **number of questions solved** on y, and
mean ± one sample standard deviation across the three seeds. A plot point is not
shown until every requested seed is complete. No fitted curves, extrapolation, or
percentage-axis substitution is used.

Eval2's x axis remains the **allocated** individual budget, not the lower actual
consumption after early stopping. Its summaries also record unused budget and the
number of early-stopped questions. Per-question attempt seeds do not depend on
batch membership, although GPU floating-point/batching effects mean that changing
the batch composition is not a promise of bitwise-identical generated responses.
Eval1 must still generate all four responses; Eval3 already skips solved questions.

When enabling Eval2 early stopping mid-run, pause the original run at point
boundaries and use a fresh output root. Set `MAXRL_EVAL_REUSE_EVAL1_FROM` to the old
root when launching: completed Eval1 points are checksum-verified, independently
audited, and copied with their original manifest/summary provenance. All inputs,
package versions, grading and seed code, and non-Eval2 settings must match.
The original results are retained; Eval2/3 points are never imported by this option.

```bash
python -m pytest tests/utils/test_math_eval_matrix_on_cpu.py -q
python qwen3_experiments/plot_math_eval_matrix.py --output-root /path/to/results
```

CPU tests use deterministic fake generation, not actual model results.
