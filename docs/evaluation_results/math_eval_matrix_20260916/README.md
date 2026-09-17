# Five-model math evaluation — 2026-09-16 snapshot

**700/700 completion:** [results and figures](../math_eval_matrix_20260916_h100_completion/README.md) and [all raw trajectories](../math_eval_matrix_20260916_h100_completion/RAW_TRAJECTORIES.md) are now available. This directory retains the original 541-point metrics and receipts.

**Partial run: 541 of 700 budget/seed points completed.** The 48-hour holder
expired on 2026-09-16 at 06:41 CDT (11:41 UTC). The evaluation was interrupted by
the allocation time limit, not an out-of-memory failure. Completed results are
retained; unfinished points are not represented as zero scores.

| Evaluation | Completed / requested points | Status |
|---|---:|---|
| Eval1: mean@4 under a per-response cap | 100 / 100 | Complete |
| Eval2: solved within an individual cumulative budget | 300 / 300 | Complete |
| Eval3: solved within a shared cumulative budget | 141 / 300 | Partial |

An individual point is one model × dataset × evaluation × budget × seed.
Each curve point requires all three requested seeds; partial seed groups are
retained in the data but **not plotted**.

## Results

- [Eval1 tables and evaluation figures](results.md).
- [Eval1 mean@4 values](eval1_mean_at_4.csv).
- Eval2: [PNG](eval2_questions_solved.png), [PDF](eval2_questions_solved.pdf),
  [SVG](eval2_questions_solved.svg), [three-seed statistics](eval2_questions_solved.csv).
- Partial Eval3: [PNG](eval3_questions_solved.png), [PDF](eval3_questions_solved.pdf),
  [SVG](eval3_questions_solved.svg), [three-seed statistics](eval3_questions_solved.csv).
- [All 541 completed per-seed points](all_points.csv).
- [Completion counts and early-stop token accounting](snapshot.json).
- [Point-level checksums and provenance](point_receipts.csv).

Eval3 completion by dataset: AMC22+23 **75/75**, Minerva Math **63/75**,
MATH-500 **3/75**, OlympiadBench **0/75**. Comparisons involving missing Eval3
points cannot be drawn from this snapshot.

## Protocol

All five models receive the same ordinary math prompt and tokenizer/chat template;
L1-Exact receives **no length instruction**. Eval1 reports mean correctness across
four independent responses, **not pass@4**. Grading uses the repository's
MathVerify scorer with a one-second per-item deadline.

Eval2 stops each question after its first correct response or budget exhaustion.
Failed attempts and EOS tokens count; prompt tokens do not. Saved tokens are not
transferred to other questions. Its x axis is the allocated **Individual Budget**,
not actual consumption. Early stopping used 204,292,516 of 364,024,320 allocated
output tokens, a **43.88% reduction relative to exhausting all allowances**; this
is not a measured wall-clock speedup.

Eval3 shares `number_of_questions × B` tokens across seeded shuffled sweeps and
skips solved questions. Its x axis is **Averaged Shared Budget**. Both plots show
the **number of questions solved**, with mean ± one sample standard deviation
across seeds 0, 1, and 2, rather than percentages or confidence intervals.

Datasets: MATH-500 (500), Minerva Math (272), text-only English OlympiadBench (674),
and the L1/DeepScaleR-style AMC22+23 subset (83). The latter is not the separate
89-question original-wording AMC selection.

Hardware: four NVIDIA GH200 GPUs, one model inference engine per GPU. Decoding:
BF16, temperature 0.6, top-p 0.95, top-k -1; Eval2/3 per-response cap 4096 tokens.

## Reproducibility and scope

- [Evaluation instructions](../../../qwen3_experiments/math_eval_matrix.md).
- [Pinned five-model/four-dataset configuration](../../../qwen3_experiments/math_eval_matrix.json).
- [Run manifest](run_manifest.json): model/dataset commit hashes, file hashes,
  package versions, decoding/grading settings, and evaluator source hashes.
- [Earlier Eval1 manifest](eval1_source_manifest.json): provenance for the 55
  completed Eval1 points reused after the Eval2 early-stop update. They were
  checksum-verified and independently re-audited before reuse; Eval1 was unchanged.

The CSV receipts identify each completed point and record hashes of its original
summary and raw artifacts. The **541 completed points' original trajectories are
available as [GitHub Release assets](https://github.com/Jiahao-Zhangg/maxrl/releases/tag/math-eval-matrix-20260916-raw-541)**,
with download links, content descriptions, and checksum instructions in
[Raw trajectories](RAW_TRAJECTORIES.md). Responses, output token IDs, gold answers,
scores, and budget ledgers are retained; only machine-local summary metadata is
redacted. The raw data are not committed to Git history.

This publication excludes full input dataset snapshots, checkpoints, training
logs, credentials, incomplete attempts, and local filesystem paths. It is an
analysis snapshot, not a standalone resumable run directory.

The evaluator source hashes in `run_manifest.json` describe the actual executed
generation/accounting code. Documentation, plotting, and publication metadata can
be updated without changing those generation results.

Validation:

```bash
python -m pytest tests/utils/test_math_eval_matrix_on_cpu.py tests/utils/test_math500_eval_series_on_cpu.py -q -p no:cacheprovider
bash -n qwen3_experiments/run_math_eval_matrix_on_holder.sh
```

The CPU suite tests accounting, early stopping, per-question seeds, independent
raw-ledger audits, resume/reuse, and plotting. It does not rerun model inference.
