# Compression models: individual budgets relative to base mean length

This queue starts after the current six-dataset cross-budget queue has completed
all 150 points, passed its result audits, and cleaned its model caches.

Models: Qwen3-1.7B base, MaxRL, ER, L+0, L+4096, and f_cov (L0 = 0). The five
trained models use the public, pinned step-100 `-final` repositories. Base uses
the pinned official Qwen3-1.7B checkpoint.

Datasets run in this order: Minerva Math, OlympiadBench, AMC22+23, AIME24, AIME25,
and AIME26. Finish all six models and all four budgets in one dataset before
starting the next: 24 points per dataset, 144 total. MATH500 is excluded.

## Budget definition

For each dataset, L is the base model's mean response length from the audited
32k, four-responses-per-question reference table. It includes all responses,
including incorrect and unfinished thinking. Use the mean exactly as published
to one decimal place, then compute `ceil(multiplier * L)` integer output tokens.
The four multipliers are 0.5, 1, 2, and 3. Every model uses the same budgets for
that dataset. Token accounting includes thinking, answers and EOS, but not input.

The reference generation used temperature/top-p/top-k 0.6/0.95/-1 and seed 0.
This new evaluation, including base, consistently uses 0.6/0.95/20 and seed 0.

| Dataset | Base mean L | 0.5 × L | 1 × L | 2 × L | 3 × L |
| --- | ---: | ---: | ---: | ---: | ---: |
| Minerva Math | 7,064.2 | 3,533 | 7,065 | 14,129 | 21,193 |
| OlympiadBench | 11,135.8 | 5,568 | 11,136 | 22,272 | 33,408 |
| AMC22+23 | 11,719.0 | 5,860 | 11,719 | 23,438 | 35,157 |
| AIME24 | 17,997.3 | 8,999 | 17,998 | 35,995 | 53,992 |
| AIME25 | 17,885.6 | 8,943 | 17,886 | 35,772 | 53,657 |
| AIME26 | 17,890.3 | 8,946 | 17,891 | 35,781 | 53,671 |

These are per-question individual budgets (`eval2`). Stop after the first correct
answer; unused allowance stays with its question. Each response is capped at
32,768 tokens or the remaining question allowance. Each budget uses fresh
sampling. Only the nonempty answer after completed thinking is graded; no extra
EOS or boxed-answer requirement. Unfinished or reopened thinking scores zero.

## Execution and recovery

Preparation copies only scripts, model identities, tokenizer-verified prompts,
reference tables and grading provenance. It stages no model weights and uses
no GPUs. Launch a persistent supervisor on the allocated compute node:

```bash
python -m qwen3_experiments.compression_relative_individual_budget prepare \
  --after-cross "$CROSS_ROOT" --base-reference "$BASE_REFERENCE" \
  --base-evaluation "$BASE_INDIVIDUAL_ROOT" \
  --output-root "$RELATIVE_ROOT" --scratch "$RELATIVE_SCRATCH"
PYTHONPATH="$RELATIVE_ROOT/runtime" python -m \
  qwen3_experiments.compression_relative_individual_budget launch \
  --output-root "$RELATIVE_ROOT"
```

Eight GPUs execute independent model/budget points, largest budgets first,
with the existing dataset barrier. Weights are downloaded to this queue's
dedicated compute-local cache only after its predecessor finishes. Each model
cache is deleted after all four points for that dataset are verified and its
workers exit. Completed points are reused after restart; failed points retry.
Other queues' model caches, including the shared base model, are preserved.

Full compressed responses, token IDs, grades and budget ledgers are retained.
Reports contain exact budgets, multipliers, pass rates, solved counts and token
usage in JSON/CSV, with a Markdown comparison per dataset. The compute-node
supervisor monitors space, offloads verified closed results to its own scratch
if needed, restores them when space is available, and restarts a failed queue.
