# Compression f_cov and MaxRL final evaluation

`compression_final_evaluation.py` appends this order to allocation 146103:

1. Wait for f_cov training and every checkpoint/rollout upload to finish.
2. f_cov step 100: nine benchmarks, then seven-dataset individual pass@budget.
3. Verify both result sets and remove only this queue's f_cov model caches.
4. MaxRL step 100: the same two evaluations, then remove its model caches.

Both checkpoints are bound to immutable commits from their verified final
archive receipts. No training plan or active trainer is changed. The queue,
retrying supervisor, GPU launchers, model downloads and conversion all run on
the assigned compute node. Model caches use a private `/tmp` directory. Results
may use a dedicated home-filesystem directory exposed through a link in the
primary repository's `outputs/`; no additional checkout or branch is created.

Nine datasets: MATH500 (500), Minerva Math (272), OlympiadBench (674), AMC22+23
(83), AIME24/25/26 (30 each), Polaris-Test (100), Polaris-Test-4-8 (100). Each
model generates 7,276 responses: four samples per question, seed 42,
32,768-token cap, temperature 0.6, top-p 0.95 and top-k 20. The main metrics and
exact 1k/2k/4k/8k/16k/32k prefixes use after-thinking grading. Reports use the
corrected Qwen3 after-thinking baseline and identify compression-trained models.

Individual pass@budget uses the first seven datasets (1,619 questions), seed 0,
budgets 8,192/16,384/32,768/49,152/65,536, and the same 0.6/0.95/20 sampling.
Each attempt is capped at 32,768 tokens or its question's remaining budget;
the question stops at its first success. Eight GPUs process independent
dataset/budget points, largest budgets first. These are 35 points per model,
70 in total. All questions retain their own allowance; budgets are not pooled.

Both evaluations grade only the answer after completed thinking. There is no
additional EOS or boxed-answer requirement. Original response texts, token IDs,
seeds, per-question token ledgers, result checksums and model identities remain
in the output directory after cache cleanup.

Prepare on the allocated compute node with the maxrl Python environment:

```bash
python -m qwen3_experiments.compression_final_evaluation prepare \
  --output-root "$EVALUATION_ROOT" \
  --f-cov-training "$FCOV_TRAINING_ROOT" --maxrl-training "$MAXRL_TRAINING_ROOT" \
  --nine-reference "$COMPLETED_NINE_ROOT" --budget-reference "$EXISTING_BUDGET_ROOT"
python -m qwen3_experiments.compression_final_evaluation launch \
  --output-root "$EVALUATION_ROOT"
```

Preparation freezes code, questions and settings without loading evaluation
models or taking GPUs. `queue_status.json` records the current model and phase.
Per-model results are under `stages/<model>/nine_datasets/` and
`stages/<model>/pass_at_budget/`; combined JSON/CSV files and the final audit
are under `report/`. The launch receipt points to the persistent supervisor.

The service retries failed queues, and a replacement queue adopts existing
phase workers before starting new work. Workers retain completed responses and
budget points across retries. The service monitors result, compute and project
disk space every 30 seconds. Below 50 GiB on the result filesystem it offloads
only closed, checksummed artifacts belonging to this queue to compute-local
storage, retaining their canonical paths as links; above 100 GiB it restores
them. It never deletes another allocation's cache or unverified evaluation data.
