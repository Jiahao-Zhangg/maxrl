# Compression final models: cross-context pass@budget

This follow-up waits for the existing evaluation queue to complete, in order:

1. f_cov: nine-dataset evaluation.
2. f_cov: individual pass@budget on seven datasets.
3. MaxRL: nine-dataset evaluation.
4. MaxRL: individual pass@budget on seven datasets.
5. The cross-context evaluation described here.

The original queue and frozen runtimes remain unchanged. The dependency checks
both nine-dataset reports (14,552 responses), all 70 individual-budget points,
their result hashes, and the original queue's two model-cache cleanup receipts.
Creating final repositories does not replace any of these evaluations.

## Cross-context protocol

The five compression models are ER, MaxRL, L+0, L+4096, and f_cov, all at step 100.
Each model comes from its public, verified `-final` Hugging Face repository.
The plan pins the repository commit, file hashes, tokenizer, dataset revisions,
prompts, and grading code.

The current queue runs Minerva Math, OlympiadBench, AMC22+23, AIME24, AIME25,
and AIME26. MATH500 is deferred. All five models and all five budgets for one
dataset must finish before the next dataset starts: 25 points per dataset,
150 scheduled points in total.

Preparation retains the original seven-dataset plan. A separate `schedule.json`
pins the prepared plan and selects an ordered subset for execution and reporting.
When changing a running queue, only the controllers use the separately frozen
controller runtime; active workers retain their original plans, code, and result
identities. The service and queue controllers can restart while those workers
continue. A deferred dataset is excluded from the completion audit and report.

Each point receives a shared output-token budget of
`dataset_size × {4096, 8192, 16384, 32768, 49152}` (4k, 8k, 16k, 32k, 48k).
Seeded shuffled sweeps visit
unsolved questions; a correct answer removes the question from later sweeps.
Generation stops when the shared budget is exhausted or all questions are solved.
The denominator includes every question, including any unvisited questions.
Every generated token counts, including EOS tokens and failed attempts; prompt
tokens do not count. Each response is capped at 32,768 tokens or the remaining
shared allowance. A question can spend more than the average budget.

Sampling uses seed 0, temperature 0.6, top-p 0.95, and top-k 20. Each budget uses
fresh generation. Math-Verify grades only the nonempty answer after completed
thinking, without an additional EOS or boxed-answer requirement. The grading
and model-specific tokenized prompts are inherited from the individual evaluation.

Eight GPUs run independent complete `(dataset, model, budget)` points, one GPU
per point. A single point's shared budget is never split across dataset shards.
When fewer than eight points remain in a dataset, the next dataset still waits.

## Compute-node execution and recovery

`compression_cross_budget.py prepare` freezes the follow-up after all five final
exports are verified. It stages no model weights. `launch` starts a persistent
service inside the existing compute allocation; the service waits for the original
queue, monitors disk space, and restarts a failed queue. Run these commands on
that compute node, using its training Python environment:

```bash
python -m qwen3_experiments.compression_cross_budget prepare \
  --after-evaluation "$ORIGINAL_EVAL_ROOT" \
  --exports "$FINAL_EXPORT_ROOT" \
  --output-root "$CROSS_EVAL_ROOT" \
  --scratch "$CROSS_SCRATCH"
python -m qwen3_experiments.compression_cross_budget launch \
  --output-root "$CROSS_EVAL_ROOT"
```

`CROSS_SCRATCH` must be a dedicated direct child of `/tmp` on the same node.
Model weights and download caches stay there. A model cache is deleted once all
five of its points for the current dataset are verified and its workers have
exited. Completed points are reused on restart; unfinished points are retried.
No cache belonging to another queue or compute node is deleted.

Durable results, including compressed full responses, token IDs, grading details,
and budget ledgers, stay in `CROSS_EVAL_ROOT`. Each completed point is independently
audited and hashed. If the results filesystem falls below 50 GiB free, the service
can move verified closed artifacts to its own compute scratch and retain readable
links; it restores them once free space exceeds 100 GiB. The service never removes
the only copy of a result.

The suite's `queue_status.json` shows its dependency or current dataset. Each
dataset has its own progress and report directory. The final `report/` contains
the full matrix as JSON, CSV, Markdown tables, and an audit.
