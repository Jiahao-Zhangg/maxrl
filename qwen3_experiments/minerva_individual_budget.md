# Minerva Math individual-budget comparison

This follow-up waits for the compression L+0 run's nine-dataset final evaluation
to finish successfully and pass its complete-result audit. Its persistent queue
and all evaluation processes stay on allocation 146103's compute node. Launch
once from the primary maxrl checkout:

```bash
bash qwen3_experiments/launch_minerva_individual_budget.sh
```

`--prepare-only` freezes and checks CPU inputs without launching. The five models are:

| Model | Pinned revision / identity |
|---|---|
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Compression L+0 step 100 | The checkpoint evaluated by the preceding nine-dataset evaluation |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| `zjhhhh/er_cost_marginrl_r1_distill_1.5b_compression_n16_b512_32k_lr1e-6_kl0_seed42-step_100` | `595b113264edb620d1a091cfc29b31984584fc6d` |
| `zjhhhh/er-r1-distill-1.5b-compression-n16-extracted-step_100` | `8183d5b14fbbce488d3a2fd1891ed37af7135a84` |

For a follow-up that evaluates only the parent run's final checkpoint, set
`MINERVA_CHECKPOINT_ONLY=true` when using the launcher, or pass
`--checkpoint-only` to the Python `prepare`/`launch` command. Set
`MINERVA_PARENT_RUN` to that run's directory. This creates five model-budget
points without preparing the comparison models; its label and final checkpoint
identity come from the preceding nine-dataset plan. The default output suffix
becomes `minerva_individual_budget_seed0_checkpoint` instead of `five_models`.
The existing five-model mode remains the default.

The same evaluator can run several datasets with `--datasets`, or with the
space-separated `MINERVA_DATASETS` launcher variable. For the L+4096 follow-up,
the selected datasets are `minervamath math500 olympiadbench amc22_23 aime24
aime25 aime26`: 1,619 questions and 35 dataset-model-budget points for the final
checkpoint. Polaris is not part of this individual-budget selection. Each
dataset retains its original question order; request seeds are derived from the
position within that dataset, so expanding the selection preserves Minerva's
existing seeds. New results use
`results/<dataset>/<model>/budget_<tokens>/`, and reports use each dataset's own
question count as the denominator.

To expand a waiting queue, prepare a distinct output directory with the expanded
`MINERVA_DATASETS` and run the launcher with `--replace-waiting` pointing to the
old directory. Replacement verifies that all original datasets, revisions,
questions, native prompt IDs, model identities, sampling and grading rules are
preserved. It is refused once the original queue has begun evaluation. The old
plan and provenance remain available with a forwarding receipt.

The checkpoint revision and merged-model hashes are pinned when that evaluation
finishes. The Qwen3 model is the same thinking model used to initialize training.
DeepSeek ER cost's four FSDP actor shards are merged on CPU; ER extracted uses
the full BF16 actor weights in its ZeRO-2 checkpoint. Optimizer states are not
needed. Weight hashes, tensor inventories, tokenizer vocabulary and native
thinking templates are checked before the expanded queue launches.

| Parameter | Value |
|---|---|
| Dataset | All 272 frozen Minerva Math questions from the nine-dataset evaluation |
| Individual cumulative budgets | 8,192 / 16,384 / 32,768 / 49,152 / 65,536 output tokens |
| Per-response cap | min(32,768, the question's remaining budget) |
| Repeated experiment seeds | One: 0 |
| Temperature / top-p / top-k | 0.6 / 0.95 / 20 |
| min-p / presence / frequency / repetition penalty | 0 / 0 / 0 / 1 |
| Prompt | Identical user messages; each model's native thinking template and frozen prompt token IDs; no system message |
| Grader | Same after-thinking Math-Verify rule and one-second deadline as the nine-dataset evaluation |
| GPU setup | 8 GPUs; one independent model-budget point per GPU, largest budgets first; BF16 |

Each question receives its own budget. Failed responses, thinking, answers, and
generated EOS tokens count; prompt tokens do not. Attempts stop at the first
correct answer or budget exhaustion. Unused tokens are never reassigned to
another question. A question passes if any generated attempt within its budget
is correct; the report is the percentage of the 272 questions that pass.

All five budget points perform fresh requests. Every attempt has a deterministic
seed derived from `(seed, question, attempt)`, shared across all five models and
budget points. Sampling count varies by question. The grader uses the nonempty
suffix after the last completed `</think>`; unfinished/reopened thinking and
empty suffixes score zero. There is no extra EOS or boxed requirement. Grader
errors fail the evaluation attempt instead of silently entering the final score.

The queue uses the existing allocation locks and starts only once all eight GPUs
are free. It preserves complete points and their checksums when retrying a failed
run, retaining partial raw attempts separately. It does not modify or restart the
training or nine-dataset controllers. To expand an older queue that has never
started generation, prepare this five-model plan in a distinct output directory
and launch with `--replace-waiting /absolute/path/to/old_minerva_output`. This
validates unchanged original settings and models, pauses and rechecks the old
waiting controller, launches the new one, then retires the old controller with a
forwarding receipt. Original plans and provenance are retained. Replacement is
rejected once any Minerva execution has begun.

The default output directory is
`outputs/per_context_rb_l0_0_qwen3_1_7b_compression_bs32_32k_146103/minerva_individual_budget_seed0_five_models/`:

- `plan.json`, `input_audit.json`: frozen questions and requested settings.
- `prompts/`, `prepared_models/`: DeepSeek native prompts, conversion receipts and source hashes.
- `queue_status.json`, `queue.log`, `launch.log`: dependency and execution status.
- `execution_manifest.json`: exact model revisions, staged files and checksums.
- `results/<model>/budget_<tokens>/`: raw responses, token IDs, per-question budgets and summary.
- `report/metrics.csv`, `report/README.md`, `report/audit.json`: all 25 comparison points and their audit.
