# Full coding GRPO and MaxRL on veRL 0.9.1

New coding training and evaluation use `maxrl-code-verl091`: veRL 0.9.1,
vLLM 0.24.0, PyTorch 2.11/CUDA 13, and Python 3.12. The environment currently
lives on the assigned compute node; preserve the installation recipe in
`verl091_upgrade.md` when an allocation is replaced. Earlier environments are
not upgraded in place.

`grpo_coding_release.py prepare` consumes the recorded benchmark inputs and the
two previously prepared holdout runs. It freezes the current runtime, copies the
original Qwen3-1.7B initialization, verifies the complete 3,200-row training
parquet, and resolves the full training configuration. Training starts from the
base model, not the one-step benchmark weights. No benchmark replay subset is
used in this run.

The training dataset is the public
`hi-todayis-jh/Nemotron-LeetCode-coding-clean-3.2k` release
`aeef423c0faef01e2fdac84628254eaa5327683f`, with its pinned LCB-format tests.
Its 3,200 rows are shuffled with seed 42. GRPO uses batch 32, eight responses per
prompt, one epoch / 100 updates, learning rate 1e-6, zero KL and entropy
coefficients, and group standard-deviation normalization. Thinking is enabled;
training sampling is temperature/top-p/top-k 1/1/-1, with 32,768 output tokens.

The successful benchmark's inference configuration is retained: eight H100
80GB GPUs, TP=1, BF16 weights/KV, concurrency 16 per GPU, 0.7 vLLM memory budget,
8,192 prefill tokens, MRV2, FlashAttention 3, FULL_AND_PIECEWISE CUDA Graph,
asynchronous scheduling, chunked prefill, and prefix caching. This still uses
synchronous GRPO parameter updates; reward execution overlaps generation.

## Training binning

The opt-in `trainer.log_training_binning` flag restores the metric names used
by `grpo_Qwen3-1.7B_Polaris-1-8-3200_bs32_n16_32k_1epoch`:

- `train_all_datasets_binning/fraction_of_prompts_in_<interval>`
- `train_binning_for_dataset_<data_source>/fraction_of_prompts_in_<interval>`

Each prompt contributes one mean accuracy over its N binary rewards.
The original 13 intervals are preserved: exact zero, powers-of-two boundaries
from 1/1024 through 1/2, the open interval (0.5, 1.0), and exact one. Empty
intervals are logged as zero. Prompt identity survives batch reordering, padding
is excluded, and an incomplete group is rejected. All metrics go to console and
the same W&B run as the training losses and timing metrics.

## Unified grading and final evaluations

Training and evaluation call the same `LiveCodeBenchGrader`, using the unchanged
official `testing_util.py` at revision
`28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`. Both use the same Python environment
and credential-free bubblewrap sandbox, ten seconds per test, binary reward,
first-failure early stop, code after the last closing thinking tag, and no EOS
requirement. A missing closing thinking tag receives zero. Infrastructure
failures are retried once and otherwise raised, rather than recorded as wrong
model answers.

After the step-100 model has been merged and publicly uploaded, the queue runs:

| Holdout | Questions |
|---|---:|
| LiveCodeBench v6, sixth batch | 175 |
| TACO test excluding the pinned official SPJ list | 782 |
| USACOBench | 307 |
| CodeContests test | 165 |

Evaluation retains the earlier question populations and input token IDs. Every
source test is retained, including public/private/generated CodeContests tests.
TACO call argument lists become newline-separated JSON arguments, and its
singleton expected-result wrapper is removed exactly once; stdin line lists
become newline-separated text. USACO test files are converted to stdin/stdout
pairs. Format mismatches fail preparation instead of dropping a test or question.

All four evaluations now use the LCB comparison and timeout rules. Previous
TACO, USACO, and CodeContests results used their respective official graders,
so their historical numbers do not have exactly the same grading protocol.
Evaluation uses vLLM 0.24.0, thinking on, 32k output tokens, 0.6/0.95/20, one
sample per question (pass@1), original source-row-index seeds, and concurrency
16 per GPU. Grading overlaps later generation batches. Responses and individual
grades are resumable and checked against the model revision and frozen plan.

## Supervision and storage

The login node only starts the persistent supervisor on the compute node.
The supervisor restarts the queue and upload monitors if they exit. GPU stages
hold the allocation's existing locks and refuse to overlap another GPU process.
The other allocation is not used.

Every ten steps, the checkpoint monitor requires all eight model/optimizer/RNG
shards, the tokenizer/config, dataloader state, and the trainer's completion
marker. It polls every two seconds, uploads to a public Hub repository, verifies
every remote file size and hash, writes a durable receipt, and deletes the local
checkpoint. A failed upload retains local data. Training recovery restores the
latest verified model, optimizer, RNG, and dataloader state.

Completed rollout JSONL files (256 rows for GRPO N=8; 512 for MaxRL N=16) are compressed, publicly uploaded, hash
verified, and deleted locally. Immutable commit IDs and original-file hashes
remain in receipts. Failed-attempt rollouts at a repeated step can be superseded
on the dataset's current branch, with earlier versions retained in Hub history.

Large files, model caches, Ray, W&B files and logs remain on compute-local disk.
Small control records are mirrored to persistent storage, with emergency reserve
space. A full status filesystem does not terminate a controller; the second
copy remains usable. Cleanup is limited to this run's verified uploaded files.

## Launch

Run preparation and supervision on the assigned compute node, using explicit
paths for the existing benchmark and two holdout runs:

```bash
"$MAXRL_CODE_PYTHON" -m qwen3_experiments.grpo_coding_release prepare \
  --root "$GRPO_RUN_DIR" --scratch "$GRPO_SCRATCH" --job-id "$GRPO_JOB_ID" \
  --benchmark-plan "$GRPO_BENCHMARK_PLAN" \
  --baseline-root "$CODING_BASELINE_ROOT" --competition-root "$COMPETITION_EVAL_ROOT" \
  --hf-prefix "$GRPO_HF_PREFIX" --experiment-name "$GRPO_EXPERIMENT"

PYTHONPATH="$GRPO_SCRATCH/runtime" "$MAXRL_CODE_PYTHON" \
  -m qwen3_experiments.grpo_coding_release supervise --root "$GRPO_RUN_DIR"
```

The supervisor must be detached from the login connection by the compute-node
launcher. `HF_TOKEN_PATH` should point to the existing authorized credential
file during preparation; only that path, not its secret contents, is recorded.

For original MaxRL, add `--algorithm maxrl --n 16` at preparation and choose a
separate root, scratch directory, experiment name, and public HF prefix. Add
`--after-root "$GRPO_RUN_DIR"` to queue it after the entire GRPO pipeline.
Its supervisor can start immediately: it waits for the pinned predecessor's
training, all four evaluation audits, and rollout archives before launching any
GPU work. It then removes the predecessor's verified uploaded final model and
remaining checkpoint copies under the allocation GPU lock. Evaluation responses,
metrics, upload receipts, and other runs remain available. Interrupted cleanup
can resume after verifying every remaining file against its uploaded hash.

See `maxrl_verl091.md` for the migrated advantage estimators and CPU checks.

## Continue GRPO for another epoch

Prepare a separate run with `--after-root "$GRPO_RUN_DIR" --continue-from-predecessor`
and its own root, scratch directory, experiment name, and public HF prefix.
The predecessor's training, four holdouts, and rollout uploads finish first.
Continuation requires the same dataset, initialization, grading rules, N, batch
size, sampling settings, and shuffle seed, plus a complete predecessor epoch.

The second run restores the verified full step-100 checkpoint into its own
scratch directory, including optimizer, scheduler/RNG, and dataloader state.
It runs steps 101–200 with `total_epochs=2` and `total_training_steps=200`.
The V1 loader retains sampler state and moves into a newly shuffled second
epoch; it does not restart epoch one's row order. The constant learning-rate
schedule remains unchanged. A separate W&B run records the continued global
step numbers and the same training accuracy bins.

Only this stage's checkpoints (110, 120, …, 200) and rollout shards (101–200)
are counted toward completion and uploaded to its own public repositories.
Recovery prefers the latest checkpoint from this second stage; a missing
predecessor checkpoint raises an error instead of starting from base weights.
The final model and all four holdouts use step 200.
