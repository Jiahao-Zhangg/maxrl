# TACO Easy GRPO and final checkpoint evaluation

The prepared queue waits for the entire existing 146102 compression-budget
follow-up queue: all 90 new points and the combined 105-point audit. It then runs
TACO GRPO, LiveCodeBench `v6`, and TACO test without official SPJ problems, in that
order. Its supervisor, training launcher, checkpoint monitor and GPU workers all
run on the assigned compute node. The existing allocation locks prevent overlap.
Before training starts, a required cleanup step verifies the completed evaluation
audit and the new training's separate base-model copy. It removes the preceding
ER, L+0 and Qwen3 evaluation caches, including downloaded checkpoint shards and
private Hub/Xet download caches. Active files are retained, cleanup receipts are
mirrored onto two filesystems, and interrupted deletion is retried. Training
cannot start until every listed evaluation cache is absent and the completion
receipt has been saved. Published checkpoints and evaluation results are retained.

Training uses the public `hi-todayis-jh/TACO-easy-subset` training split at commit
`7ef9b8ac1260cefe2c03ee054f1a44d13e37c6a5`: all 3,200 Easy questions, without a
training SPJ filter. The runtime and resolved configuration derive from the
completed Polaris GRPO run. The current replacement run starts from Qwen3-1.7B
base weights after cancellation of the earlier N=16/concurrency-32 run. Batch
size is 32, with 8 responses per prompt (256 rollouts per training step),
100 steps / one epoch, 32,768 response tokens, learning rate 1e-6, zero KL and
group standard-deviation normalization. Both training vLLM concurrency settings
are 16. The replacement has separate checkpoint, rollout, log and public Hub
destinations; it does not resume the cancelled run's checkpoints.
Training prompts are produced by directly calling the unmodified
[`pretokenizing.py:initialize()` from FlagOpen/TACO](https://github.com/FlagOpen/TACO/blob/245eba3beb2d23a07082307de303de2589e4321a/pretokenizing.py#L54).
The official question, starter-code and input-format text is used verbatim as
the user message, inside Qwen3's native thinking chat template. No custom task
instructions are added, and reference solutions never enter the prompt. The
same official content template is used for the final TACO test evaluation.
The prompt cap increases from 1,280 to 1,792 because the longest official prompt
is 1,606 tokens; the actual training dataset loader retains all 3,200 rows.

For runs initially configured with concurrency 32, if an unsuccessful training
attempt reports CUDA out-of-memory, the next attempt
uses concurrency 16 in both vLLM configuration fields and keeps that setting for
later restarts. The rollout group size is unchanged by that fallback. The current
replacement already starts at concurrency 16 and N=8. Recovery waits for complete
checkpoint uploads, restores the most recent archived optimizer/dataloader state,
and starts from scratch only if no complete checkpoint exists. Cleanup of failed
Ray workers is restricted to that run's recorded Slurm step. The fallback decision
and effective concurrency are saved with the training attempt records.

Training sets `check_eos=False`. `TacoRewardManager` requires a generated
`</think>` and a nonempty final answer. Only code after the last `</think>` is
graded, including responses without EOS. Reward is one only when every official TACO test passes; error and
timeout codes count as failure. Grading runs in an isolated bubblewrap namespace
with no network, GPU devices or credentials. The pinned official graders are
copied without modifications.
The sandbox wrapper captures both Python output and operating-system stdout/stderr
descriptors. TACO's file-I/O retry can launch a child with inherited output streams;
those prints must stay separate from the JSON result. Regression checks cover noisy
failed programs and successful file-I/O retries without changing official scores.

Every ten steps, the complete eight-rank checkpoint is uploaded to a public
repository. The monitor polls save completion every two seconds, verifies remote
file sizes and hashes, retains a durable receipt, and then deletes the local
checkpoint, including step 100. Failed uploads retain local files for retry.
Training recovery downloads the last verified checkpoint, including optimizer
and dataloader state. Final evaluation downloads the pinned step-100 actor,
merges it, verifies publication of a public native Hugging Face model, then runs:

| Evaluation | Questions | Grader |
|---|---:|---|
| LiveCodeBench `v6` (sixth batch; `test6.jsonl`) | 175 | Official LiveCodeBench |
| TACO test, all difficulties, excluding 218 official SPJ problems | 782 | Official TACO |

Both evaluations use thinking mode, 32,768 output tokens, temperature/top-p/top-k
0.6/0.95/20, one sample per question (pass@1), concurrency 32, and only the answer
after thinking for grading. They impose no additional EOS requirement. The
sampling seed is the original source-row index. Results include overall and
per-difficulty pass@1, output lengths, raw responses, token IDs, per-test outcomes
and a final audit. `v6` here is not the cumulative `release_v6` benchmark.

`prepare_taco_grpo.py` prepares pinned inputs. The prepared `plan.json` records
all runtime and input hashes, the predecessor hash, model identity, grading and
sampling settings. `run_qwen3_1_7b_taco_grpo.sh` consumes the prepared configuration;
`taco_grpo_pipeline.py queue` supervises the full sequence. Control records are
mirrored onto node-local storage; disk-full errors cannot terminate the queue.
Checkpoints, model caches, training rollouts and new evaluation payloads use
compute-local storage. Small control records and result summaries remain
accessible through the home run directory.

The persistent compute supervisor also probes write access every 30 seconds to
detect both ENOSPC and user-quota EDQUOT. Emergency reserve files provide room
for recovery records. Under pressure, audited or closed evaluation ledgers can
be copied to compute-local storage, checked byte-for-byte, and replaced by links
at their original paths. Current writers are retained. The supervisor restarts
missing evaluation/GRPO controllers and keeps operating when status mirrors are
full. Training logs, W&B files and new evaluation payloads use compute-local
storage so shared/home filesystem pressure cannot consume their output space.

`verified_rollout_cleanup.py` runs separately on the compute nodes. After a run
succeeds and its public rollout dataset is available, it verifies each closed
shard against an immutable Hub commit before removing local shards. Dataset
manifests and checksum receipts are retained. This is enabled for the new TACO
run and the already-running 146103 MaxRL run.

Validation includes CPU tests with EOS checking enabled and disabled, after-thinking extraction,
predecessor gating, independent disk-failure handling, and retention of rollout
files on remote hash mismatch. Compute-node preflight additionally resolves the
actual training configuration, checks all 3,200 actual model input token sequences
against the official formatter and Qwen3 thinking template, exercises both official
graders on standard-input and function-call tasks, and verifies timeout handling.
No GPU training or new evaluation is started during preflight.
