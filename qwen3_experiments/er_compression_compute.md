# Qwen3-1.7B ER on compression, after the evaluation queue

Start the persistent compute-node supervisor from the primary maxrl checkout:

```bash
bash qwen3_experiments/launch_er_compression_compute.sh
```

`--prepare-only` freezes and validates inputs without launching. The default
dependency is the five-model Minerva Math comparison in allocation 146103.
All 25 model-budget points must finish and pass their artifact audit before
this supervisor acquires the existing allocation locks and starts training.
Training, reward service, checkpoint monitoring and uploads run on the compute
node. Output plans and logs stay under the primary maxrl `outputs/` directory.

The training configuration follows `efficient-reasoning-official-hybrid/run_polaris.sh`. It uses
OpenRLHF v0.7.3 (`4a8683d3d5266494d20cea7f77abe044653639cb`) and its existing
constant-learning-rate scheduler adjustment. Its code is frozen into the run
directory. EOS processing follows the user's subsequent choice,
[`zjhhhh/er-r1-distill-1.5b-compression-n16-extracted-step_100`](https://huggingface.co/zjhhhh/er-r1-distill-1.5b-compression-n16-extracted-step_100/tree/8183d5b14fbbce488d3a2fd1891ed37af7135a84).
The original projects and Python environments are not modified.

| Setting | Value |
|---|---|
| Initial model | Original `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Dataset | All 3,200 rows of `zjhhhh/compression_dataset`, revision `bfdd7af1633ecc6db191a9f28f76449165a4ee06` |
| Prompt | Native Qwen3 thinking template; reference instruction suffix |
| Prompt / response cap | 1,536 / 32,768 tokens; longer prompt allowance preserves every compression row |
| Algorithm | RLOO; reference global advantage normalization |
| ER reward | Correctness × (1 − 0.1 × sigmoid((length − correct mean)/(correct population std + 1e−7))) |
| Reward grading | Full-query Math-Verify; DeepSeek checkpoint EOS policy described below; no after-thinking gate |
| Rollout / training batch | 32 prompts × 8 responses = 256; training batch 128, microbatch 1; two optimizer updates per rollout |
| Duration | One epoch: 100 rollout steps, 200 optimizer updates, 25,600 responses |
| Sampling / seed | Temperature 1, top-p 1, reference top-k default −1; seed 79 |
| Optimizer | Constant LR 1e−6, no warmup, KL 0, Adam betas 0.9/0.999, weight decay 0.01, grad clip 0.3 |
| Hardware | All eight GPUs; official Hybrid colocated vLLM/ZeRO-3 sleep/wake, BF16, vLLM memory 50% |
| Checkpoints | Every 20 rollouts: steps 20/40/60/80/100, plus final Hugging Face model |

## DeepSeek checkpoint EOS behavior

The checkpoint's pinned `train_config.json` selects unpacked rollouts with
microbatch size 1. Its `efficient-reasoning` trainer captures `all_responses`
before `Actor.process_sequences` forces EOS into the training sequence. This
produces two distinct reward inputs:

1. Correct-length statistics use raw generations, with natural EOS and correct
   full-text Math-Verify grading required for inclusion.
2. Individual correctness and ER reward use the force-EOS training sequence.
   The reference writes EOS after the last ordinary token, capped at the final
   existing position. With no EOS/padding slot, it replaces the last generated
   token; it never appends an extra token. This is applied independently to each
   response before the official Hybrid trainer pads the complete rollout batch.

The response's natural EOS is determined from its original generated token IDs.
Qwen3's prompt already contains `<|im_end|>`, which must not make an unfinished
generation eligible for the length pool. Mathematical verification still receives
the complete text, including thinking; it is not restricted to after `</think>`.
The EOS state is part of the correctness cache key so raw-pool eligibility and
individual correctness cannot overwrite one another. If no raw response qualifies
for the correct-length pool, the reference fallback uses the current response's
length, giving reward 0.95 when its forced-EOS text is correct.

The pinned checkpoint configuration and original EOS source files are preserved
under `runtime/eos_reference/`. The Math-Verify helper remains byte-identical to
the Polaris reference; the reward adapter changes only EOS routing. The selected
model, sampling settings, batch sizes, dataset, evaluation dependencies and
eight-GPU Hybrid training configuration remain as listed above.

## Complete rollout and checkpoint archival

A generation hook captures every original vLLM prompt/output token ID, finish
reason, original response text and row-level gold before modifying EOS. It also
saves `training_response_token_ids` and `force_eos_applied`. A reward hook records
the exact full query sent to the grader, binary accuracy, measured length and
shaped ER reward. Failed and unfinished responses are retained. HTTP retries do
not duplicate rollout records.

`raw_rollouts/` retains original generations. `rollout_dataset/data/` contains
one compressed JSONL shard per rollout step. Training finalization requires
all 100 shards, exactly 25,600 records and eight responses for every dataset row.
All shards are uploaded to the run's public HF dataset and checked against
remote content hashes. Raw local files remain available.

The reference checkpoint observer verifies all eight ZeRO-3 rank states,
checkpoint completion and optimizer step counters. Each checkpoint and the
final model use separate public HF repositories, as requested for all training
archives. Local checkpoints are removed only after
remote size and content-hash verification. Checkpoint files use node-local
scratch; receipts, frozen provenance and logs persist in the primary checkout.

## Automatic evaluation after ER

`er_compression_evaluation.py` runs a persistent follow-up queue on the same compute
node. It waits for successful 100-step training, all checkpoint receipts and the
verified upload of all 25,600 training responses. The final native Hugging Face
export is downloaded at its immutable, verified archive revision.

First, the queue evaluates exactly the same 1,819 questions across nine datasets
as compression L+0: four samples, seed 42, temperature 0.6, top-p 0.95, top-k 20,
and a 32,768-token response cap. It saves all 7,276 responses and grades their
exact 1k/2k/4k/8k/16k/32k prefixes. Both main-table metrics and budget-table metrics
use only the answer after thinking; the Qwen3 reference uses its corrected
after-thinking scores. Historical Polaris-trained ER/MaxRL rows are excluded.

Then Minerva Math runs individual budgets 8k/16k/32k/48k/64k with seed 0, the same
sampling settings and a 32k cap per attempt. Each question stops at its first
correct answer or when its own budget runs out. Eight independent GPU workers
process 34 questions each, preserving the original dataset position in the
per-question random seeds. Merged response ledgers are audited again across all
272 questions before publishing the five results. Neither evaluation requires
an extra EOS or boxed answer; unfinished/reopened thinking scores zero.

This evaluation policy does not change ER's full-text training reward. The
follow-up queue takes the existing GPU locks and waits for training to release
the GPUs. It neither edits the frozen training runtime nor restarts training.

Run preparation and launch **on the allocated compute node**, using the `maxrl`
environment (the evaluation backend):

```bash
python -m qwen3_experiments.er_compression_evaluation prepare \
  --training-root "$ER_TRAINING_ROOT" --output-root "$ER_EVALUATION_ROOT"
python -m qwen3_experiments.er_compression_evaluation launch \
  --output-root "$ER_EVALUATION_ROOT"
```

The follow-up directory contains `queue_status.json`, `nine_datasets/`, and
`minerva_individual_budget_seed0/`. All evaluation responses and token IDs are
retained there. Failed workers retry up to three times, reusing completed
responses or completed question shards without overwriting the training run.
