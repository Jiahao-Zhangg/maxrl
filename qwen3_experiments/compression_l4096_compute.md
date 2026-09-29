# Compression L+4096 after GRPO final evaluation

Run from the primary maxrl checkout with the maxrl Python environment:

```bash
bash qwen3_experiments/launch_compression_l4096_compute.sh
```

The default allocation is 146102, on orchard-flame-23. The login node makes one
SSH call. Input preparation, the detached supervisor, checkpoint monitor,
training launcher and Ray all run on the compute node. `--prepare-only`
freezes and validates the inputs without starting the supervisor.

The supervisor waits for `outputs/grpo_final_eval_146102/` to finish successfully.
It checks the final checkpoint receipt, successful evaluator exit, all 7,276
graded responses, six budgets across nine datasets, and report checksums.
Only then does it acquire the allocation's existing locks, verify that all eight
GPUs are idle, and launch the new training. An incomplete or failed evaluation
does not release this training run.

The launcher `run_qwen3_1_7b_compression_per_context_rb_l0_4096.sh` reuses
`run_qwen3_1_7b_compression_per_context_rb_l0_0.sh`. It changes:

- `algorithm.cost_offset_tokens=4096`, giving full-response cost `L + 4096`.
- `reward_model.reward_kwargs.check_eos=false`.
- The experiment name and output destinations.

`score_after_thinking=true` remains enabled. Only the nonempty answer after the
last completed `</think>` is graded. Missing or reopened unfinished thinking and
empty answers score zero. A completed answer can receive credit even when the
output reaches the token limit without EOS. EOS stopping remains enabled and
EOS is not appended to truncated outputs.

The run starts from the original Qwen3-1.7B revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. It uses the same pinned 3,200-row
compression dataset, 32 prompts × 16 rollouts, 1e-6 learning rate, KL=0,
1,536-token prompt cap, 32,768-token response cap and 100 training steps.
Preparation compares the resolved configuration against L+0 and rejects any
other training-setting change.

Every ten steps, the checkpoint monitor uploads a complete checkpoint to a
public Hugging Face repository. Local checkpoints are deleted only after remote
hash verification and a local consistency check. All 51,200 training rollouts
are saved and uploaded to a public dataset; their local copies are retained.
The training launcher itself queues training and archival. The separately
attached final-evaluation queues described below wait for this same run without
changing its frozen training plan or restarting its supervisor.

The default run directory is
`outputs/per_context_rb_l0_4096_no_eos_qwen3_1_7b_compression_bs32_32k_146102/`.
Its `supervisor_status.json` shows the queue/training state; `plan.json`,
`resolved_config.yaml`, `config_comparison.json` and `runtime/` preserve the
configuration and inputs. Checkpoint verification receipts are stored in
`hf_checkpoint_archive/receipts/`.

## Final evaluation follow-ups

The run now has two persistent follow-ups on allocation 146102's compute node:

1. `evaluation/`: `eval_l0_final.py` waits for successful step 100 completion and
   the verified final checkpoint upload. It evaluates the same nine datasets,
   1,819 questions and four samples per question (7,276 responses), with thinking
   on, 32,768 output tokens, temperature 0.6, top-p 0.95, top-k 20 and seed 42.
   Reports include mean@4, pass@4, mean response length and the six exact-prefix
   budget tables (1k, 2k, 4k, 8k, 16k, 32k). The Qwen3 comparison uses the frozen,
   audited after-thinking reference and records its different sampling settings.
2. `individual_budget_seed0_seven_datasets/`: after the nine-dataset result
   passes its full audit, `minerva_individual_budget.py --checkpoint-only` runs
   the same final checkpoint on Minerva Math (272), MATH500 (500), OlympiadBench
   (674), AMC22+23 (83), and AIME24/25/26 (30 each): 1,619 questions in total.
   Each question receives
   a separate cumulative budget of 8,192 / 16,384 / 32,768 / 49,152 / 65,536
   output tokens. These 35 dataset-budget points use fresh sampling with seed 0, the same
   0.6 / 0.95 / 20 settings, a 32,768-token per-response cap and stopping at the
   first correct answer. Unused tokens stay with their question.

The expanded individual-budget queue supersedes the original waiting
`minerva_individual_budget_seed0_checkpoint/` queue. Its original Minerva inputs
and seeds are preserved. Dataset-budget points are dispatched across all eight
GPUs and reported separately with their own question counts.

Both grade only the nonempty suffix after the last completed `</think>` using
Math-Verify, with no extra EOS or boxed-answer requirement. Missing, unfinished,
reopened or empty thinking suffixes score zero. Both acquire the existing GPU
holder locks and wait for idle GPUs before generation.

Each directory preserves `plan.json`, input hashes, `queue_status.json` and
`queue.log`; finished results are written to `report/README.md`, metrics files
and `report/audit.json`. `post_training_evaluation.json` in the training root
links the two queues and their frozen plans. The training plan's original
`evaluate_after_training=false` remains unchanged because these follow-ups were
attached after training started.

## Recovering an orphaned controller

If the supervisor and checkpoint monitor have exited while the original training
step is still running, `compression_supervisor_recovery.py` can adopt that step:

```bash
# Run on the allocation's compute node, using the original Slurm step ID.
python qwen3_experiments/compression_supervisor_recovery.py launch \
  --training-root outputs/per_context_rb_l0_4096_no_eos_qwen3_1_7b_compression_bs32_32k_146102 \
  --step-id 64
```

Recovery checks the frozen runtime and inputs, original launcher command and PID
identity, allocation membership and matching live trainer. It restores checkpoint
uploads and status updates without restarting training. Evaluation is released
only after the original Slurm step reports `COMPLETED` with exit `0:0`, the final
checkpoint is complete, all 51,200 rollout uploads are verified, and all ten
checkpoint archives are verified. Recovery records are kept in `control_recovery/`;
local controller logs are under the checkpoint directory's sibling
`control_recovery/`. Status writes retain a local copy and retry shared-filesystem
quota errors.
