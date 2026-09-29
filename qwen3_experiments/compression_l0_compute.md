# Qwen3 compression L+0 on allocation 146103

Work from the primary `maxrl` checkout on `agent/add-math12k-maxrl-launcher`.
With the maxrl Python environment activated, launch once from the login node:

```bash
bash qwen3_experiments/launch_compression_l0_compute.sh
```

The launcher resolves the compute node from Slurm and makes one SSH call.
Preparation, the detached supervisor, training launcher, checkpoint monitor,
evaluation queue, and Ray workers run on that compute node. Closing the login
session does not stop them. They use the existing allocation; the launcher does
not create another allocation, branch, or worktree.

The supervisor waits for the previous L+0 recovery pipeline to complete
successfully, including the verified step-100 checkpoint archive. It then
acquires the existing allocation locks and checks that all eight GPUs are free.
A failed predecessor blocks the new training run.

The new run uses Qwen3-1.7B at revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` and all 3,200 rows of
`zjhhhh/compression_dataset` at revision
`bfdd7af1633ecc6db191a9f28f76449165a4ee06`. Training uses per-context RB L+0,
32 prompts × 16 samples, 100 steps, and a 32,768-token response budget. The
prompt cap is 1,536 so all compression questions fit. Reward requires EOS and
grades the nonempty answer after the last completed thinking section, matching
the current GRPO launcher. EOS is not forced.

`run_qwen3_1_7b_compression_per_context_rb_l0_0.sh` directly enables
`trainer.rollout_dataset`: every completed step saves all 512 rollouts atomically
under `rollout_dataset/data/step_XXXXXX.jsonl.gz`. Records include the full input
and output text, reward, prompt identity, token counts, advantage, and return.
The trainer uploads the complete dataset to HF before successful exit. The
supervisor additionally checks all 100 shards, 51,200 rows, and remote hashes.
Local rollout shards remain available. The HF dataset is public:

```text
hi-todayis-jh/per-context-rb-l0-0-qwen3-1.7b-compression-bs32-32k-146103-rollouts
```

Checkpoints are saved every 10 steps. Complete checkpoints are uploaded to
public HF model repositories with suffix `-step_N`; local checkpoint shards
are removed only after remote hash verification and a local consistency check.
The final checkpoint remains local until the training process exits.

After training and all archives are verified, the evaluation queue evaluates
step 100 on the same frozen nine benchmarks as the previous L+0 evaluation:
1,819 questions, four responses each, temperature 0.6, top-p 0.95, top-k 20,
seed 42, and 32k generation. It regrades the same token sequences at 1k, 2k,
4k, 8k, 16k, and 32k. Evaluation uses the established after-thinking rule
without an additional EOS requirement. Evaluation only acquires GPUs after
training releases the allocation locks.

The comparison report and both comparison CSVs include only original
Qwen3-1.7B and `L+0 step 100 (compression)`. The historical Qwen3 sampling and
current L+0 sampling settings are identified in the report. Both main-table
accuracy metrics and every budget column use after-thinking grading. Qwen3's
audited reference is stored under `evaluation/report/qwen3_after_thinking/`;
its main table uses the reference's 32k mean@4, pass@4 and response length.
Raw L+0 metrics and the frozen historical tables retain their original provenance.

The run directory is
`outputs/per_context_rb_l0_0_qwen3_1_7b_compression_bs32_32k_146103/`:

- `plan.json`, `resolved_config.yaml`, `runtime/`: pinned inputs and a code snapshot inside the primary repository.
- `supervisor_status.json`, `status.json`, `train.log`: overall control and training progress.
- `hf_checkpoint_archive/status.json`, `hf_checkpoint_archive/receipts/`: checkpoint uploads and verification receipts.
- `rollout_dataset/`, `rollout_upload.json`: saved rollouts and final upload audit.
- `evaluation/queue_status.json`, `evaluation/report/`: queued evaluation and results.

Use `--prepare-only` to prepare and verify inputs and HF write access without
starting the supervisor. It creates an empty public rollout repository.
Environment overrides are documented at the top of the launch scripts. A
second launch into the same run directory is rejected to prevent duplicate
training. Preparation and training errors are recorded locally; no failure
silently starts a replacement training run.
