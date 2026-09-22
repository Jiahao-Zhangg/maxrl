# Full TextMaze per-context RB results — 2026-09-16

This is an aggregate snapshot of **four completed 5001-step runs**, not the earlier
100-step reduced-data pilot. All four use binary goal reward, cost `max(L, L*)`,
global token-mean aggregation, 256 prompts × 16 rollouts per update, and four
GH200 GPUs. SFT step identifies the released initialization, not the RL step.

| SFT initialization | Seed | N | RL updates | Initial goal successes / 64,000 | Final goal successes / 64,000 | Final goal rate | Final shortest-path rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2450 | 0 | 16 | 5001 | 4 | 0 | 0% | 0% |
| 2450 | 1 | 16 | 5001 | 7 | 0 | 0% | 0% |
| 3000 | 0 | 16 | 5001 | 9 | 52,446 | 81.946875% | 81.946875% |
| 3250 | 0 | 16 | 5001 | 32 | 56,273 | 87.9265625% | 87.9265625% |

Every validation point samples 64 trajectories for each of 1000 held-out mazes.
The rates above are **per-trajectory mean accuracy**, not pass@64 or the fraction
of mazes solved at least once. Final means are from step **5001**, not the best
observed checkpoint. These are individual runs, not means across seeds.

## Published data

- [Per-run summary](run_summary.csv): completion, training success totals, and
  initial/final held-out metrics.
- [Validation history](validation_metrics.csv): all 28 recorded validation points
  at steps 0, 1000, 2000, 3000, 4000, 5000, and 5001 for the four runs.
- [Provenance](provenance.json): pinned inputs, estimator-port metadata, source
  code hashes, and checksums of the original metrics/completion files.
- [Training scripts and full configuration](../../../tailrl_experiments/README.full.aarch64.md).

The tables were derived from local `metrics.jsonl` and matching `COMPLETE.json`
records. Each run has all 5001 distinct training-step records with no duplicates,
and its final checkpoint pointer is 5001 with an HF-format export present.
The recorded validation rates imply integer event counts at 64,000 samples.
This publication verifies those records and their checksums; it does **not**
independently regrade all saved validation trajectories or rehash model weights.

N=16/seed=2 and N=32/seed=0 were supported/planned variants but are not included
as completed runs. No unexecuted setting is represented by a zero score. The
snapshot contains no TailRL/GRPO/RLOO comparison run and does not establish an
advantage over those algorithms or identify a causal explanation for collapse.

All-failure groups have zero advantages, but the inherited optimizer is still
stepped. Momentum and weight decay mean zero advantage is not necessarily a
frozen policy. The sparse SFT-2450 runs sampled only 2 (seed 0) and 8 (seed 1)
successful training trajectories across 20,484,096 generated trajectories per run.

Raw maze data, prompts, trajectories, checkpoints, local paths, credentials,
operational queue state, and machine-specific environment artifacts are excluded.
Checkpoint files remain local; uploading this snapshot does not upload them to
GitHub or Hugging Face and does not restart any run.
