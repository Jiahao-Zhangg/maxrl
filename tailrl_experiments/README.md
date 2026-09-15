# TailRL text-maze pilot with per-context RB

This experiment uses TailRL's released 17x17 mazes, tiny Qwen2 model, SFT
checkpoints, binary goal verifier, and vendored verl trainer. It ports the exact
function body of MaxRL's `compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage`
into a separate TailRL checkout. The existing MaxRL trainer is not edited.

## Experiment definition

For each context (one maze), generate a fixed group of N trajectories. Let
`r_i` be 1 for a valid DONE-terminated output that reaches the goal, otherwise 0.
As in the original verifier, a collision fails immediately and reaching the goal
ends simulation. Every token before DONE must be a valid direction.

`L_i` counts all generated items before DONE (or the whole decoded output if DONE
is missing). Valid items are movement actions; invalid non-special items also
consume length. DONE, EOS, and padding do not count as moves. This counts generated
actions after an early goal visit or collision as well. `L*` is computed by BFS on
the maze and checked against the dataset's shortest-path metadata, including for
failed or malformed responses.

For each maze independently:

```text
c_i = max(L_i, L*)
M = sum_i r_i
q_hat = M / sum_i c_i                       # detached, same-context estimate
A_i = N * (1 - q_hat*c_i) / M               # success
A_i = -N * q_hat*c_i / (M + 1)              # failure
```

If M=0, all advantages are zero. There is no extra centering, standard-deviation
normalization, or cross-context estimate of `q_hat`. Actor aggregation is
`token-mean`, as in the existing MaxRL launchers. The reward stays binary; cost
enters the estimator separately. Shortest-path success is an evaluation metric,
using TailRL's number of moves to the first goal visit.

## Pilot versus the paper

| Setting | TailRL paper sweep | This pilot |
|---|---|---|
| Estimators | TailRL, GRPO, RLOO, PKPO | Per-context fixed-N RB only |
| Reward | Continuous composite_v2 | Binary goal success |
| Cost | Encoded in continuous reward | `max(L, L*)`, separate from reward |
| SFT initializations | 2450, 3000, 3250, 3350, 3400, 3450, 3550 | 2450, 3350, 3550 |
| Seeds | 0, 1, 2 | 0 |
| Prompt batch | 256 | 64 |
| Rollouts per prompt | 16 | 16 |
| Training-step setting | 5001 | 100 |
| Training data | Approximately 1.3M mazes | 8192 distinct mazes from a streamed prefix |
| Eval contexts | Released eval1000 | Fixed random 256-context subset, seed 0 |
| Eval rollouts per context | 64 during training, 4096 final ladder | 128 at steps 0 and 100 |
| Learning rate | 1e-4 | 1e-4 |
| Max response length | 180 | 180 |
| Sampling | Temperature 1, top-p 1, unrestricted top-k | Same |
| Stop tokens | EOS and DONE (id 7) | Same |
| KL / entropy coefficient | 0 / 0 | 0 / 0 |
| Actor loss aggregation | token-mean | token-mean |
| PPO microbatch | Up to 4096 trajectories | 64 trajectories |
| Attention implementation | FlashAttention 2 | PyTorch SDPA |
| Gradient checkpointing / torch.compile | Enabled defaults | Disabled for the small pilot |
| GPUs | One per arm | GPU 4 and GPU 5, one per arm; third arm queued |

The hardware/compute settings reduce memory and dependency requirements. In
particular, microbatching changes the response-length normalization within each
`token-mean` microbatch. This pilot is not an exact reproduction of the paper.
There is no TailRL control arm, so it can measure learning from sparse successes
but cannot establish an advantage over TailRL.

Training rows are not filtered by model success or path difficulty. The entire
released eval1000 and released SFT test set are excluded by grid identity before
selecting the first 8192 distinct source mazes. Dataset hashes, source IDs, and
the 256 selected eval indices are recorded in `data/pilot_manifest.json`.

## Sources and isolation

- TailRL revision: `5682c6ac03387355e017ce966693266bb148fa10`.
- Checkpoint repo: `max-rl/maze_v2_sft_ckpts_guanning`, revision
  `145e5e2d1ddb160994eb6b2daabcf2362d5d00ce`.
- Dataset repo: `max-rl/maze_17x17_diverse_1.3m`, revision
  `9b9ed56991cb045ba4227d9120dad337085db439`.
- Official configuration:
  [reproduce/_common.sh](https://github.com/Zanette-Labs/TailRL/blob/5682c6ac03387355e017ce966693266bb148fa10/experiments/text_maze/scripts/reproduce/_common.sh).

The installer records the source estimator function hash and both Git revisions
in `rb_port_manifest.json`. It also adds cost routing and local JSONL metrics,
seeds both Ray's driver actor and GPU worker, and makes the unused FlashAttention
padding import optional when SDPA is selected. Released checkpoint metadata is
repaired by TailRL's own checkpoint doctor (RoPE base, tokenizer class, DONE stop).

## Run

Choose separate directories for the source checkout, environment, and state.
From the MaxRL repository, with a Python 3.10 interpreter and `uv` available:

```bash
bash tailrl_experiments/setup_text_maze_rb.sh "$MAZE_CHECKOUT" "$MAZE_STATE_DIR" "$MAZE_VENV" "$PYTHON_3_10"

"$MAZE_VENV/bin/python" tailrl_experiments/run_text_maze_rb_sweep.py \
  --experiment "$MAZE_CHECKOUT/experiments/text_maze" --state-dir "$MAZE_STATE_DIR"

"$MAZE_VENV/bin/python" tailrl_experiments/summarize_text_maze_rb.py \
  --state-dir "$MAZE_STATE_DIR" --output-dir "$MAZE_STATE_DIR/reports/pilot"
```

For one arm, use `run_text_maze_rb.py` with `--gpu 4 --ckpt-step 2450` and
the same experiment/state arguments. `--dry-run` resolves the complete config.
`--smoke --name smoke` runs two updates on a tiny batch. A completed arm is skipped;
an interrupted arm resumes from its own saved checkpoint.

Each arm saves its resolved config, launch arguments, full-precision metrics,
step-0 and step-100 validation trajectories, FSDP checkpoints, and an HF-format
model. `pilot_sweep.json` records PIDs and completion status. No online experiment
tracker is used. Every Ray instance is created locally with exactly one visible
GPU; shutdown is scoped to that run and does not use a host-wide Ray stop.

## Validation

The focused CPU tests cover valid/invalid/shortest/detour trajectories, cost
flooring on failures, early-goal behavior, metadata validation, interleaved
context grouping, all-failure zero updates, context-wise cost-scale invariance,
and exact AST equality of the copied estimator with the MaxRL function.
The two-update GPU smoke additionally exercises rollout, reward, RB routing,
nonzero and zero gradients, validation, and checkpoint export.
