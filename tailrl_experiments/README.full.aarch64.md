# Sequential full RB training on four GH200 GPUs

Each run uses the existing per-context fixed-N RB estimator. The default is
seed 0 and SFT 3000; the manual launcher also supports other seeds and N=16/32.
The historical seed-0 queue ran **3000 → 2450 → 3250**, stopping after the
SFT 3250 arm completed all 5001 training steps and saved its final checkpoint.
The [2026-09-16 result snapshot](../docs/evaluation_results/textmaze_rb_20260916/README.md)
also includes SFT 2450 with seed 1. N=16/seed=2 and N=32/seed=0 are supported
variants, not completed runs in that snapshot. Publishing these scripts does not
schedule or resume any experiment.
Training settings come from TailRL revision
`5682c6ac03387355e017ce966693266bb148fa10`,
`experiments/text_maze/scripts/reproduce/_common.sh`, `scripts/train.sh`, and
the vendored `ppo_trainer.yaml`. The user's selected hardware is four GPUs
jointly training one combination at a time, with global prompt batch 256.

| Setting | Full run |
|---|---|
| Initial model | Released tiny Qwen2 at the selected SFT step, 3,944,704 parameters |
| Estimator / default seed | Per-context fixed-N cost-aware RB / 0 |
| Reward / cost | Binary goal success / `max(generated actions, BFS shortest distance)` |
| Training / validation contexts | 1,298,992 / released eval1000 |
| Global prompt batch / default samples per prompt | 256 / 16 |
| Actor minibatch / per-GPU microbatch | 256 prompts / 1024 trajectories |
| GPUs / updates per rollout | 4 / 1 |
| Learning rate / weight decay | 1e-4 / 0.01 |
| KL / entropy coefficient | 0 / 0 |
| Prompt / response limit | 320 / 180 tokens |
| Sampling | Temperature 1, top-p 1, unrestricted top-k; EOS and DONE stop |
| Training-step setting | 5001 |
| Save / validation frequency | 250 / 1000 steps, plus final |
| Validation samples per context | 64, including step-0 validation |
| Attention / precision | FlashAttention 2 / FP16 actor and rollout |
| Gradient checkpointing / torch.compile | Enabled / enabled |
| Checkpoints | Model, optimizer, resume metadata, HF export; retain latest 3 |

The full raw release has 1,299,992 mazes. The released eval1000 is present in
that corpus and includes the released 256 SFT test mazes. Preparation excludes
all 1000 held-out grids, preserves source ordering, and uses every remaining
maze. It does not filter by policy success or path difficulty. The streaming
converter avoids loading the complete 4 GB JSONL into memory. The manifest
records zero training/evaluation overlap and the raw/output SHA-256 hashes.

`install_text_maze_full_adapter.py` first installs the existing exact RB
function port, then supplies the global response-token count to each GPU.
For token-mean loss, each rank's loss is weighted by
`world_size * local_response_tokens / global_response_tokens`; FSDP's average
then matches a token mean over the complete global batch. This preserves the
original objective when response lengths differ across GPU shards. The RB
estimator function itself is unchanged.

Metrics and validation trajectories are written locally. The separate paper
evaluation ladder is outside the training configuration described here.
An HF-format checkpoint export is not an automatic Hugging Face upload; W&B is
disabled in this launcher. No TailRL/GRPO/RLOO control arm is included in the
published RB snapshot.

## Independent ARM environment

The isolated Conda environment is named `tailrl_rb`, with Python
3.10.15, PyTorch 2.6.0+cu126, FlashAttention 2.7.4.post1, Triton 3.2.0,
TensorDict 0.6.2, Transformers 4.57.6, and Ray 2.53.0. CUDA 12.6.1 and GCC 13
are selected by the checked-in activation/deactivation hooks. No ER or MaxRL
environment packages are changed.

For this installation, TensorDict 0.6.2's native extension was built with four
CPU compiler workers from official tag `v0.6.2`, commit
`88c86f8379bca104c4021f59cc5e62409e81abbe`. A matching FlashAttention wheel and
the corrected executable-permission Triton wheel are copied into this task's
own wheel directory. The Torch 2.6 `sm_90a` architecture-parser fix is also
applied only inside the new environment. Native wheel hashes and provenance,
the complete resolved package lock, and the Conda explicit lock are retained.

Choose independent directories and set `MAXRL_REPO`, `MAZE_CHECKOUT`,
`MAZE_STATE_DIR`, and `MAZE_OUTPUT_DIR` for your installation. Model weights
belong under `$MAZE_STATE_DIR/checkpoints/ckpt-<step>`; the prepared datasets and
`full_manifest.json` belong under `$MAZE_STATE_DIR/data`. Machine-specific wheel
directories and lock files are not published. Package pins are in
`requirements_text_maze.aarch64.txt`; matching ARM native wheels/build
prerequisites are still required in addition to that file.

Install the hooks as `etc/conda/activate.d/tailrl_rb.sh` and
`etc/conda/deactivate.d/tailrl_rb.sh` inside the isolated Conda environment.
`MAZE_CUDA_HOME`, `MAZE_HOST_CC`, and `MAZE_HOST_CXX` override site-specific
CUDA/compiler locations.

```bash
python "$MAXRL_REPO/tailrl_experiments/install_text_maze_full_adapter.py" \
  --checkout "$MAZE_CHECKOUT" --maxrl-root "$MAXRL_REPO"
python "$MAXRL_REPO/tailrl_experiments/prepare_text_maze_full.py" \
  --experiment "$MAZE_CHECKOUT/experiments/text_maze" \
  --data-dir "$MAZE_STATE_DIR/data"
```

Preparation expects the pinned raw `main_1.3M.jsonl` and released SFT `test.json`
in the data directory, and the released eval1000 in the dedicated TailRL checkout.
Only that checkout is patched; the existing MaxRL trainer is untouched.

## Run one seed/N arm

Run inside an authorized allocation with four idle GPUs:

```bash
conda activate tailrl_rb
unset PYTHONPATH PYTHONHOME RAY_ADDRESS
python "$MAXRL_REPO/tailrl_experiments/run_text_maze_rb_full.py" \
  --experiment "$MAZE_CHECKOUT/experiments/text_maze" \
  --state-dir "$MAZE_STATE_DIR" --output-dir "$MAZE_OUTPUT_DIR" \
  --ckpt-step 2450 --seed 1 --n-rollouts 16
```

`--seed` accepts an unsigned 32-bit integer. `--n-rollouts` (also spelled
`--n_rollouts`) accepts 16 or 32. N=32 keeps the 1024-trajectory per-GPU microbatch
cap, accumulating two microbatches for one global update. Run identities include
the initialization, seed, and N, so completion/resume cannot cross arms.
`--dry-run` writes the resolved configuration without training; `--smoke` runs
two updates in its own output directory.

Each arm has its own resolved `config.yaml`, metrics, validation outputs,
checkpoints, and completion record. `run_text_maze_rb_full.py --ckpt-step <step>`
selects its released initialization. Other settings remain identical unless
explicitly selecting a seed/N variant. Retries resume only that arm's checkpoint
directory. Keep environment/input storage separate from training output storage;
Ray and compiler temporary files use node-local `/tmp`. No GPU was used for
environment installation or the CPU checks.

The preparation-time CPU checks verified native imports, loading all seven SFT checkpoints and finite model
forward pass, the binary reference reward, the exact estimator port, global
token-mean gradient equivalence, full training parameters, checkpoint-gated
launching, duplicate prevention, ordered transitions after Slurm cleanup,
per-arm retries, deadline handling, recovery after a final save, and stopping
on GPU validation failure, and releasing the holder only after final checkpoint
saving and Slurm step cleanup. The released checkpoint metadata is repaired
with TailRL's checkpoint doctor: RoPE theta 1e6, fast tokenizer class, and
EOS/DONE generation termination.
The checkpoint source is pinned to Hugging Face revision
`145e5e2d1ddb160994eb6b2daabcf2362d5d00ce` of
`max-rl/maze_v2_sft_ckpts_guanning`. `state/sweep_inputs.json` records each
model's weight hash and CPU forward check.

## Historical checkpoint-gated queue and recovery

The archived queue was authorized only for holder `3114675`.
Its preflight still deliberately rejects other holder IDs; that safety guard
has not been relaxed for publication. Use the manual launcher above for newly
authorized runs, not this old operational plan. `queue_full_rb.py` waits until
ER's controller records a verified step 60
archive, completed ER shutdown, and retention of that holder. It also checks
that all four GPUs are idle and acquires the existing shared holder lock.
The plan's `cancel_holder_after_completion: true` authorizes cancellation of
holder 3114675 after the final configured arm. No additional allocation is submitted.
The plan supplies site-specific owner/account/partition, runner/input/output
paths, holder lock directory, and allocation deadline. Private plans and live
controller state are not included. The queue imports Slurm/identity/atomic-state
helpers from `qwen3_experiments/checkpoint_handoff.py`; importing that module does
not start its separate HF handoff controller. Existing pending-job priorities
and STOP markers are not changed by this publication.

After the GPUs are released, `run_full_rb_on_slurm.sh` performs native
FlashAttention/Triton backward checks, a four-rank NCCL all-reduce, and a
separate two-update four-GPU smoke run including reward, RB, validation, and
checkpoint/HF export. Each stage must pass before full training starts from
SFT 3000. The smoke run has a separate output directory and does not initialize
the full run. Successful GPU validation is reused for later arms on this holder.
GPU validation passed on holder 3114675 after ER released the node.

Failures end only the attempt's Slurm step, so Slurm reaps its processes before
the controller retries the same arm after five minutes. Full training resumes
from its latest local checkpoint. A completed arm advances only after its final
5001-step checkpoint, matching completion record, driver exit, and Slurm step
cleanup. The controller checks for four idle GPUs again before the next arm.
It preserves the allocation deadline in its plan. No new arm starts after that
deadline, and Slurm ends running work naturally. Queue state records `ALLOCATION_ENDED`, keeping
saved checkpoints, completed arms, and remaining arms for later continuation.
Every arm retains its full 5001-step target even if only part fits in the remaining time.
After the three configured arms finish, the controller rechecks the final
completion record and checkpoint, holder identity, shared launch lock, Slurm
steps, and GPU cleanup. It then cancels only holder 3114675 and waits for Slurm
to report a terminal state before recording `COMPLETE_HOLDER_RELEASED`.
Transient cancellation failures are retried, including after a controller restart.
If the cancellation flag is absent or false, completion retains the holder.
A `STOP` file in the queue directory stops future controller actions while
leaving already running work untouched.

## CPU validation

Activate the isolated environment and run from the TailRL experiment directory
with its patched `verl` first on `PYTHONPATH`:

```bash
cd "$MAZE_CHECKOUT/experiments/text_maze"
PYTHONPATH="$PWD:$MAXRL_REPO" CUDA_VISIBLE_DEVICES= python -m pytest \
  "$MAXRL_REPO/tailrl_experiments/tests" \
  "$MAXRL_REPO/tests/trainer/ppo/test_checkpoint_handoff_on_cpu.py" \
  --import-mode=importlib -q -p no:cacheprovider
```

All 85 tests passed in the 2026-09-16 publication check. Slurm/HF operations in
these tests are mocked; no training or GPU validation was launched. Running from
the MaxRL root can select its different `verl`, so that is not a valid test of
the isolated TailRL environment.
