# Qwen3-1.7B compression f_cov, L_0=0

`run_qwen3_1_7b_compression_f_cov_l0_0.sh` reuses the existing compression
thinking recipe with `algorithm.adv_estimator=f_cov`. The model is
`Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; the dataset is
the same 3,200-row `zjhhhh/compression_dataset` at
`bfdd7af1633ecc6db191a9f28f76449165a4ee06` used for compression L+0 and MaxRL.

- Full rollout batch: 32 prompts, 16 responses per prompt, 512 responses/step.
- `algorithm.cost_offset_tokens=0`: cost is the full generated response length,
  including thinking and the final answer. Global f_cov statistics use all
  prompts in the rollout batch; `f_cov_num_prompts` resolves from
  `data.train_batch_size`, not a GPU microbatch size.
- Thinking is enabled. Only the nonempty answer after completed thinking is
  graded; unfinished or reopened thinking scores zero.
- `check_eos=false`, `force_eos=false`, `ignore_eos=false`: no EOS reward gate
  or appended EOS; generation still stops naturally on EOS.
- 1,536 prompt tokens, 32,768 response tokens, temperature 0.6, top-p 0.95,
  top-k 20, learning rate 1e-6, KL=0, seed 79, token-mean loss.
- Eight GPUs, one epoch / 100 steps, checkpoint every ten steps, no in-training
  validation. All 51,200 training rollouts are saved and uploaded to a public HF
  dataset before successful trainer exit.

The training settings match the existing compression launcher with EOS checking
disabled; only the advantage estimator, its full-batch prompt count, and output
names/paths differ. The existing f_cov implementation supplies the advantages
without additional whitening or multiplication by prompt/response counts.

From the primary checkout, preview without downloads or training:

```bash
DRY_RUN=1 bash qwen3_experiments/run_qwen3_1_7b_compression_f_cov_l0_0.sh
```

Resolve the full configuration with the maxrl environment on a compute node:

```bash
bash qwen3_experiments/run_qwen3_1_7b_compression_f_cov_l0_0.sh --cfg job --resolve
```

On an allocation with eight available GPUs, the same script without preview
flags starts training. It accepts the shared launcher's `L0_DATA_DIR`,
`L0_RUN_DIR`, `L0_CHECKPOINT_DIR`, `L0_RAY_DIR`, `L0_ROLLOUT_DIR`,
`L0_ROLLOUT_HF_REPO`, and `PYTHON_BIN` settings. Checkpoints stay in
`L0_CHECKPOINT_DIR` (default: `L0_RUN_DIR/checkpoints`); checkpoint archival and
deletion use a separately attached compute supervisor. This script does not
change an existing queue or attach a supervisor by itself.

To queue the run after compression MaxRL in allocation 146103, use the compute
launcher from the primary checkout with the maxrl Python environment selected:

```bash
FCOV_PYTHON_BIN="$(command -v python)" bash qwen3_experiments/launch_compression_f_cov_compute.sh
```

Use `--prepare-only` to freeze and verify inputs without starting a supervisor.
Overrides are `FCOV_JOB_ID`, `FCOV_RUN_DIR`, `FCOV_HF_PREFIX`,
`FCOV_ROLLOUT_HF_REPO`, `FCOV_PREDECESSOR_TRAINING` and `FCOV_SSH_KNOWN_HOSTS`.
The predecessor must be this allocation's compression MaxRL run. Training starts
from the pinned initial Qwen3 weights, not the predecessor's checkpoint.

The supervisor waits for all 100 MaxRL steps, successful trainer exit, ten
verified checkpoint archives and all 51,200 verified public rollouts. It then
acquires the predecessor's supervisor/GPU locks, checks that all GPUs are idle,
and removes only predecessor rollout shards matching their pinned public Hub
commit. Initial model weights and other allocations' caches remain protected.
Cleanup receipts survive on compute-local storage and are mirrored in the new
run directory. Failed checks or incomplete cleanup cannot start f_cov.

Checkpoints are written under `/tmp/compressionfcov146103/checkpoints`, archived
publicly every ten steps and removed locally only after verification. The
deployed disk watchdog routes rollouts and logs to compute-local storage, retries
disk-full metadata writes and recovers a lost supervisor while retaining an
existing trainer. It uses a separate control directory and leaves the MaxRL and
146102 controllers running. Public destinations default to
`hi-todayis-jh/f-cov-l0-0-no-eos-qwen3-1.7b-compression-bs32-n16-32k-146103-step_N`
and the same prefix with `-rollouts`. No evaluation is added by this launcher.
