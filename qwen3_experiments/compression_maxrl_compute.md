# Qwen3-1.7B compression MaxRL after the Polaris evaluations

From the primary `maxrl` checkout on `agent/add-math12k-maxrl-launcher`, with the
maxrl environment active:

```bash
bash qwen3_experiments/launch_compression_maxrl_compute.sh
```

The default allocation is 146103. The login node only starts the compute-node
supervisor. `--prepare-only` freezes the inputs and validates the resolved
configuration without launching the supervisor or training.

The supervisor waits for the existing queue to complete Polaris L+0 step100 on
all nine datasets and all fifteen Minerva model/budget points for Polaris L+0,
ER and MaxRL. It verifies the frozen predecessor plan, completed reports,
checkpoint identities and saved rollout hashes before acquiring the shared
allocation locks and all eight GPUs.

After those evaluations finish and their queue releases its locks, the supervisor
removes the three Polaris checkpoints' downloaded and converted model caches.
Training starts only after `evaluation_model_cache_cleanup.json` records a
complete cleanup and the training inputs are verified again. The initial
Qwen3-1.7B model, evaluation responses, reports and archive receipts are retained.
Cleanup failure blocks training; it never bypasses this handoff.

Training starts from original `Qwen/Qwen3-1.7B` revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, using the same 3,200-row
`zjhhhh/compression_dataset` revision
`bfdd7af1633ecc6db191a9f28f76449165a4ee06` as the compression L+0 and ER runs.

- Plain `algorithm.adv_estimator=maxrl`; no length cost.
- `check_eos=false`, `score_after_thinking=true`: only a nonempty answer after
  completed thinking is graded. A correct answer without EOS remains eligible;
  unfinished/reopened thinking and empty answers score zero.
- Natural EOS stopping remains enabled; EOS is never forced or appended.
- 32 prompts × 16 responses, 1,536 prompt tokens, 32,768 response tokens.
- Temperature 0.6, top-p 0.95, top-k 20, seed 79, learning rate 1e-6, KL=0.
- One epoch / 100 steps; checkpoint every ten steps.
- All 51,200 training rollouts are retained and uploaded to a public HF dataset.
  Checkpoints are also public and deleted locally only after upload verification.

The training launcher is `run_qwen3_1_7b_compression_maxrl.sh`. It reuses the
compression launcher; preparation compares its resolved configuration with L+0
and permits only the advantage estimator, EOS reward gate and experiment name
to differ. Existing running jobs retain their frozen code and configuration.

The default run directory is
`outputs/maxrl_no_eos_qwen3_1_7b_compression_bs32_n16_32k_146103/`.
See `supervisor_status.json` for queue state, `plan.json` and
`config_comparison.json` for settings, `rollout_upload.json` for the rollout
verification receipt and `hf_checkpoint_archive/receipts/` for checkpoint
receipts. This entry point queues training and archival; it does not schedule
another evaluation of the newly trained model.
