# Compression pipeline disk monitoring

`compression_pipeline_watchdog.py` runs inside the same compute allocation as
the frozen Polaris evaluation queue and its following compression MaxRL trainer.
Its configuration, code, lock, event log, status, and recovery logs are copied to
node-local storage. A project-disk status mirror is best effort.

The configuration pins both plans and the recovery code, names the only writable
run directories, and explicitly protects the other allocation's outputs and the
shared Hugging Face base model. It never cleans arbitrary cache directories.
A controller is recovered only after checking its process
identity, allocation, exclusive locks, and any remaining workers. Once training
has started, recovery adopts its recorded Slurm step; it never launches a second
trainer. Recovery runs without waiting for a user response. Local `status.json`,
`events.jsonl`, and recovery receipts retain the evidence and actions.

With `autonomous_recovery=true`, a controller whose heartbeat has stopped for
five minutes can be replaced without signalling its workers. An evaluation step
can be restarted only after fifteen minutes without result progress and five
minutes of negligible CPU and GPU activity. The Slurm step, launcher identity,
allocation and output root are checked again immediately before cancellation.
Only that numeric evaluation step is cancelled; the allocation, training step,
and other node remain untouched. Existing evaluation receipts allow the queue
to continue from verified results. Downloads, generation and grading count as
activity. The `serve` parent also restarts the watchdog if it exits.

Before future outputs are opened, selected rollout, evaluation-ledger and log
paths can point to node-local storage. Existing active outputs are not moved.
Logger routes may explicitly set `skip_symlinks=true`: regular log files are
backed up, while aliases and links into external logger caches are recorded and
skipped. Other routes continue to reject unexpected links.
Checked copies are backed up to each run's `watchdog_backups/` directory when the
project disk has headroom. Under critical pressure only byte-identical duplicate
backups may be removed; canonical outputs and unuploaded rollouts remain intact.
If this is insufficient, closed nine-dataset response payloads with matching
manifest, size and SHA-256 receipts can move to node-local storage. Their canonical
paths become links only after the copy has been checked, and the original bytes
are restored when project storage has headroom. Active payloads are excluded.
Failed metadata writes keep a local copy and retry; quota errors trigger cleanup
even if the filesystem still reports free space.
The monitor keeps trying backups after the pipeline completes. Node-local data
must be backed up or verified on Hugging Face before the allocation is released.

Polaris checkpoints are evaluated in order. With
`cleanup_each_model_after_evaluation=true`, each model's five Minerva budgets
must pass artifact and checkpoint-identity checks before its cache is removed.
L+0 also requires its nine-dataset audit. Workers have exited before deletion;
results and upload receipts remain. Retries skip verified completed models
without downloading their deleted caches again. The final pretraining cleanup
gate remains in place and accepts caches already removed after evaluation.

Example (on the compute node, using a prepared configuration):

```bash
python compression_pipeline_watchdog.py serve --config /node/local/config.json
```

`prepare-storage` refuses to relocate any existing output. `inspect` performs a
monitoring cycle, including authorized recovery, so it is not a read-only command.
Read `status.json` when only status is needed.
