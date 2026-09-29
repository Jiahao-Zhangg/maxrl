# Compute-node disk and queue watchdog

`compute_disk_watchdog.py` monitors one already-running Slurm allocation. Its
configuration pins the existing training and evaluation plans and identifies
their exact compute node. `disk_watchdog.json` in the training output directory
points to the installed configuration, local status, and home backup.

- Check disk space and control-process identities every 30 seconds.
- Persist watchdog status on the compute node's local disk first, with an
  independent copy under `/home`. A full project filesystem cannot prevent
  these status writes. Space errors on either status destination are contained;
  even when both copies are unavailable, the watchdog keeps running.
- Store the adopted supervisor's mutable control records, checkpoint upload
  receipts, and both future evaluation directories on the independent home
  filesystem. Original paths remain available through symlinks. Migration is
  restricted to idle control processes, verifies every copied file, and leaves
  the original training process and frozen experiment inputs unchanged. Atomic
  updates to root-level control files explicitly follow their pinned storage
  routes so they cannot replace the symlinks with project-local files.
- The protected checkpoint monitor and evaluation queues mirror pending JSON
  writes to both locations, then retry project writes on `ENOSPC` or `EDQUOT`.
  They do not advance past an unpersisted checkpoint receipt. The adopted
  training supervisor also retains its original local-write-and-retry behavior.
- With `eager_checkpoint_upload` enabled, poll the checkpoint completion marker
  and all required rank files every `checkpoint_poll_seconds` (2 seconds in the
  installed run). Upload each complete checkpoint to its public repository,
  verify every remote file hash and unchanged local inventory, persist the
  receipt, then remove the local checkpoint immediately. Failed uploads retain
  local files and retry after 30 seconds. Idle status heartbeats remain at 30
  seconds to avoid unnecessary metadata writes.
- The final checkpoint can also be archived before the trainer exits. The
  adopted supervisor accepts its verified archive receipt as evidence of the
  completed save, including the narrow interval between deletion and updating
  the receipt. It still requires the original Slurm step's successful exit and
  final training-step log before releasing evaluation. A crash after complete
  deletion is reconciled from the verified receipt without reuploading.
- Recover an exited supervisor by adopting the original training launcher and
  frozen recovery receipt. Recover an exited waiting evaluation queue only if
  no launcher or worker for that stage is still active. Validate allocation,
  ownership, process locks, and pinned inputs before launching a replacement.
- Never start another trainer, signal a process, or operate another allocation's
  control processes. The one-time installation can replace verified idle control
  processes to enable protected I/O while preserving the original trainer.
- Below the configured critical free-space threshold, move at most one closed
  rollout shard per iteration from this run to the independent home filesystem.
  Verify its manifest, bytes, and unchanged source metadata, then atomically
  retain the original path as a symlink. This is relocation, not data deletion.
- Record meaningful state changes in `events.jsonl`; there is no chat push
  notification integration. Exit after both final-checkpoint evaluations have
  completed and their audits pass.

Automatic recovery is deliberately limited to control-process failures. A live
but blocked process is retained. Training failures, changed inputs, or an exited
evaluation controller with active GPU workers are recorded for inspection rather
than launching duplicate work. The existing trainer's active log and rollout
writer remain on their original filesystem; critical-space relocation frees
closed rollout shards without modifying active files. Shared-disk exhaustion can
still affect those training writes, but cannot fill the independently stored
control records or future evaluation outputs.

Unused model-cache cleanup across allocations must be separately verified against
both running and queued dependencies. Protect the base model, queued checkpoint
models, datasets, raw evaluation responses, reports, and frozen provenance. A
completed model's public checkpoint and checksums must be verified before removing
its reconstructable local copies. The watchdog's automatic rollout relocation
remains restricted to its own run.

Validation: `tests/utils/test_compute_disk_watchdog_on_cpu.py` covers quota retries,
independent status destinations, monitor exit codes, protected supervisor spawn,
cross-allocation rejection, live-worker exclusion, frozen-input verification,
safe rollout relocation, and final-result audit gating.
Storage-route tests also reject redirected symlinks and recognize evaluation
workers whose command lines use the resolved independent-storage path.
`tests/utils/test_compute_checkpoint_archive_on_cpu.py` checks prompt final-save
upload, incomplete-save retention, interrupted uploads, hash mismatches, invalid
archive receipts, deletion recovery, and the original supervisor's final handoff.
