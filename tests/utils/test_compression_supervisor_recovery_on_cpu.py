"""Adopt the existing training process and require its real Slurm exit before eval."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_experiments import compression_supervisor_recovery as recovery
from qwen3_experiments.grpo_compute_control import read, write


def test_progress_uses_global_step_instead_of_timing_or_output_text(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("step:62 - timing_s/step:838.5 - training/global_step:62.000 - accuracy:0.7\n"
                   "response mentions step:1234\n")
    assert recovery.last_completed_step(log) == 62


@pytest.mark.parametrize("state,code", [("FAILED", "1:0"), ("CANCELLED", "0:15"),
                                       ("OUT_OF_MEMORY", "0:9"), ("COMPLETED", "1:0")])
def test_unsuccessful_slurm_exit_never_releases_evaluation(state, code):
    with pytest.raises(RuntimeError, match="training step failed"):
        recovery.successful_exit({"state": state, "exit_code": code})


def test_launcher_exit_waits_for_accounting_and_rejects_pid_reuse():
    assert not recovery.successful_exit(None)
    assert not recovery.successful_exit({"state": "RUNNING", "exit_code": "0:0"})
    assert recovery.successful_exit({"state": "COMPLETED", "exit_code": "0:0"})
    original = {"pid": 42, "start_ticks": 123}
    assert recovery.same_process(original, original)
    assert not recovery.same_process(original, None)
    with pytest.raises(ValueError, match="reused"):
        recovery.same_process(original, {**original, "start_ticks": 456})


@pytest.mark.parametrize("final_saved", [True, False])
def test_adopted_run_only_completes_after_success_final_save_and_uploads(tmp_path, monkeypatch, final_saved):
    root = tmp_path / "run"
    plan = {"job_id": "7", "checkpoint_dir": str(tmp_path / "scratch/checkpoints"), "holder_locks": [],
            "checkpoint_steps": list(range(10, 101, 10)), "runtime": str(tmp_path / "runtime"), "input_hashes": {}}
    write(root / "plan.json", plan)
    write(root / "training_started.json", {"pid": 42})
    log = root / "train.log"
    log.write_text("training/global_step:62.000 - timing_s/step:900\n")
    launcher = {"pid": 42, "start_ticks": 123}
    receipt = root / "recovery.json"
    write(receipt, {"plan_sha256": recovery.digest(root / "plan.json"),
                    "started_sha256": recovery.digest(root / "training_started.json"),
                    "code_sha256": recovery.digest(Path(recovery.__file__)), "launcher": launcher,
                    "step_id": 64, "previous_status": {"last_completed_step": 28, "state": "training"}})
    identities = iter([launcher, None, None])
    accounts = iter([{"state": "RUNNING", "exit_code": "0:0"},
                     {"state": "COMPLETED", "exit_code": "0:0"}])
    spawns, uploads = [], []

    def spawn(_, command, path):
        spawns.append(command)
        assert command == "monitor", "Recovery must never start another trainer"
        return SimpleNamespace(pid=43, poll=lambda: 0 if (root / "training_exit.json").exists() else None)

    def audit(*_):
        assert read(root / "training_exit.json")["slurm_accounting"]["state"] == "COMPLETED"
        uploads.append(True)
        return {"state": "verified", "num_rollouts": 51200}

    control = SimpleNamespace(require_compute=lambda _: None, verify_runtime=lambda _: None,
                              now=lambda: "now", write=write, holder_locks=lambda _: nullcontext(),
                              spawn=spawn, checkpoint_complete=lambda *_: final_saved, audit_rollouts=audit)
    monkeypatch.setattr(recovery, "controller", lambda _: control)
    monkeypatch.setattr(recovery, "process_identity", lambda _: next(identities))
    monkeypatch.setattr(recovery, "slurm_step", lambda *_: next(accounts))
    monkeypatch.setattr(recovery, "archived_steps", lambda *_: plan["checkpoint_steps"])
    monkeypatch.setattr(recovery.time, "sleep", lambda _: log.write_text("training/global_step:100.000 - timing_s/step:900\n"))
    if final_saved:
        recovery.watch(root, receipt)
        status = read(root / "status.json")
        assert status["state"] == "complete" and status["exit_code"] == 0 and status["last_completed_step"] == 100
        assert uploads == [True] and spawns == ["monitor"]
        assert read(root / "training_exit.json")["launcher_exit_code"] is None
    else:
        with pytest.raises(RuntimeError, match="missing the completed final checkpoint"):
            recovery.watch(root, receipt)
        assert read(root / "status.json")["state"] == "failed"
        assert not uploads and not (root / "training_exit.json").exists()


def test_archive_receipts_must_match_this_run(tmp_path):
    plan = {"checkpoint_steps": [10], "hf_repo_prefix": "owner/l4096"}
    receipt = tmp_path / "hf_checkpoint_archive/receipts/global_step_10.json"
    write(receipt, {"repo_id": "owner/l4096-step_10", "checkpoint": "global_step_10",
                    "state": "archived_and_deleted", "verified_at": "now", "remote_commit": "a" * 40})
    assert recovery.archived_steps(tmp_path, plan) == [10]
    write(receipt, {**read(receipt), "repo_id": "owner/another-step_10"})
    with pytest.raises(ValueError, match="another run"):
        recovery.archived_steps(tmp_path, plan)
