import errno
import os

import pytest

from qwen3_experiments import er_compression_recovery as recovery
from qwen3_experiments.grpo_compute_control import read, write


def complete_run(root):
    plan = {"train_dir": str(root / "train"), "checkpoint_steps": [20, 40, 60, 80, 100],
            "hf_repo_prefix": "user/er", "rollout_hf_repo": "user/er-rollouts"}
    (root / "train").mkdir()
    for name in ("training_exit_status", "exit_status"):
        (root / "train" / name).write_text("0\n")
    write(root / "rollout_dataset/rollout_manifest.json", {
        "num_steps": 100, "num_rollouts": 25600,
        "steps": {str(i): {"file": f"data/step_{i:06d}.jsonl.gz", "num_rollouts": 256} for i in range(1, 101)},
    })
    write(root / "rollout_upload.json", {"state": "verified", "num_rollouts": 25600, "num_steps": 100,
                                         "repo_id": plan["rollout_hf_repo"], "revision": "a" * 40})
    for step in [*plan["checkpoint_steps"], None]:
        name = f"global_step{step}" if step is not None else "final_model"
        files = {"config.json": 100, "model.safetensors": 1024}
        write(root / "archive_receipts" / f"{name}.json", {
            "repo_id": plan["hf_repo_prefix"] + (f"-step_{step}" if step is not None else "-final"),
            "path": f"global_step_{step}/actor" if step is not None else "", "local_path": name,
            "revision": "b" * 40, "files": files, "sha256": {name: "c" * 64 for name in files},
        })
    return plan


def test_completion_requires_all_verified_outputs(tmp_path):
    plan = complete_run(tmp_path)
    assert recovery.validate_completion(tmp_path, plan) == {
        "exit_code": 0, "completed_rollout_steps": 100, "optimizer_updates": 200, "saved_training_rollouts": 25600,
    }


@pytest.mark.parametrize("damage", ["training_failed", "archive_failed", "missing_step", "partial_step",
                                    "wrong_upload", "wrong_checkpoint", "missing_hash", "wrong_final_format"])
def test_reject_incomplete_or_wrong_run(tmp_path, damage):
    plan = complete_run(tmp_path)
    if damage in ("training_failed", "archive_failed"):
        (tmp_path / "train" / ("training_exit_status" if damage == "training_failed" else "exit_status")).write_text("1")
    elif damage in ("missing_step", "partial_step"):
        path = tmp_path / "rollout_dataset/rollout_manifest.json"
        value = read(path)
        if damage == "missing_step":
            del value["steps"]["99"]
        else:
            value["steps"]["99"]["num_rollouts"] = 255
        write(path, value)
    else:
        path = tmp_path / ("rollout_upload.json" if damage == "wrong_upload" else "archive_receipts/final_model.json")
        value = read(path)
        if damage in ("wrong_upload", "wrong_checkpoint"):
            value["repo_id"] = "user/wrong"
        elif damage == "missing_hash":
            value["sha256"].pop("config.json")
        else:
            value["files"].pop("model.safetensors")
            value["sha256"].pop("model.safetensors")
        write(path, value)
    with pytest.raises(ValueError):
        recovery.validate_completion(tmp_path, plan)


def test_launcher_must_match_exact_run_and_allocation():
    plan = {"job_id": "7", "node": "compute", "runtime": "/frozen"}
    command = recovery.training.training_command(plan)
    started = {"pid": 42, "hostname": "compute", "command": command}
    identity = {"pid": 42, "uid": os.getuid(), "command": command, "cgroup": "slurmstepd/allocation7/extern"}
    recovery.validate_launcher(plan, started, identity, identity["cgroup"])
    for changed in ({"pid": 43}, {"uid": -1}, {"command": ["srun", "different"]}, {"cgroup": "other"}):
        with pytest.raises(ValueError):
            recovery.validate_launcher(plan, started, {**identity, **changed}, identity["cgroup"])


def test_reused_pid_is_not_original_launcher():
    identity = {"pid": 42, "start_ticks": 123}
    assert recovery.check_identity(identity, identity)
    assert not recovery.check_identity(identity, None)
    with pytest.raises(ValueError, match="reused"):
        recovery.check_identity(identity, {**identity, "start_ticks": 456})


def test_retry_shared_quota_without_losing_local_status(tmp_path, monkeypatch):
    plan = {"job_id": "7", "scratch": str(tmp_path / "scratch")}
    shared = tmp_path / "shared"
    attempts, pauses = [], []

    def simulated_write(path, value):
        if path == shared / "status.json":
            attempts.append(value)
            if len(attempts) == 1:
                raise OSError(errno.EDQUOT, "quota")
        write(path, value)

    monkeypatch.setattr(recovery, "write", simulated_write)
    monkeypatch.setattr(recovery, "require_compute", lambda job: None)
    monkeypatch.setattr(recovery.time, "sleep", pauses.append)
    recovery.durable_status(shared, plan, {"state": "complete"})
    assert len(attempts) == 2 and pauses == [30]
    assert read(tmp_path / "scratch/recovered_supervisor_status.json") == read(shared / "status.json")
