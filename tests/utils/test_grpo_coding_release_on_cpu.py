import errno
import json

import pytest

from qwen3_experiments import grpo_coding_release as release


def test_checkpoint_requires_completion_marker_and_all_rank_states(tmp_path):
    plan = {"checkpoint_dir": str(tmp_path)}
    checkpoint = tmp_path / "global_step_10"
    for relative in release.required_checkpoint_files():
        target = checkpoint / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("data")
    marker = tmp_path / "latest_checkpointed_iteration.txt"
    assert not release.checkpoint_complete(plan, 10)
    marker.write_text("9")
    assert not release.checkpoint_complete(plan, 10)
    marker.write_text("10")
    assert release.checkpoint_complete(plan, 10)
    (checkpoint / "actor/optim_world_size_8_rank_7.pt").unlink()
    assert not release.checkpoint_complete(plan, 10)


def test_control_records_survive_home_disk_full(tmp_path, monkeypatch):
    scratch, home = tmp_path / "scratch", tmp_path / "home"
    scratch.mkdir()
    home.mkdir()
    plan = {"scratch": str(scratch), "output_root": str(home)}
    original = release.write

    def disk_full(path, value):
        if path.is_relative_to(home):
            raise OSError(errno.ENOSPC, "Disk full")
        original(path, value)

    monkeypatch.setattr(release, "write", disk_full)
    assert release.persist(plan, "status.json", {"state": "training"})
    assert release.state(plan, "status.json")["state"] == "training"
    assert json.loads((scratch / "control_mirrors/status.json").read_text())["state"] == "training"


def test_successor_waits_for_four_verified_evaluations_and_rollout_archives(tmp_path):
    root, scratch = tmp_path / "predecessor", tmp_path / "scratch"
    root.mkdir()
    scratch.mkdir()
    predecessor = {"output_root": str(root), "scratch": str(scratch), "job_id": "123", "node": "compute"}
    release.write(root / "plan.json", predecessor)
    sha = release.digest(root / "plan.json")
    successor = {"job_id": "123", "node": "compute", "predecessor": {"root": str(root), "plan_sha256": sha}}
    assert not release.predecessor_complete(successor)
    release.persist(predecessor, "training_exit.json", {"exit_code": 0})
    release.persist(predecessor, "queue_status.json", {"state": "complete"})
    assert not release.predecessor_complete(successor)
    release.persist(predecessor, "rollout_status.json", {"state": "complete"})
    for name, count in release.evaluation.COUNTS.items():
        assert not release.predecessor_complete(successor)
        release.persist(predecessor, f"evaluation/{name}/audit.json", {
            "complete": True, "questions": count, "plan_sha256": sha,
        })
    assert release.predecessor_complete(successor)
    first = next(iter(release.evaluation.COUNTS))
    release.persist(predecessor, f"evaluation/{first}/audit.json", {
        "complete": True, "questions": release.evaluation.COUNTS[first], "plan_sha256": "wrong-run",
    })
    assert not release.predecessor_complete(successor)
    release.write(root / "plan.json", {**predecessor, "node": "other-compute"})
    with pytest.raises(ValueError, match="identity changed"):
        release.predecessor_complete(successor)


def test_model_cleanup_refuses_changed_files_and_can_resume_partial_deletion(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "model.safetensors"
    weight.write_bytes(b"verified weights")
    hashes = {weight.name: release.digest(weight), "already-deleted": "irrelevant"}
    weight.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="changed; retaining"):
        release.delete_verified_model_tree(model, hashes)
    assert weight.exists()
    weight.write_bytes(b"verified weights")
    release.delete_verified_model_tree(model, hashes)
    assert not model.exists()
    release.delete_verified_model_tree(model, hashes)


def test_model_cleanup_refuses_symlinks(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    other = tmp_path / "other-run.bin"
    other.write_bytes(b"in use")
    (model / "weights").symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        release.delete_verified_model_tree(model, {"weights": release.digest(other)})
    assert other.read_bytes() == b"in use"
