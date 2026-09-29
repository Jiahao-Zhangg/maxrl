"""Checkpoint completeness and failed-upload retention do not require a GPU."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("grpo_control", ROOT / "qwen3_experiments/grpo_compute_control.py")
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


def make_checkpoint(tmp_path, step=10):
    directory = tmp_path / "checkpoints"
    checkpoint = directory / f"global_step_{step}"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    for name in ("config.json", "tokenizer_config.json"):
        (actor / name).write_text("{}")
    for kind in ("model", "optim", "extra_state"):
        for rank in range(8):
            (actor / f"{kind}_world_size_8_rank_{rank}.pt").write_bytes(b"checkpoint")
    (directory / "latest_checkpointed_iteration.txt").write_text(str(step))
    return checkpoint


def test_checkpoint_requires_all_ranks_and_commit_marker(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    assert CONTROL.checkpoint_complete(checkpoint, checkpoint.parent)
    shard = checkpoint / "actor/model_world_size_8_rank_7.pt"
    shard.unlink()
    assert not CONTROL.checkpoint_complete(checkpoint, checkpoint.parent)
    shard.write_bytes(b"checkpoint")
    marker = checkpoint.parent / "latest_checkpointed_iteration.txt"
    marker.write_text("9")
    assert not CONTROL.checkpoint_complete(checkpoint, checkpoint.parent)
    marker.write_text("10")
    outside = tmp_path / "external.pt"
    outside.write_bytes(b"checkpoint")
    shard.unlink()
    shard.symlink_to(outside)
    assert not CONTROL.checkpoint_complete(checkpoint, checkpoint.parent)


@pytest.mark.parametrize("failure", ["visibility", "upload", "verify", "changed_local", None])
def test_delete_only_after_remote_hash_and_local_consistency_checks(tmp_path, failure):
    checkpoint = make_checkpoint(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    events = []
    privacy = {"private": True}

    def stage(name):
        events.append(name)
        assert checkpoint.exists()
        if failure == name:
            raise RuntimeError(name)

    def create(**kwargs):
        assert kwargs["private"] is False
        stage("create")

    def update_visibility(**kwargs):
        assert kwargs["private"] is False
        stage("visibility")
        privacy["private"] = False

    def upload(**kwargs):
        assert kwargs["path_in_repo"] == "global_step_10"
        stage("upload")

    def verify(*args):
        stage("verify")

    def check(*args):
        stage("changed_local")
        return {"state": "verified", "checkpoint": "global_step_10", "remote_commit": "a" * 40}

    api = SimpleNamespace(create_repo=create, upload_folder=upload,
                          repo_info=lambda **kwargs: SimpleNamespace(**privacy),
                          update_repo_settings=update_visibility)
    verifier = SimpleNamespace(verify_checkpoint=verify, check_local=check)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            CONTROL.archive_checkpoint(checkpoint, receipt_path, "owner/test-step_10", api, verifier)
        assert checkpoint.exists()
        assert not receipt_path.exists()
    else:
        CONTROL.archive_checkpoint(checkpoint, receipt_path, "owner/test-step_10", api, verifier)
        assert not checkpoint.exists()
        assert CONTROL.read(receipt_path)["state"] == "archived_and_deleted"
        assert events == ["create", "visibility", "upload", "verify", "changed_local"]


def test_training_step_is_launched_under_the_existing_allocation():
    command = CONTROL.training_command({"job_id": "146102", "runtime": "/tmp/runtime with spaces"})
    assert command[0] == "srun"
    assert "--jobid=146102" in command and "--gres=gpu:8" in command
    assert command[-1] == "/tmp/runtime with spaces/qwen3_experiments/run_qwen3_1_7b_polaris_1_8_3200_grpo.sh"


def test_control_process_refuses_login_node(monkeypatch):
    monkeypatch.setattr(CONTROL.subprocess, "check_output", lambda *args, **kwargs:
                        f"JobState=RUNNING NumNodes=1 UserId=test({CONTROL.os.getuid()}) BatchHost=compute-23")
    monkeypatch.setattr(CONTROL.socket, "gethostname", lambda: "login-001")
    with pytest.raises(RuntimeError, match="compute node"):
        CONTROL.require_compute("146102")
