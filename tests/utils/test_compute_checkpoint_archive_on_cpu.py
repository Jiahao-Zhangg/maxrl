"""Prompt uploads retain checkpoint safety and the real training-exit gate."""

import hashlib
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from qwen3_experiments import compute_disk_watchdog as watchdog
from qwen3_experiments import grpo_compute_control as control


class EndPoll(Exception):
    pass


@pytest.fixture
def archiver(tmp_path, monkeypatch):
    root, directory = tmp_path / "run", tmp_path / "checkpoints"
    root.mkdir()
    checkpoint = directory / "global_step_100"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"loader")
    for name in ("config.json", "tokenizer_config.json"):
        (actor / name).write_text("{}")
    for kind in ("model", "optim", "extra_state"):
        for rank in range(8):
            (actor / f"{kind}_world_size_8_rank_{rank}.pt").write_bytes(f"{kind}-{rank}".encode())
    (directory / "latest_checkpointed_iteration.txt").write_text("100")
    plan = {"job_id": "7", "output_root": str(root), "checkpoint_dir": str(directory),
            "hf_repo_prefix": "owner/current", "checkpoint_steps": [100]}
    config = {"training_root": str(root), "control_root": str(tmp_path / "control"),
              "checkpoint_poll_seconds": 2, "interval_seconds": 30}
    calls, sleeps = [], []

    def upload(**kwargs):
        assert kwargs["repo_id"] == "owner/current-step_100"
        assert control.checkpoint_complete(checkpoint, directory)
        assert not (root / "training_exit.json").exists()
        calls.append(kwargs)

    def repo_info(**kwargs):
        files = []
        if kwargs.get("files_metadata"):
            for p in checkpoint.rglob("*"):
                if p.is_file():
                    files.append(SimpleNamespace(rfilename=f"{checkpoint.name}/{p.relative_to(checkpoint)}",
                                                 size=p.stat().st_size,
                                                 lfs={"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}))
        return SimpleNamespace(private=False, sha="a" * 40, siblings=files)

    def create(**kwargs):
        assert kwargs["private"] is False

    api = SimpleNamespace(create_repo=create, upload_folder=upload, repo_info=repo_info)
    verifier = watchdog.load("prompt_archive_verifier", Path(control.__file__).with_name("verify_checkpoint_upload.py"))
    monkeypatch.setattr(control, "load_verifier", lambda plan: verifier)
    monkeypatch.setattr(control, "require_compute", lambda job: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api))

    def sleep(seconds):
        sleeps.append(seconds)
        raise EndPoll

    monkeypatch.setattr(watchdog.time, "sleep", sleep)
    return SimpleNamespace(plan=plan, config=config, root=root, directory=directory, checkpoint=checkpoint,
                           api=api, verifier=verifier, calls=calls, sleeps=sleeps,
                           receipt=root / "hf_checkpoint_archive/receipts/global_step_100.json")


def test_final_save_uploads_and_deletes_before_training_exit(archiver):
    a = archiver
    with pytest.raises(EndPoll):
        watchdog.eager_checkpoint_monitor(a.plan, a.config, control)
    assert len(a.calls) == 1 and a.sleeps == [2]
    assert not a.checkpoint.exists()
    assert watchdog.read(a.receipt)["state"] == "archived_and_deleted"
    assert watchdog.archive_receipt_ready(a.plan, a.checkpoint, a.directory)
    status = watchdog.read(a.root / "hf_checkpoint_archive/status.json")
    assert status["state"] == "monitoring" and status["archived_steps"] == [100]
    assert not (a.root / "training_exit.json").exists()


@pytest.mark.parametrize("incomplete", ["rank", "marker"])
def test_incomplete_save_is_never_uploaded(archiver, incomplete):
    a = archiver
    if incomplete == "rank":
        (a.checkpoint / "actor/model_world_size_8_rank_7.pt").unlink()
    else:
        (a.directory / "latest_checkpointed_iteration.txt").write_text("90")
    with pytest.raises(EndPoll):
        watchdog.eager_checkpoint_monitor(a.plan, a.config, control)
    assert not a.calls and a.checkpoint.exists() and a.sleeps == [2]
    assert not a.receipt.exists()


@pytest.mark.parametrize("failure", ["upload", "hash"])
def test_failed_upload_or_verification_retains_files_and_retries(archiver, failure):
    a = archiver
    if failure == "upload":
        def upload(**kwargs):
            raise ConnectionError("interrupted upload")
        a.api.upload_folder = upload
    else:
        original = a.api.repo_info

        def info(**kwargs):
            result = original(**kwargs)
            if result.siblings:
                result.siblings[0].lfs["sha256"] = "f" * 64
            return result
        a.api.repo_info = info
    with pytest.raises(EndPoll):
        watchdog.eager_checkpoint_monitor(a.plan, a.config, control)
    assert a.checkpoint.exists() and not a.receipt.exists() and a.sleeps == [30]
    assert watchdog.read(a.root / "hf_checkpoint_archive/status.json")["state"] == "retrying"


@pytest.mark.parametrize("mutation", ["repo", "rank", "commit", "path", "marker"])
def test_archive_proof_rejects_wrong_or_incomplete_receipts(archiver, mutation):
    a = archiver
    with pytest.raises(EndPoll):
        watchdog.eager_checkpoint_monitor(a.plan, a.config, control)
    receipt = watchdog.read(a.receipt)
    if mutation == "repo":
        receipt["repo_id"] = "owner/other-step_100"
    elif mutation == "rank":
        receipt["files"].pop("actor/optim_world_size_8_rank_7.pt")
    elif mutation == "commit":
        receipt["remote_commit"] = "main"
    elif mutation == "path":
        receipt["checkpoint_path"] = "/another/run/global_step_100"
    else:
        (a.directory / "latest_checkpointed_iteration.txt").write_text("90")
    watchdog.write(a.receipt, receipt)
    assert not watchdog.archive_receipt_ready(a.plan, a.checkpoint, a.directory)


def test_reconcile_receipt_after_crash_between_deletion_and_final_write(archiver):
    a = archiver
    with pytest.raises(EndPoll):
        watchdog.eager_checkpoint_monitor(a.plan, a.config, control)
    receipt = watchdog.read(a.receipt)
    receipt["state"] = "verified"
    watchdog.write(a.receipt, receipt)
    assert watchdog.archive_receipt_ready(a.plan, a.checkpoint, a.directory)
    with pytest.raises(EndPoll):
        watchdog.eager_checkpoint_monitor(a.plan, a.config, control)
    assert len(a.calls) == 1
    assert watchdog.read(a.receipt)["state"] == "archived_and_deleted"


@pytest.mark.parametrize("exit_code", [0, 1])
def test_remote_final_save_does_not_bypass_training_exit(archiver, exit_code):
    a = archiver
    with pytest.raises(EndPoll):
        watchdog.eager_checkpoint_monitor(a.plan, a.config, control)
    watchdog.write(a.root / "training_exit.json", {"exit_code": exit_code})
    assert watchdog.eager_checkpoint_monitor(a.plan, a.config, control) == exit_code


def test_original_supervisor_handoff_accepts_verified_removed_final_save(archiver, monkeypatch):
    a = archiver
    with pytest.raises(EndPoll):
        watchdog.eager_checkpoint_monitor(a.plan, a.config, control)
    assert not a.checkpoint.exists()
    source = Path(watchdog.__file__).with_name("compression_supervisor_recovery.py")
    recovery = watchdog.load("archive_handoff_original_supervisor", source)
    a.plan.update(runtime=str(a.root / "runtime"), holder_locks=[], input_hashes={})
    watchdog.write(a.root / "plan.json", a.plan)
    watchdog.write(a.root / "training_started.json", {"pid": 42})
    (a.root / "train.log").write_text("training/global_step:100.000\n")
    receipt = a.root / "recovery.json"
    watchdog.write(receipt, {"plan_sha256": watchdog.digest(a.root / "plan.json"),
                            "started_sha256": watchdog.digest(a.root / "training_started.json"),
                            "code_sha256": watchdog.digest(source), "launcher": {"pid": 42},
                            "step_id": 1, "previous_status": {"state": "training", "last_completed_step": 90}})
    frozen = SimpleNamespace(require_compute=lambda _: None, verify_runtime=lambda _: None,
                             now=watchdog.now, write=watchdog.write, holder_locks=lambda _: nullcontext(),
                             checkpoint_complete=control.checkpoint_complete, environment=lambda _: {},
                             spawn=lambda *args: pytest.fail("Must not start training"),
                             audit_rollouts=lambda *args: {"state": "verified"})
    monkeypatch.setattr(recovery, "controller", lambda _: frozen)
    monkeypatch.setattr(recovery, "process_identity", lambda _: None)
    monkeypatch.setattr(recovery, "slurm_step", lambda *args: {"state": "COMPLETED", "exit_code": "0:0"})
    monkeypatch.setattr(watchdog, "load", lambda *args: recovery)
    monkeypatch.setattr(watchdog, "verify_config", lambda _: None)
    monkeypatch.setattr(watchdog, "require_node", lambda _: None)
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *args, **kwargs:
                        SimpleNamespace(pid=43, poll=lambda: 0 if (a.root / "training_exit.json").exists() else None))
    a.config.update(eager_checkpoint_upload=True, recovery_receipt=str(receipt), code="watchdog.py",
                    stages={"supervisor": {"root": str(a.root)}})
    a.plan["python_bin"] = "/python"
    watchdog.write(a.root / "plan.json", a.plan)
    prior = watchdog.read(receipt)
    watchdog.write(receipt, {**prior, "plan_sha256": watchdog.digest(a.root / "plan.json")})
    watchdog.invoke(a.config, "supervisor", Path("config.json"))
    assert watchdog.read(a.root / "status.json")["state"] == "complete"
    assert watchdog.read(a.root / "training_exit.json")["slurm_accounting"]["state"] == "COMPLETED"
