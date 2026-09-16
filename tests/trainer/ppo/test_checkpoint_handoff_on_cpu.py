"""Exercise upload/cancel/launch safety using fake HF and Slurm backends."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_experiments.checkpoint_handoff import (
    Handoff,
    SafetyError,
    atomic_json,
    candidate_holders,
    checkpoint_complete,
    file_fingerprint,
    main,
    parse_job,
    release_source,
    remote_file_matches,
    upload_verified_checkpoint,
    validate_holder,
)


@pytest.fixture
def config(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return {
        "source_job": "1000", "source_step": "1000.5", "target_step": 150,
        "candidate_jobs": ["1001", "1002"], "owner": "testuser", "account": "test-account",
        "partition": "gpu", "holder_name": "holder", "holder_command": "/site/hold.slurm",
        "world_size": 4, "cpus": 288, "wandb_id": "same-run",
        "hf_repo_id": "owner/checkpoint-150", "hf_private": True,
        "experiment_dir": str(tmp_path / "experiment"), "repo_root": str(tmp_path),
        "resume_runner": str(tmp_path / "resume.sh"), "log_dir": str(log_dir),
        "state_path": str(log_dir / "state.json"), "receipt_path": str(log_dir / "receipt.json"),
        "watch_lock": str(log_dir / "watch.lock"), "upload_lock": str(log_dir / "upload.lock"),
    }


def make_checkpoint(config):
    experiment = Path(config["experiment_dir"])
    checkpoint = experiment / "global_step_150"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    for rank in range(4):
        for component in ("model", "optim", "extra_state"):
            (actor / f"{component}_world_size_4_rank_{rank}.pt").write_bytes(f"{component}-{rank}".encode())
    (checkpoint / "data.pt").write_bytes(b"dataloader state")
    (actor / "config.json").write_text("{}")
    (actor / "tokenizer_config.json").write_text("{}")
    (experiment / "latest_checkpointed_iteration.txt").write_text("150")
    (experiment / "wandb_id.txt").write_text(config["wandb_id"])
    return checkpoint


class FakeApi:
    def __init__(self):
        self.files = {}
        self.created = []
        self.commits = []
        self.fail_upload = False
        self.corrupt_remote = False

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def create_commit(self, **kwargs):
        if self.fail_upload:
            raise RuntimeError("upload failed")
        self.commits.append(kwargs)
        for operation in kwargs["operations"]:
            with operation.as_file() as stream:
                data = stream.read()
            digest = file_fingerprint(data)
            self.files[operation.path_in_repo] = SimpleNamespace(
                rfilename=operation.path_in_repo, size=digest["size"],
                blob_id=digest["git_blob"], lfs=SimpleNamespace(sha256=digest["sha256"]),
            )
        return SimpleNamespace(oid="verified-revision")

    def repo_info(self, **kwargs):
        if self.corrupt_remote and self.files:
            next(iter(self.files.values())).lfs.sha256 = "wrong"
        return SimpleNamespace(sha="verified-revision", siblings=list(self.files.values()))


def holder(config, job_id, state="RUNNING", start="2026-09-10T01:00:00"):
    return {
        "JobId": str(job_id), "JobState": state, "UserId": config["owner"] + "(123)",
        "Account": config["account"], "Partition": config["partition"],
        "JobName": config["holder_name"], "Command": config["holder_command"],
        "ReqTRES": "cpu=288,gres/gpu=4", "NodeList": "gpu001", "StartTime": start,
    }


class FakeSlurm:
    def __init__(self, config):
        self.jobs = {job: holder(config, job) for job in [config["source_job"], *config["candidate_jobs"]]}
        self.active = {config["source_job"]: {config["source_step"]}}
        self.cancelled = []
        self.busy = set()

    def job(self, job_id):
        return self.jobs[str(job_id)]

    def steps(self, job_id):
        return self.active.get(str(job_id), set()).copy()

    def cancel(self, target):
        self.cancelled.append(target)
        if "." in target:
            self.active[target.split(".")[0]].remove(target)
        else:
            self.jobs[target]["JobState"] = "CANCELLED"

    def idle(self, job):
        return job["JobId"] not in self.busy and not self.steps(job["JobId"])


def test_checkpoint_requires_all_ranks_metadata_and_completed_pointer(config):
    checkpoint = make_checkpoint(config)
    assert checkpoint_complete(config["experiment_dir"], 150, 4, "same-run")
    (checkpoint / "actor/optim_world_size_4_rank_2.pt").write_bytes(b"")
    assert not checkpoint_complete(config["experiment_dir"], 150, 4, "same-run")
    with pytest.raises(SafetyError, match="different W&B"):
        checkpoint_complete(config["experiment_dir"], 150, 4, "different-run")


def test_completed_pointer_is_required_even_if_files_exist(config):
    make_checkpoint(config)
    (Path(config["experiment_dir"]) / "latest_checkpointed_iteration.txt").write_text("100")
    assert not checkpoint_complete(config["experiment_dir"], 150, 4, "same-run")


def test_upload_checks_content_and_keeps_local_resume_files(config):
    checkpoint = make_checkpoint(config)
    api = FakeApi()
    receipt = upload_verified_checkpoint(config, api)
    assert checkpoint.is_dir()
    assert (checkpoint / "data.pt").is_file()
    assert receipt["revision"] == "verified-revision"
    assert "wandb_id.txt" in receipt["manifest"]
    assert "latest_checkpointed_iteration.txt" in receipt["manifest"]
    assert api.created[0]["private"] is True
    assert all(name.startswith("global_step_150/") or name in {"wandb_id.txt", "latest_checkpointed_iteration.txt"} for name in receipt["manifest"])


def test_same_size_corruption_fails_verification_and_never_cancels(config):
    make_checkpoint(config)
    api = FakeApi()
    api.corrupt_remote = True
    slurm = FakeSlurm(config)
    controller = Handoff(config, api, slurm)
    with pytest.raises(RuntimeError, match="verification failed"):
        controller.tick()
    assert not Path(config["receipt_path"]).exists()
    assert slurm.cancelled == []


@pytest.mark.parametrize("resume_after_release", [True, False])
def test_failed_upload_never_cancels_or_creates_receipt(config, resume_after_release):
    config["resume_after_release"] = resume_after_release
    make_checkpoint(config)
    api = FakeApi()
    api.fail_upload = True
    slurm = FakeSlurm(config)
    with pytest.raises(RuntimeError, match="upload failed"):
        Handoff(config, api, slurm).tick()
    assert slurm.cancelled == []
    assert not Path(config["receipt_path"]).exists()


def test_existing_conflicting_repo_is_not_overwritten(config):
    make_checkpoint(config)
    api = FakeApi()
    api.files["unrelated.bin"] = SimpleNamespace(rfilename="unrelated.bin", size=1, lfs=None, blob_id="wrong")
    with pytest.raises(SafetyError, match="conflicting HF content"):
        upload_verified_checkpoint(config, api)
    assert api.commits == []


def test_release_orders_verified_upload_then_step_then_allocation(config):
    make_checkpoint(config)
    api = FakeApi()
    slurm = FakeSlurm(config)
    controller = Handoff(config, api, slurm)
    assert controller.tick() is False
    assert Path(config["receipt_path"]).is_file()
    assert slurm.cancelled == []
    controller.tick()
    assert slurm.cancelled == ["1000.5"]
    controller.tick()
    assert slurm.cancelled == ["1000.5", "1000"]
    controller.tick()
    assert controller.state["source_released"] is True
    assert Path(config["experiment_dir"], "global_step_150").exists()


def test_source_with_other_training_is_not_cancelled(config):
    make_checkpoint(config)
    api = FakeApi()
    receipt = upload_verified_checkpoint(config, api)
    slurm = FakeSlurm(config)
    slurm.active["1000"].add("1000.6")
    with pytest.raises(SafetyError, match="another training step"):
        release_source(config, receipt, api, slurm)
    assert slurm.cancelled == []


@pytest.mark.parametrize("candidate_jobs", [[], ["1001", "1002"]])
def test_release_only_finishes_without_using_other_holders(config, monkeypatch, candidate_jobs):
    config.update(resume_after_release=False, candidate_jobs=candidate_jobs)
    config.pop("resume_runner")
    make_checkpoint(config)
    api = FakeApi()
    slurm = FakeSlurm(config)
    controller = Handoff(config, api, slurm)
    original_job = slurm.job

    def source_job_only(job_id):
        assert job_id == config["source_job"]
        return original_job(job_id)

    def forbidden(*args, **kwargs):
        pytest.fail("Release-only controller must not inspect or launch continuation candidates")

    monkeypatch.setattr(slurm, "job", source_job_only)
    monkeypatch.setattr("qwen3_experiments.checkpoint_handoff.candidate_holders", forbidden)
    monkeypatch.setattr("qwen3_experiments.checkpoint_handoff.subprocess.Popen", forbidden)
    assert controller.tick() is False  # Upload, verify, persist receipt.
    assert slurm.cancelled == []
    assert Path(config["receipt_path"]).is_file()
    assert controller.tick() is False
    assert slurm.cancelled == ["1000.5"]
    assert controller.tick() is False
    assert slurm.cancelled == ["1000.5", "1000"]
    assert controller.tick() is True
    assert controller.state["phase"] == "RELEASE_COMPLETE"
    assert controller.state["source_released"] is True
    assert not controller.state.get("launch_requested")
    assert Path(config["experiment_dir"], "global_step_150").is_dir()

    # Persisted completion also exits without starting anything after a restart.
    restarted = Handoff(config, api, slurm)
    assert restarted.tick() is True
    assert slurm.cancelled == ["1000.5", "1000"]


def test_release_only_rejects_direct_launch_and_existing_launch_intent(config):
    config["resume_after_release"] = False
    controller = Handoff(config, FakeApi(), FakeSlurm(config))
    with pytest.raises(SafetyError, match="continuation is disabled"):
        controller.launch(holder(config, "1001"))
    controller.state.update(source_released=True, launch_requested=True)
    controller.update("LAUNCHING", "prior launch needs inspection")
    restarted = Handoff(config, controller.api, controller.slurm)
    with pytest.raises(SafetyError, match="already requested"):
        restarted.tick()
    assert controller.slurm.cancelled == []


def test_release_only_preflight_needs_no_runner_or_candidate_queries(config, monkeypatch):
    config["resume_after_release"] = False
    config.pop("resume_runner")
    api = FakeApi()
    api.whoami = lambda: {"name": "owner"}
    slurm = FakeSlurm(config)
    original_job = slurm.job

    def source_job_only(job_id):
        assert job_id == config["source_job"]
        return original_job(job_id)

    monkeypatch.setattr(slurm, "job", source_job_only)
    monkeypatch.setattr("qwen3_experiments.checkpoint_handoff.Slurm", lambda: slurm)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
    config_path = Path(config["log_dir"]) / "config.json"
    atomic_json(config_path, config)
    monkeypatch.setattr("sys.argv", ["checkpoint_handoff.py", str(config_path), "--check"])
    main()
    assert slurm.cancelled == []
    assert api.created == []
    assert api.commits == []


def test_receipt_rechecked_before_cancelling(config):
    make_checkpoint(config)
    api = FakeApi()
    receipt = upload_verified_checkpoint(config, api)
    slurm = FakeSlurm(config)
    api.corrupt_remote = True
    with pytest.raises(RuntimeError, match="verification failed"):
        release_source(config, receipt, api, slurm)
    assert slurm.cancelled == []


def test_candidate_selection_skips_busy_and_prefers_earliest_allocation(config):
    slurm = FakeSlurm(config)
    slurm.jobs["1002"]["StartTime"] = "2026-09-10T00:00:00"
    assert [job["JobId"] for job in candidate_holders(config, slurm)] == ["1002", "1001"]
    slurm.busy.add("1002")
    assert [job["JobId"] for job in candidate_holders(config, slurm)] == ["1001"]
    slurm.jobs["1001"]["JobState"] = "PENDING"
    assert candidate_holders(config, slurm) == []
    assert slurm.cancelled == []


def test_launch_intent_prevents_automatic_duplicate_after_controller_restart(config):
    make_checkpoint(config)
    api = FakeApi()
    receipt = upload_verified_checkpoint(config, api)
    atomic_json(config["receipt_path"], receipt)
    controller = Handoff(config, api, FakeSlurm(config))
    controller.state.update(source_released=True, launch_requested=True)
    controller.update("LAUNCHING", "intent persisted")
    restarted = Handoff(config, api, FakeSlurm(config))
    with pytest.raises(SafetyError, match="prior launch intent"):
        restarted.tick()


def test_holder_identity_is_validated_before_use(config):
    job = holder(config, "1000")
    validate_holder(job, config, "1000")
    job["UserId"] = "someone_else(42)"
    with pytest.raises(SafetyError, match="owner"):
        validate_holder(job, config, "1000")


def test_git_blob_and_lfs_hashes_are_both_supported():
    expected = file_fingerprint(b"some data")
    assert remote_file_matches({"size": 9, "blob_id": expected["git_blob"], "lfs": None}, expected)
    assert remote_file_matches({"size": 9, "lfs": {"sha256": expected["sha256"]}}, expected)
    assert not remote_file_matches({"size": 9, "blob_id": "wrong", "lfs": None}, expected)


def test_slurm_record_parser_preserves_command_and_identity():
    parsed = parse_job("JobId=1000 UserId=testuser(123) JobState=RUNNING Command=/site/hold.slurm ReqTRES=cpu=288,gres/gpu=4")
    assert parsed["UserId"] == "testuser(123)"
    assert parsed["Command"] == "/site/hold.slurm"


def test_missing_checkpoint_waits_without_any_external_mutation(config):
    slurm = FakeSlurm(config)
    api = FakeApi()
    controller = Handoff(config, api, slurm)
    controller.tick()
    assert controller.state["phase"] == "WAITING_CHECKPOINT"
    assert slurm.cancelled == []
    assert api.created == []
    assert not Path(config["receipt_path"]).exists()
    assert json.loads(Path(config["state_path"]).read_text())["phase"] == "WAITING_CHECKPOINT"


def test_only_one_idle_holder_is_launched_and_same_run_is_confirmed(config, monkeypatch):
    make_checkpoint(config)
    api = FakeApi()
    atomic_json(config["receipt_path"], upload_verified_checkpoint(config, api))
    slurm = FakeSlurm(config)
    controller = Handoff(config, api, slurm)
    controller.state["source_released"] = True
    calls = []

    def fake_popen(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=12345, poll=lambda: None)

    monkeypatch.setattr("qwen3_experiments.checkpoint_handoff.subprocess.Popen", fake_popen)
    controller.tick()
    assert len(calls) == 1
    assert "--jobid=1001" in calls[0]
    assert calls[0][-1] == "1001"
    assert "--gpus-per-task=4" in calls[0]
    assert controller.state["selected_job"] == "1001"
    controller.tick()
    assert len(calls) == 1
    Path(controller.state["training_log"]).write_text(
        "Setting global step to 150\nruns/same-run\n"
        "'save_freq': 50\n 'test_freq': 250\n 'val_before_train': False\n150/1272\n"
    )
    controller.tick()
    assert controller.state["startup_confirmed"] is True
    assert len(calls) == 1
    assert slurm.cancelled == []
    controller.launch_lock.close()


def test_changed_local_checkpoint_blocks_release(config):
    checkpoint = make_checkpoint(config)
    api = FakeApi()
    receipt = upload_verified_checkpoint(config, api)
    (checkpoint / "data.pt").write_bytes(b"different dataloader state")
    slurm = FakeSlurm(config)
    with pytest.raises(SafetyError, match="changed after upload"):
        release_source(config, receipt, api, slurm)
    assert slurm.cancelled == []


def test_local_checkpoint_must_remain_immutable_during_upload(config):
    checkpoint = make_checkpoint(config)

    class MutatingApi(FakeApi):
        def create_commit(self, **kwargs):
            result = super().create_commit(**kwargs)
            (checkpoint / "data.pt").write_bytes(b"state changed during upload")
            return result

    with pytest.raises(SafetyError, match="changed during upload"):
        upload_verified_checkpoint(config, MutatingApi())
