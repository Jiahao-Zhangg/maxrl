"""Budget extensions preserve completed inputs and wait for existing GPU work."""

import errno
from pathlib import Path

import pytest

from qwen3_experiments import compression_budget_followup as followup


def predecessor(tmp_path):
    root = tmp_path / "previous"
    followup.write(root / "plan.json", {"models": ["previous"]})
    followup.write(root / "status.json", {"state": "complete"})
    followup.write(root / "queue_status.json", {"state": "complete"})
    followup.write(root / "report/metrics.json", [])
    followup.write(root / "report/audit.json", {"complete": True, "points": 35,
                   "all_rollout_ledgers_verified": True, "metrics_sha256": followup.digest(root / "report/metrics.json")})
    return root, {"dependency": {"root": str(root), "plan_sha256": followup.digest(root / "plan.json"),
                                 "points": 35, "queue_required": True}}


def test_no_gpu_handoff_until_full_predecessor_audit(tmp_path):
    root, plan = predecessor(tmp_path)
    assert followup.dependency_ready(plan)
    for name in ("status.json", "queue_status.json"):
        followup.write(root / name, {"state": "running"})
        assert not followup.dependency_ready(plan)
        followup.write(root / name, {"state": "complete"})
    audit = followup.read(root / "report/audit.json")
    followup.write(root / "report/audit.json", {**audit, "points": 34})
    with pytest.raises(ValueError, match="complete expected"):
        followup.dependency_ready(plan)


def test_changed_predecessor_plan_or_results_are_rejected(tmp_path):
    root, plan = predecessor(tmp_path)
    followup.write(root / "report/metrics.json", ["changed"])
    with pytest.raises(ValueError, match="complete expected"):
        followup.dependency_ready(plan)
    followup.write(root / "plan.json", {"other": True})
    with pytest.raises(ValueError, match="Predecessor plan changed"):
        followup.dependency_ready(plan)


@pytest.mark.parametrize("error", [errno.ENOSPC, errno.EDQUOT])
def test_full_status_disk_keeps_record_locally_and_retries(tmp_path, monkeypatch, error):
    root, scratch = tmp_path / "run", tmp_path / "local"
    calls = []

    def transient(path, value):
        if Path(path) == root / "status.json":
            calls.append(value)
            if len(calls) == 1:
                raise OSError(error, "full")
        followup.write(path, value)

    monkeypatch.setattr(followup.time, "sleep", lambda _: None)
    followup.protected_writer(root, scratch, transient)(root / "status.json", {"state": "running"})
    assert len(calls) == 2
    assert followup.read(scratch / "control_mirrors/status.json") == followup.read(root / "status.json")


def test_prepared_model_cannot_change_identity_or_weights(tmp_path):
    root = tmp_path / "stage"
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"pinned weights")
    spec = {"repo": "owner/model", "revision": "a" * 40, "path": str(model)}
    plan = {"models": {"er": spec}}
    followup.write(root / "plan.json", plan)
    manifest = {"plan_sha256": followup.digest(root / "plan.json"), "models": {
        "er": {**spec, "files_sha256": {"model.safetensors": followup.digest(model / "model.safetensors")}}}}
    manifest["fingerprint"] = followup.mini.fingerprint(manifest)
    followup.write(root / "execution_manifest.json", manifest)
    assert followup.staged_manifest(root, plan) == manifest
    (model / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Prepared model changed"):
        followup.staged_manifest(root, plan)


def test_cache_cleanup_refuses_external_or_shared_model_path(tmp_path, monkeypatch):
    root = tmp_path / "suite"
    stage_root = root / "stages/er"
    cache = tmp_path / "other-job-cache"
    cache.mkdir()
    (cache / "weights").write_bytes(b"needed by other job")
    suite = {"scratch": str(tmp_path / "owned")}
    stage = {"key": "er", "root": str(stage_root), "points": 30, "owned_model_cache": str(cache)}
    monkeypatch.setattr(followup, "stage_complete", lambda *args: True)
    monkeypatch.setattr(followup, "active_stage_processes", lambda *args: [])
    monkeypatch.setattr(followup.mini, "verify_plan", lambda _: {"models": {}})
    with pytest.raises(ValueError, match="private model cache"):
        followup.cleanup_stage(root, suite, stage)
    assert (cache / "weights").read_bytes() == b"needed by other job"


def test_live_worker_prevents_cache_cleanup(tmp_path, monkeypatch):
    stage = {"key": "er", "root": str(tmp_path / "stage"), "points": 30}
    monkeypatch.setattr(followup, "stage_complete", lambda *args: True)
    monkeypatch.setattr(followup, "active_stage_processes", lambda *args: [{"pid": 42}])
    monkeypatch.setattr(followup.mini, "verify_plan", lambda _: pytest.fail("Cannot touch an active stage"))
    followup.cleanup_stage(tmp_path, {}, stage)


def test_combined_report_requires_every_new_and_reused_point(tmp_path, monkeypatch):
    followup.write(tmp_path / "reused_minerva.json", {"metrics": []})
    stage = tmp_path / "stage"
    followup.write(stage / "report/metrics.json", [])
    plan = {"stages": [{"root": str(stage), "key": "er", "points": 30}],
            "all_datasets": ["minervamath", "math500"], "budgets": [8192], "new_points": 1}
    monkeypatch.setattr(followup, "stage_complete", lambda *args: True)
    with pytest.raises(ValueError, match="missing or duplicate"):
        followup.combined_report(tmp_path, plan)


@pytest.mark.parametrize("active", [False, True])
def test_queue_does_not_launch_over_predecessor_or_existing_stage(tmp_path, monkeypatch, active):
    root = tmp_path / "queue"
    root.mkdir()
    stage = root / "stage"
    plan = {"job_id": "7", "scratch": str(tmp_path / "local"), "new_points": 30, "filesystems": {},
            "stages": [{"root": str(stage), "key": "er", "points": 30}]}
    monkeypatch.setattr(followup, "verify_suite", lambda _: plan)
    monkeypatch.setattr(followup, "stage_complete", lambda *args: False)
    monkeypatch.setattr(followup.mini, "verify_plan", lambda _: {"dependency": {"root": "/previous"}})
    monkeypatch.setattr(followup, "dependency_ready", lambda _: active)
    monkeypatch.setattr(followup, "active_stage_processes", lambda *args: [{"pid": 42}] if active else [])
    monkeypatch.setattr(followup.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Duplicate GPU launch"))

    class EndLoop(Exception):
        pass

    def stop(_):
        raise EndLoop

    monkeypatch.setattr(followup.time, "sleep", stop)
    with pytest.raises(EndLoop):
        followup.queue(root)
    assert followup.read(root / "queue_status.json")["state"] == (
        "running_or_waiting_for_gpus" if active else "waiting_for_predecessor")
