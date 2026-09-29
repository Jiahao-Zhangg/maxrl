"""Disk recovery must retain live jobs and protect the other allocation."""

from contextlib import nullcontext
import errno
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from qwen3_experiments import compression_pipeline_watchdog as guard


def config(tmp_path):
    root, local, other = (tmp_path / name for name in ("run", "node", "other"))
    for folder in (root, local, other):
        folder.mkdir()
    return {"control_root": str(local), "stages": {"training": {"root": str(root)}},
            "protected_paths": [str(other)], "project_path": str(tmp_path), "routes": [],
            "backup_min_free_bytes": 0, "backup_bytes_per_cycle": 1024, "interval_seconds": 0}


def route(cfg, directory=True):
    root, local = Path(cfg["stages"]["training"]["root"]), Path(cfg["control_root"])
    value = {"source": str(root / "rollouts"), "local": str(local / "rollouts"),
             "backup": str(root / "backups/rollouts"), "directory": directory}
    cfg["routes"] = [value]
    return value


def test_training_only_watchdog_does_not_touch_evaluation(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.update(job_id="146103", node="compute", warning_bytes=100, critical_bytes=50,
               autonomous_recovery=True)
    cfg["stages"]["training"]["status"] = "status.json"
    root = Path(cfg["stages"]["training"]["root"])
    guard.write(root / "status.json", {"state": "waiting_for_predecessor", "updated_at": guard.now()})
    monkeypatch.setattr(guard, "require_node", lambda _: None)
    monkeypatch.setattr(guard, "disk", lambda _: {"available_bytes": 80})
    monkeypatch.setattr(guard, "stage_process", lambda *args: {"pid": 42})
    monkeypatch.setattr(guard, "stale_controller", lambda *args: None)
    monkeypatch.setattr(guard, "stalled_evaluation", lambda _: pytest.fail("Unrelated evaluation was inspected"))
    result = guard.tick(cfg, tmp_path / "config.json")
    assert result["issues"] == [] and result["controls"]["training"]["alive"]
    assert result["disk_pressure"] == "low"


def test_owned_outputs_cannot_escape_via_parent_or_symlink(tmp_path):
    cfg = config(tmp_path)
    root = Path(cfg["stages"]["training"]["root"])
    assert guard.owned_path(cfg, root / "status.json") == root / "status.json"
    for path in (tmp_path / "other/data", root / "../other/data"):
        with pytest.raises(ValueError):
            guard.owned_path(cfg, path)
    (root / "link").symlink_to(tmp_path / "other", target_is_directory=True)
    with pytest.raises(ValueError):
        guard.owned_path(cfg, root / "link/checkpoint")


def test_future_outputs_are_local_but_existing_outputs_are_never_moved(tmp_path):
    cfg = config(tmp_path)
    item = route(cfg)
    guard.prepare_empty_routes(cfg)
    guard.prepare_empty_routes(cfg)
    (Path(item["source"]) / "one").write_text("saved")
    assert (Path(item["local"]) / "one").read_text() == "saved"
    Path(item["source"]).unlink()
    Path(item["source"]).mkdir()
    with pytest.raises(ValueError, match="existing live output"):
        guard.prepare_empty_routes(cfg)


def test_disk_cleanup_removes_only_verified_duplicate_backup(tmp_path):
    cfg = config(tmp_path)
    item = route(cfg)
    guard.prepare_empty_routes(cfg)
    data = Path(item["local"]) / "step.json"
    data.write_text("original")
    os.utime(data, (time.time() - 200, time.time() - 200))
    assert guard.backups(cfg, "normal")["copied_bytes"] == 8
    backup = Path(item["backup"]) / "step.json"
    assert backup.read_text() == "original"
    backup.write_text("changed!")
    assert guard.backups(cfg, "critical")["reclaimed_bytes"] == 0
    assert backup.exists()
    backup.write_text("original")
    assert guard.backups(cfg, "critical")["reclaimed_bytes"] == 8
    assert not backup.exists() and data.read_text() == "original"


def test_logger_symlinks_do_not_block_backups_or_expose_external_files(tmp_path):
    cfg = config(tmp_path)
    item = route(cfg)
    item["skip_symlinks"] = True
    guard.prepare_empty_routes(cfg)
    local = Path(item["local"])
    data = local / "events.log"
    data.write_text("events")
    os.utime(data, (1, 1))
    external = tmp_path / "other/private.log"
    external.write_text("keep private")
    (local / "latest.log").symlink_to(data.name)
    (local / "debug-core.log").symlink_to(external)
    result = guard.backups(cfg, "normal")
    assert result["copied_bytes"] == 6 and len(result["skipped_symlinks"]) == 2
    assert sorted(p.name for p in Path(item["backup"]).iterdir()) == ["events.log"]
    assert guard.backups(cfg, "critical")["reclaimed_bytes"] == 6
    assert data.read_text() == "events" and external.read_text() == "keep private"
    assert (local / "debug-core.log").is_symlink() and (local / "latest.log").is_symlink()


def test_non_logger_routes_still_reject_unexpected_symlinks(tmp_path):
    cfg = config(tmp_path)
    item = route(cfg)
    guard.prepare_empty_routes(cfg)
    (Path(item["local"]) / "unexpected").symlink_to(tmp_path / "other")
    with pytest.raises(ValueError, match="Unexpected link"):
        guard.backups(cfg, "normal")


@pytest.mark.parametrize("error", [errno.EDQUOT, errno.ENOSPC])
def test_quota_retries_keep_a_local_record_and_do_not_advance(tmp_path, monkeypatch, error):
    cfg = config(tmp_path)
    path = Path(cfg["stages"]["training"]["root"]) / "status.json"
    calls = []

    def writer(path, value):
        calls.append(value)
        if len(calls) == 1:
            raise OSError(error, "full")
        guard.write(path, value)

    monkeypatch.setattr(guard, "require_node", lambda _: None)
    guard.durable_writer(writer, cfg)(path, {"state": "training"})
    assert len(calls) == 2 and guard.read(path) == {"state": "training"}
    mirror = next((Path(cfg["control_root"]) / "write_mirrors").glob("*.json"))
    assert guard.read(mirror)["value"] == guard.read(path)


def test_non_storage_failures_are_not_silently_retried(tmp_path):
    cfg = config(tmp_path)
    path = Path(cfg["stages"]["training"]["root"]) / "status.json"

    def fail(*_):
        raise OSError(errno.EACCES, "permission")

    with pytest.raises(OSError, match="permission"):
        guard.durable_writer(fail, cfg)(path, {})


@pytest.mark.parametrize("changed", ["start_ticks", "cgroup", "command"])
def test_pid_reuse_or_other_node_is_never_adopted(tmp_path, monkeypatch, changed):
    cfg = config(tmp_path)
    root = Path(cfg["stages"]["training"]["root"])
    cfg["control_cgroup"] = "allocation-ours"
    process = {"pid": 42, "uid": os.getuid(), "start_ticks": 123, "cgroup": "allocation-ours", "command": [str(root)]}
    guard.write(root / "launch.json", {"pid": 42, "identity": process})
    replacement = {**process, changed: {"start_ticks": 456, "cgroup": "allocation-other", "command": ["other"]}[changed]}
    monkeypatch.setattr(guard, "identity", lambda _: replacement)
    with pytest.raises(ValueError):
        guard.stage_process(cfg, "training")


def test_completed_or_live_queue_is_not_restarted(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    root = Path(cfg["stages"]["training"]["root"])
    cfg["stages"]["training"].update(status="status.json")
    monkeypatch.setattr(guard, "verify", lambda _: None)
    monkeypatch.setattr(guard, "require_node", lambda _: None)
    monkeypatch.setattr(guard, "locks", lambda _: nullcontext())
    monkeypatch.setattr(guard, "stage_process", lambda *_: {"pid": 42})
    monkeypatch.setattr(guard, "spawn", lambda *_: pytest.fail("Must not start a duplicate"))
    assert guard.restart(cfg, tmp_path / "config", "training") is None
    monkeypatch.setattr(guard, "stage_process", lambda *_: None)
    guard.write(root / "status.json", {"state": "complete"})
    assert guard.restart(cfg, tmp_path / "config", "training") is None


def test_missing_training_identity_never_starts_second_trainer(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    root = Path(cfg["stages"]["training"]["root"])
    cfg["stages"]["training"].update(status="status.json")
    guard.write(root / "status.json", {"state": "training"})
    guard.write(root / "training_started.json", {"pid": 42})
    monkeypatch.setattr(guard, "verify", lambda _: None)
    monkeypatch.setattr(guard, "require_node", lambda _: None)
    monkeypatch.setattr(guard, "locks", lambda _: nullcontext())
    monkeypatch.setattr(guard, "stage_process", lambda *_: None)
    monkeypatch.setattr(guard, "spawn", lambda *_: pytest.fail("Second trainer forbidden"))
    with pytest.raises(ValueError, match="refusing a new trainer"):
        guard.restart(cfg, tmp_path / "config", "training")


@pytest.mark.parametrize("success", [True, False])
def test_adoption_observes_original_training_and_requires_real_success(tmp_path, monkeypatch, success):
    cfg = config(tmp_path)
    root = Path(cfg["stages"]["training"]["root"])
    plan = {"output_root": str(root), "checkpoint_dir": str(tmp_path / "checkpoints"),
            "holder_locks": [], "job_id": "1", "checkpoint_steps": [10, 100]}
    guard.write(root / "training_started.json", {"pid": 42})
    guard.write(root / "status.json", {"state": "training", "last_completed_step": 10})
    launcher = {"pid": 42, "start_ticks": 123}
    guard.write(Path(cfg["control_root"]) / "training_identity.json", {
        "launcher": launcher, "step_id": "7", "started_sha256": guard.digest(root / "training_started.json")})
    processes, steps, events = iter([launcher, None]), iter([10, 100]), []

    def successful(record):
        if not success:
            raise RuntimeError("Original Slurm step failed")
        return True

    def audit(*args):
        assert guard.read(root / "training_exit.json")["exit_code"] == 0
        events.append("verified_rollouts")
        return {"state": "verified"}

    recovery = SimpleNamespace(same_process=lambda before, after: before == after,
                               last_completed_step=lambda _: next(steps), successful_exit=successful,
                               slurm_step=lambda *a: {"state": "COMPLETED" if success else "FAILED"},
                               archived_steps=lambda *a: [10, 100])
    control = SimpleNamespace(verify_training_inputs=lambda _: None, holder_locks=lambda _: nullcontext(), write=guard.write,
                              audit_rollouts=audit, checkpoint_complete=lambda *a: True)

    def spawn(plan, command, log):
        assert command == "monitor"
        events.append(command)
        return SimpleNamespace(poll=lambda: None)

    control.spawn = spawn
    monkeypatch.setattr(guard, "load_recovery", lambda _: recovery)
    monkeypatch.setattr(guard, "require_node", lambda _: None)
    monkeypatch.setattr(guard, "identity", lambda _: next(processes))
    monkeypatch.setattr(guard.time, "sleep", lambda _: None)
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **kw: pytest.fail("Adoption launched another process"))
    if success:
        guard.adopt_training(cfg, control, plan)
        assert guard.read(root / "status.json")["state"] == "complete"
        assert events == ["monitor", "verified_rollouts"]
    else:
        with pytest.raises(RuntimeError, match="Original Slurm step failed"):
            guard.adopt_training(cfg, control, plan)
        assert not (root / "training_exit.json").exists() and events == ["monitor"]
