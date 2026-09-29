"""Disk recovery stays within one run and never duplicates live GPU work."""

import errno
import gzip
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from qwen3_experiments import compute_disk_watchdog as watchdog


@pytest.fixture
def configured(tmp_path):
    root, control = tmp_path / "run", tmp_path / "control"
    root.mkdir()
    control.mkdir()
    watchdog.write(root / "plan.json", {"job_id": "7", "node": "compute-a"})
    stage = root / "evaluation"
    stage.mkdir()
    watchdog.write(stage / "plan.json", {"frozen": True})
    watchdog.write(stage / "launch.json", {"pid": 123})
    watchdog.write(stage / "queue_status.json", {"pid": 123, "state": "waiting", "updated_at": watchdog.now()})
    return {
        "training_root": str(root), "control_root": str(control), "node": "compute-a", "job_id": "7",
        "pinned_files": {str(root / "plan.json"): watchdog.digest(root / "plan.json")},
        "interval_seconds": 30, "python_bin": "/python", "code": str(tmp_path / "watchdog.py"),
        "hf_home": str(tmp_path / "hf"), "control_cgroup": "our-allocation",
        "stages": {"nine": {"root": str(stage), "runtime": str(stage), "receipt": "launch.json",
                            "status": "queue_status.json", "launch_lock": "launch.lock",
                            "process_lock": "queue.lock", "restart_states": ["waiting"]}},
    }


def test_config_rejects_other_node_and_other_run(configured):
    watchdog.verify_config(configured)
    with pytest.raises(ValueError, match="another training allocation"):
        watchdog.verify_config({**configured, "node": "compute-b"})
    configured["stages"]["nine"]["root"] = str(Path(configured["training_root"]).parent / "other-run")
    with pytest.raises(ValueError):
        watchdog.verify_config(configured)


def test_changed_frozen_plan_is_never_recovered(configured):
    watchdog.write(Path(configured["training_root"]) / "plan.json", {"job_id": "7", "node": "compute-a", "changed": True})
    with pytest.raises(ValueError, match="input changed"):
        watchdog.verify_config(configured)


def test_require_node_refuses_another_host_before_slurm_call(configured, monkeypatch):
    monkeypatch.setattr(watchdog.socket, "gethostname", lambda: "compute-b")
    monkeypatch.setattr(watchdog.subprocess, "check_output", lambda *a, **k: pytest.fail("Must not inspect other allocation"))
    with pytest.raises(ValueError, match="original compute node"):
        watchdog.require_node(configured)


@pytest.mark.parametrize("error", [errno.EDQUOT, errno.ENOSPC])
def test_full_disk_keeps_queue_at_same_write_and_preserves_mirror(configured, monkeypatch, error):
    root = Path(configured["training_root"])
    calls, sleeps = [], []

    def intermittent(path, value):
        calls.append((path, value))
        if len(calls) == 1:
            raise OSError(error, "full")
        watchdog.write(path, value)

    monkeypatch.setattr(watchdog.time, "sleep", sleeps.append)
    monkeypatch.setattr(watchdog, "require_node", lambda config: None)
    write = watchdog.durable_writer(intermittent, configured)
    write(root / "queue_status.json", {"state": "waiting"})
    assert len(calls) == 2 and sleeps == [30]
    assert watchdog.read(root / "queue_status.json") == {"state": "waiting"}
    assert watchdog.read(Path(configured["control_root"]) / "write_mirrors/queue_status.json") == {"state": "waiting"}


def test_other_write_errors_are_not_hidden_or_retried(configured):
    def denied(*args):
        raise PermissionError(errno.EACCES, "denied")

    with pytest.raises(PermissionError):
        watchdog.durable_writer(denied, configured)(Path(configured["training_root"]) / "status.json", {})


@pytest.mark.parametrize("case", ["live_controller", "live_launcher", "live_worker", "failed_stage"])
def test_recovery_never_overlaps_existing_work(configured, monkeypatch, case):
    stage = Path(configured["stages"]["nine"]["root"])
    state = watchdog.read(stage / "queue_status.json")
    if case == "live_launcher":
        state["launcher_pid"] = 456
    if case == "failed_stage":
        state["state"] = "failed"
    watchdog.write(stage / "queue_status.json", state)
    monkeypatch.setattr(watchdog, "require_node", lambda config: None)
    monkeypatch.setattr(watchdog, "identity", lambda pid: {} if (
        (case == "live_controller" and pid == 123) or (case == "live_launcher" and pid == 456)) else None)
    monkeypatch.setattr(watchdog, "active_stage_work", lambda *a: case == "live_worker")
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *a, **k: pytest.fail("Duplicate work launched"))
    if case == "live_controller":
        assert watchdog.restart(configured, "nine", Path("config.json")) is None
    else:
        with pytest.raises(ValueError):
            watchdog.restart(configured, "nine", Path("config.json"))


def test_dead_waiting_queue_is_recovered_from_pinned_code(configured, monkeypatch):
    monkeypatch.setattr(watchdog, "require_node", lambda config: None)
    monkeypatch.setattr(watchdog, "identity", lambda pid: None)
    monkeypatch.setattr(watchdog, "active_stage_work", lambda *a: False)
    launched = []
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda command, **kwargs: (
        launched.append((command, kwargs)) or SimpleNamespace(pid=456)))
    receipt = watchdog.restart(configured, "nine", Path("config.json"))
    assert receipt["pid"] == 456
    assert launched[0][0][3:] == ["queue", "--config", "config.json", "--kind", "nine", "--output-root",
                                  configured["stages"]["nine"]["root"]]
    assert launched[0][1]["start_new_session"]
    assert "trainer" not in " ".join(launched[0][0])
    assert watchdog.read(Path(configured["stages"]["nine"]["root"]) / "launch.json")["pid"] == 456


def test_relocation_preserves_bytes_and_skips_current_step_and_external_links(configured):
    root = Path(configured["training_root"]) / "rollout_dataset"
    (root / "data").mkdir(parents=True)
    steps = {}
    for step in (1, 2):
        path = root / f"data/step_{step:06d}.jsonl.gz"
        with gzip.open(path, "wt") as stream:
            stream.write('{"step": ' + str(step) + ', "output": "answer"}\n')
        os.utime(path, (0, 0))
        steps[str(step)] = {"file": str(path.relative_to(root)), "num_rollouts": 1}
    watchdog.write(root / "rollout_manifest.json", {"steps": steps})
    first = root / steps["1"]["file"]
    checksum = watchdog.digest(first)
    result = watchdog.relocate_closed_rollout(configured, completed_step=2)
    assert result["sha256"] == checksum and first.is_symlink() and watchdog.digest(first) == checksum
    assert not (root / steps["2"]["file"]).is_symlink()
    assert watchdog.relocate_closed_rollout(configured, completed_step=2) is None


def test_bad_rollout_manifest_cannot_delete_or_move_source(configured):
    root = Path(configured["training_root"]) / "rollout_dataset"
    path = root / "data/step_000001.jsonl.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as stream:
        stream.write('{"step": 2}\n')
    os.utime(path, (0, 0))
    watchdog.write(root / "rollout_manifest.json", {"steps": {"1": {"file": "data/step_000001.jsonl.gz", "num_rollouts": 1}}})
    with pytest.raises(ValueError, match="another step"):
        watchdog.relocate_closed_rollout(configured, 2)
    assert path.is_file() and not path.is_symlink()


def test_live_pid_from_other_allocation_is_never_adopted(configured, monkeypatch):
    monkeypatch.setattr(watchdog, "identity", lambda pid: {
        "uid": os.getuid(), "cgroup": "other-allocation", "command": [configured["stages"]["nine"]["root"]]})
    with pytest.raises(ValueError, match="this allocation"):
        watchdog.stage_status(configured, "nine")


def test_completion_requires_all_35_budget_points(configured):
    base = Path(configured["training_root"])
    configured["stages"]["budget"] = {"root": str(base / "budget")}
    watchdog.write(Path(configured["stages"]["nine"]["root"]) / "report/audit.json", {"complete": True, "responses_verified": 7276})
    watchdog.write(base / "budget/report/audit.json", {"complete": True, "points": 5})
    states = {key: {"state": "complete"} for key in configured["stages"]}
    assert not watchdog.all_complete(configured, states)
    watchdog.write(base / "budget/report/audit.json", {"complete": True, "points": 35})
    assert watchdog.all_complete(configured, states)


@pytest.mark.parametrize("full", ["home", "local", "both"])
def test_full_monitor_status_disk_does_not_exit_watchdog(configured, monkeypatch, full):
    configured["local_control_root"] = str(Path(configured["control_root"]).parent / "local-control")
    original = watchdog.write

    def limited(path, value):
        on_local = Path(path).is_relative_to(configured["local_control_root"])
        if full == "both" or on_local == (full == "local"):
            raise OSError(errno.ENOSPC, "status disk full")
        original(path, value)

    monkeypatch.setattr(watchdog, "write", limited)
    saved = watchdog.control_record(configured, "status.json", {"state": "monitoring"})
    assert len(saved) == (0 if full == "both" else 1)
    for path in saved:
        assert watchdog.read(path) == {"state": "monitoring"}


def test_monitor_wrapper_retries_quota_and_preserves_exit_code(configured, monkeypatch):
    configured["stages"]["supervisor"] = {"root": configured["training_root"]}
    root = Path(configured["training_root"])
    watchdog.write(root / "plan.json", {"runtime": str(root / "runtime")})
    impl = SimpleNamespace()
    calls = []

    def initially_full(path, value):
        calls.append(value)
        if len(calls) == 1:
            raise OSError(errno.EDQUOT, "quota")
        watchdog.write(path, value)

    def monitor(plan):
        impl.write(root / "hf_checkpoint_archive/status.json", {"state": "monitoring"})
        return 1  # Nonzero monitor exit must remain nonzero.

    monitor.__module__ = "test_frozen_monitor_impl"
    impl.write = initially_full
    monkeypatch.setitem(sys.modules, monitor.__module__, impl)
    control = SimpleNamespace(monitor=monitor, verify_runtime=lambda runtime: None)
    monkeypatch.setattr(watchdog.importlib, "import_module", lambda name: control)
    monkeypatch.setattr(watchdog, "verify_config", lambda config: None)
    monkeypatch.setattr(watchdog, "require_node", lambda config: None)
    monkeypatch.setattr(watchdog.time, "sleep", lambda seconds: None)
    assert watchdog.invoke(configured, "monitor", Path("config.json")) == 1
    assert len(calls) == 2
    mirror = Path(configured["control_root"]) / "write_mirrors/hf_checkpoint_archive/status.json"
    assert watchdog.read(mirror)["state"] == "monitoring"


def test_supervisor_spawns_protected_monitor_without_starting_training(configured, monkeypatch):
    root = Path(configured["training_root"])
    configured["stages"]["supervisor"] = {"root": str(root)}
    configured["recovery_receipt"] = str(root / "recovery/receipt.json")
    control = SimpleNamespace(environment=lambda plan: {}, write=watchdog.write,
                              spawn=lambda *args: pytest.fail("Unexpected unprotected spawn"))
    module = SimpleNamespace(controller=lambda plan: control)
    launches = []

    def watch(root, receipt):
        active = module.controller({})
        child = active.spawn({"python_bin": "/python", "runtime": str(root)}, "monitor", root / "upload.log")
        assert child.pid == 456

    module.watch = watch
    monkeypatch.setattr(watchdog, "load", lambda name, path: module)
    monkeypatch.setattr(watchdog, "verify_config", lambda config: None)
    monkeypatch.setattr(watchdog, "require_node", lambda config: None)
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda command, **kwargs: (
        launches.append(command) or SimpleNamespace(pid=456)))
    watchdog.invoke(configured, "supervisor", Path("config.json"))
    assert launches == [["/python", "-u", configured["code"], "queue", "--config", "config.json",
                         "--kind", "monitor", "--output-root", str(root)]]


def test_control_record_route_preserves_link_and_writes_independent_disk(configured):
    root = Path(configured["training_root"])
    store = root.parent / "independent"
    target = store / "control/status.json"
    target.parent.mkdir(parents=True)
    link = root / "status.json"
    link.symlink_to(target)
    configured.update(independent_storage_root=str(store), control_file_routes={str(link): str(target)})
    watchdog.project_write(configured, link, {"state": "training"})
    assert link.is_symlink() and watchdog.read(link) == {"state": "training"}
    assert watchdog.read(target) == {"state": "training"}
    link.unlink()
    link.symlink_to(root.parent / "another-job/status.json")
    with pytest.raises(ValueError, match="storage route changed"):
        watchdog.project_write(configured, link, {"state": "wrong"})


def test_migrated_evaluation_worker_is_still_detected(configured, monkeypatch):
    root = Path(configured["stages"]["nine"]["root"])
    destination = root.parent.parent / "independent_evaluation"
    root.rename(destination)
    root.symlink_to(destination, target_is_directory=True)
    monkeypatch.setattr(watchdog, "allocation_processes", lambda config: {999})
    monkeypatch.setattr(watchdog, "identity", lambda pid: {"command": ["python", "run", "--output-root", str(destination)]})
    assert watchdog.active_stage_work(configured, configured["stages"]["nine"])
