"""Autonomous recovery must use evidence and keep other allocations untouched."""

from datetime import datetime, timezone
import os
from pathlib import Path
import signal

import pytest

from qwen3_experiments import compression_pipeline_watchdog as guard


def setup(tmp_path):
    root, local, other = (tmp_path / p for p in ("evaluation", "local", "other"))
    for path in (root, local, other):
        path.mkdir()
    config = {"job_id": "146103", "node": "ours", "control_cgroup": "our-cgroup", "control_root": str(local),
              "project_path": str(tmp_path), "warning_bytes": 0, "interval_seconds": 30,
              "protected_paths": [str(other)], "stages": {"evaluation": {"root": str(root), "status": "queue_status.json"}},
              "stall_no_progress_seconds": 90, "stall_idle_seconds": 60}
    plan = {"nine_root": str(root / "nine"), "minerva_root": str(root / "minerva"),
            "scratch": str(local / "models"), "models": []}
    guard.write(root / "plan.json", plan)
    return config, root, local


def response(config, root):
    nine = root / "nine"
    guard.write(nine / "manifest.json", {"model": "pinned"})
    path = nine / "responses/one.json.gz"
    path.parent.mkdir()
    path.write_bytes(b"completed response; preserve these bytes")
    receipt = path.with_name("one.receipt.json")
    guard.write(receipt, {"id": "one", "file": path.name, "size": path.stat().st_size,
                          "sha256": guard.digest(path), "manifest_sha256": guard.digest(nine / "manifest.json")})
    for p in (path, receipt):
        os.utime(p, (1, 1))
    return path, receipt


def test_disk_pressure_moves_verified_payload_and_restores_exact_bytes(tmp_path):
    config, root, local = setup(tmp_path)
    path, receipt = response(config, root)
    content, before = path.read_bytes(), receipt.read_bytes()
    result = guard.emergency_storage(config, "critical")
    assert result["offloaded_bytes"] == len(content)
    assert path.is_symlink() and path.read_bytes() == content
    assert receipt.read_bytes() == before
    guard.emergency_storage(config, "normal")
    assert not path.is_symlink() and path.read_bytes() == content
    assert receipt.read_bytes() == before
    assert not list((local / "emergency_payloads").iterdir())


@pytest.mark.parametrize("damage", ["hash", "manifest", "escape", "active"])
def test_emergency_offload_never_removes_unverified_or_active_outputs(tmp_path, damage):
    config, root, local = setup(tmp_path)
    path, receipt = response(config, root)
    info = guard.read(receipt)
    if damage == "active":
        os.utime(path, None)
        assert guard.emergency_storage(config, "critical")["offloaded_bytes"] == 0
    else:
        info[{"hash": "sha256", "manifest": "manifest_sha256", "escape": "file"}[damage]] = (
            "../../other/secret" if damage == "escape" else "0" * 64)
        guard.write(receipt, info)
        os.utime(receipt, (1, 1))
        with pytest.raises(ValueError):
            guard.emergency_storage(config, "critical")
    assert path.is_file() and not path.is_symlink()


def test_failed_link_creation_retains_shared_original(tmp_path, monkeypatch):
    config, root, local = setup(tmp_path)
    path, receipt = response(config, root)
    content = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("metadata disk full")

    monkeypatch.setattr(Path, "symlink_to", fail)
    with pytest.raises(OSError, match="disk full"):
        guard.emergency_storage(config, "critical")
    assert path.read_bytes() == content and not path.is_symlink()
    assert (local / "emergency_payloads" / path.name).read_bytes() == content


def test_progress_ignores_heartbeats_but_observes_generation_and_grading(tmp_path):
    config, root, _ = setup(tmp_path)
    status = root / "nine/status.json"
    guard.write(status, {"completed_responses": 20, "graded_questions": 0, "updated_at": 100})
    previous = guard.evaluation_progress(config)
    guard.write(status, {"completed_responses": 20, "graded_questions": 0, "updated_at": 200})
    assert guard.evaluation_progress(config) == previous
    guard.write(status, {"completed_responses": 20, "graded_questions": 25, "updated_at": 200})
    assert guard.evaluation_progress(config) != previous


@pytest.mark.parametrize("work", ["stalled", "gpu", "cpu", "progress", "sampling_gap"])
def test_only_sustained_idle_step_is_cancelled(tmp_path, monkeypatch, work):
    config, root, local = setup(tmp_path)
    clock = [1000]
    step = {"step": "146103.25", "phase": "nine", "launcher": {"pid": 42, "start_ticks": 10}}
    cancelled = []
    monkeypatch.setattr(guard, "evaluation_steps", lambda _: [step])
    monkeypatch.setattr(guard, "allocation_processes", lambda _: {43: "25", 44: "extern"})
    monkeypatch.setattr(guard, "process_cpu", lambda _: {"start_ticks": 10, "seconds": clock[0] if work == "cpu" else 1})
    monkeypatch.setattr(guard, "evaluation_progress", lambda _: str(clock[0]) if work == "progress" else "same")
    monkeypatch.setattr(guard.time, "time", lambda: clock[0])
    monkeypatch.setattr(guard.subprocess, "check_output", lambda *a, **k: "\n".join(["80" if work == "gpu" else "0"] * 8))
    monkeypatch.setattr(guard.subprocess, "run", lambda args, **kw: cancelled.append(args))
    monkeypatch.setattr(guard, "verify", lambda _: None)
    monkeypatch.setattr(guard, "require_node", lambda _: None)
    for _ in range(4):
        guard.stalled_evaluation(config)
        clock[0] += 200 if work == "sampling_gap" else 30
    assert cancelled == ([["scancel", "146103.25"]] if work == "stalled" else [])


@pytest.mark.parametrize("damage", [None, "other_job", "other_node", "other_launcher", "batch_step"])
def test_step_validation_rejects_wrong_job_node_or_process(tmp_path, monkeypatch, damage):
    config, root, _ = setup(tmp_path)
    fields = {"StepId": "146103.25", "State": "RUNNING", "UserId": str(os.getuid()), "NodeList": "ours",
              "Name": "polaris-nine-eval", "SrunHost:Pid": "ours:42"}
    command = ["srun", "--jobid=146103", "qwen3_experiments.polaris_checkpoint_evaluation",
               "run", "--output-root", str(root), "--phase", "nine"]
    if damage == "other_job":
        fields["StepId"] = "146102.25"
    elif damage == "other_node":
        fields["NodeList"] = "other"
    elif damage == "batch_step":
        fields["StepId"] = "146103.batch"
    elif damage == "other_launcher":
        command[command.index(str(root))] = "/another/run"
    monkeypatch.setattr(guard, "allocation_processes", lambda _: {42: "extern", 43: "25"})
    monkeypatch.setattr(guard, "identity", lambda _: {"pid": 42, "uid": os.getuid(), "command": command, "cgroup": "our-cgroup"})
    monkeypatch.setattr(guard.subprocess, "check_output", lambda *a, **k: " ".join(f"{k}={v}" for k, v in fields.items()))
    if damage is None:
        assert guard.evaluation_steps(config)[0]["step"] == "146103.25"
    else:
        with pytest.raises(ValueError):
            guard.evaluation_steps(config)


@pytest.mark.parametrize("condition", ["stale", "fresh", "pid_reused", "updated_status"])
def test_stale_controller_recovery_only_signals_its_verified_pid(tmp_path, monkeypatch, condition):
    config, root, _ = setup(tmp_path)
    process = {"pid": 42, "start_ticks": 10}
    stamp = datetime.fromtimestamp(9999 if condition == "fresh" else 1, timezone.utc).isoformat()
    status = {"pid": 42, "state": "running", "updated_at": stamp}
    guard.write(root / "queue_status.json", {**status, "state": "complete"} if condition == "updated_status" else status)
    monkeypatch.setattr(guard.time, "time", lambda: 10000)
    monkeypatch.setattr(guard, "verify", lambda _: None)
    monkeypatch.setattr(guard, "require_node", lambda _: None)
    monkeypatch.setattr(guard, "stage_process", lambda *a: process)
    monkeypatch.setattr(guard, "identity", lambda _: {**process, "start_ticks": 20} if condition == "pid_reused" else process)
    signals = []
    monkeypatch.setattr(guard.os, "kill", lambda pid, signum: signals.append((pid, signum)))
    guard.stale_controller(config, "evaluation", status, process)
    assert signals == ([(42, signal.SIGTERM)] if condition == "stale" else [])
