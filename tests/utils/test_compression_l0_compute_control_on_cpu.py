"""Fail-closed handoff, GPU coordination and complete rollout archival checks."""

import fcntl
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qwen3_experiments import compression_l0_compute_control as control
from verl.utils.rollout_dataset import dump_rollout_step


def predecessor(tmp_path):
    plan = {"predecessor_status": str(tmp_path / "training.json"),
            "predecessor_pipeline_status": str(tmp_path / "pipeline.json"),
            "predecessor_receipt": str(tmp_path / "receipt.json"), "predecessor_repo": "owner/old-step_100"}
    status = {"variant": "per_context_rb_l0_0", "adv_estimator": "fixed_n_rb_offset_cost_aware_marginrl",
              "cost_offset_tokens": 0, "total_steps": 100, "state": "training", "last_completed_step": 99}
    control.write(plan["predecessor_status"], status)
    control.write(plan["predecessor_pipeline_status"], {"state": "resuming_training"})
    return plan, status


def test_handoff_requires_successful_exit_pipeline_and_verified_final_checkpoint(tmp_path):
    plan, status = predecessor(tmp_path)
    assert not control.predecessor_ready(plan)
    status.update(state="complete", exit_code=0, last_completed_step=100)
    control.write(plan["predecessor_status"], status)
    assert not control.predecessor_ready(plan)
    control.write(plan["predecessor_pipeline_status"], {"state": "complete"})
    assert not control.predecessor_ready(plan)
    receipt = {"checkpoint": "global_step_100", "repo_id": plan["predecessor_repo"],
               "state": "verified", "verified_at": "2026-09-22", "remote_commit": "a" * 40,
               "files": {f"actor/model_world_size_8_rank_{rank}.pt": {"size": 1, "sha256": "b" * 64}
                         for rank in range(8)}}
    control.write(plan["predecessor_receipt"], receipt)
    assert control.predecessor_ready(plan)
    receipt["repo_id"] = "owner/wrong-run-step_100"
    control.write(plan["predecessor_receipt"], receipt)
    with pytest.raises(AssertionError, match="another run"):
        control.predecessor_ready(plan)


@pytest.mark.parametrize("failed_file", ["predecessor_status", "predecessor_pipeline_status"])
def test_failed_predecessor_never_unlocks_new_training(tmp_path, failed_file):
    plan, _ = predecessor(tmp_path)
    control.write(plan[failed_file], {"state": "failed"})
    with pytest.raises(RuntimeError, match="has not been launched"):
        control.predecessor_ready(plan)


def test_lock_contention_releases_partially_acquired_locks(tmp_path):
    paths = [tmp_path / "first.lock", tmp_path / "occupied.lock"]
    with paths[1].open("a") as occupied:
        fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError), control.holder_locks(paths):
            pytest.fail("The occupied allocation must not be entered")
        with paths[0].open("a") as released:
            fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)


def rollout_plan(tmp_path):
    plan = {"rollout_dir": str(tmp_path), "rollout_hf_repo": "owner/run-rollouts",
            "total_steps": 2, "rows_per_step": 3}
    for step in (1, 2):
        dump_rollout_step(tmp_path, step=step, inputs=["question"] * 3,
                          outputs=["<think>full reasoning</think>answer"] * 3, scores=[0, 1, 0])
    siblings = [SimpleNamespace(rfilename=p.relative_to(tmp_path).as_posix(), size=p.stat().st_size,
                               lfs={"sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
                for p in (tmp_path / "data").glob("*")]
    api = SimpleNamespace(repo_info=lambda **kwargs: SimpleNamespace(sha="c" * 40, siblings=siblings))
    return plan, api, siblings


def test_rollout_audit_checks_every_step_row_and_remote_hash(tmp_path):
    plan, api, _ = rollout_plan(tmp_path)
    audit = control.audit_rollouts(plan, api)
    assert audit["state"] == "verified" and audit["num_rollouts"] == 6 and audit["num_steps"] == 2


@pytest.mark.parametrize("damage", ["missing_step", "wrong_count", "missing_remote", "wrong_hash", "short_shard"])
def test_rollout_audit_rejects_partial_or_corrupt_collection(tmp_path, damage):
    plan, api, siblings = rollout_plan(tmp_path)
    manifest_path = tmp_path / "rollout_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "missing_step":
        del manifest["steps"]["2"]
    elif damage == "wrong_count":
        manifest["steps"]["2"]["num_rollouts"] = 2
    elif damage == "missing_remote":
        siblings.pop()
    elif damage == "wrong_hash":
        siblings[0].lfs["sha256"] = "0" * 64
    else:
        dump_rollout_step(tmp_path, step=2, inputs=["q"] * 2, outputs=["a"] * 2, scores=[0] * 2)
        for item in siblings:
            path = tmp_path / item.rfilename
            item.size = path.stat().st_size
            item.lfs["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    # A lying manifest cannot hide a missing rollout from the full-shard audit.
    control.write(manifest_path, manifest)
    with pytest.raises(ValueError):
        control.audit_rollouts(plan, api)


def test_training_launcher_enables_all_rollouts_without_a_wrapper():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["bash", str(root / control.LAUNCHER)], cwd=root,
                            env={"PATH": "/usr/bin:/bin", "DRY_RUN": "1", "PYTHON_BIN": "/not-installed",
                                 "L0_ROLLOUT_HF_REPO": "owner/all-rollouts"},
                            text=True, capture_output=True, check=True)
    assert "trainer.rollout_dataset.enabled=true" in result.stdout
    assert "trainer.rollout_dataset.hub_repo_id=owner/all-rollouts" in result.stdout
    assert "trainer.rollout_dataset.private=false" in result.stdout
    assert "trainer.total_training_steps=100" in result.stdout


def test_training_stays_in_the_requested_allocation():
    command = control.training_command({"job_id": "146103", "node": "compute-18", "runtime": "/run with spaces"})
    assert command[0] == "srun"
    assert "--jobid=146103" in command and "--nodelist=compute-18" in command and "--gres=gpu:8" in command
    assert command[-1] == "/run with spaces/" + control.LAUNCHER


def evaluation_predecessor(tmp_path, monkeypatch):
    previous = {"job_id": "146102", "model_repo": "owner/grpo-step_100", "training_kind": "grpo",
                "final_step": 100, "questions": 1819, "total_responses": 7276,
                "caps": [1024, 2048, 4096, 8192, 16384, 32768]}
    control.write(tmp_path / "plan.json", previous)
    plan = {"job_id": "146102", "predecessor_kind": "evaluation", "predecessor_evaluation_root": str(tmp_path),
            "predecessor_plan_sha256": control.digest(tmp_path / "plan.json"),
            "predecessor_repo": previous["model_repo"]}
    monkeypatch.setattr(control.evaluation, "evaluator", lambda root: None)
    monkeypatch.setattr(control.evaluation, "verify_plan", lambda root, base: previous)
    control.write(tmp_path / "queue_status.json", {"state": "complete", "completed_responses": 7276})
    control.write(tmp_path / "status.json", {"state": "complete", "completed_responses": 7276})
    control.write(tmp_path / "training_completion.json", {
        "variant": "grpo", "adv_estimator": "grpo", "archive_exit_code": 0,
        "total_steps": 100, "state": "complete", "exit_code": 0, "last_completed_step": 100})
    control.write(tmp_path / "final_checkpoint_receipt.json", {
        "checkpoint": "global_step_100", "repo_id": previous["model_repo"], "state": "archived_and_deleted",
        "verified_at": "2026-09-24", "remote_commit": "a" * 40,
        "files": {f"actor/model_world_size_8_rank_{rank}.pt": {"size": 1, "sha256": "b" * 64}
                  for rank in range(8)}})
    (tmp_path / "report").mkdir()
    (tmp_path / "report/README.md").write_text("Completed evaluation\n")
    control.write(tmp_path / "report/per_sample.json", [])
    control.write(tmp_path / "report/metrics.json", [])
    control.write(tmp_path / "report/audit.json", {
        "complete": True, "grader_errors": {}, "all_budgets_regraded": True,
        "questions": 1819, "responses_verified": 7276, "budget_points": 54,
        "per_sample_sha256": control.digest(tmp_path / "report/per_sample.json"),
        "metrics_sha256": control.digest(tmp_path / "report/metrics.json")})
    (tmp_path / "exit_status").write_text("0\n")
    return plan


def test_grpo_evaluation_must_finish_before_compression_training(tmp_path, monkeypatch):
    plan = evaluation_predecessor(tmp_path, monkeypatch)
    for phase in ("waiting_for_training_and_verified_final_checkpoint", "waiting_for_all_eight_gpus",
                  "waiting_for_available_gpus_or_evaluating"):
        control.write(tmp_path / "queue_status.json", {"state": phase})
        assert not control.predecessor_ready(plan)
    control.write(tmp_path / "queue_status.json", {"state": "complete", "completed_responses": 7276})
    control.write(tmp_path / "status.json", {"state": "grading_prefixes"})
    assert not control.predecessor_ready(plan)
    control.write(tmp_path / "status.json", {"state": "complete", "completed_responses": 7276})
    assert control.predecessor_ready(plan)


@pytest.mark.parametrize("damage", ["queue_failed", "grading_failed", "missing_response", "bad_exit",
                                   "changed_report", "changed_plan", "wrong_model", "missing_report"])
def test_bad_evaluation_cannot_release_compression_training(tmp_path, monkeypatch, damage):
    plan = evaluation_predecessor(tmp_path, monkeypatch)
    if damage == "queue_failed":
        control.write(tmp_path / "queue_status.json", {"state": "failed"})
    elif damage == "grading_failed":
        control.write(tmp_path / "status.json", {"state": "failed"})
    elif damage == "missing_response":
        audit = control.read(tmp_path / "report/audit.json")
        control.write(tmp_path / "report/audit.json", {**audit, "responses_verified": 7275})
    elif damage == "bad_exit":
        (tmp_path / "exit_status").write_text("1\n")
    elif damage == "changed_report":
        control.write(tmp_path / "report/metrics.json", {"changed": True})
    elif damage == "changed_plan":
        control.write(tmp_path / "plan.json", {"changed": True})
    elif damage == "wrong_model":
        receipt = control.read(tmp_path / "final_checkpoint_receipt.json")
        control.write(tmp_path / "final_checkpoint_receipt.json", {**receipt, "repo_id": "owner/wrong-model"})
    else:
        (tmp_path / "report/README.md").unlink()
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        control.predecessor_ready(plan)


def test_l4096_recipe_rejects_unrequested_hyperparameter_changes():
    before = {"algorithm": {"adv_estimator": "fixed_n_rb_offset_cost_aware_marginrl", "cost_offset_tokens": 0},
              "reward_model": {"reward_kwargs": {"check_eos": True, "score_after_thinking": True}},
              "trainer": {"experiment_name": "l0", "total_training_steps": 100}}
    after = copy.deepcopy(before)
    after["algorithm"]["cost_offset_tokens"] = 4096
    after["reward_model"]["reward_kwargs"]["check_eos"] = False
    after["trainer"]["experiment_name"] = "l4096_no_eos"
    assert len(control.compare_training_recipe(before, after)) == 3
    after["trainer"]["total_training_steps"] = 200
    with pytest.raises(ValueError, match="Unexpected change"):
        control.compare_training_recipe(before, after)


def test_l4096_launcher_uses_after_thinking_without_an_eos_gate():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["bash", str(root / control.L4096_LAUNCHER)], cwd=root,
                            env={"PATH": "/usr/bin:/bin", "DRY_RUN": "1", "PYTHON_BIN": "/not-installed"},
                            text=True, capture_output=True, check=True)
    assert "algorithm.cost_offset_tokens=4096" in result.stdout
    assert "reward_model.reward_kwargs.check_eos=false" in result.stdout
    assert "reward_model.reward_kwargs.score_after_thinking=True" in result.stdout
    assert "actor_rollout_ref.rollout.force_eos=False" in result.stdout
    assert "actor_rollout_ref.rollout.ignore_eos=False" in result.stdout
    assert "trainer.rollout_dataset.private=false" in result.stdout


@pytest.mark.parametrize("estimator,offset,variant", [
    (control.RB_ESTIMATOR, 4096, "per_context_rb_l0_4096"), ("maxrl", 0, "maxrl"),
])
def test_supervisor_can_finish_training_without_scheduling_another_evaluation(tmp_path, monkeypatch, estimator, offset, variant):
    predecessor = tmp_path / "predecessor.json"
    control.write(predecessor, {"state": "evaluating"})
    plan = {"job_id": "146102", "node": "compute", "output_root": str(tmp_path), "runtime": str(tmp_path),
            "data_dir": str(tmp_path / "data"), "checkpoint_dir": str(tmp_path / "checkpoints"),
            "rollout_dir": str(tmp_path / "rollouts"), "rollout_hf_repo": "owner/rollouts",
            "ray_dir": str(tmp_path / "ray"), "python_bin": sys.executable, "hf_repo_prefix": "owner/l4096",
            "predecessor_status": str(predecessor), "input_hashes": {}, "holder_locks": [str(tmp_path / "gpu.lock")],
            "evaluate_after_training": False, "cost_offset_tokens": offset, "variant": variant,
            "adv_estimator": estimator,
            "cleanup_predecessor_models_before_training": estimator == "maxrl",
            "grading": {"check_eos": False, "score_after_thinking": True}}
    flags = {"ready": False, "archive_done": False, "launched": False, "cleaned": estimator != "maxrl"}
    monkeypatch.setattr(control, "require_compute", lambda job: "compute")
    monkeypatch.setattr(control, "verify_runtime", lambda runtime: None)
    monkeypatch.setattr(control, "predecessor_ready", lambda plan: flags["ready"])
    monkeypatch.setattr(control, "gpu_idle", lambda: True)
    monkeypatch.setattr(control, "checkpoint_complete", lambda *args: True)
    monkeypatch.setattr(control, "audit_rollouts", lambda *args: {"state": "verified"})

    def clean(_):
        assert flags["ready"] and not flags["launched"]
        flags["cleaned"] = True
        return {"state": "complete"}

    monkeypatch.setattr(control, "cleanup_before_training", clean)

    def advance(seconds):
        assert not flags["launched"]
        flags["ready"] = True

    def spawn(plan, command, log):
        assert command == "monitor", "A further evaluation was not requested"
        return SimpleNamespace(pid=1, returncode=0, poll=lambda: 0 if flags["archive_done"] else None)

    def finish():
        flags["archive_done"] = True
        return 0

    def launch(*args, **kwargs):
        assert flags["ready"] and flags["cleaned"] and not flags["launched"]
        assert kwargs["env"]["L0_CHECK_EOS"] == "false"
        assert kwargs["env"]["L0_COST_OFFSET_TOKENS"] == str(offset)
        assert kwargs["env"]["L0_ADV_ESTIMATOR"] == estimator
        flags["launched"] = True
        return SimpleNamespace(pid=2, returncode=0, poll=finish)

    monkeypatch.setattr(control.time, "sleep", advance)
    monkeypatch.setattr(control, "spawn", spawn)
    monkeypatch.setattr(control.subprocess, "Popen", launch)
    assert control.supervise(plan) == 0
    assert flags["launched"]
    assert control.read(tmp_path / "supervisor_status.json")["state"] == "complete"
    assert control.read(tmp_path / "status.json")["cost_offset_tokens"] == offset
    assert control.read(tmp_path / "status.json")["adv_estimator"] == estimator
    assert control.read(tmp_path / "training_exit.json")["exit_code"] == 0
