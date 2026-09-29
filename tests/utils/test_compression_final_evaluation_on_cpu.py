"""The final queue preserves model order, checkpoint identity and evaluation data."""

import os
from pathlib import Path

import pytest

from qwen3_experiments import compression_final_evaluation as suite


def test_finish_both_f_cov_evaluations_and_cleanup_before_maxrl(tmp_path, monkeypatch):
    models = [{"key": k} for k in suite.MODEL_ORDER]
    plan = {"models": models}
    complete = set()
    monkeypatch.setattr(suite.shared, "nine_complete", lambda root: str(root) in complete)
    monkeypatch.setattr(suite, "budget_complete", lambda root: str(root) in complete)
    for model in models:
        nroot, broot = suite.stage_roots(tmp_path, model["key"])
        assert suite.next_action(tmp_path, plan) == (model, "nine")
        complete.add(str(nroot))
        assert suite.next_action(tmp_path, plan) == (model, "budget")
        complete.add(str(broot))
        assert suite.next_action(tmp_path, plan) == (model, "cleanup")
        suite.write(tmp_path / "model_cache_cleanup" / f"{model['key']}.json", {"state": "complete"})
    assert suite.next_action(tmp_path, plan) is None


@pytest.mark.parametrize("damage", [None, "repo", "step", "rank", "unfinished"])
def test_bind_only_verified_final_checkpoint(tmp_path, monkeypatch, damage):
    model = {"key": "f_cov_step100", "repo": "owner/f-cov-step_100", "training_root": str(tmp_path / "training"),
             "training_plan_sha256": "a" * 64}
    receipt = {"repo_id": model["repo"], "checkpoint": "global_step_100", "state": "archived_and_deleted",
               "remote_commit": "b" * 40, "files": {f"actor/model_world_size_8_rank_{r}.pt":
                {"size": 10, "sha256": "c" * 64} for r in range(8)}}
    if damage == "repo":
        receipt["repo_id"] = "owner/other-step_100"
    elif damage == "step":
        receipt["checkpoint"] = "global_step_90"
    elif damage == "rank":
        receipt["files"].pop("actor/model_world_size_8_rank_7.pt")
    suite.write(tmp_path / "training/hf_checkpoint_archive/receipts/global_step_100.json", receipt)
    monkeypatch.setattr(suite.training, "training_predecessor_ready", lambda _: damage != "unfinished")
    plan = {"job_id": "146103", "node": "compute"}
    if damage:
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            suite.bind_checkpoint(tmp_path, plan, model)
    else:
        spec, actual = suite.bind_checkpoint(tmp_path, plan, model)
        assert spec["revision"] == "b" * 40 and actual == receipt
        assert suite.bind_checkpoint(tmp_path, plan, model) == (spec, actual)


def completed_model(tmp_path):
    root, scratch = tmp_path / "results", tmp_path / "node"
    model = {"key": "f_cov_step100", "repo": "owner/f-cov-step_100"}
    plan = {"models": [model], "scratch": str(scratch)}
    nroot, broot = suite.stage_roots(root, model["key"])
    suite.write(nroot / "status.json", {"state": "complete"})
    for file in ("metrics.json", "per_sample.json"):
        suite.write(nroot / "report" / file, [])
    suite.write(nroot / "report/audit.json", {"complete": True, "questions": 1819, "responses_verified": 7276,
        "budget_points": 54, "metrics_sha256": suite.digest(nroot / "report/metrics.json"),
        "per_sample_sha256": suite.digest(nroot / "report/per_sample.json")})
    suite.write(nroot / "plan.json", {})
    bplan = {"models": {model["key"]: model}, "datasets": [{"key": k} for k in suite.DATASETS],
             "budgets": suite.mini.BUDGETS, "parent_eval_root": str(nroot),
             "parent_eval_plan_sha256": suite.digest(nroot / "plan.json"), "frozen_files": {}}
    suite.write(broot / "plan.json", bplan)
    manifest = {"plan_sha256": suite.digest(broot / "plan.json"), "models": {model["key"]: {**model, "revision": "a" * 40}},
                "fingerprint": "manifest"}
    suite.write(broot / "execution_manifest.json", manifest)
    for key, budget, dataset in suite.mini.evaluation_points(bplan):
        folder = suite.mini.point_directory(broot, key, budget, dataset)
        for name in ("rollouts", "prompts"):
            suite.write(folder / f"{name}.json", ["retained results"])
        suite.write(folder / "summary.json", {"state": "complete", "seed": 0, "ledger_audit": "passed",
            "num_prompts": suite.mini.DATASETS[dataset][1], "identity": suite.mini.point_identity(key, budget, manifest, dataset),
            "artifacts": {name: {"file": f"{name}.json", "size": (folder / f"{name}.json").stat().st_size,
                                 "sha256": suite.digest(folder / f"{name}.json")} for name in ("rollouts", "prompts")}})
    suite.write(broot / "report/metrics.json", [])
    suite.write(broot / "report/audit.json", {"complete": True, "points": 35, "all_rollout_ledgers_verified": True,
                                             "metrics_sha256": suite.digest(broot / "report/metrics.json")})
    suite.write(broot / "status.json", {"state": "complete"})
    for path in (scratch / "models" / model["key"] / "model/weights", scratch / "stages" / model["key"] / "tmp/inference"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"reconstructable model cache")
    (nroot / "model").symlink_to(scratch / "models" / model["key"] / "model", target_is_directory=True)
    return root, plan, model, nroot, broot


@pytest.mark.parametrize("damage", [None, "unfinished_budget", "changed_artifact", "cache_link"])
def test_model_cleanup_requires_both_audits_and_keeps_results(tmp_path, damage):
    root, plan, model, nroot, broot = completed_model(tmp_path)
    cache = Path(plan["scratch"]) / "models" / model["key"]
    other = tmp_path / "146102/keep.txt"
    other.parent.mkdir()
    other.write_text("another allocation")
    if damage == "unfinished_budget":
        suite.write(broot / "status.json", {"state": "running"})
    elif damage == "changed_artifact":
        path = suite.mini.point_directory(broot, model["key"], 65536, "aime26") / "rollouts.json"
        path.write_text("changed")
    elif damage == "cache_link":
        (cache / "external").symlink_to(other)
    if damage:
        with pytest.raises(ValueError):
            suite.cleanup_model(root, plan, model)
        assert cache.exists()
    else:
        suite.cleanup_model(root, plan, model)
        assert not cache.exists() and not (nroot / "model").is_symlink()
        assert suite.cleanup_complete(root, model["key"])
        assert suite.budget_complete(broot) and suite.shared.nine_complete(nroot)
        assert len(list((broot / "results").rglob("rollouts.json"))) == 35
    assert other.read_text() == "another allocation"


def test_cleanup_ignores_its_own_launcher_but_detects_other_evaluation_workers(tmp_path, monkeypatch):
    key = "f_cov_step100"
    plan = {"output_root": str(tmp_path), "scratch": str(tmp_path / "scratch")}
    processes = {99991: {"pid": 99991, "uid": os.getuid(), "command": [suite.MODULE, "run-phase", "--model", key]},
                 99992: {"pid": 99992, "uid": os.getuid(), "command": ["worker", str(tmp_path / "stages" / key / "nine_datasets")]}}
    monkeypatch.setattr(suite.storage, "allocation_processes", lambda _: list(processes))
    monkeypatch.setattr(suite.storage, "identity", processes.get)
    assert len(suite.active_workers(plan, {"key": key})) == 2
    assert [p["pid"] for p in suite.active_workers(plan, {"key": key}, include_launchers=False)] == [99992]
    assert not suite.active_workers(plan, {"key": "maxrl_step100"})


@pytest.mark.parametrize("valid", [True, False])
def test_storage_recovery_moves_only_checked_closed_artifacts_and_restores(tmp_path, monkeypatch, valid):
    root = tmp_path / "results"
    model = {"key": "f_cov_step100"}
    plan = {"scratch": str(tmp_path / "node"), "models": [model], "storage_critical_bytes": 50, "storage_warning_bytes": 100}
    nroot, _ = suite.stage_roots(root, model["key"])
    suite.write(nroot / "manifest.json", {"model": "pinned"})
    path = nroot / "responses/one.json.gz"
    suite.write(path, {"response": "retained exactly"})
    before = path.read_bytes()
    suite.write(nroot / "responses/one.receipt.json", {"file": path.name, "size": len(before),
        "sha256": suite.digest(path) if valid else "0" * 64, "manifest_sha256": suite.digest(nroot / "manifest.json")})
    os.utime(path, (1, 1))
    monkeypatch.setattr(suite.storage, "disk", lambda _: {"available_bytes": 20})
    if not valid:
        with pytest.raises(ValueError, match="Unverified"):
            suite.recover_storage(root, plan)
        assert not path.is_symlink() and path.read_bytes() == before
        return
    assert suite.recover_storage(root, plan)["reclaimed_bytes"] == len(before)
    assert path.is_symlink() and path.read_bytes() == before
    monkeypatch.setattr(suite.storage, "disk", lambda _: {"available_bytes": 200})
    assert suite.recover_storage(root, plan)["restored_bytes"] == len(before)
    assert not path.is_symlink() and path.read_bytes() == before
