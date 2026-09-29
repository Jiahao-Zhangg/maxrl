"""Pinned Polaris checkpoint identities and ordered evaluation dependencies."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_experiments import polaris_checkpoint_evaluation as evaluation


def predecessor(tmp_path, monkeypatch):
    previous = tmp_path / "er"
    mroot = previous / "minerva"
    evaluation.write(previous / "plan.json", {"nine_root": str(previous / "nine"), "minerva_root": str(mroot)})
    evaluation.write(previous / "queue_status.json", {"state": "complete", "nine_responses": 7276, "minerva_points": 5})
    evaluation.write(mroot / "status.json", {"state": "complete"})
    evaluation.write(mroot / "report/metrics.json", ["verified metrics"])
    audit = {"complete": True, "points": 5, "questions_per_point": 272, "all_rollout_ledgers_verified": True,
             "metrics_sha256": evaluation.digest(mroot / "report/metrics.json")}
    evaluation.write(mroot / "report/audit.json", audit)
    monkeypatch.setattr(evaluation.shared, "nine_complete", lambda _: True)
    return {"predecessor_root": str(previous)}, previous, mroot


def test_queue_waits_for_both_current_er_evaluations(tmp_path, monkeypatch):
    plan, previous, _ = predecessor(tmp_path, monkeypatch)
    assert evaluation.predecessor_ready(plan)
    for state in ("waiting_for_er_training_and_archives", "running_or_waiting_for_gpus", "retrying"):
        evaluation.write(previous / "queue_status.json", {"state": state})
        assert not evaluation.predecessor_ready(plan)
    evaluation.write(previous / "queue_status.json", {"state": "failed"})
    with pytest.raises(RuntimeError, match="ER evaluation failed"):
        evaluation.predecessor_ready(plan)


@pytest.mark.parametrize("damage", ["nine", "count", "ledger", "hash", "incomplete"])
def test_partial_or_changed_predecessor_does_not_unlock_queue(tmp_path, monkeypatch, damage):
    plan, previous, mroot = predecessor(tmp_path, monkeypatch)
    if damage == "nine":
        monkeypatch.setattr(evaluation.shared, "nine_complete", lambda _: False)
    elif damage == "count":
        evaluation.write(previous / "queue_status.json", {"state": "complete", "nine_responses": 7276, "minerva_points": 4})
    elif damage == "hash":
        evaluation.write(mroot / "report/metrics.json", ["changed"])
    elif damage == "incomplete":
        evaluation.write(mroot / "status.json", {"state": "running"})
    else:
        audit = evaluation.read(mroot / "report/audit.json")
        evaluation.write(mroot / "report/audit.json", {**audit, "all_rollout_ledgers_verified": False})
    with pytest.raises(ValueError):
        evaluation.predecessor_ready(plan)


@pytest.mark.parametrize("model", evaluation.MODELS)
def test_each_minerva_shard_uses_its_requested_checkpoint(tmp_path, model):
    model = {**model, "revision": "a" * 40}
    plan = {"minerva_root": str(tmp_path)}
    root = tmp_path / model["key"]
    for folder in [root, *(root / "shards" / str(rank) for rank in range(8))]:
        evaluation.write(folder / "plan.json", {"models": {model["key"]: model}})
    receipt = {"repo": model["repo"], "revision": model["revision"], "merged_files": {"model.safetensors": "b" * 64}}
    evaluation.install_minerva_model(plan, model, Path("/node/model"), receipt)
    for folder in [root, *(root / "shards" / str(rank) for rank in range(8))]:
        manifest = evaluation.read(folder / "execution_manifest.json")
        assert list(manifest["models"]) == [model["key"]]
        assert manifest["models"][model["key"]]["repo"] == model["repo"]
        assert manifest["models"][model["key"]]["revision"] == "a" * 40
    with pytest.raises(ValueError, match="differs from the pinned"):
        evaluation.install_minerva_model(plan, model, Path("/node/model"), {**receipt, "revision": "c" * 40})


def test_nine_install_rejects_step80_or_another_model(tmp_path):
    plan = {"nine_root": str(tmp_path)}
    model = tmp_path / "checkpoint"
    model.mkdir()
    evaluation.write(tmp_path / "input_plan.json", {"model": {"repo": "owner/step_100", "revision": "a" * 40}})
    with pytest.raises(ValueError, match="identity mismatch"):
        evaluation.install_nine_model(plan, model, {"repo": "owner/step_80", "revision": "b" * 40})


def test_followup_runs_nine_then_fifteen_minerva_points(tmp_path, monkeypatch):
    plan = {"job_id": "146103", "node": "compute", "runtime": str(tmp_path), "python_bin": "python",
            "nine_root": str(tmp_path / "nine"), "minerva_root": str(tmp_path / "minerva")}
    ready, phases = [], []
    monkeypatch.setattr(evaluation, "require_compute", lambda _: "compute")
    monkeypatch.setattr(evaluation, "verify_plan", lambda _: plan)
    monkeypatch.setattr(evaluation.shared, "environment", lambda _: {})
    monkeypatch.setattr(evaluation, "predecessor_ready", lambda _: bool(ready))
    monkeypatch.setattr(evaluation.shared, "nine_complete", lambda _: "nine" in phases)

    def wait(_):
        if not ready:
            assert not phases
            ready.append(True)

    def child(command, **kwargs):
        phase = command[command.index("--phase") + 1]
        phases.append(phase)
        if phase == "minerva":
            assert phases == ["nine", "minerva"]
            evaluation.write(tmp_path / "minerva/report/audit.json", {"complete": True, "points": 15})
        return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)

    monkeypatch.setattr(evaluation.time, "sleep", wait)
    monkeypatch.setattr(evaluation.subprocess, "Popen", child)
    evaluation.queue(tmp_path, plan)
    assert phases == ["nine", "minerva"]
    assert evaluation.read(tmp_path / "queue_status.json")["minerva_points"] == 15


def test_models_finish_and_delete_in_order_and_resume_without_redownload(tmp_path, monkeypatch):
    models = [{**spec, "revision": "a" * 40} for spec in evaluation.MODELS]
    plan = {"job_id": "1", "holder_locks": [], "models": models, "nine_root": str(tmp_path / "nine"),
            "minerva_root": str(tmp_path / "minerva"), "cleanup_each_model_after_evaluation": True}
    events, complete = [], set()
    monkeypatch.setattr(evaluation, "require_compute", lambda _: None)
    monkeypatch.setattr(evaluation.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(evaluation, "predecessor_ready", lambda _: True)
    monkeypatch.setattr(evaluation.shared, "nine_complete", lambda _: True)
    monkeypatch.setattr(evaluation, "model_evaluation_complete", lambda p, s: s["key"] in complete)

    def prepare(root, plan, spec):
        assert all(other["key"] in complete for other in models[:models.index(spec)])
        events.append(("prepare", spec["key"]))
        return Path("/cache") / spec["key"], {}

    def run(plan, **kwargs):
        key = Path(plan["minerva_root"]).name
        events.append(("evaluate", key))
        complete.add(key)

    monkeypatch.setattr(evaluation, "prepared_model", prepare)
    monkeypatch.setattr(evaluation, "install_minerva_model", lambda *args: None)
    monkeypatch.setattr(evaluation.shared, "run_minerva", run)
    monkeypatch.setattr(evaluation, "cleanup_completed_model", lambda r, p, s: events.append(("delete", s["key"])))
    monkeypatch.setattr(evaluation, "combined_report", lambda _: events.append(("report", "all")))
    evaluation.run_phase(tmp_path, plan, "minerva")
    assert events == [(action, s["key"]) for s in models for action in ("prepare", "evaluate", "delete")] + [("report", "all")]
    events.clear()
    evaluation.run_phase(tmp_path, plan, "nine")
    evaluation.run_phase(tmp_path, plan, "minerva")
    assert events == [("delete", s["key"]) for s in models] + [("report", "all")]


@pytest.mark.parametrize("damage", ["unfinished", "nine_unfinished", "protected", "symlink", "checkpoint"])
def test_per_model_cleanup_preserves_incomplete_or_protected_cache(tmp_path, monkeypatch, damage):
    spec = {**evaluation.MODELS[0], "revision": "a" * 40}
    root, scratch = tmp_path / "eval", tmp_path / "scratch"
    folder = scratch / "models" / spec["key"]
    folder.mkdir(parents=True)
    (folder / "weights").write_text("keep")
    plan = {"models": [spec], "minerva_root": str(root / "minerva"), "nine_root": str(root / "nine"),
            "scratch": str(scratch), "base_model": str(tmp_path / "base")}
    evaluation.write(root / "plan.json", plan)
    evaluation.write(root / "prepared_models" / f"{spec['key']}.json", spec)
    monkeypatch.setattr(evaluation, "model_evaluation_complete", lambda *args: damage != "unfinished")
    monkeypatch.setattr(evaluation.shared, "nine_complete", lambda _: damage != "nine_unfinished")
    if damage == "protected":
        plan["protected_cache_paths"] = [str(folder)]
    elif damage == "symlink":
        (folder / "other_input").symlink_to(tmp_path / "other")
    elif damage == "checkpoint":
        evaluation.write(root / "prepared_models" / f"{spec['key']}.json", {**spec, "revision": "b" * 40})
    with pytest.raises(ValueError):
        evaluation.cleanup_completed_model(root, plan, spec)
    assert (folder / "weights").read_text() == "keep"


def test_per_model_cleanup_is_idempotent_and_retains_results(tmp_path, monkeypatch):
    spec = {**evaluation.MODELS[1], "revision": "a" * 40}
    root, scratch = tmp_path / "eval", tmp_path / "scratch"
    folder = scratch / "models" / spec["key"]
    folder.mkdir(parents=True)
    (folder / "weights").write_text("weights")
    plan = {"models": [spec], "minerva_root": str(root / "minerva"),
            "scratch": str(scratch), "base_model": str(tmp_path / "base")}
    evaluation.write(root / "plan.json", plan)
    evaluation.write(root / "prepared_models" / f"{spec['key']}.json", spec)
    monkeypatch.setattr(evaluation, "model_evaluation_complete", lambda *args: True)
    first = evaluation.cleanup_completed_model(root, plan, spec)
    assert first["state"] == "complete" and first["bytes"] == 7 and not folder.exists()
    assert evaluation.cleanup_completed_model(root, plan, spec) == first
    assert (root / "plan.json").exists()


@pytest.mark.parametrize("wrong_weights", [False, True])
def test_l0_cleanup_checks_and_removes_its_extra_nine_dataset_copy(tmp_path, monkeypatch, wrong_weights):
    spec = {**evaluation.MODELS[0], "revision": "a" * 40}
    root, scratch = tmp_path / "eval", tmp_path / "scratch"
    folder, extra = scratch / "models" / spec["key"], tmp_path / "slurm_tmp/rloo_final_model"
    for path in (folder, extra):
        path.mkdir(parents=True)
        (path / "weights").write_text("weights")
    plan = {"job_id": "146103", "node": "compute", "models": [spec],
            "minerva_root": str(root / "minerva"), "nine_root": str(root / "nine"),
            "scratch": str(scratch), "base_model": str(tmp_path / "base")}
    evaluation.write(root / "plan.json", plan)
    evaluation.write(root / "nine_inference_cache.json", {
        "path": str(extra), "job_id": plan["job_id"], "node": plan["node"],
        "repo": spec["repo"], "revision": spec["revision"],
    })
    evaluation.write(root / "prepared_models" / f"{spec['key']}.json",
                     {**spec, "merged_files": {"weights": evaluation.digest(folder / "weights")}})
    monkeypatch.setattr(evaluation, "model_evaluation_complete", lambda *args: True)
    monkeypatch.setattr(evaluation.shared, "nine_complete", lambda *args: True)
    if wrong_weights:
        (extra / "weights").write_text("another model")
        with pytest.raises(ValueError, match="another model"):
            evaluation.cleanup_completed_model(root, plan, spec)
        assert folder.exists() and extra.exists()
    else:
        receipt = evaluation.cleanup_completed_model(root, plan, spec)
        assert receipt["bytes"] == 14 and not folder.exists() and not extra.exists()
