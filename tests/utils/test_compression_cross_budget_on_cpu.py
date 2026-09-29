"""Keep the original evaluations, shared budgets and dataset barriers intact."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_experiments import compression_cross_budget as suite


def predecessor(tmp_path):
    previous = tmp_path / "original"
    suite.write(previous / "plan.json", {"order": ["f_cov_nine", "f_cov_individual", "maxrl_nine", "maxrl_individual"]})
    plan = {"predecessor_root": str(previous), "predecessor_plan_sha256": suite.digest(previous / "plan.json")}
    for name in ("nine", "budget"):
        suite.write(previous / "report" / f"{name}_metrics.json", [{"result": name}])
    audit = {"complete": True, "models": ["f_cov_step100", "maxrl_step100"], "nine_responses": 14552,
             "budget_points": 70, "all_rollout_ledgers_verified": True,
             **{f"{name}_metrics_sha256": suite.digest(previous / "report" / f"{name}_metrics.json")
                for name in ("nine", "budget")}}
    suite.write(previous / "report/audit.json", audit)
    for key in ("f_cov_step100", "maxrl_step100"):
        suite.write(previous / "model_cache_cleanup" / f"{key}.json", {"state": "complete"})
    return plan, previous, audit


@pytest.mark.parametrize("phase", ["f_cov_nine", "f_cov_individual", "maxrl_nine", "maxrl_individual"])
def test_wait_for_every_original_evaluation(tmp_path, phase):
    plan, previous, _ = predecessor(tmp_path)
    suite.write(previous / "queue_status.json", {"state": "running", "phase": phase})
    assert not suite.predecessor_ready(plan)
    suite.write(previous / "queue_status.json", {"state": "complete"})
    assert suite.predecessor_ready(plan)


@pytest.mark.parametrize("damage", ["plan", "nine", "individual", "responses", "ledgers", "cleanup", "metrics"])
def test_predecessor_rejects_incomplete_or_changed_evaluations(tmp_path, damage):
    plan, previous, audit = predecessor(tmp_path)
    suite.write(previous / "queue_status.json", {"state": "complete"})
    if damage == "plan":
        suite.write(previous / "plan.json", {"order": []})
    elif damage == "nine":
        audit["models"] = ["f_cov_step100"]
    elif damage == "individual":
        audit["budget_points"] = 35
    elif damage == "responses":
        audit["nine_responses"] = 7276
    elif damage == "ledgers":
        audit["all_rollout_ledgers_verified"] = False
    elif damage == "cleanup":
        suite.write(previous / "model_cache_cleanup/maxrl_step100.json", {"state": "running"})
    else:
        suite.write(previous / "report/budget_metrics.json", [])
    suite.write(previous / "report/audit.json", audit)
    with pytest.raises(ValueError):
        suite.predecessor_ready(plan)


def complete_stage(stage, *, cleanup=True):
    suite.write(stage / "status.json", {"state": "complete"})
    suite.write(stage / "report/metrics.json", [])
    suite.write(stage / "report/audit.json", {"points": 25, "complete": True, "all_rollout_ledgers_verified": True,
                                             "metrics_sha256": suite.digest(stage / "report/metrics.json")})
    if cleanup:
        for key in suite.MODELS:
            suite.write(stage / "cache_cleanup" / f"{key}.json", {"state": "complete"})


def test_finish_all_five_models_and_cleanup_before_next_dataset(tmp_path):
    plan = {"dataset_order": list(suite.DATASETS)}
    assert suite.next_dataset(tmp_path, plan) == "minervamath"
    for dataset in suite.DATASETS:
        stage = tmp_path / "datasets" / dataset
        complete_stage(stage, cleanup=False)
        assert suite.next_dataset(tmp_path, plan) == dataset
        for key in suite.MODELS[:-1]:
            suite.write(stage / "cache_cleanup" / f"{key}.json", {"state": "complete"})
        assert suite.next_dataset(tmp_path, plan) == dataset
        suite.write(stage / "cache_cleanup" / f"{suite.MODELS[-1]}.json", {"state": "complete"})
    assert suite.next_dataset(tmp_path, plan) is None


def saved_model(tmp_path):
    stage = tmp_path / "datasets/minervamath"
    scratch = tmp_path / "node"
    target = scratch / "models/er"
    plan = {"suite_root": str(tmp_path), "dataset_key": "minervamath", "num_questions": 272,
            "models": {"er": {"path": str(target), "repo": "owner/compression-final", "revision": "a" * 40}},
            "budgets": suite.BUDGETS}
    suite.write(tmp_path / "plan.json", {"scratch": str(scratch)})
    manifest = {"fingerprint": "pinned-cross-budget"}
    suite.write(stage / "execution_manifest.json", manifest)
    suite.write(target / "weights.json", {"reproducible": True})
    for budget in plan["budgets"]:
        folder = suite.mini.point_directory(stage, "er", budget, "minervamath")
        suite.write(folder / "responses.json", ["saved answer"])
        artifact = folder / "responses.json"
        suite.write(folder / "summary.json", {
            "state": "complete", "protocol": "eval3", "ledger_audit": "passed",
            "allocated_output_budget": 272 * budget,
            "identity": suite.mini.point_identity("er", budget, manifest, "minervamath"),
            "artifacts": {"rollouts": {"file": artifact.name, "size": artifact.stat().st_size,
                                       "sha256": suite.digest(artifact)}}})
    return stage, plan, target


@pytest.mark.parametrize("damage", [None, "incomplete", "individual", "budget", "artifact", "external_cache"])
def test_cleanup_only_verified_finished_model_cache(tmp_path, damage):
    stage, plan, target = saved_model(tmp_path)
    other = tmp_path / "other_node/keep.json"
    suite.write(other, ["other allocation"])
    last = suite.mini.point_directory(stage, "er", plan["budgets"][-1], "minervamath")
    summary = suite.read(last / "summary.json")
    if damage == "incomplete":
        (last / "summary.json").unlink()
    elif damage == "individual":
        suite.write(last / "summary.json", {**summary, "protocol": "eval2"})
    elif damage == "budget":
        suite.write(last / "summary.json", {**summary, "allocated_output_budget": plan["budgets"][-1]})
    elif damage == "artifact":
        suite.write(last / "responses.json", ["modified answer"])
    elif damage == "external_cache":
        plan["models"]["er"]["path"] = str(other.parent)
    if damage:
        with pytest.raises(ValueError):
            suite.cleanup_model(stage, plan, "er")
        assert target.exists()
    else:
        suite.cleanup_model(stage, plan, "er")
        assert not target.exists()
        assert suite.read(stage / "cache_cleanup/er.json")["state"] == "complete"
        assert len(list((stage / "results").rglob("responses.json"))) == 5
    assert suite.read(other) == ["other allocation"]


def test_worker_uses_one_complete_shared_budget_and_retains_unvisited_questions(tmp_path, monkeypatch):
    mini = suite.mini
    model = tmp_path / "model"
    mini.write(model / "generation_config.json", {"eos_token_id": [9]})
    rows = [{"unique_id": str(i), "ground_truth": "42", "prompt_token_ids": [i], "dataset": "minervamath"}
            for i in range(3)]
    plan = {"job_id": "123", "protocol": "eval3", "seed": 0, "budgets": [4], "engine": {},
            "datasets": [{"key": "minervamath", "rows": 3}],
            "models": {"er": {"questions_file": "questions.json", "label": "ER"}},
            "sampling": {"per_rollout_cap": 32768, "max_batch_size": 32, "temperature": .6, "top_p": .95, "top_k": 20}}
    manifest = {"fingerprint": "cross", "models": {"er": {"path": str(model)}}}
    mini.write(tmp_path / "execution_manifest.json", manifest)
    mini.write(tmp_path / "questions.json", rows)
    monkeypatch.setattr(mini, "verify_plan", lambda _: plan)
    monkeypatch.setattr(mini, "require_compute", lambda _: None)

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, fn, items, **kwargs):
            return [{"score": 0, "grader_status": "scored"} for _ in items]

    class Engine:
        def __init__(self, **kwargs):
            pass

        def generate(self, *, sampling_params, **kwargs):
            # The first request may spend the entire dataset's 3 × 4 allowance.
            assert len(sampling_params) == 1 and sampling_params[0].max_tokens == 12
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1] * 12, finish_reason="length")])]

    monkeypatch.setattr(mini, "ProcessPoolExecutor", Pool)
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace(decode=lambda *a, **k: "unfinished"))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=SimpleNamespace))
    assert mini.worker(tmp_path, "er", 0, 4, "minervamath") == 0
    summary = mini.completed_point(tmp_path, "er", 4, manifest, "minervamath")
    assert summary["protocol"] == "eval3" and summary["ledger_audit"] == "passed"
    assert summary["num_prompts"] == 3 and summary["total_rollouts"] == 1
    assert summary["allocated_output_budget"] == summary["total_output_tokens"] == 12
    assert summary["unused_output_budget"] == 0
    mini.report(tmp_path, plan, manifest)
    assert "All questions share n × budget" in (tmp_path / "report/README.md").read_text()


@pytest.mark.parametrize("damage", [None, "checksum", "escape"])
def test_storage_offloads_only_verified_owned_results_then_restores(tmp_path, monkeypatch, damage):
    stage, plan, _ = saved_model(tmp_path)
    plan.update(scratch=str(tmp_path / "node"), storage_critical_bytes=50, storage_warning_bytes=100)
    folder = suite.mini.point_directory(stage, "er", 8192, "minervamath")
    source = folder / "responses.json"
    before = source.read_bytes()
    os.utime(source, (1, 1))
    if damage:
        summary = suite.read(folder / "summary.json")
        summary["artifacts"]["rollouts"]["sha256" if damage == "checksum" else "file"] = (
            "0" * 64 if damage == "checksum" else "../../../../../../other_node/keep.json")
        suite.write(folder / "summary.json", summary)
    monkeypatch.setattr(suite.shutil, "disk_usage", lambda p: SimpleNamespace(free=20 if Path(p) == tmp_path else 100 * 1024**3))
    if damage:
        with pytest.raises(ValueError):
            suite.recover_storage(tmp_path, plan)
        assert not source.is_symlink() and source.read_bytes() == before
    else:
        assert suite.recover_storage(tmp_path, plan)["moved_bytes"] == len(before)
        assert source.is_symlink() and source.read_bytes() == before
        monkeypatch.setattr(suite.shutil, "disk_usage", lambda _: SimpleNamespace(free=200))
        assert suite.recover_storage(tmp_path, plan)["restored_bytes"] == len(before)
        assert not source.is_symlink() and source.read_bytes() == before


def test_restart_after_all_points_complete_still_cleans_model_caches(tmp_path, monkeypatch):
    suite_root, stage = tmp_path, tmp_path / "datasets/minervamath"
    plan = {"suite_root": str(suite_root)}
    monkeypatch.setattr(sys, "argv", ["cross", "run", "--output-root", str(stage)])
    monkeypatch.setattr(suite.mini, "verify_plan", lambda _: plan)
    monkeypatch.setattr(suite, "verify_suite", lambda _: {})
    monkeypatch.setattr(suite, "environment", lambda _: {})
    monkeypatch.setattr(suite.mini, "run", lambda *args, **kwargs: 0)
    calls = []
    monkeypatch.setattr(suite, "cleanup_model", lambda root, plan, key: calls.append(key))
    # main installs worker adapters; restore them after this test.
    for name in ("MODULE", "dependency_ready", "prepare_models"):
        monkeypatch.setattr(suite.mini, name, getattr(suite.mini, name))
    assert suite.main() == 0
    assert calls == list(suite.MODELS)


def test_cross_worker_accepts_4k_without_changing_individual_budgets(tmp_path, monkeypatch):
    stage = tmp_path / "datasets/minervamath"
    monkeypatch.setattr(sys, "argv", ["cross", "worker", "--output-root", str(stage),
                                     "--model", "er", "--dataset", "minervamath",
                                     "--budget", "4096", "--rank", "0"])
    monkeypatch.setattr(suite.mini, "verify_plan", lambda _: {"suite_root": str(tmp_path)})
    monkeypatch.setattr(suite, "verify_suite", lambda _: {})
    calls = []
    monkeypatch.setattr(suite.mini, "worker", lambda *args: calls.append(args) or 0)
    for name in ("MODULE", "dependency_ready", "prepare_models"):
        monkeypatch.setattr(suite.mini, name, getattr(suite.mini, name))
    assert suite.main() == 0
    assert calls == [(stage, "er", 0, 4096, "minervamath")]
    assert suite.mini.BUDGETS == [8192, 16384, 32768, 49152, 65536]


@pytest.mark.parametrize("defer_math500", [False, True])
def test_report_uses_requested_cross_budget_range(tmp_path, defer_math500):
    plan = {"models": {key: {"label": key} for key in suite.MODELS}, "dataset_order": list(suite.DATASETS)}
    suite.write(tmp_path / "plan.json", plan)
    datasets = list(suite.DATASETS[:-1] if defer_math500 else suite.DATASETS)
    if defer_math500:
        suite.write(tmp_path / "schedule.json", {"prepared_plan_sha256": suite.digest(tmp_path / "plan.json"),
                                                 "dataset_order": datasets})
    for dataset in datasets:
        stage = tmp_path / "datasets" / dataset
        complete_stage(stage)
        rows = [{"dataset": dataset, "model_key": key, "budget_tokens": budget,
                 "questions": suite.mini.DATASETS[dataset][1], "pass_at_budget_percent": 50.0}
                for key in suite.MODELS for budget in (4096, 8192, 16384, 32768, 49152)]
        suite.write(stage / "report/metrics.json", rows)
        audit = suite.read(stage / "report/audit.json")
        suite.write(stage / "report/audit.json", {**audit, "metrics_sha256": suite.digest(stage / "report/metrics.json")})
    suite.report(tmp_path, plan)
    report = (tmp_path / "report/README.md").read_text()
    assert "| Model | 4k × n | 8k × n | 16k × n | 32k × n | 48k × n |" in report
    assert "64k" not in report
    rows = suite.read(tmp_path / "report/metrics.json")
    assert len(rows) == len(datasets) * 25
    assert {row["budget_tokens"] for row in rows} == {4096, 8192, 16384, 32768, 49152}
    if defer_math500:
        assert "MATH500" not in report
        assert suite.next_dataset(tmp_path, {**plan, "dataset_order": datasets}) is None
    audit = suite.read(tmp_path / "report/audit.json")
    assert audit["points"] == len(datasets) * 25 and audit["datasets"] == datasets


@pytest.mark.parametrize("damage", [None, "plan", "order", "duplicate", "unknown"])
def test_schedule_preserves_prepared_plan_and_rejects_invalid_changes(tmp_path, damage):
    plan = {"dataset_order": list(suite.DATASETS)}
    suite.write(tmp_path / "plan.json", plan)
    before = (tmp_path / "plan.json").read_bytes()
    schedule = {"prepared_plan_sha256": suite.digest(tmp_path / "plan.json"),
                "dataset_order": list(suite.DATASETS[:-1])}
    if damage == "plan":
        schedule["prepared_plan_sha256"] = "0" * 64
    elif damage == "order":
        schedule["dataset_order"].reverse()
    elif damage == "duplicate":
        schedule["dataset_order"].append("minervamath")
    elif damage == "unknown":
        schedule["dataset_order"].append("not_prepared")
    suite.write(tmp_path / "schedule.json", schedule)
    if damage:
        with pytest.raises(ValueError):
            suite.scheduling(tmp_path, plan)
    else:
        assert suite.scheduling(tmp_path, plan)["dataset_order"] == list(suite.DATASETS[:-1])
    assert (tmp_path / "plan.json").read_bytes() == before


def test_controller_relaunch_preserves_worker_runtime(tmp_path, monkeypatch):
    worker_runtime = tmp_path / "runtime"
    controller_runtime = tmp_path / "controllers/runtime"
    plan = {"output_root": str(tmp_path), "runtime": str(worker_runtime), "scratch": str(tmp_path),
            "python_bin": sys.executable, "job_id": "123", "node": "node",
            "predecessor_root": str(tmp_path / "previous"), "dataset_order": list(suite.DATASETS)}
    suite.write(tmp_path / "plan.json", plan)
    suite.write(tmp_path / "schedule.json", {"prepared_plan_sha256": suite.digest(tmp_path / "plan.json"),
                                             "dataset_order": list(suite.DATASETS[:-1]),
                                             "controller_runtime": str(controller_runtime)})
    monkeypatch.setattr(suite, "verify_suite", lambda _: plan)
    monkeypatch.setattr(suite, "environment", lambda _: {"PYTHONPATH": str(worker_runtime)})
    calls = []
    monkeypatch.setattr(suite.subprocess, "Popen", lambda *a, **kw: calls.append((a, kw)) or SimpleNamespace(pid=1234))
    suite.launch(tmp_path)
    assert calls[0][1]["cwd"] == str(controller_runtime)
    assert calls[0][1]["env"]["PYTHONPATH"] == str(controller_runtime)
    assert suite.environment(plan)["PYTHONPATH"] == str(worker_runtime)
    assert suite.read(tmp_path / "launch.json")["dataset_order"] == list(suite.DATASETS[:-1])
