"""Audit dataset-specific individual budgets and the cross-budget handoff."""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_experiments import compression_relative_individual_budget as suite


def base_reference(root):
    root.mkdir(parents=True, exist_ok=True)
    values = ["7,064.2", "11,135.8", "11,719.0", "17,997.3", "17,885.6", "17,890.3"]
    with (root / "main.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Model", "Metric", *[suite.mini.DATASETS[key][0] for key in suite.DATASETS]])
        writer.writerow(["qwen3-1.7B", "Mean Response Length (tokens)", *values])
    suite.write(root / "audit.json", {"complete": True, "after_thinking_only": True,
                "model_repo": "Qwen/Qwen3-1.7B", "files_sha256": {"main.csv": suite.digest(root / "main.csv")}})


def test_budgets_use_decimal_reference_and_ceil_without_k_rounding(tmp_path):
    base_reference(tmp_path)
    grid = suite.budget_grid(tmp_path)
    assert grid["minervamath"]["budgets"] == [3533, 7065, 14129, 21193]
    assert grid["olympiadbench"]["budgets"] == [5568, 11136, 22272, 33408]
    assert grid["amc22_23"]["budgets"] == [5860, 11719, 23438, 35157]
    assert grid["aime24"]["budgets"] == [8999, 17998, 35995, 53992]
    assert grid["aime25"]["budgets"] == [8943, 17886, 35772, 53657]
    assert grid["aime26"]["budgets"] == [8946, 17891, 35781, 53671]
    assert suite.POINTS == 144 and suite.POINTS_PER_DATASET == 24


def test_changed_base_length_table_is_rejected(tmp_path):
    base_reference(tmp_path)
    with (tmp_path / "main.csv").open("a") as stream:
        stream.write("changed\n")
    with pytest.raises(ValueError, match="Changed base reference"):
        suite.budget_grid(tmp_path)


def cross_predecessor(root):
    suite.write(root / "plan.json", {"points": 175})
    suite.write(root / "schedule.json", {"dataset_order": list(suite.DATASETS)})
    suite.write(root / "queue_status.json", {"state": "complete"})
    all_rows = []
    for dataset in suite.DATASETS:
        stage = root / "datasets" / dataset
        rows = [{"dataset": dataset, "model_key": key, "budget_tokens": budget}
                for key in suite.cross.MODELS for budget in suite.cross.BUDGETS]
        all_rows.extend(rows)
        suite.write(stage / "status.json", {"state": "complete"})
        suite.write(stage / "report/metrics.json", rows)
        suite.write(stage / "report/audit.json", {"complete": True, "points": 25,
                    "all_rollout_ledgers_verified": True, "metrics_sha256": suite.digest(stage / "report/metrics.json")})
        for key in suite.cross.MODELS:
            suite.write(stage / "cache_cleanup" / f"{key}.json", {"state": "complete"})
    suite.write(root / "report/metrics.json", all_rows)
    suite.write(root / "report/audit.json", {"complete": True, "points": 150, "datasets": list(suite.DATASETS),
                "all_rollout_ledgers_verified": True, "metrics_sha256": suite.digest(root / "report/metrics.json")})
    return {"predecessor_root": str(root), "predecessor_plan_sha256": suite.digest(root / "plan.json"),
            "predecessor_schedule_sha256": suite.digest(root / "schedule.json")}


@pytest.mark.parametrize("damage", [None, "running", "plan", "schedule", "metrics", "audit", "cleanup"])
def test_wait_for_entire_six_dataset_cross_queue_and_cleanup(tmp_path, damage):
    plan = cross_predecessor(tmp_path)
    if damage == "running":
        suite.write(tmp_path / "queue_status.json", {"state": "running", "dataset": "aime26"})
        assert not suite.predecessor_ready(plan)
        return
    if damage in ("plan", "schedule"):
        suite.write(tmp_path / f"{damage}.json", {"changed": True})
    elif damage == "metrics":
        suite.write(tmp_path / "report/metrics.json", [])
    elif damage == "audit":
        audit = suite.read(tmp_path / "report/audit.json")
        suite.write(tmp_path / "report/audit.json", {**audit, "points": 149})
    elif damage == "cleanup":
        (tmp_path / "datasets/aime26/cache_cleanup/f_cov.json").unlink()
    if damage:
        with pytest.raises(ValueError):
            suite.predecessor_ready(plan)
    else:
        assert suite.predecessor_ready(plan)


def completed_dataset(root, dataset, budgets):
    stage = root / "datasets" / dataset
    rows = [{"dataset": dataset, "model_key": key, "budget_tokens": budget, "pass_at_budget_percent": 50.0}
            for key in suite.MODELS for budget in budgets]
    suite.write(stage / "status.json", {"state": "complete"})
    suite.write(stage / "report/metrics.json", rows)
    suite.write(stage / "report/audit.json", {"complete": True, "points": 24, "all_rollout_ledgers_verified": True,
                                             "metrics_sha256": suite.digest(stage / "report/metrics.json")})
    for key in suite.MODELS:
        suite.write(stage / "cache_cleanup" / f"{key}.json", {"state": "complete"})
    return stage


def test_dataset_barrier_and_report_cover_six_models_at_four_relative_budgets(tmp_path):
    base_reference(tmp_path / "base_reference")
    plan = {"budget_grid": suite.budget_grid(tmp_path / "base_reference"),
            "models": {key: {"label": key} for key in suite.MODELS}}
    for dataset in suite.DATASETS:
        assert suite.next_dataset(tmp_path) == dataset
        stage = completed_dataset(tmp_path, dataset, plan["budget_grid"][dataset]["budgets"])
        (stage / "cache_cleanup/base.json").unlink()
        assert suite.next_dataset(tmp_path) == dataset
        suite.write(stage / "cache_cleanup/base.json", {"state": "complete"})
    assert suite.next_dataset(tmp_path) is None
    suite.report(tmp_path, plan)
    rows = suite.read(tmp_path / "report/metrics.json")
    assert len(rows) == 144 and {r["model_key"] for r in rows} == set(suite.MODELS)
    assert {r["multiplier"] for r in rows} == {"0.5", "1", "2", "3"}
    assert {r["budget_tokens"] for r in rows if r["dataset"] == "aime24"} == {8999, 17998, 35995, 53992}
    assert "0.5× (3,533 tokens)" in (tmp_path / "report/README.md").read_text()


@pytest.mark.parametrize("damage", [None, "incomplete", "cross_protocol", "budget", "external_cache"])
def test_cleanup_only_owned_base_cache_after_four_verified_individual_points(tmp_path, damage):
    stage = tmp_path / "datasets/minervamath"
    scratch = tmp_path / "scratch"
    cache = scratch / "models/base"
    shared = tmp_path / "shared_base"
    suite.write(cache / "weights.json", {"owned": True})
    suite.write(shared / "weights.json", {"shared": True})
    suite.write(tmp_path / "plan.json", {"scratch": str(scratch)})
    manifest = {"fingerprint": "individual-relative-budget"}
    suite.write(stage / "execution_manifest.json", manifest)
    plan = {"suite_root": str(tmp_path), "dataset_key": "minervamath", "num_questions": 272,
            "budgets": [3533, 7065, 14129, 21193],
            "models": {"base": {"path": str(cache), "repo": "Qwen/Qwen3-1.7B", "revision": "pinned"}}}
    for budget in plan["budgets"]:
        folder = suite.mini.point_directory(stage, "base", budget, "minervamath")
        suite.write(folder / "response.json", {"saved": True})
        artifact = folder / "response.json"
        suite.write(folder / "summary.json", {"state": "complete", "protocol": "eval2", "ledger_audit": "passed",
            "allocated_output_budget": 272 * budget,
            "identity": suite.mini.point_identity("base", budget, manifest, "minervamath"),
            "artifacts": {"rollouts": {"file": artifact.name, "size": artifact.stat().st_size, "sha256": suite.digest(artifact)}}})
    last = folder / "summary.json"
    if damage == "incomplete":
        last.unlink()
    elif damage == "cross_protocol":
        suite.write(last, {**suite.read(last), "protocol": "eval3"})
    elif damage == "budget":
        suite.write(last, {**suite.read(last), "allocated_output_budget": 21193})
    elif damage == "external_cache":
        plan["models"]["base"]["path"] = str(shared)
    if damage:
        with pytest.raises(ValueError):
            suite.cleanup_model(stage, plan, "base")
        assert cache.exists()
    else:
        suite.cleanup_model(stage, plan, "base")
        assert not cache.exists()
    assert shared.exists()


def test_worker_keeps_budget_per_question_and_stops_after_success(tmp_path, monkeypatch):
    mini = suite.mini
    model = tmp_path / "model"
    mini.write(model / "generation_config.json", {"eos_token_id": [9]})
    rows = [{"unique_id": str(i), "ground_truth": "42", "prompt_token_ids": [i], "dataset": "minervamath"} for i in range(2)]
    plan = {"job_id": "123", "protocol": "eval2", "seed": 0, "budgets": [7], "engine": {},
            "datasets": [{"key": "minervamath", "rows": 2}],
            "models": {"base": {"questions_file": "questions.json", "label": "base"}},
            "sampling": {"per_rollout_cap": 32768, "max_batch_size": 32, "temperature": .6, "top_p": .95, "top_k": 20}}
    manifest = {"fingerprint": "relative", "models": {"base": {"path": str(model)}}}
    mini.write(tmp_path / "execution_manifest.json", manifest)
    mini.write(tmp_path / "questions.json", rows)
    monkeypatch.setattr(mini, "verify_plan", lambda _: plan)
    monkeypatch.setattr(mini, "require_compute", lambda _: None)

    class Pool:
        def __init__(self, **kwargs):
            self.first = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, fn, items, **kwargs):
            scores = [1, 0] if self.first else [0]
            self.first = False
            assert len(items) == len(scores)
            return [{"score": score, "grader_status": "scored"} for score in scores]

    class Engine:
        def __init__(self, **kwargs):
            self.first = True

        def generate(self, *, sampling_params, **kwargs):
            assert [p.max_tokens for p in sampling_params] == ([7, 7] if self.first else [4])
            lengths = [2, 3] if self.first else [4]
            self.first = False
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1] * n, finish_reason="stop")]) for n in lengths]

    monkeypatch.setattr(mini, "ProcessPoolExecutor", Pool)
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace(decode=lambda *a, **k: "text"))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=SimpleNamespace))
    assert mini.worker(tmp_path, "base", 0, 7, "minervamath") == 0
    result = mini.completed_point(tmp_path, "base", 7, manifest, "minervamath")
    assert result["protocol"] == "eval2" and result["ledger_audit"] == "passed"
    assert result["num_prompts"] == 2 and result["num_questions_solved"] == 1
    assert result["total_rollouts"] == 3 and result["total_output_tokens"] == 9
    assert result["unused_output_budget"] == 5


@pytest.mark.parametrize("budget", [3533, 8192])
def test_worker_cli_accepts_only_dataset_specific_budgets(tmp_path, monkeypatch, budget):
    plan = {"suite_root": str(tmp_path), "dataset_key": "minervamath", "budgets": [3533, 7065, 14129, 21193], "models": {"base": {}}}
    monkeypatch.setattr(sys, "argv", ["relative", "worker", "--output-root", str(tmp_path), "--model", "base",
                                     "--dataset", "minervamath", "--budget", str(budget), "--rank", "0"])
    monkeypatch.setattr(suite.mini, "verify_plan", lambda _: plan)
    monkeypatch.setattr(suite, "verify_suite", lambda _: {})
    calls = []
    monkeypatch.setattr(suite.mini, "worker", lambda *args: calls.append(args) or 0)
    for name in ("MODULE", "dependency_ready", "prepare_models"):
        monkeypatch.setattr(suite.mini, name, getattr(suite.mini, name))
    if budget == 3533:
        assert suite.main() == 0 and calls[0][3] == 3533
    else:
        with pytest.raises(ValueError, match="relative-budget plan"):
            suite.main()
        assert not calls
