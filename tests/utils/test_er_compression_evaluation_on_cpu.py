"""ER completion gates, GPU handoff, and seed-preserving Minerva sharding."""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_experiments import er_compression_evaluation as evaluation
from qwen3_experiments import math_eval_budget_engine as engine
from qwen3_experiments.math_eval_matrix_common import rollout_seed


def completed_training(root):
    training = {"hf_repo_prefix": "owner/er", "rollout_hf_repo": "owner/er-rollouts"}
    evaluation.write(root / "plan.json", training)
    state = {"state": "complete", "exit_code": 0, "completed_rollout_steps": 100,
             "optimizer_updates": 200, "saved_training_rollouts": 25600}
    evaluation.write(root / "status.json", state)
    for step in (20, 40, 60, 80, 100):
        evaluation.write(root / "archive_receipts" / f"global_step{step}.json", {
            "repo_id": f"owner/er-step_{step}", "path": f"global_step_{step}/actor", "revision": "a" * 40,
            "files": {"state.pt": 10}, "sha256": {"state.pt": "b" * 64},
        })
    final = {"repo_id": "owner/er-final", "path": "", "local_path": "final_model", "revision": "c" * 40,
             "files": {"config.json": 1, "model.safetensors": 2},
             "sha256": {"config.json": "d" * 64, "model.safetensors": "e" * 64}}
    evaluation.write(root / "archive_receipts/final_model.json", final)
    evaluation.write(root / "rollout_upload.json", {"state": "verified", "num_rollouts": 25600,
                                                   "num_steps": 100, "repo_id": "owner/er-rollouts"})
    return {"training_root": str(root), "model_repo": "owner/er-final"}, state, final


def test_waits_for_training_and_all_archives_even_if_final_export_exists(tmp_path):
    plan, state, final = completed_training(tmp_path)
    assert evaluation.training_ready(plan) == final
    for phase in ("training_or_archiving", "uploading", "waiting_for_all_evaluations"):
        evaluation.write(tmp_path / "status.json", {**state, "state": phase})
        assert evaluation.training_ready(plan) is None
    evaluation.write(tmp_path / "status.json", {**state, "state": "failed"})
    with pytest.raises(RuntimeError, match="evaluation will not start"):
        evaluation.training_ready(plan)


@pytest.mark.parametrize("damage", ["step", "optimizer", "rollout_count", "upload", "wrong_repo", "revision", "hashes", "missing_checkpoint"])
def test_rejects_incomplete_or_wrong_training_artifacts(tmp_path, damage):
    plan, state, final = completed_training(tmp_path)
    if damage in ("step", "optimizer", "rollout_count"):
        field = {"step": "completed_rollout_steps", "optimizer": "optimizer_updates", "rollout_count": "saved_training_rollouts"}[damage]
        evaluation.write(tmp_path / "status.json", {**state, field: state[field] - 1})
    elif damage == "upload":
        evaluation.write(tmp_path / "rollout_upload.json", {"state": "uploading"})
    elif damage == "missing_checkpoint":
        (tmp_path / "archive_receipts/global_step100.json").unlink()
    else:
        if damage == "wrong_repo":
            final["repo_id"] = "owner/polaris-final"
        elif damage == "revision":
            final["revision"] = "main"
        else:
            del final["sha256"]["model.safetensors"]
        evaluation.write(tmp_path / "archive_receipts/final_model.json", final)
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.training_ready(plan)


def test_phase_waits_for_gpu_locks_and_does_not_prepare_model(tmp_path, monkeypatch):
    import fcntl

    plan, _, _ = completed_training(tmp_path / "training")
    holder = tmp_path / "training.lock"
    plan.update(job_id="146103", holder_locks=[str(holder)])
    monkeypatch.setattr(evaluation, "require_compute", lambda _: "compute")
    monkeypatch.setattr(evaluation, "prepare_model", lambda *_: pytest.fail("Touched model before GPU handoff"))
    with holder.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert evaluation.run_phase(tmp_path, plan, "nine") == 75
    monkeypatch.setattr(evaluation.subprocess, "check_output", lambda *_, **__: "1234\n")
    assert evaluation.run_phase(tmp_path, plan, "nine") == 75


def test_nine_completion_checks_counts_and_actual_results(tmp_path):
    evaluation.write(tmp_path / "status.json", {"state": "running"})
    assert not evaluation.nine_complete(tmp_path)
    evaluation.write(tmp_path / "status.json", {"state": "complete"})
    evaluation.write(tmp_path / "report/metrics.json", ["metric"])
    evaluation.write(tmp_path / "report/per_sample.json", ["sample"])
    audit = {"complete": True, "questions": 1819, "responses_verified": 7276, "budget_points": 54,
             "metrics_sha256": evaluation.digest(tmp_path / "report/metrics.json"),
             "per_sample_sha256": evaluation.digest(tmp_path / "report/per_sample.json")}
    evaluation.write(tmp_path / "report/audit.json", audit)
    assert evaluation.nine_complete(tmp_path)
    evaluation.write(tmp_path / "report/audit.json", {**audit, "responses_verified": 7275})
    with pytest.raises(ValueError, match="Incomplete"):
        evaluation.nine_complete(tmp_path)
    evaluation.write(tmp_path / "report/audit.json", audit)
    evaluation.write(tmp_path / "report/per_sample.json", ["changed"])
    with pytest.raises(ValueError, match="Changed"):
        evaluation.nine_complete(tmp_path)


def fake_point(rows, budget=12):
    records = []

    class FakeEngine:
        def generate(self, *, prompt_token_ids, sampling_params, **kwargs):
            result = []
            for prompt, params in zip(prompt_token_ids, sampling_params, strict=True):
                token = 1 if prompt[0] % 2 == 0 else 0
                length = 2 if token else params.max_tokens
                result.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=[token] * length, finish_reason="stop" if token else "length")]))
            return result

    summary, prompts = engine.evaluate_point(
        protocol="eval2", budget=budget, seed=0, rows=rows,
        prompt_token_ids=[[row.get("seed_position", i)] for i, row in enumerate(rows)],
        engine=FakeEngine(), sampling_params_type=SimpleNamespace,
        tokenizer=SimpleNamespace(decode=lambda ids, **_: str(ids[0])),
        score_many=lambda items: [int(text) for text, _ in items], emit=records.append,
        sampling={"per_rollout_cap": 8, "max_batch_size": 32, "temperature": .6, "top_p": .95, "top_k": 20},
        stop_on_first_success=True,
    )
    return records, summary, prompts


@pytest.mark.parametrize("model_key", [evaluation.MODEL_KEY, "polaris_l0_step100", "polaris_er_step100", "polaris_maxrl_step100"])
def test_sharding_preserves_seeds_and_merged_budget_ledger(tmp_path, model_key):
    rows = [{"unique_id": str(p), "ground_truth": "42"} for p in range(16)]
    baseline, expected, expected_prompts = fake_point(rows)
    shards = [tmp_path / "shards" / str(rank) for rank in range(8)]
    for rank, folder in enumerate(shards):
        shard_rows = [{**row, "seed_position": p} for p, row in enumerate(rows) if p % 8 == rank]
        records, _, _ = fake_point(shard_rows)
        assert records[0]["rollout_seed"] == rollout_seed(0, rank, 0)
        engine.audit_records(records, protocol="eval2", budget=12, seed=0, rows=shard_rows,
                             per_rollout_cap=8, stop_on_first_success=True)
        evaluation.write(folder / "questions.json", shard_rows)
        evaluation.write(folder / "execution_manifest.json", {"fingerprint": "fixed"})
        directory = evaluation.mini.point_directory(folder, model_key, 12)
        directory.mkdir(parents=True)
        raw = directory / "rollouts.jsonl.gz"
        with gzip.open(raw, "wt") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        evaluation.write(directory / "summary.json", {
            "state": "complete", "identity": {"manifest": "fixed", "model": model_key, "budget": 12},
            "artifacts": {"rollouts": {"file": raw.name, "size": raw.stat().st_size, "sha256": evaluation.digest(raw)}},
        })
    merged = list(evaluation.merged_records(shards, 12, model_key))
    summary, prompts = engine.audit_records(merged, protocol="eval2", budget=12, seed=0, rows=rows,
                                            per_rollout_cap=8, stop_on_first_success=True)
    assert prompts == expected_prompts
    assert all(expected[key] == value for key, value in summary.items())
    seeds = lambda records: {(r["unique_id"], r["rollout_index"]): r["rollout_seed"] for r in records}
    assert seeds(merged) == seeds(baseline)
    assert summary["num_questions_solved"] == 8
    assert summary["total_output_tokens"] == 8 * 2 + 8 * 12
    (evaluation.mini.point_directory(shards[-1], model_key, 12) / "summary.json").unlink()
    with pytest.raises(ValueError, match="Missing Minerva shard"):
        list(evaluation.merged_records(shards, 12, model_key))


def test_queue_runs_nine_then_minerva_after_training_and_busy_retry(tmp_path, monkeypatch):
    plan = {"job_id": "146103", "node": "compute", "runtime": str(tmp_path), "python_bin": "python",
            "nine_root": str(tmp_path / "nine"), "minerva_root": str(tmp_path / "minerva")}
    ready, launched = [], []
    monkeypatch.setattr(evaluation, "require_compute", lambda _: "compute")
    monkeypatch.setattr(evaluation, "verify_plan", lambda _: plan)
    monkeypatch.setattr(evaluation, "environment", lambda _: {})
    monkeypatch.setattr(evaluation, "training_ready", lambda _: {"revision": "a" * 40} if ready else None)

    def wait(_):
        if not ready:
            assert not launched
            ready.append(True)

    def child(command, **kwargs):
        phase = command[command.index("--phase") + 1]
        launched.append(phase)
        code = 75 if len(launched) == 1 else 0
        if phase == "minerva":
            assert launched == ["nine", "nine", "minerva"]
            evaluation.write(tmp_path / "minerva/report/audit.json", {
                "complete": True, "points": 5, "all_rollout_ledgers_verified": True,
            })
        return SimpleNamespace(pid=123, returncode=code, poll=lambda: code)

    monkeypatch.setattr(evaluation.time, "sleep", wait)
    monkeypatch.setattr(evaluation.subprocess, "Popen", child)
    monkeypatch.setattr(evaluation, "nine_complete", lambda _: len(launched) >= 2)
    evaluation.queue(tmp_path, plan)
    assert launched == ["nine", "nine", "minerva"]
    assert evaluation.read(tmp_path / "queue_status.json")["state"] == "complete"


def test_queue_never_starts_minerva_after_three_nine_failures(tmp_path, monkeypatch):
    plan = {"job_id": "146103", "node": "compute", "runtime": str(tmp_path), "python_bin": "python"}
    phases = []
    monkeypatch.setattr(evaluation, "require_compute", lambda _: "compute")
    monkeypatch.setattr(evaluation, "verify_plan", lambda _: plan)
    monkeypatch.setattr(evaluation, "environment", lambda _: {})
    monkeypatch.setattr(evaluation, "training_ready", lambda _: {"revision": "a" * 40})
    monkeypatch.setattr(evaluation.time, "sleep", lambda _: None)

    def child(command, **kwargs):
        phases.append(command[command.index("--phase") + 1])
        return SimpleNamespace(pid=123, returncode=1, poll=lambda: 1)

    monkeypatch.setattr(evaluation.subprocess, "Popen", child)
    with pytest.raises(RuntimeError, match="failed three times"):
        evaluation.queue(tmp_path, plan)
    assert phases == ["nine"] * 3
    assert evaluation.read(tmp_path / "queue_status.json")["state"] == "failed"


@pytest.mark.parametrize("training_data", ["compression", "polaris"])
def test_report_uses_after_thinking_reference_and_excludes_historical_rows(tmp_path, monkeypatch, training_data):
    nine = evaluation.nine
    training = tmp_path / "training"
    evaluation.write(training / "plan.json", {"dataset_repo": "zjhhhh/compression_dataset" if training_data == "compression" else "Polaris-1-8-3200"})
    plan = {"training_root": str(training), "model_label": "ER" if training_data == "compression" else "L+0", "total_responses": 7276}
    if training_data == "polaris":
        plan["report_training_dataset"] = "Polaris-1-8-3200"
    evaluation.write(tmp_path / "plan.json", plan)
    evaluation.write(tmp_path / "prepared_inputs.json", {"model": {"repo": "owner/compression-er", "revision": "a" * 40}})
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "main_baseline.csv").write_text("Model,Metric\nPolaris ER,mean@4 (%)\n")
    (provenance / "budget_baseline.csv").write_text("model,dataset\nPolaris ER,math500\n")
    main = [{"Model": "qwen3-1.7B", "Metric": metric, **{title: "17.00" for _, title, _ in nine.DATASETS}}
            for metric in ("mean@4 (%)", "pass@4 (%)", "Mean Response Length (tokens)")]
    budgets = [{"model": "qwen3-1.7B", "dataset": key, **{f"{cap // 1024}k": "17.00" for cap in nine.CAPS}}
               for key, _, _ in nine.DATASETS]
    monkeypatch.setattr(nine, "after_thinking_reference", lambda *_: (main, budgets))
    metrics = [{"dataset": key, "cap_tokens": cap, "mean_at_4_percent": 20., "pass_at_4_percent": 25., "mean_output_tokens": 100.}
               for key, _, _ in nine.DATASETS for cap in nine.CAPS]
    base = nine.load_module("er_report_core_test", Path(nine.__file__).with_name("eval_polaris_step80.py"))
    nine.write_report(tmp_path, base, metrics)
    report = (tmp_path / "report/README.md").read_text()
    expected_label = "ER step 100 (compression)" if training_data == "compression" else "L+0 step 100 (Polaris-1-8-3200)"
    assert expected_label in report
    assert "Polaris ER" not in report and "Historical Qwen3" not in report
    assert "Both models use the same after-thinking-only scoring policy" in report
    expected_run = "compression ER" if training_data == "compression" else "Polaris L+0"
    assert f"this {expected_run} run uses top-k 20 and seed 42" in report
