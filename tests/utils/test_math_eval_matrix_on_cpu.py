"""No weights, downloads, or GPUs: test exact budget and matrix semantics."""

import copy
import gzip
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "qwen3_experiments"
sys.path.insert(0, str(SCRIPTS))
engine_module = importlib.import_module("math_eval_budget_engine")
common = importlib.import_module("math_eval_matrix_common")


class FakeTokenizer:
    def decode(self, tokens, **kwargs):
        return "correct" if tokens[0] == 1 else "wrong"


def run(protocol, budget, *, count=3, seed=0, length=3, correct=lambda p, a: False, batch=2, cap=4,
        stop_on_first_success=False):
    rows = [{"unique_id": str(index), "ground_truth": "answer"} for index in range(count)]
    records = []

    class FakeEngine:
        def generate(self, *, prompt_token_ids, sampling_params, **kwargs):
            outputs = []
            for ids, parameters in zip(prompt_token_ids, sampling_params):
                p = ids[0]
                attempt = next(a for a in range(count * budget + 4) if common.rollout_seed(seed, p, a) == parameters.seed)
                marker = int(correct(p, attempt))
                tokens = [marker] * min(length, parameters.max_tokens)
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=tokens, finish_reason="stop")]))
            return outputs

    sampling = {"per_rollout_cap": cap, "max_batch_size": batch, "temperature": 0.6, "top_p": 0.95, "top_k": -1}
    summary, prompts = engine_module.evaluate_point(
        protocol=protocol, budget=budget, seed=seed, rows=rows, prompt_token_ids=[[i] for i in range(count)],
        engine=FakeEngine(), sampling_params_type=SimpleNamespace, tokenizer=FakeTokenizer(),
        score_many=lambda items: [int(response == "correct") for response, gold in items],
        emit=records.append, sampling=sampling, stop_on_first_success=stop_on_first_success)
    audited, audited_prompts = engine_module.audit_records(records, protocol=protocol, budget=budget, seed=seed,
                                                          rows=rows, per_rollout_cap=cap,
                                                          stop_on_first_success=stop_on_first_success)
    assert prompts == audited_prompts
    assert all(summary[key] == value for key, value in audited.items())
    return summary, prompts, records, rows


def test_matrix_is_exact_cartesian_product():
    config = common.read_config(SCRIPTS / "math_eval_matrix.json")
    tasks = common.make_tasks(config)
    points = [point for task in tasks for point in common.task_points(config, task)]
    assert len(tasks) == 5 * 4 * 3
    assert len(points) == 700
    assert {p["seed"] for p in points if p["protocol"] == "eval2"} == {0, 1, 2}
    assert {p["seed"] for p in points if p["protocol"] == "eval1"} == {0}
    assert "amc22_23" in {p["dataset"] for p in points}
    assert not any(model.get("length_instruction") for model in config["models"])
    assert config["evals"]["eval2"]["stop_on_first_success"] is True


def test_seed_repeats_do_not_share_shifted_attempt_streams():
    streams = [{common.rollout_seed(seed, 5, attempt) for attempt in range(100)} for seed in (0, 1, 2)]
    assert len(set.union(*streams)) == 300
    assert common.rollout_seed(0, 5, 1) != common.rollout_seed(1, 5, 0)


def test_eval1_mean_four_not_pass_four():
    summary, prompts, records, _ = run("eval1", 4, count=2, correct=lambda p, a: p == 0 and a == 0)
    assert summary["mean_at_4_accuracy"] == 1 / 8
    assert summary["fraction_solved"] == 1 / 2
    assert len(records) == 8
    assert all(p["attempts"] == 4 for p in prompts)


def test_eval2_charges_failures_and_continues_after_success():
    summary, prompts, records, _ = run("eval2", 7, count=2, length=2, correct=lambda p, a: a == 0)
    assert summary["total_output_tokens"] == 14
    assert all(p["output_tokens"] == 7 and p["attempts"] == 4 for p in prompts)
    assert sum(r["output_tokens"] for r in records if not r["score"]) == 10
    assert {r["max_output_tokens"] for r in records if r["remaining_budget_after"] == 0} == {1}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("batch", [1, 2, 10])
def test_eval2_early_stop_preserves_solved_and_each_question_stream(seed, batch):
    settings = dict(count=3, length=2, correct=lambda p, a: (p == 0 and a == 0) or (p == 1 and a == 2),
                    seed=seed, batch=batch)
    full, _, full_records, _ = run("eval2", 7, **settings)
    stopped, prompts, records, _ = run("eval2", 7, stop_on_first_success=True, **settings)
    assert stopped["num_questions_solved"] == full["num_questions_solved"] == 2
    assert [p["output_tokens"] for p in prompts] == [2, 6, 7]
    assert [p["attempts"] for p in prompts] == [1, 3, 4]
    assert stopped["total_output_tokens"] == 15  # No redistribution of saved tokens.
    assert stopped["allocated_output_budget"] == 21
    assert stopped["unused_output_budget"] == 6
    assert stopped["early_stopped_questions"] == 2
    for record in records:
        original = next(r for r in full_records if (r["prompt_position"], r["rollout_index"]) ==
                        (record["prompt_position"], record["rollout_index"]))
        for key in ("rollout_seed", "max_output_tokens", "output_token_ids", "score", "remaining_budget_after"):
            assert record[key] == original[key]
        assert not record["solved_before"]


def test_eval2_early_stop_all_correct_and_success_at_budget_boundary():
    summary, prompts, _, _ = run("eval2", 20, length=1, correct=lambda p, a: True, stop_on_first_success=True)
    assert summary["total_output_tokens"] == 3
    assert all(p["attempts"] == 1 for p in prompts)
    summary, prompts, _, _ = run("eval2", 7, length=2, correct=lambda p, a: a == 3, stop_on_first_success=True)
    assert summary["num_questions_solved"] == 3
    assert summary["unused_output_budget"] == summary["early_stopped_questions"] == 0


def test_eval2_early_stop_audit_rejects_resampling_or_unfinished_failure():
    _, _, records, rows = run("eval2", 7, count=1, length=2, correct=lambda p, a: a == 0)
    with pytest.raises(ValueError, match="resampled a solved"):
        engine_module.audit_records(records, protocol="eval2", budget=7, seed=0, rows=rows,
                                    per_rollout_cap=4, stop_on_first_success=True)
    _, _, records, rows = run("eval2", 7, count=1, length=2)
    with pytest.raises(ValueError, match="before success or budget exhaustion"):
        engine_module.audit_records(records[:-1], protocol="eval2", budget=7, seed=0, rows=rows,
                                    per_rollout_cap=4, stop_on_first_success=True)


def test_eval1_never_early_stops_mean_at_four():
    summary, prompts, records, _ = run("eval1", 4, correct=lambda p, a: a == 0, stop_on_first_success=True)
    assert summary["mean_at_4_accuracy"] == 0.25
    assert len(records) == 12
    assert all(p["attempts"] == 4 for p in prompts)


def test_eval3_skips_solved_and_charges_failed_attempts():
    summary, prompts, records, _ = run("eval3", 5, count=2, length=2, correct=lambda p, a: p == 0)
    assert summary["total_output_tokens"] == 10
    assert prompts[0]["attempts"] == 1
    assert summary["num_questions_solved"] == 1
    assert sum(r["output_tokens"] for r in records if not r["score"]) == 8


def test_eval3_can_finish_when_all_solved():
    summary, _, _, _ = run("eval3", 20, length=1, correct=lambda p, a: True)
    assert summary["total_output_tokens"] == 3
    assert summary["all_questions_solved"]


def test_eval3_unvisited_questions_stay_in_denominator():
    summary, prompts, records, _ = run("eval3", 1, count=3, length=3, correct=lambda p, a: True)
    assert len(records) == 1
    assert summary["fraction_solved"] == 1 / 3
    assert sum(p["attempts"] == 0 for p in prompts) == 2


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("batch", [1, 3, 10])
def test_seeded_budget_accounting(seed, batch):
    summary, _, records, _ = run("eval3", 9, seed=seed, batch=batch, length=2)
    assert summary["total_output_tokens"] == 27
    assert records[-1]["global_budget_after"] == 0


def test_audit_detects_tampered_ledger():
    _, _, records, rows = run("eval2", 7)
    damaged = copy.deepcopy(records)
    damaged[0]["remaining_budget_after"] += 1
    with pytest.raises(ValueError, match="ledger"):
        engine_module.audit_records(damaged, protocol="eval2", budget=7, seed=0, rows=rows, per_rollout_cap=4)


def test_resume_rejects_wrong_identity_or_missing_artifacts(tmp_path):
    point = {"id": "task", "model": "m", "dataset": "d", "protocol": "eval2", "budget": 512, "seed": 0}
    manifest = {"fingerprint": "correct"}
    path = common.point_directory(tmp_path, point) / "summary.json"
    common.write_json(path, {"status": "complete", "identity": {"wrong": True}, "artifacts": {}})
    with pytest.raises(ValueError, match="Incompatible"):
        common.completed_point(tmp_path, point, manifest)
    common.write_json(path, {"status": "complete", "identity": common.point_identity(point, manifest),
                             "artifacts": {"raw": {"file": "missing", "size": 1, "sha256": "wrong"}}})
    with pytest.raises(ValueError, match="Incomplete"):
        common.completed_point(tmp_path, point, manifest)


def test_plot_requires_all_seeds_and_uses_counts_not_percentages():
    plot = importlib.import_module("plot_math_eval_matrix")
    config = {"datasets": [{"key": "d"}], "models": [{"key": "m"}], "evals": {"eval2": {"budgets": [512], "seeds": [0, 1, 2]}}}
    rows = [{"protocol": "eval2", "dataset": "d", "model": "m", "budget": 512, "seed": seed,
             "num_questions_solved": count} for seed, count in enumerate((10, 20, 30))]
    assert not plot.aggregate(rows[:2], config, "eval2")[0]["complete"]
    result = plot.aggregate(rows, config, "eval2")[0]
    assert result["mean_questions_solved"] == 20
    assert result["std_questions_solved"] == 10


def test_dataset_adapter_does_not_drop_questions_or_answer_alternatives():
    prepare = importlib.import_module("prepare_math_eval_matrix")
    spec = {"key": "tiny", "expected_rows": 2, "question_field": "q", "answer_field": "a"}
    rows = [{"q": "first", "a": ["x"]}, {"q": "second", "a": 2.0}]
    result = prepare.normalize_dataset(rows, spec, " suffix")
    assert [row["ground_truth"] for row in result] == ["x", "2"]
    assert result[0]["prompt"][0]["content"] == "first suffix"
    rows[0]["a"] = ["x", "y"]
    with pytest.raises(ValueError, match="alternatives"):
        prepare.normalize_dataset(rows, spec, "")


@pytest.mark.parametrize("exit_code", [0, 1])
def test_worker_records_child_completion_and_failures(tmp_path, monkeypatch, exit_code):
    runner = importlib.import_module("run_math_eval_matrix")
    task = {"id": "m__d__eval1", "model": "m", "dataset": "d", "protocol": "eval1"}
    config = {"evals": {"eval1": {"budgets": [256], "seeds": [0]}}}
    args = SimpleNamespace(output_root=tmp_path / "outputs", scratch=tmp_path / "scratch", gpu=2,
                           config=tmp_path / "config.json", retry_failed=False)
    monkeypatch.setattr(runner, "make_tasks", lambda config: [task])
    monkeypatch.setattr(runner, "completed_point", lambda *args, **kwargs: None)
    calls = []

    def fake_process(command, **kwargs):
        calls.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        return SimpleNamespace(pid=123, wait=lambda: exit_code)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_process)
    assert runner.worker(args, config, {"fingerprint": "test"}) == 0
    assert len(calls) == 1 and calls[0][1] == "2"
    suffix = ".done.json" if exit_code == 0 else ".failed.json"
    assert (args.output_root / "task_state" / (task["id"] + suffix)).is_file()
    if exit_code:
        runner.worker(args, config, {"fingerprint": "test"})
        assert len(calls) == 1  # No unrequested retry loop.


def test_plot_renders_three_formats_without_gpu(tmp_path):
    pytest.importorskip("matplotlib")
    plot = importlib.import_module("plot_math_eval_matrix")
    config = {"datasets": [{"key": "d", "label": "Synthetic CPU test", "expected_rows": 100}],
              "models": [{"key": "m", "label": "Synthetic"}],
              "evals": {"eval2": {"budgets": [512, 1024], "seeds": [0, 1, 2]}}}
    rows = [{"dataset": "d", "model": "m", "budget": b, "complete": True,
             "mean_questions_solved": n, "std_questions_solved": 2} for b, n in [(512, 10), (1024, 20)]]
    plot.render_plot(tmp_path / "cpu_test_plot", rows, config, "eval2")
    for extension in ("png", "pdf", "svg"):
        assert (tmp_path / ("cpu_test_plot." + extension)).stat().st_size > 1000


def test_holder_launcher_pins_gh200_build_environment():
    script = (SCRIPTS / "run_math_eval_matrix_on_holder.sh").read_text()
    assert "TORCH_CUDA_ARCH_LIST=9.0" in script
    assert "cuda-12.6.1" in script
    assert "gcc-native/13" in script
    assert "MAX_JOBS=4" in script
    assert "--retry-failed" in script
    assert "MAXRL_EVAL_REUSE_EVAL1_FROM" in script


def test_reuse_eval1_audits_copies_and_retains_provenance(tmp_path):
    runner = importlib.import_module("run_math_eval_matrix")
    source, destination = tmp_path / "before", tmp_path / "after"
    summary, prompts, records, rows = run("eval1", 4)
    dataset_path = destination / "datasets" / "d.json"
    common.write_json(dataset_path, rows)
    dataset = {"sha256": common.sha256(dataset_path), "path": str(dataset_path)}
    config = {"models": [{"key": "m"}], "datasets": [{"key": "d", "expected_rows": 3}],
              "evals": {"eval1": {"budgets": [4], "seeds": [0]},
                        "eval2": {"budgets": [7], "seeds": [0], "stop_on_first_success": True}},
              "sampling": {"per_rollout_cap": 4}}
    identity = {"config": config, "packages": {}, "models": {}, "datasets": {"d": {"sha256": dataset["sha256"]}},
                "source_sha256": {"qwen3_experiments/math_eval_matrix_common.py": "same",
                                  "verl/workers/reward_manager/multi_thread_naive.py": "same"}}
    manifest = {**identity, "fingerprint": common.fingerprint(identity)}
    old_identity = copy.deepcopy(identity)
    old_identity["config"]["evals"]["eval2"].pop("stop_on_first_success")
    previous = {**old_identity, "fingerprint": common.fingerprint(old_identity)}
    common.write_json(source / "manifest.json", previous)
    common.write_json(destination / "prepared_inputs.json", {"datasets": {"d": dataset}})
    task = common.make_tasks(config)[0]
    point = next(common.task_points(config, task))
    directory = common.point_directory(source, point)
    common.write_json(directory / "attempt_test" / "prompts.json", prompts)
    with gzip.open(directory / "attempt_test" / "rollouts.jsonl.gz", "wt") as stream:
        stream.writelines(json.dumps(record) + "\n" for record in records)
    artifacts = {key: {"file": "attempt_test/" + name,
                       "size": (directory / "attempt_test" / name).stat().st_size,
                       "sha256": common.sha256(directory / "attempt_test" / name)}
                 for key, name in [("prompts", "prompts.json"), ("rollouts", "rollouts.jsonl.gz")]}
    common.write_json(directory / "summary.json", {**summary, "identity": common.point_identity(point, previous),
                                                   "status": "complete", "artifacts": artifacts})
    old_summary_hash = common.sha256(directory / "summary.json")
    assert runner.reuse_eval1_points(source, destination, manifest) == 1
    reused = common.completed_point(destination, point, manifest)
    assert reused["reused_from"]["summary_sha256"] == old_summary_hash
    assert reused["reused_from"]["manifest_fingerprint"] == previous["fingerprint"]
    assert reused["mean_at_4_accuracy"] == summary["mean_at_4_accuracy"]
    assert common.sha256(directory / "summary.json") == old_summary_hash
    assert runner.reuse_eval1_points(source, destination, manifest) == 0
    changed = copy.deepcopy(identity)
    changed["config"]["sampling"]["per_rollout_cap"] = 8
    with pytest.raises(ValueError, match="identical inputs and settings"):
        runner.reuse_eval1_points(source, destination, {**changed, "fingerprint": common.fingerprint(changed)})
