"""After-thinking grading, exact individual budgets and audited queue handoff."""

from types import SimpleNamespace

import pytest

from qwen3_experiments import minerva_individual_budget as evaluator
from qwen3_experiments.eval_l0_final import answer_suffix
from qwen3_experiments.math_eval_matrix_common import rollout_seed


def make_predecessor(tmp_path):
    parent, previous = tmp_path / "training", tmp_path / "nine"
    plan = {"parent_run": str(parent), "parent_eval_root": str(previous),
            "models": {"compression_step100": {"repo": "owner/compression-step_100"}}}
    evaluator.write(parent / "status.json", {"state": "complete", "exit_code": 0, "last_completed_step": 100})
    evaluator.write(previous / "queue_status.json", {"state": "complete"})
    evaluator.write(previous / "status.json", {"state": "complete"})
    evaluator.write(previous / "report/audit.json", {"complete": True, "questions": 1819,
                                                    "responses_verified": 7276, "budget_points": 54})
    evaluator.write(previous / "final_checkpoint_receipt.json", {
        "repo_id": "owner/compression-step_100", "checkpoint": "global_step_100", "state": "archived_and_deleted",
    })
    return plan, parent, previous


@pytest.mark.parametrize("datasets", [["minervamath"], list(evaluator.DATASETS)])
def test_prepare_checkpoint_only_pins_l4096_without_preparing_other_models(tmp_path, monkeypatch, datasets):
    from transformers import AutoTokenizer

    parent, previous, root = tmp_path / "training", tmp_path / "nine", tmp_path / "minerva"
    model = tmp_path / "base"
    evaluator.write(parent / "plan.json", {
        "job_id": "123", "python_bin": "python", "evaluation_root": str(previous), "holder_locks": [],
        "model_revision": evaluator.MODEL_REVISION, "model_path": str(model), "hf_repo_prefix": "owner/l4096",
        "input_hashes": {str(model / "model.safetensors"): "a" * 64},
    })
    evaluator.write(previous / "plan.json", {"training_root": str(parent), "job_id": "123", "frozen_files": {},
                                             "model_repo": "owner/l4096-step_100", "model_label": "L+4096"})
    evaluator.write(previous / "questions.json", [
        {"id": f"{key}_{i}", "dataset": key, "gold": "42", "messages": [{"role": "user", "content": "Q"}],
         "prompt_token_ids": [1, 2], "source_row": i} for key, (_, count) in evaluator.DATASETS.items() for i in range(count)])
    evaluator.write(previous / "input_plan.json", {
        "datasets": [{"key": key, "revision": "pinned", "rows": count} for key, (_, count) in evaluator.DATASETS.items()]})
    (previous / "provenance").mkdir()
    for name in ("eval_l0_final.py", "eval_polaris_step80.py"):
        (previous / "provenance" / name).write_text("# Frozen grader\n")
    monkeypatch.setattr(evaluator, "require_compute", lambda _: "compute")
    monkeypatch.setattr(evaluator.subprocess, "check_output", lambda *_, **__: "agent/add-math12k-maxrl-launcher\n")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_, **__: SimpleNamespace(
        apply_chat_template=lambda *_, **__: [1, 2]))
    monkeypatch.setattr(evaluator, "prepare_deepseek", lambda *args: pytest.fail("Unrequested model preparation"))
    args = SimpleNamespace(parent_run=parent, output_root=root, seed=0, checkpoint_only=True, datasets=datasets)
    plan = evaluator.prepare(args)
    try:
        assert set(plan["models"]) == {"compression_step100"}
        assert plan["models"]["compression_step100"]["repo"] == "owner/l4096-step_100"
        assert plan["models"]["compression_step100"]["label"] == "Compression L+4096 step 100"
        assert len(evaluator.pending_points(root, plan, {"fingerprint": "pinned"})) == 5 * len(datasets)
        assert plan["num_questions"] == sum(evaluator.DATASETS[key][1] for key in datasets)
        assert [spec["key"] for spec in plan["datasets"]] == datasets
        for key in datasets:
            selected = evaluator.selected_rows(plan, evaluator.read(root / "questions.json"), key)
            assert [row["unique_id"] for row in selected] == [f"{key}_{i}" for i in range(evaluator.DATASETS[key][1])]
        assert evaluator.prepare(args) == plan
        args.datasets = ["math500"]
        with pytest.raises(ValueError, match="dataset selection is immutable"):
            evaluator.prepare(args)
        args.datasets = datasets
        args.checkpoint_only = False
        with pytest.raises(ValueError, match="model selection is immutable"):
            evaluator.prepare(args)
    finally:
        evaluator.Path(plan["scratch"]).rmdir()


def test_queue_requires_completed_nine_dataset_evaluation(tmp_path):
    plan, _, previous = make_predecessor(tmp_path)
    assert evaluator.dependency_ready(plan)
    for value in ("waiting_for_training_and_verified_archives", "running", "grading_prefixes"):
        evaluator.write(previous / "queue_status.json", {"state": value})
        assert not evaluator.dependency_ready(plan)


@pytest.mark.parametrize("damage", ["missing_question", "missing_response", "missing_budget", "wrong_checkpoint", "failed"])
def test_queue_does_not_use_partial_or_wrong_results(tmp_path, damage):
    plan, _, previous = make_predecessor(tmp_path)
    if damage == "failed":
        evaluator.write(previous / "status.json", {"state": "failed"})
    elif damage == "wrong_checkpoint":
        p = previous / "final_checkpoint_receipt.json"
        receipt = evaluator.read(p)
        receipt["repo_id"] = "owner/polaris-step_100"
        evaluator.write(p, receipt)
    else:
        p = previous / "report/audit.json"
        audit = evaluator.read(p)
        key = {"missing_question": "questions", "missing_response": "responses_verified", "missing_budget": "budget_points"}[damage]
        audit[key] -= 1
        evaluator.write(p, audit)
    with pytest.raises((ValueError, RuntimeError)):
        evaluator.dependency_ready(plan)


@pytest.mark.parametrize("text", ["\\boxed{42}", "<think>\\boxed{42}", "<think>x</think>  ",
                                 "<think>x</think>42<think>not finished"])
def test_invalid_thinking_cannot_earn_reward(text):
    def unexpected(_):
        pytest.fail("An invalid thinking suffix must not reach Math-Verify")

    assert evaluator.after_thinking_score(text, "42", answer_suffix, unexpected)["score"] == 0


def test_only_final_suffix_is_graded_without_extra_eos_or_box_requirement():
    received = []

    def grade(item):
        received.append(item)
        return {"correct": 1.0, "grader_status": "scored"}

    result = evaluator.after_thinking_score("<think>wrong answer 5</think> The answer is 42.", "42", answer_suffix, grade)
    assert result["score"] == 1
    assert received == [("The answer is 42.", "42")]


def test_grader_timeout_does_not_silently_count_as_a_valid_experiment():
    with pytest.raises(RuntimeError, match="TimeoutError"):
        evaluator.after_thinking_score("<think>x</think>42", "42", answer_suffix,
                                       lambda _: {"correct": 0.0, "grader_status": "TimeoutError"})


@pytest.mark.parametrize("budget", evaluator.BUDGETS)
def test_requested_budgets_cap_each_attempt_and_never_redistribute_saved_tokens(budget):
    records, caps = [], []
    rows = [{"unique_id": str(p), "ground_truth": "42"} for p in range(2)]

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "<think>x</think>42" if ids[0] == 1 else "<think>unfinished"

    class Engine:
        def generate(self, *, prompt_token_ids, sampling_params, **kwargs):
            outputs = []
            for prompt, params in zip(prompt_token_ids, sampling_params):
                p = prompt[0]
                assert params.temperature == 0.6 and params.top_p == 0.95 and params.top_k == 20
                assert params.ignore_eos is False and params.n == 1
                # Question 0 succeeds immediately; question 1 consumes its own complete allowance.
                caps.append((p, params.max_tokens))
                length = 10 if p == 0 else params.max_tokens
                ids = [int(p == 0)] * length
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids, finish_reason="stop" if p == 0 else "length")]))
            return outputs

    def score_many(items):
        return [evaluator.after_thinking_score(text, gold, answer_suffix,
                 lambda _: {"correct": 1.0, "grader_status": "scored"})["score"] for text, gold in items]

    sampler = {"per_rollout_cap": 32768, "max_batch_size": 32, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    summary, prompts = evaluator.evaluate_point(
        protocol="eval2", budget=budget, seed=0, rows=rows, prompt_token_ids=[[0], [1]],
        engine=Engine(), sampling_params_type=SimpleNamespace, tokenizer=Tokenizer(), score_many=score_many,
        emit=records.append, sampling=sampler, stop_on_first_success=True,
    )
    audit, audited_prompts = evaluator.audit_records(records, protocol="eval2", budget=budget, seed=0,
                                                     rows=rows, per_rollout_cap=32768, stop_on_first_success=True)
    assert audit["fraction_solved"] == summary["fraction_solved"] == 0.5
    assert prompts == audited_prompts
    assert [p["output_tokens"] for p in prompts] == [10, budget]
    assert summary["unused_output_budget"] == budget - 10
    assert len([r for r in records if r["prompt_position"] == 0]) == 1
    assert all(cap <= 32768 for _, cap in caps)
    assert [cap for p, cap in caps if p == 1] == ([budget] if budget <= 32768 else [32768, budget - 32768])
    assert records[0]["rollout_seed"] == rollout_seed(0, 0, 0)


def test_completed_point_rejects_changed_saved_responses(tmp_path):
    directory = evaluator.point_directory(tmp_path, "qwen3_1_7b", 8192)
    directory.mkdir(parents=True)
    raw = directory / "rollouts.jsonl.gz"
    raw.write_bytes(b"saved response")
    summary = {"state": "complete", "identity": {"manifest": "pinned", "model": "qwen3_1_7b", "budget": 8192},
               "artifacts": {"rollouts": {"file": raw.name, "size": raw.stat().st_size, "sha256": evaluator.digest(raw)}}}
    evaluator.write(directory / "summary.json", summary)
    assert evaluator.completed_point(tmp_path, "qwen3_1_7b", 8192, {"fingerprint": "pinned"}) == summary
    raw.write_bytes(b"other response")
    with pytest.raises(ValueError, match="changed"):
        evaluator.completed_point(tmp_path, "qwen3_1_7b", 8192, {"fingerprint": "pinned"})


def test_dataset_results_cannot_be_reused_for_another_benchmark(tmp_path):
    manifest = {"fingerprint": "fixed"}
    destination = evaluator.point_directory(tmp_path, "compression_step100", 8192, "math500")
    evaluator.write(destination / "summary.json", {
        "state": "complete", "artifacts": {},
        "identity": evaluator.point_identity("compression_step100", 8192, manifest, "minervamath"),
    })
    with pytest.raises(ValueError, match="different identity"):
        evaluator.completed_point(tmp_path, "compression_step100", 8192, manifest, "math500")


def test_dataset_filter_preserves_seed_order_and_rejects_mixed_or_missing_questions():
    plan = {"datasets": [{"key": "minervamath", "rows": 2}, {"key": "aime24", "rows": 1}]}
    rows = [{"unique_id": "aime24_0", "dataset": "aime24"},
            {"unique_id": "minerva_0", "dataset": "minervamath"},
            {"unique_id": "minerva_1", "dataset": "minervamath"}]
    selected = evaluator.selected_rows(plan, rows, "minervamath")
    assert [row["unique_id"] for row in selected] == ["minerva_0", "minerva_1"]
    assert [rollout_seed(0, index, 0) for index, _ in enumerate(selected)] == [rollout_seed(0, 0, 0), rollout_seed(0, 1, 0)]
    for broken in (rows[:-1], rows + rows[1:2]):
        with pytest.raises(ValueError, match="Incomplete or duplicate"):
            evaluator.selected_rows(plan, broken, "minervamath")
    with pytest.raises(ValueError, match="one dataset"):
        evaluator.selected_rows(plan, rows, None)


def test_deepseek_prefilled_thinking_is_graded_after_the_generated_closing_tag():
    result = evaluator.after_thinking_score("reasoning started in the prompt</think>42", "42", answer_suffix,
                                            lambda item: {"correct": float(item == ("42", "42")), "grader_status": "scored"})
    assert result["score"] == 1


def test_native_prompts_preserve_questions_but_use_model_specific_token_ids():
    row = {"unique_id": "minerva:0", "ground_truth": "42", "prompt": [{"role": "user", "content": "Question?"}],
           "prompt_token_ids": [11, 12]}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == row["prompt"] and kwargs["enable_thinking"] and kwargs["add_generation_prompt"]
            return "Question?<｜Assistant｜><think>\n"

        def encode(self, text, **kwargs):
            assert not kwargs["add_special_tokens"]
            return [21, 22, 23]

        def decode(self, ids, **kwargs):
            return "Question?<｜Assistant｜><think>\n"

    converted = evaluator.thinking_rows([row], Tokenizer(), "deepseek")[0]
    assert converted == {**row, "prompt_token_ids": [21, 22, 23]}
    assert row["prompt_token_ids"] == [11, 12]
    with pytest.raises(ValueError, match="Unexpected qwen3"):
        evaluator.thinking_rows([row], Tokenizer(), "qwen3")


def test_dispatch_covers_all_five_models_and_resumes_only_unfinished_points(tmp_path):
    keys = ["qwen3_1_7b", "compression_step100", *[m["key"] for m in evaluator.DEEPSEEK_MODELS]]
    plan = {"models": dict.fromkeys(keys), "budgets": evaluator.BUDGETS}
    manifest = {"fingerprint": "fixed"}
    pending = evaluator.pending_points(tmp_path, plan, manifest)
    assert len(pending) == len(set(pending)) == 25
    assert set(pending) == {(key, budget, None) for key in keys for budget in evaluator.BUDGETS}
    assert pending[:5] == [(key, 65536, None) for key in keys]
    key, budget, _ = pending[3]
    evaluator.write(evaluator.point_directory(tmp_path, key, budget) / "summary.json", {
        "state": "complete", "identity": {"manifest": "fixed", "model": key, "budget": budget}, "artifacts": {},
    })
    resumed = evaluator.pending_points(tmp_path, plan, manifest)
    assert len(resumed) == 24 and (key, budget, None) not in resumed


@pytest.mark.parametrize("expanded", [False, True])
def test_controller_dispatches_every_point_and_complete_resume_launches_no_workers(tmp_path, monkeypatch, expanded):
    import torch

    keys = ["qwen3_1_7b", "compression_step100", *[m["key"] for m in evaluator.DEEPSEEK_MODELS]]
    if expanded:
        keys = ["compression_step100"]
    plan = {"models": {key: {"label": key} for key in keys}, "budgets": evaluator.BUDGETS, "seed": 0,
            "job_id": "146103", "holder_locks": [], "runtime": str(tmp_path), "python_bin": "python"}
    manifest = {"fingerprint": "fixed"}
    if expanded:
        plan["datasets"] = [{"key": key, "rows": count} for key, (_, count) in evaluator.DATASETS.items()]
    datasets = list(evaluator.DATASETS) if expanded else [None]
    expected_points = len(keys) * len(datasets) * len(evaluator.BUDGETS)
    launched = []

    def spawn(command, **kwargs):
        key, budget = command[command.index("--model") + 1], int(command[command.index("--budget") + 1])
        dataset = command[command.index("--dataset") + 1] if "--dataset" in command else None
        count = evaluator.DATASETS[dataset][1] if dataset else 272
        launched.append((key, budget, dataset, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        evaluator.write(evaluator.point_directory(tmp_path, key, budget, dataset) / "summary.json", {
            "state": "complete", "identity": evaluator.point_identity(key, budget, manifest, dataset), "artifacts": {},
            "pass_at_budget_percent": 100 * (count // 2) / count, "num_questions_solved": count // 2, "num_prompts": count,
            "total_rollouts": count, "total_output_tokens": count * 10, "unused_output_budget": count * (budget - 10),
        })
        return SimpleNamespace(pid=len(launched), returncode=0, poll=lambda: 0)

    monkeypatch.setattr(evaluator, "require_compute", lambda _: "compute")
    monkeypatch.setattr(evaluator, "dependency_ready", lambda _: True)
    monkeypatch.setattr(evaluator, "prepare_models", lambda *_: manifest)
    monkeypatch.setattr(evaluator.subprocess, "check_output", lambda *_, **__: "")
    monkeypatch.setattr(evaluator.subprocess, "Popen", spawn)
    monkeypatch.setattr(evaluator.time, "sleep", lambda _: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert evaluator.run(tmp_path, plan) == 0
    assert len(launched) == expected_points
    assert {(key, budget, dataset) for key, budget, dataset, _ in launched} == {
        (key, b, dataset) for key in keys for b in evaluator.BUDGETS for dataset in datasets}
    assert {gpu for _, _, _, gpu in launched} == {str(i) for i in range(8)}
    assert evaluator.read(tmp_path / "report/audit.json")["points"] == expected_points
    metrics = evaluator.read(tmp_path / "report/metrics.json")
    assert all(row["questions"] == evaluator.DATASETS[row["dataset"]][1] for row in metrics)
    assert evaluator.run(tmp_path, plan) == 0
    assert len(launched) == expected_points


def make_waiting_queue(tmp_path, monkeypatch):
    previous = tmp_path / "old"
    old = {"job_id": "146103", "node": "compute", "parent_run": "training", "parent_eval_root": "nine",
           "seed": 0, "budgets": evaluator.BUDGETS, "sampling": {"temperature": .6}, "grading": {"after_thinking_only": True},
           "models": {"qwen3_1_7b": {"repo": "Qwen/Qwen3-1.7B", "revision": "pinned", "path": "weights"}}}
    expanded = {**old, "models": {**old["models"], **{m["key"]: m for m in evaluator.DEEPSEEK_MODELS}}}
    monkeypatch.setattr(evaluator, "verify_plan", lambda root: old)
    evaluator.write(previous / "queue_status.json", {"state": "waiting_for_nine_dataset_evaluation"})
    evaluator.write(previous / "launch.json", {"pid": 123})
    return previous, expanded


def test_queue_expansion_preserves_original_models_and_settings(tmp_path, monkeypatch):
    previous, expanded = make_waiting_queue(tmp_path, monkeypatch)
    assert evaluator.check_waiting_replacement(previous, expanded) == {"pid": 123}
    with pytest.raises(ValueError, match="sampling"):
        evaluator.check_waiting_replacement(previous, {**expanded, "sampling": {"temperature": .8}})
    changed = {**expanded, "models": {k: v for k, v in expanded["models"].items() if k != "qwen3_1_7b"}}
    with pytest.raises(ValueError, match="every previously requested model"):
        evaluator.check_waiting_replacement(previous, changed)


@pytest.mark.parametrize("damage", [None, "reorder", "prompt", "revision", "remove"])
def test_dataset_expansion_preserves_original_questions_prompts_and_revisions(tmp_path, monkeypatch, damage):
    previous, expanded = make_waiting_queue(tmp_path, monkeypatch)
    old = evaluator.verify_plan(previous)
    original_spec = {"key": "minervamath", "rows": 2, "revision": "pinned"}
    old["dataset"] = original_spec
    expanded["datasets"] = [dict(original_spec), {"key": "math500", "rows": 1, "revision": "math-revision"}]
    expanded["output_root"] = str(tmp_path / "new")
    # make_waiting_queue shares this preserved model between the original and replacement.
    old["models"]["qwen3_1_7b"]["questions_file"] = "questions.json"
    original_rows = [{"unique_id": f"minerva_{i}", "prompt": [{"role": "user", "content": str(i)}],
                      "prompt_token_ids": [i], "ground_truth": "42"} for i in range(2)]
    evaluator.write(previous / "questions.json", original_rows)
    new_rows = [{"unique_id": "math500_0", "dataset": "math500"},
                *[{**row, "dataset": "minervamath"} for row in original_rows]]
    if damage == "reorder":
        new_rows.reverse()
    elif damage == "prompt":
        new_rows[1] = {**new_rows[1], "prompt_token_ids": [999]}
    elif damage == "revision":
        expanded["datasets"][0]["revision"] = "changed"
    elif damage == "remove":
        expanded["datasets"] = expanded["datasets"][1:]
    evaluator.write(tmp_path / "new/questions.json", new_rows)
    if damage:
        with pytest.raises(ValueError, match="dataset|questions, prompts or seed order"):
            evaluator.check_waiting_replacement(previous, expanded)
    else:
        assert evaluator.check_waiting_replacement(previous, expanded) == {"pid": 123}


@pytest.mark.parametrize("artifact", ["execution.json", "execution_manifest.json", "results", "status.json"])
def test_replacement_is_rejected_if_generation_has_ever_started(tmp_path, monkeypatch, artifact):
    previous, expanded = make_waiting_queue(tmp_path, monkeypatch)
    (previous / artifact).touch()
    with pytest.raises(ValueError, match="execution already began"):
        evaluator.check_waiting_replacement(previous, expanded)


def test_replacement_is_rejected_if_old_queue_is_no_longer_waiting(tmp_path, monkeypatch):
    previous, expanded = make_waiting_queue(tmp_path, monkeypatch)
    evaluator.write(previous / "queue_status.json", {"state": "running_or_waiting_for_resources"})
    with pytest.raises(ValueError, match="Only a waiting"):
        evaluator.check_waiting_replacement(previous, expanded)


def test_pidfd_fallback_supports_conda_builds_without_the_os_wrapper(monkeypatch):
    import os
    import platform
    import signal
    import sys

    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        pytest.skip("Linux pidfd check")
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    descriptor = evaluator.open_pidfd(os.getpid())
    try:
        assert not os.get_inheritable(descriptor)
        signal.pidfd_send_signal(descriptor, 0)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("base_repo", ["Qwen/Qwen3-1.7B-Base", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"])
def test_deepspeed_conversion_keeps_actor_values_for_the_requested_base(tmp_path, monkeypatch, base_repo):
    import shutil

    import huggingface_hub
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    import prepare_math_eval_matrix as preparation

    base, source = tmp_path / "base", tmp_path / "source"
    base.mkdir()
    (source / "global_step100").mkdir(parents=True)
    state = {"model.embed_tokens.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    evaluator.write(base / "config.json", {"tie_word_embeddings": False})
    save_file({key: value.bfloat16() for key, value in state.items()}, base / "model.safetensors")
    evaluator.write(source / "train_config.json", {"pretrain": base_repo, "zero_stage": 2, "lora_rank": 0})
    torch.save({"module": state}, source / "global_step100/mp_rank_00_model_states.pt")

    def download(spec, names, destination):
        result = {}
        for name in names:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, path)
            result[name] = {"size": path.stat().st_size, "sha256": evaluator.digest(path)}
        return result

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=lambda *_, **__: SimpleNamespace(siblings=[])))
    monkeypatch.setattr(preparation, "download_files", download)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_, **__: SimpleNamespace(
        get_vocab=lambda: {"test": 0}, chat_template="native thinking"))
    spec = {"key": "test", "repo": "owner/checkpoint", "revision": "pinned", "format": "deepspeed", "step": 100}
    if base_repo.startswith("deepseek"):
        spec["base_repo"] = base_repo
    receipt = preparation.prepare_model(spec, tmp_path / "scratch", base)
    converted = load_file(receipt["path"] + "/model.safetensors")
    assert set(converted) == set(state)
    for key, value in state.items():
        assert converted[key].dtype == torch.bfloat16 and torch.equal(converted[key], value.bfloat16())
    assert receipt["spec"] == spec and receipt["tensor_count"] == 1
