"""Check final-checkpoint gating, GPU handoff, and after-thinking denominators."""

import copy
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "qwen3_experiments"
SPEC = importlib.util.spec_from_file_location("l0_eval_test", SCRIPTS / "eval_l0_final.py")
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def final_state():
    status = {"variant": "per_context_rb_l0_0", "adv_estimator": "fixed_n_rb_offset_cost_aware_marginrl",
              "cost_offset_tokens": 0, "total_steps": 100, "last_completed_step": 100,
              "state": "complete", "exit_code": 0}
    receipt = {"checkpoint": "global_step_100", "repo_id": "owner/l0-step_100", "state": "verified",
               "remote_commit": "a" * 40, "verified_at": "2026-09-21T10:00:00Z",
               "files": {f"actor/model_world_size_8_rank_{i}.pt": {"size": 10, "sha256": "b" * 64}
                         for i in range(8)}}
    return status, receipt


def test_checkpoint_is_released_only_after_successful_exit_and_verification():
    status, receipt = final_state()
    assert EVAL.checkpoint_ready(status, receipt, "owner/l0-step_100")
    assert not EVAL.checkpoint_ready(status, None, "owner/l0-step_100")
    for change in ({"state": "training"}, {"last_completed_step": 99}, {"exit_code": 1}, {"state": "failed"}):
        assert not EVAL.checkpoint_ready({**status, **change}, receipt, "owner/l0-step_100")
    assert not EVAL.checkpoint_ready(status, {**receipt, "state": "uploading"}, "owner/l0-step_100")


def test_wrong_checkpoint_or_incomplete_shards_are_rejected():
    status, receipt = final_state()
    for change in ({"checkpoint": "global_step_80"}, {"repo_id": "owner/another-step_100"}, {"remote_commit": "main"}):
        with pytest.raises(AssertionError):
            EVAL.checkpoint_ready(status, {**receipt, **change}, "owner/l0-step_100")
    broken = copy.deepcopy(receipt)
    del broken["files"]["actor/model_world_size_8_rank_7.pt"]
    with pytest.raises(KeyError):
        EVAL.checkpoint_ready(status, broken, "owner/l0-step_100")


def test_l4096_requires_its_own_variant_and_successful_final_archive(tmp_path):
    base = EVAL.load_module("l4096_state_test_core", SCRIPTS / "eval_polaris_step80.py")
    status, receipt = final_state()
    status.update(variant="per_context_rb_l0_4096", cost_offset_tokens=4096)
    receipt["repo_id"] = "owner/l4096-step_100"
    base.write(tmp_path / "status.json", status)
    assert EVAL.training_status(tmp_path, base, "l4096") == status
    assert EVAL.checkpoint_ready(status, receipt, receipt["repo_id"], "l4096")
    for change in ({"state": "training"}, {"last_completed_step": 99}, {"exit_code": 1}):
        assert not EVAL.checkpoint_ready({**status, **change}, receipt, receipt["repo_id"], "l4096")
    assert not EVAL.checkpoint_ready(status, None, receipt["repo_id"], "l4096")
    for change in ({"variant": "per_context_rb_l0_0"}, {"cost_offset_tokens": 0}):
        with pytest.raises(AssertionError):
            EVAL.checkpoint_ready({**status, **change}, receipt, receipt["repo_id"], "l4096")
    with pytest.raises(AssertionError):
        EVAL.checkpoint_ready(status, receipt, "owner/different-step_100", "l4096")
    with pytest.raises(AssertionError):
        EVAL.checkpoint_ready(status, receipt, receipt["repo_id"], "l0")


@pytest.mark.parametrize("label", ["L+4096", "f_cov", "MaxRL"])
def test_compression_report_uses_after_thinking_reference_and_its_own_label(tmp_path, label):
    base = EVAL.load_module("l4096_report_test_core", SCRIPTS / "eval_polaris_step80.py")
    base.write(tmp_path / "plan.json", {"model_label": label, "total_responses": 7276,
                                        "training_root": str(tmp_path / "training")})
    base.write(tmp_path / "training/plan.json", {"dataset_repo": "zjhhhh/compression_dataset"})
    base.write(tmp_path / "prepared_inputs.json", {"model": {"repo": "owner/l4096", "revision": "a" * 40}})
    (tmp_path / "provenance").mkdir()
    for name in ("main_baseline.csv", "budget_baseline.csv"):
        (tmp_path / "provenance" / name).write_text("unused\n")
    reference = tmp_path / "report/qwen3_after_thinking"
    reference.mkdir(parents=True)
    (reference / "source_report.md").write_text("Audited reference")
    base.write_csv(reference / "main.csv", [
        {"Model": "qwen3-1.7B", "Metric": metric, **{title: "50.00" for _, title, _ in EVAL.DATASETS}}
        for metric in ("mean@4 (%)", "pass@4 (%)", "Mean Response Length (tokens)")])
    base.write_csv(reference / "budgets.csv", [
        {"model": "qwen3-1.7B", "dataset": key, **{f"{cap // 1024}k": "50.00" for cap in EVAL.CAPS}}
        for key, _, _ in EVAL.DATASETS])
    base.write(reference / "audit.json", {
        "complete": True, "after_thinking_only": True, "model_repo": "Qwen/Qwen3-1.7B",
        "files_sha256": {name: base.digest(reference / name) for name in ("main.csv", "budgets.csv", "source_report.md")},
    })
    metrics = [{"dataset": key, "cap_tokens": cap, "mean_at_4_percent": 55., "pass_at_4_percent": 60.,
                "mean_output_tokens": 500.} for key, _, _ in EVAL.DATASETS for cap in EVAL.CAPS]
    EVAL.write_report(tmp_path, base, metrics)
    report = (tmp_path / "report/README.md").read_text()
    assert f"Qwen3-1.7B vs {label} step 100 (compression)" in report
    assert "same after-thinking-only scoring policy" in report
    assert "Historical Qwen3, MaxRL and ER" not in report


def test_grpo_requires_matching_training_exit_and_successful_archive(tmp_path):
    base = EVAL.load_module("grpo_state_test_core", SCRIPTS / "eval_polaris_step80.py")
    base.write(tmp_path / "plan.json", {"output_root": str(tmp_path), "total_steps": 100,
                                      "hf_repo_prefix": "owner/grpo"})
    raw = {"state": "training", "last_completed_step": 2327}
    base.write(tmp_path / "status.json", raw)
    (tmp_path / "train.log").write_text(
        "\x1b[36m(TaskRunner pid=123)\x1b[0m step:16 - timing_s/step:2327.4 - actor/loss:0\n")
    status = EVAL.training_status(tmp_path, base, "grpo")
    assert status["last_completed_step"] == 16 and status["exit_code"] is None
    _, receipt = final_state()
    receipt["repo_id"] = "owner/grpo-step_100"
    assert not EVAL.checkpoint_ready(status, receipt, receipt["repo_id"], "grpo")

    raw.update(state="complete", last_completed_step=100, training_exit_code=0, archive_exit_code=0)
    base.write(tmp_path / "status.json", raw)
    assert not EVAL.checkpoint_ready(EVAL.training_status(tmp_path, base, "grpo"), receipt,
                                     receipt["repo_id"], "grpo")
    base.write(tmp_path / "training_exit.json", {"exit_code": 1})
    assert not EVAL.checkpoint_ready(EVAL.training_status(tmp_path, base, "grpo"), receipt,
                                     receipt["repo_id"], "grpo")
    base.write(tmp_path / "training_exit.json", {"exit_code": 0})
    status = EVAL.training_status(tmp_path, base, "grpo")
    assert EVAL.checkpoint_ready(status, receipt, receipt["repo_id"], "grpo")
    for code in (None, 1):
        assert not EVAL.checkpoint_ready({**status, "archive_exit_code": code}, receipt,
                                         receipt["repo_id"], "grpo")


def test_queue_rejects_login_node_and_another_users_allocation(monkeypatch):
    plan = {"queue_node": "compute", "job_id": "123"}
    monkeypatch.setattr(EVAL.os, "uname", lambda: SimpleNamespace(nodename="login"))
    with pytest.raises(AssertionError, match="Run the queue on compute node"):
        EVAL.require_queue_node(plan)
    monkeypatch.setattr(EVAL.os, "uname", lambda: SimpleNamespace(nodename="compute"))
    monkeypatch.setattr(EVAL.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=f"JobState=RUNNING BatchHost=compute UserId=another({os.getuid() + 1})"))
    with pytest.raises(AssertionError, match="another user"):
        EVAL.require_queue_node(plan)


def test_grpo_queue_waits_for_archive_and_retries_busy_gpu_handoff(tmp_path, monkeypatch):
    base = EVAL.load_module("grpo_queue_test_core", SCRIPTS / "eval_polaris_step80.py")
    training = tmp_path / "training"
    base.write(training / "plan.json", {"output_root": str(training), "total_steps": 100,
                                       "hf_repo_prefix": "owner/grpo"})
    base.write(training / "status.json", {"state": "training", "last_completed_step": 99})
    _, receipt = final_state()
    receipt["repo_id"] = "owner/grpo-step_100"
    base.write(training / "hf_checkpoint_archive/receipts/global_step_100.json", receipt)
    plan = {"training_root": str(training), "training_kind": "grpo", "job_id": "123",
            "model_repo": receipt["repo_id"], "total_responses": 7276, "evaluation_job_name": "grpo-final-eval"}
    monkeypatch.setattr(EVAL, "verify_plan", lambda *args: plan)
    monkeypatch.setattr(EVAL, "require_queue_node", lambda *args: None)
    monkeypatch.setenv("USER", "testuser")
    monkeypatch.setattr(EVAL.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="JobState=RUNNING UserId=testuser(123)"))
    sleeps, launches = [], []

    def advance_training(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            assert not launches
            base.write(training / "training_exit.json", {"exit_code": 0})
            base.write(training / "status.json", {"state": "archiving", "last_completed_step": 100,
                                                  "training_exit_code": 0})
        elif len(sleeps) == 2:
            assert not launches
            base.write(training / "status.json", {"state": "complete", "last_completed_step": 100,
                                                  "training_exit_code": 0, "archive_exit_code": 0})
        elif len(sleeps) > 3:
            pytest.fail("Queue did not finish after the GPU handoff")

    def launch(command, **kwargs):
        launches.append(command)
        assert base.read(tmp_path / "final_checkpoint_receipt.json") == receipt
        if len(launches) == 1:
            return SimpleNamespace(pid=123, wait=lambda: 75)
        base.write(tmp_path / "report/audit.json", {"complete": True, "responses_verified": 7276})
        base.write(tmp_path / "status.json", {"state": "complete"})
        return SimpleNamespace(pid=124, wait=lambda: 0)

    monkeypatch.setattr(EVAL.time, "sleep", advance_training)
    monkeypatch.setattr(EVAL.subprocess, "Popen", launch)
    EVAL.queue(tmp_path, base)
    assert len(sleeps) == 3 and len(launches) == 2
    assert all("--job-name=grpo-final-eval" in command for command in launches)
    assert base.read(tmp_path / "queue_status.json")["state"] == "complete"


def test_grpo_queue_stops_on_failed_training_without_launching(tmp_path, monkeypatch):
    base = EVAL.load_module("grpo_failed_queue_test_core", SCRIPTS / "eval_polaris_step80.py")
    plan = {"training_root": str(tmp_path), "training_kind": "grpo", "job_id": "123",
            "model_repo": "owner/grpo-step_100", "total_responses": 7276}
    monkeypatch.setattr(EVAL, "verify_plan", lambda *args: plan)
    monkeypatch.setattr(EVAL, "require_queue_node", lambda *args: None)
    monkeypatch.setattr(EVAL, "training_status", lambda *args: {
        "variant": "grpo", "adv_estimator": "grpo", "state": "training_failed", "exit_code": 1})
    monkeypatch.setattr(EVAL.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Launched after failure"))
    with pytest.raises(RuntimeError, match="Training did not complete successfully"):
        EVAL.queue(tmp_path, base)
    assert base.read(tmp_path / "queue_status.json")["state"] == "failed"


def test_math_verify_never_sees_thinking_and_does_not_require_a_box():
    base = EVAL.load_module("l0_test_grader", SCRIPTS / "eval_polaris_step80.py")
    base.init_grader()
    assert EVAL.answer_suffix(r"<think>\boxed{42}")[0] is None
    assert EVAL.answer_suffix(r"<think>\boxed{42}</think>  ")[0] is None
    assert EVAL.answer_suffix(r"<think>\boxed{42}</think><think>42")[0] is None
    suffix, state = EVAL.answer_suffix(r"<think>\boxed{42}</think>The final answer is 7.")
    assert state == "eligible" and "42" not in suffix
    assert base.grade((suffix, "42"))["correct"] == 0
    suffix, state = EVAL.answer_suffix(r"<think>\boxed{7}</think>The final answer is 42.")
    assert base.last_box(suffix) is None
    assert base.grade((suffix, "42"))["correct"] == 1
    suffix, _ = EVAL.answer_suffix(r"<think>1</think>42<think>2</think>7")
    assert suffix == "7"


def test_truncated_response_without_eos_can_score_its_completed_answer(tmp_path, monkeypatch):
    base = EVAL.load_module("grpo_no_eos_test_core", SCRIPTS / "eval_polaris_step80.py")
    base.init_grader()
    response = "<think>wrong answer 7</think>The final answer is 42.".ljust(32768)
    tokens = [ord(char) for char in response]
    question = {"id": "toy_0000", "dataset": "toy", "gold": "42", "prompt_token_ids": [1, 2]}
    originals = {
        base.sample_id(question, index): {
            "id": base.sample_id(question, index), "question_id": question["id"],
            "sample_index": index, "dataset": "toy", "seed": base.sample_seed(question["id"], index),
            "prompt_tokens": 2, "output_token_ids": tokens, "output_tokens": len(tokens),
            "finish_reason": "length", "response": response, "prediction": None, "correct": 0,
        } for index in range(4)
    }
    monkeypatch.setattr(base, "saved_result", lambda root, identity, *args, **kwargs: originals[identity])
    monkeypatch.setattr(EVAL, "BASE", base)
    monkeypatch.setattr(EVAL, "MODEL_LABEL", "GRPO")
    monkeypatch.setattr(EVAL, "TOKENIZER", SimpleNamespace(
        decode=lambda ids, **kwargs: "".join(chr(token) for token in ids)))
    rows = EVAL.score_question((str(tmp_path), "manifest", question))
    assert len(rows) == 4 * len(EVAL.CAPS)
    assert all(row["correct"] == 1 and row["model"] == "GRPO" for row in rows)
    assert all(not row["has_complete_box"] for row in rows)


def sample_rows():
    return [{"sample_id": f"q{q}__{i}", "question_id": f"q{q}", "sample_index": i,
             "cap_tokens": 1024, "dataset": "toy", "correct": int(q == 0 and i < 2),
             "used_output_tokens": 100 if q == 0 else 1024,
             "suffix_status": "eligible" if q == 0 else "unfinished_thinking",
             "grader_status": "scored" if q == 0 else "unfinished_thinking"}
            for q in range(2) for i in range(4)]


def test_unfinished_samples_remain_in_mean_pass_and_length_denominators():
    metrics, questions = EVAL.aggregate(sample_rows(), (("toy", "Toy", 2),), (1024,))
    assert metrics[0]["mean_at_4_percent"] == 25
    assert metrics[0]["pass_at_4_percent"] == 50
    assert metrics[0]["mean_output_tokens"] == 562
    assert metrics[0]["responses"] == 8 and len(questions) == 2
    for broken in (sample_rows()[:-1], sample_rows() + sample_rows()[:1]):
        with pytest.raises(AssertionError):
            EVAL.aggregate(broken, (("toy", "Toy", 2),), (1024,))


def test_grpo_metrics_preserve_the_algorithm_label():
    rows = [{**row, "model": "GRPO"} for row in sample_rows()]
    metrics, _ = EVAL.aggregate(rows, (("toy", "Toy", 2),), (1024,))
    assert metrics[0]["model"] == "GRPO"
    rows[0]["model"] = "L+0"
    with pytest.raises(AssertionError, match="different models"):
        EVAL.aggregate(rows, (("toy", "Toy", 2),), (1024,))


@pytest.mark.parametrize("occupied_lock", [False, True])
def test_handoff_does_not_launch_over_training(tmp_path, occupied_lock):
    holder = tmp_path / "training.lock"
    (tmp_path / "plan.json").write_text(json.dumps({"holder_locks": [str(holder)]}))
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    nvidia = bin_path / "nvidia-smi"
    nvidia.write_text("#!/usr/bin/env bash\nprintf '12345\\n'\n")
    nvidia.chmod(0o755)
    with holder.open("a") as lock:
        if occupied_lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", str(SCRIPTS / "run_l0_final_eval.sh"), sys.executable, str(tmp_path)],
            env={**os.environ, "PATH": str(bin_path) + os.pathsep + os.environ["PATH"]},
            capture_output=True, text=True, timeout=10,
        )
    assert result.returncode == 75, result.stderr
    assert not (tmp_path / "exit_status").exists()
