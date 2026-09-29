"""Protect final-answer grading and resumable evaluation bookkeeping."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "qwen3_experiments/eval_polaris_step80.py"
SPEC = importlib.util.spec_from_file_location("step80_eval", SCRIPT)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def test_only_final_answer_after_completed_thinking_counts():
    assert EVAL.prediction(r"<think>Maybe \boxed{42}") is None
    assert EVAL.prediction(r"<think>\boxed{42}</think>No final answer") is None
    assert EVAL.prediction(r"<think>\boxed{42}</think>Final: \boxed{7}") == r"\boxed{7}"
    assert EVAL.prediction(r"</think><think>\boxed{42}") is None


def test_nested_and_unfinished_boxes():
    assert EVAL.last_box(r"Answer \boxed{\frac{1}{2}}.") == r"\boxed{\frac{1}{2}}"
    assert EVAL.last_box(r"\boxed{3} but actually \boxed{\frac{1}{") is None
    assert EVAL.last_box(r"\boxed{\{1,2\}}") == r"\boxed{\{1,2\}}"


def test_seed_and_worker_assignment_do_not_duplicate_samples():
    seeds = [EVAL.sample_seed(f"question_{q}", index) for q in range(864) for index in range(4)]
    assert len(set(seeds)) == 3456
    shards = [{i for i in range(3456) if i % 8 == rank} for rank in range(8)]
    assert all(len(shard) == 432 for shard in shards)
    assert len(set.union(*shards)) == sum(map(len, shards)) == 3456


def test_resumption_rejects_corrupted_result(tmp_path):
    record = {"id": "aime24_0000__0", "correct": 1.0, "output_tokens": 10, "dataset": "aime24"}
    EVAL.save_result(tmp_path, record, "manifest-a")
    assert EVAL.saved_result(tmp_path, record["id"], "manifest-a", full=True) == record
    with pytest.raises(AssertionError):
        EVAL.saved_result(tmp_path, record["id"], "manifest-b")
    path = tmp_path / "responses" / (record["id"] + ".json.gz")
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(AssertionError):
        EVAL.saved_result(tmp_path, record["id"], "manifest-a")


def test_math_verify_equivalence_and_wrong_answer():
    EVAL.init_grader()
    assert EVAL.grade((r"\boxed{0.5}", r"\frac{1}{2}"))["correct"] == 1.0
    assert EVAL.grade((r"\boxed{41}", "42"))["correct"] == 0.0
    assert EVAL.grade((None, "42"))["correct"] == 0.0
