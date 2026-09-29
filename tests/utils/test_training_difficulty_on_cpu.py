"""Protect predecessor gating and unbiased training-sample inputs."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "qwen3_experiments"
SPEC = importlib.util.spec_from_file_location("difficulty_test", SCRIPTS / "eval_training_difficulty.py")
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)
BASE = EVAL.module("difficulty_test_core", SCRIPTS / "eval_polaris_step80.py")


def test_waits_for_entire_l0_evaluation_and_released_queue(tmp_path):
    queue = {"state": "complete"}
    status = {"state": "complete", "completed_responses": 7276, "total_responses": 7276,
              "completed_questions": 1819}
    audit = {"complete": True, "responses_verified": 7276, "budget_points": 54, "grader_errors": {}}
    assert not EVAL.predecessor_ready(tmp_path, BASE)
    for name, value in [("queue_status.json", queue), ("status.json", status), ("report/audit.json", audit)]:
        BASE.write(tmp_path / name, value)
    assert EVAL.predecessor_ready(tmp_path, BASE)
    for file, original, change in [
        ("queue_status.json", queue, {"state": "waiting_for_available_gpus_or_evaluating"}),
        ("status.json", status, {"state": "generated"}),
        ("status.json", status, {"completed_responses": 7275}),
        ("report/audit.json", audit, {"complete": False}),
        ("report/audit.json", audit, {"budget_points": 53}),
        ("report/audit.json", audit, {"grader_errors": {"TimeoutError": 1}}),
    ]:
        BASE.write(tmp_path / file, {**original, **change})
        assert not EVAL.predecessor_ready(tmp_path, BASE)
        BASE.write(tmp_path / file, original)


def test_compression_gold_keeps_full_expression_without_solution_leakage():
    row = {"problem": "Find x.", "extracted": r"\frac{1}{2}",
           "solution": r"PRIVATE WORKED SOLUTION: \boxed{9}; finally \boxed{\frac{1}{2}}."}
    result = EVAL.normalize_question(row, 17, {"key": "compression_train", "answer": "extracted"}, BASE)
    assert result["gold"] == r"\frac{1}{2}"
    assert result["source_row"] == 17 and result["id"] == "compression_train_0017"
    assert result["messages"] == [{"role": "user", "content": "Find x." + BASE.SUFFIX}]
    assert "PRIVATE" not in json.dumps(result)
    with pytest.raises(AssertionError, match="Gold disagrees"):
        EVAL.normalize_question({**row, "extracted": "9"}, 17,
                                {"key": "compression_train", "answer": "extracted"}, BASE)


def test_datasets_do_not_collide_when_sampled_source_indices_match():
    first = EVAL.normalize_question({"problem": "P?", "answer": "1"}, 5,
                                    {"key": "polaris_train", "answer": "answer"}, BASE)
    second = EVAL.normalize_question({"problem": "C?", "extracted": "1", "solution": r"\boxed{1}"}, 5,
                                     {"key": "compression_train", "answer": "extracted"}, BASE)
    assert first["id"] != second["id"]
    assert BASE.sample_seed(first["id"], 0) != BASE.sample_seed(second["id"], 0)
    same_question_other_model = copy.deepcopy(first)
    assert [BASE.sample_seed(first["id"], i) for i in range(4)] == [
        BASE.sample_seed(same_question_other_model["id"], i) for i in range(4)
    ]


def test_bootstrap_difference_uses_question_scores_and_correct_sign():
    assert EVAL.bootstrap_difference([0.25] * 200, [0.75] * 200) == [-50.0, -50.0]
