import json

import pytest

from qwen3_experiments.coding_release_eval import holdout_truth, taco_io, verify_response
from qwen3_experiments.taco_eval import sandbox_command


def test_taco_call_return_wrapper_and_nested_list_are_preserved():
    tests = {"fn_name": "f", "inputs": [[[1, 2], 3], ["hello"]], "outputs": [[[4, 5]], ["ok"]]}
    converted = taco_io(tests)
    assert converted["inputs"] == ['[1, 2]\n3', '"hello"']
    assert [json.loads(value) for value in converted["outputs"]] == [[4, 5], "ok"]


def test_stdin_lists_and_every_code_contests_group_are_kept():
    assert taco_io({"inputs": [["2", "1 3"]], "outputs": [["4"]]}) == {
        "inputs": ["2\n1 3"], "outputs": ["4"], "fn_name": None}
    question = {"tests": [{"input": "1\n", "output": "2\n", "group": group}
                           for group in ["public", "private", "generated"]]}
    truth = holdout_truth("code_contests", question)
    assert len(truth["input_output"]["inputs"]) == 3
    assert truth["unit_test_timeout_seconds"] == 10
    assert truth["check_eos"] is False


def test_bad_taco_wrappers_fail_instead_of_dropping_tests():
    with pytest.raises(ValueError):
        taco_io({"fn_name": "f", "inputs": [[1]], "outputs": [[2, 3]]})


def test_node_local_python_is_mounted_after_private_tmp():
    plan = {"python_bin": "/tmp/example-env/bin/python", "bubblewrap": "bwrap",
            "official": "/tmp/official", "sandbox_runner": "/tmp/runner.py"}
    command = sandbox_command(plan, "/tmp/input.json")
    assert command.index("--tmpfs") < command.index("/tmp/example-env")
    assert "--unshare-all" in command and "--clearenv" in command


def test_response_reuse_requires_explicit_plan_compatibility_and_same_model_question():
    question = {"id": "question", "source_index": 7}
    model = {"revision": "weights"}
    record = {"id": "question", "index": 7, "model_revision": "weights", "plan_sha256": "old-policy"}
    with pytest.raises(ValueError):
        verify_response(record, question, model, "new-policy")
    verify_response(record, question, model, "new-policy", compatible_plan_hashes=["old-policy"])
    for key, value in [("id", "other-question"), ("index", 8), ("model_revision", "other-model"),
                       ("plan_sha256", "unapproved-plan")]:
        with pytest.raises(ValueError):
            verify_response({**record, key: value}, question, model, "new-policy",
                            compatible_plan_hashes=["old-policy"])
