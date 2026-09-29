import json

import pytest

from qwen3_experiments.lcb_coding_format import convert_truth, leetcode_tests, prepare_code, tree_value
from qwen3_experiments.lcb_coding_grading import GradingInfrastructureError, LiveCodeBenchGrader


def problem(test, name="solve", args="x, y", prelude=""):
    return {"entry_point": f"Solution().{name}", "starter_code": f"class Solution:\n    def {name}(self, {args}):\n        ",
            "test": "def check(candidate):\n" + test, "prompt": prelude}


def test_argument_order_and_complete_test_preservation():
    result = leetcode_tests(problem("    assert candidate(y=2, x=[1, 3]) == 4\n    assert candidate(x=[], y=0) == None"))
    assert result["input_output"] == {"inputs": ["[1,3]\n2", "[]\n0"], "outputs": ["4", "null"], "fn_name": "solve"}
    assert result["adapter"] is None


@pytest.mark.parametrize("body", [
    "    for x in range(3):\n        assert candidate(x=x, y=0) == x",
    "    assert candidate(x=1, y=2) >= 0",
    "    assert candidate(x=1) == 1",
    "    assert candidate(x=1, y=2) == dangerous()",
    "    assert candidate(x=1, y=2) == set([1])",
])
def test_unrepresentable_tests_are_rejected_instead_of_dropped(body):
    with pytest.raises((ValueError, TypeError)):
        leetcode_tests(problem(body))


def test_infinity_is_explicit_and_cannot_be_faked_by_a_dictionary():
    p = problem("    assert candidate(x=0, y=0) == -inf", prelude="inf = float('inf')")
    result = leetcode_tests(p)
    assert json.loads(result["input_output"]["outputs"][0]) == {"__lcb_float__": "-inf"}
    scope = {}
    exec(prepare_code("class Solution:\n    def solve(self, x, y): return -float('inf')", result), scope)
    assert scope["Solution"]()._maxrl_lcb_adapter_v1(0, 0) == {"__lcb_float__": "-inf"}
    exec(prepare_code("class Solution:\n    def solve(self, x, y): return {'__lcb_float__': '-inf'}", result), scope)
    with pytest.raises(TypeError):
        scope["Solution"]()._maxrl_lcb_adapter_v1(0, 0)


def test_tree_structure_keeps_missing_children_and_null_node_values():
    assert tree_value([1, None, 2, 3]) == [[1], None, [2], [3]]
    assert tree_value([]) == []
    assert tree_value([None]) == [[None]]
    # The upstream constructor ignores descendants beyond exhausted parents.
    assert tree_value([1, None, None, 99]) == [[1]]


def test_structural_and_null_cases_keep_both_assertions():
    p = problem("    assert is_same_list(candidate(x=list_node([1]), y=0), list_node([1]))\n"
                "    assert candidate(x=None, y=0) == None")
    converted = leetcode_tests(p)
    assert converted["input_output"]["outputs"] == ["[1]", "[]"]
    assert converted["input_output"]["inputs"] == ["[1]\n0", "null\n0"]
    assert converted["adapter"]["input_types"] == ["linked_list", "json"]


def test_adapter_preserves_recursive_calls_to_the_original_method():
    p = problem("    assert candidate(x=3, y=0) == inf", prelude="inf=float('inf')")
    result = leetcode_tests(p)
    scope = {}
    code = "class Solution:\n    def solve(self, x, y):\n        return self.solve(x-1,y) if x else float('inf')"
    exec(prepare_code(code, result), scope)
    assert scope["Solution"]()._maxrl_lcb_adapter_v1(3, 0) == {"__lcb_float__": "inf"}


def test_nemotron_bytes_are_preserved_and_empty_tests_fail():
    old = {"grader": "nemo_gym_code_gen", "unit_tests": {"inputs": [" 1\n\n"], "outputs": ["\nYES  \n"]}}
    converted = convert_truth(old)
    assert converted["input_output"] == {**old["unit_tests"], "fn_name": None}
    old["unit_tests"] = {"inputs": [], "outputs": []}
    with pytest.raises(ValueError, match="Empty"):
        convert_truth(old)


def test_grader_infrastructure_retries_then_raises_instead_of_reward_zero(monkeypatch):
    grader = object.__new__(LiveCodeBenchGrader)
    calls = []
    def fail(*args):
        calls.append(1)
        raise GradingInfrastructureError("sandbox unavailable")
    monkeypatch.setattr(grader, "_run", fail)
    truth = convert_truth({"grader": "nemo_gym_code_gen", "unit_tests": {"inputs": [""], "outputs": ["1"]}})
    with pytest.raises(GradingInfrastructureError):
        grader(truth, "print(1)")
    assert len(calls) == 2


@pytest.mark.parametrize("outcomes,expected", [
    ([1, 1], 1), ([0], 0), ([-2], 0), ([1, -3], 0), ([-4], 0),
    ([1], None), ([], None), ([2], None),
])
def test_result_codes_and_partial_success_never_become_false_rewards(tmp_path, monkeypatch, outcomes, expected):
    from types import SimpleNamespace
    import qwen3_experiments.lcb_coding_grading as module

    grader = object.__new__(LiveCodeBenchGrader)
    grader.plan, grader.temp_root = {}, tmp_path
    monkeypatch.setattr(module, "sandbox_command", lambda *args: ["fake-sandbox"])
    process = SimpleNamespace(returncode=0, communicate=lambda **kwargs: (json.dumps({"results": outcomes}), ""))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    truth = convert_truth({"grader": "nemo_gym_code_gen", "unit_tests": {"inputs": ["1", "2"], "outputs": ["1", "2"]}})
    if expected is None:
        with pytest.raises(GradingInfrastructureError):
            grader._run(truth, "print(1)")
    else:
        assert grader._run(truth, "print(1)")["score"] == expected
