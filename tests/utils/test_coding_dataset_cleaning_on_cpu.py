import pytest

from qwen3_experiments.coding_holdout_audit import title_key, url_identity

from qwen3_experiments.coding_dataset_cleaning import (
    balanced_fill,
    canonical_statement,
    clean_stdio_tests,
    contradictory_bounds,
    duplicate_groups,
    leetcode_test_issues,
    numeric_bound,
    statement_issues,
    stratified_sample,
)


def test_test_pairs_preserve_order_and_allow_empty_valid_output():
    cleaned, issues, duplicates = clean_stdio_tests(
        {"inputs": ["1\n", "2\n", "1\n"], "outputs": ["", "4\n", ""]}
    )
    assert not issues and duplicates == 1
    assert cleaned == {"inputs": ["1\n", "2\n"], "outputs": ["", "4\n"]}


@pytest.mark.parametrize("tests,reason", [
    ({"inputs": [], "outputs": []}, "missing_tests"),
    ({"inputs": ["a"], "outputs": []}, "unpaired_tests"),
    ({"inputs": ["a", "a"], "outputs": ["yes", "no"]}, "conflicting_expected_outputs"),
    ({"inputs": ["a"], "outputs": [None]}, "non_text_test_case"),
    ({"inputs": ["\x00"], "outputs": ["0"]}, "malformed_test_encoding"),
])
def test_unusable_tests_are_excluded(tests, reason):
    cleaned, issues, _ = clean_stdio_tests(tests)
    assert cleaned is None and reason in issues


def test_constraints_are_checked_without_executing_text():
    assert contradictory_bounds(r"Constraints: 1 \leq N \leq 10^{5}") == []
    assert contradictory_bounds("Constraints: 10 <= n <= 1")
    assert contradictory_bounds("Constraints: 1 < n <= 1")
    assert numeric_bound("-10^9") == -(10**9)
    with pytest.raises(ValueError):
        numeric_bound("__import__('os').system('false')")
    with pytest.raises(ValueError):
        numeric_bound("10^999999")


def test_statement_hash_preserves_different_constraints():
    assert canonical_statement("Input  N\n1 <= N <= 5") != canonical_statement("Input N 1 <= N <= 50")
    assert canonical_statement("a < b") != canonical_statement("a > b")


@pytest.mark.parametrize("text", [
    "2 <= n <= 1 000", "1000 <= n <= 10\u00a0000", r"2 \leq n \leq 1\,000",
    "1000 <= y < 100'000",
    "100 <= T <= 10,000", "3 <= R <= 1,000,000",
    "- 10^9 <= x <= 10^9", "- 2·10^9 <= y <= 2·10^9",
    "3 <= n <= 2⋅10^5", "-5 ⋅ 10^8 <= a_i <= 5 ⋅ 10^8",
    "2 <= n <= 2 * 10^5 - 1", "2 <= n <= 2*n", "1 <= x <= 10^{18}",
])
def test_numeric_bound_formatting_never_creates_false_contradictions(text):
    assert contradictory_bounds(text) == []


def test_contradictory_bounds_support_arithmetic_and_separators():
    assert contradictory_bounds("2,000 <= n <= 1 000")
    assert contradictory_bounds("2*10^5 <= n <= 10^5-1")
    assert contradictory_bounds("-10 <= n <= -20")


def test_non_unique_output_requires_a_special_judge_for_stdio():
    text = ("Given a collection of integers, choose an ordering of the elements that satisfies the stated condition. "
            "Constraints: 1 <= n <= 100. Input contains n integers. Output any valid permutation.")
    assert "multiple_valid_outputs_without_special_judge" in statement_issues(text)
    assert "multiple_valid_outputs_without_special_judge" not in statement_issues(text, standard_input=False)


@pytest.mark.parametrize("output", [
    "If there are multiple valid rearrangements, print any of them.",
    "Print any possible variant of the repainted stripe.",
    "If multiple solutions exist, any of them will be accepted.",
    "Output any super-permutation of length n.",
    "If there are multiple answers, you can find any.",
    "You can print the edges in any order.",
])
def test_arbitrary_answer_variants_require_a_special_judge(output):
    text = "Given an integer n, solve the stated computational task with a Python program. Input: 1 <= n <= 100. Output: " + output
    assert "multiple_valid_outputs_without_special_judge" in statement_issues(text)


@pytest.mark.parametrize("output", [
    "You can output YES and NO in any case.",
    "If there are multiple solutions, output the smallest one.",
    "Do not print any extra messages.",
    "If there are no results, you do not need to print anything.",
])
def test_unique_answer_and_formatting_instructions_are_retained(output):
    text = "Given an integer n, solve the stated computational task with a Python program. Input: 1 <= n <= 100. Output: " + output
    assert "multiple_valid_outputs_without_special_judge" not in statement_issues(text)


def test_duplicate_aliases_are_transitive_without_merging_similar_statements():
    records = [
        {"id": 0, "identity_keys": ["cf:1/A"], "statement_sha256": "a"},
        {"id": 1, "identity_keys": ["cf:1/A", "cf:2/B"], "statement_sha256": "b"},
        {"id": 2, "identity_keys": ["cf:2/B"], "statement_sha256": "c"},
        {"id": 3, "identity_keys": ["cf:3/A"], "statement_sha256": "d"},
    ]
    assert sorted(sorted(r["id"] for r in g) for g in duplicate_groups(records)) == [[0, 1, 2], [3]]


def test_balanced_fill_is_without_replacement_and_reproducible():
    pools = {"aizu": list(range(4)), "codechef": list(range(4, 8)), "hackerearth": list(range(8, 12))}
    values = balanced_fill(pools, 8)
    assert len(values) == len(set(values)) == 8
    assert values == balanced_fill(pools, 8)
    assert sum(v < 4 for v in values) == 3
    with pytest.raises(ValueError):
        balanced_fill(pools, 13)


def test_proportional_rating_sample_has_exact_quotas_without_replacement():
    pools = {1000: list(range(3)), 1100: list(range(3, 10)), 1200: list(range(10, 20))}
    selected = stratified_sample(pools, 11)
    assert len(selected) == len(set(selected)) == 11
    assert selected == stratified_sample(pools, 11)
    assert [sum(v in pool for v in selected) for pool in pools.values()] == [2, 4, 5]
    assert set(stratified_sample(pools, 20)) == set(range(20))
    assert stratified_sample({}, 0) == []
    with pytest.raises(ValueError):
        stratified_sample(pools, 21)
    with pytest.raises(ValueError):
        stratified_sample(pools, -1)


def test_leetcode_missing_assertions_cannot_pass_structural_cleaning():
    row = {"task_id": "demo", "query": "Solve the task", "prompt": "from typing import *",
           "entry_point": "f", "completion": "def f(x): return x", "test": "def check(f): pass"}
    assert leetcode_test_issues(row) == ["missing_leetcode_assertions"]
    row["test"] = "def check(f): assert f(1) == 1"
    assert leetcode_test_issues(row) == []


def test_holdout_identity_matching_handles_url_variants_and_platform_namespaces():
    assert url_identity("https://codeforces.com/problemset/problem/1575/A") == "codeforces:1575/A"
    assert url_identity("https://codeforces.com/contest/1575/problem/A") == "codeforces:1575/A"
    assert url_identity("https://www.codechef.com/problems/BFTT") == "codechef:bftt"
    assert url_identity("https://www.hackerearth.com/practice/algorithms/foo/algorithm/bar/") == "hackerearth:bar"
    assert url_identity("http://www.usaco.org/index.php?page=viewproblem2&cpid=1333") == "usaco:1333"
    assert url_identity("https://example.com/problems/1575/A") is None
    assert title_key("Zigzag-Grid-Traversal") == title_key("zigzag grid traversal")


def test_exact_holdout_policy_retains_variants_and_excludes_confirmed_originals():
    from qwen3_experiments.prepare_coding_mix import Builder

    builder = Builder.__new__(Builder)
    builder.config = {"holdout_policy": "exact_problem_only"}
    builder.additional_holdout_exclusions = {}
    builder.blocked_ids = {"codeforces:1249/B2"}
    text = "A long statement about exchanging books with different size limits. " * 4
    builder.holdout_prompts = [("taco_test", text + "The hard version adds constraints.")]
    metadata = {"origin": "nemotron", "source_row_index": 1, "identity_keys": ["codeforces:1249/B1"]}
    assert builder.holdout_matches(metadata, text) == []
    metadata["identity_keys"] = ["codeforces:1249/B2"]
    assert builder.holdout_matches(metadata, text) == ["heldout_problem_id_overlap"]
