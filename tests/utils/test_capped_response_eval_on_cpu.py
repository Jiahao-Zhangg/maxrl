"""Prefix evaluation must grade truncated answers and retain every response."""

import numpy as np
import pytest

from qwen3_experiments.capped_response_eval import mean_at_n, score_caps

VOCAB = {0: "thinking", 1: "</think>", 2: "OK", 3: "BAD", 4: "", 5: "<think>"}


def decode(tokens):
    return "".join(VOCAB[t] for t in tokens)


def grader(answer):
    return {"score": int(answer == "OK")}


def evaluate(tokens, score, caps, **kwargs):
    return score_caps(tokens, decode(tokens), score, caps, decode, grader, thinking_close_id=1, **kwargs)


def test_truncated_answer_can_be_correct_without_eos():
    result = evaluate([0, 1, 2, 3], 0, [1, 2, 3, 4])
    assert [r["score"] for r in result] == [0, 0, 1, 0]
    assert result[2]["origin"] == "regraded_prefix"
    assert result[2]["used_tokens"] == 3


def test_partial_credit_is_not_awarded_before_answer_exists():
    result = evaluate([0, 0, 1, 2], 1, [1, 2, 3, 4, 8])
    assert [r["score"] for r in result] == [0, 0, 0, 1, 1]
    assert result[-1]["used_tokens"] == 4


def test_unchanged_answer_reuses_original_score_even_if_eos_is_cut():
    result = evaluate([0, 1, 2, 4], 1, [3, 4])
    assert [r["score"] for r in result] == [1, 1]
    assert result[0]["origin"] == "unchanged_visible_answer"


def test_short_response_and_original_timeout_grade_are_preserved():
    def never_grade(_):
        raise AssertionError("An unchanged grader input should reuse its score")
    result = score_caps([0, 1, 2, 4], decode([0, 1, 2, 4]), 0, [3, 4, 9], decode, never_grade,
                        thinking_close_id=1)
    assert [r["score"] for r in result] == [0, 0, 0]


def test_later_invalid_thinking_does_not_invalidate_an_earlier_prefix():
    result = evaluate([0, 1, 2, 5, 3], 0, [3, 4, 5])
    assert [r["score"] for r in result] == [1, 0, 0]


def test_nonatomic_closing_tag_uses_full_decode_fallback():
    chars = list(map(ord, "x</think>OKBAD"))
    result = score_caps(chars, "x</think>OKBAD", 0, [11, 14],
                        lambda ids: "".join(map(chr, ids)), grader, thinking_close_id=999)
    assert result[0]["score"] == 1
    assert result[0]["origin"] == "regraded_prefix"


def test_multiple_closing_tags_use_last_tag_within_each_cap():
    result = evaluate([0, 1, 2, 1, 3], 0, [3, 4, 5])
    assert [r["score"] for r in result] == [1, 0, 0]


def test_mean100_keeps_all_failures_and_is_not_pass100():
    values = np.zeros((2, 100, 2), dtype=np.int8)
    values[0, 0, :] = 1
    np.testing.assert_array_equal(mean_at_n(values, expected_questions=2, expected_samples=100), [.005, .005])
    values[1, 0, 0] = -1
    with pytest.raises(ValueError, match="binary"):
        mean_at_n(values, expected_questions=2, expected_samples=100)


def test_missing_samples_are_rejected():
    with pytest.raises(ValueError, match="complete"):
        mean_at_n(np.zeros((2, 99, 6), dtype=np.int8), expected_questions=2, expected_samples=100)
