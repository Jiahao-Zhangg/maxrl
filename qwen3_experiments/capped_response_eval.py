"""Binary single-response evaluation from token-capped saved rollouts."""

import hashlib

import numpy as np

from qwen3_experiments.compression_verl_reward import response_after_thinking


def score_caps(token_ids, response, original_score, caps, decode, grade_answer, *, thinking_close_id=None):
    """Grade each actual prefix; no partial credit and no success conditioning.

    ``thinking_close_id`` is an optional decoder-verified atomic </think> token.
    The shortcut is used only when its count matches every visible closing tag
    in the saved full response. Otherwise decode the prefix normally.
    """
    tokens = np.asarray(token_ids)
    caps = np.asarray(caps)
    if tokens.ndim != 1 or not len(tokens) or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("Expected a nonempty integer token sequence")
    if caps.ndim != 1 or not len(caps) or not np.issubdtype(caps.dtype, np.integer) or np.any(caps <= 0):
        raise ValueError("Caps must be positive integer token counts")
    if original_score not in (0, 1) or not isinstance(response, str):
        raise ValueError("Expected stored response text and binary full-response correctness")
    full_answer = response_after_thinking(response)
    close_positions = np.flatnonzero(tokens == thinking_close_id) if thinking_close_id is not None else None
    trusted_closes = close_positions is not None and len(close_positions) == response.count("</think>")
    results = []
    for cap in caps:
        cap = int(cap)
        item = {"cap_tokens": cap, "used_tokens": min(cap, len(tokens)), "truncated": cap < len(tokens)}
        if cap >= len(tokens):
            item.update(score=int(original_score), origin="original_complete_response")
        elif trusted_closes and (not len(close_positions) or close_positions[0] >= cap):
            item.update(score=0, origin="thinking_not_closed_at_cap")
        else:
            prefix = decode(tokens[:cap].tolist())
            item["prefix_sha256"] = hashlib.sha256(prefix.encode()).hexdigest()
            answer = response_after_thinking(prefix)
            if answer is None:
                item.update(score=0, origin="missing_or_invalid_answer_at_cap")
            elif answer == full_answer:
                item.update(score=int(original_score), origin="unchanged_visible_answer")
            else:
                grade = grade_answer(answer)
                if grade["score"] not in (0, 1):
                    raise ValueError("The prefix grader must return binary correctness")
                item.update(score=int(grade["score"]), origin="regraded_prefix",
                            answer_sha256=hashlib.sha256(answer.encode()).hexdigest(), grade=grade)
        results.append(item)
    return results


def mean_at_n(scores, *, expected_questions, expected_samples):
    """Average all responses, retaining the same denominator at every cap."""
    values = np.asarray(scores)
    if values.ndim != 3 or values.shape[:2] != (expected_questions, expected_samples):
        raise ValueError("Expected a complete question-by-sample-by-cap score array")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("Every requested response and cap must have a binary score")
    return values.mean(axis=(0, 1))
