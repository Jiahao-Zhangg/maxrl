"""Exercise EOS reward gating with real MathVerify and an in-process Ray stub."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.workers.reward_manager import multi_thread_naive as reward_module

EOS = 1
CORRECT = 2
WRONG = 3
SPACE = 4


class Tokenizer:
    eos_token_id = EOS

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = {EOS: "<eos>", CORRECT: r"\boxed{1}", WRONG: r"\boxed{2}", SPACE: " "}
        return "".join(
            pieces[int(token_id)] for token_id in token_ids if not (skip_special_tokens and int(token_id) == EOS)
        )


def make_data(responses, valid_lengths=None):
    response_ids = torch.tensor(responses, dtype=torch.long)
    batch_size, width = response_ids.shape
    if valid_lengths is None:
        valid_lengths = [width] * batch_size
    response_mask = torch.arange(width).unsqueeze(0) < torch.tensor(valid_lengths).unsqueeze(1)
    # EOS in the prompt must never satisfy the response-completion check.
    prompts = torch.tensor([[EOS, SPACE]] * batch_size)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": response_ids,
            "attention_mask": torch.cat([torch.ones_like(prompts), response_mask.long()], dim=1),
        },
        non_tensors={
            "data_source": np.array(["compression"] * batch_size, dtype=object),
            "prompt_id": np.array([f"question_{i}" for i in range(batch_size)], dtype=object),
            "reward_model": np.array([{"ground_truth": "1"}] * batch_size, dtype=object),
        },
    )


@pytest.fixture
def score_batches(monkeypatch):
    """Replace only Ray transport; retain the production MathVerify scorer."""
    calls = []
    results = {}
    scorer = reward_module.MathVerifyScorer()

    def score_remote(batch, timeout_score, per_item_timeout_s):
        calls.append([i for i, _, _ in batch])
        reference = object()
        result = []
        for i, response, gold in batch:
            score = scorer.compute_score(response, gold, timeout_score, per_item_timeout_s)
            result.append((i, {"score": score, "accuracy": score}))
        results[reference] = result
        return reference

    actor = SimpleNamespace(compute_scores_batch=SimpleNamespace(remote=score_remote))
    monkeypatch.setattr(reward_module.RewardScoreActor, "remote", lambda: actor)
    monkeypatch.setattr(reward_module.ray, "get", results.__getitem__)
    monkeypatch.setattr(
        reward_module.ray,
        "wait",
        lambda refs, num_returns, timeout: (list(refs)[:num_returns], list(refs)[num_returns:]),
    )
    return calls


@pytest.mark.parametrize("check_eos", [True, "true"])
def test_missing_eos_is_zero_and_skipped_in_mixed_scoring_batches(score_batches, check_eos):
    data = make_data(
        [[CORRECT, SPACE], [CORRECT, EOS], [CORRECT, SPACE], [WRONG, EOS], [CORRECT, SPACE], [CORRECT, EOS], [CORRECT, SPACE]]
    )
    manager = reward_module.MultiThreadNaiveRewardManager(
        Tokenizer(), num_examine=0, num_reward_actors=1, batch_size=2, check_eos=check_eos
    )
    result = manager(data, return_dict=True)
    assert result["reward_tensor"].sum(dim=-1).tolist() == [0, 1, 0, 0, 0, 1, 0]
    assert result["reward_extra_info"]["zeroed_by_missing_eos"] == [1, 0, 1, 0, 1, 0, 1]
    assert result["majority_vote_accuracy_global"] == pytest.approx(2 / 7)
    assert score_batches == [[1, 3], [5]]


def test_eos_at_32000_token_limit_can_receive_reward(score_batches):
    data = make_data([[SPACE] * 31998 + [CORRECT, EOS]])
    manager = reward_module.MultiThreadNaiveRewardManager(
        Tokenizer(), num_examine=0, num_reward_actors=1, check_eos=True, max_resp_len=32000
    )
    result = manager(data, return_dict=True)
    assert result["reward_tensor"][0, -1].item() == 1
    assert result["reward_tensor"].sum().item() == 1
    assert result["reward_extra_info"]["zeroed_by_missing_eos"] == [0]
    assert score_batches == [[0]]


def test_eos_in_prompt_or_masked_padding_does_not_count(score_batches):
    data = make_data([[CORRECT, SPACE, EOS], [CORRECT, EOS, EOS]], valid_lengths=[2, 1])
    manager = reward_module.MultiThreadNaiveRewardManager(
        Tokenizer(), num_examine=0, num_reward_actors=1, check_eos=True
    )
    result = manager(data, return_dict=True)
    assert torch.count_nonzero(result["reward_tensor"]).item() == 0
    assert result["reward_extra_info"]["zeroed_by_missing_eos"] == [1, 1]
    assert result["majority_vote_accuracy_global"] == 0
    assert result["accuracy_interval_fractions_per_prompt_global"]["[0.0, 0.0]"] == 1
    assert score_batches == []


def test_contains_eos_matches_reference_without_requiring_last_token(score_batches):
    data = make_data([[CORRECT, EOS, SPACE]])
    manager = reward_module.MultiThreadNaiveRewardManager(
        Tokenizer(), num_examine=0, num_reward_actors=1, check_eos=True
    )
    assert manager(data).sum().item() == 1


@pytest.mark.parametrize("kwargs", [{}, {"check_eos": False}, {"check_eos": "false"}])
def test_disabled_eos_check_preserves_existing_scoring(score_batches, kwargs):
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = None
    manager = reward_module.MultiThreadNaiveRewardManager(tokenizer, num_examine=0, num_reward_actors=1, **kwargs)
    result = manager(make_data([[CORRECT, SPACE]]), return_dict=True)
    assert result["reward_tensor"].sum().item() == 1
    assert "zeroed_by_missing_eos" not in result["reward_extra_info"]


def test_legacy_max_length_gate_remains_available(score_batches):
    manager = reward_module.MultiThreadNaiveRewardManager(
        Tokenizer(), num_examine=0, num_reward_actors=1, zero_reward_on_max_response_length=True, max_resp_len=2
    )
    result = manager(make_data([[CORRECT, EOS]]), return_dict=True)
    assert result["reward_tensor"].sum().item() == 0
    assert result["reward_extra_info"]["zeroed_by_max_response_length"] == [1]


@pytest.mark.parametrize("return_dict", [True, False])
def test_precomputed_scores_are_gated_without_mutating_input(score_batches, return_dict):
    data = make_data([[CORRECT, EOS], [CORRECT, SPACE]])
    original_scores = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    data.batch["rm_scores"] = original_scores.clone()
    manager = reward_module.MultiThreadNaiveRewardManager(
        Tokenizer(), num_examine=0, num_reward_actors=1, check_eos=True
    )
    result = manager(data, return_dict=return_dict)
    reward_tensor = result["reward_tensor"] if return_dict else result
    assert reward_tensor.sum(dim=-1).tolist() == [1, 0]
    if return_dict:
        assert result["reward_extra_info"]["zeroed_by_missing_eos"] == [0, 1]
    torch.testing.assert_close(data.batch["rm_scores"], original_scores)
    assert score_batches == []


def test_eos_check_requires_tokenizer_eos_id():
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="tokenizer.eos_token_id"):
        reward_module.MultiThreadNaiveRewardManager(tokenizer, num_examine=0, check_eos=True)


def test_zero_item_timeout_does_not_install_or_clear_an_alarm(monkeypatch):
    scorer = reward_module.MathVerifyScorer()
    scorer._verify_func = lambda gold, prediction: (1.0, None)
    signal_calls = []
    monkeypatch.setattr(reward_module.signal, "signal", lambda *args: signal_calls.append(args))
    monkeypatch.setattr(reward_module.signal, "alarm", lambda seconds: signal_calls.append(seconds))
    assert scorer.compute_score(r"\boxed{1}", "1", timeout_score=0, per_item_timeout_s=0) == 1
    assert signal_calls == []


def test_enabled_item_timeout_still_returns_timeout_score_and_clears_alarm(monkeypatch):
    scorer = reward_module.MathVerifyScorer()

    def time_out(*args):
        raise reward_module._ItemTimeout

    scorer._verify_func = time_out
    alarms = []
    monkeypatch.setattr(reward_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(reward_module.signal, "alarm", alarms.append)
    assert scorer.compute_score(r"\boxed{1}", "1", timeout_score=0, per_item_timeout_s=1) == 0
    assert alarms == [1, 0]


@pytest.mark.parametrize("batch_timeout, expected_score", [(0, 1), (10, 0)])
def test_batch_deadline_can_be_disabled_for_slow_results(score_batches, monkeypatch, batch_timeout, expected_score):
    manager = reward_module.MultiThreadNaiveRewardManager(
        Tokenizer(), num_examine=0, num_reward_actors=1, check_eos=True,
        per_item_timeout_s=0, per_batch_timeout_s=batch_timeout,
    )
    clock = iter([0, 3600, 7200])
    monkeypatch.setattr(reward_module, "time", SimpleNamespace(time=lambda: next(clock)))
    polls = []
    cancelled = []

    def wait(refs, **kwargs):
        polls.append(True)
        return ([], refs) if len(polls) == 1 else (refs, [])

    monkeypatch.setattr(reward_module.ray, "wait", wait)
    monkeypatch.setattr(reward_module.ray, "cancel", lambda ref, **kwargs: cancelled.append(ref))
    result = manager(make_data([[CORRECT, EOS]]), return_dict=True)
    assert result["reward_tensor"].sum().item() == expected_score
    assert len(cancelled) == (0 if batch_timeout == 0 else 1)


def test_grader_still_receives_only_response_text_without_outer_timeouts(score_batches, monkeypatch):
    data = make_data([[CORRECT, EOS]])
    data.batch["prompts"][0] = torch.tensor([WRONG, SPACE])
    predictions = []
    original = reward_module.MathVerifyScorer.compute_score

    def capture(self, model_output, ground_truth, timeout_score, per_item_timeout_s):
        predictions.append(model_output)
        assert per_item_timeout_s == 0
        return original(self, model_output, ground_truth, timeout_score, per_item_timeout_s)

    monkeypatch.setattr(reward_module.MathVerifyScorer, "compute_score", capture)
    manager = reward_module.MultiThreadNaiveRewardManager(
        Tokenizer(), num_examine=0, num_reward_actors=1, check_eos=True,
        per_item_timeout_s=0, per_batch_timeout_s=0,
    )
    assert manager(data).sum().item() == 1
    assert predictions == [r"\boxed{1}"]
