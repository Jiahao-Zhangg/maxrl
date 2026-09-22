"""Exercise EOS and thinking boundaries with the real MathVerify scorer."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.workers.reward_manager import multi_thread_naive as module

EOS, CORRECT, WRONG, SPACE, OPEN, CLOSE = range(1, 7)


class Tokenizer:
    eos_token_id = EOS

    def decode(self, tokens, skip_special_tokens=True):
        pieces = {EOS: "<eos>", CORRECT: r"\boxed{1}", WRONG: r"\boxed{2}", SPACE: " ", OPEN: "<think>", CLOSE: "</think>"}
        return "".join(pieces[int(token)] for token in tokens if not (skip_special_tokens and int(token) == EOS))


def data_for(rows):
    width = max(map(len, rows))
    responses = torch.tensor([row + [EOS] * (width - len(row)) for row in rows])
    mask = torch.arange(width)[None, :] < torch.tensor(list(map(len, rows)))[:, None]
    prompts = torch.tensor([[CORRECT, EOS, OPEN]] * len(rows))
    return DataProto.from_dict(tensors={
        "prompts": prompts, "responses": responses,
        "attention_mask": torch.cat([torch.ones_like(prompts), mask.long()], dim=1),
    }, non_tensors={
        "data_source": np.array(["polaris"] * len(rows), dtype=object),
        "prompt_id": np.array([str(i) for i in range(len(rows))], dtype=object),
        "reward_model": np.array([{"ground_truth": "1"}] * len(rows), dtype=object),
    })


@pytest.fixture
def grader(monkeypatch):
    calls, results = [], {}
    scorer = module.MathVerifyScorer()

    def score(batch, timeout_score, per_item_timeout_s):
        ref = object()
        calls.extend(batch)
        results[ref] = []
        for index, response, gold in batch:
            value = scorer.compute_score(response, gold, timeout_score, per_item_timeout_s)
            results[ref].append((index, {"score": value, "accuracy": value}))
        return ref

    actor = SimpleNamespace(compute_scores_batch=SimpleNamespace(remote=score))
    monkeypatch.setattr(module.RewardScoreActor, "remote", lambda: actor)
    monkeypatch.setattr(module.ray, "get", results.__getitem__)
    monkeypatch.setattr(module.ray, "wait", lambda refs, **kwargs: (list(refs)[:1], list(refs)[1:]))
    return calls


def test_only_completed_answer_with_real_response_eos_is_graded(grader):
    rows = [
        [CORRECT, CLOSE, WRONG, EOS],  # Correct reasoning cannot rescue a wrong final answer.
        [WRONG, CLOSE, CORRECT, EOS],
        [WRONG, CLOSE, CORRECT],  # EOS in prompt/padding must not pass.
        [CORRECT, EOS],  # Unfinished thinking.
        [CORRECT, CLOSE, SPACE, EOS],  # Empty answer.
        [CORRECT, CLOSE, OPEN, CORRECT, EOS],  # Reopened, unfinished thinking.
        [CLOSE, CORRECT, EOS],  # Opening <think> is supplied by Qwen's prompt template.
    ]
    manager = module.MultiThreadNaiveRewardManager(Tokenizer(), 0, num_reward_actors=1, batch_size=2,
                                                  check_eos=True, score_after_thinking=True)
    result = manager(data_for(rows), return_dict=True)
    assert result["reward_tensor"].sum(dim=-1).tolist() == [0, 1, 0, 0, 0, 0, 1]
    assert [(i, text) for i, text, _ in grader] == [(0, r"\boxed{2}"), (1, r"\boxed{1}"), (6, r"\boxed{1}")]
    assert result["reward_extra_info"]["zeroed_by_missing_eos"] == [0, 0, 1, 0, 0, 0, 0]
    assert result["reward_extra_info"]["zeroed_by_invalid_thinking"] == [0, 0, 0, 1, 1, 1, 0]
    assert result["majority_vote_accuracy_global"] == pytest.approx(2 / 7)


def test_eos_at_token_limit_remains_eligible(grader):
    manager = module.MultiThreadNaiveRewardManager(Tokenizer(), 0, num_reward_actors=1,
                                                  check_eos=True, score_after_thinking=True, max_resp_len=4)
    assert manager(data_for([[WRONG, CLOSE, CORRECT, EOS]])).sum().item() == 1


def test_old_defaults_still_grade_without_thinking_or_eos(grader):
    manager = module.MultiThreadNaiveRewardManager(Tokenizer(), 0, num_reward_actors=1)
    assert manager(data_for([[CORRECT]])).sum().item() == 1


def test_precomputed_full_response_reward_cannot_bypass_after_thinking(grader):
    data = data_for([[CORRECT, CLOSE, WRONG, EOS]])
    data.batch["rm_scores"] = torch.ones_like(data.batch["responses"])
    manager = module.MultiThreadNaiveRewardManager(Tokenizer(), 0, num_reward_actors=1,
                                                  check_eos=True, score_after_thinking=True)
    with pytest.raises(ValueError, match="Precomputed"):
        manager(data)


def test_eos_gate_requires_a_token_id(grader):
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="eos_token_id"):
        module.MultiThreadNaiveRewardManager(tokenizer, 0, check_eos=True)
