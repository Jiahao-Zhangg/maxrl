"""Check the cross-context formula independently of PPO's loss reduction."""

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    compute_cross_context_f_cov_outcome_advantage,
    compute_fixed_n_rb_offset_cost_aware_marginrl_outcome_advantage,
    get_adv_estimator_fn,
)


def make_batch(lengths, rewards):
    mask = torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    token_rewards = torch.zeros_like(mask, dtype=torch.float32)
    token_rewards[:, 0] = torch.tensor(rewards, dtype=torch.float32)
    return mask, token_rewards


def reference(lengths, rewards, uids, offset=256):
    groups = {uid: [i for i, value in enumerate(uids) if value == uid] for uid in set(uids)}
    successes = {uid: sum(rewards[i] for i in positions) for uid, positions in groups.items()}
    mean_cost = sum(length + offset for length in lengths) / len(lengths)
    h = sum(m / (m + 1) for m in successes.values()) / len(groups)
    advantages = []
    for i, uid in enumerate(uids):
        cost_ratio = (lengths[i] + offset) / mean_cost
        m = successes[uid]
        if rewards[i]:
            advantage = len(groups[uid]) / m - cost_ratio * (h + 1 / (len(groups) * (m + 1)))
        else:
            advantage = -cost_ratio * h
        advantages.append(advantage)
    return torch.tensor(advantages), h, mean_cost


def test_exact_k256_n16_formula_with_noncontiguous_prompt_groups():
    k, n = 256, 16
    lengths = [(prompt * 7 + response * 3) % 40 + 1 for prompt in range(k) for response in range(n)]
    rewards = [int(response < prompt % (n + 1)) for prompt in range(k) for response in range(n)]
    uids = [f"prompt-{prompt}" for prompt in range(k) for _ in range(n)]
    order = np.random.default_rng(79).permutation(k * n)
    lengths, rewards, uids = ([values[i] for i in order] for values in (lengths, rewards, uids))
    mask, token_rewards = make_batch(lengths, rewards)
    advantages, returns, diagnostics = get_adv_estimator_fn("f_cov")(
        token_rewards.requires_grad_(), mask.float().requires_grad_(), np.array(uids),
        expected_group_size=n, config={"cost_offset_tokens": 256, "f_cov_num_prompts": k},
        return_diagnostics=True,
    )
    expected, h, mean_cost = reference(lengths, rewards, uids)
    torch.testing.assert_close(advantages, expected.unsqueeze(-1) * mask)
    torch.testing.assert_close(returns, advantages)
    torch.testing.assert_close(diagnostics["optimizer_trajectory_advantages"], expected)
    assert diagnostics["cross_context_h"].item() == pytest.approx(h)
    assert diagnostics["global_cost_mean"].item() == pytest.approx(mean_cost)
    assert not advantages.requires_grad
    assert all(not value.requires_grad for value in diagnostics.values() if isinstance(value, torch.Tensor))


@pytest.mark.parametrize("rewards", [[0] * 4, [1, 0, 0, 0], [1, 0, 1, 0], [1] * 4])
def test_one_prompt_reduces_to_previous_offset_formula(rewards):
    mask, token_rewards = make_batch([1, 7, 11, 40], rewards)
    kwargs = dict(token_level_rewards=token_rewards, response_mask=mask, index=np.array(["same"] * 4))
    actual, _ = compute_cross_context_f_cov_outcome_advantage(**kwargs)
    previous, _ = compute_fixed_n_rb_offset_cost_aware_marginrl_outcome_advantage(**kwargs)
    torch.testing.assert_close(actual, previous)


def test_failed_prompt_receives_negative_advantages_from_other_prompts():
    lengths = [1, 3, 19, 37]
    rewards = [0, 0, 1, 0]
    uids = ["failed", "failed", "mixed", "mixed"]
    mask, token_rewards = make_batch(lengths, rewards)
    result, _, diagnostics = compute_cross_context_f_cov_outcome_advantage(
        token_rewards, mask, uids, return_diagnostics=True,
    )
    expected, _, _ = reference(lengths, rewards, uids)
    torch.testing.assert_close(result, expected.unsqueeze(-1) * mask)
    assert torch.all(diagnostics["optimizer_trajectory_advantages"][:2] < 0)
    zero, _ = compute_cross_context_f_cov_outcome_advantage(torch.zeros_like(token_rewards), mask, uids)
    assert torch.count_nonzero(zero) == 0


def test_all_success_batch_keeps_correction_and_is_not_centered():
    k, n = 4, 3
    mask, token_rewards = make_batch([5] * (k * n), [1] * (k * n))
    result, _ = compute_cross_context_f_cov_outcome_advantage(token_rewards, mask, np.repeat(np.arange(k), n))
    expected = (k - 1) / (k * (n + 1))
    torch.testing.assert_close(result, torch.full_like(result, expected))


@pytest.mark.parametrize("estimator", ["f_cov", AdvantageEstimator.F_COV])
def test_dispatch_uses_full_cost_mask_and_does_not_apply_grpo_normalization(estimator):
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    lengths, rewards, uids = [5, 11, 7, 29], [1, 0, 0, 0], ["a", "b", "a", "b"]
    mask, token_rewards = make_batch(lengths, rewards)
    loss_mask = torch.zeros_like(mask)
    loss_mask[:, :2] = True
    data = DataProto.from_dict(
        tensors={"response_mask": mask, "loss_mask": loss_mask, "token_level_rewards": token_rewards},
        non_tensors={"uid": np.array(uids)},
    )
    result = compute_advantage(
        data, estimator, num_repeat=2, multi_turn=True, norm_adv_by_std_in_grpo=True,
        config={"cost_offset_tokens": 256, "f_cov_num_prompts": 2},
    )
    expected, h, mean_cost = reference(lengths, rewards, uids)
    torch.testing.assert_close(result.batch["advantages"], expected.unsqueeze(-1) * loss_mask)
    metrics = result.meta_info["f_cov_metrics"]
    assert metrics["f_cov/H"] == pytest.approx(h)
    assert metrics["f_cov/global_cost_mean"] == pytest.approx(mean_cost)
    assert metrics["f_cov/num_prompts"] == 2
    assert metrics["f_cov/zero_success_group_ratio"] == 0.5
    assert all(key.startswith("f_cov/") for key in metrics)


@pytest.mark.parametrize("kwargs,error", [
    ({"expected_group_size": 16}, "responses per prompt"),
    ({"expected_num_prompts": 256}, "full rollout batch"),
    ({"config": {"f_cov_num_prompts": 256}}, "full rollout batch"),
    ({"index": ["a", "a", "a", "b"]}, "same number of responses"),
    ({"index": ["a"]}, "one prompt UID"),
    ({"cost_offset_tokens": -1}, "finite and nonnegative"),
])
def test_rejects_invalid_batch_scope_or_cost(kwargs, error):
    mask, rewards = make_batch([2] * 4, [1, 0, 0, 0])
    inputs = {"token_level_rewards": rewards, "response_mask": mask, "index": ["a", "a", "b", "b"]}
    with pytest.raises(ValueError, match=error):
        compute_cross_context_f_cov_outcome_advantage(**(inputs | kwargs))


def test_rejects_nonbinary_rewards_and_mismatched_masks():
    mask, rewards = make_batch([2, 2], [0.5, 0])
    with pytest.raises(ValueError, match="binary"):
        compute_cross_context_f_cov_outcome_advantage(rewards, mask, ["a", "a"])
    with pytest.raises(ValueError, match="matching shapes"):
        compute_cross_context_f_cov_outcome_advantage(torch.zeros_like(rewards), mask, ["a", "a"], trajectory_cost_mask=torch.ones(2, 3))
