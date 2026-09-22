# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    agg_loss,
    compute_fixed_n_rb_capped_cost_aware_marginrl_outcome_advantage,
    compute_fixed_n_rb_capped_fixed_q_cost_aware_marginrl_outcome_advantage,
    compute_fixed_n_rb_capped_marginrl_costs,
    compute_fixed_n_rb_capped_marginrl_costs_from_lengths,
    compute_fixed_n_rb_capped_thinking_cost_aware_marginrl_outcome_advantage,
    compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage,
    compute_fixed_n_rb_cost_aware_marginrl_success_gated_outcome_advantage,
    compute_fixed_n_rb_efficient_reasoning_cost_marginrl_outcome_advantage,
    compute_fixed_n_rb_efficient_reasoning_cost_marginrl_success_gated_outcome_advantage,
)


def make_response_mask(lengths: list[int], width: int) -> torch.Tensor:
    return torch.arange(width).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)


def make_token_rewards(rewards: list[float], width: int) -> torch.Tensor:
    token_rewards = torch.zeros((len(rewards), width), dtype=torch.float32)
    token_rewards[:, 0] = torch.tensor(rewards)
    return token_rewards


def trajectory_advantages(advantages: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    return (advantages * response_mask).sum(dim=-1) / response_mask.sum(dim=-1)


def test_closed_form_advantages_and_group_diagnostics():
    response_mask = make_response_mask([2, 3, 5], width=5)
    advantages, returns, diagnostics = compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage(
        token_level_rewards=make_token_rewards([1.0, 0.0, 1.0], width=5),
        response_mask=response_mask,
        index=np.array(["prompt", "prompt", "prompt"]),
        expected_group_size=3,
        return_diagnostics=True,
    )

    # M=2, total cost=10, and q_hat=0.2. Raw A=[0.3, -0.2, 0],
    # then multiply by N=3 to undo the actor's sequence mean.
    torch.testing.assert_close(
        diagnostics["raw_trajectory_advantages"],
        torch.tensor([0.3, -0.2, 0.0]),
    )
    torch.testing.assert_close(
        diagnostics["optimizer_trajectory_advantages"],
        torch.tensor([0.9, -0.6, 0.0]),
    )
    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask),
        torch.tensor([0.9, -0.6, 0.0]),
    )
    torch.testing.assert_close(returns, advantages)
    torch.testing.assert_close(diagnostics["group_q_hats"], torch.tensor([0.2]))
    torch.testing.assert_close(diagnostics["group_success_counts"], torch.tensor([2.0]))
    torch.testing.assert_close(diagnostics["group_total_costs"], torch.tensor([10.0]))
    torch.testing.assert_close(diagnostics["group_cost_means"], torch.tensor([10.0 / 3.0]))
    torch.testing.assert_close(diagnostics["group_accuracies"], torch.tensor([2.0 / 3.0]))
    assert torch.count_nonzero(advantages.masked_select(~response_mask)) == 0


def test_same_group_plugin_gives_zero_all_failure_update():
    response_mask = make_response_mask([1, 4], width=4)
    advantages, returns, diagnostics = compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage(
        token_level_rewards=make_token_rewards([0.0, 0.0], width=4),
        response_mask=response_mask,
        index=np.array(["prompt", "prompt"]),
        expected_group_size=2,
        return_diagnostics=True,
    )

    torch.testing.assert_close(diagnostics["group_q_hats"], torch.tensor([0.0]))
    torch.testing.assert_close(advantages, torch.zeros_like(advantages))
    torch.testing.assert_close(returns, advantages)


def test_success_gated_variant_zeros_failure_advantages_only():
    response_mask = make_response_mask([2, 3, 5], width=5)
    advantages, returns, diagnostics = (
        compute_fixed_n_rb_cost_aware_marginrl_success_gated_outcome_advantage(
            token_level_rewards=make_token_rewards([1.0, 0.0, 1.0], width=5),
            response_mask=response_mask,
            index=np.array(["prompt", "prompt", "prompt"]),
            expected_group_size=3,
            return_diagnostics=True,
        )
    )

    # M=2, total cost=10, and q_hat=0.2. The success branch is unchanged
    # from fixed-N RB MarginRL, while the failed trajectory is gated to zero.
    torch.testing.assert_close(
        diagnostics["raw_trajectory_advantages"],
        torch.tensor([0.3, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        diagnostics["optimizer_trajectory_advantages"],
        torch.tensor([0.9, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask),
        torch.tensor([0.9, 0.0, 0.0]),
    )
    torch.testing.assert_close(returns, advantages)


def test_success_gated_dispatch_uses_distinct_metrics_and_zero_wrong_update():
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    response_mask = make_response_mask([1, 3, 2, 4], width=4)
    trajectory_rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": make_token_rewards(trajectory_rewards.tolist(), width=4),
            "response_mask": response_mask,
        },
        non_tensors={"uid": np.array(["a", "a", "b", "b"])},
    )

    result = compute_advantage(
        data=data,
        adv_estimator=AdvantageEstimator.FIXED_N_RB_COST_AWARE_MARGINRL_SUCCESS_GATED,
        num_repeat=2,
        config={},
    )
    wrong_advantages = result.batch["advantages"][trajectory_rewards == 0]
    metrics = result.meta_info["fixed_n_rb_marginrl_metrics"]

    torch.testing.assert_close(wrong_advantages, torch.zeros_like(wrong_advantages))
    assert metrics["fixed_n_rb_marginrl_success_gated/failure_advantage_abs_max"] == 0.0
    assert metrics["fixed_n_rb_marginrl_success_gated/M_t_mean"] == pytest.approx(1.5)
    assert "fixed_n_rb_marginrl/M_t_mean" not in metrics


def test_capped_costs_normalize_lengths_and_floor_at_one_quarter():
    response_mask = make_response_mask([1, 2, 8, 16], width=16)

    lengths, costs, cap_mask = compute_fixed_n_rb_capped_marginrl_costs(
        response_mask=response_mask,
        cost_reference_tokens=8,
        max_inverse_cost=4.0,
    )

    torch.testing.assert_close(lengths, torch.tensor([1.0, 2.0, 8.0, 16.0]))
    torch.testing.assert_close(costs, torch.tensor([0.25, 0.25, 1.0, 2.0]))
    torch.testing.assert_close(cap_mask, torch.tensor([True, False, False, False]))


def test_thinking_costs_use_explicit_span_lengths_and_floor_at_one_quarter():
    lengths, costs, cap_mask = compute_fixed_n_rb_capped_marginrl_costs_from_lengths(
        trajectory_lengths=torch.tensor([0.0, 511.0, 512.0, 2048.0, 4096.0]),
        cost_reference_tokens=2048,
        max_inverse_cost=4.0,
    )

    torch.testing.assert_close(
        lengths,
        torch.tensor([0.0, 511.0, 512.0, 2048.0, 4096.0]),
    )
    torch.testing.assert_close(
        costs,
        torch.tensor([0.25, 0.25, 0.25, 1.0, 2.0]),
    )
    torch.testing.assert_close(
        cap_mask,
        torch.tensor([True, True, False, False, False]),
    )


def test_thinking_cost_estimator_uses_explicit_lengths_for_plugin_rate():
    response_mask = make_response_mask([3, 3, 3, 3], width=3)
    _, _, diagnostics = (
        compute_fixed_n_rb_capped_thinking_cost_aware_marginrl_outcome_advantage(
            token_level_rewards=make_token_rewards([1.0, 0.0, 1.0, 0.0], width=3),
            response_mask=response_mask,
            index=np.array(["prompt"] * 4),
            trajectory_lengths=torch.tensor([0.0, 512.0, 2048.0, 4096.0]),
            expected_group_size=4,
            cost_reference_tokens=2048,
            max_inverse_cost=4.0,
            return_diagnostics=True,
        )
    )

    # Costs are [1/4, 1/4, 1, 2], so q_hat=2/(7/2)=4/7.
    torch.testing.assert_close(diagnostics["group_q_hats"], torch.tensor([4.0 / 7.0]))
    torch.testing.assert_close(
        diagnostics["trajectory_lengths"],
        torch.tensor([0.0, 512.0, 2048.0, 4096.0]),
    )


def test_capped_cost_estimator_uses_effective_cost_in_q_hat_and_advantages():
    response_mask = make_response_mask([1, 2, 8, 16], width=16)
    advantages, returns, diagnostics = compute_fixed_n_rb_capped_cost_aware_marginrl_outcome_advantage(
        token_level_rewards=make_token_rewards([1.0, 0.0, 1.0, 0.0], width=16),
        response_mask=response_mask,
        index=np.array(["prompt"] * 4),
        expected_group_size=4,
        cost_reference_tokens=8,
        max_inverse_cost=4.0,
        return_diagnostics=True,
    )

    # Effective costs are [1/4, 1/4, 1, 2], so M=2 and q_hat=2/(7/2)=4/7.
    torch.testing.assert_close(diagnostics["group_q_hats"], torch.tensor([4.0 / 7.0]))
    torch.testing.assert_close(
        diagnostics["raw_trajectory_advantages"],
        torch.tensor([3.0 / 7.0, -1.0 / 21.0, 3.0 / 14.0, -8.0 / 21.0]),
    )
    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask),
        4.0 * diagnostics["raw_trajectory_advantages"],
    )
    torch.testing.assert_close(returns, advantages)


def test_capped_cost_dispatch_records_tokens_costs_and_cap_ratio():
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    response_mask = make_response_mask([1, 2, 8, 16], width=16)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": make_token_rewards([1.0, 0.0, 1.0, 0.0], width=16),
            "response_mask": response_mask,
        },
        non_tensors={"uid": np.array(["prompt"] * 4)},
    )

    result = compute_advantage(
        data=data,
        adv_estimator=AdvantageEstimator.FIXED_N_RB_CAPPED_COST_AWARE_MARGINRL,
        num_repeat=4,
        config={"cost_reference_tokens": 8, "max_inverse_cost": 4.0},
    )
    metrics = result.meta_info["fixed_n_rb_marginrl_metrics"]

    assert metrics["fixed_n_rb_capped_marginrl/trajectory_tokens_mean"] == pytest.approx(6.75)
    assert metrics["fixed_n_rb_capped_marginrl/cost_mean"] == pytest.approx(0.875)
    assert metrics["fixed_n_rb_capped_marginrl/cost_min"] == pytest.approx(0.25)
    assert metrics["fixed_n_rb_capped_marginrl/cap_ratio"] == pytest.approx(0.25)


def test_thinking_cost_dispatch_uses_reward_spans_and_logs_gate_metrics():
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    response_mask = make_response_mask([3, 3, 3, 3], width=3)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": make_token_rewards(
                [1.0, 0.0, 1.0, 0.0], width=3
            ),
            "response_mask": response_mask,
        },
        non_tensors={
            "uid": np.array(["prompt"] * 4),
            "thinking_tokens": np.array([0, 512, 2048, 4096]),
            "raw_math_accuracy": np.array([1.0, 1.0, 1.0, 0.0]),
            "post_think_pre_box_tokens": np.array([10, 512, 20, -1]),
            "has_think_open": np.ones(4),
            "has_think_close": np.array([1.0, 1.0, 1.0, 0.0]),
            "has_box_after_think": np.array([1.0, 1.0, 1.0, 0.0]),
            "thinking_span_censored": np.array([0.0, 0.0, 0.0, 1.0]),
            "post_think_length_pass": np.array([1.0, 0.0, 1.0, 0.0]),
            "gated_reward": np.array([1.0, 0.0, 1.0, 0.0]),
        },
    )

    result = compute_advantage(
        data=data,
        adv_estimator=(
            AdvantageEstimator.FIXED_N_RB_CAPPED_THINKING_COST_AWARE_MARGINRL
        ),
        num_repeat=4,
        config={"cost_reference_tokens": 2048, "max_inverse_cost": 4.0},
    )
    metrics = result.meta_info["fixed_n_rb_marginrl_metrics"]
    prefix = "fixed_n_rb_capped_thinking_marginrl"

    assert metrics[f"{prefix}/trajectory_tokens_mean"] == pytest.approx(1664.0)
    assert metrics[f"{prefix}/cost_mean"] == pytest.approx(0.875)
    assert metrics[f"{prefix}/cap_ratio"] == pytest.approx(0.25)
    assert metrics[f"{prefix}/raw_math_accuracy_mean"] == pytest.approx(0.75)
    assert metrics[f"{prefix}/gated_reward_mean"] == pytest.approx(0.5)
    assert metrics[f"{prefix}/correct_reward_retention"] == pytest.approx(2.0 / 3.0)
    assert metrics[f"{prefix}/post_think_pre_box_tokens_max"] == pytest.approx(512.0)


def test_capped_fixed_q_estimator_uses_two_for_every_group():
    response_mask = make_response_mask([1, 2, 8, 16], width=16)
    advantages, returns, diagnostics = (
        compute_fixed_n_rb_capped_fixed_q_cost_aware_marginrl_outcome_advantage(
            token_level_rewards=make_token_rewards([1.0, 0.0, 1.0, 0.0], width=16),
            response_mask=response_mask,
            index=np.array(["prompt"] * 4),
            expected_group_size=4,
            cost_reference_tokens=8,
            max_inverse_cost=4.0,
            fixed_q_hat=2.0,
            return_diagnostics=True,
        )
    )

    # Effective costs are [1/4, 1/4, 1, 2], M=2, and q_hat is fixed at 2.
    torch.testing.assert_close(diagnostics["group_q_hats"], torch.tensor([2.0]))
    torch.testing.assert_close(
        diagnostics["raw_trajectory_advantages"],
        torch.tensor([1.0 / 4.0, -1.0 / 6.0, -1.0 / 2.0, -4.0 / 3.0]),
    )
    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask),
        4.0 * diagnostics["raw_trajectory_advantages"],
    )
    torch.testing.assert_close(returns, advantages)


def test_capped_fixed_q_all_failure_group_retains_cost_update():
    response_mask = make_response_mask([1, 2], width=2)
    _, _, diagnostics = compute_fixed_n_rb_capped_fixed_q_cost_aware_marginrl_outcome_advantage(
        token_level_rewards=make_token_rewards([0.0, 0.0], width=2),
        response_mask=response_mask,
        index=np.array(["prompt", "prompt"]),
        expected_group_size=2,
        cost_reference_tokens=8,
        max_inverse_cost=4.0,
        fixed_q_hat=2.0,
        return_diagnostics=True,
    )

    torch.testing.assert_close(diagnostics["group_q_hats"], torch.tensor([2.0]))
    torch.testing.assert_close(
        diagnostics["raw_trajectory_advantages"],
        torch.tensor([-0.5, -0.5]),
    )


def test_capped_fixed_q_dispatch_uses_distinct_metrics():
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    response_mask = make_response_mask([1, 2, 8, 16], width=16)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": make_token_rewards([1.0, 0.0, 1.0, 0.0], width=16),
            "response_mask": response_mask,
        },
        non_tensors={"uid": np.array(["prompt"] * 4)},
    )

    result = compute_advantage(
        data=data,
        adv_estimator=AdvantageEstimator.FIXED_N_RB_CAPPED_FIXED_Q_COST_AWARE_MARGINRL,
        num_repeat=4,
        config={
            "cost_reference_tokens": 8,
            "max_inverse_cost": 4.0,
            "fixed_q_hat": 2.0,
        },
    )
    metrics = result.meta_info["fixed_n_rb_marginrl_metrics"]

    assert metrics["fixed_n_rb_capped_fixed_q_marginrl/q_hat_mean"] == pytest.approx(2.0)
    assert metrics["fixed_n_rb_capped_fixed_q_marginrl/q_hat_std"] == pytest.approx(0.0)
    assert metrics["fixed_n_rb_capped_fixed_q_marginrl/cost_min"] == pytest.approx(0.25)
    assert metrics["fixed_n_rb_capped_fixed_q_marginrl/cap_ratio"] == pytest.approx(0.25)
    assert "fixed_n_rb_capped_marginrl/q_hat_mean" not in metrics


def test_efficient_reasoning_cost_normalizes_all_lengths_per_prompt_and_estimates_q_hat():
    response_mask = make_response_mask([2, 4, 6, 10], width=10)
    advantages, returns, diagnostics = (
        compute_fixed_n_rb_efficient_reasoning_cost_marginrl_outcome_advantage(
            token_level_rewards=make_token_rewards([1.0, 0.0, 0.0, 1.0], width=10),
            response_mask=response_mask,
            index=np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"]),
            expected_group_size=2,
            return_diagnostics=True,
        )
    )

    # Both correct and incorrect lengths contribute to each prompt's population
    # statistics: means [3, 8] and standard deviations [1, 2].
    expected_group_means = torch.tensor([3.0, 8.0])
    expected_group_stds = torch.tensor([1.0, 2.0])
    expected_relative_lengths = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    expected_costs = torch.sigmoid(expected_relative_lengths)
    expected_q_hats = torch.stack(
        [
            1.0 / expected_costs[:2].sum(),
            1.0 / expected_costs[2:].sum(),
        ]
    )
    expected_raw = torch.tensor(
        [
            1.0 - expected_q_hats[0] * expected_costs[0],
            -expected_q_hats[0] * expected_costs[1] / 2.0,
            -expected_q_hats[1] * expected_costs[2] / 2.0,
            1.0 - expected_q_hats[1] * expected_costs[3],
        ]
    )
    torch.testing.assert_close(diagnostics["trajectory_relative_lengths"], expected_relative_lengths)
    torch.testing.assert_close(diagnostics["trajectory_costs"], expected_costs)
    torch.testing.assert_close(diagnostics["group_length_means"], expected_group_means)
    torch.testing.assert_close(diagnostics["group_length_stds"], expected_group_stds)
    torch.testing.assert_close(diagnostics["group_q_hats"], expected_q_hats)
    torch.testing.assert_close(diagnostics["raw_trajectory_advantages"], expected_raw)
    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask),
        2.0 * expected_raw,
    )
    torch.testing.assert_close(returns, advantages)


def test_efficient_reasoning_cost_supports_fixed_q_hat_without_gating_failures():
    response_mask = make_response_mask([2, 4, 6, 10], width=10)
    advantages, returns, diagnostics = (
        compute_fixed_n_rb_efficient_reasoning_cost_marginrl_outcome_advantage(
            token_level_rewards=make_token_rewards([1.0, 0.0, 0.0, 0.0], width=10),
            response_mask=response_mask,
            index=np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"]),
            expected_group_size=2,
            config={"efficient_reasoning_fixed_q_hat": 0.8},
            return_diagnostics=True,
        )
    )

    expected_costs = torch.sigmoid(torch.tensor([-1.0, 1.0, -1.0, 1.0]))
    expected_raw = torch.stack(
        (
            1.0 - 0.8 * expected_costs[0],
            -0.8 * expected_costs[1] / 2.0,
            -0.8 * expected_costs[2],
            -0.8 * expected_costs[3],
        )
    )
    torch.testing.assert_close(diagnostics["trajectory_costs"], expected_costs)
    torch.testing.assert_close(diagnostics["group_q_hats"], torch.full((2,), 0.8))
    torch.testing.assert_close(diagnostics["raw_trajectory_advantages"], expected_raw)
    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask),
        2.0 * expected_raw,
    )
    torch.testing.assert_close(returns, advantages)


def test_efficient_reasoning_success_gating_zeroes_only_failure_advantages():
    response_mask = make_response_mask([2, 4, 6, 10], width=10)
    rewards = make_token_rewards([1.0, 0.0, 0.0, 1.0], width=10)
    kwargs = {
        "token_level_rewards": rewards,
        "response_mask": response_mask,
        "index": np.array(["prompt"] * 4),
        "expected_group_size": 4,
        "return_diagnostics": True,
    }
    _, _, ungated_diagnostics = (
        compute_fixed_n_rb_efficient_reasoning_cost_marginrl_outcome_advantage(**kwargs)
    )
    advantages, returns, gated_diagnostics = (
        compute_fixed_n_rb_efficient_reasoning_cost_marginrl_success_gated_outcome_advantage(
            **kwargs
        )
    )
    success_mask = gated_diagnostics["trajectory_rewards"].bool()
    failure_mask = ~success_mask

    torch.testing.assert_close(
        gated_diagnostics["trajectory_costs"], ungated_diagnostics["trajectory_costs"]
    )
    torch.testing.assert_close(
        gated_diagnostics["group_q_hats"], ungated_diagnostics["group_q_hats"]
    )
    torch.testing.assert_close(
        gated_diagnostics["raw_trajectory_advantages"][success_mask],
        ungated_diagnostics["raw_trajectory_advantages"][success_mask],
    )
    torch.testing.assert_close(
        gated_diagnostics["raw_trajectory_advantages"][failure_mask],
        torch.zeros_like(gated_diagnostics["raw_trajectory_advantages"][failure_mask]),
    )
    torch.testing.assert_close(
        gated_diagnostics["optimizer_trajectory_advantages"][failure_mask],
        torch.zeros_like(
            gated_diagnostics["optimizer_trajectory_advantages"][failure_mask]
        ),
    )
    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask)[failure_mask],
        torch.zeros_like(trajectory_advantages(advantages, response_mask)[failure_mask]),
    )
    torch.testing.assert_close(returns, advantages)


def test_efficient_reasoning_cost_all_failure_group_has_zero_advantage():
    response_mask = make_response_mask([1, 4], width=4)
    advantages, returns, diagnostics = (
        compute_fixed_n_rb_efficient_reasoning_cost_marginrl_outcome_advantage(
            token_level_rewards=make_token_rewards([0.0, 0.0], width=4),
            response_mask=response_mask,
            index=np.array(["prompt", "prompt"]),
            expected_group_size=2,
            return_diagnostics=True,
        )
    )

    torch.testing.assert_close(advantages, torch.zeros_like(advantages))
    torch.testing.assert_close(returns, advantages)
    expected_relative_lengths = torch.tensor([-1.0, 1.0])
    torch.testing.assert_close(diagnostics["trajectory_costs"], torch.sigmoid(expected_relative_lengths))
    torch.testing.assert_close(diagnostics["trajectory_relative_lengths"], expected_relative_lengths)
    torch.testing.assert_close(diagnostics["group_length_means"], torch.tensor([2.5]))
    torch.testing.assert_close(diagnostics["group_length_stds"], torch.tensor([1.5]))
    torch.testing.assert_close(diagnostics["group_q_hats"], torch.zeros(1))


def test_efficient_reasoning_cost_constant_length_group_uses_neutral_cost():
    response_mask = make_response_mask([4, 4], width=4)
    advantages, returns, diagnostics = (
        compute_fixed_n_rb_efficient_reasoning_cost_marginrl_outcome_advantage(
            token_level_rewards=make_token_rewards([1.0, 0.0], width=4),
            response_mask=response_mask,
            index=np.array(["prompt", "prompt"]),
            expected_group_size=2,
            return_diagnostics=True,
        )
    )

    torch.testing.assert_close(diagnostics["trajectory_relative_lengths"], torch.zeros(2))
    torch.testing.assert_close(diagnostics["trajectory_costs"], torch.full((2,), 0.5))
    torch.testing.assert_close(diagnostics["group_length_means"], torch.tensor([4.0]))
    torch.testing.assert_close(diagnostics["group_length_stds"], torch.zeros(1))
    torch.testing.assert_close(diagnostics["group_q_hats"], torch.ones(1))
    torch.testing.assert_close(
        diagnostics["raw_trajectory_advantages"],
        torch.tensor([0.5, -0.25]),
    )
    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask),
        torch.tensor([1.0, -0.5]),
    )
    torch.testing.assert_close(returns, advantages)


def test_efficient_reasoning_cost_dispatch_logs_per_prompt_metrics():
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    lengths = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    response_mask = make_response_mask(lengths.int().tolist(), width=12)
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": make_token_rewards(rewards.tolist(), width=12),
            "response_mask": response_mask,
        },
        non_tensors={"uid": np.array(["prompt"] * 6)},
    )

    result = compute_advantage(
        data=data,
        adv_estimator=AdvantageEstimator.FIXED_N_RB_EFFICIENT_REASONING_COST_MARGINRL,
        num_repeat=6,
        config={"efficient_reasoning_epsilon": 1e-7},
    )
    metrics = result.meta_info["fixed_n_rb_marginrl_metrics"]
    prefix = "fixed_n_rb_er_cost_marginrl"
    expected_group_std = lengths.std(unbiased=False).item()
    expected_relative_lengths = (lengths - lengths.mean()) / (expected_group_std + 1e-7)
    expected_costs = torch.sigmoid(expected_relative_lengths)
    expected_q_hat = rewards.sum() / expected_costs.sum()
    expected_raw_advantages = torch.where(
        rewards.bool(),
        (1.0 - expected_q_hat * expected_costs) / rewards.sum(),
        -expected_q_hat * expected_costs / (rewards.sum() + 1.0),
    )

    torch.testing.assert_close(
        trajectory_advantages(result.batch["advantages"], response_mask),
        6.0 * expected_raw_advantages,
    )
    assert metrics[f"{prefix}/group_length_mean_mean"] == pytest.approx(lengths.mean().item())
    assert metrics[f"{prefix}/group_length_std_mean"] == pytest.approx(expected_group_std)
    assert metrics[f"{prefix}/relative_length_mean"] == pytest.approx(0.0, abs=1e-7)
    assert metrics[f"{prefix}/relative_length_std"] == pytest.approx(1.0)
    assert metrics[f"{prefix}/cost_mean"] == pytest.approx(0.5)
    assert metrics[f"{prefix}/failure_advantage_abs_max"] == pytest.approx(
        expected_raw_advantages[rewards == 0].abs().max().item()
    )
    assert metrics[f"{prefix}/q_hat_mean"] == pytest.approx(expected_q_hat.item())
    assert metrics[f"{prefix}/early_eos_rate"] == pytest.approx(1.0 / 6.0)
    assert metrics[f"{prefix}/early_eos_fail_rate"] == pytest.approx(0.5)
    assert metrics[f"{prefix}/mean_len_fail"] == pytest.approx(3.0)
    assert metrics[f"{prefix}/mean_len_success"] == pytest.approx(9.0)
    expected_optimizer_advantages = 6.0 * expected_raw_advantages
    assert metrics[f"{prefix}/adv_short_fail"] == pytest.approx(
        expected_optimizer_advantages[0].item()
    )
    assert metrics[f"{prefix}/adv_normal_fail"] == pytest.approx(
        expected_optimizer_advantages[1].item()
    )
    assert metrics[f"{prefix}/adv_success"] == pytest.approx(
        expected_optimizer_advantages[rewards == 1].mean().item()
    )
    assert metrics[f"{prefix}/frac_negative_adv_success"] == pytest.approx(
        (expected_optimizer_advantages[rewards == 1] < 0).float().mean().item()
    )
    assert f"{prefix}/relative_cost_alpha" not in metrics
    assert f"{prefix}/correct_cost_count" not in metrics


def test_efficient_reasoning_success_gated_dispatch_logs_early_eos_metrics():
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    response_mask = make_response_mask([2, 4, 6, 8], width=8)
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0])
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": make_token_rewards(rewards.tolist(), width=8),
            "response_mask": response_mask,
        },
        non_tensors={"uid": np.array(["prompt"] * 4)},
    )

    result = compute_advantage(
        data=data,
        adv_estimator=(
            AdvantageEstimator.FIXED_N_RB_EFFICIENT_REASONING_COST_MARGINRL_SUCCESS_GATED
        ),
        num_repeat=4,
        config={"efficient_reasoning_epsilon": 1e-7},
    )
    metrics = result.meta_info["fixed_n_rb_marginrl_metrics"]
    prefix = "fixed_n_rb_er_cost_marginrl_success_gated"
    optimizer_advantages = trajectory_advantages(result.batch["advantages"], response_mask)

    torch.testing.assert_close(
        optimizer_advantages[rewards == 0],
        torch.zeros_like(optimizer_advantages[rewards == 0]),
    )
    assert metrics[f"{prefix}/early_eos_rate"] == pytest.approx(0.25)
    assert metrics[f"{prefix}/early_eos_fail_rate"] == pytest.approx(0.5)
    assert metrics[f"{prefix}/mean_len_fail"] == pytest.approx(3.0)
    assert metrics[f"{prefix}/mean_len_success"] == pytest.approx(7.0)
    assert metrics[f"{prefix}/adv_short_fail"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/adv_normal_fail"] == pytest.approx(0.0)
    assert metrics[f"{prefix}/adv_success"] == pytest.approx(
        optimizer_advantages[rewards == 1].mean().item()
    )
    assert metrics[f"{prefix}/frac_negative_adv_success"] == pytest.approx(
        (optimizer_advantages[rewards == 1] < 0).float().mean().item()
    )
    assert "fixed_n_rb_er_cost_marginrl/early_eos_rate" not in metrics


@pytest.mark.parametrize(
    ("rewards", "empty_subset_metrics"),
    [
        (
            [1.0, 1.0],
            ["early_eos_fail_rate", "mean_len_fail", "adv_short_fail", "adv_normal_fail"],
        ),
        ([0.0, 0.0], ["mean_len_success", "adv_success", "frac_negative_adv_success"]),
    ],
)
def test_efficient_reasoning_early_eos_metrics_zero_empty_subsets(
    rewards: list[float], empty_subset_metrics: list[str]
):
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    response_mask = make_response_mask([2, 4], width=4)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": make_token_rewards(rewards, width=4),
            "response_mask": response_mask,
        },
        non_tensors={"uid": np.array(["prompt", "prompt"])},
    )

    result = compute_advantage(
        data=data,
        adv_estimator=AdvantageEstimator.FIXED_N_RB_EFFICIENT_REASONING_COST_MARGINRL,
        num_repeat=2,
        config={"efficient_reasoning_epsilon": 1e-7},
    )
    metrics = result.meta_info["fixed_n_rb_marginrl_metrics"]
    prefix = "fixed_n_rb_er_cost_marginrl"

    for metric_name in empty_subset_metrics:
        assert metrics[f"{prefix}/{metric_name}"] == pytest.approx(0.0)


def test_cost_scale_does_not_change_advantages():
    response_mask = make_response_mask([2, 3, 5], width=5)
    kwargs = {
        "token_level_rewards": make_token_rewards([1.0, 0.0, 1.0], width=5),
        "response_mask": response_mask,
        "index": np.array(["prompt", "prompt", "prompt"]),
        "expected_group_size": 3,
        "return_diagnostics": True,
    }
    _, _, raw_cost_diagnostics = compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage(**kwargs)
    _, _, scaled_cost_diagnostics = compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage(
        **kwargs,
        trajectory_cost_mask=response_mask.float() / 4096.0,
    )

    torch.testing.assert_close(
        raw_cost_diagnostics["raw_trajectory_advantages"],
        scaled_cost_diagnostics["raw_trajectory_advantages"],
    )


def test_token_mean_preserves_group_gradient_with_inverse_mean_length_scale():
    response_mask = make_response_mask([2, 3, 5], width=5)
    _, _, diagnostics = compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage(
        token_level_rewards=make_token_rewards([1.0, 0.0, 1.0], width=5),
        response_mask=response_mask,
        index=np.array(["prompt", "prompt", "prompt"]),
        expected_group_size=3,
        return_diagnostics=True,
    )
    token_terms = torch.arange(1, 16, dtype=torch.float32).reshape(3, 5)
    optimizer_loss_mat = diagnostics["optimizer_trajectory_advantages"].unsqueeze(-1) * token_terms
    sequence_aggregated = agg_loss(optimizer_loss_mat, response_mask, "seq-mean-token-sum")
    token_aggregated = agg_loss(optimizer_loss_mat, response_mask, "token-mean")
    sequence_sums = (token_terms * response_mask).sum(dim=-1)
    group_gradient = (diagnostics["raw_trajectory_advantages"] * sequence_sums).sum()
    mean_response_length = response_mask.sum() / response_mask.shape[0]

    torch.testing.assert_close(sequence_aggregated, group_gradient)
    torch.testing.assert_close(token_aggregated, group_gradient / mean_response_length)


def test_dispatch_records_requested_and_applicable_wandb_metrics():
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    response_mask = make_response_mask([1, 3, 2, 4], width=4)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": make_token_rewards([1.0, 0.0, 1.0, 1.0], width=4),
            "response_mask": response_mask,
        },
        non_tensors={"uid": np.array(["a", "a", "b", "b"])},
    )

    result = compute_advantage(
        data=data,
        adv_estimator=AdvantageEstimator.FIXED_N_RB_COST_AWARE_MARGINRL,
        num_repeat=2,
        config={},
    )
    metrics = result.meta_info["fixed_n_rb_marginrl_metrics"]

    assert metrics["fixed_n_rb_marginrl/q_hat_mean"] == pytest.approx((0.25 + 2.0 / 6.0) / 2.0)
    assert metrics["fixed_n_rb_marginrl/q_hat_std"] == pytest.approx((2.0 / 6.0 - 0.25) / 2.0)
    assert metrics["fixed_n_rb_marginrl/M_t_mean"] == pytest.approx(1.5)
    assert metrics["fixed_n_rb_marginrl/M_t_std"] == pytest.approx(0.5)
    assert metrics["fixed_n_rb_marginrl/total_cost_mean"] == pytest.approx(5.0)
    assert metrics["fixed_n_rb_marginrl/total_cost_std"] == pytest.approx(1.0)
    assert metrics["fixed_n_rb_marginrl/cost_mean"] == pytest.approx(2.5)
    assert metrics["fixed_n_rb_marginrl/cost_std"] == pytest.approx(5.0**0.5 / 2.0)
    assert metrics["fixed_n_rb_marginrl/accuracy_mean"] == pytest.approx(0.75)
    assert metrics["fixed_n_rb_marginrl/accuracy_std"] == pytest.approx(0.25)
    assert metrics["fixed_n_rb_marginrl/trajectory_tokens_min"] == pytest.approx(1.0)
    assert metrics["fixed_n_rb_marginrl/trajectory_tokens_max"] == pytest.approx(4.0)
    assert metrics["fixed_n_rb_marginrl/zero_success_group_ratio"] == pytest.approx(0.0)


def test_requires_binary_rewards():
    with pytest.raises(ValueError, match="binary trajectory rewards"):
        compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage(
            token_level_rewards=make_token_rewards([0.5, 1.0], width=2),
            response_mask=make_response_mask([1, 2], width=2),
            index=np.array(["prompt", "prompt"]),
            expected_group_size=2,
        )


def test_requires_positive_trajectory_costs():
    with pytest.raises(ValueError, match="strictly positive"):
        compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage(
            token_level_rewards=make_token_rewards([0.0, 1.0], width=2),
            response_mask=make_response_mask([0, 2], width=2),
            index=np.array(["prompt", "prompt"]),
            expected_group_size=2,
        )


def test_requires_fixed_expected_rollout_count():
    with pytest.raises(ValueError, match="rollout count mismatch"):
        compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage(
            token_level_rewards=make_token_rewards([0.0, 1.0], width=2),
            response_mask=make_response_mask([1, 2], width=2),
            index=np.array(["prompt", "prompt"]),
            expected_group_size=16,
        )
