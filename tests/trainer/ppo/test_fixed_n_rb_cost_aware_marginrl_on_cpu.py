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
    compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage,
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


def test_sequence_token_sum_and_group_scaling_match_theoretical_loss():
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
    actual = agg_loss(optimizer_loss_mat, response_mask, "seq-mean-token-sum")
    sequence_sums = (token_terms * response_mask).sum(dim=-1)
    expected = (diagnostics["raw_trajectory_advantages"] * sequence_sums).sum()

    torch.testing.assert_close(actual, expected)


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
