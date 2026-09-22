# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
    compute_cost_aware_maxrl_costs,
    compute_cost_aware_maxrl_outcome_advantage,
)
from verl.trainer.ppo.metric_utils import compute_cost_aware_maxrl_metrics


def make_response_mask(lengths: list[int], width: int) -> torch.Tensor:
    return torch.arange(width).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)


def make_token_rewards(rewards: list[float], width: int) -> torch.Tensor:
    token_rewards = torch.zeros((len(rewards), width), dtype=torch.float32)
    token_rewards[:, 0] = torch.tensor(rewards)
    return token_rewards


def trajectory_advantages(advantages: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    return (advantages * response_mask).sum(dim=-1) / response_mask.sum(dim=-1)


def test_costs_apply_inverse_cost_cap_and_report_cap_ratio():
    response_mask = make_response_mask([1, 2, 8, 16], width=16)

    lengths, costs, inverse_costs, cap_mask = compute_cost_aware_maxrl_costs(
        response_mask=response_mask,
        cost_reference_tokens=8,
        max_inverse_cost=4.0,
    )

    torch.testing.assert_close(lengths, torch.tensor([1.0, 2.0, 8.0, 16.0]))
    torch.testing.assert_close(costs, torch.tensor([0.125, 0.25, 1.0, 2.0]))
    torch.testing.assert_close(inverse_costs, torch.tensor([4.0, 4.0, 1.0, 0.5]))
    torch.testing.assert_close(cap_mask, torch.tensor([True, False, False, False]))

    metrics = compute_cost_aware_maxrl_metrics(lengths, costs, cap_mask)
    assert metrics == {
        "cost_aware_maxrl/trajectory_tokens_mean": pytest.approx(6.75),
        "cost_aware_maxrl/trajectory_tokens_max": pytest.approx(16.0),
        "cost_aware_maxrl/trajectory_tokens_min": pytest.approx(1.0),
        "cost_aware_maxrl/cost_mean": pytest.approx(0.84375),
        "cost_aware_maxrl/cost_max": pytest.approx(2.0),
        "cost_aware_maxrl/cost_min": pytest.approx(0.125),
        "cost_aware_maxrl/cap_ratio": pytest.approx(0.25),
    }


def test_inverse_cost_cap_can_be_disabled():
    response_mask = make_response_mask([1, 16], width=16)

    _, _, inverse_costs, cap_mask = compute_cost_aware_maxrl_costs(
        response_mask=response_mask,
        cost_reference_tokens=8,
        max_inverse_cost=None,
    )

    torch.testing.assert_close(inverse_costs, torch.tensor([8.0, 0.5]))
    assert not cap_mask.any()


def test_advantage_uses_group_success_count_and_capped_inverse_cost():
    response_mask = make_response_mask([1, 2, 8, 16], width=16)
    token_rewards = make_token_rewards([1.0, 0.0, 1.0, 0.0], width=16)

    advantages, returns = compute_cost_aware_maxrl_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=np.array(["prompt"] * 4),
        cost_reference_tokens=8,
        max_inverse_cost=4.0,
    )

    # N=4 and K=2: A_i = 2 * r_i * capped_inverse_cost_i - 1.
    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask),
        torch.tensor([7.0, -1.0, 1.0, -1.0]),
    )
    torch.testing.assert_close(returns, advantages)
    assert torch.count_nonzero(advantages.masked_select(~response_mask)) == 0


def test_all_success_group_still_prefers_lower_cost_trajectory():
    response_mask = make_response_mask([2, 8], width=8)
    token_rewards = make_token_rewards([1.0, 1.0], width=8)

    advantages, _ = compute_cost_aware_maxrl_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=np.array(["prompt", "prompt"]),
        cost_reference_tokens=8,
        max_inverse_cost=4.0,
    )

    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask),
        torch.tensor([3.0, 0.0]),
    )


def test_advantage_is_computed_independently_for_each_prompt_group():
    response_mask = make_response_mask([2, 16, 4, 8, 16], width=16)
    token_rewards = make_token_rewards([1.0, 0.0, 1.0, 1.0, 0.0], width=16)

    advantages, _ = compute_cost_aware_maxrl_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=np.array(["a", "a", "b", "b", "b"]),
        config={"cost_reference_tokens": 8, "max_inverse_cost": 4.0},
    )

    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask),
        torch.tensor([7.0, -1.0, 2.0, 0.5, -1.0]),
    )


def test_zero_success_group_has_zero_advantage():
    response_mask = make_response_mask([1, 8], width=8)
    token_rewards = make_token_rewards([0.0, 0.0], width=8)

    advantages, returns = compute_cost_aware_maxrl_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=np.array(["prompt", "prompt"]),
        cost_reference_tokens=4,
        max_inverse_cost=4.0,
    )

    torch.testing.assert_close(advantages, torch.zeros_like(advantages))
    torch.testing.assert_close(returns, advantages)


def test_trainer_dispatch_records_metrics_for_each_step():
    from verl import DataProto
    from verl.trainer.ppo.core_algos import AdvantageEstimator
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
        adv_estimator=AdvantageEstimator.COST_AWARE_MAXRL,
        config={"cost_reference_tokens": 8, "max_inverse_cost": 4.0},
    )

    assert result.meta_info["cost_aware_maxrl_metrics"]["cost_aware_maxrl/cap_ratio"] == pytest.approx(0.25)
    torch.testing.assert_close(
        trajectory_advantages(result.batch["advantages"], response_mask),
        torch.tensor([7.0, -1.0, 1.0, -1.0]),
    )


@pytest.mark.parametrize("value", [0, -1, float("inf")])
def test_cost_reference_tokens_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match="cost_reference_tokens"):
        compute_cost_aware_maxrl_costs(
            response_mask=make_response_mask([1], width=1),
            cost_reference_tokens=value,
            max_inverse_cost=4.0,
        )
