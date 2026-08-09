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

from itertools import product

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_rao_blackwellized_betas,
    compute_rb_cost_aware_maxrl_outcome_advantage,
    compute_rb_cost_probabilities,
)
from verl.trainer.ppo.metric_utils import compute_rb_cost_aware_maxrl_metrics


def make_response_mask(lengths: list[int], width: int) -> torch.Tensor:
    return torch.arange(width).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)


def make_token_rewards(rewards: list[float], width: int) -> torch.Tensor:
    token_rewards = torch.zeros((len(rewards), width), dtype=torch.float32)
    token_rewards[:, 0] = torch.tensor(rewards)
    return token_rewards


def trajectory_advantages(advantages: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    return (advantages * response_mask).sum(dim=-1) / response_mask.sum(dim=-1)


def brute_force_betas(cost_probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = cost_probabilities.tolist()
    betas = []
    for excluded_position, excluded_probability in enumerate(probabilities):
        other_probabilities = [probability for position, probability in enumerate(probabilities) if position != excluded_position]
        expected_reciprocal_count = 0.0
        for outcomes in product((0, 1), repeat=len(other_probabilities)):
            outcome_probability = 1.0
            for outcome, probability in zip(outcomes, other_probabilities):
                outcome_probability *= probability if outcome else 1.0 - probability
            expected_reciprocal_count += outcome_probability / (1 + sum(outcomes))
        betas.append(excluded_probability * expected_reciprocal_count)
    return torch.tensor(betas, dtype=cost_probabilities.dtype)


@pytest.mark.parametrize(
    "cost_probabilities",
    [
        torch.tensor([0.2, 0.5, 0.8]),
        torch.tensor([0.0, 1.0, 0.25, 0.75]),
        torch.tensor([0.4]),
    ],
)
def test_beta_dp_matches_exact_enumeration(cost_probabilities):
    actual = compute_rao_blackwellized_betas(cost_probabilities)
    expected = brute_force_betas(cost_probabilities)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    expected_sum = 1.0 - torch.prod(1.0 - cost_probabilities)
    torch.testing.assert_close(actual.sum(), expected_sum, atol=1e-6, rtol=1e-6)


def test_cost_probabilities_use_maximum_response_length():
    lengths, cost_probabilities = compute_rb_cost_probabilities(
        response_mask=make_response_mask([1, 2, 4, 8], width=8),
        cost_max_tokens=8,
    )

    torch.testing.assert_close(lengths, torch.tensor([1.0, 2.0, 4.0, 8.0]))
    torch.testing.assert_close(cost_probabilities, torch.tensor([0.125, 0.25, 0.5, 1.0]))


def test_successful_group_uses_no_additional_minus_one_control_variate():
    response_mask = make_response_mask([1, 4], width=4)
    token_rewards = make_token_rewards([1.0, 0.0], width=4)

    advantages, returns, diagnostics = compute_rb_cost_aware_maxrl_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=np.array(["prompt", "prompt"]),
        cost_max_tokens=4,
        return_diagnostics=True,
    )

    # kappa=[0.25, 1], beta=[0.125, 0.875], N=2, K=1.
    torch.testing.assert_close(diagnostics["betas"], torch.tensor([0.125, 0.875]))
    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask),
        torch.tensor([1.75, -1.75]),
    )
    torch.testing.assert_close(returns, advantages)
    assert torch.count_nonzero(advantages.masked_select(~response_mask)) == 0


def test_zero_success_group_is_fully_gated_off():
    response_mask = make_response_mask([1, 4], width=4)
    token_rewards = make_token_rewards([0.0, 0.0], width=4)

    advantages, returns, diagnostics = compute_rb_cost_aware_maxrl_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=np.array(["prompt", "prompt"]),
        cost_max_tokens=4,
        return_diagnostics=True,
    )

    torch.testing.assert_close(advantages, torch.zeros_like(advantages))
    torch.testing.assert_close(returns, advantages)
    torch.testing.assert_close(diagnostics["betas"], torch.tensor([0.125, 0.875]))
    torch.testing.assert_close(diagnostics["zero_success_groups"], torch.tensor([True]))


def test_prompt_groups_use_independent_beta_dps_and_success_gates():
    response_mask = make_response_mask([1, 4, 1, 2, 4], width=4)
    rewards = [1.0, 0.0, 0.0, 0.0, 0.0]

    advantages, _, diagnostics = compute_rb_cost_aware_maxrl_outcome_advantage(
        token_level_rewards=make_token_rewards(rewards, width=4),
        response_mask=response_mask,
        index=np.array(["a", "a", "b", "b", "b"]),
        config={"rb_cost_max_tokens": 4},
        return_diagnostics=True,
    )

    torch.testing.assert_close(
        trajectory_advantages(advantages, response_mask),
        torch.tensor([1.75, -1.75, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(diagnostics["zero_success_groups"], torch.tensor([False, True]))


def test_trainer_dispatch_records_rb_metrics_for_each_step():
    from verl import DataProto
    from verl.trainer.ppo.core_algos import AdvantageEstimator
    from verl.trainer.ppo.ray_trainer import compute_advantage

    response_mask = make_response_mask([1, 4, 1, 2], width=4)
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": make_token_rewards([1.0, 0.0, 0.0, 0.0], width=4),
            "response_mask": response_mask,
        },
        non_tensors={"uid": np.array(["a", "a", "b", "b"])},
    )

    result = compute_advantage(
        data=data,
        adv_estimator=AdvantageEstimator.RB_COST_AWARE_MAXRL,
        config={"rb_cost_max_tokens": 4},
    )

    metrics = result.meta_info["rb_cost_aware_maxrl_metrics"]
    assert metrics["rb_cost_aware_maxrl/trajectory_tokens_mean"] == pytest.approx(2.0)
    assert metrics["rb_cost_aware_maxrl/kappa_mean"] == pytest.approx(0.5)
    assert metrics["rb_cost_aware_maxrl/zero_success_group_ratio"] == pytest.approx(0.5)


def test_rb_metrics_validate_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        compute_rb_cost_aware_maxrl_metrics(
            trajectory_lengths=torch.ones(2),
            cost_probabilities=torch.ones(1),
            betas=torch.ones(2),
            zero_success_groups=torch.zeros(1),
        )


@pytest.mark.parametrize("value", [0, -1, float("inf")])
def test_cost_max_tokens_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match="cost_max_tokens"):
        compute_rb_cost_probabilities(
            response_mask=make_response_mask([1], width=1),
            cost_max_tokens=value,
        )
