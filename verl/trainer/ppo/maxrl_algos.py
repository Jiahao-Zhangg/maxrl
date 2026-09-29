"""Original MaxRL and fixed-N cost variants preserved from the pre-0.9.1 trainer.

The controller computes these weights once, on the complete rollout batch, before
GPU sharding. Cost always counts the full response, including thinking tokens.
"""

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

MAXRL_ESTIMATORS = frozenset({
    "maxrl", "fixed_n_rb_cost_aware_marginrl",
    "fixed_n_rb_offset_cost_aware_marginrl", "f_cov",
})


def validate_maxrl_training_config(config):
    """Fail early on modes that change the migrated fixed-batch estimators."""
    algorithm = config.algorithm
    if algorithm.adv_estimator not in MAXRL_ESTIMATORS:
        return
    rollout = config.actor_rollout_ref.rollout
    if rollout.n < 2:
        raise ValueError("MaxRL training requires at least two responses per prompt")
    if algorithm.adv_estimator != "maxrl":
        offset = float(algorithm.get("cost_offset_tokens", 256.0))
        if not math.isfinite(offset) or offset < 0:
            raise ValueError("cost_offset_tokens must be finite and nonnegative")
    if algorithm.use_kl_in_reward or algorithm.get("use_pf_ppo", False):
        raise ValueError("MaxRL variants require unmodified outcome rewards")
    correction = algorithm.get("rollout_correction") or {}
    if correction.get("rollout_rs"):
        raise ValueError("MaxRL cost variants do not support rejection-modified response masks")
    filtering = algorithm.get("filter_groups") or {}
    if filtering.get("enable", False):
        raise ValueError("MaxRL variants require the original unfiltered prompt batch")
    v1 = config.trainer.v1
    mode = v1.trainer_mode
    if mode != "sync" or v1.get(mode, {}).get("parameter_sync_step", 1) != 1:
        raise ValueError("Migrated MaxRL variants require synchronous full-batch training")
    if algorithm.adv_estimator == "f_cov":
        expected = algorithm.get("f_cov_num_prompts")
        if expected is not None and expected != config.data.train_batch_size:
            raise ValueError("f_cov_num_prompts must match the full data.train_batch_size")
        algorithm.f_cov_num_prompts = config.data.train_batch_size


@torch.no_grad()
def compute_maxrl_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    expected_group_size: Optional[int] = None,
    **kwargs,
):
    """Original (reward - group mean) / (group mean + epsilon), without whitening.

    Keep the legacy singleton behavior for direct callers. Training supplies N
    and rejects incomplete groups instead of accidentally applying that fallback.
    ``norm_adv_by_std_in_grpo`` is intentionally ignored, as in original MaxRL.
    """
    if response_mask.ndim != 2 or token_level_rewards.shape != response_mask.shape:
        raise ValueError("MaxRL rewards and response mask must have matching rank-2 shapes")
    if index is None or len(index) != response_mask.shape[0] or len(index) == 0:
        raise ValueError("MaxRL requires one prompt UID per response")
    scores = token_level_rewards.sum(dim=-1)
    if not torch.isfinite(scores).all():
        raise ValueError("MaxRL requires finite trajectory rewards")
    groups = defaultdict(list)
    for row, uid in enumerate(index):
        groups[uid].append(row)
    weights = torch.empty_like(scores)
    for positions in groups.values():
        if expected_group_size is not None and len(positions) != expected_group_size:
            raise ValueError(f"MaxRL expected {expected_group_size} responses per prompt, found {len(positions)}")
        rewards = scores[positions]
        mean = rewards.mean() if len(positions) > 1 else scores.new_tensor(0.0)
        weights[positions] = (rewards - mean) / (mean + epsilon)
    advantages = weights.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_fixed_n_rb_offset_marginrl_costs(
    response_mask: torch.Tensor,
    cost_offset_tokens: float = 256.0,
):
    """Compute fixed-N trajectory costs as response length plus a token offset."""
    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must be rank 2, got shape {tuple(response_mask.shape)}")

    cost_offset_tokens = float(cost_offset_tokens)
    if not math.isfinite(cost_offset_tokens) or cost_offset_tokens < 0:
        raise ValueError(f"cost_offset_tokens must be finite and nonnegative, got {cost_offset_tokens}")

    trajectory_lengths = response_mask.sum(dim=-1).detach().to(dtype=torch.float32)
    if not torch.isfinite(trajectory_lengths).all().item() or torch.any(trajectory_lengths <= 0).item():
        raise ValueError("fixed-N trajectory lengths must be finite and strictly positive")

    return trajectory_lengths, trajectory_lengths + cost_offset_tokens


def compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_cost_mask: Optional[torch.Tensor] = None,
    trajectory_costs: Optional[torch.Tensor] = None,
    trajectory_lengths: Optional[torch.Tensor] = None,
    inverse_cost_cap_mask: Optional[torch.Tensor] = None,
    expected_group_size: Optional[int] = None,
    success_gated: bool = False,
    fixed_q_hat: Optional[float] = None,
    return_diagnostics: bool = False,
    **kwargs,
):
    """Compute fixed-rollout Rao--Blackwellized cost-aware MarginRL advantages.

    For each prompt group, let ``M`` be its number of successful trajectories,
    ``c_i`` its response-token costs, and ``q_hat = M / sum_i c_i`` the detached
    same-group rate estimate. The raw trajectory coefficient is

    ``(1 - q_hat * c_i) / M`` for a success and
    ``-q_hat * c_i / (M + 1)`` for a failure. When ``success_gated`` is
    enabled, the failure coefficient is replaced by zero while the success
    coefficient and same-group ``q_hat`` remain unchanged. Supplying
    ``fixed_q_hat`` replaces the same-group plug-in estimate by that detached
    constant; this retains a cost update for all-failure groups.

    Raw coefficients are multiplied by the fixed group size before being
    broadcast over response tokens. With ``seq-mean-token-sum`` this directly
    realizes the prompt-averaged group-sum estimator. The experiment launchers
    default to ``token-mean`` to match original MaxRL; within each optimizer
    microbatch that preserves the full trajectory scores and divides the
    update by the microbatch's mean response length. An all-failure group has
    ``q_hat == 0`` and receives zero advantage.
    """
    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must be rank 2, got shape {tuple(response_mask.shape)}")
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(
            "token_level_rewards and response_mask must have matching shapes, "
            f"got {tuple(token_level_rewards.shape)} and {tuple(response_mask.shape)}"
        )
    if trajectory_costs is None:
        if trajectory_cost_mask is None:
            trajectory_cost_mask = response_mask
        if trajectory_cost_mask.shape != response_mask.shape:
            raise ValueError(
                "trajectory_cost_mask and response_mask must have matching shapes, "
                f"got {tuple(trajectory_cost_mask.shape)} and {tuple(response_mask.shape)}"
            )
        trajectory_costs = trajectory_cost_mask.sum(dim=-1).detach().to(dtype=torch.float32)
        if trajectory_lengths is None:
            trajectory_lengths = trajectory_costs
    else:
        trajectory_costs = trajectory_costs.detach().to(device=response_mask.device, dtype=torch.float32)
        if trajectory_costs.ndim != 1 or trajectory_costs.shape[0] != response_mask.shape[0]:
            raise ValueError(
                "trajectory_costs must have shape (batch_size,), "
                f"got {tuple(trajectory_costs.shape)}"
            )
        if trajectory_lengths is None:
            trajectory_lengths = response_mask.sum(dim=-1)

    trajectory_lengths = trajectory_lengths.detach().to(device=response_mask.device, dtype=torch.float32)
    if trajectory_lengths.ndim != 1 or trajectory_lengths.shape != trajectory_costs.shape:
        raise ValueError(
            "trajectory_lengths and trajectory_costs must have matching one-dimensional shapes, "
            f"got {tuple(trajectory_lengths.shape)} and {tuple(trajectory_costs.shape)}"
        )
    if inverse_cost_cap_mask is None:
        inverse_cost_cap_mask = torch.zeros_like(trajectory_costs, dtype=torch.bool)
    else:
        inverse_cost_cap_mask = inverse_cost_cap_mask.detach().to(device=response_mask.device, dtype=torch.bool)
        if inverse_cost_cap_mask.shape != trajectory_costs.shape:
            raise ValueError(
                "inverse_cost_cap_mask and trajectory_costs must have matching shapes, "
                f"got {tuple(inverse_cost_cap_mask.shape)} and {tuple(trajectory_costs.shape)}"
            )
    if index is None:
        raise ValueError("index is required for fixed_n_rb_cost_aware_marginrl prompt grouping")
    if len(index) != token_level_rewards.shape[0]:
        raise ValueError(
            f"index length {len(index)} does not match batch size {token_level_rewards.shape[0]}"
        )
    if expected_group_size is not None:
        expected_group_size = int(expected_group_size)
        if expected_group_size <= 0:
            raise ValueError(f"expected_group_size must be positive, got {expected_group_size}")
    if fixed_q_hat is not None:
        fixed_q_hat = float(fixed_q_hat)
        if not math.isfinite(fixed_q_hat) or fixed_q_hat < 0:
            raise ValueError(f"fixed_q_hat must be finite and nonnegative, got {fixed_q_hat}")

    scores = token_level_rewards.sum(dim=-1).detach().to(dtype=torch.float32)
    if not torch.isfinite(trajectory_costs).all().item() or torch.any(trajectory_costs <= 0).item():
        raise ValueError("fixed-N trajectory costs must be finite and strictly positive")

    is_zero = torch.isclose(scores, torch.zeros_like(scores), rtol=0.0, atol=1e-6)
    is_one = torch.isclose(scores, torch.ones_like(scores), rtol=0.0, atol=1e-6)
    if not torch.all(is_zero | is_one).item():
        invalid_scores = scores[~(is_zero | is_one)][:8].cpu().tolist()
        raise ValueError(
            "fixed_n_rb_cost_aware_marginrl requires binary trajectory rewards; "
            f"found values such as {invalid_scores}"
        )
    binary_rewards = is_one.to(dtype=torch.float32)

    with torch.no_grad():
        id2positions = defaultdict(list)
        for position, prompt_id in enumerate(index):
            id2positions[prompt_id].append(position)
        if not id2positions:
            raise ValueError("fixed_n_rb_cost_aware_marginrl requires at least one prompt group")

        observed_group_sizes = {len(positions) for positions in id2positions.values()}
        if len(observed_group_sizes) != 1:
            raise ValueError(
                "fixed_n_rb_cost_aware_marginrl requires one fixed rollout count per prompt; "
                f"found group sizes {sorted(observed_group_sizes)}"
            )
        observed_group_size = next(iter(observed_group_sizes))
        if expected_group_size is not None and observed_group_size != expected_group_size:
            raise ValueError(
                "fixed_n_rb_cost_aware_marginrl rollout count mismatch: "
                f"expected {expected_group_size}, found {observed_group_size}"
            )

        raw_trajectory_advantages = torch.zeros_like(scores)
        optimizer_trajectory_advantages = torch.zeros_like(scores)
        group_q_hats = []
        group_success_counts = []
        group_total_costs = []
        group_cost_means = []
        group_cost_stds = []
        group_accuracies = []

        for positions in id2positions.values():
            position_tensor = torch.as_tensor(positions, dtype=torch.long, device=scores.device)
            group_rewards = binary_rewards.index_select(0, position_tensor)
            group_costs = trajectory_costs.index_select(0, position_tensor)
            group_size = len(positions)
            success_count = group_rewards.sum()
            total_cost = group_costs.sum()
            if fixed_q_hat is None:
                q_hat = (success_count / total_cost).detach()
            else:
                q_hat = success_count.new_tensor(fixed_q_hat).detach()
            q_cost = q_hat * group_costs

            # torch.where evaluates both branches, so use a safe denominator
            # even though the success branch is inactive when M == 0.
            safe_success_count = success_count.clamp_min(1.0)
            success_advantages = (1.0 - q_cost) / safe_success_count
            if success_gated:
                failure_advantages = torch.zeros_like(q_cost)
            else:
                failure_advantages = -q_cost / (success_count + 1.0)
            group_raw_advantages = torch.where(
                group_rewards.to(dtype=torch.bool),
                success_advantages,
                failure_advantages,
            )
            group_optimizer_advantages = group_raw_advantages * group_size

            raw_trajectory_advantages.index_copy_(0, position_tensor, group_raw_advantages)
            optimizer_trajectory_advantages.index_copy_(0, position_tensor, group_optimizer_advantages)
            group_q_hats.append(q_hat)
            group_success_counts.append(success_count)
            group_total_costs.append(total_cost)
            group_cost_means.append(group_costs.mean())
            group_cost_stds.append(group_costs.std(unbiased=False))
            group_accuracies.append(success_count / group_size)

        advantages = optimizer_trajectory_advantages.unsqueeze(-1) * response_mask
        diagnostics = {
            "trajectory_lengths": trajectory_lengths,
            "trajectory_costs": trajectory_costs,
            "inverse_cost_cap_mask": inverse_cost_cap_mask,
            "trajectory_rewards": binary_rewards,
            "raw_trajectory_advantages": raw_trajectory_advantages,
            "optimizer_trajectory_advantages": optimizer_trajectory_advantages,
            "group_q_hats": torch.stack(group_q_hats),
            "group_success_counts": torch.stack(group_success_counts),
            "group_total_costs": torch.stack(group_total_costs),
            "group_cost_means": torch.stack(group_cost_means),
            "group_cost_stds": torch.stack(group_cost_stds),
            "group_accuracies": torch.stack(group_accuracies),
        }

    if return_diagnostics:
        return advantages, advantages, diagnostics
    return advantages, advantages


def compute_fixed_n_rb_offset_cost_aware_marginrl_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_cost_mask: Optional[torch.Tensor] = None,
    expected_group_size: Optional[int] = None,
    config=None,
    cost_offset_tokens: float = 256.0,
    return_diagnostics: bool = False,
    **kwargs,
):
    """Compute fixed-N RB MarginRL with ``c_i = L_i + L_0``.

    Use all N rollouts in the same-group estimate ``q_hat = M / sum_i c_i``.
    After the existing N-fold scaling, successful trajectories receive
    ``1 / p_hat - (L_i + L_0) / (mean_L + L_0)`` and failures receive
    ``-M / (M + 1) * (L_i + L_0) / (mean_L + L_0)``, where ``p_hat = M / N``.
    All-failure groups receive zero advantage. Costs use the full response
    length even when a separate response_mask selects tokens for the loss.
    """
    if config is not None:
        cost_offset_tokens = config.get("cost_offset_tokens", cost_offset_tokens)

    cost_mask = response_mask if trajectory_cost_mask is None else trajectory_cost_mask
    if cost_mask.shape != response_mask.shape:
        raise ValueError(
            "trajectory_cost_mask and response_mask must have matching shapes, "
            f"got {tuple(cost_mask.shape)} and {tuple(response_mask.shape)}"
        )
    trajectory_lengths, trajectory_costs = compute_fixed_n_rb_offset_marginrl_costs(
        response_mask=cost_mask,
        cost_offset_tokens=cost_offset_tokens,
    )
    return compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        trajectory_costs=trajectory_costs,
        trajectory_lengths=trajectory_lengths,
        expected_group_size=expected_group_size,
        return_diagnostics=return_diagnostics,
        **kwargs,
    )


@torch.no_grad()
def compute_cross_context_f_cov_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_cost_mask: Optional[torch.Tensor] = None,
    expected_group_size: Optional[int] = None,
    expected_num_prompts: Optional[int] = None,
    config=None,
    cost_offset_tokens: float = 256.0,
    return_diagnostics: bool = False,
    **kwargs,
):
    """Compute final cross-context plug-in advantages on the full rollout batch.

    For K prompts with N responses each, let M_k count correct responses,
    c_ki = L_ki + L_0, C = mean(c_ki), and H = mean_k(M_k / (M_k + 1)).
    Correct responses receive N / M_k - c_ki / C * (H + 1 / (K * (M_k + 1)));
    incorrect responses receive -c_ki / C * H, including all-failure groups.
    These are already optimizer advantages: do not multiply by N or K again,
    whiten them, or recompute their global statistics inside GPU microbatches.
    Costs use the full response mask even when the loss selects fewer tokens.
    """
    if response_mask.ndim != 2 or token_level_rewards.shape != response_mask.shape:
        raise ValueError("f_cov rewards and response mask must have matching rank-2 shapes")
    if index is None or len(index) != response_mask.shape[0]:
        raise ValueError("f_cov requires one prompt UID per response")
    if config is not None:
        cost_offset_tokens = config.get("cost_offset_tokens", cost_offset_tokens)
        expected_num_prompts = config.get("f_cov_num_prompts", expected_num_prompts)
    cost_mask = response_mask if trajectory_cost_mask is None else trajectory_cost_mask
    if cost_mask.shape != response_mask.shape:
        raise ValueError("f_cov cost and loss masks must have matching shapes")
    lengths, costs = compute_fixed_n_rb_offset_marginrl_costs(cost_mask, cost_offset_tokens)
    scores = token_level_rewards.sum(dim=-1).detach().to(device=costs.device, dtype=torch.float32)
    is_zero = torch.isclose(scores, torch.zeros_like(scores), rtol=0.0, atol=1e-6)
    is_one = torch.isclose(scores, torch.ones_like(scores), rtol=0.0, atol=1e-6)
    if not torch.all(is_zero | is_one).item():
        raise ValueError("f_cov requires binary trajectory rewards")
    rewards = is_one.to(dtype=torch.float32)

    groups = defaultdict(list)
    for position, prompt_uid in enumerate(index):
        groups[prompt_uid].append(position)
    if not groups:
        raise ValueError("f_cov requires at least one prompt group")
    group_sizes = {len(positions) for positions in groups.values()}
    if len(group_sizes) != 1:
        raise ValueError("f_cov requires the same number of responses for every prompt")
    group_size = next(iter(group_sizes))
    num_prompts = len(groups)
    if expected_group_size is not None and group_size != int(expected_group_size):
        raise ValueError(f"f_cov expected {expected_group_size} responses per prompt, found {group_size}")
    if expected_num_prompts is not None and num_prompts != int(expected_num_prompts):
        raise ValueError(
            f"f_cov expected {expected_num_prompts} prompts in the full rollout batch, found {num_prompts}"
        )

    row_groups = torch.empty(len(index), dtype=torch.long, device=costs.device)
    for group_id, positions in enumerate(groups.values()):
        row_groups[positions] = group_id
    success_counts = torch.zeros(num_prompts, dtype=torch.float32, device=costs.device)
    success_counts.scatter_add_(0, row_groups, rewards)
    global_cost_mean = costs.mean()
    cross_context_h = (success_counts / (success_counts + 1.0)).mean()
    success_correction = 1.0 / (num_prompts * (success_counts + 1.0))
    row_success_counts = success_counts[row_groups]
    cost_ratio = costs / global_cost_mean
    trajectory_advantages = (
        rewards * group_size / row_success_counts.clamp_min(1.0)
        - cost_ratio * (cross_context_h + rewards * success_correction[row_groups])
    )
    advantages = trajectory_advantages.unsqueeze(-1) * response_mask
    if return_diagnostics:
        return advantages, advantages, {
            "trajectory_lengths": lengths,
            "trajectory_costs": costs,
            "trajectory_rewards": rewards,
            "optimizer_trajectory_advantages": trajectory_advantages,
            "group_success_counts": success_counts,
            "global_cost_mean": global_cost_mean,
            "cross_context_h": cross_context_h,
            "success_correction": success_correction,
            "group_size": group_size,
        }
    return advantages, advantages
