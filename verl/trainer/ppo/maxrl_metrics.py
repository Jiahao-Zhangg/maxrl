"""Detached diagnostics for the migrated MaxRL cost variants."""

import torch


def compute_fixed_n_rb_cost_aware_marginrl_metrics(
    trajectory_lengths: torch.Tensor,
    trajectory_costs: torch.Tensor,
    inverse_cost_cap_mask: torch.Tensor,
    trajectory_rewards: torch.Tensor,
    raw_trajectory_advantages: torch.Tensor,
    optimizer_trajectory_advantages: torch.Tensor,
    group_q_hats: torch.Tensor,
    group_success_counts: torch.Tensor,
    group_total_costs: torch.Tensor,
    group_cost_means: torch.Tensor,
    group_cost_stds: torch.Tensor,
    group_accuracies: torch.Tensor,
    metric_prefix: str = "fixed_n_rb_marginrl",
) -> dict[str, float]:
    """Summarize fixed-N RB cost-aware MarginRL statistics for W&B."""
    if not metric_prefix:
        raise ValueError("fixed-N RB MarginRL metric prefix must be nonempty")
    trajectory_tensors = (
        trajectory_lengths,
        trajectory_costs,
        inverse_cost_cap_mask,
        trajectory_rewards,
        raw_trajectory_advantages,
        optimizer_trajectory_advantages,
    )
    group_tensors = (
        group_q_hats,
        group_success_counts,
        group_total_costs,
        group_cost_means,
        group_cost_stds,
        group_accuracies,
    )
    if any(tensor.numel() == 0 for tensor in (*trajectory_tensors, *group_tensors)):
        raise ValueError("fixed-N RB MarginRL metrics require trajectories and prompt groups")
    if any(tensor.shape != trajectory_costs.shape for tensor in trajectory_tensors):
        raise ValueError("fixed-N RB MarginRL trajectory diagnostics must have matching shapes")
    if any(tensor.shape != group_q_hats.shape for tensor in group_tensors[1:]):
        raise ValueError("fixed-N RB MarginRL group diagnostics must have matching shapes")

    lengths = trajectory_lengths.detach().float()
    costs = trajectory_costs.detach().float()
    cap_mask = inverse_cost_cap_mask.detach().float()
    rewards = trajectory_rewards.detach().float()
    raw_advantages = raw_trajectory_advantages.detach().float()
    optimizer_advantages = optimizer_trajectory_advantages.detach().float()
    q_hats = group_q_hats.detach().float()
    success_counts = group_success_counts.detach().float()
    total_costs = group_total_costs.detach().float()
    cost_means = group_cost_means.detach().float()
    cost_stds = group_cost_stds.detach().float()
    accuracies = group_accuracies.detach().float()

    def population_std(values: torch.Tensor) -> float:
        return values.std(unbiased=False).item()

    failure_mask = rewards == 0
    failure_advantage_abs_max = (
        raw_advantages[failure_mask].abs().max().item()
        if failure_mask.any().item()
        else 0.0
    )

    return {
        # Applicable trajectory-length metrics retained from prior variants.
        f"{metric_prefix}/trajectory_tokens_mean": lengths.mean().item(),
        f"{metric_prefix}/trajectory_tokens_max": lengths.max().item(),
        f"{metric_prefix}/trajectory_tokens_min": lengths.min().item(),
        # Requested detached rate and sufficient statistics, summarized over prompts.
        f"{metric_prefix}/q_hat_mean": q_hats.mean().item(),
        f"{metric_prefix}/q_hat_std": population_std(q_hats),
        f"{metric_prefix}/q_hat_max": q_hats.max().item(),
        f"{metric_prefix}/q_hat_min": q_hats.min().item(),
        f"{metric_prefix}/M_t_mean": success_counts.mean().item(),
        f"{metric_prefix}/M_t_std": population_std(success_counts),
        f"{metric_prefix}/M_t_max": success_counts.max().item(),
        f"{metric_prefix}/M_t_min": success_counts.min().item(),
        f"{metric_prefix}/total_cost_mean": total_costs.mean().item(),
        f"{metric_prefix}/total_cost_std": population_std(total_costs),
        f"{metric_prefix}/total_cost_max": total_costs.max().item(),
        f"{metric_prefix}/total_cost_min": total_costs.min().item(),
        f"{metric_prefix}/cost_mean": costs.mean().item(),
        f"{metric_prefix}/cost_std": population_std(costs),
        f"{metric_prefix}/cost_max": costs.max().item(),
        f"{metric_prefix}/cost_min": costs.min().item(),
        f"{metric_prefix}/cap_ratio": cap_mask.mean().item(),
        f"{metric_prefix}/group_cost_mean_std": population_std(cost_means),
        f"{metric_prefix}/group_cost_std_mean": cost_stds.mean().item(),
        f"{metric_prefix}/group_cost_std_std": population_std(cost_stds),
        f"{metric_prefix}/accuracy_mean": accuracies.mean().item(),
        f"{metric_prefix}/accuracy_std": population_std(accuracies),
        f"{metric_prefix}/accuracy_max": accuracies.max().item(),
        f"{metric_prefix}/accuracy_min": accuracies.min().item(),
        f"{metric_prefix}/zero_success_group_ratio": (success_counts == 0).float().mean().item(),
        f"{metric_prefix}/failure_advantage_abs_max": failure_advantage_abs_max,
        f"{metric_prefix}/raw_advantage_mean": raw_advantages.mean().item(),
        f"{metric_prefix}/raw_advantage_std": population_std(raw_advantages),
        f"{metric_prefix}/raw_advantage_max": raw_advantages.max().item(),
        f"{metric_prefix}/raw_advantage_min": raw_advantages.min().item(),
        f"{metric_prefix}/optimizer_advantage_mean": optimizer_advantages.mean().item(),
        f"{metric_prefix}/optimizer_advantage_std": population_std(optimizer_advantages),
        f"{metric_prefix}/optimizer_advantage_max": optimizer_advantages.max().item(),
        f"{metric_prefix}/optimizer_advantage_min": optimizer_advantages.min().item(),
    }


def compute_cross_context_f_cov_metrics(
    trajectory_lengths,
    trajectory_costs,
    trajectory_rewards,
    optimizer_trajectory_advantages,
    group_success_counts,
    global_cost_mean,
    cross_context_h,
    success_correction,
    group_size,
) -> dict[str, float]:
    """Report the global coupling statistics and final optimizer weights."""
    metrics = {
        "f_cov/H": cross_context_h.item(),
        "f_cov/global_cost_mean": global_cost_mean.item(),
        "f_cov/num_prompts": float(group_success_counts.numel()),
        "f_cov/responses_per_prompt": float(group_size),
        "f_cov/accuracy": trajectory_rewards.mean().item(),
        "f_cov/zero_success_group_ratio": (group_success_counts == 0).float().mean().item(),
        "f_cov/success_correction_mean": success_correction.mean().item(),
    }
    for name, values in (
        ("trajectory_tokens", trajectory_lengths),
        ("cost", trajectory_costs),
        ("M", group_success_counts),
        ("optimizer_advantage", optimizer_trajectory_advantages),
    ):
        values = values.detach().float()
        metrics.update({
            f"f_cov/{name}_mean": values.mean().item(),
            f"f_cov/{name}_std": values.std(unbiased=False).item(),
            f"f_cov/{name}_min": values.min().item(),
            f"f_cov/{name}_max": values.max().item(),
        })
    return metrics
