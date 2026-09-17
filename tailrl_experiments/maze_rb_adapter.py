"""Connect the unchanged MaxRL per-context fixed-N RB estimator to TailRL."""

import json
import os
from pathlib import Path

import numpy as np
import torch


def apply_maze_rb_advantage(data, num_repeat):
    from verl.trainer.ppo.core_algos import compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage

    required = ("uid", "trajectory_cost", "generated_action_length", "shortest_distance")
    missing = set(required) - data.non_tensor_batch.keys()
    if missing:
        raise ValueError(f"Missing per-trajectory maze fields: {sorted(missing)}")
    device = data.batch["response_mask"].device
    costs = torch.as_tensor(np.asarray(data.non_tensor_batch["trajectory_cost"], dtype=float), device=device)
    lengths = torch.as_tensor(np.asarray(data.non_tensor_batch["generated_action_length"], dtype=float), device=device)
    distances = torch.as_tensor(np.asarray(data.non_tensor_batch["shortest_distance"], dtype=float), device=device)
    if not torch.equal(costs, torch.maximum(lengths, distances)):
        raise ValueError("Maze costs must equal max(L, L*) on every trajectory")
    advantages, returns, diag = compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch["uid"],
        trajectory_costs=costs,
        trajectory_lengths=lengths,
        inverse_cost_cap_mask=lengths < distances,
        expected_group_size=num_repeat,
        return_diagnostics=True,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    counts = diag["group_success_counts"]
    data.meta_info["maze_rb_metrics"] = {
        "rb/group_count": counts.numel(),
        "rb/zero_success_group_fraction": (counts == 0).float().mean().item(),
        "rb/single_success_group_fraction": (counts == 1).float().mean().item(),
        "rb/success_count_mean": counts.mean().item(),
        "rb/q_hat_mean": diag["group_q_hats"].mean().item(),
        "rb/cost_mean": costs.mean().item(),
        "rb/cost_floor_fraction": (lengths < distances).float().mean().item(),
        "rb/advantage_abs_max": advantages.abs().max().item(),
        "rb/advantage_abs_mean": diag["optimizer_trajectory_advantages"].abs().mean().item(),
        "rb/successes": counts.sum().item(),
        "rb/rollouts": costs.numel(),
    }
    if "is_shortest" in data.non_tensor_batch:
        shortest = torch.as_tensor(np.asarray(data.non_tensor_batch["is_shortest"], dtype=bool), device=device)
        trajectory_advantages = diag["optimizer_trajectory_advantages"]
        data.meta_info["maze_rb_metrics"].update({
            "rb/shortest_successes": shortest.sum().item(),
            "rb/shortest_positive_advantages": (shortest & (trajectory_advantages > 0)).sum().item(),
        })
    return data


def append_metrics(data, step):
    """Save full-precision metrics locally, without an external tracking service."""
    destination = os.environ.get("TAILRL_RB_METRICS_FILE")
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps({"step": int(step), "metrics": data}, default=lambda x: x.item()) + "\n")
