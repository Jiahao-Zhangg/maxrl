"""Run with TailRL's text_maze directory first on PYTHONPATH."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from src.maze_binary_goal_cost_reward import compute_score

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage as rb
from verl.trainer.ppo.ray_trainer import compute_advantage

MAZE = ("GRID_START WALL WALL WALL WALL WALL NEWLINE "
        "WALL START PATH GOAL WALL NEWLINE "
        "WALL PATH PATH PATH WALL NEWLINE "
        "WALL WALL WALL WALL WALL NEWLINE "
        "WALL WALL WALL WALL WALL NEWLINE GRID_END PATH_START")


@pytest.mark.parametrize("solution,reward,length,cost,shortest", [
    ("RIGHT RIGHT DONE", 1, 2, 2, 1),
    ("DOWN RIGHT RIGHT UP DONE", 1, 4, 4, 0),
    ("UP DONE", 0, 1, 2, 0),
    ("RIGHT DONE", 0, 1, 2, 0),
    ("RIGHT RIGHT", 0, 2, 2, 0),
    ("RIGHT RIGHT LEFT LEFT DONE", 1, 4, 4, 1),
    ("RIGHT RIGHT DONE anything else", 1, 2, 2, 1),
    ("RIGHT INVALID DONE", 0, 2, 2, 0),
    ("DONE", 0, 0, 2, 0),
])
def test_reward_is_binary_and_cost_counts_generated_actions(solution, reward, length, cost, shortest):
    score = compute_score("maze_17", solution, MAZE)
    assert score["score"] == reward
    assert score["generated_action_length"] == length
    assert score["trajectory_cost"] == cost
    assert score["shortest_distance"] == 2
    assert score["is_shortest"] == shortest


def test_bad_shortest_metadata_fails_instead_of_changing_cost():
    with pytest.raises(ValueError, match="disagrees"):
        compute_score("maze_17", "RIGHT RIGHT DONE", MAZE, {"optimal_path_length": 3})


def test_adapter_keeps_interleaved_contexts_separate_and_all_failures_zero():
    # a: M=1, total cost=6, q=1/6; b: no successes, hence no update.
    batch = DataProto.from_dict(
        tensors={"response_mask": torch.ones(4, 2),
                 "token_level_rewards": torch.tensor([[0., 1.], [0., 0.], [0., 0.], [0., 0.]])},
        non_tensors={"uid": np.array(["a", "b", "a", "b"]),
                     "trajectory_cost": np.array([2., 10., 4., 20.]),
                     "generated_action_length": np.array([2., 10., 4., 20.]),
                     "shortest_distance": np.array([2., 5., 2., 5.])},
    )
    output = compute_advantage(batch, "fixed_n_rb_cost_aware_marginrl", num_repeat=2)
    torch.testing.assert_close(output.batch["advantages"][:, 0], torch.tensor([4/3, 0., -2/3, 0.]))
    assert output.meta_info["maze_rb_metrics"]["rb/zero_success_group_fraction"] == 0.5


def test_port_has_identical_function_body_to_existing_maxrl():
    source = Path(__file__).resolve().parents[2] / "verl/trainer/ppo/core_algos.py"
    text = source.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == rb.__name__)
    expected = ast.dump(ast.parse(ast.get_source_segment(text, node)), include_attributes=False)
    actual_node = ast.parse(inspect.getsource(rb)).body[0]
    actual_node.decorator_list = []
    actual = ast.dump(ast.Module(body=[actual_node], type_ignores=[]), include_attributes=False)
    assert actual == expected


def test_per_context_cost_rescaling_leaves_same_group_estimator_unchanged():
    rewards = torch.tensor([[1.], [0.], [1.], [0.]])
    mask = torch.ones_like(rewards)
    groups = np.array(["a", "a", "b", "b"])
    costs = torch.tensor([2., 3., 8., 12.])
    original, _ = rb(rewards, mask, groups, trajectory_costs=costs)
    scaled, _ = rb(rewards, mask, groups, trajectory_costs=costs * torch.tensor([3., 3., 7., 7.]))
    torch.testing.assert_close(original, scaled)
