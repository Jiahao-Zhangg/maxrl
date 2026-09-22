"""Binary goal reward and a separate max(generated actions, shortest distance) cost."""

from functools import lru_cache

from verl.utils.reward_score.maze import MazeEnv, _bfs_optimal_length, judge_maze


@lru_cache(maxsize=32768)
def shortest_distance(ground_truth):
    env = MazeEnv.from_sequence(ground_truth)
    if env is None:
        raise ValueError("Cannot parse the reference maze")
    distance = _bfs_optimal_length(env.grid, env.start, env.goal)
    if distance is None or distance <= 0:
        raise ValueError("Maze must have a reachable, distinct goal")
    return distance


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if not data_source.startswith("maze"):
        raise ValueError(f"Unsupported data source: {data_source}")
    # Preserve the original binary judge, including DONE/format requirements,
    # failure on collision, and success at the first visit to the goal.
    result = judge_maze(solution_str, ground_truth)
    distance = shortest_distance(ground_truth)
    if extra_info is not None and "optimal_path_length" in extra_info:
        if int(extra_info["optimal_path_length"]) != distance:
            raise ValueError("Dataset shortest distance disagrees with BFS")
    tokens = solution_str.strip().split()
    prefix = tokens[:tokens.index("DONE")] if "DONE" in tokens else tokens
    # For a valid output every item is one action. Invalid non-special items
    # also consume length. DONE/EOS/padding are not maze moves. Count the full
    # generated prefix even if simulation hit a wall or reached the goal early.
    length = len(prefix)
    cost = max(length, distance)
    result.update(
        generated_action_length=float(length),
        shortest_distance=float(distance),
        trajectory_cost=float(cost),
        cost_floor_active=float(length < distance),
        success_per_cost=float(result["score"]) / cost,
    )
    return result
