"""Seeded finite-pool Monte Carlo replay with proportional boundary credit."""

import itertools

import numpy as np


def individual_orders(lengths, scores, orders, budgets):
    """Score the first correct response in each supplied answer permutation."""
    L = np.asarray(lengths, dtype=np.int64)[orders]
    R = np.asarray(scores, dtype=np.int8)[orders]
    present = np.any(R, axis=1)
    first = np.argmax(R, axis=1)
    row = np.arange(len(orders))
    correct_lengths = L[row, first]
    starts = np.cumsum(L, axis=1)[row, first] - correct_lengths
    answer = np.clip((np.asarray(budgets)[None, :] - starts[:, None]) / correct_lengths[:, None], 0, 1)
    answer[~present] = 0
    return answer


def event_path(lengths, scores, question_ids, budgets, question_count):
    """Deduplicate successes and assign partial credit to the boundary answer."""
    L = np.asarray(lengths, dtype=np.int64)
    R = np.asarray(scores, dtype=np.int8)
    Q = np.asarray(question_ids, dtype=np.int64)
    if not len(L):
        return np.zeros(len(budgets), dtype=np.float64)
    assert np.all(L > 0) and np.all((R == 0) | (R == 1))
    positions = np.flatnonzero(R)
    if not len(positions):
        return np.zeros(len(budgets), dtype=np.float64)
    _, first = np.unique(Q[positions], return_index=True)
    positions = positions[first]
    cumulative = np.cumsum(L)
    start = cumulative[positions] - L[positions]
    contribution = np.clip((np.asarray(budgets)[:, None] - start[None, :]) / L[positions][None, :], 0, 1)
    return contribution.sum(axis=1) / question_count


def individual(lengths, scores, budgets, positions, dataset_index, repeats=100, seed=42):
    result = np.zeros((repeats, len(budgets)))
    for L, R, pos in zip(lengths, scores, positions, strict=True):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1, dataset_index, int(pos)]))
        orders = np.asarray([rng.permutation(len(L)) for _ in range(repeats)])
        result += individual_orders(L, R, orders, budgets)
    return result / len(lengths)


def cross_exit_once(lengths, scores, targets, response_rng, sweep_rng):
    n = len(lengths)
    orders = [response_rng.permutation(len(L)) for L in lengths]
    cursor = np.zeros(n, dtype=np.int64)
    solved = np.zeros(n, dtype=bool)
    active = np.arange(n)
    spent = 0
    event_l, event_r, event_q = [], [], []
    while len(active) and spent < targets[-1]:
        for q in sweep_rng.permutation(active):
            if cursor[q] >= len(orders[q]):
                raise ValueError(f"Unsolved question {q} exhausted at {spent} < {targets[-1]}")
            a = orders[q][cursor[q]]
            cursor[q] += 1
            L, R = int(lengths[q][a]), int(scores[q][a])
            event_l.append(L)
            event_r.append(R)
            event_q.append(q)
            spent += L
            solved[q] = bool(R)
            if spent >= targets[-1]:
                break
        active = active[~solved[active]]
    assert spent >= targets[-1] or np.all(solved)
    values = event_path(event_l, event_r, event_q, targets, n)
    return values, len(event_l)


def cross_exit(lengths, scores, budgets, dataset_index, repeats=100, seed=42):
    targets = np.asarray(budgets, dtype=np.int64) * len(lengths)
    result, events = [], []
    for trial in range(repeats):
        response_rng = np.random.default_rng(np.random.SeedSequence([seed, 2, dataset_index, trial, 0]))
        sweep_rng = np.random.default_rng(np.random.SeedSequence([seed, 2, dataset_index, trial, 1]))
        row, count = cross_exit_once(lengths, scores, targets, response_rng, sweep_rng)
        result.append(row)
        events.append(count)
    return np.asarray(result), events


def cross_no_exit(lengths, scores, budgets, dataset_index, repeats=100, seed=42):
    assert len({len(L) for L in lengths}) == 1, "Global pair pool must remain balanced by question"
    n = len(lengths)
    L, R = np.concatenate(lengths), np.concatenate(scores)
    Q = np.repeat(np.arange(n), len(lengths[0]))
    targets = np.asarray(budgets, dtype=np.int64) * n
    assert int(L.sum()) >= targets[-1]
    result = []
    for trial in range(repeats):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3, dataset_index, trial]))
        order = rng.permutation(len(L))
        result.append(event_path(L[order], R[order], Q[order], targets, n))
    return np.asarray(result)


def validate():
    # Enumerate all six orders of a small pool, including a fractional boundary.
    orders = np.asarray(list(itertools.permutations(range(3))))
    actual = individual_orders([2, 3, 5], [0, 1, 0], orders, [1, 3, 6, 12]).mean(axis=0)
    expected = [1 / 9, 7 / 18, 5 / 9, 1]
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    # A solved question's later answers spend tokens but earn no duplicate credit.
    actual = event_path([4, 6, 8], [1, 1, 1], [0, 0, 1], [2, 4, 7, 10, 14, 18], 2)
    np.testing.assert_allclose(actual, [0.25, 0.5, 0.5, 0.5, 0.75, 1], atol=1e-14, rtol=0)
    np.testing.assert_array_equal(event_path([5, 5], [0, 0], [0, 1], [2, 7, 10], 2), [0, 0, 0])
    # Sampling stops a solved question; the costly second answer is never visited.
    L, R = [[2, 100], [3, 3, 3]], [[1, 1], [0, 0, 0]]

    class IdentityRng:
        @staticmethod
        def permutation(x):
            return np.arange(x) if np.isscalar(x) else np.asarray(x)

    values, count = cross_exit_once(L, R, [1, 2, 5, 8], IdentityRng(), IdentityRng())
    np.testing.assert_allclose(values, [0.25, 0.5, 0.5, 0.5], atol=1e-14, rtol=0)
    assert count == 3
    # Pool exhaustion must be an error, not a silently skipped question.
    try:
        cross_exit_once([[1], [1]], [[0], [1]], [10], IdentityRng(), IdentityRng())
    except ValueError:
        pass
    else:
        raise AssertionError("Failed to reject unsupported unsolved exhaustion")
    # Independent trial randomness is reproducible without model-specific seeds.
    a = cross_no_exit([[2, 3], [4, 5]], [[0, 1], [1, 0]], [1, 3], 0, repeats=100)
    b = cross_no_exit([[2, 3], [4, 5]], [[0, 1], [1, 0]], [1, 3], 0, repeats=100)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.diff(a, axis=1) >= -1e-14)
    return {
        "passed": True,
        "enumerated_individual_orders": 6,
        "boundary_partial_credit": True,
        "question_success_deduplicated": True,
        "solved_questions_removed_in_exit_mode": True,
        "unsolved_exhaustion_rejected": True,
        "seed_reproducibility": True,
        "monotonic_shared_trial_budgets": True,
    }
