#!/usr/bin/env python3
"""Audit raw MATH-500 rollouts and compare offset256 with the saved cap8 baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from run_math500_offset256_comparison import (
    BASELINE_LABEL,
    BASELINE_REPO,
    CHECKPOINT,
    LABEL,
    PROTOCOLS,
    REPO_ROOT,
    result_paths,
    validate_summary,
    write_json,
)

TITLES = {
    "eval1": "Eval 1: single-rollout accuracy (mean@4)",
    "eval2": "Eval 2: per-question list success rate",
    "eval3_skip_solved": "Eval 3(1): shared budget, skip solved questions",
    "eval3_iid": "Eval 3(2): shared budget, IID draws including solved questions",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit(directory: Path, protocol: str, budget: int, label: str) -> dict:
    summary = validate_summary(directory, protocol, budget, label)
    paths = result_paths(directory, protocol, budget, label)
    records = read_jsonl(paths[0])
    if not all(row["score"] in (0, 1) for row in records):
        raise ValueError("Expected binary MathVerify scores")
    if protocol == "eval1":
        counts = Counter(row["prompt_index"] for row in records)
        assert len(records) == 2000 and len(counts) == 500 and set(counts.values()) == {4}
        assert all(0 < row["output_tokens"] <= budget for row in records)
        slots = {(row["prompt_index"], row["sample_index"]) for row in records}
        assert len(slots) == 2000
        accuracy = sum(row["score"] for row in records) / 2000
    else:
        prompts = read_jsonl(paths[1])
        assert len(prompts) == 500 and {row["prompt_position"] for row in prompts} == set(range(500))
        solved = set()
        lengths = defaultdict(int)
        attempts = Counter()
        rng = random.Random(0)
        shared_remaining = 500 * budget
        permutations = {}
        last_rank = {}
        for index, row in enumerate(records):
            position = row["prompt_position"]
            assert 0 <= position < 500
            assert row["rollout_index"] == attempts[position]
            assert row["rollout_seed"] == (1_000_003 * position + attempts[position]) % (2**31 - 1)
            length = row["output_tokens"]
            assert 0 < length <= row["max_output_tokens"] <= 4096
            if protocol == "eval2":
                remaining = budget - lengths[position]
                assert row["remaining_budget_before"] == remaining
                assert row["max_output_tokens"] == min(4096, remaining)
                assert row["remaining_budget_after"] == remaining - length
            else:
                assert row["request_sequence_index"] == index
                assert row["global_budget_before"] == shared_remaining
                shared_remaining -= length
                assert shared_remaining >= 0 and row["global_budget_after"] == shared_remaining
                assert row["solved_before"] == (position in solved)
                if protocol == "eval3_iid":
                    assert row["question_draw_index"] == index and position == rng.randrange(500)
                else:
                    assert position not in solved
                    round_index = row["round_index"]
                    if round_index not in permutations:
                        order = list(range(500))
                        random.Random((2_000_033 * round_index) % (2**31 - 1)).shuffle(order)
                        permutations[round_index] = order
                    rank = row["permutation_rank"]
                    assert rank > last_rank.get(round_index, -1)
                    assert permutations[round_index][rank] == position
                    last_rank[round_index] = rank
            lengths[position] += length
            attempts[position] += 1
            if row["score"] > 0:
                solved.add(position)
        for row in prompts:
            position = row["prompt_position"]
            assert row["total_output_tokens"] == lengths[position]
            assert row["passed" if protocol == "eval2" else "solved"] == (position in solved)
            if protocol == "eval2":
                assert lengths[position] == budget and row["list_size"] == attempts[position]
            else:
                assert row["attempts"] == attempts[position]
        if protocol == "eval2":
            assert sum(lengths.values()) == 500 * budget
        else:
            assert shared_remaining == summary["global_output_budget_remaining"]
            assert sum(lengths.values()) == summary["global_output_budget_used"]
        accuracy = len(solved) / 500
    assert math.isclose(accuracy, summary[PROTOCOLS[protocol][2]], abs_tol=1e-12)
    return {
        "accuracy": accuracy,
        "response_count": len(records),
        "mean_response_tokens": sum(row["output_tokens"] for row in records) / len(records),
        "summary_file": str(paths[-1]),
        "raw_rollout_audit": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    all_results = {}
    markdown = [
        "# MATH-500: offset256 vs cap8 (step 150)",
        "",
        f"- [Offset256](https://huggingface.co/{CHECKPOINT})",
        f"- [Cap8 / hard clip min tokens=256](https://huggingface.co/{BASELINE_REPO})",
        "",
        "All results use all 500 questions, temperature 0.6, top-p 0.95, top-k -1, seed 0, and the historical training MathVerify scorer (one-second timeout). Cap8 results are reused unchanged.",
        "",
        "Eval 1 averages four sampled binary scores per question (mean@4, not pass@4). "
        "Eval 2 gives each question its own total budget and scores whether any answer succeeds. "
        "Eval 3 gives all questions a shared budget of 500 × b; its denominator remains 500, including unvisited questions. "
        "Eval 2/3 cap each response at 4096 or the remaining budget; only generated tokens are charged.",
        "",
    ]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    for protocol, (baseline_dir, budgets, _) in PROTOCOLS.items():
        results = {"cap8": {}, "offset256": {}}
        for name, label, directory in (
            ("cap8", BASELINE_LABEL, REPO_ROOT / "outputs" / baseline_dir / "results"),
            ("offset256", LABEL, root / protocol / "results"),
        ):
            for budget in budgets:
                results[name][str(budget)] = audit(directory, protocol, budget, label)
        all_results[protocol] = results
        markdown += [f"## {TITLES[protocol]}", "", "| Model | " + " | ".join(map(str, budgets)) + " |", "|---|" + "---:|" * len(budgets)]
        for name, display in (("cap8", "Cap8"), ("offset256", "Offset256")):
            markdown.append(f"| {display} | " + " | ".join(f"{results[name][str(b)]['accuracy']:.2%}" for b in budgets) + " |")
        differences = [(results["offset256"][str(b)]["accuracy"] - results["cap8"][str(b)]["accuracy"]) * 100 for b in budgets]
        markdown += ["| Offset256 − Cap8 (pp) | " + " | ".join(f"{d:+.2f}" for d in differences) + " |", ""]
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for name, display, color, marker in (("cap8", "Cap8 (hard clip min tokens=256)", "#2864dc", "o"), ("offset256", "Offset256", "#e4662d", "s")):
            ax.plot(range(len(budgets)), [results[name][str(b)]["accuracy"] for b in budgets], label=display, color=color, marker=marker, linewidth=1.5, markersize=5)
        ax.set_xticks(range(len(budgets)), list(map(str, budgets)))
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_xlabel("Maximum output tokens per response" if protocol == "eval1" else ("Total output-token budget per question" if protocol == "eval2" else "Reference budget b (shared total = 500 × b)"))
        ax.set_ylabel("Mean@4 accuracy" if protocol == "eval1" else "Fraction of all 500 questions solved")
        ax.set_title(TITLES[protocol])
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(root / f"{protocol}_offset256_vs_cap8.png", dpi=180)
        plt.close(fig)
        markdown += [f"![{TITLES[protocol]}]({protocol}_offset256_vs_cap8.png)", ""]
    markdown += ["## Validation", "", "All 38 model/budget result sets passed raw-rollout checks for score aggregation, prompt counts, token accounting, and (where applicable) the seeded question stream and success-aware skipping. Eval 3 was formerly named Eval 4 in this repository.", ""]
    write_json(root / "comparison.json", all_results)
    (root / "comparison.md").write_text("\n".join(markdown))
    print(root / "comparison.md")


if __name__ == "__main__":
    main()
