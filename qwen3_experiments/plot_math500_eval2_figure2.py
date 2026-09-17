#!/usr/bin/env python3
"""Plot Eval2 individual-token-budget results with the shared Figure 2 style.

Edit the adjacent JSON to adjust appearance. The renderer is shared with Eval3;
this module independently reads and audits the Eval2 per-question budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from plot_math500_eval3_figure2 import check, read_jsonl, render, write_csv

DEFAULT_CONFIG = Path(__file__).with_suffix(".json")
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FIELDS = (
    "dataset", "dataset_sha256", "num_prompts", "per_rollout_output_cap",
    "temperature", "top_p", "top_k", "seed", "grader", "grader_timeout_seconds",
    "max_prompt_len", "packages", "seed_scheme",
)


def audit_run(summary_path: Path, summary: dict) -> tuple[dict, list]:
    """Verify every question's own cumulative budget and at-least-one success."""
    stem = summary_path.name.removesuffix("_summary.json")
    prompts_path = summary_path.with_name(f"{stem}_prompts.jsonl")
    rollouts_path = summary_path.with_name(f"{stem}_rollouts.jsonl")
    prompts = list(read_jsonl(prompts_path))
    num_prompts = summary["num_prompts"]
    budget = summary["total_output_budget_per_prompt"]
    prompt_map = {row["prompt_position"]: row for row in prompts}
    check(len(prompts) == num_prompts and set(prompt_map) == set(range(num_prompts)),
          f"Incomplete or duplicate prompt summaries: {prompts_path}")
    identities = [(prompt_map[i]["prompt_index"], prompt_map[i]["unique_id"]) for i in range(num_prompts)]
    check(len(set(identities)) == num_prompts, f"Duplicate question IDs: {prompts_path}")

    attempts = defaultdict(int)
    tokens = defaultdict(int)
    successes = defaultdict(int)
    first_success = {}
    count = 0
    for row in read_jsonl(rollouts_path):
        count += 1
        position = row["prompt_position"]
        check(position in prompt_map, "Invalid question position")
        check(identities[position] == (row["prompt_index"], row["unique_id"]), "Question ID mismatch")
        check(row["rollout_index"] == attempts[position], "Per-question attempt sequence mismatch")
        expected_seed = (summary["seed"] + 1_000_003 * position + attempts[position]) % (2**31 - 1)
        check(row["rollout_seed"] == expected_seed, "Response seed mismatch")
        remaining = budget - tokens[position]
        check(row["remaining_budget_before"] == remaining, "Per-question remaining budget mismatch")
        cap = min(summary["per_rollout_output_cap"], remaining)
        check(row["max_output_tokens"] == cap, "Response cap differs from the per-question remaining budget")
        check(0 < row["output_tokens"] <= cap, "Invalid response length")
        tokens[position] += row["output_tokens"]
        check(row["cumulative_output_tokens"] == tokens[position], "Per-question cumulative token mismatch")
        check(row["remaining_budget_after"] == budget - tokens[position], "Per-question budget after mismatch")
        check(row["score"] in (0, 1), "Non-binary response score")
        if row["score"] > 0:
            successes[position] += 1
            first_success.setdefault(position, row["rollout_index"])
        attempts[position] += 1

    for position, prompt in prompt_map.items():
        check(tokens[position] == budget, "A question did not consume exactly its individual budget")
        check(prompt["total_output_tokens"] == tokens[position], "Prompt token summary mismatch")
        check(prompt["list_size"] == attempts[position], "Prompt response-list size mismatch")
        check(prompt["num_successes"] == successes[position], "Prompt success count mismatch")
        check(prompt["passed"] == (successes[position] > 0), "Prompt passed flag mismatch")
        check(prompt["first_success_rollout_index"] == first_success.get(position), "First-success index mismatch")
    solved = sum(successes[position] > 0 for position in range(num_prompts))
    check(solved == summary["num_prompts_passed"], "Solved count differs from summary")
    check(count == summary["total_rollouts"], "Response count differs from summary")
    check(sum(tokens.values()) == summary["total_generated_output_tokens"], "Total token count differs from summary")
    check(math.isclose(summary["list_size_mean"], count / num_prompts, abs_tol=1e-10), "Mean list size differs")
    check(math.isclose(summary["response_tokens_mean"], sum(tokens.values()) / count, abs_tol=1e-10),
          "Mean response length differs")
    return {
        "summary_file": str(summary_path),
        "rollouts_checked": count,
        "questions_checked": num_prompts,
        "individual_token_budget": budget,
        "all_individual_budgets_exhausted": True,
        "used_tokens": sum(tokens.values()),
        "solved_questions": solved,
        "success_aggregation_verified": True,
        "response_seeds_verified": True,
    }, identities


def load_points(config: dict, repo_root: Path, audit: bool) -> tuple[list[dict], dict]:
    axes = config["axes"]
    check(axes["x_metric"] in ("total_output_budget_per_prompt", "actual_tokens_per_question"),
          "Eval2 requires an individual-budget x metric")
    check(axes["y_metric"] in ("num_prompts_solved", "fraction_solved"), "Unsupported y metric")
    points, audits = [], []
    protocol, reference_identities = None, None
    ids = [model["id"] for model in config["models"]]
    check(len(ids) == len(set(ids)), "Model IDs must be unique")
    for model in config["models"]:
        for budget in config["budgets"]:
            stem = f"{model['model_key']}_total_budget_{budget}_rollout_cap_{config['rollout_cap']}"
            summary_path = repo_root / model["result_dir"] / f"{stem}_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = {
                "model_label": model["model_key"], "num_prompts": config["num_prompts"],
                "total_output_budget_per_prompt": budget, "per_rollout_output_cap": config["rollout_cap"],
            }
            for key, value in expected.items():
                check(summary.get(key) == value, f"Unexpected {key}: {summary_path}")
            current_protocol = {key: summary[key] for key in PROTOCOL_FIELDS}
            if protocol is None:
                protocol = current_protocol
            check(current_protocol == protocol, f"Evaluation protocol differs: {summary_path}")
            used, solved = summary["total_generated_output_tokens"], summary["num_prompts_passed"]
            check(used == budget * config["num_prompts"], "Total tokens differ from the individual budgets")
            check(0 <= solved <= config["num_prompts"], "Invalid solved count")
            check(math.isclose(summary["pass_at_realized_list_size"], solved / config["num_prompts"], abs_tol=1e-12),
                  "Solved fraction differs from summary")
            if audit:
                result, identities = audit_run(summary_path, summary)
                result["summary_file"] = str(summary_path.relative_to(repo_root))
                if reference_identities is None:
                    reference_identities = identities
                check(identities == reference_identities, "Question IDs differ across model/budget conditions")
                audits.append(result)
            points.append({
                "model_id": model["id"], "model": model["label"], "model_key": model["model_key"],
                "training_step": model["training_step"],
                "total_output_budget_per_prompt": budget,
                "actual_tokens_per_question": used / config["num_prompts"],
                "num_prompts_solved": solved,
                "fraction_solved": solved / config["num_prompts"],
                "num_prompts": config["num_prompts"],
                "total_generated_output_tokens": used,
                "total_rollouts": summary["total_rollouts"],
                "mean_list_size": summary["list_size_mean"],
                "per_rollout_output_cap": summary["per_rollout_output_cap"],
                "summary_file": str(summary_path.relative_to(repo_root)),
                "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "checkpoint_repo": summary["checkpoint_repo"],
            })
    return points, {
        "evaluation": "Eval2 individual cumulative token budget",
        "protocol": protocol, "condition_count": len(points),
        "raw_audit_requested": audit,
        "raw_rollouts_checked": sum(row["rollouts_checked"] for row in audits),
        "conditions": audits,
        "x_metric": axes["x_metric"], "y_metric": axes["y_metric"],
        "x_definition": "Individual cumulative output-token budget per question; all questions exhaust this budget",
        "y_definition": "Number of questions with at least one correct response in their generated list, out of 500"
        if axes["y_metric"] == "num_prompts_solved" else "Fraction of questions with at least one correct response",
        "generation_protocol": "Each question uses its own full budget, including further samples after a correct response",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--line-mode", choices=("paper-fit", "connect", "none"))
    parser.add_argument("--audit", action="store_true", help="Verify each question's raw responses and budget")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.line_mode:
        config["lines"]["mode"] = args.line_mode
    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or repo_root / config["output_dir"]).resolve()
    points, validation = load_points(config, repo_root, args.audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    fits = render(points, config, output_dir)
    write_csv(output_dir / "points.csv", points)
    write_csv(output_dir / "fits.csv", fits)
    (output_dir / "config_used.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    report_name = "audit.json" if args.audit else "summary_checks.json"
    (output_dir / report_name).write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir), "models": len(config["models"]), "points": len(points),
        "line_mode": config["lines"]["mode"], "raw_rollouts_checked": validation["raw_rollouts_checked"],
    }, indent=2))


if __name__ == "__main__":
    main()
