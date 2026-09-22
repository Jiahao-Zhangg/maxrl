#!/usr/bin/env python3
"""Plot Eval1 mean@4 results with the shared Figure 2 style.

Edit the adjacent JSON for appearance. Each point averages 500 questions with
four responses per question; --audit verifies the original saved responses.
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
    "dataset", "dataset_sha256", "num_prompts", "num_samples_per_prompt",
    "num_scored_responses", "temperature", "top_p", "top_k", "seed", "grader",
    "grader_timeout_seconds", "max_prompt_len", "packages",
)
X_METRICS = {
    "mean_output_tokens": "Actual mean output tokens over every response, including incorrect and truncated responses",
    "max_output_tokens": "Per-response output-token generation limit",
}


def audit_run(summary_path: Path, summary: dict) -> tuple[dict, list]:
    """Check complete sample slots, question identities, token lengths and scores."""
    stem = summary_path.name.removesuffix("_summary.json")
    samples_path = summary_path.with_name(f"{stem}_samples.jsonl")
    num_samples = summary["num_samples_per_prompt"]
    num_prompts = summary["num_prompts"]
    cap = summary["max_output_len"]
    slots, identities = set(), {}
    counts = defaultdict(int)
    tokens, correct, capped, longest = 0, 0, 0, 0
    solved_by_sample = [0] * num_samples
    for row in read_jsonl(samples_path):
        prompt, sample = row["prompt_index"], row["sample_index"]
        slot = (prompt, sample)
        check(sample in range(num_samples) and slot not in slots, f"Invalid sample slot: {samples_path}")
        slots.add(slot)
        counts[prompt] += 1
        identities.setdefault(prompt, row["unique_id"])
        check(identities[prompt] == row["unique_id"], f"Question ID mismatch: {samples_path}")
        length, score = row["output_tokens"], row["score"]
        check(isinstance(length, int) and 0 < length <= cap, f"Invalid response length: {samples_path}")
        check(score in (0, 1), f"Non-binary response score: {samples_path}")
        tokens += length
        correct += int(score)
        capped += length == cap
        longest = max(longest, length)
        solved_by_sample[sample] += int(score)

    count = len(slots)
    check(count == num_prompts * num_samples == summary["num_scored_responses"],
          f"Incomplete evaluation: {samples_path}")
    check(len(counts) == num_prompts and set(counts.values()) == {num_samples},
          f"Missing questions or unequal sample counts: {samples_path}")
    check(len(set(identities.values())) == num_prompts, f"Duplicate question IDs: {samples_path}")
    expected = {
        "correct_responses": correct,
        "mean_at_4_accuracy": correct / count,
        "mean_output_tokens": tokens / count,
        "max_observed_output_tokens": longest,
        "budget_exhaustion_rate": capped / count,
    }
    for field, value in expected.items():
        check(math.isclose(summary[field], value, rel_tol=0, abs_tol=1e-10),
              f"Raw responses disagree with {field}: {summary_path}")
    return {
        "summary_file": str(summary_path),
        "samples_checked": count,
        "questions_checked": num_prompts,
        "samples_per_question": num_samples,
        "max_output_tokens": cap,
        "actual_total_output_tokens": tokens,
        "mean_output_tokens": tokens / count,
        "correct_responses": correct,
        "mean_at_4_accuracy": correct / count,
        "mean_questions_solved": correct / num_samples,
        "solved_counts_by_sample_index": solved_by_sample,
        "complete_unique_sample_slots": True,
    }, sorted(identities.items())


def load_points(config: dict, repo_root: Path, audit: bool) -> tuple[list[dict], dict]:
    axes = config["axes"]
    check(axes["x_metric"] in X_METRICS, "Unsupported Eval1 x metric")
    check(axes["y_metric"] in ("mean_questions_solved", "fraction_solved"), "Unsupported Eval1 y metric")
    check(config["num_samples_per_prompt"] == 4, "This figure uses mean@4")
    points, audits = [], []
    protocol, reference_identities = None, None
    ids = [model["id"] for model in config["models"]]
    check(len(ids) == len(set(ids)), "Model IDs must be unique")
    budgets = config["budgets"]
    check(len(budgets) >= 2 and len(budgets) == len(set(budgets)), "Need distinct output caps")
    for model in config["models"]:
        for budget in budgets:
            stem = f"{model['model_key']}_max_tokens_{budget}"
            summary_path = repo_root / model["result_dir"] / f"{stem}_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = {
                "model_label": model["model_key"], "num_prompts": config["num_prompts"],
                "max_output_len": budget, "num_samples_per_prompt": config["num_samples_per_prompt"],
                "num_scored_responses": config["num_prompts"] * config["num_samples_per_prompt"],
            }
            for field, value in expected.items():
                check(summary.get(field) == value, f"Unexpected {field}: {summary_path}")
            current_protocol = {field: summary[field] for field in PROTOCOL_FIELDS}
            if protocol is None:
                protocol = current_protocol
            check(current_protocol == protocol, f"Evaluation protocol differs: {summary_path}")
            mean_tokens, accuracy = summary["mean_output_tokens"], summary["mean_at_4_accuracy"]
            correct, responses = summary["correct_responses"], summary["num_scored_responses"]
            check(0 < mean_tokens <= budget, f"Invalid actual mean length: {summary_path}")
            check(0 <= correct <= responses and int(correct) == correct, f"Invalid correct count: {summary_path}")
            check(math.isclose(accuracy, correct / responses, rel_tol=0, abs_tol=1e-12),
                  f"Mean@4 differs from the response scores: {summary_path}")
            if audit:
                result, identities = audit_run(summary_path, summary)
                result["summary_file"] = str(summary_path.relative_to(repo_root))
                if reference_identities is None:
                    reference_identities = identities
                check(identities == reference_identities, "Question IDs differ across model/cap conditions")
                audits.append(result)
            points.append({
                "model_id": model["id"], "model": model["label"], "model_key": model["model_key"],
                "training_step": model["training_step"],
                "max_output_tokens": budget,
                "mean_output_tokens": mean_tokens,
                "mean_questions_solved": correct / config["num_samples_per_prompt"],
                "fraction_solved": accuracy,
                "mean_at_4_accuracy": accuracy,
                "correct_responses": int(correct),
                "num_prompts": config["num_prompts"],
                "num_samples_per_prompt": config["num_samples_per_prompt"],
                "num_responses": responses,
                "summary_file": str(summary_path.relative_to(repo_root)),
                "samples_file": str(summary_path.with_name(f"{stem}_samples.jsonl").relative_to(repo_root)),
                "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "checkpoint_repo": summary["checkpoint_repo"],
            })
    return points, {
        "evaluation": "Eval1 single-response mean@4",
        "protocol": protocol, "condition_count": len(points),
        "raw_audit_requested": audit,
        "raw_samples_checked": sum(row["samples_checked"] for row in audits),
        "conditions": audits,
        "x_metric": axes["x_metric"], "y_metric": axes["y_metric"],
        "x_definition": X_METRICS[axes["x_metric"]],
        "y_definition": "Mean number of questions solved = num_prompts * mean@4 = correct responses / 4"
        if axes["y_metric"] == "mean_questions_solved" else "Single-response mean@4 accuracy",
        "cap_definition": "Per-response output-token generation limit",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--line-mode", choices=("paper-fit", "connect", "none"))
    parser.add_argument("--audit", action="store_true", help="Verify all original per-question response samples")
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
        "line_mode": config["lines"]["mode"], "raw_samples_checked": validation["raw_samples_checked"],
    }, indent=2))


if __name__ == "__main__":
    main()
