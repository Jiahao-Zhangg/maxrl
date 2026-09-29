"""Evaluate AtCoder contest families with eight samples per question."""

import argparse
import csv
from pathlib import Path

from qwen3_experiments import competition_eval as base
from qwen3_experiments.code_grading import final_code
from qwen3_experiments.codecontests_rating_mean8_eval import grade
from qwen3_experiments.nemotron_mean8_eval import aggregate, guard

MODULE = "qwen3_experiments.codecontests_atcoder_mean8_eval"


def aggregate_atcoder(questions, grades, expected_groups):
    """Verify task identity and scope before reporting sample accuracy."""
    aggregate(questions, grades)
    identities = {}
    groups = {family: ([], []) for family in expected_groups}
    for question, result in zip(questions, grades):
        family = question["contest_family"]
        if question["upstream_source"] != 5 or family not in expected_groups:
            raise ValueError("Question is outside the requested AtCoder scope")
        key = question["question_hash_id"]
        identity = (question["atcoder_problem_id"], question["atcoder_contest_id"],
                    family, question["original_source_index"], question["atcoder_problem_index"])
        if key in identities and identities[key] != identity:
            raise ValueError("Question identity changes across samples")
        identities[key] = identity
        groups[family][0].append(question)
        groups[family][1].append(result)
    if len({identity[0] for identity in identities.values()}) != len(identities):
        raise ValueError("Duplicate AtCoder problem under different statements")
    summaries = {}
    for family, (selected, results) in groups.items():
        summary = aggregate(selected, results)
        if summary["questions"] != expected_groups[family]:
            raise ValueError("Incorrect number of AtCoder questions")
        summaries[family] = summary
    return summaries


def report(plan):
    task = plan["order"][0]
    if not base.complete(plan, task):
        return
    directory = base.folder(plan, task)
    questions = base.read(base.question_path(plan, task))
    audit = base.read(directory / "audit.json")
    grades = []
    for question in questions:
        path = directory / "grades" / f"{question['source_index']}.json"
        if base.digest(path) != audit["grade_hashes"][str(question["source_index"])]:
            raise ValueError("Grade changed after completion")
        grades.append(base.read(path))
    scope = plan["atcoder_scope"]
    groups = aggregate_atcoder(questions, grades, scope["groups"])
    summary = {"groups": groups, "total": aggregate(questions, grades),
               "model": plan["models"]["baseline"], "scope": scope,
               "plan_sha256": base.digest(Path(plan["output_root"]) / "plan.json")}
    by_hash = {q["question_hash_id"]: q for q in questions}
    for item in summary["total"]["per_question"]:
        question = by_hash[item["hash_id"]]
        for field in ("name", "atcoder_problem_id", "atcoder_contest_id", "atcoder_problem_index",
                      "contest_family", "atcoder_url"):
            item[field] = question[field]
    base.persist(plan, "mean8_summary.json", summary)
    with (Path(plan["output_root"]) / "per_question.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["total"]["per_question"][0]))
        writer.writeheader()
        writer.writerows(summary["total"]["per_question"])
    label = "/".join(family.upper() for family in scope["groups"])
    lines = [f"# CodeContests {label}: Qwen3-1.7B mean@8", "",
             f"{summary['total']['questions']} unique train-split questions sampled without replacement; seed 42; eight samples each.",
             "All task indices are eligible. This measures training difficulty, not held-out generalization.",
             "Thinking on; 32768 output tokens; temperature/top-p/top-k 0.6/0.95/20; grade after thinking only; no EOS gate.",
             "Pinned official LCB standard-input prompt with original CodeContests statements.",
             "All public/private/generated tests; official compiled CodeContests OutputsMatch; binary fail-fast.",
             "Execution uses bubblewrap with dataset time/memory limits, rather than upstream Sandbox2.", "",
             "| Contest | Questions | Correct / samples | mean@8 (%) |", "|---|---:|---:|---:|"]
    for family, group in groups.items():
        lines.append(f"| {family.upper()} | {group['questions']} | {group['correct_samples']:.0f} / "
                     f"{group['total_samples']} | {group['mean_at_8_percent']:.2f} |")
    lines.extend(["", "| Correct out of 8 | " + " | ".join(family.upper() for family in groups) + " |",
                  "|---:|" + "---:|" * len(groups)])
    for correct in range(9):
        counts = " | ".join(str(group['correct_samples_histogram'].get(correct, 0)) for group in groups.values())
        lines.append(f"| {correct} | {counts} |")
    for root in (Path(plan["output_root"]), Path(plan["scratch"]) / "control_mirrors"):
        try:
            (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
        except OSError:
            pass


def configure():
    base.MODULE = MODULE
    base.grade = grade
    base.final_code = final_code
    base.report = report


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command")
    parser.add_argument("--root", type=Path, required=True)
    args, _ = parser.parse_known_args()
    plan = base.read(args.root / "plan.json")
    configure()
    if args.command == "guard":
        base.verify(plan)
        return guard(plan)
    base.main()
    if args.command == "launch":
        child = base.launch_child(plan, "guard")
        base.persist(plan, "watchdog_launch.json", {"pid": child.pid, "node": plan["node"]})


if __name__ == "__main__":
    main()
