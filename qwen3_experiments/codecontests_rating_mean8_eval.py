"""Resumable CodeContests training-difficulty evaluation with eight samples."""

import argparse
import csv
from pathlib import Path

from qwen3_experiments import competition_eval as base
from qwen3_experiments import competition_grading
from qwen3_experiments.code_grading import final_code
from qwen3_experiments.nemotron_mean8_eval import aggregate, guard

MODULE = "qwen3_experiments.codecontests_rating_mean8_eval"


def grade(plan, question, code):
    question = {**question, "tests": base.read(question["tests_file"])}
    return competition_grading.grade(plan, question, code)


def aggregate_bands(questions, grades, expected_bands):
    """Require complete, disjoint rating groups and eight answers per question."""
    aggregate(questions, grades)
    groups = {band: ([], []) for band in expected_bands}
    question_bands = {}
    for question, result in zip(questions, grades):
        band = question["difficulty"]
        if band not in expected_bands:
            raise ValueError("Unexpected rating band")
        low, high = map(int, band.split("-"))
        if not low <= question["cf_rating"] <= high:
            raise ValueError("Question rating is outside its band")
        key = question["question_hash_id"]
        if key in question_bands and question_bands[key] != band:
            raise ValueError("Question appears in multiple rating bands")
        question_bands[key] = band
        groups[band][0].append(question)
        groups[band][1].append(result)
    summaries = {}
    for band, (selected, results) in groups.items():
        summary = aggregate(selected, results)
        if summary["questions"] != expected_bands[band]:
            raise ValueError("Incorrect number of questions in a rating band")
        summaries[band] = summary
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
    groups = aggregate_bands(questions, grades, plan["rating_bands"])
    summary = {"model": plan["models"]["baseline"], "groups": groups,
               "total": aggregate(questions, grades),
               "plan_sha256": base.digest(Path(plan["output_root"]) / "plan.json")}
    base.persist(plan, "mean8_summary.json", summary)
    by_hash = {q["question_hash_id"]: q for q in questions}
    per_question = []
    for band, group in groups.items():
        for item in group["per_question"]:
            question = by_hash[item["hash_id"]]
            per_question.append({**item, "rating_band": band, "cf_rating": question["cf_rating"],
                                 "name": question["name"]})
    with (Path(plan["output_root"]) / "per_question.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_question[0]))
        writer.writeheader()
        writer.writerows(per_question)
    lines = ["# CodeContests training difficulty: Qwen3-1.7B mean@8", "",
             "Train split; 50 randomly sampled unique statements per rating band, selection seed 42.",
             "Eight completions per question; thinking on; 32768 output tokens; temperature/top-p/top-k 0.6/0.95/20.",
             "Grade code after thinking only; no EOS gate. Original descriptions in the pinned official LCB prompt.",
             "All public/private/generated tests; official compiled CodeContests OutputsMatch with sandboxed Python execution.",
             "The execution backend is bubblewrap, with dataset time/memory limits, rather than upstream Sandbox2.",
             "This is a training-difficulty diagnostic using the dataset's cf_rating field.", "",
             "| Rating | Questions | Correct / samples | mean@8 (%) |", "|---|---:|---:|---:|"]
    for band, group in groups.items():
        lines.append(f"| {band} | {group['questions']} | {group['correct_samples']:.0f} / "
                     f"{group['total_samples']} | {group['mean_at_8_percent']:.2f} |")
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
