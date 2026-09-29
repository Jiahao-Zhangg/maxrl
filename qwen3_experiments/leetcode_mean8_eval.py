"""Resumable LeetCodeDataset mean@8, gated on a completed preceding evaluation."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from qwen3_experiments import competition_eval as base
from qwen3_experiments.code_grading import final_code
from qwen3_experiments.nemotron_mean8_eval import aggregate, guard
from qwen3_experiments.taco_eval import sandbox_command

MODULE = "qwen3_experiments.leetcode_mean8_eval"
EVALUATION_QUEUE = base.queue


def grade(plan, question, code):
    problem = base.read(question["problem_file"])
    timeout = plan["suite_timeout_seconds"]
    temp_root = Path(plan["scratch"]) / "grading_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        payload = Path(directory) / "input.json"
        base.write(payload, {"problem": problem, "code": code, "timeout": timeout})
        process = subprocess.Popen(sandbox_command(plan, payload), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout + 30)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise RuntimeError("LeetCode checker exceeded its outer execution deadline") from exc
        if process.returncode:
            raise RuntimeError(f"LeetCode sandbox exit {process.returncode}: {stderr[-2000:]}")
        result = json.loads(stdout)
        if result.get("infrastructure_error"):
            raise RuntimeError(result["infrastructure_error"])
        if result.get("task_id") != problem["task_id"] or type(result.get("passed")) is not bool:
            raise ValueError("Invalid official checker result")
        passed = result["passed"]
        result.update(score=float(passed), results=[int(passed)], seconds=time.monotonic() - started,
                      suite_timeout_seconds=timeout, checker="unmodified LeetCodeDataset check_correctness")
        return result


def aggregate_difficulties(questions, grades, expected_groups):
    """Mean sample accuracy, with complete and disjoint difficulty groups."""
    aggregate(questions, grades)
    groups = {level: ([], []) for level in expected_groups}
    identities = {}
    for question, result in zip(questions, grades):
        level = question["difficulty"]
        if level not in groups or question["dataset"] != "leetcode":
            raise ValueError("Question is outside the requested LeetCode scope")
        key = question["question_hash_id"]
        identity = (question["task_id"], question["original_source_index"], level)
        if key in identities and identities[key] != identity:
            raise ValueError("Question identity changes across samples")
        identities[key] = identity
        groups[level][0].append(question)
        groups[level][1].append(result)
    if len({identity[0] for identity in identities.values()}) != len(identities):
        raise ValueError("Duplicate LeetCode task under different statements")
    summaries = {}
    for level, (selected, results) in groups.items():
        summary = aggregate(selected, results)
        if summary["questions"] != expected_groups[level]:
            raise ValueError("Incorrect number of questions in a difficulty group")
        summaries[level] = summary
    return summaries


def predecessor_complete(plan):
    dependency = plan["predecessor"]
    root = Path(dependency["output_root"])
    if base.digest(root / "plan.json") != dependency["plan_sha256"]:
        raise ValueError("Predecessor plan changed")
    previous = base.read(root / "plan.json")
    if base.state(previous, "queue_status.json").get("state") != "complete":
        return False
    if not all(base.complete(previous, task) for task in previous["order"]):
        return False
    summary = base.read(root / "mean8_summary.json")
    if summary["plan_sha256"] != dependency["plan_sha256"]:
        raise ValueError("Predecessor summary belongs to a different plan")
    for task in previous["order"]:
        folder = base.folder(previous, task)
        audit = base.read(folder / "audit.json")
        for group, field in (("responses", "response_hashes"), ("grades", "grade_hashes")):
            if len(audit[field]) != previous["tasks"][task]["questions"]:
                raise ValueError("Predecessor audit is incomplete")
            for index, expected in audit[field].items():
                if base.digest(folder / group / f"{index}.json") != expected:
                    raise ValueError("Predecessor result changed after its audit")
    return True


def queue(plan):
    scratch = Path(plan["scratch"])
    with (scratch / "guard/dependency.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (Path(plan["output_root"]) / "cancellation.json").exists():
            try:
                if predecessor_complete(plan):
                    base.persist(plan, "dependency_receipt.json", {
                        "verified": True, "predecessor": plan["predecessor"], "verified_at": base.now()})
                    return EVALUATION_QUEUE(plan)
                base.persist(plan, "queue_status.json", {"state": "waiting_for_predecessor",
                    "predecessor": plan["predecessor"]["output_root"], "progress": base.progress(plan),
                    "pid": os.getpid()})
            except Exception as exc:
                base.persist(plan, "queue_status.json", {"state": "dependency_recovering", "error": str(exc),
                                                         "pid": os.getpid()})
            time.sleep(30)


def report(plan):
    task = plan["order"][0]
    if not base.complete(plan, task):
        return
    folder = base.folder(plan, task)
    questions = base.read(base.question_path(plan, task))
    audit = base.read(folder / "audit.json")
    grades = []
    for question in questions:
        path = folder / "grades" / f"{question['source_index']}.json"
        if base.digest(path) != audit["grade_hashes"][str(question["source_index"])]:
            raise ValueError("Grade changed after completion")
        grades.append(base.read(path))
    groups = aggregate_difficulties(questions, grades, plan["difficulty_groups"])
    summary = {"groups": groups, "total": aggregate(questions, grades), "model": plan["models"]["baseline"],
               "source": plan["sources"], "plan_sha256": base.digest(Path(plan["output_root"]) / "plan.json")}
    summary["reference_check"] = plan["reference_check"]
    reference_passed_groups = {}
    for level in plan["difficulty_groups"]:
        pairs = [(question, result) for question, result in zip(questions, grades)
                 if question["difficulty"] == level and question["reference_passed"]]
        if pairs:
            reference_passed_groups[level] = aggregate([q for q, _ in pairs], [r for _, r in pairs])
    summary["reference_passed_subset"] = reference_passed_groups
    by_hash = {question["question_hash_id"]: question for question in questions}
    for item in summary["total"]["per_question"]:
        question = by_hash[item["hash_id"]]
        item.update(task_id=question["task_id"], question_id=question["question_id"], tags=question["tags"],
                    reference_passed=question["reference_passed"])
    base.persist(plan, "mean8_summary.json", summary)
    with (Path(plan["output_root"]) / "per_question.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["total"]["per_question"][0]))
        writer.writeheader()
        writer.writerows(summary["total"]["per_question"])
    lines = ["# LeetCodeDataset: Qwen3-1.7B mean@8", "",
             "Train split; 50 Medium and 50 Hard questions; seed 42; eight independently seeded samples per question.",
             "Original dataset queries; thinking on; 32768 output tokens; temperature/top-p/top-k 0.6/0.95/20.",
             "Code after thinking only; no EOS gate. Original dataset checker, prompts, entry points and tests.",
             f"Timeout: {plan['grading']['suite_timeout_seconds']:g} seconds for the entire test suite of one answer.",
             "Execution is isolated with bubblewrap. These are dataset tests, not LeetCode's online judge.", "",
             "| Difficulty | Questions | Correct / samples | mean@8 (%) |", "|---|---:|---:|---:|"]
    for level, group in groups.items():
        lines.append(f"| {level} | {group['questions']} | {group['correct_samples']:.0f} / "
                     f"{group['total_samples']} | {group['mean_at_8_percent']:.2f} |")
    failures = plan["reference_check"]["failed_tasks"]
    if failures:
        lines.extend(["", f"The dataset's own reference solutions failed the same timeout on {len(failures)} selected tasks:",
                      ", ".join(failures) + ".",
                      "The primary metrics retain all randomly selected tasks. A separate diagnostic restricted to reference-passing tasks is recorded in mean8_summary.json."])
    lines.extend(["", "| Correct out of 8 | Medium | Hard |", "|---:|---:|---:|"])
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
    base.queue = queue


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
