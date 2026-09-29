"""Evaluate eight independently seeded completions per Nemotron question."""

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from qwen3_experiments import competition_eval as base
from qwen3_experiments.taco_eval import sandbox_command

MODULE = "qwen3_experiments.nemotron_mean8_eval"


def final_code(response):
    from lcb_integration.extraction_utils import LMStyle, extract_code

    if "</think>" not in response:
        return "", "missing_thinking_close"
    final = response.rsplit("</think>", 1)[1].strip()
    if "<think>" in final:
        return "", "unclosed_thinking"
    code = extract_code(final, LMStyle.OpenAIChat)
    return code, "ok" if code and code.strip() else "empty_final_code"


def grade(plan, question, code):
    tests = base.read(question["unit_tests_file"])
    if not tests["inputs"] or len(tests["inputs"]) != len(tests["outputs"]):
        raise ValueError("Empty or unpaired test data")
    timeout = plan["unit_test_timeout_secs"]
    wall = (timeout + 1) * len(tests["inputs"]) + 20
    temp_root = Path(plan["scratch"]) / "grading_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        payload = Path(directory) / "input.json"
        base.write(payload, {"code": code, "unit_tests": tests, "timeout": timeout})
        process = subprocess.Popen(sandbox_command(plan, payload), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=wall)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise RuntimeError("Outer verifier failed to finish after its internal deadline") from exc
        if process.returncode:
            raise RuntimeError(f"Verifier infrastructure exit {process.returncode}: {stderr[-1500:]}")
        result = json.loads(stdout)
        if result.get("infrastructure_error"):
            raise RuntimeError(result["infrastructure_error"])
        outcomes = result["results"]
        if not outcomes or len(outcomes) > len(tests["inputs"]):
            raise ValueError("Invalid verifier result count")
        result.update(score=float(len(outcomes) == len(tests["inputs"]) and all(v == 1 for v in outcomes)),
                      seconds=time.monotonic() - started, total_tests=len(tests["inputs"]))
        return result


def aggregate(questions, grades, samples_per_question=8):
    """Mean of per-question sample accuracy; never substitute any-pass accuracy."""
    if not questions or len(questions) != len(grades):
        raise ValueError("Incomplete grades")
    grouped = defaultdict(list)
    for question, grade_record in zip(questions, grades):
        if grade_record["id"] != question["id"] or grade_record["score"] not in (0, 1):
            raise ValueError("Invalid grade identity or non-binary score")
        grouped[question["question_hash_id"]].append((question, grade_record))
    per_question = []
    for key, items in grouped.items():
        if len(items) != samples_per_question or {q["sample_index"] for q, _ in items} != set(range(samples_per_question)):
            raise ValueError("Each question needs exactly eight distinct sample indices")
        items.sort(key=lambda item: item[0]["sample_index"])
        question = items[0][0]
        correct = sum(g["score"] for _, g in items)
        per_question.append({"hash_id": key, "subset_index": question["subset_index"],
                             "original_source_index": question["original_source_index"],
                             "source": question["difficulty"], "correct_samples": correct,
                             "samples": samples_per_question, "mean_at_8_percent": 100 * correct / samples_per_question,
                             "mean_output_tokens": sum(g["tokens"] for _, g in items) / samples_per_question})
    correct = sum(q["correct_samples"] for q in per_question)
    return {"questions": len(per_question), "samples_per_question": samples_per_question,
            "total_samples": len(grades), "correct_samples": correct,
            "mean_at_8_percent": sum(q["mean_at_8_percent"] for q in per_question) / len(per_question),
            "correct_samples_histogram": dict(Counter(int(q["correct_samples"]) for q in per_question)),
            "mean_output_tokens": sum(g["tokens"] for g in grades) / len(grades),
            "per_question": per_question}


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
    summary = aggregate(questions, grades)
    summary["plan_sha256"] = base.digest(Path(plan["output_root"]) / "plan.json")
    summary["model"] = plan["models"]["baseline"]
    base.persist(plan, "mean8_summary.json", summary)
    with (Path(plan["output_root"]) / "per_question.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["per_question"][0]))
        writer.writeheader()
        writer.writerows(summary["per_question"])
    lines = ["# Nemotron coding: Qwen3-1.7B mean@8", "",
             f"Mean@8: **{summary['mean_at_8_percent']:.2f}%** "
             f"({summary['correct_samples']:.0f}/{summary['total_samples']} correct completions).", "",
             "100 questions sampled within the 3,200-question training subset; 8 independently seeded completions per question.",
             "This measures training-subset difficulty, not held-out generalization.",
             "Thinking on; 32768 output tokens; temperature/top-p/top-k 0.6/0.95/20.",
             "Original dataset messages; official pinned NeMo Gym code_gen verifier; grade only after thinking; no EOS gate.",
             "The verifier uses a 10-second timeout per unit test inside an isolated namespace.", "",
             f"Dataset: https://huggingface.co/datasets/{plan['subset']['repo_id']}", "",
             "| Correct out of 8 | Questions |", "|---:|---:|"]
    for correct in range(9):
        lines.append(f"| {correct} | {summary['correct_samples_histogram'].get(correct, 0)} |")
    for root in (Path(plan["output_root"]), Path(plan["scratch"]) / "control_mirrors"):
        try:
            (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
        except OSError:
            pass


def guard(plan):
    root, scratch = Path(plan["output_root"]), Path(plan["scratch"])
    children = []
    with (scratch / "guard/outer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (root / "cancellation.json").exists():
            try:
                children = [p for p in children if p.poll() is None]
                if base.state(plan, "queue_status.json").get("state") == "complete":
                    base.persist(plan, "watchdog_status.json", {"state": "complete", "pid": os.getpid()})
                    return
                free = {"home": shutil.disk_usage(root).free, "compute": shutil.disk_usage(scratch).free}
                if min(free.values()) < (2 << 30):
                    for reserve in (root / ".disk_reserve", scratch / "control_mirrors/.disk_reserve"):
                        reserve.unlink(missing_ok=True)
                supervisors = base.active(plan, ("supervise",))
                if not supervisors:
                    child = base.launch_child(plan, "supervise")
                    children.append(child)
                    supervisors = [child.pid]
                base.persist(plan, "watchdog_status.json", {"state": "guarding", "pid": os.getpid(),
                                                            "supervisor_pids": supervisors, "free_bytes": free})
            except Exception as exc:
                # The guard stays alive even when neither filesystem accepts a status write.
                for reserve in (root / ".disk_reserve", scratch / "control_mirrors/.disk_reserve"):
                    try:
                        reserve.unlink(missing_ok=True)
                    except OSError:
                        pass
                try:
                    base.persist(plan, "watchdog_status.json", {"state": "recovering", "error": str(exc)})
                except OSError:
                    pass
            time.sleep(30)


def configure(plan):
    sys.path.insert(0, str(Path(plan["grading"]["official"]) / "resources_servers/code_gen"))
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
    configure(plan)
    if args.command == "guard":
        base.verify(plan)
        return guard(plan)
    base.main()
    if args.command == "launch":
        child = base.launch_child(plan, "guard")
        base.persist(plan, "watchdog_launch.json", {"pid": child.pid, "node": plan["node"]})


if __name__ == "__main__":
    main()
