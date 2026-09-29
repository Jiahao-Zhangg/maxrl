"""CPU-only reference, negative-control, and saved-completion migration checks."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from qwen3_experiments.code_grading import final_code
from qwen3_experiments.lcb_coding_format import convert_truth, validate_truth
from qwen3_experiments.lcb_coding_grading import LiveCodeBenchGrader


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(config):
    import pyarrow.parquet as pq

    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    rows = pq.read_table(config["dataset"]).to_pylist()
    truths = {r["id"]: validate_truth(json.loads(r["reward_model"]["ground_truth"])) for r in rows}
    assert len(rows) == len(truths) == 3200
    grader = LiveCodeBenchGrader(read(config["grading_plan"]))
    status = {"config_sha256": config_hash, "parquet_sha256": digest(config["dataset"])}

    def update(**values):
        status.update(values, updated_at=time.time())
        save(root / "status.json", status)
        print(json.dumps(status), flush=True)

    def run_stage(name, tasks, function):
        folder = root / name
        folder.mkdir(exist_ok=True)
        results = []
        pending = []
        for task in tasks:
            path = folder / f"{task['key']}.json"
            if path.exists():
                result = read(path)
                if result["config_sha256"] != config_hash:
                    raise ValueError("Cannot reuse verification from a different configuration")
                results.append(result)
            else:
                pending.append(task)
        update(state="running", stage=name, done=len(results), total=len(tasks))
        def process(task):
            result = {**function(task), "key": task["key"], "config_sha256": config_hash}
            save(folder / f"{task['key']}.json", result)
            return result
        with ThreadPoolExecutor(max_workers=config.get("workers", 8)) as pool:
            futures = [pool.submit(process, task) for task in pending]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 50 == 0 or len(results) == len(tasks):
                    update(done=len(results))
        return results

    # Exercise both native branches, errors, fail-fast, and the timeout path.
    stdin = convert_truth({"grader": "nemo_gym_code_gen",
                           "unit_tests": {"inputs": ["1\n", "2\n"], "outputs": ["2\n", "3\n"]}})
    functional = {**stdin, "input_output": {"inputs": ["1\n2", "3\n4"], "outputs": ["3", "7"], "fn_name": "add"}}
    slow = {**stdin, "unit_test_timeout_seconds": 1}
    probes = [
        ("stdin_pass", stdin, "print(int(input())+1)", 1),
        ("stdin_fail_fast", stdin, "x=int(input())\nif x==2:\n    while True: pass\nprint(0)", 0),
        ("function_pass", functional, "class Solution:\n    def add(self,a,b): return a+b", 1),
        ("function_fail", functional, "class Solution:\n    def add(self,a,b): return a-b", 0),
        ("syntax_error", stdin, "def broken(", 0),
        ("runtime_error", stdin, "raise RuntimeError('negative control')", 0),
        ("timeout", slow, "while True: pass", 0),
    ]
    smoke = []
    for name, truth, code, expected in probes:
        result = grader(truth, code)
        assert result["score"] == expected, (name, result)
        if name in {"stdin_fail_fast", "function_fail"}:
            assert result["executed_tests"] == 1
        smoke.append({"name": name, **result})
    save(root / "smoke.json", smoke)

    leetcode_rows = {}
    selected = {r["extra_info"]["problem_id"]: r["id"] for r in rows if r["extra_info"]["platform"] == "leetcode"}
    with Path(config["leetcode_source"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["task_id"] in selected:
                leetcode_rows[selected[row["task_id"]]] = row
    assert len(leetcode_rows) == 590
    references = run_stage("references", [{"key": key} for key in leetcode_rows],
                           lambda task: grader(truths[task["key"]], leetcode_rows[task["key"]]["completion"]))
    failed_references = [r for r in references if r["score"] != 1]
    if failed_references:
        save(root / "reference_failures.json", failed_references)
        raise ValueError(f"{len(failed_references)} reference solutions failed LCB")
    negatives = run_stage("negative_controls", [{"key": r["id"]} for r in rows],
                         lambda task: grader(truths[task["key"]], "raise RuntimeError('intentional negative control')"))
    assert all(r["score"] == 0 for r in negatives)

    cached_tasks = []
    for previous in config.get("cached_evaluations", []):
        questions = read(previous["questions"])
        folder = Path(previous["folder"])
        lookup = {r["extra_info"]["problem_id"] if previous["kind"] == "leetcode" else r["extra_info"]["source_hash_id"]: r["id"]
                  for r in rows if (r["extra_info"]["platform"] == "leetcode") == (previous["kind"] == "leetcode")}
        for question in questions:
            identity = question["task_id"] if previous["kind"] == "leetcode" else question["question_hash_id"]
            if identity not in lookup:
                continue
            index = str(question["source_index"])
            cached_tasks.append({"key": previous["kind"] + "_" + index, "row_id": lookup[identity],
                                 "question": question, "response_path": str(folder / "responses" / f"{index}.json"),
                                 "grade_path": str(folder / "grades" / f"{index}.json"), "kind": previous["kind"]})

    def regrade(task):
        response, old = read(task["response_path"]), read(task["grade_path"])
        if response["id"] != task["question"]["id"] or old["id"] != response["id"]:
            raise ValueError("Saved response identity mismatch")
        response_hash = digest(task["response_path"])
        if response_hash != old["response_sha256"]:
            raise ValueError("Saved response changed after its original grading")
        code, reason = final_code(response["response"])
        result = grader(truths[task["row_id"]], code) if reason == "ok" else {"score": 0, "reason": reason}
        return {"row_id": task["row_id"], "kind": task["kind"], "old_score": old["score"],
                "changed": old["score"] != result["score"], "response_sha256": response_hash, **result}

    cached = run_stage("saved_completions", cached_tasks, regrade)
    changes = [r for r in cached if r["changed"]]
    summary = {"passed": True, **status, "state": "complete", "rows": len(rows),
               "smoke_checks": len(smoke), "reference_solutions": len(references), "references_passed": len(references),
               "negative_controls": len(negatives), "negative_controls_rejected": len(negatives),
               "saved_completions": len(cached), "saved_completions_by_source": dict(Counter(r["kind"] for r in cached)),
               "saved_completion_score_changes": len(changes), "score_changes": changes,
               "grader": "unchanged pinned LiveCodeBench run_test", "per_test_timeout_seconds": 10,
               "check_eos": False, "score_after_thinking": True,
               "test_cases": sum(len(t["input_output"]["inputs"]) for t in truths.values())}
    save(root / "verification.json", summary)
    update(state="complete", stage="complete", summary="verification.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    try:
        verify(config)
    except Exception as exc:
        save(Path(config["output_root"]) / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
