"""Isolated adapters for USACOBench and the CodeContests output checker."""

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from qwen3_experiments.taco_eval import sandbox_command


def grade(plan, question, code):
    """Fail loudly on infrastructure errors instead of recording an incorrect answer."""
    kind = question["dataset"]
    if kind not in ("usaco", "code_contests"):
        raise ValueError(f"Unsupported competition: {kind}")
    tests = question["tests"]
    if not tests:
        raise ValueError("A question must have tests")
    if kind == "code_contests" and any(set(test) != {"input", "output", "group"} for test in tests):
        raise ValueError("Unpaired CodeContests test")
    wall = int(len(tests) * (question["runtime_limit"] * (30 if kind == "code_contests" else 1) + 3) + 60)
    root = Path(plan["scratch"]) / "grading_tmp"
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=root) as directory:
        payload = Path(directory) / "input.json"
        payload.write_text(json.dumps({"question": question, "code": code, "wall_timeout": wall}))
        command = sandbox_command(plan, payload)
        command[1:1] = ["--ro-bind", plan["usaco_tests"], "/tests"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=wall)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise RuntimeError("Competition grader exceeded its outer wall limit") from exc
        if process.returncode:
            raise RuntimeError(f"Competition sandbox exited {process.returncode}: {stderr[-2000:]}")
        record = json.loads(stdout)
        if record.get("infrastructure_error"):
            raise RuntimeError(record["infrastructure_error"])
        values = record["results"]
        if not values or len(values) > len(tests):
            raise ValueError("Invalid grader result count")
        record.update(score=float(len(values) == len(tests) and all(v == 1 for v in values)),
                      seconds=time.monotonic() - started, total_tests=len(tests))
        return record
