"""Shared binary training/evaluation interface to an unchanged LCB grader."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from qwen3_experiments.lcb_coding_format import LCB_TESTING_SHA256, prepare_code, validate_truth
from qwen3_experiments.taco_eval import sandbox_command


class GradingInfrastructureError(RuntimeError):
    """A failed grader must not silently become a wrong model answer."""


class LiveCodeBenchGrader:
    def __init__(self, plan):
        self.plan = dict(plan)
        source = Path(plan["official"]) / "lcb_runner/evaluation/testing_util.py"
        if hashlib.sha256(source.read_bytes()).hexdigest() != LCB_TESTING_SHA256:
            raise ValueError("LiveCodeBench source does not match the pinned version")
        runner = Path(plan["sandbox_runner"])
        if hashlib.sha256(runner.read_bytes()).hexdigest() != plan["sandbox_runner_sha256"]:
            raise ValueError("Sandbox runner differs from its preparation receipt")
        self.temp_root = Path(plan["scratch"]) / "lcb_grading_tmp"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def __call__(self, truth, code):
        if isinstance(truth, str):
            truth = json.loads(truth)
        validate_truth(truth)
        started = time.monotonic()
        last_error = None
        for _ in range(2):
            try:
                result = self._run(truth, code)
                result["seconds"] = time.monotonic() - started
                return result
            except GradingInfrastructureError as exc:
                last_error = exc
        raise last_error

    def _run(self, truth, code):
        tests = truth["input_output"]
        timeout = truth["unit_test_timeout_seconds"]
        wall = (timeout + 1) * len(tests["inputs"]) + 30
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            payload = Path(directory) / "input.json"
            payload.write_text(json.dumps({"input_output": tests, "code": prepare_code(code, truth),
                                           "timeout": timeout}))
            process = subprocess.Popen(sandbox_command(self.plan, payload), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=wall)
            except subprocess.TimeoutExpired as exc:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise GradingInfrastructureError("LCB runner exceeded its outer deadline") from exc
            if process.returncode:
                raise GradingInfrastructureError(f"LCB sandbox exit {process.returncode}: {stderr[-1500:]}")
            try:
                result = json.loads(stdout)
            except ValueError as exc:
                raise GradingInfrastructureError("LCB runner returned invalid JSON") from exc
            if result.get("infrastructure_error"):
                raise GradingInfrastructureError(result["infrastructure_error"])
        outcomes = result.get("results")
        if (not isinstance(outcomes, list) or not outcomes or len(outcomes) > len(tests["inputs"])
                or any(type(v) is not int or v not in (-4, -3, -2, -1, 0, 1) for v in outcomes)):
            raise GradingInfrastructureError("Invalid LCB result vector")
        if all(v == 1 for v in outcomes) and len(outcomes) != len(tests["inputs"]):
            raise GradingInfrastructureError("LCB returned an incomplete successful result")
        result.update(score=float(all(v == 1 for v in outcomes)), total_tests=len(tests["inputs"]),
                      executed_tests=len(outcomes), reason="livecodebench")
        return result
