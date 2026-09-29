"""Run pinned competition graders in a separate, credential-free namespace."""

import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

from qwen3_experiments.taco_eval import sandbox_command


def final_code(response):
    if "</think>" not in response:
        return "", "missing_thinking_close"
    final = response.rsplit("</think>", 1)[1].strip()
    if "<think>" in final:
        return "", "unclosed_thinking"
    if not final:
        return "", "empty_final"
    blocks = re.findall(r"```(?:python3?|py)?[ \t]*\n(.*?)```", final, re.S | re.I)
    code = (blocks[-1] if blocks else final).strip()
    return code, "ok" if code else "empty_code"


def grade(plan, input_output, code, kind="taco"):
    tests = json.loads(input_output)
    if not tests.get("inputs") or len(tests["inputs"]) != len(tests["outputs"]):
        raise ValueError("Empty or unpaired tests")
    wall = max(90, 24 * len(tests["inputs"]) + 30) if kind == "taco" else 7 * len(tests["inputs"]) + 5
    settings = {**plan, "official": plan["official_taco" if kind == "taco" else "official_lcb"]}
    temp_root = Path(plan["scratch"]) / "grading_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        payload = Path(directory) / "input.json"
        payload.write_text(json.dumps({"kind": kind, "input_output": input_output,
                                       "code": code, "wall_timeout": wall}))
        try:
            result = subprocess.run(sandbox_command(settings, payload), capture_output=True,
                                    text=True, timeout=wall + 10, start_new_session=True)
        except subprocess.TimeoutExpired:
            record = {"results": [-1], "error": "outer_wall_timeout"}
        else:
            if result.returncode and ("bwrap:" in result.stderr or "Traceback" in result.stderr):
                raise RuntimeError(f"Grader infrastructure failed: {result.stderr[-2000:]}")
            if result.returncode or not result.stdout.strip():
                record = {"results": [-2], "error": f"grader_exit_{result.returncode}"}
            else:
                record = json.loads(result.stdout)
    outcomes = record["results"]
    # Official graders use negative integers for timeout/compile/runtime errors.
    record.update(score=float(bool(outcomes) and all(value > 0 for value in outcomes)),
                  seconds=time.monotonic() - started)
    return record
