"""Async veRL reward adapter using the project's pinned, sandboxed LCB grader."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from qwen3_experiments.code_grading import final_code
from qwen3_experiments.lcb_coding_grading import LiveCodeBenchGrader


@lru_cache(maxsize=1)
def _service(grading_plan: str, workers_per_process: int):
    if type(workers_per_process) is not int or workers_per_process < 1:
        raise ValueError("workers_per_process must be a positive integer")
    grader = LiveCodeBenchGrader(json.loads(Path(grading_plan).read_text()))
    executor = ThreadPoolExecutor(max_workers=workers_per_process, thread_name_prefix="lcb")
    return grader, executor


async def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    *,
    grading_plan,
    workers_per_process=16,
):
    """Grade only code after </think>, with no EOS requirement or wrong-answer retry.

    Eight veRL reward processes with sixteen executor threads each cap sandbox
    concurrency at 128. Infrastructure failures propagate to the trainer after
    the existing grader's single infrastructure retry.
    """
    code, reason = final_code(solution_str)
    if reason != "ok":
        result = {"score": 0.0, "reason": reason, "seconds": 0.0, "executed_tests": 0}
    else:
        grader, executor = _service(str(grading_plan), workers_per_process)
        result = await asyncio.get_running_loop().run_in_executor(executor, grader, ground_truth, code)
    return {
        "score": result["score"],
        "acc": result["score"],
        "accuracy": result["score"],
        "grading_seconds": result["seconds"],
        "grading_reason": result["reason"],
        "grading_executed_tests": result["executed_tests"],
    }
