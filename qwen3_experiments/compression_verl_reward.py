"""Keep historical compression MathVerify scoring in isolated Python processes."""

import argparse
import asyncio
import atexit
import contextlib
import json
import select
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path


def response_after_thinking(text):
    """Match the historical compression reward manager's extraction exactly."""
    if "</think>" not in text:
        return None
    answer = text.rsplit("</think>", 1)[1].strip()
    if not answer or "<think>" in answer:
        return None
    return answer


class MathWorker:
    def __init__(self, grading_plan):
        self.grading_plan = str(grading_plan)
        self.child = None
        self.lock = threading.Lock()
        atexit.register(self.close)

    def close(self):
        if self.child is not None:
            child, self.child = self.child, None
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            child.stdin.close()
            child.stdout.close()

    def _read(self, timeout):
        if not select.select([self.child.stdout], [], [], timeout)[0]:
            self.close()
            raise RuntimeError("Compression math worker did not respond")
        line = self.child.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError("Compression math worker exited")
        return json.loads(line)

    def score(self, answer, ground_truth):
        with self.lock:
            if self.child is None or self.child.poll() is not None:
                self.close()
                self.child = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()),
                     "--grading-plan", self.grading_plan],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                )
                if self._read(60).get("state") != "ready":
                    raise RuntimeError("Compression math worker initialization failed")
            self.child.stdin.write(json.dumps({"answer": answer, "ground_truth": ground_truth}) + "\n")
            self.child.stdin.flush()
            return self._read(15)


@lru_cache(maxsize=1)
def service(grading_plan):
    return MathWorker(grading_plan)


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, *, grading_plan):
    answer = response_after_thinking(solution_str)
    if answer is None:
        result = {"score": 0.0, "grading_reason": "missing_or_invalid_thinking_close", "grading_seconds": 0.0}
    else:
        result = await asyncio.to_thread(service(str(grading_plan)).score, answer, ground_truth)
    return {**result, "acc": result["score"], "accuracy": result["score"]}


def worker(grading_plan):
    # Only this subprocess sees the old symbolic-math dependencies. The veRL,
    # torch and vLLM environment remains untouched, including its newer SymPy.
    import hashlib
    import importlib.metadata
    import importlib.util

    plan = json.loads(Path(grading_plan).read_text())
    sys.path.insert(0, plan["site_packages"])
    for name, expected in plan["versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Historical math dependency changed: {name}")
    path = Path(plan["scorer_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != plan["scorer_sha256"]:
        raise ValueError("Historical math scorer changed")
    spec = importlib.util.spec_from_file_location("frozen_compression_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = module.MathVerifyScorer()
    print(json.dumps({"state": "ready", "versions": plan["versions"]}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        start = time.monotonic()
        with contextlib.redirect_stdout(sys.stderr):
            score = scorer.compute_score(request["answer"], request["ground_truth"], 0.0,
                                         plan["per_item_timeout_seconds"])
        print(json.dumps({"score": float(score), "grading_reason": "math_verify",
                          "grading_seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grading-plan", required=True)
    worker(parser.parse_args().grading_plan)
