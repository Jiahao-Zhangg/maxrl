"""Insert one audited TACO stage without changing frozen math-evaluation inputs."""

import argparse
from pathlib import Path

from qwen3_experiments import compression_budget_followup as followup
from qwen3_experiments import taco_eval as taco


def gated_dependency(plan, taco_root, blocked_root, original):
    if Path(plan["output_root"]).resolve() == blocked_root.resolve() and not taco.complete(taco_root):
        return False
    return original(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    plan = taco.verify(root)
    budget = Path(plan["budget_root"])
    if taco.digest(budget / "plan.json") != plan["budget_plan_sha256"]:
        raise ValueError("Original budget queue plan changed")
    dependency = followup.dependency_ready
    blocked = Path(plan["blocked_stage_root"])
    followup.dependency_ready = lambda stage: gated_dependency(stage, root, blocked, dependency)
    original_writer = followup.protected_writer

    def annotated_writer(*args):
        persist = original_writer(*args)

        def write(path, value):
            if (Path(path) == budget / "queue_status.json" and value.get("state") == "waiting_for_predecessor"
                    and value.get("stage_root") == str(blocked) and not taco.complete(root)):
                value = {**value, "predecessor": str(root), "priority_insert": "TACO Easy100 + Medium100; seqs16/32"}
            return persist(path, value)

        return write

    followup.protected_writer = annotated_writer
    return followup.queue(budget)


if __name__ == "__main__":
    raise SystemExit(main())
