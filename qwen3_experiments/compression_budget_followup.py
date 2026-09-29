"""Append pinned compression-model budget evaluations to one compute allocation."""

import argparse
import csv
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from qwen3_experiments import compute_disk_watchdog as storage
from qwen3_experiments import minerva_individual_budget as mini

MODULE = "qwen3_experiments.compression_budget_followup"
read, write, digest, now = storage.read, storage.write, storage.digest, storage.now


def verify_suite(root):
    plan = read(root / "plan.json")
    if root.resolve() != Path(plan["output_root"]).resolve():
        raise ValueError("Queue output path differs from its pinned plan")
    for relative, checksum in plan["frozen_files"].items():
        if digest(root / relative) != checksum:
            raise ValueError(f"Changed follow-up input: {relative}")
    if mini.require_compute(plan["job_id"]) != plan["node"]:
        raise ValueError("Follow-up is restricted to its original compute node")
    return plan


def dependency_ready(plan):
    expected = plan["dependency"]
    previous = Path(expected["root"])
    if digest(previous / "plan.json") != expected["plan_sha256"]:
        raise ValueError("Predecessor plan changed")
    names = ["status.json", "report/audit.json"]
    if expected.get("queue_required"):
        names.append("queue_status.json")
    records = {name: read(previous / name) if (previous / name).exists() else {} for name in names}
    if any(record.get("state") in ("failed", "blocked_by_training_failure") for record in records.values()):
        raise RuntimeError("Predecessor failed; preserving its outputs and GPU ownership")
    if any(records[name].get("state") != "complete" for name in names if name != "report/audit.json"):
        return False
    audit = records["report/audit.json"]
    if not audit:
        return False
    if (not audit.get("complete") or audit.get("points") != expected["points"]
            or not audit.get("all_rollout_ledgers_verified")
            or digest(previous / "report/metrics.json") != audit["metrics_sha256"]):
        raise ValueError("Predecessor does not have the complete expected budget audit")
    return True


def protected_writer(root, scratch, original):
    def persist(path, value):
        relative = Path(path).relative_to(root)
        backup = Path(scratch) / "control_mirrors" / relative
        while True:
            for target in (backup, Path(path)):
                try:
                    original(target, value)
                except OSError as exc:
                    if exc.errno not in (errno.EDQUOT, errno.ENOSPC):
                        raise
                else:
                    if target == Path(path):
                        return
            time.sleep(30)
    return persist


def staged_manifest(root, plan):
    manifest = read(root / "execution_manifest.json")
    if manifest["plan_sha256"] != digest(root / "plan.json") or set(manifest["models"]) != set(plan["models"]):
        raise ValueError("Prepared models belong to a different plan")
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if mini.fingerprint(payload) != manifest["fingerprint"]:
        raise ValueError("Prepared model manifest changed")
    for key, model in manifest["models"].items():
        if any(model[field] != plan["models"][key][field] for field in ("repo", "revision", "path")):
            raise ValueError("Prepared model identity differs from the requested checkpoint")
        for name, checksum in model["files_sha256"].items():
            if digest(Path(model["path"]) / name) != checksum:
                raise ValueError(f"Prepared model changed: {key}/{name}")
    return manifest


def stage_complete(root, expected_points):
    status = read(root / "status.json") if (root / "status.json").exists() else {}
    if status.get("state") != "complete":
        return False
    audit = read(root / "report/audit.json")
    if (not audit.get("complete") or audit.get("points") != expected_points
            or not audit.get("all_rollout_ledgers_verified")
            or digest(root / "report/metrics.json") != audit["metrics_sha256"]):
        raise ValueError("Completed stage has an invalid result audit")
    return True


def active_stage_processes(plan, stage):
    needles = {str(stage), str(stage.resolve())}
    result = []
    for pid in storage.allocation_processes(plan):
        item = storage.identity(pid)
        if item is None or pid == os.getpid() or item["uid"] != os.getuid():
            continue
        if any(arg in needles for arg in item["command"]):
            result.append(item)
    return result


def cleanup_stage(suite_root, suite, stage):
    root = Path(stage["root"])
    receipt_path = suite_root / "cache_cleanup" / f"{stage['key']}.json"
    if receipt_path.exists() and read(receipt_path).get("state") == "deleted":
        return
    if not stage_complete(root, stage["points"]) or active_stage_processes(suite, root):
        return
    plan = mini.verify_plan(root)
    cache = Path(stage["owned_model_cache"])
    if cache.is_symlink() or cache.resolve() != Path(suite["scratch"]).resolve() / "models" / stage["key"]:
        raise ValueError("Cleanup path is outside this queue's private model cache")
    for model in plan["models"].values():
        Path(model["path"]).resolve().relative_to(cache.resolve())
    # Recheck completed point artifacts before discarding reconstructable weights.
    manifest = read(root / "execution_manifest.json")
    for key, budget, dataset in mini.evaluation_points(plan):
        if mini.completed_point(root, key, budget, manifest, dataset) is None:
            raise ValueError("Cannot clean a model before all its point artifacts are verified")
    size = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()) if cache.exists() else 0
    write(receipt_path, {"state": "deleting_verified_unused_cache", "path": str(cache), "bytes": size,
                         "model_identity": stage["model"], "updated_at": now()})
    if cache.exists():
        shutil.rmtree(cache)
    write(receipt_path, {"state": "deleted", "path": str(cache), "bytes": size,
                         "model_identity": stage["model"], "updated_at": now()})


def combined_report(root, plan):
    rows = read(root / "reused_minerva.json")["metrics"]
    for stage in plan["stages"]:
        folder = Path(stage["root"])
        if not stage_complete(folder, stage["points"]):
            raise ValueError("Cannot report an incomplete evaluation stage")
        rows.extend(read(folder / "report/metrics.json"))
    identities = {(row["dataset"], row["model_key"], row["budget_tokens"]) for row in rows}
    expected = {(dataset, stage["key"], budget) for dataset in plan["all_datasets"]
                for stage in plan["stages"] for budget in plan["budgets"]}
    if identities != expected or len(rows) != len(expected):
        raise ValueError("Combined results contain missing or duplicate model/dataset/budget points")
    order = {key: index for index, key in enumerate(plan["all_datasets"])}
    rows.sort(key=lambda row: (order[row["dataset"]], row["model_key"], row["budget_tokens"]))
    report = root / "report"
    report.mkdir(exist_ok=True)
    write(report / "metrics.json", rows)
    fields = ["dataset", "model_key", "model", "budget_tokens", "pass_at_budget_percent", "questions_solved",
              "questions", "mean_tokens_per_question", "seed", "source"]
    with (report / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Compression models: individual pass@budget", "",
             "Thinking on; temperature/top-p/top-k = 0.6/0.95/20; seed 0; after-thinking grading only; "
             "no extra EOS requirement. Each question retains its own budget and stops at its first correct answer. "
             "The per-response cap is 32,768 tokens. Existing audited Minerva results are reused.", "",
             "| Dataset | Model | Budget | Pass@budget (%) | Solved |",
             "|---|---|---:|---:|---:|"]
    lines += [f"| {row['dataset']} | {row['model']} | {row['budget_tokens']} | "
              f"{row['pass_at_budget_percent']:.2f} | {row['questions_solved']}/{row['questions']} |" for row in rows]
    (report / "README.md").write_text("\n".join(lines) + "\n")
    write(report / "audit.json", {"complete": True, "points": len(rows), "new_points": plan["new_points"],
                                  "reused_minerva_points": len(read(root / "reused_minerva.json")["metrics"]),
                                  "all_rollout_ledgers_verified": True, "metrics_sha256": digest(report / "metrics.json")})


def queue(root):
    plan = verify_suite(root)
    persist = protected_writer(root, plan["scratch"], write)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children = {}
        while True:
            verify_suite(root)
            state = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": plan["job_id"],
                     "updated_at": now(), "new_points": plan["new_points"], "completed_points": 0,
                     "disks": {name: storage.disk(path) for name, path in plan["filesystems"].items()}}
            try:
                for stage in plan["stages"]:
                    folder = Path(stage["root"])
                    if stage_complete(folder, stage["points"]):
                        state["completed_points"] += stage["points"]
                        cleanup_stage(root, plan, stage)
                        continue
                    child = children.get(stage["key"])
                    if child is not None and child.poll() is not None:
                        state["last_exit_code"] = child.returncode
                        children.pop(stage["key"])
                    stage_plan = mini.verify_plan(folder)
                    state.update(current_model=stage["key"], stage_root=str(folder))
                    if not dependency_ready(stage_plan):
                        state.update(state="waiting_for_predecessor", predecessor=stage_plan["dependency"]["root"])
                        break
                    active = active_stage_processes(plan, folder)
                    if not active:
                        command = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                                   f"--nodelist={plan['node']}", "--cpus-per-task=96", "--gres=gpu:8",
                                   "--kill-on-bad-exit=1", "--job-name=compression-budget-followup",
                                   plan["python_bin"], "-u", "-m", MODULE, "run", "--output-root", str(folder)]
                        env = mini.environment(stage_plan)
                        env["HF_HOME"] = plan["hf_home"]
                        with (folder / "launch.log").open("ab", buffering=0) as log:
                            child = subprocess.Popen(command, cwd=plan["runtime"], env=env, stdin=subprocess.DEVNULL,
                                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        children[stage["key"]] = child
                        persist(folder / "queue_launch.json", {"pid": child.pid, "identity": storage.identity(child.pid),
                                                               "command": command, "launched_at": now()})
                        active = [{"pid": child.pid}]
                    progress = read(folder / "status.json") if (folder / "status.json").exists() else {}
                    state["completed_points"] += progress.get("completed_points", 0)
                    state.update(state="running_or_waiting_for_gpus", active_pids=[item["pid"] for item in active])
                    break
                else:
                    combined_report(root, plan)
                    state.update(state="complete", finished_at=now(), report=str(root / "report/README.md"))
                persist(root / "queue_status.json", state)
                if state["state"] == "complete":
                    return 0
            except Exception as exc:
                persist(root / "queue_status.json", {**state, "state": "retrying", "error": str(exc)})
                storage.safe_print(f"{now()} follow-up queue will retry: {exc}")
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("queue", "run", "worker"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--budget", type=int, choices=mini.BUDGETS)
    parser.add_argument("--dataset", choices=mini.DATASETS)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.command == "queue":
        return queue(root)
    plan = mini.verify_plan(root)
    if mini.require_compute(plan["job_id"]) != plan["node"]:
        raise ValueError("Evaluation invoked on a different compute node")
    mini.MODULE = MODULE
    mini.dependency_ready = dependency_ready
    mini.prepare_models = staged_manifest
    mini.write = protected_writer(root, plan["scratch"], write)
    if args.command == "worker":
        if args.model not in plan["models"] or args.rank is None or args.dataset is None or args.budget is None:
            parser.error("Worker requires a pinned model, dataset, rank and budget")
        return mini.worker(root, args.model, args.rank, args.budget, args.dataset)
    return mini.run(root, plan)


if __name__ == "__main__":
    raise SystemExit(main())
