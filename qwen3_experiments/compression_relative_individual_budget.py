"""Run individual budgets scaled by base-model mean length after cross-budget evaluation."""

import argparse
import csv
import fcntl
import json
import os
import shutil
import socket
import subprocess
import time
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

from qwen3_experiments import compression_cross_budget as cross
from qwen3_experiments import compute_disk_watchdog as storage
from qwen3_experiments import minerva_individual_budget as mini

MODULE = "qwen3_experiments.compression_relative_individual_budget"
DATASETS = ("minervamath", "olympiadbench", "amc22_23", "aime24", "aime25", "aime26")
MODELS = ("base", "maxrl", "er", "l0", "l4096", "f_cov")
MULTIPLIERS = ("0.5", "1", "2", "3")
POINTS_PER_DATASET = len(MODELS) * len(MULTIPLIERS)
POINTS = len(DATASETS) * POINTS_PER_DATASET
read, write, digest, now = storage.read, storage.write, storage.digest, storage.now


def budget_grid(reference):
    audit = read(reference / "audit.json")
    if not audit.get("complete") or not audit.get("after_thinking_only") or audit["model_repo"] != "Qwen/Qwen3-1.7B":
        raise ValueError("Expected the audited base-model after-thinking reference")
    for name, checksum in audit["files_sha256"].items():
        if digest(reference / name) != checksum:
            raise ValueError(f"Changed base reference: {name}")
    with (reference / "main.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream)
                if r["Model"] == "qwen3-1.7B" and r["Metric"] == "Mean Response Length (tokens)"]
    if len(rows) != 1:
        raise ValueError("Expected one base mean-response-length row")
    result = {}
    for dataset in DATASETS:
        mean = Decimal(rows[0][mini.DATASETS[dataset][0]].replace(",", ""))
        if not mean.is_finite() or mean <= 0:
            raise ValueError("Mean response lengths must be positive and finite")
        budgets = [int((mean * Decimal(m)).to_integral_value(rounding=ROUND_CEILING)) for m in MULTIPLIERS]
        if len(set(budgets)) != len(MULTIPLIERS):
            raise ValueError("Relative budgets must be distinct")
        result[dataset] = {"base_mean_response_length": str(mean), "multipliers": list(MULTIPLIERS),
                           "budgets": budgets, "rounding": "ceil_to_integer_output_token"}
    return result


def verify_suite(root):
    plan = read(root / "plan.json")
    storage.require_node(plan)
    if root != Path(plan["output_root"]) or plan["dataset_order"] != list(DATASETS):
        raise ValueError("Relative-budget queue location or dataset order changed")
    if list(plan["models"]) != list(MODELS) or plan["points"] != POINTS or plan["protocol"] != "eval2":
        raise ValueError("Expected six models and 144 individual-budget points")
    if (root / "launch.json").exists() and read(root / "launch.json")["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Launched relative-budget plan changed")
    for relative, checksum in plan["frozen_files"].items():
        if digest(root / relative) != checksum:
            raise ValueError(f"Frozen relative-budget input changed: {relative}")
    if budget_grid(root / "base_reference") != plan["budget_grid"]:
        raise ValueError("Per-dataset budgets do not match the pinned base mean lengths")
    return plan


def predecessor_ready(plan):
    previous = Path(plan["predecessor_root"])
    if digest(previous / "plan.json") != plan["predecessor_plan_sha256"]:
        raise ValueError("Predecessor cross-budget plan changed")
    if digest(previous / "schedule.json") != plan["predecessor_schedule_sha256"]:
        raise ValueError("Predecessor cross-budget schedule changed")
    if read(previous / "schedule.json")["dataset_order"] != list(DATASETS):
        raise ValueError("Expected the six-dataset cross-budget predecessor")
    state = previous / "queue_status.json"
    if not state.exists() or read(state).get("state") != "complete":
        return False
    audit = read(previous / "report/audit.json")
    if (not audit.get("complete") or audit.get("points") != 150 or audit.get("datasets") != list(DATASETS)
            or not audit.get("all_rollout_ledgers_verified")
            or digest(previous / "report/metrics.json") != audit["metrics_sha256"]):
        raise ValueError("The entire current cross-budget queue must finish and pass its audit")
    if not all(cross.stage_complete(previous / "datasets" / dataset) for dataset in DATASETS):
        raise ValueError("Cross-budget dataset audits and model-cache cleanup must finish")
    return True


def stage_complete(stage):
    if not (stage / "status.json").exists() or read(stage / "status.json").get("state") != "complete":
        return False
    audit = read(stage / "report/audit.json")
    if (not audit.get("complete") or audit.get("points") != POINTS_PER_DATASET
            or not audit.get("all_rollout_ledgers_verified")
            or digest(stage / "report/metrics.json") != audit["metrics_sha256"]):
        raise ValueError("Dataset needs all 24 verified individual-budget results")
    return all((stage / "cache_cleanup" / f"{key}.json").exists()
               and read(stage / "cache_cleanup" / f"{key}.json").get("state") == "complete" for key in MODELS)


def next_dataset(root):
    return next((key for key in DATASETS if not stage_complete(root / "datasets" / key)), None)


def stage_dependency(plan):
    root = Path(plan["suite_root"])
    suite = verify_suite(root)
    return predecessor_ready(suite) and next_dataset(root) == plan["dataset_key"]


def prepare(args):
    from transformers import AutoTokenizer

    previous = args.after_cross.resolve()
    parent = cross.verify_suite(previous)
    if cross.scheduling(previous, parent)["dataset_order"] != list(DATASETS):
        raise ValueError("This follow-up expects the current six-dataset cross-budget schedule")
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=source, text=True).strip() != "agent/add-math12k-maxrl-launcher":
        raise ValueError("Use the primary maxrl launcher branch")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        return verify_suite(root)
    scratch = args.scratch
    if not scratch.is_absolute() or scratch.parent != Path("/tmp") or scratch.is_symlink():
        raise ValueError("Use a dedicated compute-local scratch directory")
    scratch.mkdir(parents=True, exist_ok=True)
    reference = args.base_reference.resolve()
    grid = budget_grid(reference)
    shutil.copytree(reference, root / "base_reference", dirs_exist_ok=True)
    base_plan_path = args.base_evaluation.resolve() / "plan.json"
    base = read(base_plan_path)["models"]["qwen3_1_7b"]
    audit = read(reference / "audit.json")
    if base["repo"] != audit["model_repo"] or base["revision"] != audit["model_revision"]:
        raise ValueError("Base checkpoint differs from the mean-length reference")
    write(root / "base_model_source.json", {"plan": str(base_plan_path), "plan_sha256": digest(base_plan_path), "model": base})
    models = {}
    for key in MODELS:
        model = base if key == "base" else parent["models"][key]
        if key != "base" and (not model["repo"].endswith("-final") or "compression" not in model["repo"]):
            raise ValueError("Use the five compression final checkpoints")
        models[key] = {**model, "format": "huggingface", "path": str(scratch / "models" / key),
                       "label": "Qwen3-1.7B (base)" if key == "base" else model["label"],
                       "family": "qwen3", "questions_file": "questions.json"}
    original = read(Path(parent["predecessor_root"]) / "plan.json")
    tokenizer = AutoTokenizer.from_pretrained(original["base_model"], local_files_only=True)
    runtime = root / "runtime/qwen3_experiments"
    runtime.mkdir(parents=True, exist_ok=True)
    scripts = ("compression_relative_individual_budget.py", "compression_cross_budget.py", "minerva_individual_budget.py",
               "math_eval_budget_engine.py", "math_eval_matrix_common.py", "prepare_math_eval_matrix.py",
               "grpo_compute_control.py", "compute_disk_watchdog.py")
    for name in scripts:
        shutil.copy2(source / "qwen3_experiments" / name, runtime / name)
    fixed = [*runtime.glob("*.py"), *(root / "base_reference").glob("*"), root / "base_model_source.json"]
    for dataset in DATASETS:
        old_stage = previous / "datasets" / dataset
        old = mini.verify_plan(old_stage)
        rows = read(old_stage / "questions.json")
        if len(rows) != mini.DATASETS[dataset][1] or mini.thinking_rows(rows, tokenizer, "qwen3") != rows:
            raise ValueError("Frozen dataset or native thinking prompts changed")
        stage = root / "datasets" / dataset
        stage.mkdir(parents=True, exist_ok=True)
        write(stage / "questions.json", rows)
        (stage / "runtime").symlink_to(root / "runtime", target_is_directory=True)
        shutil.copytree(old_stage / "provenance", stage / "provenance", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        frozen = {"questions.json": digest(stage / "questions.json")}
        frozen.update({f"runtime/qwen3_experiments/{name}": digest(runtime / name) for name in scripts})
        frozen.update({"provenance/" + p.name: digest(p) for p in (stage / "provenance").glob("*.py")})
        stage_plan = {"job_id": parent["job_id"], "node": parent["node"], "python_bin": parent["python_bin"],
                      "output_root": str(stage), "suite_root": str(root), "dataset_key": dataset,
                      "runtime": str(root / "runtime"), "scratch": str(scratch / "datasets" / dataset),
                      "hf_home": parent["hf_home"], "holder_locks": old["holder_locks"],
                      "parent_eval_root": str(previous), "parent_eval_plan_sha256": digest(previous / "plan.json"),
                      "models": models, "datasets": old["datasets"], "num_questions": len(rows),
                      **grid[dataset], "seed": 0, "protocol": "eval2", "stop_on_first_success": True,
                      "sampling": old["sampling"], "engine": old["engine"], "grading": old["grading"],
                      "num_gpus": 8, "frozen_files": frozen}
        write(stage / "plan.json", stage_plan)
        manifest = {"plan_sha256": digest(stage / "plan.json"), "models": models, "protocol": "eval2"}
        manifest["fingerprint"] = mini.fingerprint(manifest)
        write(stage / "execution_manifest.json", manifest)
        fixed += [stage / "questions.json", stage / "plan.json", stage / "execution_manifest.json",
                  *(stage / "provenance").glob("*.py")]
    plan = {"job_id": parent["job_id"], "node": parent["node"], "python_bin": parent["python_bin"],
            "output_root": str(root), "runtime": str(root / "runtime"), "scratch": str(scratch),
            "hf_home": parent["hf_home"], "predecessor_root": str(previous),
            "predecessor_plan_sha256": digest(previous / "plan.json"),
            "predecessor_schedule_sha256": digest(previous / "schedule.json"),
            "dataset_order": list(DATASETS), "models": models, "budget_grid": grid,
            "points": POINTS, "protocol": "eval2", "seed": 0,
            "storage_critical_bytes": 50 * 1024**3, "storage_warning_bytes": 100 * 1024**3,
            "frozen_files": {p.relative_to(root).as_posix(): digest(p) for p in fixed if p.is_file()}, "created_at": now()}
    write(root / "plan.json", plan)
    write(root / "input_audit.json", {"points": POINTS, "dataset_order": list(DATASETS), "models": list(MODELS),
          "budget_grid": grid, "reference_sampling": audit["sampling"], "new_sampling": stage_plan["sampling"],
          "questions": sum(mini.DATASETS[key][1] for key in DATASETS), "thinking_prompts_verified": True,
          "rounding": "ceil of multiplier times the mean as printed in the audited 32k base reference",
          "models_staged_before_predecessor_completion": False})
    shutil.copy2(source / "qwen3_experiments/compression_relative_individual_budget.md", root / "README.md")
    return verify_suite(root)


def prepare_models(stage, plan):
    from transformers import AutoTokenizer
    from qwen3_experiments.prepare_math_eval_matrix import download_files

    manifest = read(stage / "execution_manifest.json")
    if manifest["plan_sha256"] != digest(stage / "plan.json") or manifest["protocol"] != "eval2":
        raise ValueError("Individual-budget execution identity changed")
    for key, model in plan["models"].items():
        if all(mini.completed_point(stage, key, budget, manifest, plan["dataset_key"]) for budget in plan["budgets"]):
            continue
        if shutil.disk_usage(Path(plan["scratch"])).free < 64 * 1024**3:
            raise RuntimeError("Insufficient compute scratch for model staging")
        target, hashes = Path(model["path"]), model["files_sha256"]
        if not all((target / name).is_file() and digest(target / name) == value for name, value in hashes.items()):
            downloaded = download_files({**model, "key": key}, list(hashes), target)
            if any(downloaded[name]["sha256"] != value for name, value in hashes.items()):
                raise ValueError("Downloaded model differs from the pinned checkpoint")
        tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
        rows = read(stage / "questions.json")
        if mini.thinking_rows(rows, tokenizer, "qwen3") != rows:
            raise ValueError("Model tokenizer changed the frozen thinking prompts")
        write(stage / "prepared_models" / f"{key}.json", {"repo": model["repo"], "revision": model["revision"],
                                                          "files_sha256": hashes, "verified_at": now()})
    return manifest


def cleanup_model(stage, plan, key):
    manifest = read(stage / "execution_manifest.json")
    for budget in plan["budgets"]:
        point = mini.completed_point(stage, key, budget, manifest, plan["dataset_key"])
        if (not point or point.get("protocol") != "eval2" or point.get("ledger_audit") != "passed"
                or point["allocated_output_budget"] != plan["num_questions"] * budget):
            raise ValueError("Cleanup requires all four verified individual-budget points")
    suite = read(Path(plan["suite_root"]) / "plan.json")
    target = Path(plan["models"][key]["path"])
    if target.is_symlink() or target.resolve() != Path(suite["scratch"]) / "models" / key:
        raise ValueError("Cleanup must stay inside this queue's model cache")
    if target.exists():
        shutil.rmtree(target)
    write(stage / "cache_cleanup" / f"{key}.json", {"state": "complete", "model": key,
          "repo": plan["models"][key]["repo"], "revision": plan["models"][key]["revision"], "finished_at": now()})


def active_processes(plan, stage):
    result = []
    for pid in storage.allocation_processes(plan):
        item = storage.identity(pid)
        if item and pid != os.getpid() and item["uid"] == os.getuid() and MODULE in item["command"] and str(stage) in item["command"]:
            result.append(item)
    return result


def report(root, plan):
    rows = []
    expected = set()
    for dataset in DATASETS:
        stage = root / "datasets" / dataset
        if not stage_complete(stage):
            raise ValueError("Cannot report an incomplete relative-budget dataset")
        grid = plan["budget_grid"][dataset]
        multipliers = dict(zip(grid["budgets"], grid["multipliers"], strict=True))
        for row in read(stage / "report/metrics.json"):
            rows.append({**row, "multiplier": multipliers[row["budget_tokens"]],
                         "base_mean_response_length": grid["base_mean_response_length"]})
        expected.update((dataset, key, budget) for key in MODELS for budget in grid["budgets"])
    if len(rows) != POINTS or {(r["dataset"], r["model_key"], r["budget_tokens"]) for r in rows} != expected:
        raise ValueError("Missing or duplicate relative-budget evaluation points")
    rows.sort(key=lambda r: (DATASETS.index(r["dataset"]), MODELS.index(r["model_key"]), r["budget_tokens"]))
    destination = root / "report"
    write(destination / "metrics.json", rows)
    with (destination / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(destination / "audit.json", {"complete": True, "points": POINTS, "datasets": list(DATASETS),
          "models": list(MODELS), "all_rollout_ledgers_verified": True, "metrics_sha256": digest(destination / "metrics.json")})
    lines = ["# Individual pass@budget relative to base mean response length", "",
             "Each question receives ceil(multiplier × base mean response length) output tokens. "
             "The same dataset-specific budgets apply to all six models. Seed 0; temperature/top-p/top-k "
             "0.6/0.95/20; each response is capped at 32k or the remaining allowance. Stop at first success; "
             "unused budget stays with that question. After-thinking grading only; no extra EOS requirement. "
             "All budgets use fresh sampling. Percentages are pass@budget."]
    for dataset in DATASETS:
        grid = plan["budget_grid"][dataset]
        lines += ["", f"## {mini.DATASETS[dataset][0]}", "",
                  f"Base mean response length: {grid['base_mean_response_length']} output tokens.", "",
                  "| Model | " + " | ".join(f"{m}× ({b:,} tokens)" for m, b in zip(MULTIPLIERS, grid["budgets"], strict=True)) + " |",
                  "|---|---:|---:|---:|---:|"]
        for key in MODELS:
            points = [r for r in rows if r["dataset"] == dataset and r["model_key"] == key]
            lines.append("| " + plan["models"][key]["label"] + " | " + " | ".join(f"{r['pass_at_budget_percent']:.2f}%" for r in points) + " |")
    lines += ["", "[Metrics CSV](metrics.csv) · [Metrics JSON](metrics.json) · [Audit](audit.json)"]
    (destination / "README.md").write_text("\n".join(lines) + "\n")


def queue(root):
    plan = verify_suite(root)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children = []
        while True:
            state = {"pid": os.getpid(), "node": socket.gethostname(), "updated_at": now(), "points": POINTS}
            try:
                verify_suite(root)
                children = [child for child in children if child.poll() is None]
                if not predecessor_ready(plan):
                    state.update(state="waiting_for_cross_budget", predecessor=plan["predecessor_root"])
                else:
                    dataset = next_dataset(root)
                    if dataset is None:
                        report(root, plan)
                        write(root / "queue_status.json", {**state, "state": "complete"})
                        return
                    stage = root / "datasets" / dataset
                    active = active_processes(plan, stage)
                    if not active:
                        command = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                                   f"--nodelist={plan['node']}", "--cpus-per-task=96", "--gres=gpu:8",
                                   "--kill-on-bad-exit=1", f"--job-name=compression-relative-{dataset}",
                                   plan["python_bin"], "-u", "-m", MODULE, "run", "--output-root", str(stage)]
                        with (Path(plan["scratch"]) / f"{dataset}.log").open("ab", buffering=0) as log:
                            child = subprocess.Popen(command, cwd=plan["runtime"], env=cross.environment(plan),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        children.append(child)
                        write(root / "phase_launches" / f"{dataset}.json", {"pid": child.pid, "command": command, "launched_at": now()})
                        active = [{"pid": child.pid}]
                    state.update(state="running_or_waiting_for_gpus", dataset=dataset, active_pids=[p["pid"] for p in active])
                write(root / "queue_status.json", state)
            except Exception as exc:
                write(root / "queue_status.json", {**state, "state": "retrying", "error": str(exc)})
            time.sleep(30)


def serve(root):
    plan = verify_suite(root)
    local = Path(plan["scratch"])
    with (local / "service.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            with (local / "queue.log").open("ab", buffering=0) as log:
                child = subprocess.Popen([plan["python_bin"], "-u", "-m", MODULE, "queue", "--output-root", str(root)],
                    cwd=plan["runtime"], env=cross.environment(plan), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(local / "queue_launch.json", {"pid": child.pid, "launched_at": now()})
            while child.poll() is None:
                storage.require_node(plan)
                try:
                    recovered = cross.recover_storage(root, plan)
                except Exception as exc:
                    recovered = {"error": str(exc)}
                write(local / "service_status.json", {"state": "supervising", "pid": os.getpid(), "queue_pid": child.pid,
                      "updated_at": now(), "storage_recovery": recovered,
                      "disks": {name: storage.disk(path) for name, path in
                                (("results", root), ("compute", local), ("project", plan["hf_home"]))}})
                time.sleep(30)
            if child.returncode == 0 and read(root / "queue_status.json").get("state") == "complete":
                while list((local / "offload_receipts").glob("*.json")):
                    cross.recover_storage(root, plan)
                    time.sleep(30)
                return
            time.sleep(30)


def launch(root):
    plan = verify_suite(root)
    env = cross.environment(plan)
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            raise ValueError("Relative-budget service already launched")
        with (Path(plan["scratch"]) / "service.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([plan["python_bin"], "-u", "-m", MODULE, "serve", "--output-root", str(root)],
                cwd=plan["runtime"], env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"pid": child.pid, "job_id": plan["job_id"], "node": plan["node"],
                   "plan_sha256": digest(root / "plan.json"), "points": POINTS, "launched_at": now()}
        write(root / "launch.json", receipt)
        write(Path(plan["predecessor_root"]) / "after_queue_relative_individual_budget.json", {"root": str(root), **receipt})
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "serve", "queue", "run", "worker"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--after-cross", type=Path)
    parser.add_argument("--base-reference", type=Path)
    parser.add_argument("--base-evaluation", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.command == "prepare":
        prepare(args)
        return 0
    if args.command in ("launch", "serve", "queue"):
        return {"launch": launch, "serve": serve, "queue": queue}[args.command](root)
    plan = mini.verify_plan(root)
    verify_suite(Path(plan["suite_root"]))
    mini.MODULE, mini.dependency_ready, mini.prepare_models = MODULE, stage_dependency, prepare_models
    if args.command == "worker":
        if args.dataset != plan["dataset_key"] or args.budget not in plan["budgets"] or args.model not in plan["models"]:
            raise ValueError("Worker must select a point from the dataset's relative-budget plan")
        return mini.worker(root, args.model, args.rank, args.budget, args.dataset)
    cross.environment(plan)
    result = mini.run(root, plan, on_model_complete=lambda key: cleanup_model(root, plan, key))
    if result == 0:
        for key in MODELS:
            receipt = root / "cache_cleanup" / f"{key}.json"
            if not receipt.exists() or read(receipt).get("state") != "complete":
                cleanup_model(root, plan, key)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
