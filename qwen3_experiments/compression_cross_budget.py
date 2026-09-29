"""Append dataset-first cross-context budgets after the existing final evaluations."""

import argparse
import csv
import fcntl
import json
import os
import shutil
import socket
import subprocess
import time
from collections import Counter
from pathlib import Path

from qwen3_experiments import compute_disk_watchdog as storage
from qwen3_experiments import minerva_individual_budget as mini

MODULE = "qwen3_experiments.compression_cross_budget"
DATASETS = ("minervamath", "olympiadbench", "amc22_23", "aime24", "aime25", "aime26", "math500")
MODELS = ("er", "maxrl", "l0", "l4096", "f_cov")
BUDGETS = [4096, 8192, 16384, 32768, 49152]
read, write, digest, now = storage.read, storage.write, storage.digest, storage.now


def scheduling(root, plan):
    path = root / "schedule.json"
    if not path.exists():
        return {"dataset_order": list(plan.get("dataset_order", DATASETS))}
    schedule = read(path)
    if schedule["prepared_plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Schedule does not match the prepared cross-budget plan")
    selected = schedule["dataset_order"]
    prepared = plan.get("dataset_order", DATASETS)
    if not selected or selected != [dataset for dataset in prepared if dataset in selected]:
        raise ValueError("Schedule must retain the prepared order without duplicate or unknown datasets")
    for relative, checksum in schedule.get("controller_files", {}).items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != checksum:
            raise ValueError(f"Frozen controller changed: {relative}")
    return schedule


def verify_suite(root):
    plan = read(root / "plan.json")
    storage.require_node(plan)
    if root != Path(plan["output_root"]) or plan["dataset_order"] != list(DATASETS):
        raise ValueError("Dataset order or queue location changed")
    if list(plan["models"]) != list(MODELS) or plan["points"] != 175:
        raise ValueError("Expected all five final models and 175 cross-budget points")
    if plan["budgets"] != BUDGETS:
        raise ValueError("Expected cross budgets of 4k, 8k, 16k, 32k and 48k")
    if (root / "launch.json").exists() and read(root / "launch.json")["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Launched cross-budget plan changed")
    if (root / "launch.json").exists():
        receipt = read(root / "launch.json")
        if receipt.get("schedule_sha256") and digest(root / "schedule.json") != receipt["schedule_sha256"]:
            raise ValueError("Launched cross-budget schedule changed")
    for relative, checksum in plan["frozen_files"].items():
        if digest(root / relative) != checksum:
            raise ValueError(f"Frozen cross-budget input changed: {relative}")
    scheduling(root, plan)
    return plan


def predecessor_ready(plan):
    previous = Path(plan["predecessor_root"])
    if digest(previous / "plan.json") != plan["predecessor_plan_sha256"]:
        raise ValueError("Original f_cov/MaxRL evaluation plan changed")
    path = previous / "queue_status.json"
    if not path.exists() or read(path).get("state") != "complete":
        return False
    audit = read(previous / "report/audit.json")
    if (not audit.get("complete") or audit.get("models") != ["f_cov_step100", "maxrl_step100"]
            or audit.get("nine_responses") != 14552 or audit.get("budget_points") != 70
            or not audit.get("all_rollout_ledgers_verified")):
        raise ValueError("Both original nine-dataset and individual-budget evaluations must finish")
    for name in ("nine", "budget"):
        if digest(previous / f"report/{name}_metrics.json") != audit[f"{name}_metrics_sha256"]:
            raise ValueError("Original evaluation results changed")
    for key in ("f_cov_step100", "maxrl_step100"):
        if read(previous / "model_cache_cleanup" / f"{key}.json").get("state") != "complete":
            raise ValueError("Original evaluation model cache cleanup must finish")
    return True


def stage_complete(stage):
    path = stage / "status.json"
    if not path.exists() or read(path).get("state") != "complete":
        return False
    audit = read(stage / "report/audit.json")
    if (audit.get("points") != 25 or not audit.get("complete")
            or not audit.get("all_rollout_ledgers_verified")
            or digest(stage / "report/metrics.json") != audit.get("metrics_sha256")):
        raise ValueError("Dataset does not have all 25 verified cross-budget results")
    return all((stage / "cache_cleanup" / f"{key}.json").exists()
               and read(stage / "cache_cleanup" / f"{key}.json").get("state") == "complete"
               for key in MODELS)


def next_dataset(root, plan):
    for dataset in plan["dataset_order"]:
        if not stage_complete(root / "datasets" / dataset):
            return dataset
    return None


def stage_dependency(plan):
    root = Path(plan["suite_root"])
    suite = verify_suite(root)
    return predecessor_ready(suite) and next_dataset(root, suite) == plan["dataset_key"]


def environment(plan):
    env = mini.environment(plan)
    local = Path(plan["scratch"])
    env.update(HF_HOME=plan["hf_home"], HF_HUB_CACHE=str(local / "hub"),
               HF_XET_CACHE=str(local / "xet"), HF_ASSETS_CACHE=str(local / "assets"),
               HF_HUB_DISABLE_XET="1")
    for path in (env["TMPDIR"], env["TRITON_CACHE_DIR"], env["HF_HUB_CACHE"], env["HF_XET_CACHE"], env["HF_ASSETS_CACHE"]):
        Path(path).mkdir(parents=True, exist_ok=True)
    return env


def controller_environment(plan):
    env = environment(plan)
    schedule = scheduling(Path(plan["output_root"]), plan)
    if "controller_runtime" in schedule:
        runtime = Path(schedule["controller_runtime"])
        if not runtime.resolve().is_relative_to(Path(plan["output_root"]).resolve()):
            raise ValueError("Controller runtime must belong to this queue")
        env["PYTHONPATH"] = str(runtime)
    return env


def prepare(args):
    from transformers import AutoTokenizer

    previous = args.after_evaluation.resolve()
    parent = read(previous / "plan.json")
    storage.require_node(parent)
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=source, text=True).strip() != "agent/add-math12k-maxrl-launcher":
        raise ValueError("Use the primary maxrl launcher branch")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        return verify_suite(root)
    reference = previous / "stages/f_cov_step100/pass_at_budget"
    old = mini.verify_plan(reference)
    all_rows = read(reference / "questions.json")
    tokenizer = AutoTokenizer.from_pretrained(parent["base_model"], local_files_only=True)
    if mini.thinking_rows(all_rows, tokenizer, "qwen3") != all_rows:
        raise ValueError("Frozen thinking prompts changed")
    if Counter(row["dataset"] for row in all_rows) != {key: mini.DATASETS[key][1] for key in DATASETS}:
        raise ValueError("Expected all seven complete datasets")
    exports = args.exports.resolve()
    exported = read(exports / "audit.json")
    if exported.get("state") != "complete" or not exported.get("all_public"):
        raise ValueError("All five final exports must be verified first")
    scratch = Path(args.scratch)
    if not scratch.is_absolute() or scratch.parent != Path("/tmp") or scratch.is_symlink():
        raise ValueError("Use a dedicated node-local scratch directory")
    scratch.mkdir(parents=True, exist_ok=True)
    runtime = root / "runtime/qwen3_experiments"
    runtime.mkdir(parents=True, exist_ok=True)
    scripts = ("compression_cross_budget.py", "minerva_individual_budget.py", "math_eval_budget_engine.py",
               "math_eval_matrix_common.py", "grpo_compute_control.py", "compute_disk_watchdog.py",
               "prepare_math_eval_matrix.py")
    for name in scripts:
        shutil.copy2(source / "qwen3_experiments" / name, runtime / name)
    models = {}
    for key in MODELS:
        receipt = read(exports / "receipts" / f"{key}.json")
        if receipt.get("state") != "published_verified" or not receipt.get("public") or receipt["training_step"] != 100:
            raise ValueError("Final model is not verified and public")
        if not receipt["repo"].endswith("-final") or "compression" not in receipt["repo"]:
            raise ValueError("Use compression final model repositories")
        write(root / "export_receipts" / f"{key}.json", receipt)
        models[key] = {"label": {"er": "ER", "maxrl": "MaxRL", "l0": "L+0", "l4096": "L+4096", "f_cov": "f_cov"}[key],
                       "repo": receipt["repo"], "revision": receipt["revision"], "format": "huggingface",
                       "path": str(scratch / "models" / key), "files_sha256": receipt["files_sha256"],
                       "family": "qwen3", "questions_file": "questions.json"}
    fixed = [*runtime.glob("*.py"), *(root / "export_receipts").glob("*.json")]
    by_key = {spec["key"]: spec for spec in old["datasets"]}
    locks = list(dict.fromkeys([*parent["holder_locks"], str(previous / "run.lock")]))
    for dataset in DATASETS:
        stage = root / "datasets" / dataset
        stage.mkdir(parents=True, exist_ok=True)
        rows = [row for row in all_rows if row["dataset"] == dataset]
        write(stage / "questions.json", rows)
        runtime_link = stage / "runtime"
        if runtime_link.is_symlink():
            if runtime_link.resolve() != root / "runtime":
                raise ValueError("Dataset runtime points outside this queue")
        else:
            runtime_link.symlink_to(root / "runtime", target_is_directory=True)
        (stage / "provenance").mkdir(exist_ok=True)
        for name in ("eval_l0_final.py", "eval_polaris_step80.py"):
            shutil.copy2(reference / "provenance" / name, stage / "provenance" / name)
        frozen = {"questions.json": digest(stage / "questions.json")}
        frozen.update({f"runtime/qwen3_experiments/{name}": digest(runtime / name) for name in scripts})
        frozen.update({"provenance/" + p.name: digest(p) for p in (stage / "provenance").glob("*.py")})
        plan = {"job_id": parent["job_id"], "node": parent["node"], "python_bin": parent["python_bin"],
                "output_root": str(stage), "suite_root": str(root), "dataset_key": dataset,
                "runtime": str(root / "runtime"), "scratch": str(scratch / "datasets" / dataset),
                "hf_home": parent["hf_home"], "holder_locks": locks,
                "parent_eval_root": str(previous), "parent_eval_plan_sha256": digest(previous / "plan.json"),
                "models": models, "datasets": [by_key[dataset]], "num_questions": len(rows), "budgets": BUDGETS,
                "seed": 0, "protocol": "eval3", "stop_on_first_success": True,
                "sampling": old["sampling"], "engine": old["engine"], "grading": old["grading"],
                "num_gpus": 8, "frozen_files": frozen}
        write(stage / "plan.json", plan)
        manifest = {"plan_sha256": digest(stage / "plan.json"), "models": models, "protocol": "eval3"}
        manifest["fingerprint"] = mini.fingerprint(manifest)
        write(stage / "execution_manifest.json", manifest)
        fixed += [stage / "questions.json", stage / "plan.json", stage / "execution_manifest.json",
                  *(stage / "provenance").glob("*.py")]
    suite = {"job_id": parent["job_id"], "node": parent["node"], "python_bin": parent["python_bin"],
             "output_root": str(root), "runtime": str(root / "runtime"), "scratch": str(scratch),
             "hf_home": parent["hf_home"], "predecessor_root": str(previous),
             "predecessor_plan_sha256": digest(previous / "plan.json"), "dataset_order": list(DATASETS),
             "models": models, "budgets": BUDGETS, "seed": 0, "protocol": "eval3", "points": 175,
             "storage_critical_bytes": 50 * 1024**3, "storage_warning_bytes": 100 * 1024**3,
             "frozen_files": {p.relative_to(root).as_posix(): digest(p) for p in fixed}, "created_at": now()}
    write(root / "plan.json", suite)
    write(root / "input_audit.json", {"points": 175, "dataset_order": list(DATASETS),
          "dataset_counts": {key: mini.DATASETS[key][1] for key in DATASETS},
          "global_output_budgets": {key: [mini.DATASETS[key][1] * b for b in BUDGETS] for key in DATASETS},
          "original_queue_retained": True, "thinking_prompts_verified": len(all_rows),
          "models_staged_before_predecessor_completion": False})
    return verify_suite(root)


def prepare_models(stage, plan):
    from transformers import AutoTokenizer
    from qwen3_experiments.prepare_math_eval_matrix import download_files

    manifest = read(stage / "execution_manifest.json")
    if manifest["plan_sha256"] != digest(stage / "plan.json") or manifest["protocol"] != "eval3":
        raise ValueError("Cross-budget execution identity changed")
    for key, model in plan["models"].items():
        if all(mini.completed_point(stage, key, b, manifest, plan["dataset_key"]) for b in plan["budgets"]):
            continue
        target = Path(model["path"])
        if shutil.disk_usage(Path(plan["scratch"])).free < 64 * 1024**3:
            raise RuntimeError("Insufficient compute scratch space for model staging")
        hashes = model["files_sha256"]
        if not all((target / name).is_file() and digest(target / name) == checksum for name, checksum in hashes.items()):
            downloaded = download_files({**model, "key": key}, list(hashes), target)
            if any(downloaded[name]["sha256"] != checksum for name, checksum in hashes.items()):
                raise ValueError("Downloaded final model differs from its verified export")
        tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
        rows = read(stage / "questions.json")
        if mini.thinking_rows(rows, tokenizer, "qwen3") != rows:
            raise ValueError("Final model tokenizer changed the frozen prompts")
        write(stage / "prepared_models" / f"{key}.json", {"repo": model["repo"], "revision": model["revision"],
                                                          "files_sha256": hashes, "verified_at": now()})
    return manifest


def cleanup_model(stage, plan, key):
    manifest = read(stage / "execution_manifest.json")
    for budget in plan["budgets"]:
        point = mini.completed_point(stage, key, budget, manifest, plan["dataset_key"])
        if (not point or point.get("protocol") != "eval3" or point.get("ledger_audit") != "passed"
                or point["allocated_output_budget"] != plan["num_questions"] * budget):
            raise ValueError("Model cleanup requires all five verified shared-budget points")
    suite = read(Path(plan["suite_root"]) / "plan.json")
    target = Path(plan["models"][key]["path"])
    if target.resolve() != Path(suite["scratch"]) / "models" / key or target.is_symlink():
        raise ValueError("Cleanup must stay inside this queue's model cache")
    if target.exists():
        shutil.rmtree(target)
    write(stage / "cache_cleanup" / f"{key}.json", {"state": "complete", "model": key,
          "repo": plan["models"][key]["repo"], "revision": plan["models"][key]["revision"], "finished_at": now()})


def active_processes(plan, stage):
    needle = str(stage)
    result = []
    for pid in storage.allocation_processes(plan):
        process = storage.identity(pid)
        if (process and pid != os.getpid() and process["uid"] == os.getuid()
                and MODULE in process["command"] and needle in process["command"]):
            result.append(process)
    return result


def report(root, plan):
    datasets = scheduling(root, plan)["dataset_order"]
    rows = []
    for dataset in datasets:
        stage = root / "datasets" / dataset
        if not stage_complete(stage):
            raise ValueError("Cannot report an incomplete dataset")
        rows.extend(read(stage / "report/metrics.json"))
    expected = {(dataset, key, b) for dataset in datasets for key in MODELS for b in BUDGETS}
    if len(rows) != len(expected) or {(r["dataset"], r["model_key"], r["budget_tokens"]) for r in rows} != expected:
        raise ValueError("Incomplete cross-budget matrix")
    for row in rows:
        row["total_budget_tokens"] = row["questions"] * row["budget_tokens"]
    order = {key: index for index, key in enumerate(MODELS)}
    rows.sort(key=lambda row: (datasets.index(row["dataset"]), order[row["model_key"]], row["budget_tokens"]))
    destination = root / "report"
    write(destination / "metrics.json", rows)
    with (destination / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(destination / "audit.json", {"complete": True, "points": len(expected), "datasets": datasets,
                                       "all_rollout_ledgers_verified": True,
                                       "metrics_sha256": digest(destination / "metrics.json")})
    lines = ["# Compression final models: cross-context pass@budget", "",
             "Each point shares (4k, 8k, 16k, 32k or 48k) × dataset size output tokens. "
             "One seed (0); 1k = 1,024 tokens; each response is capped at 32k. "
             "Seeded shuffled sweeps skip solved questions. Temperature / top-p / top-k: 0.6 / 0.95 / 20. "
             "Only the answer after completed thinking is graded; no extra EOS requirement."]
    for dataset in datasets:
        lines += ["", f"## {mini.DATASETS[dataset][0]}", "",
                  "| Model | " + " | ".join(f"{budget // 1024}k × n" for budget in BUDGETS) + " |",
                  "|---|---:|---:|---:|---:|---:|"]
        for key in MODELS:
            points = [row for row in rows if row["dataset"] == dataset and row["model_key"] == key]
            lines.append("| " + plan["models"][key]["label"] + " | "
                         + " | ".join(f"{row['pass_at_budget_percent']:.2f}%" for row in points) + " |")
    lines += ["", "[Metrics CSV](metrics.csv) · [Metrics JSON](metrics.json) · [Audit](audit.json)"]
    (destination / "README.md").write_text("\n".join(lines) + "\n")


def queue(root):
    plan = verify_suite(root)
    selected = scheduling(root, plan)["dataset_order"]
    scheduled_plan = {**plan, "dataset_order": selected}
    points = len(selected) * len(MODELS) * len(BUDGETS)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children = []
        while True:
            state = {"pid": os.getpid(), "node": socket.gethostname(), "updated_at": now(),
                     "points": points, "dataset_order": selected}
            try:
                verify_suite(root)
                children = [child for child in children if child.poll() is None]
                if not predecessor_ready(plan):
                    state.update(state="waiting_for_original_nine_and_individual_evaluations",
                                 predecessor=plan["predecessor_root"])
                else:
                    dataset = next_dataset(root, scheduled_plan)
                    if dataset is None:
                        report(root, plan)
                        write(root / "queue_status.json", {**state, "state": "complete"})
                        return
                    stage = root / "datasets" / dataset
                    active = active_processes(plan, stage)
                    if not active:
                        command = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                                   f"--nodelist={plan['node']}", "--cpus-per-task=96", "--gres=gpu:8",
                                   "--kill-on-bad-exit=1", f"--job-name=compression-cross-{dataset}",
                                   plan["python_bin"], "-u", "-m", MODULE, "run", "--output-root", str(stage)]
                        with (Path(plan["scratch"]) / f"{dataset}.log").open("ab", buffering=0) as log:
                            child = subprocess.Popen(command, cwd=plan["runtime"], env=environment(plan),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        children.append(child)
                        write(root / "phase_launches" / f"{dataset}.json", {"pid": child.pid, "command": command, "launched_at": now()})
                        active = [{"pid": child.pid}]
                    state.update(state="running_or_waiting_for_gpus", dataset=dataset, active_pids=[p["pid"] for p in active])
                write(root / "queue_status.json", state)
            except Exception as exc:
                write(root / "queue_status.json", {**state, "state": "retrying", "error": str(exc)})
            time.sleep(30)


def recover_storage(root, plan):
    local = Path(plan["scratch"])
    free = shutil.disk_usage(root).free
    moved = restored = 0
    if free < plan["storage_critical_bytes"]:
        for summary in (root / "datasets").glob("*/results/*/*/budget_*/summary.json"):
            value = read(summary)
            if value.get("state") != "complete" or value.get("ledger_audit") != "passed":
                continue
            for item in value["artifacts"].values():
                relative = Path(item["file"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Result artifact must stay inside its evaluation point")
                source = summary.parent / relative
                if source.is_symlink() or source.resolve() != source or time.time() - source.stat().st_mtime < 120:
                    continue
                if source.stat().st_size != item["size"] or digest(source) != item["sha256"]:
                    raise ValueError("Cannot offload an unverified result")
                key = mini.fingerprint(str(source))
                target = local / "offloaded" / key
                target.parent.mkdir(parents=True, exist_ok=True)
                if shutil.disk_usage(local).free < item["size"] + 64 * 1024**3:
                    raise RuntimeError("Insufficient node-local space for verified result offload")
                shutil.copy2(source, target)
                if digest(target) != item["sha256"]:
                    raise ValueError("Result offload failed verification")
                write(local / "offload_receipts" / f"{key}.json", {"source": str(source), "target": str(target), **item})
                link = source.with_name(source.name + ".offloading")
                link.unlink(missing_ok=True)
                link.symlink_to(target)
                link.replace(source)
                moved += item["size"]
                if moved >= 512 * 1024**2:
                    return {"moved_bytes": moved, "restored_bytes": 0}
    elif free > plan["storage_warning_bytes"]:
        for receipt in (local / "offload_receipts").glob("*.json"):
            item = read(receipt)
            source, target = Path(item["source"]), Path(item["target"])
            source.relative_to(root / "datasets")
            target.relative_to(local / "offloaded")
            if (".." in source.parts or ".." in target.parts or source.parent.resolve() != source.parent
                    or target.resolve() != target):
                raise ValueError("Offload receipt points outside this queue")
            if not source.is_symlink() or source.readlink() != target or digest(target) != item["sha256"]:
                raise ValueError("Offloaded result identity changed")
            temporary = source.with_name(source.name + ".restoring")
            shutil.copy2(target, temporary)
            if digest(temporary) != item["sha256"]:
                raise ValueError("Result restoration failed verification")
            temporary.replace(source)
            target.unlink()
            receipt.unlink()
            restored += item["size"]
            if restored >= 512 * 1024**2:
                break
    return {"moved_bytes": moved, "restored_bytes": restored}


def serve(root):
    plan = verify_suite(root)
    local = Path(plan["scratch"])
    control_env = controller_environment(plan)
    with (local / "service.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            with (local / "queue.log").open("ab", buffering=0) as log:
                child = subprocess.Popen([plan["python_bin"], "-u", "-m", MODULE, "queue", "--output-root", str(root)],
                    cwd=control_env["PYTHONPATH"], env=control_env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(local / "queue_launch.json", {"pid": child.pid, "launched_at": now()})
            while child.poll() is None:
                storage.require_node(plan)
                try:
                    recovered = recover_storage(root, plan)
                except Exception as exc:
                    recovered = {"error": str(exc)}
                write(local / "service_status.json", {"state": "supervising", "pid": os.getpid(), "queue_pid": child.pid,
                      "updated_at": now(), "disks": {name: storage.disk(path) for name, path in
                      (("results", root), ("compute", local), ("project", "/project/flame"))}, "storage_recovery": recovered})
                time.sleep(30)
            status = read(root / "queue_status.json") if (root / "queue_status.json").exists() else {}
            if child.returncode == 0 and status.get("state") == "complete":
                while list((local / "offload_receipts").glob("*.json")):
                    recover_storage(root, plan)
                    time.sleep(30)
                return
            time.sleep(30)


def launch(root):
    plan = verify_suite(root)
    control_env = controller_environment(plan)
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            raise ValueError("Cross-budget service is already launched")
        with (Path(plan["scratch"]) / "service.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([plan["python_bin"], "-u", "-m", MODULE, "serve", "--output-root", str(root)],
                cwd=control_env["PYTHONPATH"], env=control_env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"pid": child.pid, "job_id": plan["job_id"], "node": plan["node"],
                   "plan_sha256": digest(root / "plan.json"), "launched_at": now()}
        if (root / "schedule.json").exists():
            receipt["schedule_sha256"] = digest(root / "schedule.json")
            receipt["dataset_order"] = scheduling(root, plan)["dataset_order"]
        write(root / "launch.json", receipt)
        write(Path(plan["predecessor_root"]) / "after_queue_cross_budget.json", {"root": str(root), **receipt})
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "serve", "queue", "run", "worker"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--after-evaluation", type=Path)
    parser.add_argument("--exports", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--budget", type=int, choices=BUDGETS)
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
    mini.MODULE = MODULE
    mini.dependency_ready = stage_dependency
    mini.prepare_models = prepare_models
    if args.command == "worker":
        return mini.worker(root, args.model, args.rank, args.budget, args.dataset)
    environment(plan)
    result = mini.run(root, plan, on_model_complete=lambda key: cleanup_model(root, plan, key))
    if result == 0:
        # A restart with all points already saved skips the worker loop and its callbacks.
        for key in MODELS:
            receipt = root / "cache_cleanup" / f"{key}.json"
            if not receipt.exists() or read(receipt).get("state") != "complete":
                cleanup_model(root, plan, key)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
