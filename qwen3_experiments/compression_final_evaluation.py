"""Evaluate compression f_cov then MaxRL: nine benchmarks and seven budget sweeps."""

import argparse
from collections import Counter
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

from qwen3_experiments import compression_budget_followup as followup
from qwen3_experiments import compression_l0_compute_control as training
from qwen3_experiments import compression_pipeline_watchdog as storage
from qwen3_experiments import er_compression_evaluation as shared
from qwen3_experiments import eval_l0_final as nine
from qwen3_experiments import minerva_individual_budget as mini
from qwen3_experiments import polaris_checkpoint_evaluation as checkpoint

MODULE = "qwen3_experiments.compression_final_evaluation"
read, write, digest, now = storage.read, storage.write, storage.digest, storage.now
MODEL_ORDER = ("f_cov_step100", "maxrl_step100")
DATASETS = ("math500", "minervamath", "olympiadbench", "amc22_23", "aime24", "aime25", "aime26")


def verify_plan(root):
    plan = read(root / "plan.json")
    if root.resolve() != Path(plan["output_root"]):
        raise ValueError("Evaluation root changed")
    launch = root / "launch.json"
    if launch.exists() and read(launch)["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Queued evaluation plan changed")
    if [model["key"] for model in plan["models"]] != list(MODEL_ORDER):
        raise ValueError("Evaluate f_cov completely before MaxRL")
    for relative, checksum in plan["frozen_files"].items():
        if digest(root / relative) != checksum:
            raise ValueError(f"Frozen evaluation input changed: {relative}")
    for model in plan["models"]:
        if digest(Path(model["training_root"]) / "plan.json") != model["training_plan_sha256"]:
            raise ValueError("Preceding training plan changed")
    storage.require_node(plan)
    return plan


def predecessor_plan(plan, model):
    return {"job_id": plan["job_id"], "node": plan["node"],
            "predecessor_training_root": model["training_root"],
            "predecessor_plan_sha256": model["training_plan_sha256"]}


def training_ready(plan):
    return all(training.training_predecessor_ready(predecessor_plan(plan, model)) for model in plan["models"])


def stage_roots(root, key):
    return root / "stages" / key / "nine_datasets", root / "stages" / key / "pass_at_budget"


def prepare(args):
    from transformers import AutoTokenizer

    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=source, text=True).strip() != "agent/add-math12k-maxrl-launcher":
        raise ValueError("Use the primary maxrl launcher branch")
    roots = (args.f_cov_training.resolve(), args.maxrl_training.resolve())
    originals = [read(path / "plan.json") for path in roots]
    first = originals[0]
    node = training.require_compute(first["job_id"])
    for plan, estimator in zip(originals, ("f_cov", "maxrl")):
        if (plan["job_id"] != first["job_id"] or plan["node"] != node or plan["adv_estimator"] != estimator
                or plan["dataset_repo"] != "zjhhhh/compression_dataset" or plan["total_steps"] != 100
                or plan["model_revision"] != mini.MODEL_REVISION):
            raise ValueError("Expected this allocation's compression f_cov and MaxRL runs")
    if first["predecessor_training_root"] != str(roots[1]):
        raise ValueError("f_cov must follow the selected MaxRL training")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with training.holder_locks([root / "prepare.lock"]):
        if (root / "plan.json").exists():
            plan = verify_plan(root)
            if [m["training_root"] for m in plan["models"]] != list(map(str, roots)):
                raise ValueError("Existing queue belongs to different training runs")
            return plan
        reference = args.nine_reference.resolve()
        ref_plan = nine.verify_plan(reference, nine.evaluator(reference))
        if not shared.nine_complete(reference):
            raise ValueError("Expected a completed and audited nine-dataset reference")
        questions, inputs = read(reference / "questions.json"), read(reference / "input_plan.json")
        if (Counter(q["dataset"] for q in questions) != {k: n for k, _, n in nine.DATASETS}
                or len({q["id"] for q in questions}) != 1819
                or digest(reference / "questions.json") != inputs["questions_sha256"]):
            raise ValueError("Nine-dataset questions changed")
        tokenizer = AutoTokenizer.from_pretrained(first["model_path"], local_files_only=True)
        for q in questions:
            if tokenizer.apply_chat_template(q["messages"], add_generation_prompt=True, enable_thinking=True) != q["prompt_token_ids"]:
                raise ValueError("Frozen thinking prompts do not match Qwen3")
        budget_reference = mini.verify_plan(args.budget_reference.resolve())
        if (budget_reference["seed"] != 0 or budget_reference["budgets"] != mini.BUDGETS
                or not budget_reference["grading"]["after_thinking_only"]
                or budget_reference["grading"]["require_eos"] or not budget_reference["stop_on_first_success"]):
            raise ValueError("Budget reference must use the established after-thinking protocol")
        runtime = root / "runtime"
        names = ("compression_final_evaluation.py", "compression_budget_followup.py",
                 "compression_l0_compute_control.py", "compression_pipeline_watchdog.py", "compute_disk_watchdog.py",
                 "er_compression_evaluation.py", "polaris_checkpoint_evaluation.py", "eval_l0_final.py",
                 "eval_polaris_step80.py", "minerva_individual_budget.py", "math_eval_budget_engine.py",
                 "math_eval_matrix_common.py", "grpo_compute_control.py", "prepare_math_eval_matrix.py",
                 "verified_rollout_cleanup.py")
        for name in names:
            shared.copy_file(source / "qwen3_experiments" / name, runtime / "qwen3_experiments" / name)
        shared.copy_file(source / "scripts/model_merger.py", runtime / "scripts/model_merger.py")
        scratch = Path("/tmp") / f"compressionfinaleval{first['job_id']}"
        models, fixed = [], []
        for key, label, previous, old in zip(MODEL_ORDER, ("f_cov", "MaxRL"), roots, originals):
            model = {"key": key, "label": label, "repo": old["hf_repo_prefix"] + "-step_100",
                     "format": "fsdp8", "step": 100, "training_root": str(previous),
                     "training_plan_sha256": digest(previous / "plan.json")}
            models.append(model)
            nroot, broot = stage_roots(root, key)
            for name in ("questions.json", "provenance/main_baseline.csv", "provenance/budget_baseline.csv",
                         "provenance/eval_rloo_final.py"):
                shared.copy_file(reference / name, nroot / name)
            for name in ("eval_l0_final.py", "eval_polaris_step80.py"):
                for folder in (nroot, broot):
                    shared.copy_file(runtime / "qwen3_experiments" / name, folder / "provenance" / name)
            for name in ("main.csv", "budgets.csv", "source_report.md", "audit.json"):
                shared.copy_file(reference / "report/qwen3_after_thinking" / name,
                                 nroot / "report/qwen3_after_thinking" / name)
            new_inputs = {**copy.deepcopy(inputs), "model": {"repo": model["repo"], "revision": None}, "model_label": label}
            write(nroot / "input_plan.json", new_inputs)
            nplan = {"training_root": str(previous), "training_kind": "archived_checkpoint", "model_label": label,
                     "model_repo": model["repo"], "model_format": "fsdp8", "final_step": 100,
                     "report_training_dataset": "compression", "repository": str(runtime),
                     "reference_model": old["model_path"], "job_id": old["job_id"], "queue_node": node,
                     "num_gpus": 8, "questions": 1819, "total_responses": 7276, "sampling": ref_plan["sampling"],
                     "sampling_seed": 42, "samples_per_question": 4, "caps": list(nine.CAPS),
                     "merger_sha256": digest(runtime / "scripts/model_merger.py"),
                     "frozen_files": {p.relative_to(nroot).as_posix(): digest(p) for p in nroot.rglob("*") if p.is_file()}}
            write(nroot / "plan.json", nplan)
            rows = [{"unique_id": q["id"], "ground_truth": q["gold"], "prompt": q["messages"],
                     "prompt_token_ids": q["prompt_token_ids"], "source_row": q["source_row"], "dataset": q["dataset"]}
                    for q in questions if q["dataset"] in DATASETS]
            write(broot / "questions.json", rows)
            specs = {s["key"]: s for s in inputs["datasets"]}
            bplan = {"job_id": old["job_id"], "node": node, "python_bin": sys.executable,
                     "suite_root": str(root), "output_root": str(broot), "runtime": str(runtime),
                     "scratch": str(scratch / "stages" / key), "holder_locks": first["holder_locks"],
                     "parent_eval_root": str(nroot), "parent_eval_plan_sha256": digest(nroot / "plan.json"),
                     "models": {key: {"label": f"Compression {label} step 100", "repo": model["repo"], "revision": None,
                                     "path": str(scratch / "models" / key / "model"),
                                     "family": "qwen3", "questions_file": "questions.json"}},
                     "datasets": [specs[k] for k in DATASETS], "num_questions": len(rows), "budgets": mini.BUDGETS,
                     "seed": 0, "protocol": "eval2", "stop_on_first_success": True, "num_gpus": 8,
                     "scheduling": budget_reference["scheduling"],
                     **{k: copy.deepcopy(budget_reference[k]) for k in ("sampling", "engine", "grading")},
                     "frozen_files": {p.relative_to(broot).as_posix(): digest(p) for p in broot.rglob("*") if p.is_file()}}
            write(broot / "plan.json", bplan)
            nine.verify_plan(nroot, nine.evaluator(nroot))
            mini.verify_plan(broot)
            fixed.extend(p for folder in (nroot, broot) for p in folder.rglob("*") if p.is_file())
        plan = {"job_id": first["job_id"], "node": node, "python_bin": sys.executable, "output_root": str(root),
                "runtime": str(runtime), "scratch": str(scratch), "models": models,
                "hf_home": os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")),
                "holder_locks": list(dict.fromkeys([*first["holder_locks"], str(roots[0] / "supervisor.lock")])),
                "base_model": first["model_path"], "datasets": list(DATASETS), "budgets": mini.BUDGETS,
                "nine_questions_per_model": 1819, "nine_responses_per_model": 7276, "budget_points_per_model": 35,
                "storage_warning_bytes": 100 * 1024**3, "storage_critical_bytes": 50 * 1024**3,
                "created_at": now(), "frozen_files": {p.relative_to(root).as_posix(): digest(p)
                    for p in [*fixed, *(p for p in runtime.rglob("*") if p.is_file())]}}
        write(root / "plan.json", plan)
        write(root / "input_audit.json", {"thinking_prompts_verified": 1819, "nine_responses_per_model": 7276,
              "budget_dataset_counts": {k: mini.DATASETS[k][1] for k in DATASETS}, "budget_points_per_model": 35,
              "model_order": list(MODEL_ORDER), "checkpoint_revisions": "Bound to verified step-100 archive commits at execution"})
        return verify_plan(root)


def bind_checkpoint(root, plan, model):
    if not training.training_predecessor_ready(predecessor_plan(plan, model)):
        raise RuntimeError("Training must complete before binding its final checkpoint")
    receipt = read(Path(model["training_root"]) / "hf_checkpoint_archive/receipts/global_step_100.json")
    if (receipt["repo_id"] != model["repo"] or receipt["checkpoint"] != "global_step_100"
            or receipt["state"] != "archived_and_deleted" or not re.fullmatch(r"[0-9a-f]{40}", receipt["remote_commit"])):
        raise ValueError("Wrong final checkpoint archive")
    for rank in range(8):
        item = receipt["files"][f"actor/model_world_size_8_rank_{rank}.pt"]
        if item["size"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Final checkpoint lacks all eight verified ranks")
    nroot, _ = stage_roots(root, model["key"])
    shared.immutable_json(nroot / "final_checkpoint_receipt.json", receipt)
    return {**model, "revision": receipt["remote_commit"]}, receipt


def prepare_model(root, plan, model):
    spec, archived = bind_checkpoint(root, plan, model)
    nroot, _ = stage_roots(root, model["key"])
    from huggingface_hub import HfApi

    if HfApi().model_info(spec["repo"], revision=spec["revision"]).private:
        raise ValueError("Expected a public project checkpoint")
    path, receipt = checkpoint.prepared_model(root, {**plan, "nine_root": str(nroot)}, spec)
    for name, item in receipt["source_files"].items():
        info = archived["files"][name.removeprefix("global_step_100/")]
        if item != {"size": info["size"], "sha256": info["sha256"]}:
            raise ValueError("Downloaded weights differ from the archived final checkpoint")
    destination = nroot / "model"
    if not destination.is_symlink():
        if destination.exists():
            raise ValueError("Unexpected nine-dataset model directory")
        destination.symlink_to(path, target_is_directory=True)
    if destination.resolve() != path.resolve():
        raise ValueError("Nine-dataset model points to another cache")
    shared.immutable_json(nroot / "model_receipt.json", receipt)
    inputs = read(nroot / "input_plan.json")
    inputs["model"]["revision"] = spec["revision"]
    inputs.update(plan_sha256=digest(nroot / "plan.json"),
                  checkpoint_receipt_sha256=digest(nroot / "final_checkpoint_receipt.json"))
    shared.immutable_json(nroot / "prepared_inputs.json", inputs)
    return receipt


def budget_dependency(plan):
    return shared.nine_complete(Path(plan["parent_eval_root"]))


def budget_manifest(root, plan):
    nroot = Path(plan["parent_eval_root"])
    archived, merged = read(nroot / "final_checkpoint_receipt.json"), read(nroot / "model_receipt.json")
    if merged != read(nroot / "manifest.json")["model"] or merged["revision"] != archived["remote_commit"]:
        raise ValueError("Budget evaluation must use the exact nine-dataset checkpoint")
    models = {}
    for key, model in plan["models"].items():
        if model["repo"] != merged["repo"]:
            raise ValueError("Budget checkpoint identity mismatch")
        for name, checksum in merged["merged_files"].items():
            if digest(Path(model["path"]) / name) != checksum:
                raise ValueError("Prepared budget model changed")
        models[key] = {**model, "revision": merged["revision"], "files_sha256": merged["merged_files"]}
    manifest = {"plan_sha256": digest(root / "plan.json"), "models": models,
                "final_checkpoint_receipt_sha256": digest(nroot / "final_checkpoint_receipt.json")}
    manifest["fingerprint"] = mini.fingerprint(manifest)
    shared.immutable_json(root / "execution_manifest.json", manifest)
    return manifest


def budget_complete(root):
    return followup.stage_complete(root, 35)


def cleanup_complete(root, key):
    path = root / "model_cache_cleanup" / f"{key}.json"
    return path.exists() and read(path).get("state") == "complete"


def next_action(root, plan):
    for model in plan["models"]:
        nroot, broot = stage_roots(root, model["key"])
        if not shared.nine_complete(nroot):
            return model, "nine"
        if not budget_complete(broot):
            return model, "budget"
        if not cleanup_complete(root, model["key"]):
            return model, "cleanup"
    return None


def cleanup_model(root, plan, model):
    key = model["key"]
    nroot, broot = stage_roots(root, key)
    if not shared.nine_complete(nroot) or not budget_complete(broot):
        raise ValueError("Both evaluations must finish before cache cleanup")
    manifest = read(broot / "execution_manifest.json")
    bplan = mini.verify_plan(broot)
    if (manifest["plan_sha256"] != digest(broot / "plan.json")
            or manifest["models"][key]["repo"] != model["repo"]):
        raise ValueError("Completed budget results belong to another model")
    for point_key, budget, dataset in mini.evaluation_points(bplan):
        point = mini.completed_point(broot, point_key, budget, manifest, dataset)
        if (point is None or point.get("seed") != 0 or point.get("ledger_audit") != "passed"
                or point.get("num_prompts") != mini.DATASETS[dataset][1]
                or set(point.get("artifacts", {})) != {"rollouts", "prompts"}):
            raise ValueError("Missing verified budget artifacts")
    scratch = Path(plan["scratch"])
    targets = [scratch / "models" / key, scratch / "stages" / key]
    for target in targets:
        if target.resolve() != target or not target.is_relative_to(scratch) or not scratch.is_absolute() or scratch == Path("/"):
            raise ValueError("Model cache escaped the queue's private scratch")
        if target.exists() and any(p.is_symlink() or p.stat().st_uid != os.getuid() for p in [target, *target.rglob("*")]):
            raise ValueError("Unexpected link or ownership in the model cache")
    receipt_path = root / "model_cache_cleanup" / f"{key}.json"
    receipt = {"state": "cleaning", "model": model["repo"], "revision": manifest["models"][key]["revision"],
               "paths": list(map(str, targets)), "started_at": now(),
               "nine_audit_sha256": digest(nroot / "report/audit.json"),
               "budget_audit_sha256": digest(broot / "report/audit.json")}
    write(receipt_path, receipt)
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
    link = nroot / "model"
    if link.is_symlink() and link.readlink() == targets[0] / "model":
        link.unlink()
    write(receipt_path, {**receipt, "state": "complete", "finished_at": now()})


def environment(plan, key=None):
    selected = {**plan, "scratch": str(Path(plan["scratch"]) / "stages" / key)} if key else plan
    env = shared.environment(selected)
    env["HF_HOME"] = plan["hf_home"]
    for name in ("TMPDIR", "TRITON_CACHE_DIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    return env


def active_workers(plan, model, include_launchers=True):
    stage = Path(plan["output_root"]) / "stages" / model["key"]
    cache = Path(plan["scratch"]) / "models" / model["key"]
    matches = []
    for pid in storage.allocation_processes(plan):
        process = storage.identity(pid)
        if process is None or pid == os.getpid() or process["uid"] != os.getuid():
            continue
        command = process["command"]
        if (any(arg == str(stage) or arg.startswith(str(stage) + "/") or arg.startswith(str(cache) + "/") for arg in command)
                or (include_launchers and MODULE in command and "run-phase" in command and model["key"] in command)):
            matches.append(process)
    return matches


def run_phase(root, plan, key, phase):
    if not training_ready(plan):
        return 75
    action = next_action(root, plan)
    if action is None:
        return 0
    model, expected = action
    if model["key"] != key or expected != phase:
        raise ValueError("Requested phase would skip the prescribed evaluation order")
    nroot, broot = stage_roots(root, key)
    if phase == "budget":
        mini.MODULE = MODULE
        mini.dependency_ready = budget_dependency
        mini.prepare_models = budget_manifest
        return mini.run(broot, mini.verify_plan(broot))
    try:
        with training.holder_locks([root / "run.lock", *plan["holder_locks"]]):
            if not training.gpu_idle():
                return 75
            if phase == "cleanup":
                if active_workers(plan, model, include_launchers=False):
                    return 75
                cleanup_model(root, plan, model)
                return 0
            prepare_model(root, plan, model)
            base = nine.evaluator(nroot)
            runner = nine.load_module("compression_final_nine_runner", nroot / "provenance/eval_rloo_final.py")
            runner.report = nine.generation_complete
            runner.run(nroot, base)
            nine.regrade(nroot, base)
    except BlockingIOError:
        return 75
    return 0


def combined_report(root, plan):
    nine_rows, budget_rows = [], []
    for model in plan["models"]:
        nroot, broot = stage_roots(root, model["key"])
        if not shared.nine_complete(nroot) or not budget_complete(broot) or not cleanup_complete(root, model["key"]):
            raise ValueError("Cannot report an incomplete model")
        nine_rows.extend({**row, "model_key": model["key"]} for row in read(nroot / "report/metrics.json"))
        budget_rows.extend(read(broot / "report/metrics.json"))
    expected = {(model["key"], key, budget) for model in plan["models"] for key in DATASETS for budget in mini.BUDGETS}
    if len(budget_rows) != 70 or {(r["model_key"], r["dataset"], r["budget_tokens"]) for r in budget_rows} != expected:
        raise ValueError("Incomplete two-model seven-dataset budget comparison")
    report = root / "report"
    write(report / "nine_metrics.json", nine_rows)
    write(report / "budget_metrics.json", budget_rows)
    base = nine.evaluator(stage_roots(root, MODEL_ORDER[0])[0])
    base.write_csv(report / "nine_metrics.csv", [{k: v for k, v in row.items() if k != "grader_status"} for row in nine_rows])
    base.write_csv(report / "budget_metrics.csv", budget_rows)
    write(report / "audit.json", {"complete": True, "models": list(MODEL_ORDER), "nine_responses": 14552,
          "budget_points": 70, "all_rollout_ledgers_verified": True,
          "nine_metrics_sha256": digest(report / "nine_metrics.json"), "budget_metrics_sha256": digest(report / "budget_metrics.json")})


def queue(root):
    plan = verify_plan(root)
    children = {}
    with training.holder_locks([root / "queue.lock"]):
        while True:
            state = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": plan["job_id"], "updated_at": now()}
            try:
                verify_plan(root)
                if not training_ready(plan):
                    state["state"] = "waiting_for_f_cov_training_and_uploads"
                else:
                    action = next_action(root, plan)
                    if action is None:
                        combined_report(root, plan)
                        write(root / "queue_status.json", {**state, "state": "complete", "nine_responses": 14552, "budget_points": 70})
                        return 0
                    model, phase = action
                    key = model["key"]
                    state.update(current_model=key, phase=phase)
                    child = children.get((key, phase))
                    if child is not None and child.poll() is not None:
                        state["last_exit_code"] = child.returncode
                        children.pop((key, phase))
                    active = active_workers(plan, model)
                    if not active:
                        command = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                                   f"--nodelist={plan['node']}", "--cpus-per-task=96", "--gres=gpu:8", "--kill-on-bad-exit=1",
                                   f"--job-name=compression-{key}-{phase}", plan["python_bin"], "-u", "-m", MODULE,
                                   "run-phase", "--output-root", str(root), "--model", key, "--phase", phase]
                        log_path = Path(plan["scratch"]) / "logs" / f"{key}_{phase}.log"
                        log_path.parent.mkdir(exist_ok=True)
                        with log_path.open("ab", buffering=0) as log:
                            child = subprocess.Popen(command, cwd=plan["runtime"], env=environment(plan, key),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        children[(key, phase)] = child
                        write(root / "phase_launches" / f"{key}_{phase}.json", {"pid": child.pid, "command": command, "launched_at": now()})
                        active = [{"pid": child.pid}]
                    state.update(state="running_or_waiting_for_gpus", active_pids=[p["pid"] for p in active])
                write(root / "queue_status.json", state)
            except Exception as exc:
                write(root / "queue_status.json", {**state, "state": "retrying", "error": str(exc)})
            time.sleep(30)


def closed_artifacts(root, plan):
    """Only committed response artifacts can move during a storage emergency."""
    for model in plan["models"]:
        nroot, broot = stage_roots(root, model["key"])
        manifest = nroot / "manifest.json"
        if manifest.exists():
            fingerprint = digest(manifest)
            for path in (nroot / "responses").glob("*.receipt.json"):
                receipt = read(path)
                name = receipt["file"]
                if Path(name).name != name or receipt.get("manifest_sha256") != fingerprint:
                    raise ValueError("Invalid closed nine-dataset response receipt")
                yield nroot / "responses" / name, receipt["size"], receipt["sha256"]
        for path in (broot / "results").rglob("summary.json"):
            summary = read(path)
            if summary.get("state") != "complete" or summary.get("ledger_audit") != "passed":
                continue
            for artifact in summary["artifacts"].values():
                relative = Path(artifact["file"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Invalid closed budget artifact path")
                yield path.parent / relative, artifact["size"], artifact["sha256"]


def recover_storage(root, plan):
    """Free results-disk space by moving only verified closed files to this node."""
    local = Path(plan["scratch"])
    free = storage.disk(root)["available_bytes"]
    reclaimed = restored = 0
    if free < plan["storage_critical_bytes"]:
        for source, size, checksum in closed_artifacts(root, plan):
            source.relative_to(root / "stages")
            if source.is_symlink():
                continue
            if source.resolve() != source or source.stat().st_uid != os.getuid():
                raise ValueError("Emergency cleanup must stay inside this queue's results")
            if time.time() - source.stat().st_mtime < 120:
                continue
            if source.stat().st_size != size or digest(source) != checksum:
                raise ValueError("Unverified evaluation results cannot be moved")
            key = mini.fingerprint(str(source))
            destination = local / "offloaded" / key
            copied = storage.copy_checked(source, destination)
            if copied["sha256"] != checksum:
                raise ValueError("Artifact changed while moving to node storage")
            record = {"source": str(source), "local": str(destination), "sha256": checksum, "size": size}
            storage.write(local / "offload_receipts" / f"{key}.json", record)
            link = source.with_name(source.name + ".offload_link")
            if link.is_symlink():
                link.unlink()
            link.symlink_to(destination)
            link.replace(source)
            reclaimed += size
            if reclaimed >= 512 * 1024**2:
                break
    elif free > plan["storage_warning_bytes"]:
        for receipt in (local / "offload_receipts").glob("*.json"):
            record = read(receipt)
            source, saved = Path(record["source"]), Path(record["local"])
            source.relative_to(root / "stages")
            saved.relative_to(local / "offloaded")
            if source.is_symlink():
                if source.readlink() != saved or digest(saved) != record["sha256"]:
                    raise ValueError("Offloaded artifact identity changed")
                temporary = source.with_name(source.name + ".restoring")
                storage.copy_checked(saved, temporary)
                temporary.replace(source)
            if source.is_file() and not source.is_symlink() and digest(source) == record["sha256"]:
                saved.unlink(missing_ok=True)
                receipt.unlink()
                restored += record["size"]
            if restored >= 512 * 1024**2:
                break
    return {"reclaimed_bytes": reclaimed, "restored_bytes": restored}


def serve(root):
    plan = verify_plan(root)
    local = Path(plan["scratch"])
    local.mkdir(exist_ok=True)
    with training.holder_locks([local / "service.lock"]):
        while True:
            verify_plan(root)
            command = [plan["python_bin"], "-u", "-m", MODULE, "queue", "--output-root", str(root)]
            with (local / "queue.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(command, cwd=plan["runtime"], env=environment(plan), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            launch_receipt = {"pid": child.pid, "launched_at": now(), "command": command}
            storage.write(local / "queue_launch.json", launch_receipt)
            try:
                storage.write(root / "queue_launch.json", launch_receipt)
            except OSError as exc:
                if exc.errno not in storage.SPACE_ERRORS:
                    raise
            while child.poll() is None:
                storage.require_node(plan)
                disks = {"results": storage.disk(root), "compute": storage.disk(local), "project": storage.disk("/project/flame")}
                try:
                    recovery = recover_storage(root, plan)
                except Exception as exc:
                    recovery = {"error": str(exc)}
                storage.write(local / "service_status.json", {"pid": os.getpid(), "queue_pid": child.pid,
                              "state": "supervising", "updated_at": now(), "disks": disks, "storage_recovery": recovery})
                mirror = local / "control_mirrors/queue_status.json"
                if mirror.exists() and disks["results"]["available_bytes"] > plan["storage_critical_bytes"]:
                    stamp = max(datetime.fromisoformat(read(mirror)["updated_at"]).timestamp(),
                                datetime.fromisoformat(launch_receipt["launched_at"]).timestamp())
                    process = storage.identity(child.pid)
                    if (time.time() - stamp > 600 and process and process["uid"] == os.getuid()
                            and MODULE in process["command"] and "queue" in process["command"]
                            and str(root) in process["command"]):
                        # The detached Slurm phase is retained and adopted by the next queue.
                        child.send_signal(signal.SIGTERM)
                time.sleep(30)
            status = read(root / "queue_status.json") if (root / "queue_status.json").exists() else {}
            if child.returncode == 0 and status.get("state") == "complete":
                while list((local / "offload_receipts").glob("*.json")):
                    storage.require_node(plan)
                    recover_storage(root, plan)
                    time.sleep(30)
                return 0
            time.sleep(30)


def launch(root, plan):
    environment(plan)
    with training.holder_locks([root / "launch.lock"]):
        if (root / "launch.json").exists():
            raise RuntimeError("Evaluation supervisor is already launched")
        command = [plan["python_bin"], "-u", "-m", MODULE, "serve", "--output-root", str(root)]
        with (Path(plan["scratch"]) / "service.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=plan["runtime"], env=environment(plan), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"pid": child.pid, "job_id": plan["job_id"], "node": plan["node"],
                   "plan_sha256": digest(root / "plan.json"), "launched_at": now()}
        write(root / "launch.json", receipt)
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "serve", "queue", "run-phase", "worker"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--f-cov-training", type=Path)
    parser.add_argument("--maxrl-training", type=Path)
    parser.add_argument("--nine-reference", type=Path)
    parser.add_argument("--budget-reference", type=Path)
    parser.add_argument("--model", choices=MODEL_ORDER)
    parser.add_argument("--phase", choices=("nine", "budget", "cleanup"))
    parser.add_argument("--budget", type=int, choices=mini.BUDGETS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.command == "prepare":
        if not all((args.f_cov_training, args.maxrl_training, args.nine_reference, args.budget_reference)):
            parser.error("Preparation needs both training runs and both reference evaluations")
        plan = prepare(args)
        print(json.dumps({"state": "prepared", "node": plan["node"], "output_root": str(root)}))
        return 0
    if args.command == "worker":
        plan = mini.verify_plan(root)
        if args.model not in plan["models"] or args.rank is None or args.dataset is None or args.budget is None:
            parser.error("Worker requires model, GPU rank, dataset and budget")
        return mini.worker(root, args.model, args.rank, args.budget, args.dataset)
    plan = verify_plan(root)
    global write
    write = followup.protected_writer(root, plan["scratch"], storage.write)
    if args.command == "launch":
        return launch(root, plan)
    if args.command == "serve":
        return serve(root)
    if args.command == "queue":
        return queue(root)
    if args.model is None or args.phase is None:
        parser.error("run-phase requires model and phase")
    return run_phase(root, plan, args.model, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
