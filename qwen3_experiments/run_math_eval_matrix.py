#!/usr/bin/env python3
"""Four-GPU resumable model × protocol × dataset evaluation scheduler."""

from __future__ import annotations

import argparse
import copy
import fcntl
import gzip
import importlib.metadata
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from math_eval_budget_engine import audit_records, evaluate_point
from math_eval_matrix_common import (
    completed_point, fingerprint, make_tasks, point_directory, point_identity,
    read_config, read_json, sha256, task_points, write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER = None
GRADER_TIMEOUT = 1


def init_scorer(timeout):
    global SCORER, GRADER_TIMEOUT
    from verl.workers.reward_manager.multi_thread_naive import MathVerifyScorer

    SCORER, GRADER_TIMEOUT = MathVerifyScorer(), timeout


def score_one(item):
    return SCORER.compute_score(model_output=item[0], ground_truth_unboxed=item[1],
                                timeout_score=0.0, per_item_timeout_s=GRADER_TIMEOUT)


def establish_manifest(config, root):
    prepared = read_json(root / "prepared_inputs.json")
    if prepared["config"] != config:
        raise ValueError("Prepared inputs do not match this matrix config")
    if set(prepared["models"]) != {m["key"] for m in config["models"]} or set(prepared["datasets"]) != {d["key"] for d in config["datasets"]}:
        raise ValueError("Some matrix inputs are not prepared")
    packages = {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers", "math-verify", "datasets")}
    source_paths = [Path(__file__), Path(__file__).with_name("math_eval_budget_engine.py"),
                    Path(__file__).with_name("math_eval_matrix_common.py"),
                    REPO_ROOT / "verl/workers/reward_manager/multi_thread_naive.py"]
    source_hashes = {str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_paths}
    identity = {"config": config, "packages": packages, "source_sha256": source_hashes,
                "models": {key: {k: v for k, v in value.items() if k != "path"} for key, value in prepared["models"].items()},
                "datasets": {key: {k: v for k, v in value.items() if k != "path"} for key, value in prepared["datasets"].items()}}
    manifest = {**identity, "fingerprint": fingerprint(identity)}
    path = root / "manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Existing results use different inputs, code, or settings; use a fresh output root")
    write_json(path, manifest)
    write_json(root / "tasks.json", make_tasks(config))
    return manifest, prepared


def point_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def reuse_eval1_points(source, root, manifest):
    """Explicitly reuse audited Eval1 points when only Eval2 early stopping changed.

    Keep the source run immutable and record its manifest/summary hashes. Source
    changes in the runner/budget engine are expected for this Eval2-only update;
    the shared seed/identity helper and grader must still match exactly.
    """
    source = source.resolve()
    if source == root.resolve():
        raise ValueError("Eval1 reuse requires a different output root")
    previous = read_json(source / "manifest.json")
    for item in (previous, manifest):
        if item["fingerprint"] != fingerprint({k: v for k, v in item.items() if k != "fingerprint"}):
            raise ValueError("Invalid manifest fingerprint")
    configs = [copy.deepcopy(item["config"]) for item in (previous, manifest)]
    for config in configs:
        config["evals"]["eval2"].pop("stop_on_first_success", None)
    if configs[0] != configs[1]:
        raise ValueError("Eval1 reuse requires identical inputs and settings except Eval2 early stopping")
    for key in ("packages", "models", "datasets"):
        if previous[key] != manifest[key]:
            raise ValueError(f"Eval1 reuse requires identical {key}")
    for path in ("qwen3_experiments/math_eval_matrix_common.py", "verl/workers/reward_manager/multi_thread_naive.py"):
        if previous["source_sha256"][path] != manifest["source_sha256"][path]:
            raise ValueError(f"Eval1 reuse requires the same seed/identity helper and grader: {path}")
    prepared = read_json(root / "prepared_inputs.json")
    imported = 0
    for task in make_tasks(manifest["config"]):
        if task["protocol"] != "eval1":
            continue
        dataset = prepared["datasets"][task["dataset"]]
        if sha256(Path(dataset["path"])) != dataset["sha256"]:
            raise ValueError("Dataset changed before Eval1 reuse")
        rows = read_json(Path(dataset["path"]))
        for point in task_points(manifest["config"], task):
            if completed_point(root, point, manifest) is not None:
                continue
            summary = completed_point(source, point, previous)
            if summary is None:
                continue
            origin = point_directory(source, point)
            audit, prompts = audit_records(point_records(origin / summary["artifacts"]["rollouts"]["file"]),
                                           protocol="eval1", budget=point["budget"], seed=point["seed"], rows=rows,
                                           per_rollout_cap=manifest["config"]["sampling"]["per_rollout_cap"])
            if any(summary[key] != value for key, value in audit.items()) or prompts != read_json(origin / summary["artifacts"]["prompts"]["file"]):
                raise ValueError("Source Eval1 ledger failed independent audit")
            destination = point_directory(root, point)
            for artifact in summary["artifacts"].values():
                relative = Path(artifact["file"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe source artifact path")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(origin / relative, target)
            summary["reused_from"] = {"output_root": str(source), "manifest_fingerprint": previous["fingerprint"],
                                      "summary_sha256": sha256(origin / "summary.json"),
                                      "reason": "Eval1 unchanged; Eval2 stop-on-first-success update"}
            summary["identity"] = point_identity(point, manifest)
            write_json(destination / "summary.json", summary)
            completed_point(root, point, manifest)
            imported += 1
    print(f"Reused {imported} independently audited Eval1 points from {source}", flush=True)
    return imported


def run_task(args, config, manifest, prepared):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tasks = {task["id"]: task for task in make_tasks(config)}
    task = tasks[args.task]
    stop_on_first_success = config["evals"][task["protocol"]].get("stop_on_first_success", False)
    if not isinstance(stop_on_first_success, bool):
        raise ValueError("stop_on_first_success must be a boolean")
    root = args.output_root
    model = prepared["models"][task["model"]]
    dataset = prepared["datasets"][task["dataset"]]
    dataset_path = Path(dataset["path"])
    if sha256(dataset_path) != dataset["sha256"]:
        raise ValueError("Dataset changed since preparation")
    rows = read_json(dataset_path)
    base_key = next(m["key"] for m in config["models"] if m["format"] == "huggingface")
    tokenizer_path = prepared["models"][base_key]["path"]
    # All five models use exactly the same ordinary prompt and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompts = [tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True) for row in rows]
    longest = max(map(len, prompts))
    if longest > config["sampling"]["max_prompt_len"]:
        raise ValueError(f"Prompt would exceed configured limit: {longest}; no truncation is allowed")
    pending = [point for point in task_points(config, task) if completed_point(root, point, manifest) is None]
    if not pending:
        return 0
    sampler = config["sampling"]
    engine = LLM(model=model["path"], tokenizer=tokenizer_path, tensor_parallel_size=1,
                 dtype="bfloat16", gpu_memory_utilization=sampler["gpu_memory_utilization"],
                 max_model_len=longest + max(sampler["per_rollout_cap"], max(config["evals"]["eval1"]["budgets"])),
                 max_num_batched_tokens=sampler["max_num_batched_tokens"],
                 max_num_seqs=sampler["max_batch_size"], enforce_eager=False,
                 enable_chunked_prefill=True, enable_prefix_caching=True,
                 disable_log_stats=True, seed=0, trust_remote_code=False,
                 generation_config="vllm")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=config["grader"]["workers"], mp_context=context,
                             initializer=init_scorer, initargs=(config["grader"]["timeout_seconds"],)) as pool:
        for point in pending:
            if (root / "PAUSE").exists():
                return 75
            if shutil.disk_usage(root).free < 5 * 2**30:
                raise RuntimeError("Less than 5 GiB available for persistent results; refusing to generate more")
            directory = point_directory(root, point)
            attempt = directory / ("attempt_" + uuid.uuid4().hex[:12])
            attempt.mkdir(parents=True)
            rollouts = attempt / "rollouts.jsonl.gz"
            started = time.time()
            progress_time = 0.0

            def progress(counters):
                nonlocal progress_time
                if time.monotonic() - progress_time >= 30:
                    state = {"state": "running", "point": point, "gpu": args.gpu, "pid": os.getpid(),
                             "started_at": started, "updated_at": time.time(), **counters}
                    write_json(root / "progress" / f"gpu_{args.gpu}.json", state)
                    print(json.dumps(state), flush=True)
                    progress_time = time.monotonic()

            print(f"START {point}", flush=True)
            with gzip.open(rollouts, "wt", encoding="utf-8", compresslevel=3) as stream:
                def emit(record):
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

                summary, prompts_summary = evaluate_point(
                    protocol=point["protocol"], budget=point["budget"], seed=point["seed"],
                    rows=rows, prompt_token_ids=prompts, engine=engine,
                    sampling_params_type=SamplingParams, tokenizer=tokenizer,
                    score_many=lambda items: pool.map(score_one, items, chunksize=1),
                    emit=emit, sampling=sampler, progress=progress,
                    stop_on_first_success=stop_on_first_success)
            audit, audited_prompts = audit_records(point_records(rollouts), protocol=point["protocol"],
                                                   budget=point["budget"], seed=point["seed"], rows=rows,
                                                   per_rollout_cap=sampler["per_rollout_cap"],
                                                   stop_on_first_success=stop_on_first_success)
            if audited_prompts != prompts_summary or any(summary[key] != value for key, value in audit.items()):
                raise ValueError("Saved response ledger does not reproduce the computed result")
            prompts_path = attempt / "prompts.json"
            write_json(prompts_path, prompts_summary)
            summary.update({"status": "complete", "identity": point_identity(point, manifest),
                            "started_at": started, "completed_at": time.time(),
                            "elapsed_seconds": time.time() - started, "longest_prompt_tokens": longest,
                            "raw_rollout_audit": "passed", "artifacts": {
                                name: {"file": str(path.relative_to(directory)), "size": path.stat().st_size, "sha256": sha256(path)}
                                for name, path in (("rollouts", rollouts), ("prompts", prompts_path))}})
            write_json(directory / "summary.json", summary)
            write_json(root / "progress" / f"gpu_{args.gpu}.json", {"state": "point_complete", "point": point,
                       "gpu": args.gpu, "updated_at": time.time(), "summary": str(directory / "summary.json")})
            print(f"COMPLETE {point}: solved={summary['num_questions_solved']}/{len(rows)}, mean@4={summary['mean_at_4_accuracy']}", flush=True)
    return 0


def worker(args, config, manifest):
    root = args.output_root
    for task in make_tasks(config):
        if (root / "PAUSE").exists():
            break
        lock_path = root / "task_state" / f"{task['id']}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            failure = lock_path.with_suffix(".failed.json")
            if failure.exists() and not args.retry_failed:
                continue
            if all(completed_point(root, point, manifest, check_hashes=False) is not None for point in task_points(config, task)):
                continue
            stamp = time.strftime("%Y%m%d_%H%M%S")
            log_path = root / "logs" / f"{task['id']}_{stamp}_gpu{args.gpu}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config),
                       "--output-root", str(root), "--scratch", str(args.scratch),
                       "--task", task["id"], "--gpu", str(args.gpu)]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            ipc = args.scratch / f"g{args.gpu}"
            ipc.mkdir(parents=True, exist_ok=True)
            env.update({"TMPDIR": str(ipc), "TRITON_CACHE_DIR": str(ipc / "triton"),
                        "VLLM_CACHE_ROOT": str(ipc / "vllm"), "TORCHINDUCTOR_CACHE_DIR": str(ipc / "inductor")})
            with log_path.open("w") as stream:
                child = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                         pass_fds=(lock.fileno(),))
                write_json(root / "progress" / f"gpu_{args.gpu}.json", {"state": "loading", "task": task,
                           "gpu": args.gpu, "pid": child.pid, "log": str(log_path), "updated_at": time.time()})
                code = child.wait()
            if code == 75:
                break
            if code:
                write_json(failure, {"task": task, "exit_code": code, "log": str(log_path), "time": time.time()})
                print(f"FAILED {task['id']}: {log_path}", flush=True)
            else:
                write_json(lock_path.with_suffix(".done.json"), {"task": task, "log": str(log_path), "time": time.time()})
    write_json(root / "progress" / f"gpu_{args.gpu}.json", {"state": "worker_finished", "gpu": args.gpu, "updated_at": time.time()})
    return 0


def report_status(args, config, manifest, children):
    points = [point for task in make_tasks(config) for point in task_points(config, task)]
    complete = sum(completed_point(args.output_root, point, manifest, check_hashes=False) is not None for point in points)
    failed = [read_json(path) for path in (args.output_root / "task_state").glob("*.failed.json")]
    state = "complete" if complete == len(points) else ("running" if any(child.poll() is None for child in children) else "incomplete")
    result = {"state": state, "completed_points": complete, "total_points": len(points), "failed_tasks": failed,
              "worker_exit_codes": [child.poll() for child in children],
              "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": os.uname().nodename,
              "updated_at": time.time(), "manifest_fingerprint": manifest["fingerprint"]}
    write_json(args.output_root / "status.json", result)
    return result


def coordinate(args, config, manifest):
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
    if busy:
        raise RuntimeError("GPUs are occupied; refusing to overlap another workload")
    if len(args.gpus) != len(set(args.gpus)):
        raise ValueError("GPU IDs must be unique")
    args.output_root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    children, streams = [], []

    def interrupted(signum, frame):
        # Only signal the process groups created by this scheduler.
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for gpu in args.gpus:
            stream = (args.output_root / "logs" / f"worker_gpu{gpu}_{int(time.time())}.log").open("w")
            streams.append(stream)
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config),
                       "--output-root", str(args.output_root), "--scratch", str(args.scratch), "--worker", "--gpu", str(gpu)]
            if args.retry_failed:
                command.append("--retry-failed")
            children.append(subprocess.Popen(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
        last_plot = 0.0
        while True:
            status = report_status(args, config, manifest, children)
            print(json.dumps({key: value for key, value in status.items() if key != "failed_tasks"}), flush=True)
            finished = all(child.poll() is not None for child in children)
            if args.plot_python and (finished or time.monotonic() - last_plot >= 300):
                result = subprocess.run([str(args.plot_python), str(Path(__file__).with_name("plot_math_eval_matrix.py")),
                                         "--output-root", str(args.output_root)], cwd=REPO_ROOT, timeout=120, check=False)
                if result.returncode:
                    print(f"Report generation failed with code {result.returncode}; raw results are retained", flush=True)
                last_plot = time.monotonic()
            if finished:
                return 0 if status["state"] == "complete" else 1
            time.sleep(30)
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for stream in streams:
            stream.close()
        report_status(args, config, manifest, children)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--reuse-eval1-from", type=Path)
    parser.add_argument("--plot-python", type=Path)
    args = parser.parse_args()
    args.config, args.output_root, args.scratch = args.config.resolve(), args.output_root.resolve(), args.scratch.resolve()
    config = read_config(args.config)
    if args.worker or args.task:
        manifest, prepared = read_json(args.output_root / "manifest.json"), read_json(args.output_root / "prepared_inputs.json")
        if manifest["config"] != config or prepared["config"] != config:
            raise ValueError("Config changed while this matrix was running")
        return run_task(args, config, manifest, prepared) if args.task else worker(args, config, manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, _ = establish_manifest(config, args.output_root)
        if args.reuse_eval1_from:
            reuse_eval1_points(args.reuse_eval1_from, args.output_root, manifest)
        return coordinate(args, config, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
