#!/usr/bin/env python3
"""Run the existing MATH-500 evaluators for offset256 without changing cap8 results."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL = "fixed_n_rb_offset256_step150"
CHECKPOINT = "zjhhhh/fixed-n-rb-offset256-qwen3-1.7b-base-math12k-754f8ca2-step_150"
REVISION = "8c299693da88a1a686933bdd77a4b812b0dcc118"
BASELINE_LABEL = "fixed_n_rb_capped_cap8_token_mean_step150"
BASELINE_REPO = "zjhhhh/fixed-n-rb-capped-cost-aware-marginrl-qwen3-1.7b-base-math12k-cap8-token-mean-step_150"
DATASET_SHA256 = "7e674a2fb0e85770931fed4b467c7236a674adb4af58f6c426fa9fd60d736af3"
PACKAGES = {
    "torch": "2.6.0+cu124",
    "vllm": "0.8.4",
    "transformers": "4.51.3",
    "math-verify": "0.9.0",
    "datasets": "3.5.0",
}
PROTOCOLS = {
    "eval1": ("eval_math500_step100", (256, 512, 1024, 2048, 4096), "mean_at_4_accuracy"),
    "eval2": ("eval_math500_total_budget_12288", (2048, 4096, 8192, 12288), "pass_at_realized_list_size"),
    "eval3_skip_solved": ("eval_math500_cross_context_budget", (256, 512, 1024, 2048, 4096), "fraction_solved"),
    "eval3_iid": ("eval_math500_cross_context_random_replacement", (256, 512, 1024, 2048, 4096), "fraction_solved"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def result_stem(protocol: str, budget: int, label: str = LABEL) -> str:
    if protocol == "eval1":
        return f"{label}_max_tokens_{budget}"
    if protocol == "eval2":
        return f"{label}_total_budget_{budget}_rollout_cap_4096"
    stem = f"{label}_budget_per_prompt_{budget}_global_budget_{500 * budget}_rollout_cap_4096"
    if protocol == "eval3_iid":
        stem += "_question_selection_random_with_replacement"
    return stem


def result_paths(directory: Path, protocol: str, budget: int, label: str = LABEL) -> list[Path]:
    stem = result_stem(protocol, budget, label)
    suffixes = ("samples.jsonl", "summary.json") if protocol == "eval1" else ("rollouts.jsonl", "prompts.jsonl", "summary.json")
    return [directory / f"{stem}_{suffix}" for suffix in suffixes]


def validate_summary(directory: Path, protocol: str, budget: int, label: str = LABEL) -> dict:
    paths = result_paths(directory, protocol, budget, label)
    if not all(path.is_file() and path.stat().st_size for path in paths):
        raise ValueError(f"Missing or incomplete result: {paths[-1]}")
    summary = json.loads(paths[-1].read_text())
    expected = {
        "model_label": label,
        "checkpoint_repo": CHECKPOINT if label == LABEL else BASELINE_REPO,
        "dataset": "HuggingFaceH4/MATH-500",
        "dataset_sha256": DATASET_SHA256,
        "num_prompts": 500,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": -1,
        "seed": 0,
        "max_prompt_len": 1024,
        "longest_prompt_tokens": 816,
        "grader": "verl.workers.reward_manager.multi_thread_naive.MathVerifyScorer",
        "grader_timeout_seconds": 1,
        "packages": PACKAGES,
    }
    if protocol == "eval1":
        expected.update(max_output_len=budget, num_samples_per_prompt=4, num_scored_responses=2000)
    else:
        expected["per_rollout_output_cap"] = 4096
        if protocol == "eval2":
            expected.update(total_output_budget_per_prompt=budget, total_generated_output_tokens=500 * budget)
        else:
            expected.update(budget_per_prompt_reference=budget, total_global_output_budget=500 * budget)
            if protocol == "eval3_iid":
                expected.update(question_selection="random-with-replacement", success_aware_skipping=False, question_draws_include_already_solved_prompts=True)
            else:
                expected["evaluation"] = "eval4_cross_context_global_budget"
                # Early sweep summaries predate this flag; the raw audit still
                # verifies that none of their solved questions were resampled.
                if label == LABEL or "success_aware_skipping" in summary:
                    expected["success_aware_skipping"] = True
            if label == LABEL:
                expected["checkpoint_revision"] = REVISION
            if not (summary.get("global_budget_exhausted") or summary.get("all_prompts_solved_early")):
                raise ValueError(f"Budget was not exhausted: {paths[-1]}")
    mismatches = {key: (summary.get(key), value) for key, value in expected.items() if summary.get(key) != value}
    if mismatches:
        raise ValueError(f"Result configuration mismatch: {paths[-1]}: {mismatches}")
    return summary


def command(args: argparse.Namespace, protocol: str, budget: int) -> list[str]:
    scripts = {
        "eval1": "eval_math500_output_budget.py",
        "eval2": "eval_math500_total_token_budget.py",
        "eval3_skip_solved": "eval_math500_cross_context_budget.py",
        "eval3_iid": "eval_math500_cross_context_budget.py",
    }
    cmd = [
        str(args.python),
        str(REPO_ROOT / "qwen3_experiments" / scripts[protocol]),
        "--model-path",
        str(args.model_path),
        "--model-label",
        LABEL,
        "--checkpoint-repo",
        CHECKPOINT,
        "--dataset",
        str(REPO_ROOT / "data/math500/test.parquet"),
        "--output-dir",
        str(args.output_root / protocol / "results"),
        "--max-prompt-len",
        "1024",
        "--grader-workers",
        "8",
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k=-1",
        "--seed",
        "0",
    ]
    if protocol == "eval1":
        cmd += ["--max-output-len", str(budget), "--num-samples", "4"]
    elif protocol == "eval2":
        cmd += ["--total-output-budget", str(budget), "--per-rollout-cap", "4096"]
    else:
        selection = "sweep" if protocol == "eval3_skip_solved" else "random-with-replacement"
        cmd += ["--budget-per-prompt", str(budget), "--per-rollout-cap", "4096", "--max-batch-size", "512", "--question-selection", selection, "--checkpoint-revision", REVISION, "--grader-timeout", "1", "--gpu-memory-utilization", "0.7"]
    return cmd


def idle_gpus() -> set[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    idle = set()
    for line in result.stdout.splitlines():
        index, memory, utilization = (int(part.strip()) for part in line.split(","))
        if memory < 512 and utilization == 0:
            idle.add(index)
    return idle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--ipc-root", type=Path, required=True, help="Short local directory for Unix-domain sockets")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for key in ("python", "model_path", "artifact_root", "ipc_root", "output_root", "report_python"):
        setattr(args, key, getattr(args, key).resolve())
    if len(os.fsencode(str(args.ipc_root))) + 48 >= 107:
        raise ValueError("--ipc-root is too long for vLLM Unix-domain socket paths")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (args.model_path / "config.json").is_file() or not list(args.model_path.glob("*.safetensors")):
            raise ValueError("The offset256 model must be merged before evaluation")
        if sha256(REPO_ROOT / "data/math500/test.parquet") != DATASET_SHA256:
            raise ValueError("MATH-500 changed since the cap8 evaluation")
        version_code = "import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in " + repr(list(PACKAGES)) + "}))"
        environment = {**os.environ, "PYTHONNOUSERSITE": "1", "XDG_CACHE_HOME": str(args.artifact_root / "runtime" / "common")}
        versions = json.loads(subprocess.check_output([str(args.python), "-c", version_code], env=environment, text=True))
        if versions != PACKAGES:
            raise ValueError(f"Evaluation environment differs from cap8: {versions}")

        baseline_hashes = {}
        jobs = []
        complete = []
        for protocol, (baseline_dir, budgets, _) in PROTOCOLS.items():
            directory = args.output_root / protocol / "results"
            directory.mkdir(parents=True, exist_ok=True)
            for budget in budgets:
                baseline = REPO_ROOT / "outputs" / baseline_dir / "results"
                validate_summary(baseline, protocol, budget, BASELINE_LABEL)
                for path in result_paths(baseline, protocol, budget, BASELINE_LABEL):
                    baseline_hashes[str(path)] = sha256(path)
                paths = result_paths(directory, protocol, budget)
                job = {"protocol": protocol, "budget": budget, "key": f"{protocol}_{budget}"}
                if any(path.exists() for path in paths):
                    validate_summary(directory, protocol, budget)
                    complete.append(job)
                else:
                    jobs.append(job)
        # Start expensive points first; each request retains the historical seed.
        factors = {"eval1": 0.12, "eval2": 0.12, "eval3_skip_solved": 0.32, "eval3_iid": 0.28}
        jobs.sort(key=lambda job: job["budget"] * factors[job["protocol"]], reverse=True)
        scripts = ["eval_math500_output_budget.py", "eval_math500_total_token_budget.py", "eval_math500_cross_context_budget.py"]
        manifest = {
            "checkpoint_repo": CHECKPOINT,
            "checkpoint_revision": REVISION,
            "model_path": str(args.model_path),
            "packages": versions,
            "dataset_sha256": DATASET_SHA256,
            "model_artifact_sha256": {path.name: sha256(path) for path in sorted(args.model_path.iterdir()) if path.is_file()},
            "baseline_artifact_sha256": baseline_hashes,
            "gpus": args.gpus,
            "script_sha256": {name: sha256(REPO_ROOT / "qwen3_experiments" / name) for name in scripts},
            "commands": {job["key"]: command(args, job["protocol"], job["budget"]) for job in jobs},
        }
        write_json(args.output_root / "manifest.json", manifest)
        active = {}
        failed = []
        previously_idle = set()
        while (jobs and not failed) or active:
            for gpu, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                job = item["job"]
                if code == 0:
                    try:
                        validate_summary(args.output_root / job["protocol"] / "results", job["protocol"], job["budget"])
                    except Exception as exc:
                        code = 1
                        job["error"] = str(exc)
                (complete if code == 0 else failed).append({**job, "returncode": code})
                print(f"Finished {job['key']} on GPU {gpu}: exit={code}", flush=True)
                del active[gpu]

            available = idle_gpus().intersection(args.gpus).difference(active)
            disk_ok = shutil.disk_usage(args.artifact_root).free > 20 * 2**30 and shutil.disk_usage(args.output_root).free > 5 * 2**30
            for gpu in sorted(available.intersection(previously_idle)) if disk_ok else []:
                if not jobs or failed:
                    break
                job = jobs.pop(0)
                runtime = args.artifact_root / "runtime" / f"gpu_{gpu}"
                runtime.mkdir(parents=True, exist_ok=True)
                ipc_directory = args.ipc_root / f"g{gpu}"
                ipc_directory.mkdir(parents=True, exist_ok=True)
                env = {
                    **environment,
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "TOKENIZERS_PARALLELISM": "false",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
                    "XDG_CACHE_HOME": str(runtime),
                    "TMPDIR": str(ipc_directory),
                    "TRITON_CACHE_DIR": str(runtime / "triton"),
                    "TORCHINDUCTOR_CACHE_DIR": str(runtime / "inductor"),
                    "VLLM_CACHE_ROOT": str(runtime / "vllm"),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HUB_CACHE": str(args.artifact_root / "hf-hub"),
                    "HF_XET_CACHE": str(args.artifact_root / "hf-xet"),
                    "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                }
                log_path = args.output_root / f"{job['key']}.log"
                log = log_path.open("a")
                process = subprocess.Popen(command(args, job["protocol"], job["budget"]), cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = {"job": job, "process": process, "log": log}
                print(f"Started {job['key']} on GPU {gpu}: pid={process.pid}", flush=True)
            previously_idle = available
            write_json(
                args.output_root / "status.json",
                {
                    "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "supervisor_pid": os.getpid(),
                    "complete": complete,
                    "failed": failed,
                    "pending": jobs,
                    "active": [{**item["job"], "gpu": gpu, "pid": item["process"].pid} for gpu, item in active.items()],
                    "disk_ok_for_new_jobs": disk_ok,
                },
            )
            if jobs or active:
                time.sleep(10)
        for filename, expected_hash in baseline_hashes.items():
            if sha256(Path(filename)) != expected_hash:
                raise RuntimeError(f"A baseline artifact changed during evaluation: {filename}")
        if failed:
            raise RuntimeError(f"Some evaluation points failed; inspect status.json: {failed}")
        subprocess.run([str(args.report_python), str(REPO_ROOT / "qwen3_experiments/summarize_math500_offset256_comparison.py"), "--output-root", str(args.output_root)], cwd=REPO_ROOT, env=environment, check=True)
        print("All 19 offset256 points completed, baseline artifacts unchanged, comparison report written.", flush=True)


if __name__ == "__main__":
    main()
