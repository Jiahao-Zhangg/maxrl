#!/usr/bin/env python3
"""Resume a historical MATH-500 Eval1 or Eval2 series on one dedicated GPU."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path

from run_math500_offset256_comparison import BASELINE_LABEL, BASELINE_REPO, DATASET_SHA256, PACKAGES, PROTOCOLS, REPO_ROOT, idle_gpus, result_paths, sha256, write_json


def audit_point(directory: Path, protocol: str, budget: int, label: str, checkpoint: str) -> dict:
    paths = result_paths(directory, protocol, budget, label)
    if not all(path.is_file() and path.stat().st_size for path in paths):
        raise ValueError(f"Incomplete result set: {paths[-1]}")
    summary = json.loads(paths[-1].read_text())
    expected = {
        "model_label": label,
        "checkpoint_repo": checkpoint,
        "dataset": "HuggingFaceH4/MATH-500",
        "dataset_sha256": DATASET_SHA256,
        "num_prompts": 500,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": -1,
        "seed": 0,
        "max_prompt_len": 1024,
        "longest_prompt_tokens": 816,
        "packages": PACKAGES,
        "grader": "verl.workers.reward_manager.multi_thread_naive.MathVerifyScorer",
        "grader_timeout_seconds": 1,
    }
    if protocol == "eval1":
        expected.update(max_output_len=budget, num_samples_per_prompt=4, num_scored_responses=2000)
    else:
        expected.update(total_output_budget_per_prompt=budget, per_rollout_output_cap=4096, total_generated_output_tokens=500 * budget)
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"{paths[-1]}: {key}={summary.get(key)!r}; expected {value!r}")
    attempts, lengths, successes = Counter(), Counter(), Counter()
    slots = set()
    response_count, score_sum = 0, 0
    with paths[0].open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row["score"] in (0, 1)
            length = row["output_tokens"]
            if protocol == "eval1":
                position = row["prompt_index"]
                assert 0 <= row["sample_index"] < 4 and 0 < length <= budget
                slot = (position, row["sample_index"])
                assert slot not in slots
                slots.add(slot)
            else:
                position = row["prompt_position"]
                assert 0 <= position < 500
                assert row["rollout_index"] == attempts[position]
                assert row["rollout_seed"] == (1_000_003 * position + attempts[position]) % (2**31 - 1)
                remaining = budget - lengths[position]
                assert row["max_output_tokens"] == min(4096, remaining)
                assert 0 < length <= row["max_output_tokens"]
                assert row["remaining_budget_before"] == remaining
                assert row["remaining_budget_after"] == remaining - length
                assert row["cumulative_output_tokens"] == lengths[position] + length
            attempts[position] += 1
            lengths[position] += length
            successes[position] += int(row["score"])
            response_count += 1
            score_sum += row["score"]
    if protocol == "eval1":
        assert response_count == 2000 and len(attempts) == 500 and set(attempts.values()) == {4}
        assert score_sum == summary["correct_responses"]
        accuracy = score_sum / 2000
    else:
        prompts = [json.loads(line) for line in paths[1].read_text().splitlines()]
        assert len(prompts) == 500 and {row["prompt_position"] for row in prompts} == set(range(500))
        for row in prompts:
            position = row["prompt_position"]
            assert row["list_size"] == attempts[position]
            assert row["total_output_tokens"] == lengths[position] == budget
            assert row["num_successes"] == successes[position]
            assert row["passed"] == (successes[position] > 0)
        assert response_count == summary["total_rollouts"]
        assert sum(lengths.values()) == 500 * budget
        passed = sum(value > 0 for value in successes.values())
        assert passed == summary["num_prompts_passed"]
        accuracy = passed / 500
    assert math.isclose(accuracy, summary[PROTOCOLS[protocol][2]], abs_tol=1e-12)
    return {"accuracy": accuracy, "response_count": response_count, "mean_response_tokens": sum(lengths.values()) / response_count, "raw_rollout_audit": "passed", "summary_file": str(paths[-1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("eval1", "eval2"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    for option in ("python", "model-path", "artifact-root", "ipc-root", "output-root", "source-manifest"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("model-label", "checkpoint-repo", "checkpoint-revision"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    for key in ("python", "model_path", "artifact_root", "ipc_root", "output_root", "source_manifest"):
        setattr(args, key, getattr(args, key).resolve())
    if len(os.fsencode(str(args.ipc_root))) + 48 >= 107:
        raise ValueError("Use a shorter --ipc-root for vLLM Unix-domain sockets")
    root = args.output_root / args.protocol
    directory = root / "results"
    runtime = args.artifact_root / "runtime" / f"gpu{args.gpu}"
    for path in (directory, runtime, args.ipc_root):
        path.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "TMPDIR": str(args.ipc_root),
        "XDG_CACHE_HOME": str(runtime),
        "TRITON_CACHE_DIR": str(runtime / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(runtime / "inductor"),
        "VLLM_CACHE_ROOT": str(runtime / "vllm"),
        "HF_HUB_CACHE": str(args.artifact_root / "hf-hub"),
        "HF_XET_CACHE": str(args.artifact_root / "hf-xet"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    baseline_dir, budgets, _ = PROTOCOLS[args.protocol]
    evaluator = REPO_ROOT / "qwen3_experiments" / ("eval_math500_output_budget.py" if args.protocol == "eval1" else "eval_math500_total_token_budget.py")
    dataset = REPO_ROOT / "data/math500/test.parquet"
    baseline, results = {}, {}
    state = {"state": "preparing", "protocol": args.protocol, "gpu": args.gpu, "supervisor_pid": os.getpid(), "active_pid": None, "failed": []}

    def refresh() -> None:
        for budget in budgets:
            summary = result_paths(directory, args.protocol, budget, args.model_label)[-1]
            if str(budget) not in results and summary.is_file():
                results[str(budget)] = audit_point(directory, args.protocol, budget, args.model_label, args.checkpoint_repo)
        state.update(updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), completed_points=len(results), total_points=len(budgets))
        write_json(root / "status.json", state)
        write_json(root / "results.json", {"model": results, "cap8": baseline})
        description = "Single-rollout accuracy (mean@4)" if args.protocol == "eval1" else "Per-question list success rate (at least one correct answer)"
        lines = [
            f"# MATH-500 {args.protocol}: {description}",
            "",
            f"Model: [{args.model_label}](https://huggingface.co/{args.checkpoint_repo}), revision `{args.checkpoint_revision}`.",
            "",
            "All 500 questions; temperature 0.6, top-p 0.95, top-k -1, seed 0, historical MathVerify scorer with one-second timeout. "
            + ("Each question has four samples at each output cap." if args.protocol == "eval1" else "Each question consumes its own total output-token budget, including after a success; each response is capped at 4096 or its remaining budget."),
            "",
            "| Model | " + " | ".join(map(str, budgets)) + " |",
            "|---|" + "---:|" * len(budgets),
            "| Cap8 | " + " | ".join(f"{baseline[str(b)]['accuracy']:.2%}" for b in budgets) + " |",
            f"| {args.model_label} | " + " | ".join(f"{results[str(b)]['accuracy']:.2%}" if str(b) in results else "pending" for b in budgets) + " |",
            "| Difference vs Cap8 (pp) | " + " | ".join(f"{100 * (results[str(b)]['accuracy'] - baseline[str(b)]['accuracy']):+.2f}" if str(b) in results else "pending" for b in budgets) + " |",
            "",
            f"Validated budget points: {len(results)}/{len(budgets)}. All responses and per-question results are retained.",
            "",
        ]
        temporary = root / "results.md.tmp"
        temporary.write_text("\n".join(lines))
        temporary.replace(root / "results.md")

    def interrupted(signum, frame) -> None:
        raise InterruptedError(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    child = None
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            source = json.loads(args.source_manifest.read_text())
            if source["checkpoint_repo"] != args.checkpoint_repo or source["checkpoint_revision"] != args.checkpoint_revision:
                raise ValueError("Prepared model provenance does not match the requested checkpoint")
            model_hashes = {p.name: sha256(p) for p in sorted(args.model_path.iterdir()) if p.is_file()}
            if model_hashes != source["model_sha256"]:
                raise ValueError("Prepared model differs from its verified Eval3 snapshot")
            if sha256(dataset) != DATASET_SHA256:
                raise ValueError("MATH-500 has changed")
            version_code = "import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in " + repr(list(PACKAGES)) + "}))"
            versions = json.loads(subprocess.check_output([str(args.python), "-c", version_code], env=environment, text=True))
            if versions != PACKAGES:
                raise ValueError(f"Evaluation packages differ from the historical snapshot: {versions}")
            baseline_hashes = {}
            for budget in budgets:
                baseline_path = REPO_ROOT / "outputs" / baseline_dir / "results"
                baseline[str(budget)] = audit_point(baseline_path, args.protocol, budget, BASELINE_LABEL, BASELINE_REPO)
                baseline_hashes.update({str(p): sha256(p) for p in result_paths(baseline_path, args.protocol, budget, BASELINE_LABEL)})
            manifest = {
                "checkpoint_repo": args.checkpoint_repo,
                "checkpoint_revision": args.checkpoint_revision,
                "model_sha256": model_hashes,
                "packages": versions,
                "dataset_sha256": DATASET_SHA256,
                "evaluator_sha256": sha256(evaluator),
                "baseline_sha256": baseline_hashes,
                "gpu": args.gpu,
                "commands": {},
            }
            manifest_path = root / "manifest.json"
            if manifest_path.is_file():
                previous = json.loads(manifest_path.read_text())
                for key, value in manifest.items():
                    if key != "commands" and previous.get(key) != value:
                        raise ValueError(f"Cannot resume: manifest field {key} changed")
                manifest["commands"] = previous.get("commands", {})
            write_json(manifest_path, manifest)
            refresh()
            for budget in budgets:
                if str(budget) in results:
                    continue
                state.update(state="waiting_for_gpu", budget=budget, active_pid=None)
                refresh()
                idle_checks = 0
                while idle_checks < 2:
                    disk_ok = shutil.disk_usage(args.artifact_root).free > 20 * 2**30 and shutil.disk_usage(root).free > 5 * 2**30
                    idle_checks = idle_checks + 1 if disk_ok and args.gpu in idle_gpus() else 0
                    if idle_checks < 2:
                        time.sleep(10)
                        refresh()
                refresh()
                if str(budget) in results:
                    continue
                if any(p.exists() for p in result_paths(directory, args.protocol, budget, args.model_label)):
                    raise ValueError(f"Refusing to overwrite incomplete output for budget {budget}")
                if sha256(evaluator) != manifest["evaluator_sha256"]:
                    raise ValueError("The evaluator changed during the run")
                cmd = [
                    str(args.python),
                    str(evaluator),
                    "--model-path",
                    str(args.model_path),
                    "--model-label",
                    args.model_label,
                    "--checkpoint-repo",
                    args.checkpoint_repo,
                    "--dataset",
                    str(dataset),
                    "--output-dir",
                    str(directory),
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
                cmd += ["--max-output-len", str(budget), "--num-samples", "4"] if args.protocol == "eval1" else ["--total-output-budget", str(budget), "--per-rollout-cap", "4096"]
                manifest["commands"][str(budget)] = cmd
                write_json(manifest_path, manifest)
                with (root / f"budget_{budget}.log").open("a") as log:
                    child = subprocess.Popen(cmd, cwd=REPO_ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                    state.update(state="running", active_pid=child.pid)
                    print(f"Started {args.protocol} b={budget} on GPU {args.gpu}: pid={child.pid}", flush=True)
                    while child.poll() is None:
                        refresh()
                        time.sleep(10)
                    if child.returncode:
                        raise RuntimeError(f"Budget {budget} exited with code {child.returncode}; inspect budget_{budget}.log")
                    refresh()
                    if str(budget) not in results:
                        raise ValueError(f"No complete result for budget {budget}")
                child = None
                print(f"Finished {args.protocol} b={budget}: raw rollout audit passed", flush=True)
            if sha256(evaluator) != manifest["evaluator_sha256"] or sha256(dataset) != DATASET_SHA256:
                raise ValueError("Evaluation inputs changed during this run")
            if any(sha256(Path(path)) != digest for path, digest in baseline_hashes.items()):
                raise ValueError("A baseline artifact changed during this run")
            state.update(state="complete", active_pid=None)
            refresh()
            print(f"All {len(budgets)} budget points completed and audited: {root / 'results.md'}", flush=True)
        except BaseException as exc:
            state.update(state="failed", failed=[str(exc)])
            write_json(root / "status.json", state)
            raise
        finally:
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    main()
