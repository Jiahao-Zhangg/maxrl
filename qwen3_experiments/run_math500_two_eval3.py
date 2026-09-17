#!/usr/bin/env python3
"""Run and audit either or both historical MATH-500 shared-budget protocols."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import random
import shutil
import signal
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from run_math500_offset256_comparison import DATASET_SHA256, PACKAGES, REPO_ROOT, result_paths, sha256, write_json

BUDGETS = (256, 512, 1024, 2048, 4096)
SELECTIONS = {"eval3_skip_solved": "sweep", "eval3_iid": "random-with-replacement"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_result(args: argparse.Namespace, protocol: str, budget: int) -> dict:
    paths = result_paths(args.output_root / protocol / "results", protocol, budget, args.model_label)
    if not all(path.is_file() and path.stat().st_size for path in paths):
        raise ValueError(f"Incomplete result set: {paths[-1]}")
    summary = json.loads(paths[-1].read_text())
    expected = {
        "model_label": args.model_label,
        "checkpoint_repo": args.checkpoint_repo,
        "checkpoint_revision": args.checkpoint_revision,
        "dataset": "HuggingFaceH4/MATH-500",
        "dataset_sha256": DATASET_SHA256,
        "num_prompts": 500,
        "budget_per_prompt_reference": budget,
        "total_global_output_budget": 500 * budget,
        "per_rollout_output_cap": 4096,
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
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"{paths[-1]}: {key}={summary.get(key)!r}, expected {value!r}")
    if summary.get("question_selection", "sweep") != SELECTIONS[protocol]:
        raise ValueError("Wrong question-selection strategy")
    if not (summary["global_budget_exhausted"] or summary["all_prompts_solved_early"]):
        raise ValueError("The evaluation stopped before consuming its budget")
    rollouts, prompts = read_jsonl(paths[0]), read_jsonl(paths[1])
    assert len(prompts) == 500 and {p["prompt_position"] for p in prompts} == set(range(500))
    remaining = budget * 500
    solved = set()
    attempts = Counter()
    lengths = Counter()
    permutations = {}
    last_rank = defaultdict(lambda: -1)
    rng = random.Random(0)
    for index, record in enumerate(rollouts):
        position = record["prompt_position"]
        assert 0 <= position < 500
        assert record["request_sequence_index"] == index
        assert record["rollout_index"] == attempts[position]
        assert record["rollout_seed"] == (1_000_003 * position + attempts[position]) % (2**31 - 1)
        assert record["score"] in (0, 1)
        assert 0 < record["output_tokens"] <= record["max_output_tokens"] <= 4096
        assert record["global_budget_before"] == remaining
        remaining -= record["output_tokens"]
        assert remaining >= 0 and record["global_budget_after"] == remaining
        assert record["solved_before"] == (position in solved)
        if protocol == "eval3_iid":
            assert record["question_draw_index"] == index
            assert position == rng.randrange(500)
        else:
            assert position not in solved
            current_round = record["round_index"]
            if current_round not in permutations:
                order = list(range(500))
                random.Random((2_000_033 * current_round) % (2**31 - 1)).shuffle(order)
                permutations[current_round] = order
            rank = record["permutation_rank"]
            assert rank > last_rank[current_round] and permutations[current_round][rank] == position
            last_rank[current_round] = rank
        attempts[position] += 1
        lengths[position] += record["output_tokens"]
        if record["score"] > 0:
            solved.add(position)
        assert record["solved_after"] == (position in solved)
    for prompt in prompts:
        position = prompt["prompt_position"]
        assert prompt["solved"] == (position in solved)
        assert prompt["attempts"] == attempts[position]
        assert prompt["total_output_tokens"] == lengths[position]
    assert len(rollouts) == summary["total_rollouts"]
    assert remaining == summary["global_output_budget_remaining"]
    assert sum(lengths.values()) == summary["global_output_budget_used"]
    assert math.isclose(len(solved) / 500, summary["fraction_solved"], abs_tol=1e-12)
    return {"budget": budget, "fraction_solved": len(solved) / 500, "num_prompts_solved": len(solved), "total_rollouts": len(rollouts), "global_output_budget_used": sum(lengths.values()), "summary_file": str(paths[-1]), "raw_rollout_audit": "passed"}


def gpu_is_idle(gpu: int) -> bool:
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    )
    memory, utilization = (int(value.strip()) for value in output.strip().split(","))
    return memory < 512 and utilization == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("python", "model-path", "artifact-root", "ipc-root", "output-root"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("model-label", "checkpoint-repo", "checkpoint-revision"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--protocol", choices=("both", *SELECTIONS), default="both")
    parser.add_argument("--source-manifest", type=Path, help="Optional verified prepared-model manifest")
    args = parser.parse_args()
    for key in ("python", "model_path", "artifact_root", "ipc_root", "output_root"):
        setattr(args, key, getattr(args, key).resolve())
    if len(os.fsencode(str(args.ipc_root))) + 48 >= 107:
        raise ValueError("Use a shorter --ipc-root for vLLM Unix-domain sockets")
    selections = SELECTIONS if args.protocol == "both" else {args.protocol: SELECTIONS[args.protocol]}
    control_root = args.output_root if args.protocol == "both" else args.output_root / args.protocol
    for directory in (control_root, args.artifact_root / "runtime", args.ipc_root):
        directory.mkdir(parents=True, exist_ok=True)
    runtime = args.artifact_root / "runtime"
    if args.protocol != "both":
        runtime = runtime / f"gpu{args.gpu}"
        runtime.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "XDG_CACHE_HOME": str(runtime),
        "TMPDIR": str(args.ipc_root),
        "TRITON_CACHE_DIR": str(runtime / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(runtime / "inductor"),
        "VLLM_CACHE_ROOT": str(runtime / "vllm"),
        "HF_HUB_CACHE": str(args.artifact_root / "hf-hub"),
        "HF_XET_CACHE": str(args.artifact_root / "hf-xet"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    dataset = REPO_ROOT / "data/math500/test.parquet"
    evaluator = REPO_ROOT / "qwen3_experiments/eval_math500_cross_context_budget.py"
    results = {protocol: {} for protocol in selections}
    state = {"state": "preparing", "gpu": args.gpu, "supervisor_pid": os.getpid(), "active_pid": None, "failed": []}

    def refresh() -> None:
        for protocol in selections:
            for budget in BUDGETS:
                summary = result_paths(args.output_root / protocol / "results", protocol, budget, args.model_label)[-1]
                if str(budget) not in results[protocol] and summary.is_file():
                    results[protocol][str(budget)] = audit_result(args, protocol, budget)
        state["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["completed_points"] = sum(map(len, results.values()))
        state["total_points"] = len(BUDGETS) * len(selections)
        write_json(control_root / "status.json", state)
        write_json(control_root / "results.json", results)
        lines = [
            "# MATH-500: shared-budget evaluations",
            "",
            f"Model: [{args.model_label}](https://huggingface.co/{args.checkpoint_repo}); revision `{args.checkpoint_revision}`.",
            "",
            "All 500 questions; total generated-token budget = 500 × b. Each response is capped at 4096 tokens "
            "or the remaining budget. Temperature 0.6, top-p 0.95, top-k -1, seed 0, and the historical MathVerify scorer "
            "with a one-second timeout. Unvisited questions remain unsolved in the denominator.",
            "",
        ]
        for protocol, title in (("eval3_skip_solved", "Eval 3(1): shuffled sweeps, skip solved questions"), ("eval3_iid", "Eval 3(2): IID draws, including already-solved questions")):
            if protocol not in selections:
                continue
            values = [f"{results[protocol][str(b)]['fraction_solved']:.2%}" if str(b) in results[protocol] else "pending" for b in BUDGETS]
            lines += [f"## {title}", "", "| Model | " + " | ".join(map(str, BUDGETS)) + " |", "|---|" + "---:|" * len(BUDGETS), f"| {args.model_label} | " + " | ".join(values) + " |", ""]
        lines += [f"Validated budget points: {state['completed_points']}/{state['total_points']}. Raw rollouts and per-question results are retained.", ""]
        temporary = control_root / "results.md.tmp"
        temporary.write_text("\n".join(lines))
        temporary.replace(control_root / "results.md")

    def interrupted(signum, frame) -> None:
        raise InterruptedError(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    with (control_root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sha256(dataset) != DATASET_SHA256:
            raise ValueError("MATH-500 differs from the historical evaluation")
        version_code = "import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in " + repr(list(PACKAGES)) + "}))"
        versions = json.loads(subprocess.check_output([str(args.python), "-c", version_code], env=env, text=True))
        if versions != PACKAGES:
            raise ValueError(f"Pinned evaluation packages differ: {versions}")
        if not (args.model_path / "config.json").is_file() or not list(args.model_path.glob("*.safetensors")):
            raise ValueError("The model has not been prepared")
        evaluator_hash = sha256(evaluator)
        manifest = {
            "checkpoint_repo": args.checkpoint_repo,
            "checkpoint_revision": args.checkpoint_revision,
            "gpu": args.gpu,
            "packages": versions,
            "dataset_sha256": DATASET_SHA256,
            "evaluator_sha256": evaluator_hash,
            "model_sha256": {p.name: sha256(p) for p in sorted(args.model_path.iterdir()) if p.is_file()},
            "commands": {},
        }
        if args.source_manifest is not None:
            source = json.loads(args.source_manifest.read_text())
            for key in ("checkpoint_repo", "checkpoint_revision", "model_sha256"):
                if source.get(key) != manifest[key]:
                    raise ValueError(f"Prepared checkpoint verification failed: {key}")
        manifest_path = control_root / "manifest.json"
        if manifest_path.is_file():
            previous_manifest = json.loads(manifest_path.read_text())
            for key, value in manifest.items():
                if key != "commands" and previous_manifest.get(key) != value:
                    raise ValueError(f"Cannot resume: manifest field {key} changed")
            manifest["commands"] = previous_manifest.get("commands", {})
        write_json(manifest_path, manifest)
        refresh()
        child = None
        try:
            for protocol, selection in selections.items():
                if len(results[protocol]) == len(BUDGETS):
                    continue
                state.update(state="waiting_for_gpu", protocol=protocol, active_pid=None)
                refresh()
                idle_checks = 0
                while idle_checks < 2:
                    disk_ok = shutil.disk_usage(args.artifact_root).free > 20 * 2**30 and shutil.disk_usage(args.output_root).free > 5 * 2**30
                    idle_checks = idle_checks + 1 if disk_ok and gpu_is_idle(args.gpu) else 0
                    if idle_checks < 2:
                        time.sleep(10)
                        refresh()
                # An evaluator from an earlier supervisor may finish while we wait.
                refresh()
                if len(results[protocol]) == len(BUDGETS):
                    continue
                if sha256(evaluator) != evaluator_hash:
                    raise ValueError("The evaluator changed during this run")
                directory = args.output_root / protocol / "results"
                directory.mkdir(parents=True, exist_ok=True)
                cmd = [
                    str(args.python),
                    str(evaluator),
                    "--model-path",
                    str(args.model_path),
                    "--model-label",
                    args.model_label,
                    "--checkpoint-repo",
                    args.checkpoint_repo,
                    "--checkpoint-revision",
                    args.checkpoint_revision,
                    "--dataset",
                    str(dataset),
                    "--output-dir",
                    str(directory),
                    "--budget-per-prompt",
                    *map(str, BUDGETS),
                    "--per-rollout-cap",
                    "4096",
                    "--max-prompt-len",
                    "1024",
                    "--max-batch-size",
                    "512",
                    "--question-selection",
                    selection,
                    "--grader-workers",
                    "8",
                    "--grader-timeout",
                    "1",
                    "--temperature",
                    "0.6",
                    "--top-p",
                    "0.95",
                    "--top-k=-1",
                    "--seed",
                    "0",
                    "--gpu-memory-utilization",
                    "0.7",
                ]
                manifest["commands"][protocol] = cmd
                write_json(manifest_path, manifest)
                with (control_root / f"{protocol}.log").open("a") as log:
                    child = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                    state.update(state="running", protocol=protocol, active_pid=child.pid)
                    print(f"Started {protocol} on GPU {args.gpu}: pid={child.pid}", flush=True)
                    while child.poll() is None:
                        refresh()
                        time.sleep(10)
                    if child.returncode:
                        raise RuntimeError(f"{protocol} exited with status {child.returncode}; inspect {protocol}.log")
                    refresh()
                    if len(results[protocol]) != len(BUDGETS):
                        raise ValueError(f"Missing completed budgets for {protocol}")
                    print(f"Finished {protocol}: five budget points audited.", flush=True)
                child = None
            if sha256(evaluator) != evaluator_hash or sha256(dataset) != DATASET_SHA256:
                raise ValueError("Evaluation inputs changed while running")
            state.update(state="complete", active_pid=None)
            refresh()
            print(f"All {state['total_points']} budget points completed and audited: {control_root / 'results.md'}", flush=True)
        except BaseException as exc:
            state.update(state="failed", failed=[str(exc)])
            write_json(control_root / "status.json", state)
            raise
        finally:
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    main()
