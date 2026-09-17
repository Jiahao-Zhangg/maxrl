#!/usr/bin/env python3
"""Fill the shared-budget grid using the historical Eval2 evaluator on one GPU."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from run_math500_eval1_eval2 import audit_point
from run_math500_offset256_comparison import DATASET_SHA256, PACKAGES, REPO_ROOT, idle_gpus, result_paths, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    models = {model["id"]: model for model in config["models"]}
    selected = [models[key] for key in args.models]
    root = Path(config["output_root"])
    results = root / "results"
    runtime = Path(config["artifact_root"]) / "runtime" / f"gpu{args.gpu}"
    ipc = Path(config["ipc_root"]) / f"g{args.gpu}"
    for path in (results, root / "logs", runtime, ipc):
        path.mkdir(parents=True, exist_ok=True)
    evaluator = REPO_ROOT / "qwen3_experiments/eval_math500_total_token_budget.py"
    dataset = REPO_ROOT / "data/math500/test.parquet"
    if sha256(dataset) != DATASET_SHA256:
        raise ValueError("MATH-500 differs from the historical dataset")
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO_ROOT),
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "TMPDIR": str(ipc),
        "XDG_CACHE_HOME": str(runtime), "TRITON_CACHE_DIR": str(runtime / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(runtime / "inductor"), "VLLM_CACHE_ROOT": str(runtime / "vllm"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    version_code = "import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in " + repr(list(PACKAGES)) + "}))"
    versions = json.loads(subprocess.check_output([config["python"], "-c", version_code], env=environment, text=True))
    if versions != PACKAGES:
        raise ValueError(f"Evaluation packages differ from the historical results: {versions}")
    state = {"state": "starting", "gpu": args.gpu, "supervisor_pid": os.getpid(),
             "completed": {}, "total_points": len(selected) * len(config["budgets"])}
    manifest = {"packages": versions, "dataset_sha256": DATASET_SHA256,
                "evaluator_sha256": sha256(evaluator), "gpu": args.gpu, "models": {}, "results": {}}

    def update(**fields: object) -> None:
        state.update(fields, updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(root / f"status_gpu{args.gpu}.json", state)
        write_json(root / f"manifest_gpu{args.gpu}.json", manifest)

    with (root / f"gpu{args.gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for model in selected:
                label, checkpoint = model["model_key"], model["checkpoint_repo"]
                model_path = Path(model["model_path"])
                prepared_path = Path(model["prepared_manifest"])
                update(state="waiting_for_model", model=label, active_pid=None)
                print(f"GPU {args.gpu}: waiting for verified {label}", flush=True)
                while not prepared_path.is_file():
                    time.sleep(10)
                    update()
                prepared = json.loads(prepared_path.read_text())
                if prepared["checkpoint_repo"] != checkpoint or prepared["checkpoint_revision"] != model["checkpoint_revision"]:
                    raise ValueError(f"Unexpected prepared checkpoint: {label}")
                hashes = {path.name: sha256(path) for path in sorted(model_path.iterdir()) if path.is_file()}
                if hashes != prepared["model_sha256"] or not list(model_path.glob("*.safetensors")):
                    raise ValueError(f"Prepared model failed integrity verification: {label}")
                manifest["models"][label] = {"checkpoint_repo": checkpoint, "checkpoint_revision": model["checkpoint_revision"],
                                             "model_path": str(model_path), "model_sha256": hashes}
                for budget in config["budgets"]:
                    key = f"{label}/{budget}"
                    targets = result_paths(results, "eval2", budget, label)
                    if all(path.is_file() for path in targets):
                        provenance = {"kind": "resumed"}
                    else:
                        if any(path.exists() for path in targets):
                            raise FileExistsError(f"Incomplete result requires inspection: {key}")
                        sources = result_paths(Path(model["source_result_dir"]), "eval2", budget, label)
                        if all(path.is_file() for path in sources):
                            audit_point(Path(model["source_result_dir"]), "eval2", budget, label, checkpoint)
                            for source, target in zip(sources, targets, strict=True):
                                shutil.copy2(source, target)
                            provenance = {"kind": "reused", "source_files": {str(path): sha256(path) for path in sources}}
                        else:
                            update(state="waiting_for_gpu", budget=budget)
                            while args.gpu not in idle_gpus():
                                time.sleep(10)
                                update()
                            if sha256(evaluator) != manifest["evaluator_sha256"]:
                                raise ValueError("Evaluator changed during this series")
                            command = [config["python"], str(evaluator), "--model-path", str(model_path),
                                       "--model-label", label, "--checkpoint-repo", checkpoint,
                                       "--dataset", str(dataset), "--output-dir", str(results),
                                       "--total-output-budget", str(budget), "--per-rollout-cap", "4096",
                                       "--max-prompt-len", "1024", "--grader-workers", "8",
                                       "--temperature", "0.6", "--top-p", "0.95", "--top-k=-1", "--seed", "0"]
                            log_path = root / "logs" / f"{label}_budget_{budget}.log"
                            provenance = {"kind": "new", "command": command, "log_file": str(log_path)}
                            print(f"GPU {args.gpu}: starting {key}", flush=True)
                            with log_path.open("a") as log:
                                child = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                                update(state="running", budget=budget, active_pid=child.pid)
                                while child.poll() is None:
                                    time.sleep(5)
                                    update()
                                if child.returncode:
                                    raise RuntimeError(f"Evaluation failed with exit {child.returncode}: {log_path}")
                    audit = audit_point(results, "eval2", budget, label, checkpoint)
                    manifest["results"][key] = {**provenance, **audit,
                                                "files": {str(path): sha256(path) for path in targets}}
                    state["completed"][key] = audit["accuracy"]
                    update(state="audited", active_pid=None)
                    print(f"GPU {args.gpu}: {key} audited, solved={round(500 * audit['accuracy'])}", flush=True)
            update(state="complete", active_pid=None)
        except BaseException as exc:
            update(state="failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
