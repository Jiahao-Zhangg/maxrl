"""Run isolated, single-step GRPO comparisons on an existing Slurm allocation.

The plan supplies all paths and versions. Each variant uses the same resolved
training configuration except for its Python environment, V0/V1 selection, and
version-specific engine arguments. This does not alter the normal launchers.
"""

import argparse
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def gpu_samples(path, stopped):
    with Path(path).open("a", buffering=1) as stream:
        while not stopped.is_set():
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, timeout=20)
            stream.write(json.dumps({"time": time.time(), "gpus": result.stdout.strip(),
                                     "returncode": result.returncode}) + "\n")
            stopped.wait(2)


def summarize(folder):
    text = (folder / "train.log").read_text(errors="replace")
    metrics = {}
    for line in text.splitlines():
        if "step:1 - " in line:
            for key, value in re.findall(r"(?:step:1 - | - )([^:]+):([-+\d.eE]+)", line):
                try:
                    metrics[key.strip()] = float(value)
                except ValueError:
                    pass
    graph_lines = [line for line in text.splitlines()
                   if any(word in line.lower() for word in
                          ("captur", "cuda graph", "cudagraph", "preempt", "v1 llm engine", "v0 llm engine"))]
    output = {"metrics": metrics, "graph_and_preemption_log": graph_lines[-100:],
              "preemption_log_lines": sum("preempt" in line.lower() for line in text.splitlines())}
    if "response_length/mean" in metrics and metrics.get("timing_s/gen", 0) > 0:
        output["output_tokens_per_second_including_reshard"] = (
            metrics["response_length/mean"] * 256 / metrics["timing_s/gen"])
    peaks = {}
    for line in (folder / "gpu.jsonl").read_text().splitlines():
        for row in json.loads(line)["gpus"].splitlines():
            fields = [v.strip() for v in row.split(",")]
            if len(fields) == 4:
                peaks[fields[0]] = max(peaks.get(fields[0], 0), float(fields[1]))
    output["peak_gpu_memory_mib"] = peaks
    write_json(folder / "summary.json", output)
    return output


def train_entry(plan, variant):
    import importlib.metadata

    # Slurm exports both vendor masks on this NVIDIA cluster. Ray needs only
    # CUDA_VISIBLE_DEVICES and sets that mask separately for each GPU worker.
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    folder = Path(plan["scratch"]) / variant["name"]
    write_json(folder / "process.json", {
        "pid": os.getpid(), "node": socket.gethostname(),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "python": sys.executable,
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "vllm", "transformers", "flash-attn", "ray", "tensordict")},
    })
    import runpy

    sys.argv = ["verl.trainer.main_ppo", "--config-path", str(folder),
                "--config-name", "resolved_config"]
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")


def run_variant(plan, variant):
    scratch = Path(plan["scratch"])
    folder = scratch / variant["name"]
    folder.mkdir(exist_ok=True)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("RAY_", "VLLM_", "SLURM_", "WANDB_")):
            env.pop(key)
    env.update({
        "PATH": str(Path(variant["python"]).parent) + os.pathsep + env.get("PATH", ""),
        "PYTHONPATH": plan["runtime"], "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_USE_V1": str(variant["v1"]),
        "RAY_DEDUP_LOGS": "0", "HYDRA_FULL_ERROR": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "VLLM_LOGGING_LEVEL": "INFO",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "NCCL_DEBUG": "WARN", "SEED": str(plan["seed"]),
        "HF_HOME": str(scratch / "hf"), "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled",
        "TMPDIR": str(folder / "tmp"), "TRITON_CACHE_DIR": str(folder / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(folder / "inductor"),
        "VLLM_CACHE_ROOT": str(folder / "vllm_cache"),
    })
    env.update(variant.get("environment", {}))
    for key in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "PYTHONHOME"):
        env.pop(key, None)
    for key in ("TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", "HF_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    command = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
               f"--nodelist={plan['node']}", "--cpus-per-task=192", "--gres=gpu:8",
               "--kill-on-bad-exit=1", f"--chdir={plan['artifact_root']}", variant["python"], "-u", "-m",
               variant.get("train_module", "qwen3_experiments.grpo_graph_benchmark"),
               "--plan", plan["plan_path"],
               "--train-variant", variant["name"]]
    started = time.time()
    write_json(folder / "status.json", {"state": "running", "started": started, "command": command})
    stopped = threading.Event()
    monitor = threading.Thread(target=gpu_samples, args=(folder / "gpu.jsonl", stopped), daemon=True)
    monitor.start()
    with (folder / "train.log").open("ab", buffering=0) as stream:
        child = subprocess.Popen(command, cwd=plan["runtime"], env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=plan.get("variant_timeout_seconds", 7200))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    stopped.set()
    monitor.join(timeout=25)
    output = summarize(folder)
    state = "complete" if code == 0 and "timing_s/step" in output["metrics"] else "failed"
    status = {"state": state, "exit_code": code, "started": started, "finished": time.time(),
              "wall_seconds": time.time() - started, "summary": str(folder / "summary.json")}
    write_json(folder / "status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--train-variant")
    parser.add_argument("--variant", help="Run just this variant; used while preparing another environment")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if socket.gethostname() != plan["node"]:
        raise RuntimeError("Benchmark must run on its allocated compute node")
    if args.train_variant:
        variant = next(v for v in plan["variants"] if v["name"] == args.train_variant)
        train_entry(plan, variant)
        return
    with (Path(plan["scratch"]) / "benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = {}
        for variant in plan["variants"]:
            if args.variant and args.variant != variant["name"]:
                continue
            busy = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
            if busy.strip():
                raise RuntimeError("GPUs already have compute processes; refusing to disturb them")
            result[variant["name"]] = run_variant(plan, variant)
            write_json(Path(plan["scratch"]) / "status.json", result)


if __name__ == "__main__":
    main()
