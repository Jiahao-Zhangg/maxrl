"""Compute-only veRL 0.9.1 coding RL, verified uploads, recovery, and four holdouts."""

import argparse
import errno
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path

from qwen3_experiments import coding_release_eval as evaluation
from qwen3_experiments import verify_checkpoint_upload as verifier
from qwen3_experiments.grpo_compute_control import ensure_public_repository, require_compute
from qwen3_experiments.taco_eval import digest, now, read, write
from qwen3_experiments.taco_grpo_pipeline import finish_verified_deletion, verify_uploaded_folder
from qwen3_experiments.taco_resilience import cuda_oom, log_tail, process, terminate_failed_step
from qwen3_experiments.upload_training_rollouts_to_hf import complete_records

MODULE = "qwen3_experiments.grpo_coding_release"
SPACE_ERRORS = (errno.ENOSPC, errno.EDQUOT, errno.EIO)


def persist(plan, name, value):
    value = {**value, "updated_at": now()}
    successes = 0
    for root in (Path(plan["scratch"]) / "control_mirrors", Path(plan["output_root"])):
        try:
            write(root / name, value)
            successes += 1
        except OSError as exc:
            if exc.errno not in SPACE_ERRORS:
                raise
            try:
                (root / ".disk_reserve").unlink(missing_ok=True)
                write(root / name, value)
                successes += 1
            except OSError:
                pass
    return bool(successes)


def state(plan, name):
    copies = []
    for root in (Path(plan["scratch"]) / "control_mirrors", Path(plan["output_root"])):
        try:
            copies.append(read(root / name))
        except (OSError, ValueError):
            pass
    return max(copies, key=lambda item: item.get("updated_at", ""), default={})


def safe_error(plan, name, exc):
    try:
        persist(plan, name, {"state": "retrying", "error": str(exc), "pid": os.getpid()})
        print(traceback.format_exc(), flush=True)
    except OSError:
        pass


@contextmanager
def lock_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def gpu_guard(plan):
    with ExitStack() as stack:
        stack.enter_context(lock_file(Path(plan["scratch"]) / "guard/gpu.lock"))
        for path in plan["holder_locks"]:
            # Existing shared locks are opened read-only so disk pressure cannot
            # prevent coordination with the other queues.
            stream = stack.enter_context(Path(path).open("r"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        )
        if any(line.strip().isdigit() for line in result.splitlines()):
            raise RuntimeError("Another process is using the allocation GPUs")
        yield


def environment(plan, *, gpu=None, training=False):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("RAY_", "VLLM_", "SLURM_", "WANDB_", "MAXRL_", "GRPO_")):
            if key != "WANDB_API_KEY":
                env.pop(key)
    for key in ("PYTHONHOME", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        env.pop(key, None)
    scratch = Path(plan["scratch"])
    env.update(
        PATH=str(Path(plan["python_bin"]).parent) + ":/usr/local/cuda/bin:" + env.get("PATH", ""),
        PYTHONPATH=plan["runtime"], PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_USE_V2_MODEL_RUNNER="1",
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_HOME="/usr/local/cuda", CUDA_MODULE_LOADING="LAZY",
        NCCL_DEBUG="WARN", TORCH_NCCL_ASYNC_ERROR_HANDLING="1", RAY_DEDUP_LOGS="0", HYDRA_FULL_ERROR="1",
        TMPDIR=str(scratch / "tmp"), TRITON_CACHE_DIR=str(scratch / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(scratch / "inductor"), VLLM_CACHE_ROOT=str(scratch / "vllm_cache"),
        FLASHINFER_WORKSPACE_BASE=str(scratch / "flashinfer"), HF_HUB_CACHE=str(scratch / "hf/hub"),
        HF_XET_CACHE=str(scratch / "hf/xet"), HF_DATASETS_CACHE=str(scratch / "hf/datasets"),
        HF_TOKEN_PATH=plan["hf_token_path"],
        GRPO_BENCHMARK_METRICS_DIR=str(scratch / "engine_metrics"),
    )
    if training:
        env.update(WANDB_MODE="online", WANDB_DIR=str(scratch / "wandb"),
                   WANDB_RUN_ID=plan["wandb_id"], WANDB_RESUME="allow", WANDB_INIT_TIMEOUT="120")
    else:
        env["WANDB_MODE"] = "disabled"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def assert_versions():
    versions = {name: importlib.metadata.version(name) for name in ("verl", "vllm", "torch", "transformers", "ray")}
    if versions["verl"] != "0.9.1" or versions["vllm"] != "0.24.0":
        raise ValueError(f"Expected maxrl-code-verl091: {versions}")
    return versions


def prepare(args):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    from huggingface_hub import constants as hf_constants
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from qwen3_experiments.lcb_coding_format import validate_truth
    from qwen3_experiments.prepare_lcb_grading import prepare as prepare_grading
    from verl.utils.dataset.rl_dataset import RLHFDataset

    node = require_compute(args.job_id)
    versions = assert_versions()
    algorithm = args.algorithm
    rollout_n = args.n if args.n is not None else (8 if algorithm == "grpo" else 16)
    if rollout_n < 2:
        raise ValueError("Grouped coding RL requires N >= 2")
    root, scratch = args.root.absolute(), args.scratch.absolute()
    if (root / "plan.json").exists():
        raise ValueError("Run is already prepared; use its frozen plan")
    root.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    benchmark = read(args.benchmark_plan)
    source = Path(__file__).resolve().parents[1]
    runtime = scratch / "runtime"
    runtime.mkdir(exist_ok=True)
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=source
    ).decode().split("\0")
    runtime_hashes = {}
    for relative in sorted(set(files) - {""}):
        path = Path(relative)
        if path.parts[0] not in ("verl", "qwen3_experiments", "scripts", "tests") and relative not in (
            "pyproject.toml", "setup.py", "uv.lock", "AGENTS.md",
        ):
            continue
        target = runtime / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / path, target)
        runtime_hashes[relative] = digest(target)
    for name in ("data", "checkpoints", "rollouts", "logs", "tmp", "triton", "inductor", "vllm_cache",
                 "flashinfer", "hf", "wandb", "guard", "control_mirrors", "engine_metrics"):
        (scratch / name).mkdir(exist_ok=True)
    for directory in (root, scratch / "control_mirrors"):
        with (directory / ".disk_reserve").open("wb") as stream:
            stream.write(b"\0" * (8 << 20))
    data = scratch / "data/train.parquet"
    shutil.copy2(benchmark["dataset"], data)
    if digest(data) != benchmark["dataset_sha256"]:
        raise ValueError("Training data differs from the verified release")
    rows = pq.read_table(data).to_pylist()
    if len(rows) != 3200:
        raise ValueError("Expected all 3,200 training questions")
    for row in rows:
        validate_truth(json.loads(row["reward_model"]["ground_truth"]))
    validation = scratch / "data/unused_validation.parquet"
    shutil.copy2(benchmark["validation_dataset"], validation)
    api = HfApi()
    account = api.whoami()["name"]
    old_grading = read(benchmark["grading_plan"])
    grading_plan = prepare_grading(scratch / "grading", old_grading["bubblewrap"], sys.executable)
    model = scratch / "base_model"
    shutil.copytree(benchmark["model"], model, dirs_exist_ok=True)
    base_hashes = {p.name: digest(p) for p in model.iterdir() if p.is_file()}
    baseline = read(args.baseline_root / "plan.json")
    plan = {
        "job_id": str(args.job_id), "node": node, "output_root": str(root), "scratch": str(scratch),
        "runtime": str(runtime), "python_bin": sys.executable, "versions": versions, "created_at": now(),
        "dataset": str(data), "dataset_sha256": digest(data), "dataset_rows": len(rows),
        "dataset_repo": "hi-todayis-jh/Nemotron-LeetCode-coding-clean-3.2k",
        "dataset_revision": "aeef423c0faef01e2fdac84628254eaa5327683f",
        "model": str(model), "base_model_hashes": base_hashes, "grading_plan": str(grading_plan),
        "checkpoint_dir": str(scratch / "checkpoints"), "checkpoint_steps": list(range(10, 101, 10)),
        "hf_repo_prefix": args.hf_prefix, "hf_account": account, "public": True,
        "hf_token_path": hf_constants.HF_TOKEN_PATH,
        "holder_locks": baseline["holder_locks"], "total_steps": 100, "expected_rollouts_per_step": 32 * rollout_n,
        "wandb_id": hashlib.sha256(str(root).encode()).hexdigest()[:8], "experiment_name": args.experiment_name,
        "sampling": {"n": 1, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32768,
                     "ignore_eos": False, "skip_special_tokens": True},
        "runtime_files": runtime_hashes, "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "training": {"algorithm": algorithm, "batch_size": 32, "n": rollout_n,
                     "max_num_seqs": 16, "temperature": 1.0,
                     "top_p": 1.0, "top_k": -1, "shuffle": True, "seed": 42, "epochs": 1},
        "grading": {"name": "livecodebench", "check_eos": False, "score_after_thinking": True,
                    "unit_test_timeout_seconds": 10, "binary": True, "workers": 128},
    }
    if args.after_root is not None:
        predecessor_path = args.after_root.absolute() / "plan.json"
        predecessor = read(predecessor_path)
        if predecessor["job_id"] != plan["job_id"] or predecessor["node"] != node:
            raise ValueError("Predecessor must belong to the same compute allocation")
        if predecessor["output_root"] == plan["output_root"]:
            raise ValueError("A run cannot depend on itself")
        plan["predecessor"] = {"root": predecessor["output_root"], "plan_sha256": digest(predecessor_path)}
    plan["holdouts"] = evaluation.prepare_holdouts(plan, args.baseline_root, args.competition_root)
    max_prompt = max(v["max_prompt_tokens"] for v in plan["holdouts"].values())
    plan["evaluation_engine"] = {
        "dtype": "bfloat16", "tensor_parallel_size": 1, "max_model_len": max(35840, max_prompt + 32768),
        "max_num_batched_tokens": 8192, "max_num_seqs": 16, "gpu_memory_utilization": 0.7,
        "enforce_eager": False, "enable_chunked_prefill": True, "enable_prefix_caching": True,
        "async_scheduling": True, "seed": 42, "disable_log_stats": False,
        "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 3},
        "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8, 16]},
    }
    with initialize_config_dir(str(runtime / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer")
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.load(runtime / "qwen3_experiments/grpo_verl091_benchmark.yaml"),
                            OmegaConf.load(runtime / "qwen3_experiments/grpo_verl091_training.yaml"))
    overrides = {
        "algorithm.adv_estimator": algorithm,
        "algorithm.norm_adv_by_std_in_grpo": algorithm == "grpo",
        "algorithm.cost_offset_tokens": args.cost_offset_tokens,
        "algorithm.f_cov_num_prompts": 32 if algorithm == "f_cov" else None,
        "actor_rollout_ref.rollout.n": rollout_n,
        "data.train_files": str(data), "data.val_files": str(validation),
        "actor_rollout_ref.model.path": str(model),
        "reward.custom_reward_function.path": str(runtime / "qwen3_experiments/lcb_verl_reward.py"),
        "reward.custom_reward_function.reward_kwargs.grading_plan": str(grading_plan),
        "trainer.default_local_dir": plan["checkpoint_dir"], "trainer.rollout_data_dir": str(scratch / "rollouts"),
        "trainer.experiment_name": args.experiment_name,
        "ray_kwargs.ray_init._temp_dir": f"/tmp/cr_{plan['wandb_id']}",
        "actor_rollout_ref.rollout.agent.custom_async_server.path":
            str(runtime / "qwen3_experiments/grpo_release_server.py"),
        "actor_rollout_ref.rollout.agent.custom_async_server.name": "BenchmarkvLLMHttpServer",
        "ray_kwargs.ray_init.runtime_env.env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": "1", "CUDA_HOME": "/usr/local/cuda",
            "GRPO_BENCHMARK_METRICS_DIR": str(scratch / "engine_metrics"),
            "FLASHINFER_WORKSPACE_BASE": str(scratch / "flashinfer"), "CUDA_MODULE_LOADING": "LAZY",
        },
    }
    for key, value in overrides.items():
        OmegaConf.update(config, key, value, force_add=True)
    from verl.trainer.ppo.maxrl_algos import validate_maxrl_training_config

    validate_maxrl_training_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = RLHFDataset([str(data)], tokenizer, config.data)
    if len(dataset) != 3200:
        raise ValueError("Prompt filtering removed training questions")
    config_path = scratch / "resolved_config.yaml"
    OmegaConf.save(config, config_path, resolve=True)
    plan["config_path"], plan["config_sha256"] = str(config_path), digest(config_path)
    plan["grading_plan_sha256"] = digest(grading_plan)
    write(scratch / "plan.json", plan)
    shutil.copy2(scratch / "plan.json", root / "plan.json")
    shutil.copy2(config_path, root / "resolved_config.yaml")
    write(root / "runtime_manifest.json", runtime_hashes)
    write(root / "prepare.json", {"state": "prepared", "node": node, "versions": versions,
                                 "dataset_rows": len(dataset), "holdouts": plan["holdouts"]})
    print(json.dumps({"prepared": str(root), "scratch": str(scratch), "rows": len(dataset)}, indent=2))


def required_checkpoint_files(world_size=8):
    files = {"data.pt", "actor/huggingface/config.json", "actor/huggingface/tokenizer_config.json",
             "actor/fsdp_config.json"}
    files.update(f"actor/{kind}_world_size_{world_size}_rank_{rank}.pt"
                 for kind in ("model", "optim", "extra_state") for rank in range(world_size))
    return files


def checkpoint_complete(plan, step):
    directory = Path(plan["checkpoint_dir"])
    checkpoint = directory / f"global_step_{step}"
    try:
        latest = int((directory / "latest_checkpointed_iteration.txt").read_text())
    except (OSError, ValueError):
        return False
    if latest < step or checkpoint.is_symlink():
        return False
    return all((checkpoint / name).is_file() and not (checkpoint / name).is_symlink()
               and (checkpoint / name).stat().st_size > 0 for name in required_checkpoint_files())


def archive_receipt(plan, step):
    receipt = state(plan, f"checkpoints/global_step_{step}.json")
    if (receipt.get("state") not in ("verified", "archived_and_deleted")
            or receipt.get("repo_id") != f"{plan['hf_repo_prefix']}-step_{step}"
            or not required_checkpoint_files().issubset(receipt.get("files", {}))
            or not receipt.get("remote_commit")):
        return None
    return receipt


def monitor_checkpoints(plan):
    from huggingface_hub import HfApi

    with lock_file(Path(plan["scratch"]) / "guard/checkpoints.lock"):
        api = HfApi()
        while True:
            try:
                for step in plan["checkpoint_steps"]:
                    checkpoint = Path(plan["checkpoint_dir"]) / f"global_step_{step}"
                    name = f"checkpoints/global_step_{step}.json"
                    receipt = archive_receipt(plan, step)
                    if receipt:
                        if receipt["state"] == "verified":
                            finish_verified_deletion(checkpoint, receipt)
                            persist(plan, name, {**receipt, "state": "archived_and_deleted", "deleted_at": now()})
                        continue
                    if not checkpoint_complete(plan, step):
                        continue
                    repo = f"{plan['hf_repo_prefix']}-step_{step}"
                    persist(plan, "checkpoint_status.json", {"state": "uploading", "step": step})
                    ensure_public_repository(api, repo)
                    api.upload_folder(repo_id=repo, folder_path=str(checkpoint), path_in_repo=checkpoint.name,
                                      ignore_patterns=[".cache/**"], commit_message=f"Archive {checkpoint.name}")

                    def durable_receipt(path, value, name=name):
                        if not persist(plan, name, value):
                            raise OSError(errno.ENOSPC, "No durable checkpoint receipt destination")

                    verifier.write_receipt = durable_receipt
                    receipt = verifier.verify_checkpoint(checkpoint, repo, Path(plan["output_root"]) / name, api)
                    finish_verified_deletion(checkpoint, receipt)
                    persist(plan, name, {**receipt, "state": "archived_and_deleted", "deleted_at": now()})
                archived = [s for s in plan["checkpoint_steps"] if archive_receipt(plan, s)]
                persist(plan, "checkpoint_status.json", {"state": "complete" if len(archived) == 10 else "monitoring",
                                                        "archived_steps": archived, "pid": os.getpid()})
                if len(archived) == 10:
                    return
                time.sleep(2)
            except Exception as exc:
                safe_error(plan, "checkpoint_status.json", exc)
                time.sleep(20)


def verify_remote_file(api, path, repo, relative, *, repo_type="dataset"):
    info = api.repo_info(repo_id=repo, repo_type=repo_type, files_metadata=True)
    item = next((f for f in info.siblings if f.rfilename == relative), None)
    if item is None or item.size != path.stat().st_size:
        raise ValueError("Uploaded file is missing or has the wrong size")
    checksum = digest(path)
    if item.lfs:
        expected = item.lfs["sha256"] if isinstance(item.lfs, dict) else item.lfs.sha256
        if expected != checksum:
            raise ValueError("Uploaded file SHA256 mismatch")
    elif hashlib.sha1(f"blob {item.size}\0".encode() + path.read_bytes()).hexdigest() != item.blob_id:
        raise ValueError("Uploaded Git blob hash mismatch")
    return {"repo_id": repo, "remote_commit": info.sha, "remote_path": relative,
            "sha256": checksum, "bytes": item.size}


def monitor_rollouts(plan):
    from huggingface_hub import HfApi

    with lock_file(Path(plan["scratch"]) / "guard/rollouts.lock"):
        api, repo = HfApi(), plan["hf_repo_prefix"] + "-rollouts"
        initialized = False
        while True:
            try:
                if not initialized:
                    ensure_public_repository(api, repo, "dataset")
                    api.upload_file(
                        repo_id=repo, repo_type="dataset", path_in_repo="README.md",
                        path_or_fileobj=("---\nlicense: apache-2.0\n---\n# Coding GRPO rollouts\n\n"
                                         f"{plan['experiment_name']}\n\n"
                                         "One verified gzip JSONL shard per training step; 256 responses per shard.\n"
                                         "LCB binary grading after thinking, without an EOS gate.\n").encode(),
                    )
                    initialized = True
                for path in sorted((Path(plan["scratch"]) / "rollouts").glob("*.jsonl")):
                    if not path.stem.isdigit() or not 1 <= int(path.stem) <= plan["total_steps"]:
                        continue
                    step = int(path.stem)
                    complete = complete_records(path, step, plan["expected_rollouts_per_step"])
                    if complete is None:
                        continue
                    records, signature = complete
                    compressed = path.with_suffix(".jsonl.gz")
                    with path.open("rb") as source, gzip.open(compressed, "wb", compresslevel=6) as target:
                        shutil.copyfileobj(source, target)
                    relative = f"data/step_{step:06d}.jsonl.gz"
                    api.upload_file(repo_id=repo, repo_type="dataset", path_in_repo=relative,
                                    path_or_fileobj=str(compressed), commit_message=f"Rollouts step {step}")
                    receipt = verify_remote_file(api, compressed, repo, relative)
                    receipt.update(step=step, rows=len(records), source_sha256=digest(path),
                                   original_path=str(path), state="verified")
                    if not persist(plan, f"rollout_receipts/{step}.json", receipt):
                        raise OSError(errno.ENOSPC, "Cannot retain rollout receipt")
                    current = path.stat()
                    if [current.st_ino, current.st_size, current.st_mtime_ns] != signature:
                        raise ValueError("Rollout changed while uploading")
                    path.unlink()
                    compressed.unlink()
                    persist(plan, f"rollout_receipts/{step}.json", {**receipt, "state": "archived_and_deleted"})
                archived = [step for step in range(1, 101)
                            if state(plan, f"rollout_receipts/{step}.json").get("state") == "archived_and_deleted"]
                persist(plan, "rollout_status.json", {"state": "complete" if len(archived) == 100 else "monitoring",
                                                     "archived_steps": archived, "pid": os.getpid()})
                if len(archived) == 100 and state(plan, "training_exit.json").get("exit_code") == 0:
                    return
                time.sleep(15)
            except Exception as exc:
                safe_error(plan, "rollout_status.json", exc)
                time.sleep(20)


def restore_checkpoint(plan, step, *, models_only=False):
    from huggingface_hub import snapshot_download

    receipt = archive_receipt(plan, step)
    if receipt is None:
        raise ValueError("No verified checkpoint to restore")
    root = Path(plan["scratch"]) / ("merge_source" if models_only else "resume_source")
    names = [name for name in receipt["files"] if not models_only or
             (name.startswith("actor/") and not any(t in name for t in ("optim_world", "extra_state_world")))]
    snapshot_download(repo_id=receipt["repo_id"], revision=receipt["remote_commit"], local_dir=root,
                      allow_patterns=[f"global_step_{step}/{name}" for name in names], max_workers=4)
    checkpoint = root / f"global_step_{step}"
    for name in names:
        if digest(checkpoint / name) != receipt["files"][name]["sha256"]:
            raise ValueError("Restored checkpoint hash mismatch")
    return checkpoint


def train_entry(plan, attempt):
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    assert_versions()
    sys.argv = ["verl.trainer.main_ppo", "--config-path", str(Path(plan["scratch"]) / "attempts"),
                "--config-name", f"attempt_{attempt}"]
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")


def train(plan):
    from omegaconf import OmegaConf

    if not predecessor_complete(plan):
        raise RuntimeError("Predecessor training and all four evaluations must finish first")
    with gpu_guard(plan):
        cleanup_predecessor_checkpoints(plan)
        if shutil.disk_usage(plan["scratch"]).free < 100 << 30:
            raise RuntimeError("Need at least 100 GiB of local disk before training")
        previous = state(plan, "training_exit.json")
        if previous.get("exit_code") == 0:
            return
        for step in plan["checkpoint_steps"]:
            if checkpoint_complete(plan, step) and not archive_receipt(plan, step):
                raise RuntimeError("Complete checkpoint is still being uploaded; retry after verification")
        archived = [s for s in plan["checkpoint_steps"] if archive_receipt(plan, s)]
        if archived and max(archived) == plan["total_steps"]:
            persist(plan, "training_exit.json", {"exit_code": 0, "recovered_final_checkpoint": True})
            return
        attempt = previous.get("attempt", 0) + 1
        config = OmegaConf.load(plan["config_path"])
        resume_step = max(archived, default=0)
        if resume_step:
            config.trainer.resume_mode = "resume_path"
            config.trainer.resume_from_path = str(restore_checkpoint(plan, resume_step))
        attempts = Path(plan["scratch"]) / "attempts"
        attempts.mkdir(exist_ok=True)
        OmegaConf.save(config, attempts / f"attempt_{attempt}.yaml", resolve=True)
        record = {"attempt": attempt, "pid": os.getpid(), "started_at": now(), "resume_step": resume_step,
                  "slurm_step_id": os.environ.get("SLURM_STEP_ID"), "cgroup": Path("/proc/self/cgroup").read_text()}
        persist(plan, "training_active.json", record)
        path = Path(plan["scratch"]) / "logs" / f"train_attempt_{attempt}.log"
        command = [plan["python_bin"], "-u", "-m", MODULE, "train-entry", "--root", plan["output_root"],
                   "--attempt", str(attempt)]
        with path.open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=plan["runtime"], env=environment(plan, training=True),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            code = child.wait()
        record.update(exit_code=code, finished_at=now(), log=str(path), cuda_oom=cuda_oom(log_tail(path)))
        persist(plan, f"attempts/attempt_{attempt}.json", record)
        persist(plan, "training_exit.json", record)
        if code:
            raise RuntimeError(f"Training attempt {attempt} failed with exit {code}")


def final_model(plan):
    from huggingface_hub import HfApi

    old = state(plan, "final_model.json")
    if old:
        for name, checksum in old["files_sha256"].items():
            if digest(Path(old["path"]) / name) != checksum:
                raise ValueError("Final model changed")
        return old
    checkpoint = restore_checkpoint(plan, 100, models_only=True)
    destination = Path(plan["scratch"]) / "final_model"
    subprocess.run([plan["python_bin"], "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                    "--local_dir", str(checkpoint / "actor"), "--target_dir", str(destination)],
                   cwd=plan["runtime"], env=environment(plan), check=True)
    api, repo = HfApi(), plan["hf_repo_prefix"] + "-final"
    ensure_public_repository(api, repo)
    api.upload_folder(repo_id=repo, folder_path=str(destination), ignore_patterns=[".cache/**"])
    record = verify_uploaded_folder(api, destination, repo)
    if not persist(plan, "final_model.json", record):
        raise OSError(errno.ENOSPC, "Cannot preserve final model receipt")
    shutil.rmtree(checkpoint.parent)
    return record


def launch(plan, action, *, slurm=False, dataset=None, rank=None):
    command = [plan["python_bin"], "-u", "-m", MODULE, action, "--root", plan["output_root"]]
    if dataset is not None:
        command += ["--dataset", dataset]
    if rank is not None:
        command += ["--rank", str(rank)]
    if slurm:
        command = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                   f"--nodelist={plan['node']}", "--cpus-per-task=192", "--gres=gpu:8", "--kill-on-bad-exit=1",
                   f"--job-name=coding-{plan['training'].get('algorithm', 'grpo')}-verl091", *command]
    path = Path(plan["scratch"]) / "logs" / f"{action}_{dataset or 'control'}_{rank}.log"
    with path.open("ab", buffering=0) as log:
        child = subprocess.Popen(command, cwd=plan["runtime"], env=environment(plan, gpu=rank),
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    ident = process(child.pid)
    if ident:
        persist(plan, f"processes/{action}_{dataset or 'control'}_{rank}.json",
                {"pid": child.pid, "start_ticks": ident["start_ticks"], "command": command, "log": str(path)})
    return child


def alive(plan, action, dataset=None, rank=None):
    record = state(plan, f"processes/{action}_{dataset or 'control'}_{rank}.json")
    current = process(record.get("pid", -1))
    return bool(current and current["uid"] == os.getuid() and current["start_ticks"] == record.get("start_ticks")
                and MODULE in current["args"] and plan["output_root"] in current["args"])


def evaluate(plan, dataset):
    with gpu_guard(plan):
        final_model(plan)
        children = [launch(plan, "eval-worker", dataset=dataset, rank=rank) for rank in range(8)]
        try:
            while any(child.poll() is None for child in children):
                failed = [child.returncode for child in children if child.returncode not in (None, 0)]
                if failed:
                    raise RuntimeError(f"Evaluation workers failed: {failed}")
                time.sleep(5)
            if any(child.returncode for child in children):
                raise RuntimeError("Evaluation failed")
        finally:
            for child in children:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for child in children:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    pass
                # The driver may have exited before its vLLM subprocesses.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        metrics = evaluation.summarize(plan, dataset)
        persist(plan, f"evaluation/{dataset}/metrics.json", metrics)
        persist(plan, f"evaluation/{dataset}/audit.json", {
            "complete": True, "questions": metrics["questions"], "plan_sha256": plan["plan_sha256"],
            "metrics_sha256": digest(Path(plan["scratch"]) / "evaluation" / dataset / "metrics.json"),
        })


def wait_child(plan, action, dataset=None):
    if alive(plan, action, dataset):
        while alive(plan, action, dataset):
            time.sleep(10)
        return
    child = launch(plan, action, slurm=True, dataset=dataset)
    code = child.wait()
    if code:
        if action == "train":
            terminate_failed_step(plan, state(plan, "training_exit.json"))
        raise RuntimeError(f"{action} {dataset or ''} exited {code}")


def queue(plan):
    with lock_file(Path(plan["scratch"]) / "guard/queue.lock"):
        while True:
            try:
                if state(plan, "training_exit.json").get("exit_code") != 0:
                    persist(plan, "queue_status.json", {"state": "training", "pid": os.getpid()})
                    wait_child(plan, "train")
                    continue
                if archive_receipt(plan, 100) is None:
                    persist(plan, "queue_status.json", {"state": "waiting_for_final_upload", "pid": os.getpid()})
                    time.sleep(5)
                    continue
                for dataset in evaluation.COUNTS:
                    audit = state(plan, f"evaluation/{dataset}/audit.json")
                    if audit.get("complete") and audit.get("plan_sha256") == plan["plan_sha256"]:
                        continue
                    persist(plan, "queue_status.json", {"state": "evaluating", "dataset": dataset, "pid": os.getpid()})
                    wait_child(plan, "evaluate", dataset)
                    if not state(plan, f"evaluation/{dataset}/audit.json").get("complete"):
                        raise RuntimeError("Evaluation exited without its completion audit")
                if state(plan, "rollout_status.json").get("state") != "complete":
                    persist(plan, "queue_status.json", {"state": "waiting_for_rollout_uploads"})
                    time.sleep(15)
                    continue
                results = {name: state(plan, f"evaluation/{name}/metrics.json") for name in evaluation.COUNTS}
                persist(plan, "results.json", results)
                persist(plan, "queue_status.json", {"state": "complete", "pid": os.getpid()})
                return
            except Exception as exc:
                safe_error(plan, "queue_status.json", exc)
                time.sleep(30)


def predecessor_complete(plan):
    """Gate successors on the pinned predecessor's entire queue, not training exit."""
    dependency = plan.get("predecessor")
    if dependency is None:
        return True
    path = Path(dependency["root"]) / "plan.json"
    if digest(path) != dependency["plan_sha256"]:
        raise ValueError("Predecessor plan identity changed")
    predecessor = read(path)
    if predecessor["job_id"] != plan["job_id"] or predecessor["node"] != plan["node"]:
        raise ValueError("Predecessor compute allocation changed")
    if state(predecessor, "queue_status.json").get("state") != "complete":
        return False
    if state(predecessor, "training_exit.json").get("exit_code") != 0:
        return False
    if state(predecessor, "rollout_status.json").get("state") != "complete":
        return False
    for dataset, count in evaluation.COUNTS.items():
        audit = state(predecessor, f"evaluation/{dataset}/audit.json")
        if not (audit.get("complete") and audit.get("plan_sha256") == dependency["plan_sha256"]
                and audit.get("questions") == count):
            return False
    return True


def delete_verified_model_tree(directory, hashes):
    """Delete only remaining files matching a durable, verified Hub receipt."""
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Refusing redirected model cleanup")
    if not directory.exists():
        return
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Refusing model-cache symlink")
        relative = path.relative_to(directory)
        if not path.is_file() or ".cache" in relative.parts:
            continue
        if digest(path) != hashes.get(relative.as_posix()):
            raise ValueError(f"Model cache changed; retaining {path}")
    shutil.rmtree(directory)


def cleanup_predecessor_checkpoints(plan):
    """Called under the allocation GPU lock after every predecessor eval finished."""
    if not plan.get("predecessor") or state(plan, "predecessor_cleanup.json").get("state") == "complete":
        return
    if not predecessor_complete(plan):
        raise RuntimeError("Predecessor evaluations are incomplete; cannot clean its models")
    predecessor = read(Path(plan["predecessor"]["root"]) / "plan.json")
    if alive(predecessor, "train") or any(alive(predecessor, "evaluate", name) for name in evaluation.COUNTS):
        raise RuntimeError("Predecessor GPU stage has not exited yet")
    scratch = Path(predecessor["scratch"])
    final = state(predecessor, "final_model.json")
    if (final.get("path") != str(scratch / "final_model")
            or final.get("repo") != predecessor["hf_repo_prefix"] + "-final"
            or not final.get("revision") or not final.get("files_sha256")):
        raise ValueError("No verified public final-model receipt for predecessor cleanup")
    targets = [(scratch / "final_model", final["files_sha256"])]
    receipts = {"final_model": final}
    for name in ("checkpoints", "resume_source", "merge_source"):
        parent = scratch / name
        if parent.is_symlink():
            raise ValueError("Refusing redirected checkpoint-cache cleanup")
        for path in sorted(parent.glob("global_step_*")):
            step = int(path.name.removeprefix("global_step_"))
            receipt = archive_receipt(predecessor, step)
            if receipt is None:
                raise ValueError(f"Checkpoint has no verified upload: {path}")
            targets.append((path, {key: value["sha256"] for key, value in receipt["files"].items()}))
            receipts[path.relative_to(scratch).as_posix()] = receipt
    record = {"predecessor": plan["predecessor"], "paths": [str(path) for path, _ in targets],
              "verified_uploads": receipts}
    if not persist(plan, "predecessor_cleanup.json", {**record, "state": "verified"}):
        raise OSError(errno.ENOSPC, "Cannot preserve predecessor cleanup receipts")
    for path, hashes in targets:
        delete_verified_model_tree(path, hashes)
    if not persist(plan, "predecessor_cleanup.json", {**record, "state": "complete"}):
        raise OSError(errno.ENOSPC, "Cannot preserve predecessor cleanup completion")


def supervise(plan):
    with lock_file(Path(plan["scratch"]) / "guard/supervisor.lock"):
        children = {}
        while True:
            try:
                if not predecessor_complete(plan):
                    waiting = {"state": "waiting_for_predecessor", "pid": os.getpid(),
                               "predecessor": plan["predecessor"], "node": socket.gethostname()}
                    persist(plan, "supervisor_status.json", waiting)
                    persist(plan, "queue_status.json", waiting)
                    time.sleep(15)
                    continue
                if state(plan, "queue_status.json").get("state") == "complete":
                    persist(plan, "supervisor_status.json", {"state": "complete", "pid": os.getpid()})
                    return
                services = [("monitor-checkpoints", "checkpoint_status.json"),
                            ("monitor-rollouts", "rollout_status.json"), ("queue", "queue_status.json")]
                for action, status_name in services:
                    child = children.get(action)
                    if child is not None and child.poll() is None:
                        continue
                    if state(plan, status_name).get("state") == "complete" or alive(plan, action):
                        continue
                    children[action] = launch(plan, action)
                persist(plan, "supervisor_status.json", {
                    "state": "supervising", "pid": os.getpid(), "node": socket.gethostname(),
                    "local_free_bytes": shutil.disk_usage(plan["scratch"]).free,
                    "home_free_bytes": shutil.disk_usage(plan["output_root"]).free,
                })
                time.sleep(15)
            except Exception as exc:
                safe_error(plan, "supervisor_status.json", exc)
                time.sleep(15)


def load_plan(root):
    plan = read(Path(root) / "plan.json")
    require_compute(plan["job_id"])
    if plan["output_root"] != str(Path(root).absolute()):
        raise ValueError("Wrong run root")
    plan["plan_sha256"] = digest(Path(root) / "plan.json")
    if digest(plan["config_path"]) != plan["config_sha256"]:
        raise ValueError("Training configuration changed")
    if digest(plan["grading_plan"]) != plan["grading_plan_sha256"]:
        raise ValueError("Grading configuration changed")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "supervise", "queue", "monitor-checkpoints", "monitor-rollouts",
                                          "train", "train-entry", "evaluate", "eval-worker"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--benchmark-plan", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--competition-root", type=Path)
    parser.add_argument("--hf-prefix")
    parser.add_argument("--experiment-name")
    parser.add_argument("--algorithm", default="grpo", choices=[
        "grpo", "maxrl", "fixed_n_rb_offset_cost_aware_marginrl", "f_cov",
    ])
    parser.add_argument("--n", type=int, help="Responses per prompt; defaults to 8 for GRPO, 16 for MaxRL")
    parser.add_argument("--cost-offset-tokens", type=float, default=256.0)
    parser.add_argument("--after-root", type=Path, help="Wait for this frozen run and all four evaluations")
    parser.add_argument("--dataset", choices=list(evaluation.COUNTS))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--attempt", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
        return
    plan = load_plan(args.root)
    if args.action == "eval-worker":
        assert_versions()
        evaluation.worker(plan, args.dataset, args.rank)
    elif args.action == "train-entry":
        train_entry(plan, args.attempt)
    elif args.action == "evaluate":
        evaluate(plan, args.dataset)
    else:
        globals()[args.action.replace("-", "_")](plan)


if __name__ == "__main__":
    main()
