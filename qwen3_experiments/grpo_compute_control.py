"""Compute-node-only GRPO supervision and verified checkpoint upload/cleanup."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

STEPS = list(range(10, 101, 10))


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_runtime(runtime):
    manifest = read(runtime / "grpo_runtime_manifest.json")
    for name, checksum in manifest["files"].items():
        if digest(runtime / name) != checksum:
            raise ValueError(f"Frozen runtime changed: {name}")
    return manifest


def snapshot(source, overlay, runtime):
    """Freeze L+0's code and overlay the reviewed GRPO reward manager."""
    source, overlay, runtime = map(lambda p: Path(p).resolve(), (source, overlay, runtime))
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with (runtime.parent / "runtime_prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if runtime.exists():
            verify_runtime(runtime)
            return
        temporary = runtime.with_name(runtime.name + f".{os.getpid()}.tmp")
        temporary.mkdir()
        try:
            shutil.copytree(source / "verl", temporary / "verl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            scripts = temporary / "qwen3_experiments"
            scripts.mkdir()
            for name in ("run_qwen3_1_7b_polaris_1_8_3200_maxrl.sh",
                         "run_qwen3_1_7b_polaris_1_8_3200_per_context_rb_l0_0.sh", "verify_checkpoint_upload.py"):
                shutil.copy2(source / "qwen3_experiments" / name, scripts / name)
            for name in ("run_qwen3_1_7b_polaris_1_8_3200_grpo.sh", "grpo_compute_control.py"):
                shutil.copy2(overlay / "qwen3_experiments" / name, scripts / name)
            reward_path = Path("verl/workers/reward_manager/multi_thread_naive.py")
            shutil.copy2(overlay / reward_path, temporary / reward_path)
            files = {p.relative_to(temporary).as_posix(): digest(p) for p in sorted(temporary.rglob("*")) if p.is_file()}
            write(temporary / "grpo_runtime_manifest.json", {
                "source": str(source), "overlay": str(overlay), "created_at": now(), "files": files,
            })
            temporary.rename(runtime)
        except BaseException:
            shutil.rmtree(temporary)
            raise


def require_compute(job_id):
    description = subprocess.check_output(["scontrol", "show", "job", str(job_id), "-o"], text=True)
    fields = dict(word.split("=", 1) for word in description.split() if "=" in word)
    if (fields.get("JobState") != "RUNNING" or fields.get("NumNodes") != "1"
            or not fields.get("UserId", "").endswith(f"({os.getuid()})")):
        raise RuntimeError("Expected a running single-node allocation owned by this user")
    node = fields["BatchHost"]
    if socket.gethostname().split(".")[0] != node:
        raise RuntimeError(f"Control processes must run on compute node {node}")
    return node


def environment(plan):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("MAXRL_", "GRPO_", "RAY_", "VLLM_", "SLURM_", "WANDB_")) and key != "WANDB_API_KEY":
            env.pop(key)
    for key in ("PYTHONHOME", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "MASTER_ADDR", "MASTER_PORT",
                "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        env.pop(key, None)
    env.update({
        "PATH": str(Path(plan["python_bin"]).parent) + os.pathsep + env.get("PATH", ""),
        "PYTHONPATH": plan["runtime"], "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "GRPO_RUNTIME_REPO": plan["runtime"], "GRPO_L0_REPO_ROOT": plan["source_repo"],
        "GRPO_RUN_DIR": plan["output_root"], "GRPO_DATA_DIR": plan["data_dir"],
        "GRPO_MODEL_PATH": plan["model_path"], "GRPO_RAY_DIR": plan["ray_dir"],
        "GRPO_CHECKPOINT_DIR": plan["checkpoint_dir"], "WANDB_MODE": "online",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    return env


def training_command(plan):
    return ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=128",
            "--gres=gpu:8", "--kill-on-bad-exit=1", "--job-name=grpo-polaris", "bash",
            str(Path(plan["runtime"]) / "qwen3_experiments/run_qwen3_1_7b_polaris_1_8_3200_grpo.sh")]


def prepare(args):
    import yaml
    from huggingface_hub import HfApi

    node = require_compute(args.job_id)
    overlay = Path(__file__).resolve().parents[1]
    source = overlay.parent / "maxrl-eval-145514"
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = HfApi().whoami()
    write(root / "hf_auth_check.json", {"account": identity["name"], "hostname": node, "checked_at": now()})
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists():
            plan = read(root / "plan.json")
            if plan["job_id"] != args.job_id or plan["hf_repo_prefix"] != args.hf_prefix:
                raise ValueError("Existing run has different allocation or upload destination")
            verify_runtime(Path(plan["runtime"]))
            return plan
        original = source / "outputs/per_context_rb_l0_0_qwen3_1_7b_polaris_1_8_3200_bs32_32k_146103"
        original_plan = read(source / "outputs/queued_per_context_rb_l0_0_after_maxrl_145514/plan.json")
        model_manifest = read(original_plan["model_manifest"])
        runtime = root / "runtime"
        snapshot(source, overlay, runtime)
        data_dir = root / "data"
        data_dir.mkdir(exist_ok=True)
        for name in ("train.parquet", "unused_validation.parquet"):
            shutil.copy2(original / "data" / name, data_dir / name)
            assert digest(data_dir / name) == digest(original / "data" / name)
        scratch = Path(f"/tmp/grpo{args.job_id}")
        if (scratch / "checkpoints").exists() and any((scratch / "checkpoints").iterdir()):
            raise RuntimeError("Checkpoint destination is occupied by an earlier run")
        plan = {
            "job_id": args.job_id, "node": node, "output_root": str(root), "source_repo": str(source),
            "runtime": str(runtime), "python_bin": sys.executable, "data_dir": str(data_dir),
            "model_path": original_plan["model_source_dir"], "model_revision": original_plan["base_model_revision"],
            "checkpoint_dir": str(scratch / "checkpoints"), "ray_dir": str(scratch / "ray"),
            "hf_repo_prefix": args.hf_prefix, "new_hf_repositories_private": False,
            "save_freq": 10, "checkpoint_steps": STEPS, "total_steps": 100, "evaluation_enabled": False,
            "grading": {"check_eos": True, "score_after_thinking": True, "force_eos": False},
            "created_at": now(), "holder_locks": [
                str(source / f"outputs/logs/gpu_holder_{args.job_id}.launch.lock"),
                str(overlay.parent / f"maxrl-tailrl-full/outputs/text_maze_full_n32/control/holder_{args.job_id}.lock"),
            ],
        }
        script = runtime / "qwen3_experiments/run_qwen3_1_7b_polaris_1_8_3200_grpo.sh"
        preview = subprocess.run(["bash", str(script), "--cfg", "job", "--resolve"], env=environment(plan),
                                 cwd=runtime, text=True, capture_output=True, timeout=180)
        (root / "config_preview.stderr").write_text(preview.stderr)
        preview.check_returncode()
        config = yaml.safe_load(preview.stdout)
        assert config["algorithm"]["adv_estimator"] == "grpo"
        assert config["algorithm"]["norm_adv_by_std_in_grpo"] is True
        assert config["reward_model"]["reward_kwargs"] == {"check_eos": True, "score_after_thinking": True}
        assert config["trainer"]["save_freq"] == 10 and config["trainer"]["total_training_steps"] == 100
        assert config["trainer"]["resume_mode"] == "disable"
        assert not any(config["trainer"][key] for key in ("val_before_train", "val_on_last_step", "eval_on_last_step"))
        assert config["trainer"]["test_freq"] == -1
        before = yaml.safe_load((original / "resolved_config.yaml").read_text())
        allowed = {"algorithm.adv_estimator", "trainer.experiment_name", "trainer.save_freq", "trainer.default_local_dir",
                   "actor_rollout_ref.model.path", "critic.model.tokenizer_path", "reward_model.model.input_tokenizer",
                   "reward_model.reward_kwargs", "data.train_files", "data.val_files", "ray_init.ray_dir"}
        differences = []

        def compare(left, right, key=""):
            if key in allowed:
                if left != right:
                    differences.append({"path": key, "l0": left, "grpo": right})
            elif isinstance(left, dict) and isinstance(right, dict):
                for field in set(left) | set(right):
                    compare(left.get(field), right.get(field), f"{key}.{field}" if key else field)
            elif left != right:
                raise ValueError(f"Unexpected change from L+0: {key}: {left!r} -> {right!r}")

        compare(before, config)
        (root / "resolved_config.yaml").write_text(preview.stdout)
        write(root / "config_comparison.json", differences)
        plan["input_hashes"] = {str(path): digest(path) for path in (
            data_dir / "train.parquet", data_dir / "unused_validation.parquet", root / "resolved_config.yaml",
        )}
        for name, expected in model_manifest["files_sha256"].items():
            path = Path(plan["model_path"]) / name
            if digest(path) != expected:
                raise ValueError(f"Initial model differs from the pinned L+0 model: {name}")
            plan["input_hashes"][str(path)] = expected
        write(root / "plan.json", plan)
        return plan


def checkpoint_complete(checkpoint, directory, world_size=8):
    checkpoint, directory = Path(checkpoint), Path(directory).resolve()
    if checkpoint.is_symlink() or checkpoint.parent.resolve() != directory:
        return False
    match = re.fullmatch(r"global_step_(\d+)", checkpoint.name)
    if not match or not checkpoint.is_dir():
        return False
    try:
        latest = int((directory / "latest_checkpointed_iteration.txt").read_text().strip())
    except (OSError, ValueError):
        return False
    if int(match[1]) > latest:
        return False
    required = [checkpoint / "data.pt", checkpoint / "actor/config.json", checkpoint / "actor/tokenizer_config.json"]
    required += [checkpoint / "actor" / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
                 for kind in ("model", "optim", "extra_state") for rank in range(world_size)]
    return all(path.is_file() and not path.is_symlink() and path.stat().st_size > 0 for path in required)


def load_verifier(plan):
    path = Path(plan["runtime"]) / "qwen3_experiments/verify_checkpoint_upload.py"
    spec = importlib.util.spec_from_file_location("grpo_checkpoint_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_public_repository(api, repo_id, repo_type="model"):
    """Apply the user's public-storage policy before uploading training artifacts."""
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False, exist_ok=True)
    if api.repo_info(repo_id=repo_id, repo_type=repo_type).private:
        api.update_repo_settings(repo_id=repo_id, repo_type=repo_type, private=False)
        if api.repo_info(repo_id=repo_id, repo_type=repo_type).private:
            raise RuntimeError(f"Expected a public {repo_type} repository: {repo_id}")


def archive_checkpoint(checkpoint, receipt_path, repo_id, api, verifier):
    ensure_public_repository(api, repo_id)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(checkpoint),
                      path_in_repo=checkpoint.name, ignore_patterns=[".cache/**"],
                      commit_message=f"Archive {checkpoint.name}")
    verifier.verify_checkpoint(checkpoint, repo_id, receipt_path, api)
    receipt = verifier.check_local(checkpoint, receipt_path)
    shutil.rmtree(checkpoint)
    receipt.update(state="archived_and_deleted", deleted_at=now())
    write(receipt_path, receipt)


def monitor(plan):
    from huggingface_hub import HfApi

    require_compute(plan["job_id"])
    root, directory = Path(plan["output_root"]), Path(plan["checkpoint_dir"])
    folder = root / "hf_checkpoint_archive"
    folder.mkdir(exist_ok=True)
    receipts = folder / "receipts"
    receipts.mkdir(exist_ok=True)
    api, verifier = HfApi(), load_verifier(plan)
    with (folder / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            state = {"pid": os.getpid(), "hostname": socket.gethostname(), "state": "monitoring", "updated_at": now()}
            pending, errors = [], []
            for step in STEPS:
                checkpoint = directory / f"global_step_{step}"
                if not checkpoint.exists():
                    continue
                pending.append(step)
                if not checkpoint_complete(checkpoint, directory):
                    continue
                # The supervisor verifies the final save before releasing it for removal.
                if step == 100 and not (root / "training_exit.json").exists():
                    continue
                try:
                    state.update(state="uploading", step=step)
                    write(folder / "status.json", state)
                    archive_checkpoint(checkpoint, receipts / f"global_step_{step}.json",
                                       f"{plan['hf_repo_prefix']}-step_{step}", api, verifier)
                except Exception as exc:
                    errors.append({"step": step, "error": str(exc)})
                    print(f"Retaining checkpoint {step} for retry: {exc}", flush=True)
                    break
            archived = [step for step in STEPS if (receipts / f"global_step_{step}.json").exists()
                        and read(receipts / f"global_step_{step}.json")["state"] == "archived_and_deleted"]
            state.update(state="retrying" if errors else "monitoring", archived_steps=archived,
                         errors=errors, updated_at=now())
            if (root / "training_exit.json").exists() and not errors:
                training = read(root / "training_exit.json")
                remaining = [step for step in STEPS if (directory / f"global_step_{step}").exists()]
                if not remaining:
                    complete = training["exit_code"] == 0 and archived == STEPS
                    state.update(state="complete" if complete else "training_failed_or_missing_checkpoint")
                    write(folder / "status.json", state)
                    return 0 if complete else 1
                if any(not checkpoint_complete(directory / f"global_step_{step}", directory) for step in remaining):
                    state.update(state="incomplete_checkpoint_retained", remaining_steps=remaining)
                    write(folder / "status.json", state)
                    return 1
            write(folder / "status.json", state)
            time.sleep(30)


def supervise(plan):
    require_compute(plan["job_id"])
    root = Path(plan["output_root"])
    verify_runtime(Path(plan["runtime"]))
    for path, checksum in plan["input_hashes"].items():
        if digest(path) != checksum:
            raise ValueError(f"Prepared input changed: {path}")
    with ExitStack() as stack:
        for path in [root / "supervisor.lock", *map(Path, plan["holder_locks"])]:
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "status.json").exists():
            raise RuntimeError("Training was already started in this run directory")
        busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        if busy.strip():
            raise RuntimeError("Allocated GPUs are occupied; refusing to overlap another run")
        state = {"state": "initializing", "pid": os.getpid(), "hostname": socket.gethostname(),
                 "job_id": plan["job_id"], "last_completed_step": 0, "started_at": now(), "evaluation_enabled": False}
        write(root / "status.json", state)
        archive = root / "hf_checkpoint_archive"
        archive.mkdir(exist_ok=True)
        env = environment(plan)
        training_log = stack.enter_context((root / "train.log").open("ab", buffering=0))
        archive_log = stack.enter_context((archive / "upload.log").open("ab", buffering=0))
        child = subprocess.Popen(training_command(plan), cwd=plan["runtime"], env=env,
                                 stdin=subprocess.DEVNULL, stdout=training_log, stderr=subprocess.STDOUT)
        state["training_launcher_pid"] = child.pid

        def start_monitor():
            return subprocess.Popen([plan["python_bin"], "-u", str(Path(__file__).resolve()), "monitor",
                                     "--plan", str(root / "plan.json")], env=env, stdin=subprocess.DEVNULL,
                                    stdout=archive_log, stderr=subprocess.STDOUT)

        archiver = start_monitor()
        log_reader = stack.enter_context((root / "train.log").open("r", errors="replace"))
        while True:
            text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", log_reader.read())
            steps = [int(value) for value in re.findall(r"\bstep:\s*(\d+)\b", text)]
            if steps:
                state["last_completed_step"] = max(state["last_completed_step"], *steps)
                state["state"] = "training"
            state.update(checkpoint_monitor_pid=archiver.pid, updated_at=now())
            write(root / "status.json", state)
            if child.poll() is not None:
                break
            if archiver.poll() is not None:
                state["checkpoint_monitor_restarts"] = state.get("checkpoint_monitor_restarts", 0) + 1
                archiver = start_monitor()
            time.sleep(10)
        final = Path(plan["checkpoint_dir"]) / "global_step_100"
        success = child.returncode == 0 and checkpoint_complete(final, final.parent)
        exit_code = child.returncode if child.returncode else (0 if success else 1)
        write(root / "training_exit.json", {"exit_code": exit_code, "launcher_exit_code": child.returncode, "finished_at": now()})
        state.update(state="archiving" if success else "training_failed", training_exit_code=exit_code, updated_at=now())
        if success:
            state["last_completed_step"] = 100
        write(root / "status.json", state)
        if archiver.poll() is not None:
            archiver = start_monitor()
        archive_code = archiver.wait()
        state.update(state="complete" if success and archive_code == 0 else "failed",
                     archive_exit_code=archive_code, finished_at=now(), updated_at=now())
        write(root / "status.json", state)
        return 0 if state["state"] == "complete" else 1


def launch(plan):
    require_compute(plan["job_id"])
    root = Path(plan["output_root"])
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            raise RuntimeError("Persistent supervisor already launched")
        script = Path(plan["runtime"]) / "qwen3_experiments/grpo_compute_control.py"
        with (root / "supervisor.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([plan["python_bin"], "-u", str(script), "supervise", "--plan", str(root / "plan.json")],
                                     cwd=plan["runtime"], env=environment(plan), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"pid": child.pid, "hostname": socket.gethostname(), "job_id": plan["job_id"], "launched_at": now()}
        write(root / "launch.json", receipt)
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("snapshot", "prepare", "launch", "supervise", "monitor"))
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--overlay-repo", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--job-id", default="146102")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--hf-prefix")
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.source_repo, args.overlay_repo, args.runtime_dir)
    elif args.command in ("prepare", "launch"):
        if not args.output_root or not args.hf_prefix:
            parser.error("prepare/launch requires --output-root and --hf-prefix")
        plan = prepare(args)
        if args.command == "launch":
            launch(plan)
        else:
            print(json.dumps({"state": "prepared", "hostname": plan["node"], "plan": str(args.output_root / "plan.json")}))
    else:
        plan = read(args.plan)
        raise SystemExit({"supervise": supervise, "monitor": monitor}[args.command](plan))


if __name__ == "__main__":
    main()
