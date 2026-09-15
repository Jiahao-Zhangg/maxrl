"""Launch one Math12K experiment after a specified run and its uploads finish."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from qwen3_experiments.upload_training_rollouts_to_hf import process_matches, write_json_atomic


def gpu_blockers(indices):
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True,
    ).splitlines()
    devices = {int(row.split(",")[0]): row.split(",")[1].strip() for row in gpu_rows}
    if not set(indices) <= set(devices):
        raise RuntimeError(f"Requested GPU indices are unavailable: {indices}")
    selected = {devices[index] for index in indices}
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True,
    ).splitlines()
    return [row.strip() for row in rows if row.split(",")[0].strip() in selected]


def predecessor_status(plan):
    predecessor = plan["predecessor"]
    if process_matches(predecessor["pid"], predecessor["start_time"]):
        return "waiting_for_predecessor"
    checkpoint_dir = Path(predecessor["checkpoint_dir"])
    exit_file = checkpoint_dir / "logs/training.exit_status"
    if not exit_file.exists() or exit_file.read_text().strip() != "0":
        return "predecessor_failed"
    archive_log = checkpoint_dir / "logs/checkpoint_upload.log"
    if not archive_log.exists() or "Checkpoint archival is complete" not in archive_log.read_text():
        return "waiting_for_checkpoint_uploads"
    if any(path.is_dir() for path in checkpoint_dir.glob("global_step_*")):
        return "waiting_for_checkpoint_cleanup"
    rollout_file = Path(predecessor["rollout_upload_state"])
    if not rollout_file.exists():
        return "waiting_for_rollout_uploads"
    rollout = json.loads(rollout_file.read_text())
    expected = set(range(1, predecessor["final_step"] + 1))
    if rollout.get("status") != "complete" or set(rollout.get("uploaded_steps", [])) != expected:
        return "waiting_for_rollout_uploads"
    return "ready"


def verify_predecessor_uploads(plan, api):
    predecessor = plan["predecessor"]
    for step in predecessor["checkpoint_steps"]:
        repo = f"{predecessor['checkpoint_hf_prefix']}-step_{step}"
        info = api.repo_info(repo_id=repo, repo_type="model", files_metadata=True)
        files = {item.rfilename: item.size for item in info.siblings}
        prefix = f"global_step_{step}"
        expected = [f"{prefix}/data.pt"] + [
            f"{prefix}/actor/{kind}_world_size_{predecessor['num_gpus']}_rank_{rank}.pt"
            for kind in ("model", "optim", "extra_state")
            for rank in range(predecessor["num_gpus"])
        ]
        if any(not files.get(name) for name in expected):
            raise RuntimeError(f"Predecessor checkpoint is incomplete on HF: {repo}")
    rollout = json.loads(Path(predecessor["rollout_upload_state"]).read_text())
    identity = rollout["identity"]
    if identity["final_step"] != predecessor["final_step"] or identity["expected_rows"] != predecessor["rows_per_step"]:
        raise RuntimeError("Predecessor rollout collection has unexpected step or row counts")
    info = api.repo_info(repo_id=identity["repo_id"], repo_type="dataset", files_metadata=True)
    remote = {item.rfilename: item.size for item in info.siblings}
    dataset_dir = Path(identity["dataset_dir"])
    expected = ["README.md", "rollout_manifest.json"] + [
        f"data/step_{step:06d}.jsonl.gz" for step in range(1, predecessor["final_step"] + 1)
    ]
    for name in expected:
        local = dataset_dir / name
        if not local.is_file() or remote.get(name) != local.stat().st_size:
            raise RuntimeError(f"Predecessor rollout upload is not verified: {name}")


def launch_environment(plan, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    for key in list(env):
        if key.startswith(("MAXRL_", "WANDB_", "RAY_", "VLLM_")) and key != "WANDB_API_KEY":
            env.pop(key)
    for key in ("PYTHONPATH", "PYTHONHOME", "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        env.pop(key, None)
    python = Path(plan["python_bin"])
    output = Path(plan["output_root"])
    env.update({
        "PATH": str(python.parent) + os.pathsep + env.get("PATH", ""),
        "CONDA_PREFIX": str(python.parent.parent),
        "CONDA_DEFAULT_ENV": python.parent.parent.name,
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in plan["gpu_indices"]),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "MAXRL_SKIP_ENV_SETUP": "1",
        "MAXRL_DATA_DIR": plan["data_dir"],
        "MAXRL_OUTPUT_DIR": str(output),
        "MAXRL_RAY_DIR": plan["ray_dir"],
        "RAY_TMPDIR": plan["ray_dir"],
        "RAY_ADDRESS": "local",
        "TMPDIR": plan["tmp_dir"],
        "MAXRL_MODEL_PATH": plan["model_path"],
        "MAXRL_COST_OFFSET_TOKENS": str(plan["cost_offset_tokens"]),
        "MAXRL_TOTAL_TRAINING_STEPS": str(plan["total_steps"]),
        "MAXRL_SAVE_FREQ": str(plan["save_freq"]),
        "MAXRL_TEST_FREQ": str(plan["save_freq"]),
        "MAXRL_EXPERIMENT_NAME": plan["experiment_name"],
        "MAXRL_UPLOAD_CHECKPOINTS": "1",
        "MAXRL_CHECKPOINT_HF_REPO_PREFIX": plan["checkpoint_hf_prefix"],
        "MAXRL_SAVE_ROLLOUT_DATASET": "1",
        "MAXRL_ROLLOUT_DATASET_HF_REPO": plan["rollout_hf_repo"],
        "MAXRL_ROLLOUT_DATASET_DIR": str(output / "rollout_dataset"),
        "WANDB_RUN_ID": plan["wandb_run_id"],
        "WANDB_RESUME": "never",
        "WANDB_MODE": "online",
        "WANDB_ENTITY": plan["wandb_entity"],
        "WANDB_DIR": str(output / "wandb"),
    })
    return env


class RunQueue:
    def __init__(self, plan, state_file, api, inherited=None):
        self.plan = plan
        self.state_file = Path(state_file)
        self.api = api
        self.inherited = inherited
        self.child = None
        self.idle_confirmations = 0
        identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        self.state = {"plan_sha256": identity, "phase": "waiting"}
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text())
            if self.state.get("plan_sha256") != identity:
                raise ValueError("The saved queue belongs to a different launch plan")

    def record(self, status, **details):
        changed = self.state.get("status") != status
        self.state.update(status=status, updated_at=time.time(), **details)
        write_json_atomic(self.state_file, self.state)
        if changed:
            print(json.dumps({"status": status, **details}), flush=True)

    def preflight(self):
        from huggingface_hub.errors import RepositoryNotFoundError
        from huggingface_hub.utils import validate_repo_id

        for path in (self.plan["launcher"], self.plan["python_bin"]):
            if not Path(path).is_file():
                raise RuntimeError(f"Missing launch dependency: {path}")
        output = Path(self.plan["output_root"])
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Refusing to overwrite existing run outputs: {output}")
        if Path(self.plan["ray_dir"]).exists():
            raise RuntimeError("The planned Ray directory already exists")
        # Python multiprocessing appends a socket suffix to TMPDIR.
        if len(os.fsencode(self.plan["tmp_dir"])) > 60:
            raise RuntimeError("Use a short TMPDIR for multiprocessing Unix sockets")
        for name in ("math12k/train.parquet", "aime25/test.parquet", "math500/test.parquet"):
            path = Path(self.plan["data_dir"]) / name
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"Missing prepared dataset: {path}")
        targets = [(self.plan["rollout_hf_repo"], "dataset")] + [
            (f"{self.plan['checkpoint_hf_prefix']}-step_{step}", "model")
            for step in range(self.plan["save_freq"], self.plan["total_steps"] + 1, self.plan["save_freq"])
        ]
        for repo, repo_type in targets:
            validate_repo_id(repo)
            try:
                self.api.repo_info(repo_id=repo, repo_type=repo_type)
            except RepositoryNotFoundError:
                continue
            raise RuntimeError(f"Refusing to overwrite an existing HF repository: {repo}")

    def tick(self):
        if self.state["phase"] != "waiting":
            return False
        if self.state_file.with_name("STOP").exists():
            self.record("cancelled", phase="cancelled")
            return False
        status = predecessor_status(self.plan)
        if status != "ready":
            self.idle_confirmations = 0
            self.record(status, phase="blocked" if status == "predecessor_failed" else "waiting")
            return status != "predecessor_failed"
        blockers = gpu_blockers(self.plan["gpu_indices"])
        if blockers:
            self.idle_confirmations = 0
            self.record("waiting_for_gpus", blockers=blockers)
            return True
        output = Path(self.plan["output_root"])
        ancestor = output
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < self.plan.get("min_output_free_gib", 50) * 2**30:
            self.record("waiting_for_disk")
            return True
        self.idle_confirmations += 1
        if self.idle_confirmations < 2:
            self.record("confirming_idle")
            return True
        verify_predecessor_uploads(self.plan, self.api)
        self.preflight()
        # Recheck after network operations, immediately before taking the GPUs.
        if predecessor_status(self.plan) != "ready" or gpu_blockers(self.plan["gpu_indices"]):
            self.idle_confirmations = 0
            self.record("waiting_for_gpus")
            return True
        self.record("launching", phase="launching")
        output.mkdir(parents=True, exist_ok=True)
        Path(self.plan["tmp_dir"]).mkdir(parents=True, exist_ok=True)
        (output / "wandb").mkdir(exist_ok=True)
        command = ["bash", self.plan["launcher"], "trainer.resume_mode=disable"]
        with (output / "launch.log").open("a", buffering=1) as log:
            self.child = subprocess.Popen(
                command, cwd=self.plan["repo_root"], env=launch_environment(self.plan, self.inherited),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
        self.record("launched", phase="launched", launcher_pid=self.child.pid, command=command)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--check", action="store_true", help="Validate static inputs without launching or writing queue state")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    from huggingface_hub import HfApi

    queue = RunQueue(json.loads(Path(args.plan).read_text()), args.state_file, HfApi())
    if args.check:
        queue.preflight()
        print(json.dumps({"predecessor_status": predecessor_status(queue.plan), "preflight": "passed"}))
        return 0
    queue.state_file.parent.mkdir(parents=True, exist_ok=True)
    with queue.state_file.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                if not queue.tick():
                    break
            except Exception as error:
                queue.record("attention", error=f"{type(error).__name__}: {error}")
                if queue.state["phase"] != "waiting":
                    return 1
            time.sleep(args.poll_seconds)
        if queue.child is not None:
            status = queue.child.wait()
            queue.record("complete" if status == 0 else "failed", phase="finished", exit_code=status)
            return status
        return 1 if queue.state["phase"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
