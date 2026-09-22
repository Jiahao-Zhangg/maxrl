"""Verify a checkpoint on HF before releasing Slurm and optionally resuming.

The JSON configuration supplies all site-specific paths and job IDs. This
controller never deletes checkpoints, submits allocations, or changes training
code in a running process. Set resume_after_release to false to finish after
releasing the source, without using another holder. Run it inside a persistent
terminal session.
"""

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import subprocess
import time
from pathlib import Path


class SafetyError(RuntimeError):
    """Require human intervention instead of making a potentially unsafe change."""


TERMINAL_STATES = {
    "CANCELLED", "COMPLETED", "FAILED", "TIMEOUT", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checkpoint_complete(experiment_dir, step, world_size, wandb_id):
    experiment = Path(experiment_dir)
    checkpoint = experiment / f"global_step_{step}"
    required = [checkpoint / "data.pt", checkpoint / "actor/config.json", checkpoint / "actor/tokenizer_config.json"]
    required += [
        checkpoint / "actor" / f"{component}_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
        for component in ("model", "optim", "extra_state")
    ]
    try:
        latest = (experiment / "latest_checkpointed_iteration.txt").read_text().strip()
        saved_run = (experiment / "wandb_id.txt").read_text().strip()
        if saved_run != wandb_id:
            raise SafetyError("Checkpoint belongs to a different W&B run")
        if not latest.isdigit() or int(latest) < step:
            return False
        return all(path.is_file() and not path.is_symlink() and path.stat().st_size > 0 for path in required)
    except FileNotFoundError:
        return False


def file_fingerprint(source):
    """Check both LFS SHA-256 and ordinary Git blob IDs without downloading weights."""
    size = len(source) if isinstance(source, bytes) else Path(source).stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode())
    stream = io.BytesIO(source) if isinstance(source, bytes) else Path(source).open("rb")
    with stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    return {"size": size, "sha256": sha256.hexdigest(), "git_blob": git_blob.hexdigest()}


def field(item, name):
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def remote_file_matches(remote, expected):
    if field(remote, "size") != expected["size"]:
        return False
    lfs = field(remote, "lfs")
    if lfs is not None:
        return field(lfs, "sha256") == expected["sha256"]
    return field(remote, "blob_id") == expected["git_blob"]


def verify_remote(api, repo_id, revision, manifest):
    info = api.repo_info(repo_id=repo_id, repo_type="model", revision=revision, files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings}
    bad = [name for name, expected in manifest.items() if name not in remote or not remote_file_matches(remote[name], expected)]
    if bad:
        raise RuntimeError(f"HF verification failed for {bad[:4]}")
    return info.sha


def upload_verified_checkpoint(config, api):
    """Upload only the selected checkpoint and resumable metadata; keep local files."""
    from huggingface_hub import CommitOperationAdd

    experiment = Path(config["experiment_dir"])
    checkpoint = experiment / f"global_step_{config['target_step']}"
    if not checkpoint_complete(experiment, config["target_step"], config["world_size"], config["wandb_id"]):
        raise RuntimeError("Checkpoint is not complete")
    if checkpoint.is_symlink():
        raise SafetyError("Refusing a symlinked checkpoint")
    sources = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise SafetyError(f"Refusing checkpoint symlink: {path}")
        if path.is_file() and ".cache" not in path.relative_to(checkpoint).parts:
            sources[path.relative_to(experiment).as_posix()] = path
    sources["wandb_id.txt"] = (config["wandb_id"] + "\n").encode()
    sources["latest_checkpointed_iteration.txt"] = (str(config["target_step"]) + "\n").encode()
    local_files = {
        str(source.relative_to(experiment)): {
            "size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns,
        }
        for source in sources.values() if isinstance(source, Path)
    }
    manifest = {name: file_fingerprint(source) for name, source in sources.items()}
    repo_id = config["hf_repo_id"]
    api.create_repo(repo_id=repo_id, repo_type="model", private=config.get("hf_private", True), exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    for remote in info.siblings:
        if remote.rfilename == ".gitattributes":
            continue
        expected = manifest.get(remote.rfilename)
        if expected is None or not remote_file_matches(remote, expected):
            raise SafetyError(f"Refusing to overwrite conflicting HF content: {remote.rfilename}")
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=source) for name, source in sources.items()]
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        parent_commit=info.sha,
        commit_message=f"Archive complete training checkpoint at step {config['target_step']}",
        num_threads=4,
    )
    revision = verify_remote(api, repo_id, commit.oid, manifest)
    for name, expected in local_files.items():
        stat = (experiment / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise SafetyError(f"Local checkpoint changed during upload: {name}")
    return {
        "repo_id": repo_id, "revision": revision, "step": config["target_step"],
        "wandb_id": config["wandb_id"], "manifest": manifest,
        "local_files": local_files,
    }


def verify_receipt(config, receipt, api):
    if (receipt["repo_id"], receipt["step"], receipt["wandb_id"]) != (config["hf_repo_id"], config["target_step"], config["wandb_id"]):
        raise SafetyError("Upload receipt does not match this handoff")
    if not checkpoint_complete(config["experiment_dir"], config["target_step"], config["world_size"], config["wandb_id"]):
        raise SafetyError("Local resume checkpoint is no longer complete")
    for name, expected in receipt["local_files"].items():
        stat = (Path(config["experiment_dir"]) / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise SafetyError(f"Local checkpoint changed after upload: {name}")
    verify_remote(api, receipt["repo_id"], receipt["revision"], receipt["manifest"])


def parse_job(record):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", record))


def validate_holder(job, config, job_id):
    expected = {
        "JobId": str(job_id), "JobName": config["holder_name"],
        "Account": config["account"], "Partition": config["partition"],
        "Command": config["holder_command"],
    }
    if any(job.get(key) != value for key, value in expected.items()):
        raise SafetyError(f"Allocation {job_id} no longer matches the authorized holder")
    if not job.get("UserId", "").startswith(config["owner"] + "("):
        raise SafetyError(f"Allocation {job_id} has an unexpected owner")
    tres = dict(part.split("=", 1) for part in job.get("ReqTRES", "").split(",") if "=" in part)
    if tres.get("gres/gpu") != str(config["world_size"]):
        raise SafetyError(f"Allocation {job_id} has an unexpected GPU count")


class Slurm:
    @staticmethod
    def command(args, timeout=25):
        result = subprocess.run(args, check=True, text=True, capture_output=True, timeout=timeout)
        return result.stdout.strip()

    def job(self, job_id):
        result = subprocess.run(["scontrol", "show", "job", "-o", str(job_id)], text=True, capture_output=True, timeout=25)
        if result.returncode == 0 and result.stdout.strip():
            return parse_job(result.stdout)
        # An RPC failure must never be mistaken for a released allocation.
        accounting = self.command(["sacct", "-n", "-P", "-X", "-j", str(job_id), "-o", "JobIDRaw,State"])
        for line in accounting.splitlines():
            parts = line.split("|")
            if parts[0] == str(job_id) and parts[1].split()[0] in TERMINAL_STATES:
                return {"JobId": str(job_id), "JobState": parts[1].split()[0]}
        raise RuntimeError(f"Cannot reliably query allocation {job_id}")

    def steps(self, job_id):
        output = self.command(["squeue", "--steps", "--noheader", "--jobs", str(job_id), "--format=%i"])
        return {line.strip() for line in output.splitlines() if re.fullmatch(rf"{job_id}\.\d+", line.strip())}

    def cancel(self, target):
        self.command(["scancel", str(target)])

    def idle(self, job):
        job_id = job["JobId"]
        if self.steps(job_id):
            return False
        node = job.get("NodeList", "")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]+", node):
            raise SafetyError(f"Unexpected node list for {job_id}: {node}")
        output = self.command([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node,
            "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits",
        ])
        values = [tuple(int(value.strip()) for value in line.split(",")) for line in output.splitlines() if line.strip()]
        return len(values) == 4 and all(util <= 5 and memory <= 64 for _, util, memory in values) and not self.steps(job_id)


def release_source(config, receipt, api, slurm):
    """Return True only after the source allocation is confirmed released."""
    verify_receipt(config, receipt, api)
    source = str(config["source_job"])
    job = slurm.job(source)
    if job["JobState"] in TERMINAL_STATES:
        return True
    validate_holder(job, config, source)
    if job["JobState"] == "COMPLETING":
        return False
    if job["JobState"] != "RUNNING":
        raise SafetyError(f"Unexpected source state: {job['JobState']}")
    steps = slurm.steps(source)
    if steps - {config["source_step"]}:
        raise SafetyError("Source holder contains another training step; retaining allocation")
    if config["source_step"] in steps:
        slurm.cancel(config["source_step"])
        return False
    # Recheck identity and active work immediately before the allocation cancel.
    validate_holder(slurm.job(source), config, source)
    if slurm.steps(source):
        return False
    slurm.cancel(source)
    return False


def candidate_holders(config, slurm):
    ready = []
    for job_id in config["candidate_jobs"]:
        job = slurm.job(job_id)
        if job["JobState"] in TERMINAL_STATES:
            continue
        validate_holder(job, config, job_id)
        if job["JobState"] == "RUNNING" and slurm.idle(job):
            ready.append(job)
    return sorted(ready, key=lambda job: (job.get("StartTime", ""), int(job["JobId"])))


class Handoff:
    def __init__(self, config, api, slurm=None):
        self.config = config
        self.api = api
        self.slurm = slurm or Slurm()
        self.state_path = Path(config["state_path"])
        self.receipt_path = Path(config["receipt_path"])
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        identity = {key: config[key] for key in ("source_job", "source_step", "target_step", "wandb_id", "hf_repo_id", "candidate_jobs")}
        if self.state and self.state.get("identity") != identity:
            raise SafetyError("Saved watcher state belongs to a different handoff")
        self.state["identity"] = identity
        self.last_message = None
        self.process = None
        self.launch_lock = None

    def update(self, phase, message, **extra):
        self.state.update(phase=phase, message=message, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        atomic_json(self.state_path, self.state)
        if (phase, message) != self.last_message:
            print(f"{self.state['updated_at']} {phase}: {message}", flush=True)
            self.last_message = (phase, message)

    def launch(self, job):
        config = self.config
        if not config.get("resume_after_release", True):
            raise SafetyError("Automatic continuation is disabled")
        job_id = job["JobId"]
        stem = f"maxrl_qwen3_0_6b_polaris_bs16_32k_resume{config['target_step']}_save50_job{job_id}"
        log_dir = Path(config["log_dir"])
        lock_path = log_dir / f"maxrl_qwen3_0_6b_polaris_bs16_32k_job{job_id}.watch.lock"
        lock = lock_path.open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return False
        self.launch_lock = lock
        latest_job = self.slurm.job(job_id)
        validate_holder(latest_job, config, job_id)
        if latest_job["JobState"] != "RUNNING" or not self.slurm.idle(latest_job):
            lock.close()
            self.launch_lock = None
            return False
        train_log = log_dir / f"{stem}.training.log"
        marker = log_dir / f"{stem}.started"
        if train_log.exists() or marker.exists():
            raise SafetyError(f"Continuation already has a launch marker for {job_id}; refusing duplicate launch")
        # Persist intent BEFORE starting srun. A crash in this small window
        # requires reconciliation, never an automatic duplicate launch.
        marker.touch(exist_ok=False)
        self.update("LAUNCHING", f"Resuming step {config['target_step']} on holder {job_id}",
                    launch_requested=True, selected_job=job_id, training_log=str(train_log), launch_time=time.time())
        with train_log.open("xb") as output:
            self.process = subprocess.Popen(
                ["srun", f"--jobid={job_id}", "--overlap", "--nodes=1", "--ntasks=1",
                 f"--cpus-per-task={config['cpus']}", f"--gpus-per-task={config['world_size']}", "--unbuffered",
                 f"--chdir={config['repo_root']}", "/usr/bin/bash", "-l", config["resume_runner"], job_id],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),),
            )
        self.update("STARTING_CONTINUATION", f"srun launched on {job_id}; waiting for checkpoint restoration", srun_pid=self.process.pid)
        return True

    def tick(self):
        config = self.config
        resume_after_release = config.get("resume_after_release", True)
        if not resume_after_release and self.state.get("launch_requested"):
            raise SafetyError("Continuation was already requested; inspect the recorded srun before declaring release-only completion")
        if not self.receipt_path.exists():
            complete = checkpoint_complete(config["experiment_dir"], config["target_step"], config["world_size"], config["wandb_id"])
            if not complete:
                job = self.slurm.job(config["source_job"])
                if job["JobState"] in TERMINAL_STATES:
                    raise SafetyError("Source allocation ended before the requested checkpoint became complete")
                validate_holder(job, config, config["source_job"])
                if config["source_step"] not in self.slurm.steps(config["source_job"]):
                    raise SafetyError("Source training stopped before the requested checkpoint became complete")
                self.update("WAITING_CHECKPOINT", f"Waiting for complete step {config['target_step']}; current training is untouched")
                return False
            # Keep ordering explicit: complete checkpoint -> verified upload ->
            # stop training -> release holder -> optional single continuation.
            self.update("UPLOADING", f"Uploading step {config['target_step']} to {config['hf_repo_id']}; retaining all local files")
            with Path(config["upload_lock"]).open("a") as upload_lock:
                fcntl.flock(upload_lock, fcntl.LOCK_EX)
                receipt = upload_verified_checkpoint(config, self.api)
            atomic_json(self.receipt_path, receipt)
            self.update("UPLOAD_VERIFIED", f"Verified all HF files at commit {receipt['revision']}; local resume state retained")
            return False
        receipt = json.loads(self.receipt_path.read_text())
        if not self.state.get("source_released"):
            self.update("RELEASING_SOURCE", f"HF verified; stopping only {config['source_step']} and releasing {config['source_job']}")
            if not release_source(config, receipt, self.api, self.slurm):
                return False
            self.state["source_released"] = True
            if resume_after_release:
                self.update("WAITING_ALLOCATION", "Source allocation released; waiting for one authorized idle holder", source_released=True)
                return False
        if not resume_after_release:
            self.update("RELEASE_COMPLETE", "Checkpoint verified on HF and source allocation released; automatic continuation is disabled")
            return True
        if not self.state.get("launch_requested"):
            ready = candidate_holders(config, self.slurm)
            if not ready:
                states = [self.slurm.job(job)["JobState"] for job in config["candidate_jobs"]]
                if all(state in TERMINAL_STATES for state in states):
                    raise SafetyError("All authorized next allocations ended; no replacement was submitted")
                self.update("WAITING_ALLOCATION", "Waiting for an allocated, idle candidate; no other holder will be cancelled")
                return False
            verify_receipt(config, receipt, self.api)
            for job in ready:
                if self.launch(job):
                    break
            return False
        if self.process is None:
            raise SafetyError("A prior launch intent exists; inspect the recorded srun before restarting this controller")
        code = self.process.poll()
        if code is not None:
            self.update("CONTINUATION_EXITED", f"Continuation srun exited with code {code}; checkpoints retained", exit_code=code)
            return True
        if not self.state.get("startup_confirmed"):
            log = Path(self.state["training_log"]).read_text(errors="replace")
            required = [f"Setting global step to {config['target_step']}", f"runs/{config['wandb_id']}",
                        "'save_freq': 50", "'test_freq': 250", "'val_before_train': False",
                        f"{config['target_step']}/1272"]
            if all(text in log for text in required):
                self.update("CONTINUATION_RUNNING", f"Restored step {config['target_step']}; same W&B; save every 50 steps", startup_confirmed=True)
            elif time.time() - self.state["launch_time"] > 1200:
                self.update("CONTINUATION_NEEDS_CHECK", "Startup confirmation is delayed; keeping the process and refusing duplicate launches")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--check", action="store_true", help="Read-only preflight; do not upload, cancel, or launch")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not isinstance(config.get("resume_after_release", True), bool):
        raise SafetyError("resume_after_release must be a JSON boolean")
    if str(config["source_job"]) in [str(job) for job in config["candidate_jobs"]]:
        raise SafetyError("Source allocation cannot also be a continuation candidate")
    if config["source_step"].split(".")[0] != str(config["source_job"]):
        raise SafetyError("Source step does not belong to the source allocation")
    if len(set(config["candidate_jobs"])) != len(config["candidate_jobs"]):
        raise SafetyError("Duplicate continuation candidates")
    from huggingface_hub import HfApi

    api = HfApi()
    if args.check:
        if api.whoami()["name"] != config["hf_repo_id"].split("/")[0]:
            raise SafetyError("HF token is not for the intended account")
        slurm = Slurm()
        candidates = config["candidate_jobs"] if config.get("resume_after_release", True) else []
        for job_id in [config["source_job"], *candidates]:
            job = slurm.job(job_id)
            validate_holder(job, config, job_id)
            print(f"Validated holder {job_id}: {job['JobState']}")
        if config.get("resume_after_release", True) and not Path(config["resume_runner"]).is_file():
            raise SafetyError("Continuation runner is missing")
        print("Preflight passed; no upload, cancellation, or launch performed")
        return
    Path(config["log_dir"]).mkdir(parents=True, exist_ok=True)
    with Path(config["watch_lock"]).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handoff = Handoff(config, api)
        while True:
            try:
                if handoff.tick():
                    break
            except SafetyError as error:
                handoff.update("NEEDS_ATTENTION", str(error))
                raise SystemExit(1) from error
            except Exception as error:
                handoff.update("RETRYING", f"{type(error).__name__}: {str(error)[:600]}; no ungated cancellation or duplicate launch")
            time.sleep(config.get("poll_seconds", 30))


if __name__ == "__main__":
    main()
