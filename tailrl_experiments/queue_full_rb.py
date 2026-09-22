"""Run sequential four-GPU RB arms after ER archival, until the holder ends."""

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from qwen3_experiments.checkpoint_handoff import TERMINAL_STATES, SafetyError, Slurm, atomic_json, validate_holder
from tailrl_experiments.run_text_maze_rb_full import SFT_STEPS, final_checkpoint_saved, is_complete, run_name


def now():
    return datetime.now(timezone.utc).isoformat()


def process_start(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] not in ("Z", "X") else None
    except (FileNotFoundError, TypeError, ValueError, ProcessLookupError):
        return None


class Queue:
    def __init__(self, plan_path, slurm=None):
        self.plan_path = Path(plan_path)
        self.config = json.loads(self.plan_path.read_text())
        self.path = self.plan_path.with_name("state.json")
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}
        identity = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        if self.state and self.state["config_sha256"] != identity:
            raise SafetyError("Queue state belongs to another experiment")
        self.state["config_sha256"] = identity
        self.order = self.config.get("checkpoint_order", [3000])
        if (
            not self.order
            or self.order[0] != 3000
            or self.order[1:] != sorted(self.order[1:])
            or len(set(self.order)) != len(self.order)
            or any(step not in SFT_STEPS for step in self.order)
        ):
            raise SafetyError("Expected SFT 3000 first, then distinct released checkpoints in ascending order")
        completed = self.state.setdefault("completed_checkpoints", [])
        if completed != self.order[: len(completed)]:
            raise SafetyError("Completed arms do not match the configured order")
        expected = self.order[len(completed)] if len(completed) < len(self.order) else None
        if self.state.setdefault("checkpoint_step", expected) != expected:
            raise SafetyError("Current arm does not follow the completed arms")
        self.slurm = slurm or Slurm()
        self.child = None
        self.holder_lock = None

    @property
    def run_dir(self):
        return Path(self.config["output_dir"]) / run_name(self.state["checkpoint_step"])

    def time_available(self):
        end = self.config.get("allocation_end_utc")
        return end is None or time.time() < datetime.fromisoformat(end).timestamp()

    def record(self, phase, message, **extra):
        changed = (phase, message) != (self.state.get("phase"), self.state.get("message"))
        self.state.update(phase=phase, message=message, updated_utc=now(), **extra)
        atomic_json(self.path, self.state)
        if changed:
            event = {k: self.state[k] for k in ("phase", "message", "updated_utc", "checkpoint_step")}
            with self.path.with_name("events.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)

    def ready(self):
        config = self.config
        if not Path(config["cpu_ready"]).is_file():
            return False
        source = json.loads(Path(config["prerequisite_state"]).read_text())
        receipt = source.get("receipt", {}).get("checkpoint", {})
        return (
            source.get("cancel_source_holder") is False
            and source.get("source_training_stopped") is True
            and source.get("source_holder_retained") is True
            and receipt.get("step") == 60
            and receipt.get("repo") == config["prerequisite_repo"]
        )

    def preflight(self):
        config = self.config
        if config["job_id"] != "3114675" or config["world_size"] != 4:
            raise SafetyError("Expected the retained four-GPU holder 3114675")
        for key in ("runner", "python", "prerequisite_state", "cpu_ready"):
            if not Path(config[key]).is_file():
                raise SafetyError(f"Missing queue dependency: {key}")
        job = self.slurm.job(config["job_id"])
        if job["JobState"] not in TERMINAL_STATES:
            validate_holder(job, config, config["job_id"])

    def progress(self):
        path = self.run_dir / "metrics.jsonl"
        if not path.exists():
            return 0
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 256 * 1024))
            lines = stream.read().decode(errors="replace").splitlines()
        steps = [self.state.get("completed_step", 0)]
        for line in lines:
            try:
                item = json.loads(line)
                if "training/global_step" in item.get("metrics", {}):
                    steps.append(int(item["step"]))
            except (ValueError, KeyError):
                continue
        return max(steps)

    def attempt_alive(self):
        alive = self.child is not None and self.child.poll() is None
        if self.child is None and self.state.get("controller_host") == socket.gethostname():
            start = self.state.get("srun_start")
            alive = start is not None and process_start(self.state.get("srun_pid")) == start
        return alive or bool(self.slurm.steps(self.config["job_id"]))

    def release_attempt(self):
        if self.holder_lock is not None:
            self.holder_lock.close()
            self.holder_lock = None
        self.child = None

    def finish(self):
        if self.state["completed_checkpoints"] != self.order:
            raise SafetyError("Cannot finish the queue before every configured arm completes")
        if self.config.get("cancel_holder_after_completion") is not True:
            self.record("COMPLETE", "All configured RB arms completed; holder remains allocated")
            return False
        final_step = self.order[-1]
        final_dir = Path(self.config["output_dir"]) / run_name(final_step)
        if not is_complete(final_dir, final_step) or not final_checkpoint_saved(final_dir):
            raise SafetyError("Final RB checkpoint must be saved before releasing the holder")
        config = self.config
        job = self.slurm.job(config["job_id"])
        if job["JobState"] in TERMINAL_STATES:
            self.record(
                "COMPLETE_HOLDER_RELEASED",
                "All configured RB arms completed and the holder is released",
                holder_state=job["JobState"],
                holder_released_utc=now(),
            )
            return False
        validate_holder(job, config, config["job_id"])
        if job["JobState"] == "COMPLETING":
            self.record("WAITING_HOLDER_RELEASE", "Holder cancellation is completing")
            return True
        lock_path = Path(config["holder_lock_dir"]) / f"gpu_holder_{config['job_id']}.launch.lock"
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.record("WAITING_HOLDER_IDLE", "RB queue completed; waiting for the shared holder lock")
                return True
            latest = self.slurm.job(config["job_id"])
            validate_holder(latest, config, config["job_id"])
            if latest["JobState"] != "RUNNING" or self.attempt_alive() or not self.slurm.idle(latest):
                self.record("WAITING_HOLDER_IDLE", "RB queue completed; waiting for active work and GPU cleanup")
                return True
            # All launchers sharing this holder honor the same lock. Recheck
            # ownership and Slurm steps immediately before cancelling the job.
            validate_holder(self.slurm.job(config["job_id"]), config, config["job_id"])
            if self.slurm.steps(config["job_id"]):
                return True
            self.record(
                "CANCELLING_HOLDER",
                "Final RB checkpoint saved and training stopped; cancelling the authorized holder",
                holder_cancel_requested=True,
            )
            self.slurm.cancel(config["job_id"])
            self.record("WAITING_HOLDER_RELEASE", "Holder cancellation submitted; waiting for Slurm confirmation")
            return True

    def advance(self):
        finished = self.state["checkpoint_step"]
        completed = [*self.state["completed_checkpoints"], finished]
        receipts = {
            **self.state.get("completion_receipts", {}),
            str(finished): {"path": str(self.run_dir / "COMPLETE.json"), "observed_utc": now()},
        }
        next_step = self.order[len(completed)] if len(completed) < len(self.order) else None
        cancel_holder = self.config.get("cancel_holder_after_completion") is True
        self.record(
            "READY_NEXT_ARM" if next_step is not None else "READY_TO_RELEASE_HOLDER" if cancel_holder else "COMPLETE",
            (
                f"SFT {finished} RB arm completed; next SFT {next_step}"
                if next_step is not None
                else (
                    "All configured RB arms completed; preparing to release the holder"
                    if cancel_holder
                    else "All configured RB arms completed; holder remains allocated"
                )
            ),
            completed_checkpoints=completed,
            completion_receipts=receipts,
            checkpoint_step=next_step,
            completed_step=0 if next_step is not None else 5001,
            attempt=0,
            launch_requested=False,
            srun_pid=None,
            srun_start=None,
            retry_after=0,
        )
        return next_step is not None or cancel_holder

    def watch(self):
        complete = is_complete(self.run_dir, self.state["checkpoint_step"])
        step = 5001 if complete else self.progress()
        if self.attempt_alive():
            phase = "WAITING_ARM_SHUTDOWN" if complete else "TRAINING" if step > 0 else "GPU_VALIDATION_OR_STARTUP"
            message = (
                "RB arm completed; waiting for its Slurm step to exit"
                if complete
                else "RB attempt active; monitoring progress"
            )
            self.record(phase, message, completed_step=step)
            return True
        self.release_attempt()
        if complete:
            return self.advance()
        self.record(
            "RETRYING",
            "Attempt ended before completion; retrying from its latest checkpoint",
            launch_requested=False,
            retry_after=time.time() + 300,
            completed_step=step,
        )
        return True

    def launch(self, job):
        config = self.config
        lock = (Path(config["holder_lock_dir"]) / f"gpu_holder_{config['job_id']}.launch.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return False
        try:
            latest = self.slurm.job(config["job_id"])
            validate_holder(latest, config, config["job_id"])
            if (
                latest["JobState"] != "RUNNING"
                or not self.time_available()
                or not self.ready()
                or not self.slurm.idle(latest)
            ):
                return False
            self.record(
                "LAUNCHING",
                f"All four GPUs idle; launching full RB from SFT {self.state['checkpoint_step']}",
                launch_requested=True,
                controller_host=socket.gethostname(),
                srun_pid=None,
                srun_start=None,
                attempt=self.state.get("attempt", 0) + 1,
            )
            env = os.environ.copy()
            for key in ("PYTHONPATH", "PYTHONHOME", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_STEP_ID"):
                env.pop(key, None)
            with self.path.with_name(f"slurm_ckpt_{self.state['checkpoint_step']}.log").open("ab") as stream:
                self.child = subprocess.Popen(
                    [
                        "srun",
                        f"--jobid={config['job_id']}",
                        "--overlap",
                        "--nodes=1",
                        "--ntasks=1",
                        "--cpus-per-task=288",
                        "--gpus-per-task=4",
                        "--network=no_vni",
                        f"--job-name=tailrl-rb-{self.state['checkpoint_step']}",
                        "--unbuffered",
                        f"--chdir={config['repo_root']}",
                        "/usr/bin/bash",
                        "-l",
                        config["runner"],
                        config["job_id"],
                        str(self.plan_path),
                        str(self.state["checkpoint_step"]),
                    ],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    pass_fds=(lock.fileno(),),
                )
            self.holder_lock = lock
            self.record(
                "GPU_VALIDATION_OR_STARTUP",
                "RB attempt active; monitoring progress",
                srun_pid=self.child.pid,
                srun_start=process_start(self.child.pid),
            )
            return True
        finally:
            if self.holder_lock is not lock:
                lock.close()

    def tick(self):
        if self.path.with_name("STOP").exists():
            self.record("STOPPED", "Queue stopped by request; running work is left untouched")
            return False
        if self.state["checkpoint_step"] is None:
            return self.finish()
        job = self.slurm.job(self.config["job_id"])
        if job["JobState"] in TERMINAL_STATES or not self.time_available():
            self.record(
                "ALLOCATION_ENDED",
                "Holder ended; saved checkpoints and remaining arms retained for later continuation",
                completed_step=self.progress(),
            )
            self.release_attempt()
            return False
        validate_holder(job, self.config, self.config["job_id"])
        if self.state.get("launch_requested") or is_complete(self.run_dir, self.state["checkpoint_step"]):
            return self.watch()
        if not self.ready():
            self.record("WAITING_ER_STEP60", "Waiting for ER checkpoint upload, verification, and training shutdown")
            return True
        if time.time() < self.state.get("retry_after", 0):
            return True
        if job["JobState"] == "RUNNING" and self.slurm.idle(job) and self.launch(job):
            return True
        self.record("WAITING_FOUR_IDLE_GPUS", "ER checkpoint verified; waiting for four idle GPUs and the holder lock")
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    queue = Queue(args.plan)
    queue.preflight()
    if args.check:
        print("RB queue preflight passed; no GPU work launched")
        return
    with args.plan.with_name("controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                if not queue.tick():
                    break
            except Exception as error:
                queue.record("RETRYING", f"{type(error).__name__}: {str(error)[:500]}")
            time.sleep(15)


if __name__ == "__main__":
    main()
