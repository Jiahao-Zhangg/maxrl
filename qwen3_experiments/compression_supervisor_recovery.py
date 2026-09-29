"""Recover supervision of an existing compression Slurm step without restarting training."""

import argparse
import errno
import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def controller(plan):
    sys.path.insert(0, plan["runtime"])
    module = importlib.import_module("qwen3_experiments.compression_l0_compute_control")
    if Path(module.__file__).resolve() != Path(plan["runtime"]) / "qwen3_experiments/compression_l0_compute_control.py":
        raise ValueError("Recovery must use the original frozen training controller")
    return module


def verify_inputs(root, plan, control):
    launch_path = root / "launch.json"
    if launch_path.exists():
        expected = read(launch_path).get("plan_sha256")
        if expected and expected != digest(root / "plan.json"):
            raise ValueError("Training plan changed after launch")
    control.verify_runtime(Path(plan["runtime"]))
    for path, checksum in plan["input_hashes"].items():
        if control.digest(path) != checksum:
            raise ValueError(f"Frozen training input changed: {path}")


def process_identity(pid):
    folder = Path("/proc") / str(pid)
    try:
        stat = (folder / "stat").read_text().rsplit(") ", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {"pid": int(pid), "uid": folder.stat().st_uid, "start_ticks": int(stat[19]),
                "command": (folder / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
                "cgroup": (folder / "cgroup").read_text().strip()}
    except FileNotFoundError:
        return None


def same_process(expected, actual):
    if actual is not None and actual != expected:
        raise ValueError("Original launcher PID was reused or its identity changed")
    return actual is not None


def last_completed_step(path):
    with Path(path).open("rb") as stream:
        stream.seek(max(0, Path(path).stat().st_size - 512 * 1024))
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stream.read().decode(errors="replace"))
    values = re.findall(r"(?<![\w/])training/global_step:(\d+)(?:\.0+)?(?=\s|$)", text)
    return max(map(int, values), default=0)


def slurm_step(job_id, step_id):
    identity = f"{job_id}.{step_id}"
    output = subprocess.check_output(
        ["sacct", "-j", identity, "--noheader", "--parsable2", "--format=JobIDRaw,State,ExitCode"],
        text=True, timeout=30,
    )
    rows = [line.strip().split("|") for line in output.splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == identity]
    if not rows:
        return None
    if len(rows) != 1 or len(rows[0]) < 3:
        raise ValueError("Ambiguous Slurm step accounting")
    return {"job_step": identity, "state": rows[0][1], "exit_code": rows[0][2]}


def successful_exit(record):
    if record is None or record["state"] in ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING"):
        return False
    if record["state"] != "COMPLETED" or record["exit_code"] != "0:0":
        raise RuntimeError(f"Original training step failed: {record}")
    return True


def matching_trainer(plan, step_id):
    output = subprocess.check_output(["scontrol", "listpids", f"{plan['job_id']}.{step_id}"], text=True, timeout=30)
    matches = []
    for line in output.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 3 or fields[1:3] != [str(plan["job_id"]), str(step_id)] or not fields[0].isdigit():
            continue
        identity = process_identity(int(fields[0]))
        if identity is None or "verl.trainer.main_ppo" not in identity["command"]:
            continue
        overrides = dict(token.split("=", 1) for token in identity["command"] if "=" in token)
        if overrides.get("trainer.default_local_dir", "").strip("'\"") == plan["checkpoint_dir"]:
            matches.append(identity)
    if len(matches) != 1:
        raise ValueError("Slurm step does not contain this run's unique active trainer")
    return matches[0]


def durable_write(control, plan, path, value):
    local = Path(plan["checkpoint_dir"]).parent / "control_recovery" / path.name
    control.write(local, value)
    while True:
        try:
            control.write(path, value)
            return
        except OSError as exc:
            if exc.errno not in (errno.EDQUOT, errno.ENOSPC):
                raise
            control.require_compute(plan["job_id"])
            time.sleep(30)


def archived_steps(root, plan):
    completed = []
    for step in plan["checkpoint_steps"]:
        path = root / "hf_checkpoint_archive/receipts" / f"global_step_{step}.json"
        if not path.exists():
            continue
        receipt = read(path)
        if receipt["repo_id"] != f"{plan['hf_repo_prefix']}-step_{step}" or receipt["checkpoint"] != f"global_step_{step}":
            raise ValueError("Checkpoint archive belongs to another run or step")
        if receipt["state"] == "archived_and_deleted":
            if not receipt.get("verified_at") or not re.fullmatch(r"[0-9a-f]{40}", receipt.get("remote_commit", "")):
                raise ValueError("Checkpoint archive is not verified")
            completed.append(step)
    return completed


def watch(root, receipt_path):
    plan, recovery = read(root / "plan.json"), read(receipt_path)
    control = controller(plan)
    control.require_compute(plan["job_id"])
    if (digest(root / "plan.json") != recovery["plan_sha256"]
            or digest(root / "training_started.json") != recovery["started_sha256"]
            or digest(Path(__file__)) != recovery["code_sha256"]):
        raise ValueError("Recovery inputs changed")
    verify_inputs(root, plan, control)
    status = {**recovery["previous_status"], "pid": os.getpid(), "hostname": socket.gethostname(),
              "recovered_at": control.now(), "training_restarted": False}
    state = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": plan["job_id"],
             "training_launcher_pid": recovery["launcher"]["pid"], "slurm_step_id": recovery["step_id"],
             "recovery_receipt": str(receipt_path), "training_restarted": False}
    monitor = None

    def update(phase, **values):
        status.update(state=phase, updated_at=control.now(), **values)
        state.update(state=phase, updated_at=control.now(), last_completed_step=status["last_completed_step"],
                     checkpoint_monitor_pid=monitor.pid if monitor else None)
        durable_write(control, plan, root / "status.json", status)
        durable_write(control, plan, root / "supervisor_status.json", state)

    def maintain_monitor():
        nonlocal monitor
        if monitor is None or monitor.poll() is not None:
            monitor = control.spawn(plan, "monitor", Path(plan["checkpoint_dir"]).parent / "control_recovery/upload.log")

    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with control.holder_locks(plan["holder_locks"]):
            try:
                update("training")
                while True:
                    control.require_compute(plan["job_id"])
                    maintain_monitor()
                    alive = same_process(recovery["launcher"], process_identity(recovery["launcher"]["pid"]))
                    status["last_completed_step"] = last_completed_step(root / "train.log")
                    if not alive:
                        accounting = slurm_step(plan["job_id"], recovery["step_id"])
                        if successful_exit(accounting):
                            final = Path(plan["checkpoint_dir"]) / "global_step_100"
                            if status["last_completed_step"] != 100 or not control.checkpoint_complete(final, final.parent):
                                raise RuntimeError("Successful Slurm exit is missing the completed final checkpoint")
                            durable_write(control, plan, root / "training_exit.json", {
                                "exit_code": 0, "launcher_exit_code": None, "slurm_accounting": accounting,
                                "exit_evidence": "Original Slurm training step completed with exit 0:0",
                                "finished_at": control.now(),
                            })
                            break
                    update("training" if alive else "waiting_for_training_exit")
                    time.sleep(15)
                update("verifying_rollout_upload")
                from huggingface_hub import HfApi

                durable_write(control, plan, root / "rollout_upload.json", control.audit_rollouts(plan, HfApi()))
                while True:
                    control.require_compute(plan["job_id"])
                    if monitor.poll() == 0 and archived_steps(root, plan) == plan["checkpoint_steps"]:
                        break
                    maintain_monitor()
                    update("waiting_for_checkpoint_archive")
                    time.sleep(15)
                update("complete", exit_code=0, finished_at=control.now())
            except BaseException as exc:
                update("failed", error=str(exc), finished_at=control.now())
                raise


def launch(root, step_id):
    plan = read(root / "plan.json")
    control = controller(plan)
    control.require_compute(plan["job_id"])
    verify_inputs(root, plan, control)
    with ExitStack() as stack:
        for name in ("launch.lock", "supervisor.lock", "hf_checkpoint_archive/monitor.lock"):
            held = stack.enter_context((root / name).open("a"))
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous, status, started = (read(root / name) for name in ("launch.json", "status.json", "training_started.json"))
        if status["state"] != "training" or process_identity(previous["pid"]) is not None:
            raise ValueError("Only an orphaned active training run may be adopted")
        identity = process_identity(started["pid"])
        own = process_identity(os.getpid())
        if (identity is None or identity["command"] != control.training_command(plan)
                or identity["uid"] != os.getuid() or identity["cgroup"] != own["cgroup"]
                or "slurmstepd" not in own["cgroup"] or started["hostname"].split(".")[0] != plan["node"]):
            raise ValueError("Existing launcher does not match the frozen training run and allocation")
        trainer = matching_trainer(plan, step_id)
        accounting = slurm_step(plan["job_id"], step_id)
        if accounting is None or accounting["state"] != "RUNNING":
            raise ValueError("Expected the original Slurm training step to be running")
        folder = root / "control_recovery" / str(time.time_ns())
        folder.mkdir(parents=True)
        code = folder / Path(__file__).name
        shutil.copy2(Path(__file__), code)
        receipt_path = folder / "recovery.json"
        control.write(receipt_path, {
            "plan_sha256": digest(root / "plan.json"), "started_sha256": digest(root / "training_started.json"),
            "code_sha256": digest(code), "launcher": identity, "trainer": trainer, "step_id": step_id,
            "previous_launch": previous, "previous_status": status, "created_at": control.now(),
        })
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(root / "launch.json") != previous:
            raise ValueError("Another controller already recovered this run")
        local = Path(plan["checkpoint_dir"]).parent / "control_recovery"
        local.mkdir(parents=True, exist_ok=True)
        with (local / "supervisor.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([plan["python_bin"], "-u", str(code), "watch", "--training-root", str(root),
                                      "--receipt", str(receipt_path)], cwd=plan["runtime"], env=control.environment(plan),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {**previous, "pid": child.pid, "plan_sha256": digest(root / "plan.json"),
                   "recovery_receipt": str(receipt_path), "launched_at": control.now()}
        control.write(root / "launch.json", receipt)
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "watch"))
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--step-id", type=int)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.command == "launch":
        if args.step_id is None:
            parser.error("launch requires --step-id")
        launch(args.training_root.resolve(), args.step_id)
    else:
        if args.receipt is None:
            parser.error("watch requires --receipt")
        watch(args.training_root.resolve(), args.receipt)


if __name__ == "__main__":
    main()
