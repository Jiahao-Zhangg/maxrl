"""Adopt an orphaned ER launcher without restarting or changing its training."""

import argparse
import errno
import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

from qwen3_experiments import er_compression_compute as training
from qwen3_experiments.grpo_compute_control import digest, now, read, require_compute, write


def process_identity(pid, proc_root=Path("/proc")):
    folder = proc_root / str(pid)
    try:
        stat = (folder / "stat").read_text().rsplit(") ", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {"pid": int(pid), "uid": folder.stat().st_uid, "start_ticks": int(stat[19]),
                "command": (folder / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
                "cgroup": (folder / "cgroup").read_text().strip()}
    except FileNotFoundError:
        return None


def validate_launcher(plan, started, identity, own_cgroup):
    if identity is None:
        raise ValueError("The existing ER launcher is no longer running")
    expected = training.training_command(plan)
    if (identity["pid"] != started["pid"] or identity["uid"] != os.getuid()
            or identity["command"] != expected or started["command"] != expected
            or identity["cgroup"] != own_cgroup or "slurmstepd" not in own_cgroup
            or started["hostname"].split(".")[0] != plan["node"]):
        raise ValueError("Existing launcher identity/allocation does not match the frozen ER run")


def check_identity(expected, actual):
    if actual is not None and actual != expected:
        raise ValueError("The original launcher PID was reused or its identity changed")
    return actual is not None


def validate_completion(root, plan):
    train = Path(plan["train_dir"])
    for name in ("training_exit_status", "exit_status"):
        if (train / name).read_text().strip() != "0":
            raise ValueError("Training and archival must both finish successfully")
    manifest = read(root / "rollout_dataset/rollout_manifest.json")
    if (manifest.get("num_steps") != 100 or manifest.get("num_rollouts") != 25600
            or set(manifest["steps"]) != {str(i) for i in range(1, 101)}
            or any(row != {"file": f"data/step_{i:06d}.jsonl.gz", "num_rollouts": 256}
                   for i in range(1, 101) for row in [manifest["steps"][str(i)]])):
        raise ValueError("All 100 ER rollout steps must be saved")
    uploaded = read(root / "rollout_upload.json")
    if (uploaded.get("state") != "verified" or uploaded.get("num_rollouts") != 25600
            or uploaded.get("num_steps") != 100 or uploaded.get("repo_id") != plan["rollout_hf_repo"]
            or not re.fullmatch(r"[0-9a-f]{40}", uploaded.get("revision", ""))):
        raise ValueError("The complete training-rollout upload must be verified")
    for step in [*plan["checkpoint_steps"], None]:
        name = f"global_step{step}" if step is not None else "final_model"
        suffix = f"step_{step}" if step is not None else "final"
        receipt = read(root / "archive_receipts" / f"{name}.json")
        if (receipt.get("repo_id") != plan["hf_repo_prefix"] + "-" + suffix
                or receipt.get("path") != (f"global_step_{step}/actor" if step is not None else "")
                or not re.fullmatch(r"[0-9a-f]{40}", receipt.get("revision", ""))
                or not receipt.get("files") or receipt["files"].keys() != receipt.get("sha256", {}).keys()
                or any(size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"][name])
                       for name, size in receipt["files"].items())):
            raise ValueError("Wrong or incomplete checkpoint archive receipt")
        if step is None and (receipt.get("local_path") != "final_model"
                             or "config.json" not in receipt["files"]
                             or not any(name.endswith(".safetensors") for name in receipt["files"])):
            raise ValueError("The final ER model export is incomplete")
    return {"exit_code": 0, "completed_rollout_steps": 100, "optimizer_updates": 200,
            "saved_training_rollouts": 25600}


def durable_status(root, plan, value):
    # Keep the supervisor alive if the shared filesystem briefly reaches quota.
    local = Path(plan["scratch"]) / "recovered_supervisor_status.json"
    write(local, value)
    while True:
        try:
            write(root / "status.json", value)
            return
        except OSError as exc:
            if exc.errno not in (errno.EDQUOT, errno.ENOSPC):
                raise
            require_compute(plan["job_id"])
            time.sleep(30)


def watch(root, receipt_path):
    plan, recovery = training.verify_plan(root), read(receipt_path)
    require_compute(plan["job_id"])
    if (recovery["plan_sha256"] != digest(root / "plan.json")
            or recovery["started_sha256"] != digest(root / "training_started.json")
            or recovery["code_sha256"] != digest(Path(__file__))):
        raise ValueError("Recovery receipt or frozen training identity changed")
    state = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": plan["job_id"],
             "started_at": now(), "recovery_receipt": str(receipt_path),
             "launcher_pid": recovery["launcher"]["pid"], "training_restarted": False}

    def update(phase, **values):
        state.update(state=phase, updated_at=now(), **values)
        durable_status(root, plan, state)

    with ExitStack() as stack:
        for path in [root / "supervisor.lock", *map(Path, plan["holder_locks"])]:
            held = stack.enter_context(path.open("a"))
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                require_compute(plan["job_id"])
                alive = check_identity(recovery["launcher"], process_identity(recovery["launcher"]["pid"]))
                exit_path = Path(plan["train_dir"]) / "exit_status"
                if exit_path.exists() and exit_path.read_text().strip() != "0":
                    raise RuntimeError(f"Original ER training/archival failed: exit {exit_path.read_text().strip()}")
                if not alive:
                    if not exit_path.exists():
                        raise RuntimeError("Original launcher exited without recording its final exit status")
                    training.verify_plan(root)
                    update("complete", **validate_completion(root, plan), finished_at=now())
                    return
                manifest = read(root / "rollout_dataset/rollout_manifest.json")
                update("training_or_archiving", saved_rollout_steps=len(manifest["steps"]))
                time.sleep(15)
        except BaseException as exc:
            update("failed", error=str(exc), finished_at=now())
            raise


def launch(root):
    plan = training.verify_plan(root)
    require_compute(plan["job_id"])
    with ExitStack() as stack:
        for name in ("launch.lock", "supervisor.lock"):
            held = stack.enter_context((root / name).open("a"))
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous, status = read(root / "launch.json"), read(root / "status.json")
        if (status.get("state") != "training_or_archiving" or previous["pid"] != status.get("pid")
                or process_identity(previous["pid"]) is not None):
            raise ValueError("Only a stopped supervisor of an existing active run may be recovered")
        started = read(root / "training_started.json")
        identity = process_identity(started["pid"])
        validate_launcher(plan, started, identity, process_identity(os.getpid())["cgroup"])
        folder = root / "control_recovery" / str(time.time_ns())
        folder.mkdir(parents=True)
        code = folder / Path(__file__).name
        shutil.copy2(Path(__file__), code)
        receipt_path = folder / "recovery.json"
        write(receipt_path, {"plan_sha256": digest(root / "plan.json"), "code_sha256": digest(code),
                             "started_sha256": digest(root / "training_started.json"), "launcher": identity,
                             "previous_launch": previous, "previous_status": status, "created_at": now()})
    # Release the old supervisor lock before the adopter takes it. launch.lock
    # plus the new launch receipt prevent a second recovery from being accepted.
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(root / "launch.json") != previous:
            raise ValueError("Another controller recovered this run")
        with (Path(plan["scratch"]) / "recovered_supervisor.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([plan["python_bin"], "-u", str(code), "watch", "--training-root", str(root),
                                      "--receipt", str(receipt_path)], cwd=plan["runtime"],
                                     env=training.environment(plan), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {**previous, "pid": child.pid, "launched_at": now(), "recovery_receipt": str(receipt_path)}
        write(root / "launch.json", receipt)
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "watch"))
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    root = args.training_root.resolve()
    if args.command == "launch":
        launch(root)
    else:
        if args.receipt is None:
            parser.error("watch requires --receipt")
        watch(root, args.receipt)


if __name__ == "__main__":
    main()
