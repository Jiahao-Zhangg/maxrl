"""Watch one compute allocation and recover its existing, frozen control queues.

This process never signals a process, starts a trainer, or touches another run's
cache. Its own records live outside the project filesystem it is monitoring.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import errno
import fcntl
import gzip
import hashlib
import importlib
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

SPACE_ERRORS = (errno.EDQUOT, errno.ENOSPC)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def identity(pid):
    folder = Path("/proc") / str(pid)
    try:
        fields = (folder / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "uid": folder.stat().st_uid, "start_ticks": int(fields[19]),
                "command": (folder / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
                "cgroup": (folder / "cgroup").read_text().strip()}
    except FileNotFoundError:
        return None


def require_node(config):
    if socket.gethostname().split(".")[0] != config["node"]:
        raise ValueError("Watchdog is restricted to its original compute node")
    text = subprocess.check_output(["scontrol", "show", "job", config["job_id"], "-o"], text=True, timeout=30)
    fields = dict(word.split("=", 1) for word in text.split() if "=" in word)
    if (fields.get("JobState") != "RUNNING" or fields.get("BatchHost") != config["node"]
            or fields.get("NumNodes") != "1" or not fields.get("UserId", "").endswith(f"({os.getuid()})")):
        raise ValueError("Original allocation is no longer available on the expected node")
    members = allocation_processes(config)
    if os.getpid() not in members:
        raise ValueError("Watchdog must belong to its original Slurm allocation")


def allocation_processes(config):
    text = subprocess.check_output(["scontrol", "listpids", config["job_id"]], text=True, timeout=30)
    return {int(parts[0]) for line in text.splitlines()[1:]
            if len(parts := line.split()) >= 3 and parts[0].isdigit() and parts[1] == config["job_id"]}


def verify_config(config):
    root = Path(config["training_root"])
    plan = read(root / "plan.json")
    if plan["job_id"] != config["job_id"] or plan["node"] != config["node"]:
        raise ValueError("Watchdog targets another training allocation")
    for path, checksum in config["pinned_files"].items():
        if digest(path) != checksum:
            raise ValueError(f"Watchdog input changed: {path}")
    for stage in config["stages"].values():
        Path(stage["root"]).relative_to(root)
    if Path(config["control_root"]).is_relative_to(root):
        raise ValueError("Watchdog records must be outside the training output directory")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def control_locations(config):
    paths = [Path(config["control_root"])]
    if config.get("local_control_root"):
        paths.insert(0, Path(config["local_control_root"]))
    return paths


def control_record(config, relative, value):
    """Independent local and home copies; a full status disk cannot stop control."""
    saved = []
    for root in control_locations(config):
        try:
            write(root / relative, value)
            saved.append(str(root / relative))
        except OSError as exc:
            if exc.errno not in SPACE_ERRORS:
                raise
    return saved


def safe_print(message):
    try:
        print(message, flush=True)
    except OSError as exc:
        if exc.errno not in SPACE_ERRORS:
            raise


def routed_path(config, path):
    path = Path(path)
    destination = config.get("control_file_routes", {}).get(str(path))
    if destination is None:
        return path
    target = Path(destination)
    if (path.parent != Path(config["training_root"])
            or path.name not in ("status.json", "supervisor_status.json", "training_exit.json", "rollout_upload.json", "launch.json")
            or not target.is_relative_to(Path(config["independent_storage_root"]) / "control")
            or not path.is_symlink() or path.resolve() != target.resolve()):
        raise ValueError("Control storage route changed or escaped its allowed directory")
    return target


def project_write(config, path, value):
    write(routed_path(config, path), value)


def durable_writer(original, config):
    """Keep a queue alive on quota errors without advancing past a failed write."""
    root = Path(config["training_root"])

    def retry(path, value):
        relative = Path(path).relative_to(root)
        control_record(config, Path("write_mirrors") / relative, value)
        blocked = False
        while True:
            try:
                original(path, value)
                if blocked:
                    control_record(config, f"blocked_writes/{os.getpid()}.json", {
                        "pid": os.getpid(), "state": "recovered", "path": str(path), "updated_at": now()})
                    safe_print(f"{now()} storage write recovered: {relative}")
                return
            except OSError as exc:
                if exc.errno not in SPACE_ERRORS:
                    raise
                if not blocked:
                    safe_print(f"{now()} waiting for project space: {relative}")
                    blocked = True
                control_record(config, f"blocked_writes/{os.getpid()}.json", {
                    "pid": os.getpid(), "state": "waiting_for_disk", "path": str(path), "updated_at": now()})
                time.sleep(config["interval_seconds"])
                require_node(config)

    return retry


def archive_receipt_ready(plan, checkpoint, directory, world_size=8):
    """A verified archive proves a completed save even after its local removal."""
    checkpoint, directory = Path(checkpoint), Path(directory)
    match = re.fullmatch(r"global_step_(\d+)", checkpoint.name)
    if (not match or int(match[1]) not in plan["checkpoint_steps"] or checkpoint.is_symlink()
            or directory.resolve() != Path(plan["checkpoint_dir"]).resolve()
            or checkpoint.parent.resolve() != directory.resolve()):
        return False
    step = int(match[1])
    try:
        if int((directory / "latest_checkpointed_iteration.txt").read_text().strip()) < step:
            return False
        receipt = read(Path(plan["output_root"]) / "hf_checkpoint_archive/receipts" / f"global_step_{step}.json")
        if (receipt["state"] not in ("verified", "archived_and_deleted")
                or receipt["checkpoint"] != checkpoint.name
                or receipt["checkpoint_path"] != str(checkpoint.resolve())
                or receipt["repo_id"] != f"{plan['hf_repo_prefix']}-step_{step}"
                or not re.fullmatch(r"[0-9a-f]{40}", receipt["remote_commit"])
                or not receipt["verified_at"]):
            return False
        files = receipt["files"]
        required = {"data.pt", "actor/config.json", "actor/tokenizer_config.json"}
        required.update(f"actor/{kind}_world_size_{world_size}_rank_{rank}.pt"
                        for kind in ("model", "optim", "extra_state") for rank in range(world_size))
        return (required.issubset(files) and receipt["file_count"] == len(files)
                and receipt["total_bytes"] == sum(item["size"] for item in files.values())
                and all(item["size"] > 0 and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                        for name, item in files.items() if name in required))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def eager_checkpoint_monitor(plan, config, implementation):
    """Archive complete saves immediately, including the final save before exit."""
    from huggingface_hub import HfApi

    poll = config["checkpoint_poll_seconds"]
    if not 0 < poll <= 30:
        raise ValueError("Checkpoint polling interval must be within (0, 30] seconds")
    implementation.require_compute(plan["job_id"])
    root, directory = Path(plan["output_root"]), Path(plan["checkpoint_dir"])
    folder = root / "hf_checkpoint_archive"
    receipts = folder / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    api, verifier = HfApi(), implementation.load_verifier(plan)
    verifier.write_receipt = durable_writer(verifier.write_receipt, config)
    steps = plan["checkpoint_steps"]
    with (folder / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_heartbeat = 0
        while True:
            state = {"pid": os.getpid(), "hostname": socket.gethostname(), "state": "monitoring",
                     "updated_at": now(), "checkpoint_poll_seconds": poll, "eager_final_upload": True}
            errors, changed = [], False
            for step in steps:
                checkpoint = directory / f"global_step_{step}"
                receipt_path = receipts / f"global_step_{step}.json"
                if not checkpoint.exists():
                    # Recover a crash after removal but before marking the receipt.
                    if (receipt_path.exists() and read(receipt_path).get("state") == "verified"
                            and archive_receipt_ready(plan, checkpoint, directory)):
                        implementation.write(receipt_path, {**read(receipt_path), "state": "archived_and_deleted",
                                                            "deletion_reconciled_at": now()})
                        changed = True
                    continue
                if not implementation.checkpoint_complete(checkpoint, directory):
                    continue
                try:
                    implementation.require_compute(plan["job_id"])
                    state.update(state="uploading", step=step, updated_at=now())
                    implementation.write(folder / "status.json", state)
                    implementation.archive_checkpoint(checkpoint, receipt_path,
                                                      f"{plan['hf_repo_prefix']}-step_{step}", api, verifier)
                    changed = True
                except Exception as exc:
                    errors.append({"step": step, "error": str(exc)})
                    safe_print(f"Retaining checkpoint {step} for retry: {exc}")
                    break
            archived = [step for step in steps if (receipts / f"global_step_{step}.json").exists()
                        and read(receipts / f"global_step_{step}.json").get("state") == "archived_and_deleted"]
            state.update(state="retrying" if errors else "monitoring", archived_steps=archived,
                         errors=errors, updated_at=now())
            if (root / "training_exit.json").exists() and not errors:
                remaining = [step for step in steps if (directory / f"global_step_{step}").exists()]
                if not remaining:
                    complete = read(root / "training_exit.json")["exit_code"] == 0 and archived == steps
                    state.update(state="complete" if complete else "training_failed_or_missing_checkpoint")
                    implementation.write(folder / "status.json", state)
                    return 0 if complete else 1
                if any(not implementation.checkpoint_complete(directory / f"global_step_{step}", directory)
                       for step in remaining):
                    state.update(state="incomplete_checkpoint_retained", remaining_steps=remaining)
                    implementation.write(folder / "status.json", state)
                    return 1
            if changed or errors or time.monotonic() - last_heartbeat >= config["interval_seconds"]:
                implementation.require_compute(plan["job_id"])
                implementation.write(folder / "status.json", state)
                last_heartbeat = time.monotonic()
            time.sleep(config["interval_seconds"] if errors else poll)


def invoke(config, kind, config_path):
    verify_config(config)
    require_node(config)
    stage = config["stages"]["supervisor" if kind == "monitor" else kind]
    root = Path(stage["root"])
    if kind == "supervisor":
        receipt = Path(config["recovery_receipt"])
        module = load("watchdog_original_recovery", receipt.parent / "compression_supervisor_recovery.py")
        original_controller = module.controller

        def controller(plan):
            control = original_controller(plan)
            original_spawn = control.spawn
            original_write = control.write
            control.write = lambda path, value: original_write(routed_path(config, path), value)
            if config.get("eager_checkpoint_upload"):
                original_complete = control.checkpoint_complete

                def checkpoint_complete(checkpoint, directory, world_size=8):
                    # The original supervisor still requires the real Slurm exit
                    # and final training step; only its local-file proof changes.
                    try:
                        local_complete = original_complete(checkpoint, directory, world_size)
                    except FileNotFoundError:  # A verified upload is being removed.
                        local_complete = False
                    return local_complete or (
                        Path(checkpoint).name == "global_step_100"
                        and archive_receipt_ready(plan, checkpoint, directory, world_size))

                control.checkpoint_complete = checkpoint_complete

            def spawn(plan, command, log_path):
                if command != "monitor":
                    return original_spawn(plan, command, log_path)
                with Path(log_path).open("ab", buffering=0) as log:
                    return subprocess.Popen([
                        plan["python_bin"], "-u", config["code"], "queue", "--config", str(config_path),
                        "--kind", "monitor", "--output-root", str(root)], cwd=plan["runtime"],
                        env=control.environment(plan), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)

            control.spawn = spawn
            return control

        module.controller = controller
        module.watch(root, receipt)
    elif kind == "monitor":
        plan = read(root / "plan.json")
        sys.path.insert(0, plan["runtime"])
        control = importlib.import_module("qwen3_experiments.compression_l0_compute_control")
        control.verify_runtime(Path(plan["runtime"]))
        implementation = sys.modules[control.monitor.__module__]
        implementation.write = durable_writer(implementation.write, config)
        if config.get("eager_checkpoint_upload"):
            return eager_checkpoint_monitor(plan, config, implementation)
        return control.monitor(plan)
    elif kind == "nine":
        module = load("watchdog_nine", root / "provenance/eval_l0_final.py")
        base = module.evaluator(root)
        base.write = durable_writer(base.write, config)
        module.queue(root, base)
    elif kind == "budget":
        plan = read(root / "plan.json")
        sys.path.insert(0, plan["runtime"])
        module = load("watchdog_budget", Path(plan["runtime"]) / "qwen3_experiments/minerva_individual_budget.py")
        module.write = durable_writer(module.write, config)
        module.queue(root, module.verify_plan(root))


def stage_environment(config, kind):
    stage = config["stages"][kind]
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("SLURM_", "RAY_", "VLLM_", "CUDA_", "GRPO_", "L0_", "MAXRL_")):
            env.pop(key)
    env.pop("PYTHONHOME", None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               HF_HOME=config["hf_home"], PYTHONPATH=stage["runtime"],
               PATH=str(Path(config["python_bin"]).parent) + os.pathsep + env.get("PATH", ""))
    return env


def active_stage_work(config, stage):
    needles = {stage["root"], stage["runtime"]}
    needles.update(str(Path(path).resolve()) for path in tuple(needles))
    for pid in allocation_processes(config):
        item = identity(pid)
        if item is None or pid == os.getpid():
            continue
        if any(arg == needle or arg.startswith(needle + "/") for arg in item["command"] for needle in needles):
            return True
    return False


def restart(config, kind, config_path):
    """Recover only a dead controller; never overlap an existing launcher/worker."""
    verify_config(config)
    require_node(config)
    stage = config["stages"][kind]
    root = Path(stage["root"])
    receipt_path = root / stage["receipt"]
    with ExitStack() as stack:
        for name in (stage["launch_lock"], stage["process_lock"]):
            held = stack.enter_context((root / name).open("a"))
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = read(receipt_path)
        if identity(receipt["pid"]) is not None:
            return None
        state = read(root / stage["status"])
        if state.get("state") in ("complete", "superseded"):
            return None
        if kind != "supervisor":
            if state.get("state") not in stage["restart_states"]:
                raise ValueError("Queue did not stop while waiting; its work needs inspection")
            if identity(state.get("launcher_pid", -1)) is not None or active_stage_work(config, stage):
                raise ValueError("Existing evaluation work is still active; refusing a duplicate launch")
        else:
            if state.get("state") in ("failed", "training_failed"):
                raise ValueError("Training failure is not a control-only recovery")
            launcher = identity(config["training_launcher"]["pid"])
            if launcher is not None and launcher != config["training_launcher"]:
                raise ValueError("Original training launcher identity changed")
        attempt = Path(config["control_root"]) / "restarts" / f"{time.time_ns()}_{kind}"
        attempt.mkdir(parents=True)
        write(attempt / "previous_launch.json", receipt)
        write(attempt / "previous_status.json", state)
        # Persist this before spawning, so a full filesystem cannot leave an
        # unrecorded child. A prepared receipt retains the old, dead PID.
        project_write(config, receipt_path, {**receipt, "pending_recovery": str(attempt)})
        fcntl.flock(held, fcntl.LOCK_UN)  # The child acquires the process lock.
        command = [config["python_bin"], "-u", config["code"], "queue", "--config", str(config_path),
                   "--kind", kind, "--output-root", str(root)]
        with (attempt / "control.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=stage["runtime"], env=stage_environment(config, kind),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        restored = {**receipt, "pid": child.pid, "launched_at": now(), "watchdog_recovery": str(attempt)}
        write(attempt / "launch.json", restored)
        # Also keep the authoritative child identity off the project disk.
        write(Path(config["control_root"]) / f"{kind}_launch.json", restored)
        try:
            project_write(config, receipt_path, restored)
        except OSError as exc:
            if exc.errno not in SPACE_ERRORS:
                raise
        return restored


def relocate_closed_rollout(config, completed_step):
    """Move at most one immutable shard from this run, retaining its exact path."""
    root = Path(config["training_root"])
    source_dir = root / "rollout_dataset"
    manifest = read(source_dir / "rollout_manifest.json")
    for step, info in sorted(manifest["steps"].items(), key=lambda pair: int(pair[0])):
        if int(step) >= completed_step or info["file"] != f"data/step_{int(step):06d}.jsonl.gz":
            continue
        source = source_dir / info["file"]
        if source.is_symlink() or not source.is_file() or source.resolve().parent != source_dir.resolve() / "data":
            continue
        before = source.stat()
        if time.time() - before.st_mtime < 120:
            continue
        with gzip.open(source, "rt") as stream:
            count = 0
            for line in stream:
                if json.loads(line)["step"] != int(step):
                    raise ValueError("Rollout shard contains another step")
                count += 1
        if count != info["num_rollouts"]:
            raise ValueError("Completed rollout shard does not match its manifest")
        destination = Path(config["control_root"]) / "relocated_rollouts" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        checksum = digest(source)
        if destination.exists():
            if digest(destination) != checksum:
                raise ValueError("Relocation destination has different contents")
        else:
            temporary = destination.with_suffix(".copying")
            shutil.copy2(source, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            if digest(temporary) != checksum:
                raise ValueError("Relocated rollout checksum differs")
            temporary.replace(destination)
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("Rollout changed during relocation")
        link = source.with_name(source.name + ".watchdog_link")
        if link.is_symlink():
            if link.readlink() != destination:
                raise ValueError("Unexpected pending relocation link")
            link.unlink()
        link.symlink_to(destination)
        link.replace(source)
        result = {"source": str(source), "destination": str(destination), "sha256": checksum,
                  "bytes": before.st_size, "relocated_at": now()}
        write(destination.with_suffix(".receipt.json"), result)
        return result
    return None


def disk(path):
    info = os.statvfs(path)
    return {"available_bytes": info.f_bavail * info.f_frsize, "total_bytes": info.f_blocks * info.f_frsize,
            "available_inodes": info.f_favail}


def stage_status(config, kind):
    stage = config["stages"][kind]
    root = Path(stage["root"])
    state = read(root / stage["status"])
    receipt = read(root / stage["receipt"])
    local = Path(config["control_root"]) / f"{kind}_launch.json"
    if local.exists() and identity(read(local)["pid"]) is not None:
        receipt = read(local)
    process = identity(receipt["pid"])
    if process is not None:
        if process["uid"] != os.getuid() or process["cgroup"] != config["control_cgroup"]:
            raise ValueError("Controller PID does not belong to this allocation")
        if not any(arg in (stage["root"], str(Path(stage["root"]).resolve())) for arg in process["command"]):
            raise ValueError("Controller PID does not belong to this run")
    stamp = state.get("updated_at")
    timestamp = stamp if isinstance(stamp, (int, float)) else datetime.fromisoformat(stamp).timestamp()
    return {"pid": receipt["pid"], "alive": process is not None, "state": state.get("state"),
            "heartbeat_age_seconds": round(time.time() - timestamp, 1)}


def all_complete(config, states):
    if not all(state["state"] == "complete" for state in states.values()):
        return False
    nine = read(Path(config["stages"]["nine"]["root"]) / "report/audit.json")
    budget = read(Path(config["stages"]["budget"]["root"]) / "report/audit.json")
    return nine["complete"] and nine["responses_verified"] == 7276 and budget["complete"] and budget["points"] == 35


def sync_launch(config, kind):
    """Publish a recovery receipt once project writes are possible again."""
    local = Path(config["control_root"]) / f"{kind}_launch.json"
    if not local.exists():
        return
    recovered = read(local)
    if identity(recovered["pid"]) is None:
        return
    stage = config["stages"][kind]
    root = Path(stage["root"])
    with (root / stage["launch_lock"]).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = read(root / stage["receipt"])
        if current == recovered:
            return
        if current["pid"] != recovered["pid"] and identity(current["pid"]) is not None:
            raise ValueError("Another live controller owns the launch receipt")
        project_write(config, root / stage["receipt"], recovered)


def watch(config, config_path):
    verify_config(config)
    require_node(config)
    control, root = Path(config["control_root"]), Path(config["training_root"])
    children, previous, retries = {}, None, {}
    with (control / "watchdog.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            snapshot = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": config["job_id"],
                        "updated_at": now(), "state": "monitoring", "issues": []}
            try:
                require_node(config)
                snapshot["disks"] = {name: disk(path) for name, path in config["filesystems"].items()}
                free = snapshot["disks"]["project"]["available_bytes"]
                snapshot["disk_pressure"] = "critical" if free < config["critical_bytes"] else (
                    "low" if free < config["warning_bytes"] else "normal")
                training = read(root / "status.json")
                snapshot["training_step"] = training["last_completed_step"]
                if free < config["critical_bytes"]:
                    try:
                        relocated = relocate_closed_rollout(config, training["last_completed_step"])
                        if relocated:
                            snapshot["relocated_rollout"] = relocated
                    except OSError as exc:
                        snapshot["issues"].append(f"Rollout relocation deferred: {exc}")
                states = snapshot["controls"] = {}
                for kind in config["stages"]:
                    try:
                        sync_launch(config, kind)
                        states[kind] = stage_status(config, kind)
                        if states[kind]["state"] in ("complete", "superseded"):
                            continue
                        if states[kind]["alive"]:
                            if states[kind]["heartbeat_age_seconds"] > 120:
                                snapshot["issues"].append(f"{kind}: heartbeat is stale; live process retained")
                            continue
                        if time.time() < retries.get(kind, 0):
                            continue
                        receipt = restart(config, kind, config_path)
                        retries[kind] = time.time() + 120
                        if receipt:
                            children[kind] = receipt["pid"]
                            snapshot.setdefault("recovered", {})[kind] = receipt
                            states[kind] = stage_status(config, kind)
                    except Exception as exc:
                        retries[kind] = time.time() + 120
                        snapshot["issues"].append(f"{kind}: {exc}")
                monitor = read(root / "hf_checkpoint_archive/status.json")
                snapshot["checkpoint_monitor"] = {"pid": monitor["pid"], "alive": identity(monitor["pid"]) is not None,
                                                  "state": monitor["state"], "errors": monitor.get("errors", [])}
                if (not snapshot["checkpoint_monitor"]["alive"] and training["state"] != "complete"
                        and time.time() - datetime.fromisoformat(monitor["updated_at"]).timestamp() > 90):
                    snapshot["issues"].append("Checkpoint monitor exited; existing supervisor owns its restart")
                if len(states) == len(config["stages"]) and all_complete(config, states):
                    snapshot["state"] = "complete"
            except Exception as exc:
                snapshot["state"] = "needs_attention"
                snapshot["issues"].append(str(exc))
            # Reap only children this watchdog launched, without signalling any process.
            for kind, pid in list(children.items()):
                try:
                    exited, _ = os.waitpid(pid, os.WNOHANG)
                    if exited:
                        children.pop(kind)
                except ChildProcessError:
                    children.pop(kind)
            signature = json.dumps({"state": snapshot["state"], "pressure": snapshot.get("disk_pressure"),
                                    "issues": snapshot["issues"], "recovered": snapshot.get("recovered", {}),
                                    "stages": {k: {"alive": v["alive"], "state": v["state"]}
                                               for k, v in snapshot.get("controls", {}).items()}}, sort_keys=True)
            control_record(config, "status.json", snapshot)
            if signature != previous:
                for destination in control_locations(config):
                    try:
                        with (destination / "events.jsonl").open("a") as stream:
                            stream.write(json.dumps(snapshot) + "\n")
                    except OSError as exc:
                        if exc.errno not in SPACE_ERRORS:
                            raise
                previous = signature
            if snapshot["state"] == "complete":
                return
            time.sleep(config["interval_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("watch", "queue"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=("supervisor", "monitor", "nine", "budget"))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = read(args.config)
    if args.command == "watch":
        watch(config, args.config)
    else:
        kind = "supervisor" if args.kind == "monitor" else args.kind
        if kind is None or str(args.output_root) != config["stages"][kind]["root"]:
            parser.error("Control invocation must match its pinned stage")
        return invoke(config, args.kind, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
