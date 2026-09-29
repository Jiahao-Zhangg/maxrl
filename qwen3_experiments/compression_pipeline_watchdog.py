"""Keep a frozen evaluation/training pipeline alive through project disk pressure.

The watchdog and recovery logs live on the allocated compute node. A configuration
explicitly names writable outputs and protected inputs. Recovery may replace a
stale controller or cancel a verified idle evaluation step, never the allocation
or an active trainer. Completed results and checkpoint identities are retained.
"""

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
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
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
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


def allocation_processes(config):
    output = subprocess.check_output(["scontrol", "listpids", config["job_id"]], text=True, timeout=30)
    return {int(parts[0]): parts[2] for line in output.splitlines()[1:]
            if len(parts := line.split()) >= 3 and parts[0].isdigit() and parts[1] == config["job_id"]}


def require_node(config):
    if socket.gethostname().split(".")[0] != config["node"]:
        raise ValueError("Wrong compute node")
    output = subprocess.check_output(["scontrol", "show", "job", config["job_id"], "-o"], text=True, timeout=30)
    fields = dict(word.split("=", 1) for word in output.split() if "=" in word)
    if (fields.get("JobState") != "RUNNING" or fields.get("BatchHost") != config["node"]
            or fields.get("NumNodes") != "1" or not fields.get("UserId", "").endswith(f"({os.getuid()})")):
        raise ValueError("Original allocation is no longer available")
    if os.getpid() not in allocation_processes(config):
        raise ValueError("Watchdog is outside its Slurm allocation")


def overlaps(first, second):
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def owned_path(config, path):
    """Check both the logical path and its resolved target against the allowlist."""
    path = Path(path).absolute()
    allowed = [Path(s["root"]) for s in config["stages"].values()] + [Path(config["control_root"])]
    for candidate in (path, path.resolve()):
        if not any(candidate.is_relative_to(root) for root in allowed):
            raise ValueError(f"Outside this pipeline's writable outputs: {path}")
        if any(overlaps(candidate, Path(p).resolve()) for p in config["protected_paths"]):
            raise ValueError(f"Protected input or other allocation: {path}")
    return path


def verify(config):
    for path, checksum in config["pinned_files"].items():
        if digest(path) != checksum:
            raise ValueError(f"Frozen watchdog input changed: {path}")
    for stage in config["stages"].values():
        root = Path(stage["root"])
        plan = read(root / "plan.json")
        if plan["job_id"] != config["job_id"] or plan["node"] != config["node"]:
            raise ValueError("Stage belongs to another allocation")
        if digest(root / "plan.json") != stage["plan_sha256"]:
            raise ValueError("Stage plan changed")
    for route in config["routes"]:
        owned_path(config, route["source"])
        owned_path(config, route["local"])
        owned_path(config, route["backup"])


def durable_writer(original, config):
    def retry(path, value):
        path = owned_path(config, path)
        key = hashlib.sha256(str(path).encode()).hexdigest()
        write(Path(config["control_root"]) / "write_mirrors" / f"{key}.json", {"path": str(path), "value": value})
        blocked = Path(config["control_root"]) / "blocked_writes" / f"{key}.json"
        while True:
            try:
                original(path, value)
                blocked.unlink(missing_ok=True)
                return
            except OSError as exc:
                if exc.errno not in SPACE_ERRORS:
                    raise
                write(blocked, {"path": str(path), "pid": os.getpid(), "errno": exc.errno, "updated_at": now()})
                # A failed NFS temporary can keep returning EDQUOT after space
                # has returned. Remove only this writer's unpublished inode.
                for name in (f"{path.name}.{os.getpid()}.tmp", f".{path.name}.{os.getpid()}.tmp"):
                    temporary = path.with_name(name)
                    if temporary.is_file() and not temporary.is_symlink() and temporary.stat().st_uid == os.getuid():
                        temporary.unlink()
                time.sleep(config["interval_seconds"])
                require_node(config)
    return retry


@contextmanager
def locks(paths):
    with ExitStack() as stack:
        for path in paths:
            stream = stack.enter_context(Path(path).open("a"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def disk(path):
    info = os.statvfs(path)
    return {"available_bytes": info.f_bavail * info.f_frsize, "available_inodes": info.f_favail}


def copy_checked(source, destination):
    """Publish a checked copy only if the original remained unchanged."""
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".watchdog_copy")
    try:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        checksum = digest(temporary)
        after = source.stat()
        if ((before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_ino, after.st_size, after.st_mtime_ns) or checksum != digest(source)):
            raise ValueError("Output changed while copying")
        temporary.replace(destination)
        return {"sha256": checksum, "size": before.st_size, "mtime_ns": before.st_mtime_ns}
    finally:
        temporary.unlink(missing_ok=True)


def redirect_closed_logs(config, kind):
    """Called only with dead workers and the stage locks, to replace stale NFS inodes."""
    local = Path(config["control_root"])
    for name in config["stages"][kind].get("recovery_logs", []):
        source = owned_path(config, name)
        destination = local / "recovered_logs" / kind / source.relative_to(config["stages"][kind]["root"])
        if source.is_symlink():
            if source.readlink() != destination:
                # Preconfigured local storage is already safe.
                if not any(source == Path(r["source"]) and source.readlink() == Path(r["local"]) for r in config["routes"]):
                    raise ValueError("Unexpected log link")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            if source.stat().st_uid != os.getuid() or not source.is_file():
                raise ValueError("Log has unexpected type or owner")
            copy_checked(source, destination)
        else:
            destination.touch()
        link = source.with_name(source.name + ".watchdog_link")
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.readlink() != destination:
                raise ValueError("Unexpected pending log link")
            link.unlink()
        link.symlink_to(destination)
        link.replace(source)


def prepare_empty_routes(config):
    """Only newly prepared, unused output paths can be routed automatically."""
    for route in config["routes"]:
        source, local = (owned_path(config, route[k]) for k in ("source", "local"))
        if source.is_symlink():
            if source.readlink() != local:
                raise ValueError("Output link has an unexpected target")
            continue
        if source.exists():
            raise ValueError(f"Refusing to move an existing live output: {source}")
        source.parent.mkdir(parents=True, exist_ok=True)
        local.parent.mkdir(parents=True, exist_ok=True)
        if route["directory"]:
            local.mkdir(exist_ok=True)
        else:
            local.touch(exist_ok=True)
        source.symlink_to(local, target_is_directory=route["directory"])


def backups(config, pressure):
    """Reclaim only our duplicate backups; never remove canonical results."""
    state_file = Path(config["control_root"]) / "backups.json"
    state = read(state_file) if state_file.exists() else {}
    copied = reclaimed = 0
    skipped_symlinks = []
    for route in config["routes"]:
        source, local, backup = (owned_path(config, route[k]) for k in ("source", "local", "backup"))
        if not source.is_symlink() or source.readlink() != local:
            raise ValueError("Storage route changed")
        files = sorted(local.rglob("*")) if route["directory"] else [local]
        for path in files:
            if path.is_symlink():
                if route.get("skip_symlinks", False):
                    # Loggers create aliases and links into their global cache.
                    # Back up regular files without following those links.
                    skipped_symlinks.append(str(path))
                    continue
                raise ValueError("Unexpected link inside node-local outputs")
            if not path.is_file() or any(part.startswith(".") for part in path.relative_to(local.parent).parts):
                continue
            if path.name.endswith((".tmp", ".copying", ".watchdog_copy")):
                continue
            target = backup / path.relative_to(local) if route["directory"] else backup
            owned_path(config, target)
            key, stat = str(target), path.stat()
            previous = state.get(key)
            if pressure == "critical":
                if target.is_file() and not target.is_symlink():
                    # Reclaim only a byte-identical duplicate. A changing log's
                    # older snapshot is retained until a fresh copy is possible.
                    if target.stat().st_size == stat.st_size and digest(target) == digest(path):
                        reclaimed += target.stat().st_size
                        target.unlink()
                        state.pop(key, None)
                continue
            if disk(config["project_path"])["available_bytes"] < config["backup_min_free_bytes"]:
                break
            if (previous and target.exists() and previous["mtime_ns"] == stat.st_mtime_ns
                    and previous["size"] == stat.st_size):
                continue
            if time.time() - stat.st_mtime < 90:
                continue
            try:
                state[key] = copy_checked(path, target)
                copied += stat.st_size
            except ValueError:
                continue  # An active writer advanced; retry its backup later.
            if copied >= config["backup_bytes_per_cycle"]:
                write(state_file, state)
                return {"copied_bytes": copied, "reclaimed_bytes": reclaimed, "skipped_symlinks": skipped_symlinks}
    write(state_file, state)
    return {"copied_bytes": copied, "reclaimed_bytes": reclaimed, "skipped_symlinks": skipped_symlinks}


def emergency_storage(config, pressure):
    """Offload only closed, checksummed response payloads; keep their exact paths."""
    local = Path(config["control_root"])
    index = local / "emergency_index"
    entries = {entry["source"]: entry for path in index.glob("*.json") if (entry := read(path))}
    moved = restored = files = 0
    limit = config.get("emergency_bytes_per_cycle", 256 * 1024**2)
    file_limit = config.get("emergency_files_per_cycle", 256)
    if pressure == "critical":
        stage = config["stages"].get("evaluation")
        if not stage:
            return {"offloaded_bytes": 0, "restored_bytes": 0}
        root = Path(stage["root"])
        plan = read(root / "plan.json")
        nine = owned_path(config, plan["nine_root"])
        fingerprint = digest(nine / "manifest.json")
        for receipt_path in sorted((nine / "responses").glob("*.receipt.json")):
            receipt = read(receipt_path)
            name = receipt["file"]
            if (Path(name).name != name or name != receipt["id"] + ".json.gz"
                    or receipt.get("manifest_sha256") != fingerprint):
                raise ValueError("Emergency storage found a mismatched response receipt")
            source = owned_path(config, nine / "responses" / name)
            if source.is_symlink():
                continue
            before = source.stat()
            if time.time() - max(before.st_mtime, receipt_path.stat().st_mtime) < 120:
                continue
            if before.st_uid != os.getuid() or before.st_size != receipt["size"] or digest(source) != receipt["sha256"]:
                raise ValueError("Emergency storage refuses an unverified response")
            destination = local / "emergency_payloads" / name
            info = copy_checked(source, destination)
            if info["sha256"] != receipt["sha256"]:
                raise ValueError("Response changed during offload")
            entry = {"source": str(source), "local": str(destination), **info, "state": "copied", "at": now()}
            write(index / f"{name}.json", entry)
            # Create the replacement first. If metadata allocation fails, leave
            # the shared original intact and retry after duplicate cleanup.
            link = source.with_name(source.name + ".watchdog_link")
            if link.is_symlink() and link.readlink() == destination:
                link.unlink()
            link.symlink_to(destination)
            current = source.stat()
            if (current.st_ino, current.st_size, current.st_mtime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns):
                link.unlink()
                raise ValueError("Response was modified during offload")
            link.replace(source)
            entry["state"] = "offloaded"
            write(index / f"{name}.json", entry)
            moved += before.st_size
            files += 1
            if moved >= limit or files >= file_limit:
                break
    elif disk(config["project_path"])["available_bytes"] >= config["warning_bytes"] + limit:
        for entry in entries.values():
            source, payload = (owned_path(config, entry[key]) for key in ("source", "local"))
            if entry["state"] not in ("copied", "offloaded") or not source.is_symlink():
                continue
            if not source.is_symlink() or source.readlink() != payload or digest(payload) != entry["sha256"]:
                raise ValueError("Emergency payload identity changed")
            if disk(config["project_path"])["available_bytes"] < config["warning_bytes"] + limit:
                break
            # copy_checked replaces the symlink atomically with the checked copy.
            info = copy_checked(payload, source)
            if info["sha256"] != entry["sha256"]:
                raise ValueError("Restored response checksum differs")
            entry.update(state="restored", restored_at=now())
            write(index / f"{payload.name}.json", entry)
            restored += entry["size"]
            payload.unlink()
            files += 1
            if restored >= limit or files >= file_limit:
                break
    return {"offloaded_bytes": moved, "restored_bytes": restored}


def evaluation_progress(config):
    root = Path(config["stages"]["evaluation"]["root"])
    plan = read(root / "plan.json")
    nine, minerva = Path(plan["nine_root"]), Path(plan["minerva_root"])
    paths = [root / "progress.json", nine / "status.json", *sorted((nine / "progress").glob("*.json"))]
    for spec in plan["models"]:
        folder = minerva / spec["key"]
        paths.append(folder / "status.json")
        paths.extend(sorted((folder / "shards").glob("*/progress/*.json")))
    values = {}
    for path in paths:
        if path.exists():
            data = read(path)
            values[str(path)] = {k: v for k, v in data.items()
                                 if k not in ("updated_at", "started_at", "finished_at", "pid", "worker_pids")}
    values["saved_responses"] = len(list((nine / "responses").glob("*.receipt.json")))
    # Observe model downloads/conversion too; these legitimately use no GPU.
    cache = Path(plan["scratch"]) / "models"
    values["model_files"] = [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                             for p in sorted(cache.rglob("*")) if p.is_file()
                             and p.suffix in (".pt", ".safetensors", ".incomplete")]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def evaluation_steps(config):
    root = config["stages"]["evaluation"]["root"]
    members = allocation_processes(config)
    output = subprocess.check_output(["scontrol", "show", "step", config["job_id"], "-o"], text=True, timeout=30)
    result = []
    for line in output.splitlines():
        fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
        phase = {"polaris-nine-eval": "nine", "polaris-minerva-eval": "minerva"}.get(fields.get("Name"))
        if phase is None:
            continue
        step = fields.get("StepId", "")
        prefix, _, suffix = step.partition(".")
        if (prefix != config["job_id"] or not suffix.isdigit() or fields.get("State") != "RUNNING"
                or fields.get("NodeList") != config["node"] or fields.get("UserId") != str(os.getuid())):
            raise ValueError("Evaluation step identity does not match this allocation")
        host, _, pid = fields.get("SrunHost:Pid", "").rpartition(":")
        if host != config["node"] or not pid.isdigit():
            raise ValueError("Unknown evaluation launcher")
        process = identity(int(pid))
        if process is None:
            continue  # Accounting may briefly retain a finishing step.
        args = process["command"]
        if (process["uid"] != os.getuid() or int(pid) not in members
                or process["cgroup"] != config["control_cgroup"]
                or f"--jobid={config['job_id']}" not in args or root not in args
                or "qwen3_experiments.polaris_checkpoint_evaluation" not in args
                or "--phase" not in args or args[args.index("--phase") + 1] != phase):
            raise ValueError("Refusing to cancel an unrecognized launcher")
        result.append({"step": step, "phase": phase, "launcher": process})
    return result


def process_cpu(pid):
    process = identity(pid)
    if process is None:
        return None
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        return {"start_ticks": process["start_ticks"], "seconds": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")}
    except FileNotFoundError:
        return None


def stalled_evaluation(config):
    """Require sustained lack of results, CPU work AND GPU work before cancelling."""
    local = Path(config["control_root"])
    state_path = local / "evaluation_activity.json"
    steps = evaluation_steps(config)
    if len(steps) != 1:
        state_path.unlink(missing_ok=True)
        return {"state": "no_unique_active_evaluation_step"}
    step, timestamp = steps[0], time.time()
    members = allocation_processes(config)
    cpus = {str(pid): sample for pid, number in members.items() if step["step"] == f"{config['job_id']}.{number}"
            and (sample := process_cpu(pid)) is not None}
    if not cpus:
        return {"state": "step_processes_not_visible"}
    gpu_text = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=30)
    gpu = [int(line.strip()) for line in gpu_text.splitlines() if line.strip()]
    if len(gpu) != 8:
        raise ValueError("Expected all eight GPUs for idle detection")
    progress = evaluation_progress(config)
    old = read(state_path) if state_path.exists() else {}
    same = old.get("step") == step and timestamp - old.get("at", 0) <= max(120, 3 * config["interval_seconds"])
    unchanged = same and old.get("progress") == progress
    cpu_busy = True
    if same and all(pid in old["cpus"] and old["cpus"][pid]["start_ticks"] == s["start_ticks"] for pid, s in cpus.items()):
        cpu_seconds = sum(max(0, s["seconds"] - old["cpus"][pid]["seconds"]) for pid, s in cpus.items())
        cpu_busy = cpu_seconds / max(1, timestamp - old["at"]) > 0.10
    idle = max(gpu) <= 1 and not cpu_busy
    state = {"step": step, "at": timestamp, "cpus": cpus, "progress": progress,
             "last_progress_at": old["last_progress_at"] if unchanged else timestamp,
             "idle_since": (old.get("idle_since") or timestamp) if same and idle else None,
             "gpu_utilization": gpu, "cpu_busy": cpu_busy}
    write(state_path, state)
    if (not unchanged or not state["idle_since"]
            or timestamp - state["last_progress_at"] < config.get("stall_no_progress_seconds", 900)
            or timestamp - state["idle_since"] < config.get("stall_idle_seconds", 300)):
        return {"state": "observing", "step": step["step"], "idle_since": state["idle_since"],
                "last_progress_at": state["last_progress_at"]}
    verify(config)
    require_node(config)
    if evaluation_steps(config) != [step] or evaluation_progress(config) != progress:
        return {"state": "progress_resumed_before_recovery"}
    receipt = local / "stall_recoveries" / f"{time.time_ns()}.json"
    write(receipt, {"state": "cancelling_idle_step", "evidence": state, "at": now()})
    # Only a numeric, independently verified job.step is eligible. Never cancel
    # the allocation, batch/extern step, trainer, or another node's work.
    subprocess.run(["scancel", step["step"]], check=True, timeout=30)
    write(receipt, {"state": "cancel_requested", "evidence": state, "at": now()})
    state_path.unlink(missing_ok=True)
    return {"state": "cancel_requested", "step": step["step"], "receipt": str(receipt)}


def stale_controller(config, kind, status, process):
    """Replace only the controller; retain any existing workers or trainer."""
    if status.get("pid") != process["pid"] or status.get("state") == "complete":
        return None
    stamp = status.get("updated_at")
    if not stamp or time.time() - datetime.fromisoformat(stamp).timestamp() < config.get("stale_controller_seconds", 300):
        return None
    root = Path(config["stages"][kind]["root"])
    if kind == "training":
        if not (root / "training_started.json").exists() and status.get("state") != "waiting_for_predecessor":
            return None  # A launch transition cannot be assumed to be idle.
        if (root / "training_started.json").exists() and not (Path(config["control_root"]) / "training_identity.json").exists():
            remember_training(config)
            if not (Path(config["control_root"]) / "training_identity.json").exists():
                return None
    verify(config)
    require_node(config)
    if stage_process(config, kind) != process or read(root / config["stages"][kind]["status"]) != status:
        return None
    receipt = Path(config["control_root"]) / "stall_recoveries" / f"{time.time_ns()}_{kind}.json"
    write(receipt, {"state": "replacing_stale_controller", "process": process, "status": status, "at": now()})
    if identity(process["pid"]) != process:
        return None
    os.kill(process["pid"], signal.SIGTERM)
    return {"state": "controller_restart_requested", "kind": kind, "pid": process["pid"], "receipt": str(receipt)}


def stage_process(config, kind):
    stage = config["stages"][kind]
    local = Path(config["control_root"]) / f"{kind}_launch.json"
    receipts = [read(Path(stage["root"]) / "launch.json")]
    if local.exists():
        receipts.insert(0, read(local))
    for receipt in receipts:
        process = identity(receipt["pid"])
        if process is None:
            continue
        if (process["uid"] != os.getuid() or process["cgroup"] != config["control_cgroup"]
                or not any(arg in (stage["root"], str(Path(stage["root"]) / "plan.json"))
                           for arg in process["command"])):
            raise ValueError("Controller PID belongs to another process or allocation")
        if receipt.get("identity") and receipt["identity"] != process:
            raise ValueError("Controller PID was reused")
        return process
    return None


def active_work(config, kind):
    stage = config["stages"][kind]
    matches = []
    for pid in allocation_processes(config):
        process = identity(pid)
        if process is None or pid == os.getpid():
            continue
        if any(arg == stage["root"] or arg.startswith(stage["root"] + "/")
               for arg in process["command"]):
            # The checkpoint monitor is allowed to outlive its supervisor.
            if kind == "training" and "monitor" in process["command"]:
                continue
            matches.append(process)
    return matches


def controller_environment(config, kind):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("SLURM_", "RAY_", "VLLM_", "CUDA_", "L0_", "MAXRL_", "FCOV_", "GRPO_")):
            env.pop(key)
    env.pop("PYTHONHOME", None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH=config["stages"][kind]["runtime"], HF_HOME=config["hf_home"])
    return env


def spawn(config, config_path, kind, root, attempt):
    command = [config["python_bin"], "-u", config["code"], "invoke", "--config", str(config_path),
               "--kind", kind, "--output-root", str(root)]
    stage_kind = "evaluation" if kind == "evaluation" else "training"
    with (attempt / "controller.log").open("ab", buffering=0) as log:
        child = subprocess.Popen(command, cwd=config["stages"][stage_kind]["runtime"],
                                 env=controller_environment(config, stage_kind), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    # Wait briefly for exec, so the saved identity is not the pre-exec fork.
    for _ in range(50):
        process = identity(child.pid)
        if process is None or process["command"] == command:
            break
        time.sleep(0.02)
    return child


def restart(config, config_path, kind):
    verify(config)
    require_node(config)
    stage, local = config["stages"][kind], Path(config["control_root"])
    root = Path(stage["root"])
    with locks([root / "launch.lock"]):
        if stage_process(config, kind):
            return None
        state = read(root / stage["status"])
        if state.get("state") == "complete":
            return None
        started = root / "training_started.json"
        if kind == "training" and started.exists():
            # Recovery adopts the original launcher; invoke never calls supervise
            # when training_started exists, including after the launcher exits.
            if not (local / "training_identity.json").exists():
                raise ValueError("Original training identity is not recorded; refusing a new trainer")
        elif active_work(config, kind):
            raise ValueError("Existing stage workers still active; retaining them")
        names = [stage["process_lock"]]
        if kind == "evaluation":
            names.append("run.lock")
        with locks([root / name for name in names]):
            previous = read(root / "launch.json")
            attempt = local / "recoveries" / f"{time.time_ns()}_{kind}"
            attempt.mkdir(parents=True)
            write(attempt / "before.json", {"launch": previous, "status": state})
            if kind == "evaluation":
                redirect_closed_logs(config, kind)
            if kind == "training" and not started.exists() and (root / "training_exit.json").exists():
                failure = read(root / "training_exit.json")
                write(attempt / "pretraining_exit.json", failure)
                (root / "training_exit.json").unlink()
        child = spawn(config, config_path, kind, root, attempt)
        receipt = {**previous, "pid": child.pid, "identity": identity(child.pid), "launched_at": now(),
                   "watchdog_recovery": str(attempt), "controller_log": str(attempt / "controller.log")}
        write(local / f"{kind}_launch.json", receipt)
        write(attempt / "launch.json", receipt)
        try:
            write(root / "launch.json", receipt)
        except OSError as exc:
            if exc.errno not in SPACE_ERRORS:
                raise
        return receipt


def load_recovery(config):
    path = Path(config["recovery_code"])
    spec = importlib.util.spec_from_file_location("pipeline_training_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remember_training(config):
    root = Path(config["stages"]["training"]["root"])
    started = root / "training_started.json"
    destination = Path(config["control_root"]) / "training_identity.json"
    if not started.exists() or destination.exists():
        return
    plan, launch = read(root / "plan.json"), read(started)
    launcher = identity(launch["pid"])
    if launcher is None:
        return
    if (launcher["uid"] != os.getuid() or launcher["cgroup"] != config["control_cgroup"]
            or f"--jobid={config['job_id']}" not in launcher["command"]
            or str(Path(plan["runtime"]) / plan["launcher"]) not in launcher["command"]):
        raise ValueError("Training launcher does not match this pipeline")
    trainers = []
    for pid, step in allocation_processes(config).items():
        process = identity(pid)
        if process and "verl.trainer.main_ppo" in process["command"]:
            overrides = dict(arg.split("=", 1) for arg in process["command"] if "=" in arg)
            if overrides.get("trainer.default_local_dir", "").strip("'\"") == plan["checkpoint_dir"]:
                trainers.append((process, step))
    if not trainers:
        return  # The launcher may still be loading Ray and the model.
    if len(trainers) != 1:
        raise ValueError("Ambiguous training process identity")
    write(destination, {"launcher": launcher, "trainer": trainers[0][0], "step_id": trainers[0][1],
                        "started_sha256": digest(started), "recorded_at": now()})


def adopt_training(config, control, plan):
    """Observe the existing Slurm step and finish uploads; never launch training."""
    from huggingface_hub import HfApi

    root = Path(plan["output_root"])
    original = read(Path(config["control_root"]) / "training_identity.json")
    recovery = load_recovery(config)
    if digest(root / "training_started.json") != original["started_sha256"]:
        raise ValueError("Original training start receipt changed")
    control.verify_training_inputs(plan)
    with locks([root / "supervisor.lock"]), control.holder_locks(plan["holder_locks"]):
        status = read(root / "status.json")
        status.update(pid=os.getpid(), training_restarted=False)

        def update(state):
            status.update(state=state, updated_at=now())
            control.write(root / "status.json", status)
            control.write(root / "supervisor_status.json", {**status, "training_launcher_pid": original["launcher"]["pid"]})

        monitor = None
        while True:
            require_node(config)
            if monitor is None or monitor.poll() is not None:
                monitor = control.spawn(plan, "monitor", root / "hf_checkpoint_archive/upload.log")
            alive = recovery.same_process(original["launcher"], identity(original["launcher"]["pid"]))
            status["last_completed_step"] = recovery.last_completed_step(root / "train.log")
            if not alive and recovery.successful_exit(recovery.slurm_step(plan["job_id"], original["step_id"])):
                final = Path(plan["checkpoint_dir"]) / "global_step_100"
                archived = recovery.archived_steps(root, plan)
                if status["last_completed_step"] != 100 or not (
                        100 in archived or control.checkpoint_complete(final, final.parent)):
                    raise ValueError("Successful training exit lacks a verified final checkpoint")
                control.write(root / "training_exit.json", {"exit_code": 0, "finished_at": now(),
                              "slurm_accounting": recovery.slurm_step(plan["job_id"], original["step_id"])})
                break
            update("training")
            time.sleep(config["interval_seconds"])
        update("verifying_rollout_upload")
        control.write(root / "rollout_upload.json", control.audit_rollouts(plan, HfApi()))
        while recovery.archived_steps(root, plan) != plan["checkpoint_steps"]:
            if monitor.poll() is not None:
                monitor = control.spawn(plan, "monitor", root / "hf_checkpoint_archive/upload.log")
            update("waiting_for_checkpoint_archive")
            time.sleep(config["interval_seconds"])
        status.update(exit_code=0, finished_at=now())
        update("complete")


class ExistingMonitor:
    def __init__(self, process, root):
        self.process, self.pid, self.root = process, process["pid"], root

    def poll(self):
        actual = identity(self.pid)
        if actual is not None:
            if actual != self.process:
                raise ValueError("Monitor PID was reused")
            return None
        return 0 if read(self.root / "hf_checkpoint_archive/status.json").get("state") == "complete" else 1

    @property
    def returncode(self):
        return self.poll()


def invoke(config, config_path, kind):
    verify(config)
    require_node(config)
    stage_kind = "evaluation" if kind == "evaluation" else "training"
    stage = config["stages"][stage_kind]
    root, runtime = Path(stage["root"]), stage["runtime"]
    sys.path.insert(0, runtime)
    shared = importlib.import_module("qwen3_experiments.grpo_compute_control")
    shared.write = durable_writer(shared.write, config)
    if kind == "evaluation":
        module = importlib.import_module("qwen3_experiments.polaris_checkpoint_evaluation")
        module.write = shared.write
        module.queue(root, module.verify_plan(root))
        return
    control = importlib.import_module("qwen3_experiments.compression_l0_compute_control")
    control.write = shared.write
    plan = read(root / "plan.json")
    control.verify_training_inputs(plan)
    if kind == "monitor":
        sys.exit(shared.monitor(plan))

    def spawn_monitor(_plan, command, _log):
        if command != "monitor":
            raise ValueError("This training plan must not launch an additional evaluation queue")
        status_path = root / "hf_checkpoint_archive/status.json"
        if status_path.exists():
            process = identity(read(status_path)["pid"])
            if process is not None:
                if (process["uid"] != os.getuid() or process["cgroup"] != config["control_cgroup"]
                        or "monitor" not in process["command"]
                        or not any(arg in (str(root), str(root / "plan.json")) for arg in process["command"])):
                    raise ValueError("Unexpected checkpoint monitor identity")
                return ExistingMonitor(process, root)
        attempt = Path(config["control_root"]) / "monitors" / str(time.time_ns())
        attempt.mkdir(parents=True)
        return spawn(config, config_path, "monitor", root, attempt)

    control.spawn = spawn_monitor
    if (root / "training_started.json").exists():
        adopt_training(config, control, plan)
    else:
        control.supervise(plan)


def tick(config, config_path):
    require_node(config)
    local = Path(config["control_root"])
    snapshot = {"pid": os.getpid(), "node": config["node"], "job_id": config["job_id"], "updated_at": now(),
                "state": "monitoring", "issues": [], "controls": {}}
    snapshot["disks"] = {"project": disk(config["project_path"]), "compute": disk(local)}
    free = snapshot["disks"]["project"]["available_bytes"]
    pressure = "critical" if free < config["critical_bytes"] else "low" if free < config["warning_bytes"] else "normal"
    if config.get("autonomous_recovery"):
        for request in (local / "blocked_writes").glob("*.json"):
            item = read(request)
            process = identity(item["pid"])
            if process is None:
                request.unlink()
            elif process["uid"] == os.getuid() and process["cgroup"] == config["control_cgroup"]:
                owned_path(config, item["path"])
                pressure = "critical"  # EDQUOT can occur even when df shows free space.
    snapshot["disk_pressure"] = pressure
    try:
        snapshot["backups"] = backups(config, pressure)
    except Exception as exc:
        snapshot["issues"].append(f"storage: {exc}")
    if config.get("autonomous_recovery"):
        actions = [("emergency_storage", lambda: emergency_storage(config, pressure))]
        if "evaluation" in config["stages"]:
            actions.append(("evaluation_activity", lambda: stalled_evaluation(config)))
        for name, action in actions:
            try:
                snapshot[name] = action()
            except Exception as exc:
                snapshot["issues"].append(f"{name}: {exc}")
    try:
        remember_training(config)
    except Exception as exc:
        snapshot["issues"].append(f"training_identity: {exc}")
    for kind, stage in config["stages"].items():
        try:
            status = read(Path(stage["root"]) / stage["status"])
            process = stage_process(config, kind)
            snapshot["controls"][kind] = {"pid": process["pid"] if process else None, "alive": process is not None,
                                          "state": status.get("state")}
            if status.get("state") == "complete":
                continue
            if process is not None:
                if config.get("autonomous_recovery"):
                    action = stale_controller(config, kind, status, process)
                    if action:
                        snapshot.setdefault("recovery_actions", []).append(action)
                        continue
                recovered = local / f"{kind}_launch.json"
                if recovered.exists() and read(recovered)["pid"] == process["pid"]:
                    launch_path = Path(stage["root"]) / "launch.json"
                    if read(launch_path) != read(recovered):
                        with locks([Path(stage["root"]) / "launch.lock"]):
                            write(launch_path, read(recovered))
                stamp = status.get("updated_at", status.get("started_at"))
                if stamp and time.time() - datetime.fromisoformat(stamp).timestamp() > 180:
                    snapshot["issues"].append(f"{kind}: live controller heartbeat is stale; process retained")
                continue
            retry_path = local / f"{kind}_retry.json"
            retry = read(retry_path) if retry_path.exists() else {"attempts": 0, "next_at": 0}
            if time.time() < retry["next_at"]:
                continue
            retry.update(attempts=retry["attempts"] + 1,
                         next_at=time.time() + min(120, 60 * 2 ** min(retry["attempts"], 4)))
            write(retry_path, retry)
            restored = restart(config, config_path, kind)
            if restored:
                snapshot.setdefault("recovered", {})[kind] = restored
        except Exception as exc:
            snapshot["issues"].append(f"{kind}: {exc}")
    if snapshot["controls"] and all(s["state"] == "complete" for s in snapshot["controls"].values()):
        if len(snapshot["controls"]) == len(config["stages"]):
            snapshot["state"] = "complete"
    return snapshot


def watch(config, config_path):
    verify(config)
    require_node(config)
    local = Path(config["control_root"])
    previous = None
    with locks([local / "watchdog.lock"]):
        while True:
            try:
                snapshot = tick(config, config_path)
            except Exception as exc:
                snapshot = {"pid": os.getpid(), "updated_at": now(), "state": "needs_attention", "issues": [str(exc)]}
            write(local / "status.json", snapshot)
            if config.get("status_mirror"):
                try:
                    write(owned_path(config, config["status_mirror"]), snapshot)
                except OSError as exc:
                    if exc.errno not in SPACE_ERRORS:
                        # The local record remains authoritative even if a
                        # project administrator changes its permissions.
                        snapshot["issues"].append(f"status mirror: {exc}")
            signature = json.dumps({k: snapshot.get(k) for k in ("state", "disk_pressure", "issues", "recovered", "recovery_actions")}, sort_keys=True)
            if signature != previous:
                with (local / "events.jsonl").open("a") as stream:
                    stream.write(json.dumps(snapshot) + "\n")
                previous = signature
            # Reap only this watchdog's recovery children.
            try:
                while os.waitpid(-1, os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                pass
            # Continue after completion to back up node-local artifacts whenever
            # project headroom returns; no completed stage is launched again.
            time.sleep(config["interval_seconds"])


def serve(config, config_path):
    """Keep the watchdog itself running, independently of the interactive session."""
    verify(config)
    require_node(config)
    local = Path(config["control_root"])
    with locks([local / "service.lock"]):
        attempts = 0
        while True:
            require_node(config)
            # A pre-existing watchdog is retained; the service becomes its
            # custodian until it exits instead of starting a duplicate.
            receipt = read(local / "launch.json") if (local / "launch.json").exists() else None
            process = identity(receipt["pid"]) if receipt else None
            child = None
            if process is not None:
                if (process != receipt.get("identity") or config["code"] not in process["command"]
                        or "watch" not in process["command"] or process["cgroup"] != config["control_cgroup"]):
                    raise ValueError("Existing watchdog identity changed")
            else:
                with locks([local / "watchdog.lock"]):
                    pass
                command = [config["python_bin"], "-u", config["code"], "watch", "--config", str(config_path)]
                with (local / "watchdog.log").open("ab", buffering=0) as log:
                    child = subprocess.Popen(command, cwd=local, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                for _ in range(50):
                    process = identity(child.pid)
                    if process is None or process["command"] == command:
                        break
                    time.sleep(0.02)
                receipt = {"pid": child.pid, "identity": process, "launched_at": now(),
                           "config_sha256": digest(config_path), "service_pid": os.getpid()}
                write(local / "launch.json", receipt)
                attempts += 1
            while process is not None:
                write(local / "service_status.json", {"pid": os.getpid(), "watchdog_pid": receipt["pid"],
                      "state": "supervising", "watchdog_launches": attempts, "updated_at": now()})
                time.sleep(config["interval_seconds"])
                require_node(config)
                actual = identity(receipt["pid"])
                if actual is not None and actual != process:
                    raise ValueError("Watchdog PID was reused")
                process = actual
            if child is not None:
                child.wait()
            write(local / "service_status.json", {"pid": os.getpid(), "state": "restarting_watchdog", "updated_at": now()})
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "watch", "invoke", "prepare-storage", "inspect"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=("evaluation", "training", "monitor"))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = read(args.config)
    if args.command == "serve":
        serve(config, args.config)
    elif args.command == "watch":
        watch(config, args.config)
    elif args.command == "invoke":
        invoke(config, args.config, args.kind)
    elif args.command == "prepare-storage":
        verify(config)
        require_node(config)
        prepare_empty_routes(config)
    else:
        verify(config)
        require_node(config)
        print(json.dumps(tick(config, args.config), indent=2))


if __name__ == "__main__":
    main()
