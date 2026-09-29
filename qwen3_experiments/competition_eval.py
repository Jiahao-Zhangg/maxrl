"""Persistent, resumable paired USACOBench and CodeContests evaluation."""

import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

from qwen3_experiments.code_grading import final_code
from qwen3_experiments.competition_grading import grade

MODULE = "qwen3_experiments.competition_eval"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def persist(plan, name, value):
    value = {**value, "updated_at": now()}
    successes = 0
    for root in (Path(plan["output_root"]), Path(plan["scratch"]) / "control_mirrors"):
        try:
            write(root / name, value)
            successes += 1
        except OSError as exc:
            if exc.errno not in (errno.ENOSPC, errno.EDQUOT, errno.EIO):
                raise
            # Releasing only this run's reserve works even when status cannot be written.
            (root / ".disk_reserve").unlink(missing_ok=True)
            try:
                write(root / name, value)
                successes += 1
            except OSError:
                pass
    return bool(successes)


def state(plan, name):
    for root in (Path(plan["output_root"]), Path(plan["scratch"]) / "control_mirrors"):
        try:
            return read(root / name)
        except (OSError, ValueError):
            pass
    return {}


def environment(plan, gpu=None):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("RAY_", "VLLM_", "SLURM_", "CUDA_", "WANDB_", "MAXRL_")):
            env.pop(key)
    scratch = Path(plan["scratch"])
    env.update(PATH=str(Path(plan["python_bin"]).parent) + ":" + env.get("PATH", ""),
               PYTHONPATH=plan["runtime"], PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VLLM_WORKER_MULTIPROC_METHOD="spawn",
               VLLM_ATTENTION_BACKEND="FLASH_ATTN", VLLM_USE_V1="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               TMPDIR=str(scratch / "tmp"), TRITON_CACHE_DIR=str(scratch / "triton"),
               HF_HOME=str(scratch / "hf_home"), NCCL_DEBUG="WARN")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def active(plan, commands, task=None):
    pids = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            if entry.stat().st_uid != os.getuid() or int(entry.name) == os.getpid():
                continue
            args = (entry / "cmdline").read_bytes().decode().split("\0")
            if (MODULE in args and plan["output_root"] in args and any(c in args for c in commands)
                    and (task is None or task in args)):
                pids.append(int(entry.name))
        except (OSError, UnicodeError):
            pass
    return pids


def gpu_busy():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return any(line.strip().isdigit() for line in result.stdout.splitlines())


def launch_child(plan, command, task=None, rank=None, slurm=False):
    args = [plan["python_bin"], "-u", "-m", MODULE, command, "--root", plan["output_root"]]
    if task:
        args += ["--task", task]
    if rank is not None:
        args += ["--rank", str(rank)]
    if slurm:
        args = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                f"--nodelist={plan['node']}", "--cpus-per-task=96", "--gres=gpu:8",
                "--kill-on-bad-exit=1", "--job-name=usaco-codecontests-eval", *args]
    log_path = Path(plan["scratch"]) / "logs" / f"{command}_{task or 'control'}_{rank}.log"
    with log_path.open("ab", buffering=0) as log:
        return subprocess.Popen(args, cwd=plan["runtime"], env=environment(plan, rank),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)


def question_path(plan, task):
    return Path(plan["output_root"]) / "data" / f"{plan['tasks'][task]['dataset']}.json"


def folder(plan, task):
    return Path(plan["output_root"]) / "evaluation" / task


def verify_response(plan, task, question, record):
    model = plan["models"][plan["tasks"][task]["model"]]
    if (record["id"] != question["id"] or record["index"] != question["source_index"]
            or record["model_revision"] != model["revision"]
            or record["plan_sha256"] != digest(Path(plan["output_root"]) / "plan.json")):
        raise ValueError("Existing response belongs to another question, model or plan")


def worker(plan, task, rank):
    from vllm import LLM, SamplingParams

    questions = read(question_path(plan, task))[rank::8]
    directory = folder(plan, task) / "responses"
    pending = []
    for question in questions:
        path = directory / f"{question['source_index']}.json"
        if path.exists():
            verify_response(plan, task, question, read(path))
        else:
            pending.append(question)
    if not pending:
        return
    model = plan["models"][plan["tasks"][task]["model"]]
    engine = LLM(model=model["path"], tokenizer=model["path"], **plan["evaluation_engine"])
    plan_hash = digest(Path(plan["output_root"]) / "plan.json")
    # Use the same batch size and per-question seeds for both checkpoints.
    for offset in range(0, len(pending), 64):
        batch = pending[offset:offset + 64]
        outputs = engine.generate([{"prompt_token_ids": q["prompt_token_ids"]} for q in batch],
                                  [SamplingParams(**plan["sampling"], seed=q["source_index"]) for q in batch],
                                  use_tqdm=False)
        if len(outputs) != len(batch):
            raise ValueError("Missing generation outputs")
        for question, output in zip(batch, outputs):
            if len(output.outputs) != 1:
                raise ValueError("Expected one completion per question")
            sample = output.outputs[0]
            write(directory / f"{question['source_index']}.json", {
                "id": question["id"], "index": question["source_index"], "response": sample.text,
                "token_ids": list(sample.token_ids), "finish_reason": sample.finish_reason,
                "model_revision": model["revision"], "plan_sha256": plan_hash, "finished_at": now(),
            })


def complete(plan, task):
    audit_path = folder(plan, task) / "audit.json"
    if not audit_path.exists():
        return False
    audit = read(audit_path)
    if (audit["questions"] != plan["tasks"][task]["questions"]
            or audit["plan_sha256"] != digest(Path(plan["output_root"]) / "plan.json")
            or audit["metrics_sha256"] != digest(folder(plan, task) / "metrics.json")):
        raise ValueError("Evaluation completion audit mismatch")
    return audit["complete"]


def evaluate(plan, task):
    directory = folder(plan, task)
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        own = stack.enter_context((Path(plan["scratch"]) / "guard/evaluate.lock").open("a"))
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in plan["holder_locks"]:
            lock = stack.enter_context(Path(path).open("r"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if gpu_busy():
            return 75
        children = [launch_child(plan, "worker", task, rank) for rank in range(8)]
        try:
            while any(p.poll() is None for p in children):
                failed = [p.returncode for p in children if p.returncode not in (None, 0)]
                if failed:
                    raise RuntimeError(f"Generation worker failed: {failed}")
                time.sleep(5)
            if any(p.returncode for p in children):
                raise RuntimeError("Generation worker failed")
        finally:
            for process in children:
                if process.poll() is None:
                    process.terminate()
            for process in children:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, 9)
                    process.wait()
    questions = read(question_path(plan, task))
    grading = plan["grading"]

    def grade_one(question):
        index = question["source_index"]
        response_path = directory / "responses" / f"{index}.json"
        response = read(response_path)
        verify_response(plan, task, question, response)
        response_hash = digest(response_path)
        destination = directory / "grades" / f"{index}.json"
        if destination.exists():
            previous = read(destination)
            if previous["id"] == question["id"] and previous["response_sha256"] == response_hash:
                return previous
            raise ValueError("Existing grade has a different response identity")
        code, reason = final_code(response["response"])
        result = (grade(grading, question, code) if reason == "ok"
                  else {"results": [], "score": 0.0, "error": reason})
        record = {**result, "id": question["id"], "difficulty": question["difficulty"],
                  "tokens": len(response["token_ids"]), "response_sha256": response_hash,
                  "finished_at": now()}
        write(destination, record)
        return record

    with ThreadPoolExecutor(max_workers=plan["grading_workers"]) as pool:
        records = list(pool.map(grade_one, questions))
    groups = {"all": records}
    for record in records:
        groups.setdefault(record["difficulty"], []).append(record)
    metrics = {name: {"questions": len(group), "correct": sum(r["score"] for r in group),
                      "pass_at_1_percent": 100 * sum(r["score"] for r in group) / len(group),
                      "mean_output_tokens": sum(r["tokens"] for r in group) / len(group)}
               for name, group in groups.items()}
    write(directory / "metrics.json", metrics)
    write(directory / "audit.json", {
        "complete": True, "questions": len(questions), "completed_at": now(),
        "plan_sha256": digest(Path(plan["output_root"]) / "plan.json"),
        "metrics_sha256": digest(directory / "metrics.json"),
        "response_hashes": {str(q["source_index"]): digest(directory / "responses" / f"{q['source_index']}.json")
                            for q in questions},
        "grade_hashes": {str(q["source_index"]): digest(directory / "grades" / f"{q['source_index']}.json")
                         for q in questions},
    })
    return 0


def progress(plan):
    return {task: {"questions": item["questions"],
                   "responses": len(list((folder(plan, task) / "responses").glob("*.json"))),
                   "graded": len(list((folder(plan, task) / "grades").glob("*.json")))}
            for task, item in plan["tasks"].items()}


def report(plan):
    metrics = {task: read(folder(plan, task) / "metrics.json") for task in plan["order"] if complete(plan, task)}
    persist(plan, "comparison.json", {"tasks": metrics})
    lines = ["# USACOBench and CodeContests paired evaluation", "",
             "Thinking on; 32768 output tokens; temperature/top-p/top-k 0.6/0.95/20; one sample per question.",
             "Grade code after the final thinking close, without an EOS requirement. No retrieval or reflection.",
             "USACO: official 307-problem paper subset and official Python checker.",
             "CodeContests: all 165 test problems, public/private/generated tests; official compiled output comparator,",
             "with bubblewrap Python execution and dataset time/memory limits (not the upstream Sandbox2 backend).", "",
             "| Task | Difficulty | Questions | Correct | pass@1 (%) | Mean output tokens |",
             "|---|---|---:|---:|---:|---:|"]
    for task, groups in metrics.items():
        for group, value in groups.items():
            lines.append(f"| {task} | {group} | {value['questions']} | {value['correct']:.0f} | "
                         f"{value['pass_at_1_percent']:.2f} | {value['mean_output_tokens']:.1f} |")
    for root in (Path(plan["output_root"]), Path(plan["scratch"]) / "control_mirrors"):
        try:
            (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
        except OSError:
            pass


def queue(plan):
    children = []
    with (Path(plan["scratch"]) / "guard/queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (Path(plan["output_root"]) / "cancellation.json").exists():
            try:
                children = [p for p in children if p.poll() is None]
                current = next((task for task in plan["order"] if not complete(plan, task)), None)
                if current is None:
                    report(plan)
                    persist(plan, "queue_status.json", {"state": "complete", "progress": progress(plan)})
                    return
                pids = active(plan, ("evaluate", "worker"))
                status = "evaluating"
                if not pids:
                    if gpu_busy():
                        status = "waiting_for_free_gpus"
                    else:
                        child = launch_child(plan, "evaluate", current, slurm=True)
                        children.append(child)
                        previous = state(plan, f"attempts/{current}.json")
                        persist(plan, f"attempts/{current}.json", {
                            "attempt": previous.get("attempt", 0) + 1, "pid": child.pid, "started_at": now()})
                        pids = [child.pid]
                persist(plan, "queue_status.json", {"state": status, "task": current, "active_pids": pids,
                                                    "progress": progress(plan), "pid": os.getpid()})
                report(plan)
            except Exception as exc:
                persist(plan, "queue_status.json", {"state": "recovering", "error": str(exc),
                                                    "progress": progress(plan), "pid": os.getpid()})
                print(traceback.format_exc(), flush=True)
            time.sleep(30)


def supervise(plan):
    children = []
    with (Path(plan["scratch"]) / "guard/supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (Path(plan["output_root"]) / "cancellation.json").exists():
            children = [p for p in children if p.poll() is None]
            if state(plan, "queue_status.json").get("state") == "complete":
                persist(plan, "supervisor_status.json", {"state": "complete", "pid": os.getpid()})
                return
            queues = active(plan, ("queue",))
            if not queues:
                child = launch_child(plan, "queue")
                children.append(child)
                queues = [child.pid]
            disks = {}
            for label, root in (("home", Path(plan["output_root"])),
                                ("compute", Path(plan["scratch"]) / "control_mirrors")):
                try:
                    free = shutil.disk_usage(root).free
                    if free < (2 << 30):
                        (root / ".disk_reserve").unlink(missing_ok=True)
                    disks[label] = {"free_bytes": shutil.disk_usage(root).free}
                except OSError as exc:
                    disks[label] = {"error": str(exc)}
            persist(plan, "supervisor_status.json", {"state": "supervising", "pid": os.getpid(),
                                                     "queue_pids": queues, "disks": disks})
            time.sleep(30)


def verify(plan, full=False):
    if socket.gethostname().split(".")[0] != plan["node"]:
        raise ValueError("Run this controller only on its designated compute node")
    for path, expected in plan["frozen_files"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen artifact changed: {path}")
    if full:
        for model in plan["models"].values():
            for name, expected in model["files_sha256"].items():
                if digest(Path(model["path"]) / name) != expected:
                    raise ValueError(f"Model artifact changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "supervise", "queue", "evaluate", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    plan = read(args.root / "plan.json")
    if (args.root / "cancellation.json").exists():
        raise SystemExit("Evaluation cancelled")
    if args.command == "launch":
        verify(plan, full=True)
        preflight = read(args.root / "preflight.json")
        if preflight["state"] != "passed" or preflight["plan_sha256"] != digest(args.root / "plan.json"):
            raise ValueError("Missing preflight for this plan")
        with (Path(plan["scratch"]) / "guard/launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if active(plan, ("supervise", "queue", "evaluate", "worker")):
                raise ValueError("Evaluation is already running")
            child = launch_child(plan, "supervise")
            record = {"started_at": now(), "supervisor_pid": child.pid,
                      "plan_sha256": digest(args.root / "plan.json"), "node": plan["node"]}
            persist(plan, "launch.json", record)
            print(json.dumps(record))
    elif args.command == "supervise":
        supervise(plan)
    elif args.command == "queue":
        queue(plan)
    elif args.command == "worker":
        worker(plan, args.task, args.rank)
    else:
        verify(plan)
        sys.exit(evaluate(plan, args.task))


if __name__ == "__main__":
    main()
