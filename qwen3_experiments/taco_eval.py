"""Pinned TACO pass@1 evaluation and paired vLLM concurrency benchmark."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback

MODULE = "qwen3_experiments.taco_eval"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def persist(root, plan, name, value):
    """Mirror control records away from the shared filesystem; keep retrying ENOSPC."""
    while True:
        saved = False
        for target in (Path(plan["scratch"]) / "control_mirrors" / name, root / name):
            try:
                write(target, value)
                saved |= target == root / name
            except OSError as exc:
                if exc.errno not in (errno.ENOSPC, errno.EDQUOT):
                    raise
        if saved:
            return
        time.sleep(15)


def verify(root, compute=True):
    plan = read(root / "plan.json")
    if str(root.resolve()) != plan["output_root"]:
        raise ValueError("Unexpected output root")
    receipt = root / "launch.json"
    if receipt.exists() and read(receipt)["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("TACO plan changed after launch")
    for name, checksum in plan["frozen_files"].items():
        if digest(root / name) != checksum:
            raise ValueError(f"Changed frozen TACO input: {name}")
    if compute:
        from qwen3_experiments.minerva_individual_budget import require_compute

        if require_compute(plan["job_id"]) != plan["node"]:
            raise ValueError("TACO is restricted to its assigned compute node")
    return plan


def test_case_issue(row):
    try:
        tests = json.loads(row["input_output"])
    except (KeyError, TypeError, ValueError):
        return "invalid_input_output_json"
    if not isinstance(tests, dict) or not all(isinstance(tests.get(key), list) for key in ("inputs", "outputs")):
        return "invalid_test_lists"
    if not tests["inputs"]:
        return "empty_test_cases"
    if len(tests["inputs"]) != len(tests["outputs"]):
        return "unpaired_test_cases"
    return None


def select_indices(rows, seed=0, *, eligible_indices=None):
    allowed = set(range(len(rows))) if eligible_indices is None else set(eligible_indices)
    if not allowed <= set(range(len(rows))):
        raise ValueError("Eligibility indices must refer to original dataset rows")
    rng = random.Random(seed)
    groups = []
    for level in ("EASY", "MEDIUM"):
        eligible = [i for i, row in enumerate(rows) if row["difficulty"] == level and i in allowed]
        if len(eligible) < 100:
            raise ValueError(f"Need 100 {level} questions; only {len(eligible)} available")
        groups.append(rng.sample(eligible, 100))
    # Four balanced, fixed 50-question shards; each GPU sees both concurrency settings.
    return [index for rank in range(4) for pair in zip(groups[0][rank::4], groups[1][rank::4])
            for index in pair]


def prompt_for(row):
    inputs = json.loads(row["input_output"])
    starter = row.get("starter_code") or ""
    interface = ("Use Call-Based format. Implement the function or class in the starter code."
                 if inputs.get("fn_name") else
                 "Use Standard Input format. Read from standard input and write to standard output.")
    return ("Solve the following programming problem in Python.\n\nQUESTION:\n" + row["question"]
            + ("\n\nStarter code:\n```python\n" + starter + "\n```" if starter else "")
            + "\n\n" + interface
            + "\nReturn the complete solution in a single Python code block in your final answer.")


def extract_final_code(text):
    if "</think>" not in text:
        return "", "missing_thinking_close"
    final = text.rsplit("</think>", 1)[1].strip()
    # Never inspect a code block inside the reasoning portion.
    blocks = re.findall(r"```(?:python3?|py)?[ \t]*\n(.*?)```", final, flags=re.DOTALL | re.IGNORECASE)
    code = blocks[-1].strip() if blocks else final
    return code, "ok" if code else "empty_final"


def concurrency_for(round_number, rank):
    return 16 if (rank < 2) == (round_number == 0) else 32


def sandbox_command(plan, payload):
    prefix = str(Path(plan["python_bin"]).parent.parent)
    command = [plan["bubblewrap"], "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
    for path in ("/usr", "/lib", "/lib64", "/bin"):
        if Path(path).exists():
            command.extend(["--ro-bind", path, path])
    # Mount the private scratch filesystem before exposing an environment that
    # may itself live under /tmp on the compute node.
    command += ["--tmpfs", "/tmp", "--ro-bind", prefix, prefix, "--ro-bind", plan["official"], "/official",
                "--ro-bind", plan["sandbox_runner"], "/runner.py", "--ro-bind", str(payload), "/input.json",
                "--proc", "/proc", "--dev", "/dev", "--clearenv"]
    for key, value in {"PATH": prefix + "/bin:/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp",
                       "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}.items():
        command.extend(["--setenv", key, value])
    return command + ["--chdir", "/tmp", plan["python_bin"], "/runner.py"]


def grade_code(plan, sample, code):
    tests = json.loads(sample["input_output"])
    # The official standard-input grader may retry three times, then resynthesize.
    wall_timeout = max(90, 24 * len(tests["inputs"]) + 30)
    temporary_root = Path(plan["scratch"]) / "grading_tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
        payload = Path(directory) / "input.json"
        write(payload, {"input_output": sample["input_output"], "code": code, "wall_timeout": wall_timeout})
        start = time.monotonic()
        try:
            result = subprocess.run(sandbox_command(plan, payload), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, timeout=wall_timeout, start_new_session=True)
        except subprocess.TimeoutExpired:
            return {"results": [-1], "grader_exception": "outer_wall_timeout", "seconds": time.monotonic() - start}
        if result.returncode and ("bwrap:" in result.stderr or "Traceback" in result.stderr):
            raise RuntimeError(f"Isolated grader failed ({result.returncode}): {result.stderr[-2000:]}")
        if result.returncode or not result.stdout.strip():
            return {"results": [-2], "grader_exception": f"grader_process_exit:{result.returncode}",
                    "seconds": time.monotonic() - start}
        record = json.loads(result.stdout.strip())
        if not record.get("results"):
            raise ValueError("Official grader returned no test outcomes")
        record["seconds"] = time.monotonic() - start
        return record


def worker_complete(root, concurrency, rank):
    folder = root / "generation" / f"seqs_{concurrency}" / f"shard_{rank}"
    summary_path = folder / "summary.json"
    if not summary_path.exists():
        return None
    summary = read(summary_path)
    if summary["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Generation receipt belongs to another plan")
    if summary["state"] == "oom":
        return summary
    if summary["state"] != "complete" or summary["count"] != 50 or digest(folder / "responses.jsonl") != summary["responses_sha256"]:
        raise ValueError("Invalid generation completion receipt")
    records = [json.loads(line) for line in (folder / "responses.jsonl").read_text().splitlines()]
    expected = {row["id"] for row in read(root / "questions.json")[rank * 50:(rank + 1) * 50]}
    if len(records) != 50 or {row["id"] for row in records} != expected:
        raise ValueError("Missing or duplicated TACO responses")
    samples = {row["id"]: row for row in read(root / "questions.json")}
    for row in records:
        if (row["seed"] != samples[row["id"]]["seed"] or row["difficulty"] != samples[row["id"]]["difficulty"]
                or not 0 < row["output_tokens"] == len(row["token_ids"]) <= 32768):
            raise ValueError("Response has the wrong seed, difficulty, or token count")
    return summary


def worker(root, rank, concurrency):
    plan = verify(root)
    folder = root / "generation" / f"seqs_{concurrency}" / f"shard_{rank}"
    folder.mkdir(parents=True, exist_ok=True)
    previous = worker_complete(root, concurrency, rank)
    if previous:
        return
    samples = read(root / "questions.json")[rank * 50:(rank + 1) * 50]
    stop = threading.Event()
    memory = {"peak_gpu_used_mib": 0, "gpu_total_mib": None, "gpu_uuid": None}

    def monitor():
        import pynvml

        pynvml.nvmlInit()
        visible = os.environ["CUDA_VISIBLE_DEVICES"]
        handle = (pynvml.nvmlDeviceGetHandleByIndex(int(visible)) if visible.isdigit()
                  else pynvml.nvmlDeviceGetHandleByUUID(visible))
        memory["gpu_uuid"] = pynvml.nvmlDeviceGetUUID(handle)
        while not stop.is_set():
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory["gpu_total_mib"] = info.total / 1024**2
            memory["peak_gpu_used_mib"] = max(memory["peak_gpu_used_mib"], info.used / 1024**2)
            stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    metadata = {"rank": rank, "max_num_seqs": concurrency, "plan_sha256": digest(root / "plan.json"),
                "pid": os.getpid(), "started_at": now(), "count": 0}
    try:
        from vllm import LLM, SamplingParams

        init_start = time.monotonic()
        llm = LLM(model=plan["model"]["path"], tokenizer=plan["model"]["path"],
                  max_num_seqs=concurrency, **plan["engine"])
        engine = llm.llm_engine
        metadata["initialization_seconds"] = time.monotonic() - init_start
        engine.add_request("warmup", {"prompt_token_ids": samples[0]["warmup_token_ids"]},
                           SamplingParams(temperature=0, max_tokens=32, seed=0))
        while engine.has_unfinished_requests():
            engine.step()
        start = time.monotonic()
        for sample in samples:
            engine.add_request(sample["id"], {"prompt_token_ids": sample["prompt_token_ids"]},
                               SamplingParams(**plan["sampling"], seed=sample["seed"]))
        mapping = {row["id"]: row for row in samples}
        peak_running, total_tokens, count = 0, 0, 0
        last_update = 0
        with (folder / "responses.jsonl").open("w") as stream:
            while engine.has_unfinished_requests():
                outputs = engine.step()
                peak_running = max(peak_running, sum(len(scheduler.running) for scheduler in engine.scheduler))
                for output in outputs:
                    if not output.finished:
                        continue
                    generated = output.outputs[0]
                    sample = mapping[output.request_id]
                    record = {"id": output.request_id, "difficulty": sample["difficulty"], "seed": sample["seed"],
                              "text": generated.text, "token_ids": list(generated.token_ids),
                              "output_tokens": len(generated.token_ids), "finish_reason": generated.finish_reason,
                              "stop_reason": generated.stop_reason, "finished_after_seconds": time.monotonic() - start}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    count += 1
                    total_tokens += record["output_tokens"]
                if time.monotonic() - last_update > 30:
                    persist(root, plan, str(folder.relative_to(root) / "progress.json"),
                            {**metadata, **memory, "state": "running", "count": count, "updated_at": now(),
                             "peak_running_sequences": peak_running, "output_tokens": total_tokens})
                    last_update = time.monotonic()
        elapsed = time.monotonic() - start
        if count != 50 or not memory["gpu_total_mib"]:
            raise ValueError("Incomplete generation or missing GPU telemetry")
        metadata.update(state="complete", count=count, generation_seconds=elapsed, output_tokens=total_tokens,
                        output_tokens_per_second=total_tokens / elapsed, peak_running_sequences=peak_running,
                        preemptions=sum(s.num_cumulative_preemption for s in engine.scheduler),
                        responses_sha256=digest(folder / "responses.jsonl"))
    except Exception as exc:
        traceback.print_exc()
        if "out of memory" not in str(exc).lower() and "outofmemory" not in type(exc).__name__.lower():
            raise
        metadata.update(state="oom", error=f"{type(exc).__name__}: {exc}")
    finally:
        stop.set()
        thread.join(timeout=5)
    persist(root, plan, str(folder.relative_to(root) / "summary.json"), {**metadata, **memory, "finished_at": now()})


def grade_response(root, plan, sample, response, concurrency):
    path = root / "grades" / f"seqs_{concurrency}" / f"{sample['id']}.json"
    code, extraction = extract_final_code(response["text"])
    identity = hashlib.sha256(json.dumps({"response": response, "tests": sample["input_output"],
                                         "plan_sha256": digest(root / "plan.json")}, sort_keys=True).encode()).hexdigest()
    if path.exists():
        result = read(path)
        if result["identity"] != identity:
            raise ValueError("Cached grade belongs to different output or tests")
        return result
    result = (grade_code(plan, sample, code) if extraction == "ok"
              else {"results": [0], "grader_exception": None, "seconds": 0})
    result.update(id=sample["id"], difficulty=sample["difficulty"], code=code, extraction=extraction,
                  identity=identity, correct=all(value > 0 for value in result["results"]))
    persist(root, plan, str(path.relative_to(root)), result)
    return result


def build_report(root, plan):
    sys.path.insert(0, plan["official"])
    from compute_metric import compute_metrics

    samples = {row["id"]: row for row in read(root / "questions.json")}
    modes = {}
    for concurrency in (16, 32):
        shards = [worker_complete(root, concurrency, rank) for rank in range(4)]
        if any(item is None for item in shards):
            raise ValueError("Missing benchmark shard outcome")
        complete = all(item["state"] == "complete" for item in shards)
        mode = {"shards": shards, "complete": complete, "oom": any(item["state"] == "oom" for item in shards),
                "peak_gpu_used_mib": max(item["peak_gpu_used_mib"] for item in shards)}
        if complete:
            records = [json.loads(line) for rank in range(4) for line in
                       (root / "generation" / f"seqs_{concurrency}" / f"shard_{rank}" / "responses.jsonl").read_text().splitlines()]
            grades = []
            with ThreadPoolExecutor(max_workers=8) as pool:
                pending = [pool.submit(grade_response, root, plan, samples[record["id"]], record, concurrency) for record in records]
                for future in as_completed(pending):
                    grades.append(future.result())
                    persist(root, plan, "status.json", {"state": "grading", "max_num_seqs": concurrency,
                                                       "graded": len(grades), "total": 200, "updated_at": now()})
            mode["accuracy"] = {}
            for level in ("EASY", "MEDIUM", "ALL"):
                selected = [row for row in grades if level == "ALL" or row["difficulty"] == level]
                official = compute_metrics({row["id"]: [row["results"]] for row in selected}, k_list=[1])
                mode["accuracy"][level] = {"questions": len(selected), "correct": sum(row["correct"] for row in selected),
                                            "pass_at_1_percent": float(official["pass@1"]) * 100,
                                            "missing_thinking_close": sum(row["extraction"] == "missing_thinking_close" for row in selected),
                                            "grader_exceptions": sum(row["grader_exception"] is not None for row in selected)}
            mode["generation_gpu_seconds"] = sum(item["generation_seconds"] for item in shards)
            mode["output_tokens"] = sum(item["output_tokens"] for item in shards)
            mode["tokens_per_gpu_second"] = mode["output_tokens"] / mode["generation_gpu_seconds"]
            mode["preemptions"] = sum(item["preemptions"] for item in shards)
        modes[str(concurrency)] = mode
    if not modes["16"]["complete"]:
        raise RuntimeError("Concurrency 16 failed; cannot claim the requested 200-question evaluation is complete")
    report = {"model": plan["model"], "dataset": plan["dataset"], "sampling": plan["sampling"],
              "n": 1, "grading": "unmodified official TACO, code after </think>, no additional EOS requirement",
              "modes": modes, "primary_max_num_seqs": 32 if modes["32"]["complete"] else 16,
              "scope": "Inference only; does not establish actor/optimizer training memory capacity.",
              "finished_at": now()}
    if (root / "data_quality_note.json").exists():
        report["data_quality_note"] = read(root / "data_quality_note.json")
    if modes["32"]["complete"]:
        report["throughput_ratio_32_over_16"] = modes["32"]["tokens_per_gpu_second"] / modes["16"]["tokens_per_gpu_second"]
        report["generation_time_ratio_32_over_16"] = modes["32"]["generation_gpu_seconds"] / modes["16"]["generation_gpu_seconds"]
    persist(root, plan, "report/metrics.json", report)
    lines = [f"# Qwen3-1.7B: TACO {plan['dataset']['split']} Easy 100 + Medium 100", "", report["grading"], "",
             "One sample per question per condition; conditions are never pooled into pass@2.", "",
             "| max_num_seqs | Easy pass@1 | Medium pass@1 | Overall pass@1 | tokens/GPU-second | Peak GPU MiB | OOM |",
             "|---:|---:|---:|---:|---:|---:|---|"]
    for key, mode in modes.items():
        if mode["complete"]:
            scores = [mode["accuracy"][level]["pass_at_1_percent"] for level in ("EASY", "MEDIUM", "ALL")]
            lines.append(f"| {key} | {scores[0]:.2f}% | {scores[1]:.2f}% | {scores[2]:.2f}% | "
                         f"{mode['tokens_per_gpu_second']:.2f} | {mode['peak_gpu_used_mib']:.0f} | No |")
        else:
            lines.append(f"| {key} | incomplete | incomplete | incomplete | — | {mode['peak_gpu_used_mib']:.0f} | {mode['oom']} |")
    lines += ["", report["scope"], "", "Generation timing excludes engine initialization, warmup and grading.",
              "Each physical GPU runs the same 50 questions at both settings; order is counterbalanced.",
              "The official checker is unchanged; no questions were filtered by model outcomes."]
    if plan["dataset"].get("exclude_spj"):
        lines += ["SPJ-tagged problems were excluded using the pinned official test-set annotations before sampling "
                  "100 Easy and 100 Medium questions. Both concurrency conditions use the same retained questions."]
    if plan["dataset"]["split"] == "train":
        lines += ["Questions are sampled from the original training split to compare training difficulty. "
                  "No SPJ filtering is applied; the official SPJ annotations cover only the test split. "
                  "These are training-pool diagnostics, not held-out TACO test benchmark scores."]
        if plan["dataset"].get("test_case_filter"):
            lines += ["Only records lacking nonempty, paired input/output test lists are excluded before sampling; "
                      "question text, expected answers, SPJ problems and reference/model outcomes are not used to filter. "
                      "See structural_test_audit.json for excluded original row indices and reasons."]
    if "data_quality_note" in report:
        note = report["data_quality_note"]
        lines += ["", f"Data check: official reference solution 0 for {note['question_id']} passes "
                  f"{note['passed_tests']}/{note['total_tests']} attached tests. One recorded input declares five "
                  "cases while its expected output has four lines. Original tests and sample selection are unchanged; "
                  "see data_quality_note.json."]
        if note.get("included_in_current_sample") is False:
            lines += ["That data issue was found during the initial preflight; its question is not in the current sample."]
    (root / "report/README.md").write_text("\n".join(lines) + "\n")
    artifacts = {str(path.relative_to(root)): digest(path) for folder in ("generation", "grades")
                 for path in (root / folder).rglob("*") if path.name in ("summary.json", "responses.jsonl") or path.parent.parent.name == "grades"}
    persist(root, plan, "report/audit.json", {"complete": True, "plan_sha256": digest(root / "plan.json"),
                                             "metrics_sha256": digest(root / "report/metrics.json"),
                                             "artifacts_sha256": artifacts, "questions_per_condition": 200})


def complete(root):
    if not (root / "report/audit.json").exists() or not (root / "status.json").exists():
        return False
    audit = read(root / "report/audit.json")
    return (read(root / "status.json").get("state") == "complete" and audit.get("complete") is True
            and audit["plan_sha256"] == digest(root / "plan.json")
            and audit["metrics_sha256"] == digest(root / "report/metrics.json")
            and all(digest(root / path) == checksum for path, checksum in audit.get("artifacts_sha256", {}).items()))


def run(root):
    plan = verify(root)
    from qwen3_experiments.compression_budget_followup import dependency_ready

    if not dependency_ready(plan):
        return 75
    with ExitStack() as stack:
        try:
            for path in [root / "run.lock", *map(Path, plan["holder_locks"])]:
                lock = stack.enter_context(path.open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        if busy.strip():
            return 75
        for name, checksum in plan["model"]["files_sha256"].items():
            if digest(Path(plan["model"]["path"]) / name) != checksum:
                raise ValueError(f"Changed model file: {name}")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
        if len(visible) != 8:
            raise ValueError("Expected the full eight-GPU allocation")
        children = []
        try:
            for round_number in range(2):
                active = []
                for rank in range(4):
                    concurrency = concurrency_for(round_number, rank)
                    if worker_complete(root, concurrency, rank):
                        continue
                    log = stack.enter_context((root / f"worker_round{round_number}_rank{rank}.log").open("ab", buffering=0))
                    child = subprocess.Popen([plan["python_bin"], "-u", "-m", MODULE, "worker", "--output-root", str(root),
                                              "--rank", str(rank), "--concurrency", str(concurrency)],
                                             cwd=plan["runtime"], env={**os.environ, "CUDA_VISIBLE_DEVICES": visible[rank]},
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active.append(child)
                    children.append(child)
                while any(child.poll() is None for child in active):
                    failed = [child.returncode for child in active if child.poll() not in (None, 0)]
                    if failed:
                        raise RuntimeError(f"TACO generation worker exited: {failed}")
                    persist(root, plan, "status.json", {"state": "generating", "round": round_number,
                                                       "worker_pids": [child.pid for child in active if child.poll() is None],
                                                       "updated_at": now()})
                    time.sleep(15)
                if any(child.returncode for child in active):
                    raise RuntimeError("TACO generation worker failed")
            build_report(root, plan)
            persist(root, plan, "status.json", {"state": "complete", "finished_at": now()})
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            for child in children:
                if child.poll() is None:
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
    return 0


def queue(root):
    plan = verify(root)
    from qwen3_experiments import compression_budget_followup as followup
    from qwen3_experiments import compute_disk_watchdog as storage
    from qwen3_experiments import minerva_individual_budget as mini

    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                verify(root)
                state = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": plan["job_id"], "updated_at": now()}
                if complete(root):
                    persist(root, plan, "queue_status.json", {**state, "state": "complete"})
                    return 0
                if not followup.dependency_ready(plan):
                    persist(root, plan, "queue_status.json", {**state, "state": "waiting_for_er_final", "predecessor": plan["dependency"]["root"]})
                else:
                    active = []
                    for pid in storage.allocation_processes(plan):
                        item = storage.identity(pid)
                        if (item and item["uid"] == os.getuid() and MODULE in item["command"]
                                and str(root) in item["command"] and any(x in item["command"] for x in ("run", "worker"))):
                            active.append(pid)
                    if not active:
                        command = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                                   f"--nodelist={plan['node']}", "--cpus-per-task=96", "--gres=gpu:8", "--kill-on-bad-exit=1",
                                   "--job-name=taco-priority", plan["python_bin"], "-u", "-m", MODULE, "run", "--output-root", str(root)]
                        env = mini.environment(plan)
                        env["PYTHONPATH"] = plan["runtime"] + os.pathsep + plan["budget_runtime"]
                        env["HF_HUB_OFFLINE"] = "1"
                        with (root / "launch.log").open("ab", buffering=0) as log:
                            child = subprocess.Popen(command, cwd=plan["runtime"], env=env, stdin=subprocess.DEVNULL,
                                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        active = [child.pid]
                        persist(root, plan, "execution_launch.json", {"pid": child.pid, "command": command, "launched_at": now()})
                    persist(root, plan, "queue_status.json", {**state, "state": "running_or_waiting_for_gpus", "active_pids": active})
            except Exception as exc:
                traceback.print_exc()
                persist(root, plan, "queue_status.json", {"pid": os.getpid(), "state": "retrying", "error": str(exc), "updated_at": now()})
            time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("queue", "run", "worker"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(4))
    parser.add_argument("--concurrency", type=int, choices=(16, 32))
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.command == "worker":
        if args.rank is None or args.concurrency is None:
            parser.error("Worker requires rank and concurrency")
        worker(root, args.rank, args.concurrency)
        return 0
    return queue(root) if args.command == "queue" else run(root)


if __name__ == "__main__":
    raise SystemExit(main())
