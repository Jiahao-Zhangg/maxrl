"""Run nine benchmarks, then eight-GPU Minerva budgets after verified ER training."""

import argparse
import fcntl
import gzip
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

from qwen3_experiments import eval_l0_final as nine
from qwen3_experiments import minerva_individual_budget as mini
from qwen3_experiments.grpo_compute_control import digest, now, read, require_compute, write
from qwen3_experiments.math_eval_matrix_common import fingerprint

MODULE = "qwen3_experiments.er_compression_evaluation"
MODEL_KEY = "er_compression_step100"
MODEL_LABEL = "Compression ER step 100"
SHARDS = 8


def immutable_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen evaluation identity changed: {path}")
    else:
        write(path, value)


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f"Prepared source changed: {target}")
    else:
        shutil.copy2(source, target)


def verify_plan(root):
    plan = read(root / "plan.json")
    if digest(Path(plan["training_root"]) / "plan.json") != plan["training_plan_sha256"]:
        raise ValueError("The running ER training plan changed")
    launch = root / "launch.json"
    if launch.exists() and read(launch)["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Evaluation plan changed after launch")
    for relative, checksum in plan["frozen_files"].items():
        if digest(root / relative) != checksum:
            raise ValueError(f"Frozen evaluation input changed: {relative}")
    return plan


def validate_archive(receipt, repo, path):
    if (receipt["repo_id"] != repo or receipt["path"] != path
            or not re.fullmatch(r"[0-9a-f]{40}", receipt["revision"])):
        raise ValueError("Wrong ER checkpoint archive identity")
    if not receipt["files"] or receipt["files"].keys() != receipt["sha256"].keys():
        raise ValueError("Incomplete ER checkpoint archive manifest")
    for name, size in receipt["files"].items():
        if (Path(name).is_absolute() or ".." in Path(name).parts or size <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"][name])):
            raise ValueError("Invalid ER archive file metadata")


def training_ready(plan):
    root = Path(plan["training_root"])
    state = read(root / "status.json")
    if state.get("state") == "failed" or state.get("exit_code") not in (None, 0):
        raise RuntimeError("ER training or archival failed; evaluation will not start")
    if state.get("state") != "complete":
        return None
    expected = {"exit_code": 0, "completed_rollout_steps": 100,
                "optimizer_updates": 200, "saved_training_rollouts": 25600}
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("ER completion is missing the complete training/rollout audit")
    training = read(root / "plan.json")
    for step in (20, 40, 60, 80, 100):
        validate_archive(read(root / "archive_receipts" / f"global_step{step}.json"),
                         training["hf_repo_prefix"] + f"-step_{step}", f"global_step_{step}/actor")
    receipt = read(root / "archive_receipts/final_model.json")
    validate_archive(receipt, plan["model_repo"], "")
    if (receipt.get("local_path") != "final_model" or "config.json" not in receipt["files"]
            or not any(name.endswith(".safetensors") for name in receipt["files"])):
        raise ValueError("Final ER export is not a complete Hugging Face model")
    rollouts = read(root / "rollout_upload.json")
    if (rollouts.get("state") != "verified" or rollouts.get("num_rollouts") != 25600
            or rollouts.get("num_steps") != 100 or rollouts.get("repo_id") != training["rollout_hf_repo"]):
        raise ValueError("All ER training rollouts must be archived before evaluation")
    return receipt


def nine_complete(root):
    if not (root / "status.json").exists() or read(root / "status.json").get("state") != "complete":
        return False
    audit = read(root / "report/audit.json")
    if (not audit.get("complete") or audit.get("questions") != 1819
            or audit.get("responses_verified") != 7276 or audit.get("budget_points") != 54):
        raise ValueError("Incomplete nine-benchmark audit")
    for name, key in (("metrics.json", "metrics_sha256"), ("per_sample.json", "per_sample_sha256")):
        if digest(root / "report" / name) != audit[key]:
            raise ValueError(f"Changed nine-benchmark results: {name}")
    return True


def environment(plan):
    env = mini.environment(plan)
    for key in list(env):
        if key.startswith("ER_") or key == "STEP80_LOCAL_MODEL":
            env.pop(key)
    return env


def prepare(args):
    source = Path(__file__).resolve().parents[1]
    training_root = args.training_root.resolve()
    training = read(training_root / "plan.json")
    node = require_compute(training["job_id"])
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=source, text=True).strip() != "agent/add-math12k-maxrl-launcher":
        raise RuntimeError("Use the primary maxrl launcher branch")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists():
            plan = verify_plan(root)
            if plan["training_root"] != str(training_root):
                raise ValueError("Evaluation belongs to another training run")
            return plan
        reference_minerva = Path(training["predecessor_root"])
        minerva_reference = read(reference_minerva / "plan.json")
        reference = Path(minerva_reference["parent_eval_root"])
        if not nine_complete(reference):
            raise ValueError("The L+0 reference evaluation must already be complete")
        reference_plan = read(reference / "plan.json")
        for relative, checksum in reference_plan["frozen_files"].items():
            if digest(reference / relative) != checksum:
                raise ValueError(f"Changed reference evaluation input: {relative}")
        questions = read(reference / "questions.json")
        if Counter(q["dataset"] for q in questions) != {key: count for key, _, count in nine.DATASETS}:
            raise ValueError("Expected the exact nine benchmark inventories")
        input_plan = read(reference / "input_plan.json")
        if digest(reference / "questions.json") != input_plan["questions_sha256"]:
            raise ValueError("Reference questions failed checksum verification")
        runtime = root / "runtime"
        for name in ("er_compression_evaluation.py", "eval_l0_final.py", "eval_polaris_step80.py",
                     "minerva_individual_budget.py", "math_eval_budget_engine.py", "math_eval_matrix_common.py",
                     "grpo_compute_control.py"):
            copy_file(source / "qwen3_experiments" / name, runtime / "qwen3_experiments" / name)
        repo = training["hf_repo_prefix"] + "-final"
        nroot, mroot = root / "nine_datasets", root / "minerva_individual_budget_seed0"
        for name in ("questions.json", "provenance/main_baseline.csv", "provenance/budget_baseline.csv",
                     "provenance/eval_rloo_final.py"):
            copy_file(reference / name, nroot / name)
        for name in ("eval_l0_final.py", "eval_polaris_step80.py"):
            copy_file(source / "qwen3_experiments" / name, nroot / "provenance" / name)
        for name in ("source_report.md", "main.csv", "budgets.csv", "audit.json"):
            copy_file(reference / "report/qwen3_after_thinking" / name, nroot / "report/qwen3_after_thinking" / name)
        nine.after_thinking_reference(nroot, nine.evaluator(nroot))
        input_plan["model"] = {"repo": repo, "revision": None}
        write(nroot / "input_plan.json", input_plan)
        nplan = {
            "training_root": str(training_root), "training_kind": "er", "model_label": "ER",
            "model_repo": repo, "model_format": "huggingface", "final_step": 100,
            "repository": str(source), "reference_model": training["model_path"],
            "job_id": training["job_id"], "queue_node": node, "num_gpus": SHARDS,
            "questions": 1819, "total_responses": 7276, "sampling": input_plan["sampling"],
            "sampling_seed": 42, "samples_per_question": 4, "caps": list(nine.CAPS),
            "merger_sha256": digest(source / "scripts/model_merger.py"),
            "frozen_files": {p.relative_to(nroot).as_posix(): digest(p) for p in nroot.rglob("*")
                             if p.is_file() and p != nroot / "plan.json"},
        }
        write(nroot / "plan.json", nplan)
        rows = [{"unique_id": q["id"], "ground_truth": q["gold"], "prompt": q["messages"],
                 "prompt_token_ids": q["prompt_token_ids"], "source_row": q["source_row"]}
                for q in questions if q["dataset"] == "minervamath"]
        if rows != read(reference_minerva / "questions.json"):
            raise ValueError("Minerva questions/order differ from the seed-0 baseline")
        write(mroot / "questions.json", rows)
        model = {"label": MODEL_LABEL, "repo": repo, "revision": None, "family": "qwen3", "questions_file": "questions.json"}
        mplan = {
            "job_id": training["job_id"], "node": node, "parent_eval_root": str(nroot),
            "parent_eval_plan_sha256": digest(nroot / "plan.json"), "models": {MODEL_KEY: model},
            "budgets": mini.BUDGETS, "num_questions": 272, "seed": 0,
            "sampling": minerva_reference["sampling"], "engine": minerva_reference["engine"],
            "grading": minerva_reference["grading"], "protocol": "eval2", "stop_on_first_success": True,
            "question_shards": SHARDS, "seed_position_policy": "Original full-dataset position, preserved across shards",
            "frozen_files": {"questions.json": digest(mroot / "questions.json")},
        }
        write(mroot / "plan.json", mplan)
        for rank in range(SHARDS):
            shard = mroot / "shards" / str(rank)
            write(shard / "questions.json", [{**row, "seed_position": index} for index, row in enumerate(rows) if index % SHARDS == rank])
            for name in ("eval_l0_final.py", "eval_polaris_step80.py"):
                copy_file(nroot / "provenance" / name, shard / "provenance" / name)
            shard_plan = {**mplan, "num_questions": len(rows[rank::SHARDS]), "budgets": sorted(mini.BUDGETS, reverse=True),
                          "frozen_files": {p.relative_to(shard).as_posix(): digest(p) for p in shard.rglob("*")
                                           if p.is_file() and p != shard / "plan.json"}}
            write(shard / "plan.json", shard_plan)
        scratch = str(Path("/tmp") / ("erpost" + str(training["job_id"])))
        plan = {
            "job_id": training["job_id"], "node": node, "training_root": str(training_root),
            "training_plan_sha256": digest(training_root / "plan.json"), "model_repo": repo,
            "runtime": str(runtime), "scratch": scratch, "python_bin": minerva_reference["python_bin"],
            "holder_locks": training["holder_locks"], "reference_evaluation": str(reference),
            "nine_root": str(nroot), "minerva_root": str(mroot), "created_at": now(),
            "frozen_files": {p.relative_to(root).as_posix(): digest(p) for folder in (runtime, nroot, mroot)
                             for p in folder.rglob("*") if p.is_file()},
        }
        write(root / "plan.json", plan)
        write(root / "validation.json", {"state": "prepared", "host": node, "questions": 1819,
                                         "nine_responses": 7276, "minerva_questions": 272, "minerva_budgets": mini.BUDGETS,
                                         "gpu_question_shards": SHARDS, "preserves_original_question_seeds": True,
                                         "training_plan_unchanged": True, "after_thinking_only": True})
        (root / "README.md").write_text(
            "# ER compression: automatic evaluation after training\n\n"
            "Wait for successful 100-step ER training, all checkpoint archives and all 25,600 uploaded rollouts. "
            "Pin the final Hugging Face export to its verified upload revision.\n\n"
            "1. Nine benchmarks: 1,819 questions, four responses each, 32,768 tokens, seed 42, "
            "temperature 0.6 / top-p 0.95 / top-k 20. Save every response and regrade exact "
            "1k/2k/4k/8k/16k/32k prefixes. Main table uses the same after-thinking grading.\n"
            "2. Minerva Math: 272 questions, seed 0, budgets 8k/16k/32k/48k/64k, the same sampling, "
            "at most 32k per attempt, stop on first success. Eight GPUs process disjoint question shards "
            "with the original per-question seeds; unused budgets stay with their questions.\n\n"
            "Both evaluations grade only the nonempty answer after completed thinking; unfinished/reopened thinking "
            "scores zero, with no extra EOS/boxed requirement. ER training keeps its full-text reward policy.\n\n"
            "[Queue status](queue_status.json) · [Nine datasets](nine_datasets/report/README.md) · "
            "[Minerva pass@budget](minerva_individual_budget_seed0/report/README.md)\n")
        return plan


def prepare_model(root, plan, receipt):
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    nroot = Path(plan["nine_root"])
    immutable_json(nroot / "final_checkpoint_receipt.json", receipt)
    model_path = Path(plan["scratch"]) / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    destination = nroot / "model"
    if not destination.exists():
        destination.symlink_to(model_path, target_is_directory=True)
    if destination.resolve() != model_path.resolve():
        raise ValueError("Unexpected final model directory")
    info = HfApi().model_info(receipt["repo_id"], revision=receipt["revision"], files_metadata=True)
    if info.private or info.sha != receipt["revision"]:
        raise ValueError("Expected the verified public ER final-model revision")
    remote = {entry.rfilename: entry for entry in info.siblings}

    def download(name):
        path = model_path / name
        if not path.is_file() or digest(path) != receipt["sha256"][name]:
            path = Path(hf_hub_download(receipt["repo_id"], name, revision=receipt["revision"], local_dir=model_path))
        if path.stat().st_size != receipt["files"][name] or remote[name].size != receipt["files"][name]:
            raise ValueError(f"Final export file size mismatch: {name}")
        if digest(path) != receipt["sha256"][name]:
            raise ValueError(f"Final export checksum mismatch: {name}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, receipt["files"]))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    inputs = read(nroot / "input_plan.json")
    if fingerprint(tokenizer.chat_template) != inputs["chat_template_sha256"]:
        raise ValueError("ER tokenizer changed the native thinking template")
    for q in read(nroot / "questions.json"):
        if tokenizer.apply_chat_template(q["messages"], add_generation_prompt=True, enable_thinking=True) != q["prompt_token_ids"]:
            raise ValueError("ER tokenizer changed a frozen evaluation prompt")
    if read(model_path / "generation_config.json")["eos_token_id"] != [151645, 151643]:
        raise ValueError("ER model EOS configuration changed")
    if read(model_path / "config.json")["max_position_embeddings"] < 40960:
        raise ValueError("ER model context is too short")
    merged = {"repo": receipt["repo_id"], "revision": receipt["revision"],
              "format": "Native Hugging Face BF16 final export after 100 rollouts / 200 optimizer updates",
              "merged_files": receipt["sha256"], "archive_receipt_sha256": digest(nroot / "final_checkpoint_receipt.json")}
    immutable_json(nroot / "model_receipt.json", merged)
    inputs["model"]["revision"] = receipt["revision"]
    inputs["plan_sha256"] = digest(nroot / "plan.json")
    inputs["checkpoint_receipt_sha256"] = digest(nroot / "final_checkpoint_receipt.json")
    immutable_json(nroot / "prepared_inputs.json", inputs)
    return merged


def prepare_minerva(plan):
    nroot, mroot = Path(plan["nine_root"]), Path(plan["minerva_root"])
    merged = read(nroot / "model_receipt.json")
    if read(nroot / "manifest.json")["model"] != merged:
        raise ValueError("Minerva model does not match the nine-benchmark model")
    for folder in [mroot, *(mroot / "shards" / str(rank) for rank in range(SHARDS))]:
        p = read(folder / "plan.json")
        model = {**p["models"][MODEL_KEY], "path": str((nroot / "model").resolve()),
                 "revision": merged["revision"], "files_sha256": merged["merged_files"]}
        manifest = {"plan_sha256": digest(folder / "plan.json"), "models": {MODEL_KEY: model},
                    "final_checkpoint_receipt_sha256": digest(nroot / "final_checkpoint_receipt.json")}
        manifest["fingerprint"] = fingerprint(manifest)
        immutable_json(folder / "execution_manifest.json", manifest)


def merged_records(shards, budget, model_key=MODEL_KEY):
    sequence = 0
    for rank, folder in enumerate(shards):
        rows = read(folder / "questions.json")
        manifest = read(folder / "execution_manifest.json")
        summary = mini.completed_point(folder, model_key, budget, manifest)
        if summary is None:
            raise ValueError(f"Missing Minerva shard {rank}, budget {budget}")
        directory = mini.point_directory(folder, model_key, budget)
        path = directory / summary["artifacts"]["rollouts"]["file"]
        for record in mini.point_records(path):
            yield {**record, "prompt_position": rows[record["prompt_position"]]["seed_position"],
                   "request_sequence_index": sequence}
            sequence += 1


def finish_minerva(root):
    plan, manifest, rows = read(root / "plan.json"), read(root / "execution_manifest.json"), read(root / "questions.json")
    if len(plan["models"]) != 1:
        raise ValueError("Each sharded evaluation must contain exactly one model")
    model_key = next(iter(plan["models"]))
    shards = [root / "shards" / str(rank) for rank in range(SHARDS)]
    for budget in plan["budgets"]:
        directory = mini.point_directory(root, model_key, budget)
        directory.mkdir(parents=True, exist_ok=True)
        rollouts = directory / "rollouts.jsonl.gz"
        temporary = directory / "rollouts.jsonl.gz.tmp"
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as stream:
            for record in merged_records(shards, budget, model_key):
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        summary, prompts = mini.audit_records(mini.point_records(temporary), protocol="eval2", budget=budget,
                                              seed=0, rows=rows, per_rollout_cap=mini.RESPONSE_CAP,
                                              stop_on_first_success=True)
        temporary.replace(rollouts)
        prompt_path = directory / "prompts.json"
        write(prompt_path, prompts)
        summary.update({"state": "complete", "identity": {"manifest": manifest["fingerprint"], "model": model_key, "budget": budget},
                        "seed": 0, "finished_at": now(), "pass_at_budget_percent": 100 * summary["fraction_solved"],
                        "ledger_audit": "passed", "question_shards": SHARDS,
                        "artifacts": {name: {"file": p.name, "size": p.stat().st_size, "sha256": digest(p)}
                                      for name, p in (("rollouts", rollouts), ("prompts", prompt_path))}})
        write(directory / "summary.json", summary)
    mini.report(root, plan, manifest)
    write(root / "status.json", {"state": "complete", "points": 5, "finished_at": now()})


def run_minerva(plan, *, prepared=False):
    root = Path(plan["minerva_root"])
    if not prepared:
        prepare_minerva(plan)
    models = read(root / "plan.json")["models"]
    if len(models) != 1:
        raise ValueError("Each sharded evaluation must contain exactly one model")
    model_key = next(iter(models))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if len(visible) != SHARDS:
        raise ValueError("Expected all eight GPUs")
    children = []
    with ExitStack() as stack:
        try:
            for rank, gpu in enumerate(visible):
                shard = root / "shards" / str(rank)
                log = stack.enter_context((shard / "worker.log").open("ab", buffering=0))
                child = subprocess.Popen(
                    [plan["python_bin"], "-u", "-m", mini.MODULE, "worker", "--output-root", str(shard),
                     "--model", model_key, "--rank", str(rank)], cwd=plan["runtime"],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None, 0) for c in children):
                    raise RuntimeError("An ER Minerva evaluation worker failed")
                completed = sum((mini.point_directory(root / "shards" / str(rank), model_key, budget) / "summary.json").exists()
                                for rank in range(SHARDS) for budget in mini.BUDGETS)
                write(root / "status.json", {"state": "running", "completed_shards": completed,
                                             "total_shards": 40, "worker_pids": [c.pid for c in children], "updated_at": now()})
                time.sleep(15)
            if any(c.returncode != 0 for c in children):
                raise RuntimeError("An ER Minerva evaluation worker failed")
            finish_minerva(root)
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            for child in children:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def run_phase(root, plan, phase):
    require_compute(plan["job_id"])
    receipt = training_ready(plan)
    if receipt is None:
        return 75
    with ExitStack() as stack:
        try:
            for path in [root / "run.lock", *map(Path, plan["holder_locks"])]:
                path.parent.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context(path.open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        if busy.strip():
            return 75
        import torch

        if torch.cuda.device_count() != SHARDS:
            raise RuntimeError("Expected eight allocated GPUs")
        nroot = Path(plan["nine_root"])
        prepare_model(root, plan, receipt)
        if phase == "nine":
            if not nine_complete(nroot):
                base = nine.evaluator(nroot)
                shared = nine.load_module("er_shared_nine_runner", nroot / "provenance/eval_rloo_final.py")
                shared.report = nine.generation_complete
                shared.run(nroot, base)
                nine.regrade(nroot, base)
            if not nine_complete(nroot):
                raise RuntimeError("Nine-benchmark evaluation did not complete")
        else:
            if not nine_complete(nroot):
                raise RuntimeError("Minerva must wait for the complete nine-benchmark evaluation")
            run_minerva(plan)
    return 0


def queue(root, plan):
    require_compute(plan["job_id"])
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": plan["job_id"], "started_at": now()}

        def update(state_name, **values):
            state.update(state=state_name, **values, updated_at=now())
            write(root / "queue_status.json", state)

        try:
            while training_ready(plan) is None:
                require_compute(plan["job_id"])
                update("waiting_for_er_training_and_archives")
                time.sleep(30)
            verify_plan(root)
            for phase in ("nine", "minerva"):
                failures = 0
                while True:
                    require_compute(plan["job_id"])
                    update("running_or_waiting_for_gpus", phase=phase, attempts=failures + 1)
                    command = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                               f"--nodelist={plan['node']}", "--cpus-per-task=96", "--gres=gpu:8", "--kill-on-bad-exit=1",
                               f"--job-name=er-{phase}-eval", plan["python_bin"], "-u", "-m", MODULE,
                               "run", "--output-root", str(root), "--phase", phase]
                    with (root / f"{phase}.log").open("ab", buffering=0) as log:
                        child = subprocess.Popen(command, cwd=plan["runtime"], env=environment(plan),
                                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                        update("running_or_waiting_for_gpus", phase=phase, launcher_pid=child.pid)
                        while child.poll() is None:
                            update("running_or_waiting_for_gpus", phase=phase)
                            time.sleep(15)
                    if child.returncode == 0:
                        if phase == "nine":
                            if not nine_complete(Path(plan["nine_root"])):
                                raise ValueError("Nine-dataset child exited without complete results")
                        else:
                            audit = read(Path(plan["minerva_root"]) / "report/audit.json")
                            if not audit.get("complete") or audit.get("points") != 5 or not audit.get("all_rollout_ledgers_verified"):
                                raise ValueError("Minerva child exited without five audited budgets")
                        break
                    if child.returncode != 75:
                        failures += 1
                    if failures >= 3:
                        raise RuntimeError(f"ER {phase} evaluation failed three times; saved responses are preserved")
                    update("retrying", phase=phase, last_exit_code=child.returncode)
                    time.sleep(30)
            update("complete", finished_at=now(), nine_responses=7276, minerva_points=5)
        except BaseException as exc:
            update("failed", error=str(exc), finished_at=now())
            raise


def launch(root, plan):
    require_compute(plan["job_id"])
    for name in ("tmp", "triton"):
        (Path(plan["scratch"]) / name).mkdir(parents=True, exist_ok=True)
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            raise RuntimeError("ER follow-up queue already launched; inspect queue_status.json")
        with (root / "queue.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([plan["python_bin"], "-u", "-m", MODULE, "queue", "--output-root", str(root)],
                                     cwd=plan["runtime"], env=environment(plan), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"pid": child.pid, "hostname": socket.gethostname(), "job_id": plan["job_id"],
                   "plan_sha256": digest(root / "plan.json"), "launched_at": now()}
        write(root / "launch.json", receipt)
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "queue", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--phase", choices=("nine", "minerva"))
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.command == "prepare":
        if args.training_root is None:
            parser.error("prepare requires --training-root")
        prepare(args)
    else:
        plan = verify_plan(root)
        if args.command == "launch":
            launch(root, plan)
        elif args.command == "queue":
            queue(root, plan)
        else:
            if args.phase is None:
                parser.error("run requires --phase")
            sys.exit(run_phase(root, plan, args.phase))


if __name__ == "__main__":
    main()
