"""Compute-node queue for individual budgets after the nine-benchmark evaluation."""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import csv
import fcntl
from functools import partial
import gzip
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

# The existing budget engine also supports direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grpo_compute_control import digest, now, read, require_compute, write
from math_eval_budget_engine import audit_records, evaluate_point
from math_eval_matrix_common import fingerprint

MODULE = "qwen3_experiments.minerva_individual_budget"
BUDGETS = [8192, 16384, 32768, 49152, 65536]
RESPONSE_CAP = 32768
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATASETS = {
    "minervamath": ("Minerva Math", 272),
    "math500": ("MATH500", 500),
    "olympiadbench": ("OlympiadBench", 674),
    "amc22_23": ("AMC22+23", 83),
    "aime24": ("AIME24", 30),
    "aime25": ("AIME25", 30),
    "aime26": ("AIME26", 30),
}
DEEPSEEK_MODELS = [
    {"key": "deepseek_r1_1_5b", "label": "DeepSeek-R1-Distill-Qwen-1.5B",
     "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
     "revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562", "format": "huggingface"},
    {"key": "deepseek_er_cost_step100", "label": "DeepSeek ER cost step 100",
     "repo": "zjhhhh/er_cost_marginrl_r1_distill_1.5b_compression_n16_b512_32k_lr1e-6_kl0_seed42-step_100",
     "revision": "595b113264edb620d1a091cfc29b31984584fc6d", "format": "fsdp", "step": 100},
    {"key": "deepseek_er_extracted_step100", "label": "DeepSeek ER extracted step 100",
     "repo": "zjhhhh/er-r1-distill-1.5b-compression-n16-extracted-step_100",
     "revision": "8183d5b14fbbce488d3a2fd1891ed37af7135a84", "format": "deepspeed", "step": 100,
     "base_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
]
GRADING = GENERATION = None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def environment(plan):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("L0_", "GRPO_", "MAXRL_", "RAY_", "VLLM_", "SLURM_", "WANDB_")):
            env.pop(key)
    for key in ("PYTHONHOME", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        env.pop(key, None)
    env.update({
        "PATH": str(Path(plan["python_bin"]).parent) + os.pathsep + env.get("PATH", ""),
        "PYTHONPATH": plan["runtime"], "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "VLLM_USE_V1": "0", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TMPDIR": str(Path(plan["scratch"]) / "tmp"), "TRITON_CACHE_DIR": str(Path(plan["scratch"]) / "triton"),
    })
    return env


def verify_plan(root):
    plan = read(root / "plan.json")
    receipt = root / "launch.json"
    if receipt.exists() and read(receipt)["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Queued plan changed after launch")
    for relative, checksum in plan["frozen_files"].items():
        if digest(root / relative) != checksum:
            raise ValueError(f"Prepared evaluation input changed: {relative}")
    if digest(Path(plan["parent_eval_root"]) / "plan.json") != plan["parent_eval_plan_sha256"]:
        raise ValueError("Predecessor nine-benchmark plan changed")
    return plan


def dataset_specs(plan):
    return plan.get("datasets", [plan.get("dataset", {"key": "minervamath", "rows": 272})])


def evaluation_points(plan, largest_first=False):
    # Keep the original on-disk layout readable for frozen single-dataset plans.
    datasets = [spec["key"] for spec in plan["datasets"]] if "datasets" in plan else [None]
    return [(key, budget, dataset) for budget in sorted(plan["budgets"], reverse=largest_first)
            for dataset in datasets for key in plan["models"]]


def selected_rows(plan, rows, dataset):
    if "datasets" in plan:
        specs = {spec["key"]: spec for spec in plan["datasets"]}
        if dataset not in specs:
            raise ValueError("Worker needs one dataset from its frozen plan")
        result = [row for row in rows if row.get("dataset") == dataset]
        expected = specs[dataset]["rows"]
    else:
        result, expected = rows, plan.get("num_questions", 272)
    if len(result) != expected or len({row["unique_id"] for row in result}) != expected:
        raise ValueError("Incomplete or duplicate questions in a dataset")
    # Filtering preserves each dataset's original question order and rollout seeds.
    return result


def thinking_rows(rows, tokenizer, family):
    """Keep identical user messages but freeze each model's native thinking IDs."""
    result = []
    for row in rows:
        rendered = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True,
                                                 enable_thinking=True, tokenize=False)
        ending = {"qwen3": "<|im_start|>assistant\n", "deepseek": "<｜Assistant｜><think>\n"}[family]
        if not rendered.endswith(ending):
            raise ValueError(f"Unexpected {family} thinking prompt")
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if tokenizer.decode(ids, skip_special_tokens=False) != rendered or len(ids) + RESPONSE_CAP > 40960:
            raise ValueError("Native prompt round trip failed or context overflow")
        if family == "qwen3" and ids != row["prompt_token_ids"]:
            raise ValueError("Qwen3 prompt changed from the frozen nine-dataset evaluation")
        result.append({**row, "prompt_token_ids": ids})
    return result


def prepare_deepseek(root, scratch, rows):
    from transformers import AutoTokenizer
    from prepare_math_eval_matrix import prepare_model

    models, audit = {}, {}
    base = None
    for spec in DEEPSEEK_MODELS:
        print(f"Preparing {spec['repo']} at {spec['revision']}", flush=True)
        receipt = prepare_model(spec, scratch / "prepared", base)
        path = Path(receipt["path"])
        if spec["format"] == "huggingface":
            base = path
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        probe = "reasoning</think>42"
        if tokenizer.decode(tokenizer.encode(probe, add_special_tokens=False), skip_special_tokens=True) != probe:
            raise ValueError("Response decoding would remove the after-thinking grading delimiter")
        prepared = thinking_rows(rows, tokenizer, "deepseek")
        relative = f"prompts/{spec['key']}.json"
        write(root / relative, prepared)
        write(root / "prepared_models" / f"{spec['key']}.json", receipt)
        models[spec["key"]] = {**spec, "path": str(path), "files_sha256": receipt["files_sha256"],
                               "family": "deepseek", "questions_file": relative}
        audit[spec["key"]] = {"questions": len(prepared), "native_thinking_prompts_match": True,
                              "longest_prompt_tokens": max(len(q["prompt_token_ids"]) for q in prepared),
                              "model_revision": spec["revision"], "tensor_count": receipt["tensor_count"],
                              "after_thinking_delimiter_preserved": True,
                              "chat_template_sha256": receipt["chat_template_sha256"]}
    if len({digest(root / m["questions_file"]) for m in models.values()}) != 1:
        raise ValueError("DeepSeek checkpoints disagree on native prompts or token IDs")
    return models, audit


def prepare(args):
    from transformers import AutoTokenizer

    parent = read(args.parent_run / "plan.json")
    node = require_compute(parent["job_id"])
    source = Path(__file__).resolve().parents[1]
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=source, text=True).strip()
    if branch != "agent/add-math12k-maxrl-launcher":
        raise RuntimeError("Use the primary maxrl launcher branch")
    root = args.output_root.resolve()
    checkpoint_only = args.checkpoint_only
    selected_datasets = getattr(args, "datasets", ["minervamath"])
    if (not selected_datasets or len(set(selected_datasets)) != len(selected_datasets)
            or any(key not in DATASETS for key in selected_datasets)):
        raise ValueError("Choose distinct supported datasets")
    expected_models = {"compression_step100"}
    if not checkpoint_only:
        expected_models.update(["qwen3_1_7b", *(spec["key"] for spec in DEEPSEEK_MODELS)])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists():
            plan = verify_plan(root)
            if plan["parent_run"] != str(args.parent_run.resolve()) or plan["seed"] != args.seed:
                raise ValueError("Existing directory belongs to a different experiment")
            if set(plan["models"]) != expected_models:
                raise ValueError("Existing model selection is immutable; prepare a distinct output directory")
            if [spec["key"] for spec in dataset_specs(plan)] != selected_datasets:
                raise ValueError("Existing dataset selection is immutable; prepare a distinct output directory")
            return plan
        previous = Path(parent["evaluation_root"])
        previous_plan = read(previous / "plan.json")
        if (Path(previous_plan["training_root"]).resolve() != args.parent_run.resolve()
                or previous_plan["model_repo"] != parent["hf_repo_prefix"] + "-step_100"
                or str(previous_plan["job_id"]) != str(parent["job_id"])):
            raise ValueError("Nine-benchmark plan belongs to a different training run")
        for relative, checksum in previous_plan["frozen_files"].items():
            if digest(previous / relative) != checksum:
                raise ValueError(f"Nine-benchmark input changed: {relative}")
        if parent["model_revision"] != MODEL_REVISION:
            raise ValueError("Expected the same pinned Qwen3-1.7B initial model")
        records = [q for q in read(previous / "questions.json") if q["dataset"] in selected_datasets]
        counts = {key: DATASETS[key][1] for key in selected_datasets}
        if Counter(q["dataset"] for q in records) != counts or len({q["id"] for q in records}) != len(records):
            raise ValueError("Expected every distinct frozen question in the requested datasets")
        tokenizer = AutoTokenizer.from_pretrained(parent["model_path"], local_files_only=True)
        for q in records:
            ids = tokenizer.apply_chat_template(q["messages"], add_generation_prompt=True, enable_thinking=True)
            if ids != q["prompt_token_ids"] or len(ids) + RESPONSE_CAP > 40960:
                raise ValueError("Thinking prompt mismatch or context overflow")
        rows = [{"unique_id": q["id"], "ground_truth": q["gold"], "prompt": q["messages"],
                 "prompt_token_ids": q["prompt_token_ids"], "source_row": q["source_row"],
                 "dataset": q["dataset"]} for q in records]
        write(root / "questions.json", rows)
        scripts = root / "runtime/qwen3_experiments"
        scripts.mkdir(parents=True, exist_ok=True)
        for name in ("minerva_individual_budget.py", "math_eval_budget_engine.py", "math_eval_matrix_common.py",
                     "grpo_compute_control.py"):
            shutil.copy2(source / "qwen3_experiments" / name, scripts / name)
        provenance = root / "provenance"
        provenance.mkdir(exist_ok=True)
        for name in ("eval_l0_final.py", "eval_polaris_step80.py"):
            shutil.copy2(previous / "provenance" / name, provenance / name)
        for relative in ("qwen3_experiments/prepare_math_eval_matrix.py", "scripts/model_merger.py"):
            shutil.copy2(source / relative, provenance / Path(relative).name)
        scratch = Path(f"/tmp/minervaib{parent['job_id']}seed{args.seed}_{fingerprint(str(root))[:12]}")
        scratch.mkdir(parents=True, exist_ok=True)
        deepseek, deepseek_audit = ({}, {}) if checkpoint_only else prepare_deepseek(root, scratch, rows)
        initial = Path(parent["model_path"])
        initial_hashes = {Path(p).name: checksum for p, checksum in parent["input_hashes"].items()
                          if Path(p).parent == initial}
        if not any(name.endswith(".safetensors") for name in initial_hashes):
            raise ValueError("Initial model has no pinned weight files")
        by_key = {spec["key"]: spec for spec in read(previous / "input_plan.json")["datasets"]}
        specs = [by_key[key] for key in selected_datasets]
        if any(spec["rows"] != counts[spec["key"]] for spec in specs):
            raise ValueError("Dataset manifest and frozen question counts disagree")
        plan = {
            "job_id": parent["job_id"], "node": node, "python_bin": parent["python_bin"],
            "parent_run": str(args.parent_run.resolve()), "parent_eval_root": str(previous),
            "parent_eval_plan_sha256": digest(previous / "plan.json"),
            "output_root": str(root), "runtime": str(root / "runtime"),
            "scratch": str(scratch), "holder_locks": parent["holder_locks"],
            "models": {
                "qwen3_1_7b": {"label": "Qwen3-1.7B", "repo": "Qwen/Qwen3-1.7B", "revision": MODEL_REVISION,
                               "path": str(initial), "files_sha256": initial_hashes,
                               "family": "qwen3", "questions_file": "questions.json"},
                "compression_step100": {"label": f"Compression {previous_plan.get('model_label', 'L+0')} step 100",
                                        "repo": previous_plan["model_repo"],
                                        "revision": None, "path": str(previous / "model"),
                                        "family": "qwen3", "questions_file": "questions.json"},
                **deepseek,
            },
            "datasets": specs, "num_questions": len(rows), "budgets": BUDGETS, "seed": args.seed,
            "protocol": "eval2", "stop_on_first_success": True,
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                         "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
                         "per_rollout_cap": RESPONSE_CAP, "max_batch_size": 32},
            "engine": {"dtype": "bfloat16", "tensor_parallel_size": 1, "max_model_len": 40960,
                       "max_num_seqs": 32, "max_num_batched_tokens": 8192, "gpu_memory_utilization": 0.85,
                       "enable_chunked_prefill": True, "enable_prefix_caching": True, "generation_config": "vllm"},
            "grading": {"after_thinking_only": True, "require_eos": False, "require_box": False,
                        "timeout_seconds": 1, "invalid_thinking_score": 0},
            "num_gpus": 8, "scheduling": "one GPU per dataset-model-budget point, largest budgets first", "created_at": now(),
            "frozen_files": {p.relative_to(root).as_posix(): digest(p)
                             for p in [root / "questions.json", *scripts.glob("*.py"), *provenance.glob("*.py"),
                                       *(root / "prompts").glob("*.json"), *(root / "prepared_models").glob("*.json")]},
        }
        if checkpoint_only:
            plan["models"] = {"compression_step100": plan["models"]["compression_step100"]}
        write(root / "plan.json", plan)
        write(root / "input_audit.json", {"questions": len(rows), "native_thinking_prompts_match": True,
                                         "longest_prompt_tokens": max(len(q["prompt_token_ids"]) for q in rows),
                                         "dataset_counts": counts, "checked_on": node,
                                         "dataset_revisions": {spec["key"]: spec["revision"] for spec in specs},
                                         "deepseek": deepseek_audit, "model_budget_points": len(evaluation_points(plan))})
        return plan


def dependency_ready(plan):
    """Only the successful, fully audited nine-benchmark run unlocks this queue."""
    parent, previous = Path(plan["parent_run"]), Path(plan["parent_eval_root"])
    paths = [parent / "status.json", previous / "queue_status.json", previous / "status.json", previous / "report/audit.json"]
    values = [read(p) if p.exists() else {} for p in paths]
    if any(v.get("state") in ("failed", "blocked_by_training_failure") for v in values):
        raise RuntimeError("The preceding training or nine-benchmark evaluation failed")
    training, queue, status, audit = values
    if not (training.get("state") == queue.get("state") == status.get("state") == "complete"):
        return False
    if (training.get("exit_code") != 0 or training.get("last_completed_step") != 100
            or not audit.get("complete") or audit.get("responses_verified") != 7276
            or audit.get("questions") != 1819 or audit.get("budget_points") != 54):
        raise ValueError("Predecessor completion does not have the expected nine-benchmark audit")
    receipt = read(previous / "final_checkpoint_receipt.json")
    if (receipt["repo_id"] != plan["models"]["compression_step100"]["repo"]
            or receipt["checkpoint"] != "global_step_100"
            or receipt["state"] not in ("verified", "archived_and_deleted")):
        raise ValueError("Predecessor evaluated a different checkpoint")
    return True


def prepare_models(root, plan):
    from transformers import AutoTokenizer

    previous = Path(plan["parent_eval_root"])
    receipt, merged = read(previous / "final_checkpoint_receipt.json"), read(previous / "model_receipt.json")
    if (merged["repo"] != receipt["repo_id"] or merged["revision"] != receipt["remote_commit"]
            or read(previous / "manifest.json")["model"] != merged):
        raise ValueError("Merged final model does not match the nine-benchmark model")
    for name, item in merged["source_files"].items():
        archived = receipt["files"][name.removeprefix("global_step_100/")]
        if item != {"size": archived["size"], "sha256": archived["sha256"]}:
            raise ValueError("Merged model sources differ from the verified checkpoint archive")
    models = {}
    for key, model in plan["models"].items():
        hashes = model.get("files_sha256", merged["merged_files"])
        target = Path(plan["scratch"]) / "models" / key
        target.mkdir(parents=True, exist_ok=True)
        for name, checksum in hashes.items():
            source, dest = Path(model["path"]) / name, target / name
            if digest(source) != checksum:
                raise ValueError(f"Pinned model file changed: {source}")
            if not dest.is_file() or digest(dest) != checksum:
                temporary = dest.with_name(dest.name + ".tmp")
                shutil.copy2(source, temporary)
                if digest(temporary) != checksum:
                    raise ValueError("Model staging checksum failed")
                temporary.replace(dest)
        models[key] = {**model, "path": str(target), "files_sha256": hashes,
                       "revision": model["revision"] or receipt["remote_commit"]}
        tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
        rows = read(root / model["questions_file"])
        if thinking_rows(rows, tokenizer, model["family"]) != rows:
            raise ValueError(f"Model tokenizer changed the frozen thinking prompt: {key}")
    manifest = {"plan_sha256": digest(root / "plan.json"), "models": models,
                "final_checkpoint_receipt_sha256": digest(previous / "final_checkpoint_receipt.json")}
    manifest["fingerprint"] = fingerprint(manifest)
    path = root / "execution_manifest.json"
    if path.exists() and read(path) != manifest:
        raise ValueError("Model identity changed between evaluation attempts")
    write(path, manifest)
    return manifest


def initialize_grader(root):
    global GRADING, GENERATION
    root = Path(root)
    GRADING = load_module("minerva_suffix_grading", root / "provenance/eval_l0_final.py")
    GENERATION = load_module("minerva_math_grading", root / "provenance/eval_polaris_step80.py")
    GENERATION.init_grader()


def after_thinking_score(text, gold, suffix_function, grade_function):
    suffix, status = suffix_function(text)
    if suffix is None:
        return {"score": 0.0, "suffix_status": status, "grader_status": status}
    result = grade_function((suffix, gold))
    if result["grader_status"] != "scored":
        raise RuntimeError(f"Math-Verify did not complete: {result['grader_status']}")
    return {"score": result["correct"], "suffix_status": status, "grader_status": "scored",
            "answer_suffix_sha256": fingerprint(suffix)}


def grade_one(item):
    return after_thinking_score(*item, GRADING.answer_suffix, GENERATION.grade)


def point_directory(root, key, budget, dataset=None):
    folder = root / "results"
    if dataset is not None:
        folder /= dataset
    return folder / key / f"budget_{budget}"


def point_identity(key, budget, manifest, dataset=None):
    identity = {"manifest": manifest["fingerprint"], "model": key, "budget": budget}
    if dataset is not None:
        identity["dataset"] = dataset
    return identity


def completed_point(root, key, budget, manifest, dataset=None):
    directory = point_directory(root, key, budget, dataset)
    path = directory / "summary.json"
    if not path.exists():
        return None
    value = read(path)
    if value.get("state") != "complete" or value.get("identity") != point_identity(key, budget, manifest, dataset):
        raise ValueError("Saved evaluation point has a different identity")
    for info in value["artifacts"].values():
        file = directory / info["file"]
        if file.stat().st_size != info["size"] or digest(file) != info["sha256"]:
            raise ValueError(f"Saved evaluation result changed: {file}")
    return value


def point_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def worker(root, key, rank, selected_budget=None, dataset=None):
    # Slurm can export AMD visibility variables even in an NVIDIA allocation.
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        os.environ.pop(name, None)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    plan, manifest = verify_plan(root), read(root / "execution_manifest.json")
    require_compute(plan["job_id"])
    protocol = plan.get("protocol", "eval2")
    if protocol not in ("eval2", "eval3"):
        raise ValueError("Budget worker supports individual or shared-context budgets")
    rows = selected_rows(plan, read(root / plan["models"][key]["questions_file"]), dataset)
    model_path = manifest["models"][key]["path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, clean_up_tokenization_spaces=False)
    eos = read(Path(model_path) / "generation_config.json")["eos_token_id"]
    if isinstance(eos, int):
        eos = [eos]
    params = partial(SamplingParams, stop_token_ids=eos, min_p=0.0, presence_penalty=0.0,
                     frequency_penalty=0.0, repetition_penalty=1.0)
    llm = None
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=context, initializer=initialize_grader,
                             initargs=(str(root),)) as pool:
        for budget in ([selected_budget] if selected_budget is not None else plan["budgets"]):
            directory = point_directory(root, key, budget, dataset)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "point.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if completed_point(root, key, budget, manifest, dataset):
                    continue
                state = {"pid": os.getpid(), "hostname": socket.gethostname(), "rank": rank,
                         "model": key, "dataset": dataset, "budget": budget, "seed": plan["seed"]}

                def progress(counters):
                    write(root / "progress" / f"worker_{rank}.json", {
                        **state, "state": "running", "updated_at": now(), **counters,
                    })

                progress({})
                if llm is None:
                    llm = LLM(model=model_path, tokenizer=model_path, seed=plan["seed"],
                              trust_remote_code=False, disable_log_stats=True, **plan["engine"])
                attempt = directory / ("attempt_" + uuid.uuid4().hex[:12])
                attempt.mkdir()
                rollouts, details = attempt / "rollouts.jsonl.gz", deque()
                started = time.time()

                def score_many(items):
                    results = list(pool.map(grade_one, items, chunksize=1))
                    details.extend(results)
                    return [v["score"] for v in results]

                with gzip.open(rollouts, "wt", encoding="utf-8", compresslevel=1) as stream:
                    def emit(record):
                        info = details.popleft()
                        if info["score"] != record["score"]:
                            raise ValueError("Grading and response records became misaligned")
                        record.update(info)
                        record["dataset"] = dataset or dataset_specs(plan)[0]["key"]
                        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                        stream.flush()

                    summary, prompts = evaluate_point(
                        protocol=protocol, budget=budget, seed=plan["seed"], rows=rows,
                        prompt_token_ids=[q["prompt_token_ids"] for q in rows], engine=llm,
                        sampling_params_type=params, tokenizer=tokenizer, score_many=score_many,
                        emit=emit, sampling=plan["sampling"], progress=progress, stop_on_first_success=True,
                    )
                if details:
                    raise ValueError("Unmatched grading records")
                audited, audited_prompts = audit_records(
                    point_records(rollouts), protocol=protocol, budget=budget, seed=plan["seed"], rows=rows,
                    per_rollout_cap=RESPONSE_CAP, stop_on_first_success=True,
                )
                if audited_prompts != prompts or any(summary[k] != v for k, v in audited.items()):
                    raise ValueError("Saved rollout ledger does not reproduce the result")
                prompt_path = attempt / "prompts.json"
                write(prompt_path, prompts)
                summary.update({
                    "protocol": protocol,
                    "unused_output_budget": summary["allocated_output_budget"] - summary["total_output_tokens"],
                    "state": "complete", "identity": point_identity(key, budget, manifest, dataset),
                    "seed": plan["seed"], "elapsed_seconds": time.time() - started, "finished_at": now(),
                    "pass_at_budget_percent": 100 * summary["fraction_solved"], "ledger_audit": "passed",
                    "artifacts": {name: {"file": p.relative_to(directory).as_posix(), "size": p.stat().st_size,
                                         "sha256": digest(p)}
                                  for name, p in (("rollouts", rollouts), ("prompts", prompt_path))},
                })
                write(directory / "summary.json", summary)
                write(root / "progress" / f"worker_{rank}.json", {**state, "state": "point_complete", "updated_at": now()})
                print(f"COMPLETE {dataset or 'minervamath'} {key} budget={budget}: "
                      f"{summary['num_questions_solved']}/{len(rows)}", flush=True)
    return 0


def report(root, plan, manifest):
    rows = []
    specs = {spec["key"]: spec for spec in dataset_specs(plan)}
    for key, budget, dataset in evaluation_points(plan):
        summary = completed_point(root, key, budget, manifest, dataset)
        if summary is None:
            raise ValueError(f"Missing completed point: {dataset} {key} {budget}")
        dataset_key = dataset or next(iter(specs))
        if summary["num_prompts"] != specs[dataset_key]["rows"]:
            raise ValueError(f"Wrong result denominator for {dataset_key}")
        rows.append({"dataset": dataset_key, "dataset_label": DATASETS[dataset_key][0],
                     "model": plan["models"][key]["label"], "model_key": key,
                     "budget_tokens": budget, "seed": plan["seed"],
                     "pass_at_budget_percent": summary["pass_at_budget_percent"],
                     "questions_solved": summary["num_questions_solved"], "questions": summary["num_prompts"],
                     "rollouts": summary["total_rollouts"], "total_output_tokens": summary["total_output_tokens"],
                     "mean_tokens_per_question": summary["total_output_tokens"] / summary["num_prompts"],
                     "unused_output_budget": summary["unused_output_budget"]})
    order = {key: index for index, key in enumerate(specs)}
    rows.sort(key=lambda row: (order[row["dataset"]], row["model_key"], row["budget_tokens"]))
    destination = root / "report"
    destination.mkdir(exist_ok=True)
    write(destination / "metrics.json", rows)
    with (destination / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cross = plan.get("protocol") == "eval3"
    lines = ["# Cross-context shared output-token budget" if cross else "# Individual output-token budget", "",
             f"{sum(spec['rows'] for spec in specs.values()):,} questions across {len(specs)} dataset(s); "
             f"one seed ({plan['seed']}); each response is capped at 32,768 tokens or the remaining allowance. "
             + ("All questions share n × budget tokens; seeded shuffled sweeps skip solved questions. "
              "Every question, including unvisited questions, stays in the denominator. " if cross else
              "Stop after the first correct answer; unused tokens stay with their question. ") +
             "Budgets use fresh sampling; temperature / top-p / top-k = 0.6 / 0.95 / 20. "
             "Scores use only the nonempty answer after completed thinking. No extra EOS or boxed requirement.", "",
             "| Dataset | Model | Budget | Pass@budget | Solved | Mean tokens/question |",
             "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {r['dataset_label']} | {r['model']} | {r['budget_tokens']} | {r['pass_at_budget_percent']:.2f}% | "
              f"{r['questions_solved']}/{r['questions']} | {r['mean_tokens_per_question']:.1f} |" for r in rows]
    lines += ["", "[Metrics CSV](metrics.csv) · [Metrics JSON](metrics.json) · [Audit](audit.json)"]
    (destination / "README.md").write_text("\n".join(lines) + "\n")
    write(destination / "audit.json", {"complete": True, "points": len(rows),
                                      "dataset_questions": {key: spec["rows"] for key, spec in specs.items()},
                                      "all_rollout_ledgers_verified": True, "metrics_sha256": digest(destination / "metrics.json")})


def pending_points(root, plan, manifest):
    return [(key, budget, dataset) for key, budget, dataset in evaluation_points(plan, largest_first=True)
            if not completed_point(root, key, budget, manifest, dataset)]


def run(root, plan, on_model_complete=None):
    require_compute(plan["job_id"])
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "RAY_ADDRESS"):
        os.environ.pop(name, None)
    if not dependency_ready(plan):
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

        if torch.cuda.device_count() != 8:
            raise RuntimeError("Expected all eight GPUs in the existing allocation")
        manifest = prepare_models(root, plan)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
        if len(visible) != 8:
            raise ValueError("Expected eight visible GPU identifiers")
        children, active, cleaned = [], {}, set()
        pending = deque(pending_points(root, plan, manifest))
        total_points = len(evaluation_points(plan))
        try:
            logs = [stack.enter_context((root / f"worker_{rank}.log").open("ab", buffering=0)) for rank in range(8)]
            started = now()
            while pending or active:
                for rank, child in list(active.items()):
                    if child.poll() is not None:
                        if child.returncode != 0:
                            raise RuntimeError(f"Evaluation worker {rank} failed: {child.returncode}")
                        del active[rank]
                if on_model_complete is not None:
                    active_keys = {child.budget_model_key for child in active.values()}
                    for key in plan["models"]:
                        points = [(k, b, d) for k, b, d in evaluation_points(plan) if k == key]
                        if (key not in cleaned and key not in active_keys
                                and all((point_directory(root, k, b, d) / "summary.json").exists()
                                        for k, b, d in points)):
                            on_model_complete(key)
                            cleaned.add(key)
                for rank, gpu in enumerate(visible):
                    if rank in active or not pending:
                        continue
                    key, budget, dataset = pending.popleft()
                    dataset_args = ["--dataset", dataset] if dataset is not None else []
                    child = subprocess.Popen(
                        [plan["python_bin"], "-u", "-m", MODULE, "worker", "--output-root", str(root),
                         "--model", key, "--budget", str(budget), "--rank", str(rank), *dataset_args], cwd=plan["runtime"],
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, stdin=subprocess.DEVNULL,
                        stdout=logs[rank], stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    children.append(child)
                    child.budget_model_key = key
                    active[rank] = child
                write(root / "execution.json", {"hostname": socket.gethostname(), "pid": os.getpid(),
                                               "job_id": os.environ.get("SLURM_JOB_ID"), "step_id": os.environ.get("SLURM_STEP_ID"),
                                               "worker_pids": [c.pid for c in active.values()], "started_at": started,
                                               "total_points": total_points, "updated_at": now()})
                write(root / "status.json", {"state": "running", "hostname": socket.gethostname(), "updated_at": now(),
                                             "completed_points": sum((point_directory(root, k, b, d) / "summary.json").exists()
                                                                     for k, b, d in evaluation_points(plan)),
                                             "total_points": total_points})
                if pending or active:
                    time.sleep(15)
            report(root, plan, manifest)
            write(root / "status.json", {"state": "complete", "points": total_points, "finished_at": now()})
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            for child in children:
                if child.poll() is None:
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
    return 0


def queue(root, plan):
    require_compute(plan["job_id"])
    total_points = len(evaluation_points(plan))
    state = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": plan["job_id"], "started_at": now(),
             "models": list(plan["models"]), "datasets": [spec["key"] for spec in dataset_specs(plan)],
             "total_points": total_points}

    def update(**values):
        state.update(values, updated_at=now())
        write(root / "queue_status.json", state)

    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while not dependency_ready(plan):
                require_compute(plan["job_id"])
                previous = Path(plan["parent_eval_root"]) / "queue_status.json"
                update(state="waiting_for_nine_dataset_evaluation", predecessor_state=read(previous).get("state")
                       if previous.exists() else "not_started")
                time.sleep(30)
            failures = 0
            while True:
                require_compute(plan["job_id"])
                update(state="waiting_for_gpus_or_running", attempts=failures + 1)
                command = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                           f"--nodelist={plan['node']}", "--cpus-per-task=96", "--gres=gpu:8", "--kill-on-bad-exit=1",
                           "--job-name=math-individual-budget", plan["python_bin"], "-u", "-m", MODULE,
                           "run", "--output-root", str(root)]
                with (root / "launch.log").open("ab", buffering=0) as log:
                    child = subprocess.Popen(command, cwd=plan["runtime"], env=environment(plan), stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    update(launcher_pid=child.pid)
                    while child.poll() is None:
                        update(state="running_or_waiting_for_resources")
                        time.sleep(15)
                if child.returncode == 0:
                    audit = read(root / "report/audit.json")
                    if not audit["complete"] or audit["points"] != total_points:
                        raise ValueError("Incomplete individual-budget result audit")
                    update(state="complete", points=total_points, report=str(root / "report/README.md"), finished_at=now())
                    return 0
                if child.returncode != 75:
                    failures += 1
                    if failures >= 3:
                        raise RuntimeError("Individual-budget evaluation failed three times; completed points and raw attempts are preserved")
                update(state="retrying", last_exit_code=child.returncode)
                time.sleep(30)
        except BaseException as exc:
            update(state="failed", error=str(exc), finished_at=now())
            raise


def launch(root, plan):
    require_compute(plan["job_id"])
    for name in ("tmp", "triton"):
        (Path(plan["scratch"]) / name).mkdir(parents=True, exist_ok=True)
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            raise RuntimeError("Queue already launched; inspect queue_status.json")
        with (root / "queue.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([plan["python_bin"], "-u", "-m", MODULE, "queue", "--output-root", str(root)],
                                     cwd=plan["runtime"], env=environment(plan), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"pid": child.pid, "hostname": socket.gethostname(), "job_id": plan["job_id"],
                   "plan_sha256": digest(root / "plan.json"), "launched_at": now()}
        write(root / "launch.json", receipt)
        print(json.dumps(receipt), flush=True)


def check_waiting_replacement(previous, plan):
    old = verify_plan(previous)
    for key in ("job_id", "node", "parent_run", "parent_eval_root", "seed", "budgets", "sampling", "grading"):
        if old[key] != plan[key]:
            raise ValueError(f"Replacement changed the original experiment's {key}")
    for key in ("parent_eval_plan_sha256", "engine", "protocol", "stop_on_first_success", "num_gpus"):
        if old.get(key) != plan.get(key):
            raise ValueError(f"Replacement changed the original experiment's {key}")
    original_datasets = {spec["key"]: spec for spec in dataset_specs(old)}
    new_datasets = {spec["key"]: spec for spec in dataset_specs(plan)}
    if any(key not in new_datasets or any(new_datasets[key].get(field) != value for field, value in spec.items())
           for key, spec in original_datasets.items()):
        raise ValueError("Replacement must preserve every previously requested dataset and its revision")
    for relative in ("runtime/qwen3_experiments/math_eval_budget_engine.py",
                     "runtime/qwen3_experiments/math_eval_matrix_common.py",
                     "provenance/eval_l0_final.py", "provenance/eval_polaris_step80.py"):
        if relative in old.get("frozen_files", {}) and plan["frozen_files"].get(relative) != old["frozen_files"][relative]:
            raise ValueError(f"Replacement changed the original sampling or grading implementation: {relative}")
    for key, model in old["models"].items():
        if key not in plan["models"] or any(plan["models"][key][field] != model[field]
                                            for field in ("repo", "revision", "path")):
            raise ValueError("Replacement must preserve every previously requested model")
        if model.get("questions_file"):
            old_rows = read(previous / model["questions_file"])
            new_rows = read(Path(plan["output_root"]) / plan["models"][key]["questions_file"])
            for dataset in original_datasets:
                before = [row for row in old_rows if row.get("dataset", "minervamath") == dataset]
                after = [row for row in new_rows if row.get("dataset", "minervamath") == dataset]
                before = [{field: value for field, value in row.items() if field != "dataset"} for row in before]
                after = [{field: value for field, value in row.items() if field != "dataset"} for row in after]
                if before != after:
                    raise ValueError(f"Replacement changed frozen questions, prompts or seed order: {key} {dataset}")
    if read(previous / "queue_status.json").get("state") != "waiting_for_nine_dataset_evaluation":
        raise ValueError("Only a waiting Minerva queue can be replaced")
    if any((previous / name).exists() for name in ("execution.json", "execution_manifest.json", "results", "status.json")):
        raise ValueError("Minerva execution already began; refusing to replace its frozen inputs")
    return read(previous / "launch.json")


def open_pidfd(pid):
    # Some Conda Python builds omit os.pidfd_open despite kernel support.
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    import ctypes
    import platform

    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("Safe queue replacement requires Linux pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.set_inheritable(descriptor, False)
    return descriptor


def replace_waiting(root, plan, previous):
    """Prepare everything first, then replace only the still-waiting controller."""
    require_compute(plan["job_id"])
    previous = previous.resolve()
    if root == previous or (root / "launch.json").exists():
        raise ValueError("Replacement needs a distinct, prepared, unlaunched output directory")
    with (previous / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = check_waiting_replacement(previous, plan)
        if receipt["hostname"] != socket.gethostname():
            raise ValueError("Old queue belongs to another node")
        pid = receipt["pid"]
        descriptor = open_pidfd(pid)
        stopped = False
        try:
            command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
            expected = [b"-m", MODULE.encode(), b"queue", b"--output-root", str(previous).encode()]
            if command[2:7] != expected:
                raise ValueError("PID is not the expected waiting Minerva controller")
            signal.pidfd_send_signal(descriptor, signal.SIGSTOP)
            stopped = True
            for _ in range(100):
                state = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0]
                if state in ("T", "t"):
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError("Could not pause the waiting queue")
            check_waiting_replacement(previous, plan)
            # The old process is paused and has never started evaluation. If launch
            # fails, the finally clause resumes that original queue.
            launch(root, plan)
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            signal.pidfd_send_signal(descriptor, signal.SIGCONT)
            stopped = False
            import select

            if not select.select([descriptor], [], [], 10)[0]:
                raise RuntimeError("Replaced controller did not exit")
            replacement = {"state": "superseded", "replaced_by": str(root), "updated_at": now(),
                           "old_launch": receipt, "new_launch": read(root / "launch.json")}
            write(previous / "queue_status.json", replacement)
            write(root / "replaced_queue.json", {**replacement, "previous_root": str(previous)})
        finally:
            if stopped:
                signal.pidfd_send_signal(descriptor, signal.SIGCONT)
            os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "queue", "run", "worker"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-only", action="store_true",
                        help="Evaluate only the parent's final checkpoint, without the five-model comparison")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["minervamath"],
                        help="Datasets to prepare; each dataset has independent per-question budgets")
    parser.add_argument("--dataset", choices=DATASETS, help="The dataset assigned to one evaluation worker")
    parser.add_argument("--model")
    parser.add_argument("--budget", type=int, choices=BUDGETS)
    parser.add_argument("--replace-waiting", type=Path)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.command in ("prepare", "launch"):
        if args.parent_run is None or args.seed < 0:
            parser.error("--parent-run and a nonnegative --seed are required")
        plan = prepare(args)
        if args.command == "launch":
            if args.replace_waiting:
                replace_waiting(root, plan, args.replace_waiting)
            else:
                launch(root, plan)
        else:
            print(json.dumps({"state": "prepared", "hostname": plan["node"], "output_root": str(root)}))
        return 0
    plan = verify_plan(root)
    require_compute(plan["job_id"])
    if args.command == "worker":
        if args.model not in plan["models"] or args.rank is None:
            parser.error("worker requires --model and --rank")
        return worker(root, args.model, args.rank, args.budget, args.dataset)
    return {"queue": queue, "run": run}[args.command](root, plan)


if __name__ == "__main__":
    raise SystemExit(main())
