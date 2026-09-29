"""Resumable mean@4 evaluation of the pinned Polaris step-80 checkpoint."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = "hi-todayis-jh/maxrl-qwen3-1.7b-polaris-1-8-3200-bs32-32k-145514-step_80"
REVISION = "ce2a5f623e0e45e9f50f94a5001bb27fff2e6a17"
PREFIX = "global_step_80/actor/"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "max_tokens": 32768, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "repetition_penalty": 1.0}
BENCHMARKS = [
    {"key": "aime24", "repo": "math-ai/aime24", "revision": "83a7f387baaa524a8bda0022eac0541582297103",
     "file": "test-00000-of-00001.parquet", "question": "problem", "answer": "solution", "rows": 30},
    {"key": "aime25", "repo": "math-ai/aime25", "revision": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
     "file": "test.jsonl", "question": "problem", "answer": "answer", "rows": 30},
    {"key": "aime26", "repo": "math-ai/aime26", "revision": "79037aebdb6580008fb960d17cb21fd3099083e3",
     "file": "aime2026.jsonl", "question": "problem", "answer": "answer", "rows": 30},
    {"key": "olympiadbench", "repo": "math-ai/olympiadbench", "revision": "4faaf1e6ec17d11a4218a9bf4c049ecaf954dd84",
     "file": "test.parquet", "question": "question", "answer": "final_answer", "rows": 674},
]
METRIC = None


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def last_box(text):
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not matches:
        return None
    match = matches[-1]
    depth = 1
    for index in range(match.end(), len(text)):
        if text[index] not in "{}":
            continue
        previous = index - 1
        while previous >= 0 and text[previous] == "\\":
            previous -= 1
        if (index - previous - 1) % 2:
            continue
        depth += 1 if text[index] == "{" else -1
        if depth == 0:
            return text[match.start():index + 1]
    return None


def prediction(text):
    if "</think>" not in text:
        return None
    final = text.rsplit("</think>", 1)[1]
    return last_box(final) if "<think>" not in final else None


def alarm_timeout(signum, frame):
    raise TimeoutError("Math grading deadline exceeded")


def init_grader():
    global METRIC
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    METRIC = math_metric(gold_extraction_target=(LatexExtractionConfig(),),
                        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()))


def grade(item):
    box, gold = item
    if box is None:
        return {"correct": 0.0, "grader_status": "no_final_box"}
    signal.signal(signal.SIGALRM, alarm_timeout)
    signal.alarm(1)
    try:
        score, _ = METRIC([f"\\boxed{{{gold}}}"], [box])
        return {"correct": float(score), "grader_status": "scored"}
    except Exception as exc:
        return {"correct": 0.0, "grader_status": type(exc).__name__}
    finally:
        signal.alarm(0)


def prepare_model(root):
    from huggingface_hub import HfApi, hf_hub_download

    with (root / "model_prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt_path = root / "model_receipt.json"
        if receipt_path.exists():
            receipt = read(receipt_path)
            assert receipt["repo"] == MODEL and receipt["revision"] == REVISION
            for name, value in receipt["merged_files"].items():
                assert digest(root / "model" / name) == value
            print("Verified prepared model", flush=True)
            return
        api = HfApi()
        info = api.model_info(MODEL, revision=REVISION, files_metadata=True)
        available = {item.rfilename: item for item in info.siblings}
        assets = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json"]
        names = [PREFIX + f"model_world_size_8_rank_{rank}.pt" for rank in range(8)]
        names += [PREFIX + name for name in assets]
        assert all(name in available for name in names)

        def download(name):
            path = Path(hf_hub_download(MODEL, name, revision=REVISION, local_dir=root / "source_model"))
            checksum = digest(path)
            metadata = available[name]
            assert path.stat().st_size == metadata.size
            if metadata.lfs:
                assert checksum == metadata.lfs.sha256
            else:
                body = path.read_bytes()
                assert hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest() == metadata.blob_id
            print(f"Verified checkpoint file {name}", flush=True)
            return name, {"sha256": checksum, "size": path.stat().st_size}

        with ThreadPoolExecutor(max_workers=4) as pool:
            sources = dict(pool.map(download, names))
        source = root / "source_model" / PREFIX
        temporary = root / "model_in_progress"
        subprocess.run([sys.executable, str(REPO / "scripts/model_merger.py"), "merge", "--backend", "fsdp",
                        "--local_dir", str(source), "--target_dir", str(temporary)], check=True, cwd=REPO,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        # Validate every merged tensor's shape/dtype against the configured architecture.
        from accelerate import init_empty_weights
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        import torch

        cfg = AutoConfig.from_pretrained(temporary, local_files_only=True)
        with init_empty_weights():
            reference = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
        expected = {key: list(tensor.shape) for key, tensor in reference.state_dict().items()}
        actual = {}
        for path in temporary.glob("*.safetensors"):
            with safe_open(path, framework="pt", device="cpu") as stream:
                for key in stream.keys():
                    value = stream.get_slice(key)
                    assert key not in actual and value.get_dtype() == "BF16"
                    actual[key] = value.get_shape()
        if cfg.tie_word_embeddings and "lm_head.weight" not in actual:
            expected.pop("lm_head.weight")
        assert actual == expected, "Merged checkpoint tensor inventory differs from architecture"
        original_tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
        merged_tok = AutoTokenizer.from_pretrained(temporary, local_files_only=True)
        assert original_tok.get_vocab() == merged_tok.get_vocab()
        assert original_tok.chat_template == merged_tok.chat_template
        temporary.rename(root / "model")
        receipt = {"repo": MODEL, "revision": REVISION, "format": "8-rank FSDP merged to BF16",
                   "source_files": sources, "tensor_count": len(actual),
                   "tensor_inventory_sha256": fingerprint(actual),
                   "merger_sha256": digest(REPO / "scripts/model_merger.py"),
                   "merged_files": {path.name: digest(path) for path in (root / "model").iterdir() if path.is_file()}}
        write(receipt_path, receipt)
        print("Model merge and tensor/tokenizer verification complete", flush=True)


def prepare_data(root):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import pandas as pd

    hub = read(root / "polaris_test/hub_receipt.json")
    specs = BENCHMARKS + [{"key": "polaris", "repo": hub["repo_id"], "revision": hub["revision"],
        "file": "data/test-00000-of-00001.parquet", "question": "problem", "answer": "answer", "rows": 100}]
    # Tokenizer assets can be fetched before merging the model.
    tokenizer_path = Path(hf_hub_download(MODEL, PREFIX + "tokenizer_config.json", revision=REVISION,
                                        local_dir=root / "tokenizer_source")).parent
    for name in ("tokenizer.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"):
        hf_hub_download(MODEL, PREFIX + name, revision=REVISION, local_dir=root / "tokenizer_source")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    questions, inventories = [], []
    for spec in specs:
        path = Path(hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=spec["revision"]))
        rows = pd.read_parquet(path).to_dict("records") if path.suffix == ".parquet" else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) == spec["rows"]
        seen = set()
        for index, row in enumerate(rows):
            problem, answer = row[spec["question"]], row[spec["answer"]]
            if hasattr(answer, "tolist"):
                answer = answer.tolist()
            if isinstance(answer, list):
                assert len(answer) == 1
                answer = answer[0]
            answer = str(answer)
            if spec["key"] == "aime24":
                boxed = last_box(answer)
                assert boxed is not None
                answer = boxed[boxed.index("{") + 1:-1]
            if spec["key"].startswith("aime"):
                assert answer.isdigit() and 0 <= int(answer) <= 999
                answer = str(int(answer))
            if spec["key"] == "olympiadbench":
                assert row["modality"] == "Text-only" and row["subject"] == "Math" and row["language"] == "English"
                assert all(row[f"image_{i}"] is None for i in range(1, 6))
            assert isinstance(problem, str) and problem.strip() and answer.strip()
            assert problem not in seen
            seen.add(problem)
            messages = [{"role": "user", "content": problem + SUFFIX}]
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True)
            no_think = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
            assistant_prefix = tokenizer.decode(ids).rsplit("<|im_start|>assistant", 1)[-1]
            assert ids != no_think and "</think>" not in assistant_prefix
            assert len(ids) + 32768 <= 40960, "Prompt would reduce the 32k output allowance"
            questions.append({"id": f"{spec['key']}_{index:04d}", "dataset": spec["key"], "source_row": index,
                              "source_id": str(row.get("source_index", row.get("url", row.get("id", index)))),
                              "problem": problem, "gold": answer, "messages": messages, "prompt_token_ids": ids})
        inventories.append({**spec, "file_sha256": digest(path)})
    # Complete the small benchmark sets first; the ordering is independent of labels.
    order = {name: index for index, name in enumerate(["aime24", "aime25", "aime26", "polaris", "olympiadbench"])}
    questions.sort(key=lambda q: (order[q["dataset"]], q["source_row"]))
    assert len(questions) == 864
    write(root / "questions.json", questions)
    prepared = {"model": {"repo": MODEL, "revision": REVISION}, "datasets": inventories,
                "questions_sha256": digest(root / "questions.json"), "samples_per_question": 4,
                "sampling_seed": 42, "sampling": SAMPLING, "thinking": True, "prompt_suffix": SUFFIX,
                "max_model_len": 40960, "longest_prompt_tokens": max(len(q["prompt_token_ids"]) for q in questions),
                "chat_template_sha256": fingerprint(tokenizer.chat_template),
                "grading": {"library": "Math-Verify", "prediction": "last complete box after </think>",
                            "timeout_seconds": 1, "missing_final_answer_score": 0, "metric": "mean@4"}}
    write(root / "prepared_inputs.json", prepared)
    print(f"Prepared {len(questions)} questions / {len(questions) * 4} responses", flush=True)


def sample_seed(question_id, index):
    return int.from_bytes(hashlib.blake2b(f"step80-eval-v1:42:{question_id}:{index}".encode(), digest_size=8).digest(), "big") % (2**31 - 1)


def sample_id(question, index):
    return f"{question['id']}__{index}"


def saved_result(root, identity, manifest_hash, *, full=False):
    receipt_path = root / "responses" / f"{identity}.receipt.json"
    if not receipt_path.exists():
        return None
    receipt = read(receipt_path)
    path = root / "responses" / receipt["file"]
    assert receipt["manifest_sha256"] == manifest_hash
    assert path.stat().st_size == receipt["size"] and digest(path) == receipt["sha256"]
    if full:
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return receipt


def save_result(root, record, manifest_hash):
    directory = root / "responses"
    directory.mkdir(exist_ok=True)
    path = directory / f"{record['id']}.json.gz"
    temp = path.with_name(path.name + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False)
    temp.replace(path)
    write(directory / f"{record['id']}.receipt.json", {
        "manifest_sha256": manifest_hash, "id": record["id"], "file": path.name,
        "size": path.stat().st_size, "sha256": digest(path), "correct": record["correct"],
        "output_tokens": record["output_tokens"], "dataset": record["dataset"]})


def establish_manifest(root):
    prepared = read(root / "prepared_inputs.json")
    assert prepared["sampling"] == SAMPLING and prepared["samples_per_question"] == 4
    assert digest(root / "questions.json") == prepared["questions_sha256"]
    receipt = read(root / "model_receipt.json")
    for name, checksum in receipt["merged_files"].items():
        assert digest(root / "model" / name) == checksum
    manifest = {"inputs": prepared, "model": receipt, "evaluator_sha256": digest(__file__),
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers", "math-verify", "sympy", "latex2sympy2_extended")},
                "engine": {"dtype": "bfloat16", "max_model_len": 40960, "max_num_seqs": 32,
                           "max_num_batched_tokens": 8192, "gpu_memory_utilization": 0.85,
                           "enable_chunked_prefill": True, "enable_prefix_caching": True,
                           "generation_config": "vllm", "tensor_parallel_size": 1}}
    if (root / "manifest.json").exists():
        assert read(root / "manifest.json") == manifest, "Run settings changed; use a separate output directory"
    else:
        write(root / "manifest.json", manifest)
    return manifest, digest(root / "manifest.json")


def worker(root, rank, world_size):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    with (root / f"worker_{rank}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, manifest_hash = read(root / "manifest.json"), digest(root / "manifest.json")
        assert manifest["evaluator_sha256"] == digest(__file__)
        questions = read(root / "questions.json")
        pending, completed = [], 0
        for position, question in enumerate(questions):
            for index in range(4):
                if (position * 4 + index) % world_size != rank:
                    continue
                identity = sample_id(question, index)
                if saved_result(root, identity, manifest_hash):
                    completed += 1
                else:
                    pending.append((identity, question, index))
        if not pending:
            return
        progress_path = root / "progress" / f"worker_{rank}.json"
        write(progress_path, {"state": "loading_model", "rank": rank, "completed_responses": completed,
            "assigned_responses": completed + len(pending), "updated_at": time.time()})
        model_path = os.environ.get("STEP80_LOCAL_MODEL", str(root / "model"))
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        assert fingerprint(tokenizer.chat_template) == manifest["inputs"]["chat_template_sha256"]
        llm = LLM(model=model_path, tokenizer=model_path, seed=42,
                  trust_remote_code=False, disable_log_stats=True, **manifest["engine"])
        engine = llm.llm_engine
        eos = read(root / "model/generation_config.json")["eos_token_id"]
        if isinstance(eos, int):
            eos = [eos]
        for identity, question, index in pending:
            params = SamplingParams(n=1, seed=sample_seed(question["id"], index), stop_token_ids=eos,
                                    skip_special_tokens=False, **SAMPLING)
            engine.add_request(identity, {"prompt_token_ids": question["prompt_token_ids"]}, params)
        lookup = {identity: (question, index) for identity, question, index in pending}
        inflight, status_time = {}, 0
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=4, mp_context=context, initializer=init_grader) as graders:
            while engine.has_unfinished_requests() or inflight:
                outputs = engine.step() if engine.has_unfinished_requests() else []
                for output in outputs:
                    if not output.finished:
                        continue
                    question, index = lookup[output.request_id]
                    assert output.prompt_token_ids == question["prompt_token_ids"] and len(output.outputs) == 1
                    generated = output.outputs[0]
                    ids = list(generated.token_ids)
                    assert 0 < len(ids) <= 32768 and generated.finish_reason in ("stop", "length")
                    assert generated.finish_reason != "length" or len(ids) == 32768
                    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    record = {"id": output.request_id, "question_id": question["id"], "dataset": question["dataset"],
                              "sample_index": index, "seed": sample_seed(question["id"], index),
                              "prompt_tokens": len(question["prompt_token_ids"]), "output_tokens": len(ids),
                              "output_token_ids": ids, "response": text, "prediction": prediction(text),
                              "thinking_complete": "</think>" in text, "finish_reason": generated.finish_reason,
                              "stop_reason": generated.stop_reason}
                    inflight[graders.submit(grade, (record["prediction"], question["gold"]))] = record
                for future in list(inflight):
                    if future.done():
                        record = inflight.pop(future)
                        record.update(future.result())
                        save_result(root, record, manifest_hash)
                        completed += 1
                        print(f"DONE worker={rank} completed={completed} {record['id']} score={record['correct']} tokens={record['output_tokens']}", flush=True)
                if time.time() - status_time >= 30:
                    write(progress_path, {"state": "running", "rank": rank, "completed_responses": completed,
                        "updated_at": time.time()})
                    status_time = time.time()
                if not outputs and inflight and not engine.has_unfinished_requests():
                    time.sleep(0.05)
        write(progress_path, {"state": "complete", "rank": rank, "completed_responses": completed,
                             "updated_at": time.time()})


def run(root):
    import torch

    with (root / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert torch.cuda.device_count() == 8, "This run requires all eight allocated GPUs"
        manifest, manifest_hash = establish_manifest(root)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
        assert len(visible) == 8
        local_model = Path(os.environ["TMPDIR"]) / "step80_model"
        local_model.mkdir(parents=True, exist_ok=True)
        for name, checksum in manifest["model"]["merged_files"].items():
            destination = local_model / name
            if not destination.exists() or digest(destination) != checksum:
                shutil.copy2(root / "model" / name, destination)
            assert digest(destination) == checksum
        write(root / "execution.json", {"job_id": os.environ.get("SLURM_JOB_ID"), "hostname": os.uname().nodename,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(8)], "visible_devices": visible,
            "workers": 8, "started_at": time.time(), "manifest_sha256": manifest_hash})
        children, logs = [], []
        try:
            for rank, gpu in enumerate(visible):
                log = (root / f"worker_{rank}.log").open("a", buffering=1)
                logs.append(log)
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "STEP80_LOCAL_MODEL": str(local_model)}
                child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                    "--output-root", str(root), "--rank", str(rank), "--world-size", "8"],
                    stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
                children.append(child)
            while any(child.poll() is None for child in children):
                failed = [(rank, child.returncode) for rank, child in enumerate(children) if child.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f"Evaluation workers failed: {failed}")
                completed = len(list((root / "responses").glob("*.receipt.json")))
                write(root / "status.json", {"state": "running", "completed_responses": completed,
                    "total_responses": 3456, "job_id": os.environ.get("SLURM_JOB_ID"), "gpus": 8, "updated_at": time.time()})
                time.sleep(15)
            assert all(child.returncode == 0 for child in children), "Evaluation worker failed"
            report(root)
        except BaseException as exc:
            write(root / "status.json", {"state": "failed", "error": str(exc), "updated_at": time.time()})
            raise
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            for child in children:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            for log in logs:
                log.close()


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(root, partial=False):
    from transformers import AutoTokenizer

    manifest = read(root / "manifest.json")
    manifest_hash = digest(root / "manifest.json")
    assert digest(root / "questions.json") == manifest["inputs"]["questions_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)
    rows, question_rows = [], []
    for question in read(root / "questions.json"):
        samples = []
        for index in range(4):
            identity = sample_id(question, index)
            record = saved_result(root, identity, manifest_hash, full=True)
            if record is None:
                continue
            ids = record["output_token_ids"]
            assert record["id"] == identity and record["seed"] == sample_seed(question["id"], index)
            assert record["question_id"] == question["id"] and record["dataset"] == question["dataset"]
            assert record["sample_index"] == index and record["prompt_tokens"] == len(question["prompt_token_ids"])
            assert record["output_tokens"] == len(ids) and 0 < len(ids) <= 32768
            assert record["response"] == tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            assert record["prediction"] == prediction(record["response"])
            assert record["correct"] in (0.0, 1.0)
            assert record["prediction"] is not None or record["correct"] == 0.0
            assert record["finish_reason"] != "length" or len(ids) == 32768
            samples.append(record)
            rows.append({key: record[key] for key in ("id", "question_id", "dataset", "sample_index", "seed", "correct",
                         "output_tokens", "finish_reason", "thinking_complete", "grader_status")})
        if not partial:
            assert len(samples) == 4, f"Incomplete question: {question['id']}"
        if len(samples) == 4:
            question_rows.append({"question_id": question["id"], "dataset": question["dataset"],
                                  "accuracy": statistics.mean(s["correct"] for s in samples),
                                  "mean_output_tokens": statistics.mean(s["output_tokens"] for s in samples)})
    summary = []
    for spec in manifest["inputs"]["datasets"]:
        group = [row for row in rows if row["dataset"] == spec["key"]]
        complete = [row for row in question_rows if row["dataset"] == spec["key"]]
        if not group:
            continue
        summary.append({"dataset": spec["key"], "questions_complete": len(complete), "questions_total": spec["rows"],
                        "responses": len(group), "correct_responses": int(sum(row["correct"] for row in group)),
                        "accuracy_percent": 100 * statistics.mean(row["correct"] for row in group),
                        "mean_output_tokens": statistics.mean(row["output_tokens"] for row in group),
                        "length_capped_responses": sum(row["finish_reason"] == "length" for row in group),
                        "unfinished_thinking_responses": sum(not row["thinking_complete"] for row in group)})
    directory = root / ("partial_report" if partial else "report")
    directory.mkdir(exist_ok=True)
    if rows:
        write_csv(directory / "per_sample.csv", rows)
        write_csv(directory / "accuracy.csv", summary)
    if question_rows:
        write_csv(directory / "per_question.csv", question_rows)
    write(directory / "audit.json", {"responses_verified": len(rows), "complete_questions": len(question_rows),
          "checksums": "passed", "token_text_reconstruction": "passed", "seed_accounting": "passed",
          "aggregation": "passed", "grader_status_counts": dict(Counter(row["grader_status"] for row in rows)),
          "scores_independently_regraded": False, "complete": not partial})
    lines = ["# Step-80 math evaluation", "", f"Model: [{MODEL}](https://huggingface.co/{MODEL}/tree/{REVISION})", "",
             "Four independently seeded responses per question. Accuracy is correct responses / all responses (mean@4).",
             "Thinking enabled; temperature 0.6, top-p 0.95, top-k 20; 32,768 output tokens per response.",
             "The full output allowance is separate from the prompt and fits within the model's 40,960-token context.",
             "Math-Verify grades the last complete boxed answer after `</think>`; unfinished reasoning and missing final boxes score zero.",
             "", "| Dataset | Questions completed | Responses | Correct | Accuracy | Mean output tokens |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['dataset']} | {row['questions_complete']}/{row['questions_total']} | {row['responses']} | {row['correct_responses']} | {row['accuracy_percent']:.2f}% | {row['mean_output_tokens']:.0f} |")
    lines += ["", "Dataset revisions, source checksums, prompts, model merge provenance, package versions and seeds are retained in the run directory.",
              "All response texts and token IDs are retained with checksums. The final audit checks saved token/text correspondence and recomputes aggregates; it does not rerun the scorer.",
              "", "## Dataset sources", ""]
    for spec in manifest["inputs"]["datasets"]:
        lines.append(f"- [{spec['repo']}](https://huggingface.co/datasets/{spec['repo']}/tree/{spec['revision']})")
    lines += ["", "Olympiad-Bench is the 674-question English, text-only, open-ended math subset. Original answer expressions are retained.",
              "Polaris-Test contains 100 numerical, non-proof 1/8 questions held out from the pinned Polaris-1-8-3200 training dataset.",
              "No claim is made about benchmark exposure during pretraining or paraphrased overlap.", ""]
    (directory / "README.md").write_text("\n".join(lines))
    if not partial:
        assert len(rows) == 3456 and len(question_rows) == 864
        write(root / "status.json", {"state": "complete", "completed_responses": len(rows), "total_responses": 3456,
                                    "completed_questions": len(question_rows), "updated_at": time.time()})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-model", "prepare-data", "run", "worker", "report"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare-model":
        prepare_model(root)
    elif args.command == "prepare-data":
        prepare_data(root)
    elif args.command == "run":
        run(root)
    elif args.command == "worker":
        worker(root, args.rank, args.world_size)
    else:
        report(root, partial=args.partial)


if __name__ == "__main__":
    main()
