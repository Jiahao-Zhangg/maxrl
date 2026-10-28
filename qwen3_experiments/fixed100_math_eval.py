"""Portable fixed-100 compression math evaluation; see fixed100_math_eval.md."""

import argparse
import asyncio
import atexit
import concurrent.futures
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

MODEL = "hi-todayis-jh/l1-max-qwen3-1.7b-compression-100-4k-from80-mixed50-alpha3e-4-bs32-n8-32k-t1-146103-final"
MODEL_REVISION = "b9bde5a3c7d66c3ff80612842bb702f5911e5a40"
QUESTIONS_REPO = "hi-todayis-jh/compression-math-100plus-topup128k-qwen3-1.7b-verl091-146103-20261004"
QUESTIONS_REVISION = "f78ba3cf54fd10742354d0c523e8936ec282fc7a"
QUESTIONS_BLOB = "45fdf3640325e138185e0fd57258a6d785e31439"
REFERENCE_REPO = "hi-todayis-jh/compression-math-big-eval-all-models-fixed100-20261027"
REFERENCE_REVISION = "141398492e0d348652d817a80e55cb7b97c85589"
DATASETS = ["minervamath", "olympiadbench", "amc22_23", "aime24", "aime25", "aime26"]
EXCLUDED = [
    "amc22_23_0016",
    "amc22_23_0028",
    "amc22_23_0074",
    "minervamath_0135",
    "minervamath_0136",
    "minervamath_0217",
    "minervamath_0230",
]
BUDGETS = np.array([4, 8, 16, 32, 64], dtype=np.int64) * 1024
CAPS = np.array([1, 2, 4, 8, 16, 32], dtype=np.int64) * 1024
GRADER_VERSIONS = {
    "math-verify": "0.9.0",
    "latex2sympy2_extended": "1.11.0",
    "sympy": "1.13.1",
    "antlr4-python3-runtime": "4.9.3",
    "mpmath": "1.3.0",
}
GENERATION_VERSIONS = {"vllm": "0.24.0", "transformers": "5.9.0", "tokenizers": "0.22.2"}
SAMPLING = {
    "n": 1,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": 32768,
    "ignore_eos": False,
    "detokenize": True,
    "skip_special_tokens": True,
}
RUNTIME_FILES = [
    "fixed100_math_eval.py",
    "fixed100_math_analysis.py",
    "fixed100_math_replay.py",
    "fixed100_legacy_math_scorer.py",
    "compression_verl_reward.py",
    "capped_response_eval.py",
]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write(path, data):
    atomic_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_gzip(path, data):
    atomic_bytes(
        path, gzip.compress(json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), compresslevel=3, mtime=0)
    )


def sample_seed(position, attempt):
    value = f"math-eval-v1:42:{position}:{attempt}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big") % (2**31 - 1)


def record_name(question, attempt):
    return f"{question['position']:04d}_{attempt}.json.gz"


def validate_questions(questions):
    ids, positions = set(), set()
    for q in questions:
        if q["unique_id"] in ids or q["position"] in positions or q["dataset"] not in DATASETS:
            raise ValueError("Duplicate or invalid question identity")
        if not q["prompt_token_ids"] or q["position"] < 0:
            raise ValueError("Invalid original position or prompt tokens")
        ids.add(q["unique_id"])
        positions.add(q["position"])
    if not ids:
        raise ValueError("Empty question set")


def response_identity(plan, q, attempt):
    return {
        "unique_id": q["unique_id"],
        "dataset": q["dataset"],
        "position": q["position"],
        "attempt": attempt,
        "seed": sample_seed(q["position"], attempt),
        "model_repo": plan["model"]["repo"],
        "model_revision": plan["model"]["revision"],
        "plan_sha256": plan["plan_sha256"],
    }


def verify_response(record, plan, q, attempt):
    if any(record.get(k) != v for k, v in response_identity(plan, q, attempt).items()):
        raise ValueError(f"Saved response provenance differs: {q['unique_id']}:{attempt}")
    if (
        not isinstance(record.get("response"), str)
        or record["finish_reason"] not in ("stop", "length")
        or record["output_tokens"] != len(record["token_ids"])
        or not 0 < record["output_tokens"] <= SAMPLING["max_tokens"]
        or any(not isinstance(t, int) or t < 0 for t in record["token_ids"])
    ):
        raise ValueError("Invalid completed response")


def load_run(root):
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    for name, expected in plan["files_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    if plan["sampling"] != SAMPLING or plan["samples"] != 100 or plan["seed"] != 42:
        raise ValueError("Fixed-100 protocol changed")
    questions = read(root / "questions.json")
    validate_questions(questions)
    plan["plan_sha256"] = digest(root / "plan.json")
    return root, plan, questions


def prepare(args):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    root = args.run_dir.resolve()
    if (root / "plan.json").exists():
        raise ValueError("A frozen run already exists; use generate/grade to resume it")
    if min(args.shards, args.tensor_parallel_size, args.concurrency) < 1:
        raise ValueError("Shard count, tensor parallel size and concurrency must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("GPU memory utilization must be between zero and one")
    root.mkdir(parents=True, exist_ok=True)
    source = Path(hf_hub_download(QUESTIONS_REPO, "questions.json", repo_type="dataset", revision=QUESTIONS_REVISION))
    data = source.read_bytes()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if blob != QUESTIONS_BLOB:
        raise ValueError("Pinned source question manifest changed")
    questions = json.loads(data)
    if len(questions) != 1119 or not set(EXCLUDED).issubset(q["unique_id"] for q in questions):
        raise ValueError("Unexpected benchmark population")
    questions = [q for q in questions if q["unique_id"] not in EXCLUDED]
    validate_questions(questions)
    revision = args.revision or (MODEL_REVISION if args.model == MODEL else "main")
    info = HfApi().model_info(args.model, revision=revision)
    snapshot_download(
        args.model,
        revision=info.sha,
        local_dir=root / "tokenizer",
        allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"],
    )
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    for q in questions:
        ids = tokenizer.apply_chat_template(
            q["prompt"], add_generation_prompt=True, enable_thinking=True, return_dict=False
        )
        if ids != q["prompt_token_ids"]:
            raise ValueError(f"Model tokenizer changes the pinned thinking prompt: {q['unique_id']}")
    if max(len(q["prompt_token_ids"]) for q in questions) + 32768 > 40960:
        raise ValueError("Prompt plus response does not fit the configured engine context")
    write(root / "questions.json", questions)
    runtime = root / "runtime" / "qwen3_experiments"
    runtime.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_FILES:
        shutil.copy2(Path(__file__).parent / name, runtime / name)
    files = [
        root / "questions.json",
        *runtime.glob("*.py"),
        *(p for p in (root / "tokenizer").rglob("*") if p.is_file() and ".cache" not in p.parts),
    ]
    plan = {
        "format": "compression-fixed100-v1",
        "model": {"key": args.model_key, "label": args.label, "repo": args.model, "revision": info.sha},
        "samples": 100,
        "seed": 42,
        "questions": len(questions),
        "excluded_question_ids": EXCLUDED,
        "questions_source": {
            "repo": QUESTIONS_REPO,
            "revision": QUESTIONS_REVISION,
            "git_blob": QUESTIONS_BLOB,
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "sampling": SAMPLING,
        "shards": args.shards,
        "generation_versions": GENERATION_VERSIONS,
        "engine": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": "bfloat16",
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": 40960,
            "max_num_seqs": args.concurrency,
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "seed": 42,
            "generation_config": "vllm",
            "disable_log_stats": True,
            "enforce_eager": False,
        },
        "files_sha256": {str(p.relative_to(root)): digest(p) for p in files},
    }
    write(root / "plan.json", plan)
    print(
        json.dumps(
            {
                "prepared": str(root),
                "model": plan["model"],
                "questions": len(questions),
                "responses": len(questions) * 100,
                "shards": args.shards,
                "cuda_graphs": True,
            }
        )
    )


async def generate_async(root, plan, questions, shard, limit):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    jobs = [(q, a) for q in questions for a in range(100)]
    random.Random(42).shuffle(jobs)
    jobs = jobs[shard :: plan["shards"]]
    pending = []
    for q, a in jobs:
        path = root / "responses" / record_name(q, a)
        if path.exists():
            verify_response(read_gzip(path), plan, q, a)
        else:
            pending.append((q, a))
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print(json.dumps({"shard": shard, "pending": 0}))
        return
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=plan["model"]["repo"],
            revision=plan["model"]["revision"],
            tokenizer=str(root / "tokenizer"),
            **plan["engine"],
        )
    )
    queue = asyncio.Queue()
    for job in pending:
        queue.put_nowait(job)
    done = 0

    async def consume():
        nonlocal done
        while not queue.empty():
            q, a = queue.get_nowait()
            params = SamplingParams(
                **SAMPLING, seed=sample_seed(q["position"], a), output_kind=RequestOutputKind.FINAL_ONLY
            )
            output = None
            async for output in engine.generate(
                {"prompt_token_ids": q["prompt_token_ids"]}, params, request_id=f"{q['position']}-{a}"
            ):
                pass
            if output is None or not output.finished or len(output.outputs) != 1:
                raise RuntimeError("Generator returned an incomplete response")
            sample = output.outputs[0]
            record = {
                **response_identity(plan, q, a),
                "response": sample.text,
                "token_ids": list(sample.token_ids),
                "output_tokens": len(sample.token_ids),
                "finish_reason": sample.finish_reason,
            }
            verify_response(record, plan, q, a)
            await asyncio.to_thread(write_gzip, root / "responses" / record_name(q, a), record)
            done += 1
            if done % 100 == 0 or done == len(pending):
                status = {"shard": shard, "completed_this_invocation": done, "pending_at_start": len(pending)}
                write(root / "progress" / f"generate_{shard}.json", status)
                print(json.dumps(status), flush=True)

    tasks = [asyncio.create_task(consume()) for _ in range(plan["engine"]["max_num_seqs"])]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        engine.shutdown()


def generate(args):
    root, plan, questions = load_run(args.run_dir)
    if not 0 <= args.shard_index < plan["shards"]:
        raise ValueError("Shard index is outside this frozen run's shard count")
    if args.limit_requests is not None and args.limit_requests < 1:
        raise ValueError("Smoke-test request limit must be positive")
    for name, expected in plan["generation_versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Use the pinned evaluation version: {name}=={expected}")
    lock = root / f"generate_{args.shard_index}.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(generate_async(root, plan, questions, args.shard_index, args.limit_requests))


def initialize_grader(root_text):
    from tokenizers import Tokenizer

    from qwen3_experiments.compression_verl_reward import MathWorker

    global G_ROOT, G_PLAN, G_QUESTIONS, G_TOKENIZER, G_GRADER, G_CLOSE, G_FINGERPRINT
    G_ROOT, G_PLAN, G_QUESTIONS = load_run(Path(root_text))
    G_TOKENIZER = Tokenizer.from_file(str(G_ROOT / "tokenizer/tokenizer.json"))
    G_CLOSE = G_TOKENIZER.token_to_id("</think>")
    if G_CLOSE is None or G_TOKENIZER.decode([G_CLOSE], skip_special_tokens=True) != "</think>":
        G_CLOSE = None
    G_GRADER = MathWorker(G_ROOT / "grading_plan.json")
    G_FINGERPRINT = digest(G_ROOT / "grading_plan.json")
    atexit.register(G_GRADER.close)


def grade_question(index):
    from qwen3_experiments.capped_response_eval import score_caps
    from qwen3_experiments.compression_verl_reward import response_after_thinking

    q = G_QUESTIONS[index]
    lengths, scores, capped = [], [], []
    for attempt in range(100):
        name = record_name(q, attempt)
        path = G_ROOT / "responses" / name
        response = read_gzip(path)
        verify_response(response, G_PLAN, q, attempt)
        identity = {
            "response_sha256": digest(path),
            "grading_plan_sha256": G_FINGERPRINT,
            "question_id": q["unique_id"],
            "attempt": attempt,
        }
        target = G_ROOT / "grades" / name
        if target.exists():
            record = read_gzip(target)
            if any(record.get(k) != v for k, v in identity.items()):
                raise ValueError("Saved grading provenance changed")
        else:
            answer = response_after_thinking(response["response"])
            full = (
                G_GRADER.score(answer, q["ground_truth"])
                if answer is not None
                else {"score": 0.0, "grading_reason": "missing_or_invalid_thinking_close"}
            )
            decode = lambda ids: G_TOKENIZER.decode(ids, skip_special_tokens=True)
            decoded = decode(response["token_ids"])
            if decoded != response["response"] and not (
                len(response["token_ids"]) == 32768 and decoded == response["response"] + "\ufffd"
            ):
                raise ValueError("Saved token IDs do not decode to the saved response")
            caps = score_caps(
                response["token_ids"],
                response["response"],
                full["score"],
                CAPS,
                decode,
                lambda text: G_GRADER.score(text, q["ground_truth"]),
                thinking_close_id=G_CLOSE,
            )
            record = {**identity, "full": full, "caps": caps}
            write_gzip(target, record)
        values = [x["score"] for x in record["caps"]]
        if (
            len(values) != len(CAPS)
            or any(v not in (0, 1) for v in values)
            or record["full"]["score"] not in (0, 1)
            or values[-1] != record["full"]["score"]
            or [x["cap_tokens"] for x in record["caps"]] != CAPS.tolist()
        ):
            raise ValueError("Invalid binary prefix/full grades")
        lengths.append(response["output_tokens"])
        scores.append(int(record["full"]["score"]))
        capped.append(values)
    return index, lengths, scores, capped


def grade(args):
    root, plan, questions = load_run(args.run_dir)
    site = args.grader_site.resolve() if args.grader_site else root / "grader_site"
    if not site.is_dir() or args.workers < 1:
        raise ValueError("Install the isolated grading requirements and use a positive worker count")
    source = root / "runtime/qwen3_experiments/fixed100_legacy_math_scorer.py"
    grading_plan = {
        "site_packages": str(site),
        "scorer_path": str(source),
        "scorer_sha256": digest(source),
        "versions": GRADER_VERSIONS,
        "per_item_timeout_seconds": 1,
        "timeout_score": 0.0,
    }
    with (root / "grade.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "grading_plan.json").exists() and read(root / "grading_plan.json") != grading_plan:
            raise ValueError("Existing grader paths/config changed; preserve the original run location")
        write(root / "grading_plan.json", grading_plan)
        # A missing generation is an error, never silently excluded from accuracy.
        for q in questions:
            for attempt in range(100):
                if not (root / "responses" / record_name(q, attempt)).is_file():
                    raise ValueError(f"Missing response: {q['unique_id']}:{attempt}; finish all generation shards")
        lengths = np.zeros((len(questions), 100), dtype=np.int32)
        scores = np.zeros_like(lengths, dtype=np.int8)
        capped = np.zeros((len(questions), 100, len(CAPS)), dtype=np.int8)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_grader,
            initargs=(str(root),),
        ) as executor:
            futures = [executor.submit(grade_question, i) for i in range(len(questions))]
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                i, ls, rs, cs = future.result()
                lengths[i], scores[i], capped[i] = ls, rs, cs
                if done % 20 == 0 or done == len(questions):
                    print(json.dumps({"graded_questions": done, "questions": len(questions)}), flush=True)
        temporary = root / "samples.tmp.npz"
        np.savez_compressed(
            temporary,
            lengths=lengths,
            scores=scores,
            capped_scores=capped,
            caps=CAPS,
            original_positions=np.array([q["position"] for q in questions]),
        )
        temporary.replace(root / "samples.npz")
        write(
            root / "grading_receipt.json",
            {
                "complete": True,
                "responses": len(questions) * 100,
                "plan_sha256": plan["plan_sha256"],
                "samples_sha256": digest(root / "samples.npz"),
                "grading_plan_sha256": digest(root / "grading_plan.json"),
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Pin model/prompts and freeze a portable runtime; no GPU allocation")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision")
    p.add_argument("--model-key", default="l1_max_100_4000")
    p.add_argument("--label", default="L1-Max 100–4000")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.set_defaults(function=prepare)
    p = sub.add_parser("generate", help="Run one disjoint GPU shard; resume verified saved responses")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument(
        "--limit-requests", type=int, help="Optional small generation smoke test; rerun without this to finish"
    )
    p.set_defaults(function=generate)
    p = sub.add_parser("grade", help="Grade all full responses and binary token prefixes on CPUs")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--grader-site", type=Path)
    p.set_defaults(function=grade)
    p = sub.add_parser("analyze", help="Replay 100 shuffles and produce the four evaluation families")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--with-reference", action="store_true", help="Download small pinned arrays for the nine paper models"
    )
    p.add_argument("--reference-dir", type=Path, help="Use a local copy of the consolidated reference package")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.command != "prepare":
        runtime = args.run_dir / "runtime"
        if Path(__file__).resolve().parent.parent != runtime:
            # Every worker inherits this immutable runtime, independent of later checkout edits.
            command = [sys.executable, "-m", "qwen3_experiments.fixed100_math_eval", *sys.argv[1:]]
            for flag in ["--run-dir", "--grader-site", "--reference-dir"]:
                for i, argument in enumerate(command):
                    if argument == flag:
                        command[i + 1] = str(Path(command[i + 1]).resolve())
                    elif argument.startswith(flag + "="):
                        command[i] = flag + "=" + str(Path(argument.split("=", 1)[1]).resolve())
            env = {**os.environ, "PYTHONPATH": str(runtime)}
            raise SystemExit(subprocess.call(command, cwd=runtime, env=env))
    if args.command == "analyze":
        from qwen3_experiments.fixed100_math_analysis import analyze

        analyze(args)
    else:
        args.function(args)


if __name__ == "__main__":
    main()
