#!/usr/bin/env python3
"""Evaluate one merged checkpoint on MATH-500 at one output-token budget."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


_SCORER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--checkpoint-repo", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/math500/test.parquet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-output-len",
        type=int,
        required=True,
        choices=(256, 512, 1024, 2048, 4096),
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--grader-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional prompt limit for smoke tests")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_chat(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [dict(message) for message in value]


def init_scorer() -> None:
    global _SCORER
    from verl.workers.reward_manager.multi_thread_naive import MathVerifyScorer

    _SCORER = MathVerifyScorer()


def score_response(item: tuple[str, str]) -> float:
    response, ground_truth = item
    return _SCORER.compute_score(
        model_output=response,
        ground_truth_unboxed=ground_truth,
        timeout_score=0.0,
        per_item_timeout_s=1,
    )


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.num_samples != 4:
        raise ValueError("This comparison requires exactly four samples per prompt")
    if not args.model_path.joinpath("config.json").is_file():
        raise FileNotFoundError(f"Merged model is missing: {args.model_path}")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset is missing: {args.dataset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model_label}_max_tokens_{args.max_output_len}"
    samples_path = args.output_dir / f"{stem}_samples.jsonl"
    summary_path = args.output_dir / f"{stem}_summary.json"

    dataset = pd.read_parquet(args.dataset)
    if len(dataset) != 500:
        raise ValueError(f"Expected 500 MATH-500 prompts, found {len(dataset)}")
    if set(dataset["data_source"]) != {"DigitalLearningGmbH/MATH-lighteval"}:
        raise ValueError("Unexpected data_source; refusing to use a different grader route")
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(dataset):
            raise ValueError(f"--limit must be between 1 and {len(dataset)}")
        dataset = dataset.iloc[: args.limit].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt_token_ids = [
        tokenizer.apply_chat_template(
            normalize_chat(chat),
            add_generation_prompt=True,
            tokenize=True,
        )
        for chat in dataset["prompt"]
    ]
    longest_prompt = max(map(len, prompt_token_ids))
    if longest_prompt > args.max_prompt_len:
        raise ValueError(
            f"A prompt has {longest_prompt} tokens, exceeding the training cap {args.max_prompt_len}"
        )

    from vllm import LLM, SamplingParams

    generation_started = time.time()
    engine = LLM(
        model=str(args.model_path),
        tokenizer=str(args.model_path),
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        max_model_len=32000,
        max_num_batched_tokens=32000,
        enforce_eager=False,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        disable_log_stats=True,
        seed=args.seed,
        trust_remote_code=False,
    )
    sampling = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_output_len,
        ignore_eos=False,
        logprobs=0,
        detokenize=False,
    )
    request_outputs = engine.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling,
        use_tqdm=True,
    )
    generation_seconds = time.time() - generation_started

    records: list[dict[str, Any]] = []
    scoring_inputs: list[tuple[str, str]] = []
    for row_position, request_output in enumerate(request_outputs):
        row = dataset.iloc[row_position]
        ground_truth = str(row["reward_model"]["ground_truth"])
        if len(request_output.outputs) != args.num_samples:
            raise RuntimeError(
                f"Prompt {row_position} returned {len(request_output.outputs)} samples"
            )
        for sample_index, completion in enumerate(request_output.outputs):
            token_ids = list(completion.token_ids)
            response = tokenizer.decode(token_ids, skip_special_tokens=True)
            records.append(
                {
                    "prompt_index": int(row["id"]),
                    "unique_id": str(row["unique_id"]),
                    "sample_index": sample_index,
                    "ground_truth": ground_truth,
                    "response": response,
                    "output_tokens": len(token_ids),
                    "finish_reason": completion.finish_reason,
                }
            )
            scoring_inputs.append((response, ground_truth))

    scoring_started = time.time()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.grader_workers,
        mp_context=context,
        initializer=init_scorer,
    ) as pool:
        scores = list(
            tqdm(
                pool.map(score_response, scoring_inputs, chunksize=8),
                total=len(scoring_inputs),
                desc="MathVerify",
            )
        )
    scoring_seconds = time.time() - scoring_started

    for record, score in zip(records, scores, strict=True):
        record["score"] = float(score)

    token_counts = [record["output_tokens"] for record in records]
    summary = {
        "model_label": args.model_label,
        "checkpoint_repo": args.checkpoint_repo,
        "model_path": str(args.model_path.resolve()),
        "dataset": "HuggingFaceH4/MATH-500",
        "dataset_file": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "grader": "verl.workers.reward_manager.multi_thread_naive.MathVerifyScorer",
        "grader_timeout_seconds": 1,
        "num_prompts": len(dataset),
        "num_samples_per_prompt": args.num_samples,
        "num_scored_responses": len(scores),
        "max_prompt_len": args.max_prompt_len,
        "longest_prompt_tokens": longest_prompt,
        "max_output_len": args.max_output_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "mean_at_4_accuracy": float(sum(scores) / len(scores)),
        "correct_responses": float(sum(scores)),
        "mean_output_tokens": float(sum(token_counts) / len(token_counts)),
        "max_observed_output_tokens": max(token_counts),
        "budget_exhaustion_rate": float(
            sum(count == args.max_output_len for count in token_counts) / len(token_counts)
        ),
        "generation_seconds": generation_seconds,
        "scoring_seconds": scoring_seconds,
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("torch", "vllm", "transformers", "math-verify", "datasets")
        },
    }
    write_jsonl(samples_path, records)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    main()
