#!/usr/bin/env python3
"""Evaluate pass rate under a fixed per-prompt response-token budget."""

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

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

_SCORER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--checkpoint-repo", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/math500/test.parquet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-output-budget", type=int, default=3 * 4096)
    parser.add_argument("--per-rollout-cap", type=int, default=4096)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--grader-workers", type=int, default=8)
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


def rollout_seed(base_seed: int, prompt_position: int, rollout_index: int) -> int:
    """Return a deterministic seed shared across models for a rollout slot."""
    return (base_seed + 1_000_003 * prompt_position + rollout_index) % (2**31 - 1)


def main() -> None:
    args = parse_args()
    if args.total_output_budget < 1:
        raise ValueError("--total-output-budget must be positive")
    if args.per_rollout_cap < 1:
        raise ValueError("--per-rollout-cap must be positive")
    if not args.model_path.joinpath("config.json").is_file():
        raise FileNotFoundError(f"Merged model is missing: {args.model_path}")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset is missing: {args.dataset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model_label}_total_budget_{args.total_output_budget}_rollout_cap_{args.per_rollout_cap}"
    rollouts_path = args.output_dir / f"{stem}_rollouts.jsonl"
    prompts_path = args.output_dir / f"{stem}_prompts.jsonl"
    summary_path = args.output_dir / f"{stem}_summary.json"
    if any(path.exists() for path in (rollouts_path, prompts_path, summary_path)):
        raise FileExistsError(f"Refusing to overwrite existing result set: {stem}")

    dataset = pd.read_parquet(args.dataset)
    if len(dataset) != 500:
        raise ValueError(f"Expected 500 MATH-500 prompts, found {len(dataset)}")
    if set(dataset["data_source"]) != {"DigitalLearningGmbH/MATH-lighteval"}:
        raise ValueError("Unexpected data_source; refusing to use a different grader route")

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
        raise ValueError(f"A prompt has {longest_prompt} tokens, exceeding the training cap {args.max_prompt_len}")

    from vllm import LLM, SamplingParams

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

    remaining = [args.total_output_budget] * len(dataset)
    rollout_counts = [0] * len(dataset)
    rollout_records: list[dict[str, Any]] = []
    generation_started = time.time()
    generation_round = 0
    while any(value > 0 for value in remaining):
        active_positions = [position for position, value in enumerate(remaining) if value > 0]
        active_prompts = [prompt_token_ids[position] for position in active_positions]
        sampling_params = []
        for position in active_positions:
            rollout_index = rollout_counts[position]
            sampling_params.append(
                SamplingParams(
                    n=1,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=min(args.per_rollout_cap, remaining[position]),
                    ignore_eos=False,
                    logprobs=0,
                    detokenize=False,
                    seed=rollout_seed(args.seed, position, rollout_index),
                )
            )
        request_outputs = engine.generate(
            prompt_token_ids=active_prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        if len(request_outputs) != len(active_positions):
            raise RuntimeError("vLLM returned an unexpected number of requests")

        for position, parameters, request_output in zip(
            active_positions,
            sampling_params,
            request_outputs,
            strict=True,
        ):
            if len(request_output.outputs) != 1:
                raise RuntimeError("A rollout request did not return exactly one response")
            completion = request_output.outputs[0]
            token_ids = list(completion.token_ids)
            output_tokens = len(token_ids)
            if output_tokens < 1 or output_tokens > parameters.max_tokens:
                raise RuntimeError(f"Invalid rollout length {output_tokens} for cap {parameters.max_tokens}")
            before = remaining[position]
            remaining[position] -= output_tokens
            rollout_index = rollout_counts[position]
            rollout_counts[position] += 1
            row = dataset.iloc[position]
            rollout_records.append(
                {
                    "prompt_position": position,
                    "prompt_index": int(row["id"]),
                    "unique_id": str(row["unique_id"]),
                    "rollout_index": rollout_index,
                    "rollout_seed": parameters.seed,
                    "max_output_tokens": parameters.max_tokens,
                    "output_tokens": output_tokens,
                    "cumulative_output_tokens": args.total_output_budget - remaining[position],
                    "remaining_budget_before": before,
                    "remaining_budget_after": remaining[position],
                    "finish_reason": completion.finish_reason,
                    "ground_truth": str(row["reward_model"]["ground_truth"]),
                    "response": tokenizer.decode(token_ids, skip_special_tokens=True),
                }
            )
        generation_round += 1
        print(f"Completed rollout round {generation_round}: active={len(active_positions)}, remaining_prompts={sum(value > 0 for value in remaining)}")
    generation_seconds = time.time() - generation_started

    expected_tokens = len(dataset) * args.total_output_budget
    observed_tokens = sum(record["output_tokens"] for record in rollout_records)
    if observed_tokens != expected_tokens or any(remaining):
        raise RuntimeError(f"Budget accounting failed: expected {expected_tokens}, observed {observed_tokens}")

    scoring_inputs = [(record["response"], record["ground_truth"]) for record in rollout_records]
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
    for record, score in zip(rollout_records, scores, strict=True):
        record["score"] = float(score)

    records_by_prompt: list[list[dict[str, Any]]] = [[] for _ in range(len(dataset))]
    for record in rollout_records:
        records_by_prompt[record["prompt_position"]].append(record)

    prompt_records = []
    for position, records in enumerate(records_by_prompt):
        records.sort(key=lambda record: record["rollout_index"])
        row = dataset.iloc[position]
        success_indices = [record["rollout_index"] for record in records if record["score"] > 0]
        prompt_records.append(
            {
                "prompt_position": position,
                "prompt_index": int(row["id"]),
                "unique_id": str(row["unique_id"]),
                "list_size": len(records),
                "passed": bool(success_indices),
                "num_successes": len(success_indices),
                "first_success_rollout_index": (success_indices[0] if success_indices else None),
                "total_output_tokens": sum(record["output_tokens"] for record in records),
            }
        )

    list_sizes = np.asarray([record["list_size"] for record in prompt_records])
    response_lengths = np.asarray([record["output_tokens"] for record in rollout_records])
    passed = np.asarray([record["passed"] for record in prompt_records])
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
        "max_prompt_len": args.max_prompt_len,
        "longest_prompt_tokens": longest_prompt,
        "total_output_budget_per_prompt": args.total_output_budget,
        "per_rollout_output_cap": args.per_rollout_cap,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "seed_scheme": "(seed + 1000003 * prompt_position + rollout_index) mod (2^31 - 1)",
        "pass_at_realized_list_size": float(passed.mean()),
        "num_prompts_passed": int(passed.sum()),
        "total_rollouts": len(rollout_records),
        "total_generated_output_tokens": observed_tokens,
        "list_size_mean": float(list_sizes.mean()),
        "list_size_std": float(list_sizes.std()),
        "list_size_min": int(list_sizes.min()),
        "list_size_max": int(list_sizes.max()),
        "list_size_quantiles": {str(quantile): float(np.quantile(list_sizes, quantile)) for quantile in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "response_tokens_mean": float(response_lengths.mean()),
        "response_tokens_std": float(response_lengths.std()),
        "generation_rounds": generation_round,
        "generation_seconds": generation_seconds,
        "scoring_seconds": scoring_seconds,
        "packages": {package: importlib.metadata.version(package) for package in ("torch", "vllm", "transformers", "math-verify", "datasets")},
    }
    write_jsonl(rollouts_path, rollout_records)
    write_jsonl(prompts_path, prompt_records)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    main()
