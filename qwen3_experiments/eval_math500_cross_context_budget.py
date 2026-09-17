#!/usr/bin/env python3
"""Evaluate a standardized math dataset with one shared output-token budget.

For each reference budget ``b``, the evaluator receives exactly ``N * b``
generated response tokens.  It repeatedly visits a deterministic permutation of
all ``N`` prompts, samples one response for every still-unsolved prompt, and skips
that prompt in later sweeps once MathVerify accepts any response.  Every normal
response is capped at 4,096 tokens; only the final request may receive a smaller
cap so the shared budget is not exceeded.

The optional random-with-replacement strategy replaces sweeps with IID prompt
draws from a model-independent random stream. Every draw is evaluated even if
that prompt was already solved, so this strategy performs no success-aware
allocation. Repeated prompts, including repeats within one batch, are allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

_SCORER = None
_PERMUTATION_SEED_STRIDE = 2_000_033
_ROLLOUT_SEED_STRIDE = 1_000_003
_SEED_MODULUS = 2**31 - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--checkpoint-repo", required=True)
    parser.add_argument("--checkpoint-revision", default=None)
    parser.add_argument("--dataset", type=Path, default=Path("data/math500/test.parquet"))
    parser.add_argument("--dataset-name", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--budget-per-prompt",
        type=int,
        nargs="+",
        default=(256, 512, 1024, 2048, 4096),
        help="Reference budgets b; each run receives b times 500 output tokens.",
    )
    parser.add_argument("--per-rollout-cap", type=int, default=4096)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument(
        "--question-selection",
        choices=("sweep", "random-with-replacement"),
        default="sweep",
        help=("Choose deterministic shuffled sweeps (the original Eval 4) or IID prompt draws with replacement."),
    )
    parser.add_argument("--grader-workers", type=int, default=8)
    parser.add_argument("--grader-timeout", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
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


def init_scorer(grader_timeout: int) -> None:
    global _SCORER
    from verl.workers.reward_manager.multi_thread_naive import MathVerifyScorer

    _SCORER = (MathVerifyScorer(), grader_timeout)


def score_response(item: tuple[str, str]) -> float:
    response, ground_truth = item
    scorer, grader_timeout = _SCORER
    return scorer.compute_score(
        model_output=response,
        ground_truth_unboxed=ground_truth,
        timeout_score=0.0,
        per_item_timeout_s=grader_timeout,
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
    """Return a model-independent seed for one prompt/attempt slot."""
    return (base_seed + _ROLLOUT_SEED_STRIDE * prompt_position + rollout_index) % _SEED_MODULUS


def permutation_seed(base_seed: int, round_index: int) -> int:
    """Return a model-independent seed for one complete MATH-500 sweep."""
    return (base_seed + _PERMUTATION_SEED_STRIDE * round_index) % _SEED_MODULUS


def round_permutation(num_prompts: int, base_seed: int, round_index: int) -> tuple[list[int], int, str]:
    """Create the shared deterministic question order for a sweep."""
    current_seed = permutation_seed(base_seed, round_index)
    positions = list(range(num_prompts))
    random.Random(current_seed).shuffle(positions)
    serialized = json.dumps(positions, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(serialized).hexdigest()
    return positions, current_seed, digest


def random_prompt_batch(
    *,
    num_prompts: int,
    rollout_counts: list[int],
    rng: random.Random,
    next_draw_index: int,
    batch_size: int,
) -> tuple[list[tuple[int, int, int]], int]:
    """Draw an IID batch from all prompts with replacement.

    The returned tuples contain ``(draw_index, prompt_position,
    rollout_index)``.  Rollout indices remain unique when the same prompt is
    drawn more than once in a batch.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(rollout_counts) != num_prompts:
        raise ValueError("Prompt state does not match num_prompts")

    batch: list[tuple[int, int, int]] = []
    occurrences: dict[int, int] = {}
    while len(batch) < batch_size:
        position = rng.randrange(num_prompts)
        draw_index = next_draw_index
        next_draw_index += 1
        occurrence = occurrences.get(position, 0)
        rollout_index = rollout_counts[position] + occurrence
        occurrences[position] = occurrence + 1
        batch.append((draw_index, position, rollout_index))
    return batch, next_draw_index


def result_paths(
    output_dir: Path,
    model_label: str,
    budget_per_prompt: int,
    total_budget: int,
    per_rollout_cap: int,
    question_selection: str = "sweep",
) -> tuple[Path, Path, Path]:
    selection_suffix = "" if question_selection == "sweep" else f"_question_selection_{question_selection.replace('-', '_')}"
    stem = f"{model_label}_budget_per_prompt_{budget_per_prompt}_global_budget_{total_budget}_rollout_cap_{per_rollout_cap}{selection_suffix}"
    return (
        output_dir / f"{stem}_rollouts.jsonl",
        output_dir / f"{stem}_prompts.jsonl",
        output_dir / f"{stem}_summary.json",
    )


def completed_result_is_reusable(
    paths: tuple[Path, Path, Path],
    *,
    model_label: str,
    checkpoint_repo: str,
    budget_per_prompt: int,
    total_budget: int,
    per_rollout_cap: int,
    seed: int,
    dataset_name: str,
    dataset_hash: str,
    checkpoint_revision: str | None,
    dataset_revision: str | None,
    question_selection: str = "sweep",
) -> bool:
    rollouts_path, prompts_path, summary_path = paths
    existing = [path.exists() for path in paths]
    if not any(existing):
        return False
    if not all(existing):
        raise FileExistsError("Incomplete result set exists; refusing to overwrite: " + ", ".join(str(path) for path, exists in zip(paths, existing) if exists))
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    evaluation = "eval4_cross_context_global_budget" if question_selection == "sweep" else "eval4_cross_context_random_with_replacement"
    expected = {
        "evaluation": evaluation,
        "model_label": model_label,
        "checkpoint_repo": checkpoint_repo,
        "budget_per_prompt_reference": budget_per_prompt,
        "total_global_output_budget": total_budget,
        "per_rollout_output_cap": per_rollout_cap,
        "seed": seed,
        "dataset": dataset_name,
        "dataset_sha256": dataset_hash,
    }
    if question_selection != "sweep":
        expected["question_selection"] = question_selection
    if checkpoint_revision is not None:
        expected["checkpoint_revision"] = checkpoint_revision
    if dataset_revision is not None:
        expected["dataset_revision"] = dataset_revision
    mismatches = {key: (summary.get(key), value) for key, value in expected.items() if summary.get(key) != value}
    if mismatches:
        raise ValueError(f"Existing result metadata does not match: {mismatches}")
    if not (summary.get("global_budget_exhausted") or summary.get("all_prompts_solved_early")):
        raise ValueError(f"Existing result ended prematurely: {summary_path}")
    if rollouts_path.stat().st_size == 0 or prompts_path.stat().st_size == 0:
        raise ValueError(f"Existing result contains an empty JSONL file: {summary_path}")
    return True


def numeric_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": int(values.min()),
        "max": int(values.max()),
        "quantiles": {str(quantile): float(np.quantile(values, quantile)) for quantile in (0.1, 0.25, 0.5, 0.75, 0.9)},
    }


def evaluate_budget(
    *,
    args: argparse.Namespace,
    dataset: pd.DataFrame,
    dataset_hash: str,
    tokenizer: Any,
    prompt_token_ids: list[list[int]],
    longest_prompt: int,
    engine: Any,
    sampling_params_type: Any,
    scorer_pool: ProcessPoolExecutor,
    budget_per_prompt: int,
) -> dict[str, Any]:
    total_budget = budget_per_prompt * len(dataset)
    paths = result_paths(
        args.output_dir,
        args.model_label,
        budget_per_prompt,
        total_budget,
        args.per_rollout_cap,
        args.question_selection,
    )
    if completed_result_is_reusable(
        paths,
        model_label=args.model_label,
        checkpoint_repo=args.checkpoint_repo,
        budget_per_prompt=budget_per_prompt,
        total_budget=total_budget,
        per_rollout_cap=args.per_rollout_cap,
        seed=args.seed,
        dataset_name=args.dataset_name,
        dataset_hash=dataset_hash,
        checkpoint_revision=args.checkpoint_revision,
        dataset_revision=args.dataset_revision,
        question_selection=args.question_selection,
    ):
        print(f"Reusing complete Eval 4 result: {paths[2]}", flush=True)
        with paths[2].open(encoding="utf-8") as stream:
            return json.load(stream)

    solved = [False] * len(dataset)
    rollout_counts = [0] * len(dataset)
    rollout_records: list[dict[str, Any]] = []
    round_records: list[dict[str, Any]] = []
    global_remaining = total_budget
    request_sequence_index = 0
    round_index = 0
    scoring_seconds = 0.0
    evaluation_started = time.time()
    generation_seconds = 0.0
    draw_rng = random.Random(args.seed)
    next_draw_index = 0

    while global_remaining > 0 and (args.question_selection == "random-with-replacement" or not all(solved)):
        if args.question_selection == "sweep":
            permutation, current_permutation_seed, permutation_digest = round_permutation(len(dataset), args.seed, round_index)
        else:
            permutation = []
            current_permutation_seed = None
            permutation_digest = None
        unsolved_at_start = sum(not value for value in solved)
        round_output_tokens = 0
        round_sampled = 0
        round_solved = 0
        round_first_draw_index = next_draw_index
        cursor = 0
        last_sampled_rank = None

        while global_remaining > 0 and (args.question_selection == "random-with-replacement" or cursor < len(permutation)):
            full_cap_slots = global_remaining // args.per_rollout_cap
            if args.question_selection == "sweep":
                candidates = [(rank, permutation[rank]) for rank in range(cursor, len(permutation)) if not solved[permutation[rank]]]
                if not candidates:
                    cursor = len(permutation)
                    break
                batch_size = min(
                    len(candidates),
                    args.max_batch_size,
                    max(1, full_cap_slots),
                )
                batch = [(rank, position, rollout_counts[position]) for rank, position in candidates[:batch_size]]
            else:
                batch_size = min(args.max_batch_size, max(1, full_cap_slots))
                batch, next_draw_index = random_prompt_batch(
                    num_prompts=len(dataset),
                    rollout_counts=rollout_counts,
                    rng=draw_rng,
                    next_draw_index=next_draw_index,
                    batch_size=batch_size,
                )
            batch_positions = [position for _, position, _ in batch]
            batch_prompts = [prompt_token_ids[position] for position in batch_positions]
            sampling_parameters = []
            reserved_budget = global_remaining
            for _, position, rollout_index in batch:
                max_tokens = min(args.per_rollout_cap, reserved_budget)
                if max_tokens < 1:
                    raise RuntimeError("Attempted to schedule a zero-token response")
                sampling_parameters.append(
                    sampling_params_type(
                        n=1,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_tokens=max_tokens,
                        ignore_eos=False,
                        logprobs=0,
                        detokenize=False,
                        seed=rollout_seed(
                            args.seed,
                            position,
                            rollout_index,
                        ),
                    )
                )
                reserved_budget -= max_tokens
            if sum(parameter.max_tokens for parameter in sampling_parameters) > global_remaining:
                raise RuntimeError("Batch reservations exceed the remaining global budget")

            batch_generation_started = time.time()
            request_outputs = engine.generate(
                prompt_token_ids=batch_prompts,
                sampling_params=sampling_parameters,
                use_tqdm=False,
            )
            generation_seconds += time.time() - batch_generation_started
            if len(request_outputs) != len(batch_positions):
                raise RuntimeError("vLLM returned an unexpected number of requests")

            pending_records = []
            for (
                selection_index,
                position,
                rollout_index,
            ), parameters, request_output in zip(
                batch,
                sampling_parameters,
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
                row = dataset.iloc[position]
                record = {
                    "prompt_position": position,
                    "prompt_index": int(row["id"]),
                    "unique_id": str(row["unique_id"]),
                    "budget_per_prompt_reference": budget_per_prompt,
                    "total_global_output_budget": total_budget,
                    "question_selection": args.question_selection,
                    "round_index": round_index,
                    "request_sequence_index": request_sequence_index,
                    "rollout_index": rollout_index,
                    "rollout_seed": parameters.seed,
                    "max_output_tokens": parameters.max_tokens,
                    "global_budget_limited_cap": (parameters.max_tokens < args.per_rollout_cap),
                    "output_tokens": output_tokens,
                    "finish_reason": completion.finish_reason,
                    "ground_truth": str(row["reward_model"]["ground_truth"]),
                    "response": tokenizer.decode(token_ids, skip_special_tokens=True),
                }
                if args.question_selection == "sweep":
                    record.update(
                        {
                            "round_permutation_seed": current_permutation_seed,
                            "round_permutation_sha256": permutation_digest,
                            "permutation_rank": selection_index,
                        }
                    )
                else:
                    record["question_draw_index"] = selection_index
                pending_records.append(record)
                request_sequence_index += 1

            scoring_inputs = [(record["response"], record["ground_truth"]) for record in pending_records]
            scoring_started = time.time()
            chunksize = max(1, len(scoring_inputs) // (args.grader_workers * 4))
            batch_scores = list(scorer_pool.map(score_response, scoring_inputs, chunksize=chunksize))
            scoring_seconds += time.time() - scoring_started

            for record, score in zip(pending_records, batch_scores, strict=True):
                position = record["prompt_position"]
                output_tokens = record["output_tokens"]
                before = global_remaining
                global_remaining -= output_tokens
                if global_remaining < 0:
                    raise RuntimeError("Generated responses exceeded the global budget")
                record["global_budget_before"] = before
                record["global_budget_after"] = global_remaining
                record["score"] = float(score)
                record["solved_before"] = solved[position]
                newly_solved = score > 0 and not solved[position]
                record["newly_solved"] = newly_solved
                if newly_solved:
                    solved[position] = True
                    round_solved += 1
                record["solved_after"] = solved[position]
                rollout_counts[position] += 1
                round_output_tokens += output_tokens
                round_sampled += 1
                last_sampled_rank = record.get("permutation_rank") if args.question_selection == "sweep" else record["question_draw_index"]
                rollout_records.append(record)

            if args.question_selection == "sweep":
                cursor = batch[-1][0] + 1
            else:
                break

        round_record = {
            "round_index": round_index,
            "unsolved_at_start": unsolved_at_start,
            "responses_sampled": round_sampled,
            "newly_solved": round_solved,
            "output_tokens": round_output_tokens,
            "global_budget_remaining_after": global_remaining,
        }
        if args.question_selection == "sweep":
            round_record.update(
                {
                    "permutation_seed": current_permutation_seed,
                    "permutation_sha256": permutation_digest,
                    "permutation_prefix": permutation[:10],
                    "completed_full_sweep": cursor >= len(permutation),
                    "last_sampled_permutation_rank": last_sampled_rank,
                }
            )
        else:
            round_record.update(
                {
                    "first_question_draw_index": round_first_draw_index,
                    "last_question_draw_index": last_sampled_rank,
                    "draws_are_iid_with_replacement": True,
                }
            )
        round_records.append(round_record)
        progress_label = "Eval4" if args.question_selection == "sweep" else "Eval4-RWR"
        print(
            f"{progress_label} b={budget_per_prompt} round={round_index}: sampled={round_sampled}, newly_solved={round_solved}, solved={sum(solved)}/{len(solved)}, remaining={global_remaining}",
            flush=True,
        )
        round_index += 1

    evaluation_seconds = time.time() - evaluation_started
    observed_tokens = sum(record["output_tokens"] for record in rollout_records)
    if observed_tokens + global_remaining != total_budget:
        raise RuntimeError(f"Global budget accounting failed: used={observed_tokens}, remaining={global_remaining}, total={total_budget}")
    if global_remaining and not all(solved):
        raise RuntimeError("Evaluation stopped before exhausting its global budget")

    records_by_prompt: list[list[dict[str, Any]]] = [[] for _ in range(len(dataset))]
    for record in rollout_records:
        records_by_prompt[record["prompt_position"]].append(record)

    prompt_records = []
    for position, records in enumerate(records_by_prompt):
        records.sort(key=lambda record: record["rollout_index"])
        row = dataset.iloc[position]
        success_records = [record for record in records if record["score"] > 0]
        prompt_records.append(
            {
                "prompt_position": position,
                "prompt_index": int(row["id"]),
                "unique_id": str(row["unique_id"]),
                "sampled": bool(records),
                "attempts": len(records),
                "solved": bool(success_records),
                "first_success_rollout_index": (success_records[0]["rollout_index"] if success_records else None),
                "first_success_round_index": (success_records[0]["round_index"] if success_records else None),
                "first_success_request_sequence_index": (success_records[0]["request_sequence_index"] if success_records else None),
                "total_output_tokens": sum(record["output_tokens"] for record in records),
            }
        )

    attempt_counts = np.asarray([record["attempts"] for record in prompt_records], dtype=np.int64)
    prompt_output_tokens = np.asarray(
        [record["total_output_tokens"] for record in prompt_records],
        dtype=np.int64,
    )
    response_lengths = np.asarray([record["output_tokens"] for record in rollout_records], dtype=np.int64)
    solved_array = np.asarray([record["solved"] for record in prompt_records], dtype=bool)
    sampled_array = attempt_counts > 0
    evaluation = "eval4_cross_context_global_budget" if args.question_selection == "sweep" else "eval4_cross_context_random_with_replacement"
    summary = {
        "evaluation": evaluation,
        "model_label": args.model_label,
        "checkpoint_repo": args.checkpoint_repo,
        "checkpoint_revision": args.checkpoint_revision,
        "model_path": str(args.model_path.resolve()),
        "dataset": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "dataset_file": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "grader": "verl.workers.reward_manager.multi_thread_naive.MathVerifyScorer",
        "grader_timeout_seconds": args.grader_timeout,
        "num_prompts": len(dataset),
        "max_prompt_len": args.max_prompt_len,
        "longest_prompt_tokens": longest_prompt,
        "budget_per_prompt_reference": budget_per_prompt,
        "total_global_output_budget": total_budget,
        "global_output_budget_used": observed_tokens,
        "global_output_budget_remaining": global_remaining,
        "global_budget_exhausted": global_remaining == 0,
        "all_prompts_solved_early": all(solved) and global_remaining > 0,
        "per_rollout_output_cap": args.per_rollout_cap,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "question_selection": args.question_selection,
        "rollout_seed_scheme": ("(seed + 1000003 * prompt_position + rollout_index) mod (2^31 - 1)"),
        "fraction_solved": float(solved_array.mean()),
        "num_prompts_solved": int(solved_array.sum()),
        "fraction_sampled": float(sampled_array.mean()),
        "num_prompts_sampled": int(sampled_array.sum()),
        "total_rollouts": len(rollout_records),
        "attempts_per_prompt": numeric_stats(attempt_counts),
        "output_tokens_per_prompt": numeric_stats(prompt_output_tokens),
        "response_tokens": numeric_stats(response_lengths),
        "rounds": round_records,
        "generation_seconds": generation_seconds,
        "scoring_seconds": scoring_seconds,
        "evaluation_seconds": evaluation_seconds,
        "packages": {
            package: importlib.metadata.version(package)
            for package in (
                "torch",
                "vllm",
                "transformers",
                "math-verify",
                "datasets",
            )
        },
    }
    if args.question_selection == "sweep":
        summary.update(
            {
                "permutation_seed_scheme": (f"(seed + 2000033 * round_index) mod (2^31 - 1); Python random.Random(seed).shuffle(range({len(dataset)}))"),
                "permutation_schedule_is_model_independent": True,
                "success_aware_skipping": True,
                "sweeps_started": len(round_records),
                "complete_sweeps": sum(record["completed_full_sweep"] for record in round_records),
            }
        )
    else:
        summary.update(
            {
                "question_draw_seed": args.seed,
                "question_draw_scheme": (f"Python random.Random(seed).randrange({len(dataset)}); IID with replacement"),
                "question_draw_stream_is_model_independent": True,
                "question_draws_include_already_solved_prompts": True,
                "success_aware_skipping": False,
                "total_question_draws": next_draw_index,
                "generation_batches": len(round_records),
            }
        )
    rollouts_path, prompts_path, summary_path = paths
    write_jsonl(rollouts_path, rollout_records)
    write_jsonl(prompts_path, prompt_records)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    budgets = list(dict.fromkeys(args.budget_per_prompt))
    if not budgets or any(budget < 1 for budget in budgets):
        raise ValueError("--budget-per-prompt values must be positive")
    if args.per_rollout_cap < 1:
        raise ValueError("--per-rollout-cap must be positive")
    if args.max_batch_size < 1:
        raise ValueError("--max-batch-size must be positive")
    if args.grader_workers < 1:
        raise ValueError("--grader-workers must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be strictly between 0 and 1")
    if not args.model_path.joinpath("config.json").is_file():
        raise FileNotFoundError(f"Merged model is missing: {args.model_path}")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset is missing: {args.dataset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.dataset)
    required_columns = {"id", "unique_id", "prompt", "reward_model"}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing standardized columns: {sorted(missing_columns)}")
    if not len(dataset):
        raise ValueError("Dataset contains no prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset_hash = sha256(args.dataset)
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
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_len + args.per_rollout_cap,
        max_num_batched_tokens=32000,
        enforce_eager=False,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        disable_log_stats=True,
        seed=args.seed,
        trust_remote_code=False,
    )

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.grader_workers,
        mp_context=context,
        initializer=init_scorer,
        initargs=(args.grader_timeout,),
    ) as scorer_pool:
        summaries = []
        for budget_per_prompt in budgets:
            summaries.append(
                evaluate_budget(
                    args=args,
                    dataset=dataset,
                    dataset_hash=dataset_hash,
                    tokenizer=tokenizer,
                    prompt_token_ids=prompt_token_ids,
                    longest_prompt=longest_prompt,
                    engine=engine,
                    sampling_params_type=SamplingParams,
                    scorer_pool=scorer_pool,
                    budget_per_prompt=budget_per_prompt,
                )
            )

    print(
        json.dumps(
            {
                "model_label": args.model_label,
                "results": [
                    {
                        "budget_per_prompt_reference": summary["budget_per_prompt_reference"],
                        "fraction_solved": summary["fraction_solved"],
                        "global_budget_exhausted": summary["global_budget_exhausted"],
                    }
                    for summary in summaries
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    main()
