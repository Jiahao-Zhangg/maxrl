"""Prepare the pinned ER training subset for DeepSeek or Qwen3 thinking prompts."""

import argparse
import json
from pathlib import Path

import datasets

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

DATASET_REPO = "zjhhhh/compression_dataset"
DATASET_REVISION = "bfdd7af1633ecc6db191a9f28f76449165a4ee06"
EXPECTED_ROWS = 3200
INPUT_TEMPLATE = (
    "<｜begin▁of▁sentence｜><｜User｜>"
    "Please reason step by step, and put your final answer within \\boxed{{}}. "
    "Question: {}<｜Assistant｜>"
)
QWEN3_INPUT_TEMPLATE = "{}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
PROMPT_FORMATS = ("deepseek", "qwen3")


def convert_example(example, idx, prompt_format="deepseek"):
    """Keep the original fields and supply a bare gold answer for MathVerify."""
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"Unknown prompt format: {prompt_format}")
    for field in ("problem", "solution", "extracted"):
        if not isinstance(example[field], str) or not example[field].strip():
            raise ValueError(f"Row {idx} has an empty {field}; refusing to drop training rows")

    # All 3,200 exported answers match the last boxed answer in the solution.
    # Validate this relationship instead of grading against a worked solution
    # or silently trusting an unrelated answer field.
    boxed = last_boxed_only_string(example["solution"])
    if boxed is None or remove_boxed(boxed).strip() != example["extracted"].strip():
        raise ValueError(f"Row {idx}: extracted does not match the last boxed solution answer")

    template = INPUT_TEMPLATE if prompt_format == "deepseek" else QWEN3_INPUT_TEMPLATE
    return {
        "data_source": DATASET_REPO,
        "id": idx,
        # DeepSeek is already rendered; Qwen3 needs its own thinking chat template.
        "prompt": [{"role": "user", "content": template.format(example["problem"])}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": example["extracted"].strip()},
        "extra_info": {"split": "train", "index": idx},
    }


def convert_dataset(source, prompt_format="deepseek"):
    missing = {"problem", "solution", "extracted"}.difference(source.column_names)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if len(source) != EXPECTED_ROWS:
        raise ValueError(f"Expected the {EXPECTED_ROWS}-row ER training subset, found {len(source)} rows")
    # Preserve source order and all original columns; shuffle in the trainer.
    return source.map(
        convert_example, with_indices=True, fn_kwargs={"prompt_format": prompt_format},
        load_from_cache_file=False,
    )


def audit_qwen3_prompts(converted, tokenizer, max_prompt_length):
    """Reject template changes and overlong prompts instead of losing training rows."""
    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive")
    lengths = []
    for index, row in enumerate(converted):
        prompt = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
        thinking_prompt = tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=True, tokenize=False, enable_thinking=True,
        )
        non_thinking_prompt = tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        if prompt != thinking_prompt or prompt == non_thinking_prompt:
            raise ValueError(f"Row {index}: tokenizer must enable Qwen3 thinking by default")
        lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
    overlong = [index for index, length in enumerate(lengths) if length > max_prompt_length]
    if overlong:
        raise ValueError(
            f"{len(overlong)} prompts exceed {max_prompt_length} tokens (maximum {max(lengths)}); "
            "increase the prompt limit instead of filtering or truncating the compression subset"
        )
    return {
        "max_prompt_length": max_prompt_length,
        "max_observed_prompt_tokens": max(lengths),
        "verified_prompt_count": len(lengths),
        "overlong_prompts": 0,
        "enable_thinking": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--dataset_repo", default=DATASET_REPO, choices=[DATASET_REPO])
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--prompt_format", choices=PROMPT_FORMATS, default="deepseek")
    parser.add_argument("--tokenizer", help="Required for auditing Qwen3 thinking prompts")
    parser.add_argument("--tokenizer_revision")
    parser.add_argument("--max_prompt_length", type=int, default=1280)
    args = parser.parse_args()
    if args.prompt_format == "qwen3" and not args.tokenizer:
        parser.error("--tokenizer is required for --prompt_format=qwen3")

    source = datasets.load_dataset(args.dataset_repo, split="train", revision=args.revision)
    converted = convert_dataset(source, prompt_format=args.prompt_format)
    prompt_audit = {}
    if args.prompt_format == "qwen3":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
        prompt_audit = audit_qwen3_prompts(converted, tokenizer, args.max_prompt_length)
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    destination = local_dir / "train.parquet"
    converted.to_parquet(destination)
    if args.prompt_format == "qwen3":
        # A nonempty loader is required even with all validation disabled.
        converted.select([0]).to_parquet(local_dir / "unused_validation.parquet")
    metadata = {
        "dataset_repo": args.dataset_repo,
        "dataset_revision": args.revision,
        "source_rows": len(source),
        "training_rows": len(converted),
        "input_template": INPUT_TEMPLATE if args.prompt_format == "deepseek" else QWEN3_INPUT_TEMPLATE,
        "prompt_format": args.prompt_format,
        "apply_chat_template": args.prompt_format == "qwen3",
        "ground_truth_field": "extracted (validated against solution)",
    }
    if args.prompt_format == "qwen3":
        metadata.update(prompt_audit)
        metadata.update(tokenizer=args.tokenizer, tokenizer_revision=args.tokenizer_revision)
    (local_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(converted)} training rows in source order: {destination}")


if __name__ == "__main__":
    main()
