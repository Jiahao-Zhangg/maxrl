"""Prepare the exported ER training subset with its original DeepSeek prompt."""

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


def convert_example(example, idx):
    """Keep the original fields and supply a bare gold answer for MathVerify."""
    for field in ("problem", "solution", "extracted"):
        if not isinstance(example[field], str) or not example[field].strip():
            raise ValueError(f"Row {idx} has an empty {field}; refusing to drop training rows")

    # All 3,200 exported answers match the last boxed answer in the solution.
    # Validate this relationship instead of grading against a worked solution
    # or silently trusting an unrelated answer field.
    boxed = last_boxed_only_string(example["solution"])
    if boxed is None or remove_boxed(boxed).strip() != example["extracted"].strip():
        raise ValueError(f"Row {idx}: extracted does not match the last boxed solution answer")

    return {
        "data_source": DATASET_REPO,
        "id": idx,
        # Already rendered: the launcher must set data.apply_chat_template=false
        # to avoid duplicate role markers or an extra <think> prefill.
        "prompt": [{"role": "user", "content": INPUT_TEMPLATE.format(example["problem"])}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": example["extracted"].strip()},
        "extra_info": {"split": "train", "index": idx},
    }


def convert_dataset(source):
    missing = {"problem", "solution", "extracted"}.difference(source.column_names)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if len(source) != EXPECTED_ROWS:
        raise ValueError(f"Expected the {EXPECTED_ROWS}-row ER training subset, found {len(source)} rows")
    # Preserve source order and all original columns; shuffle in the trainer.
    return source.map(convert_example, with_indices=True, load_from_cache_file=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--dataset_repo", default=DATASET_REPO, choices=[DATASET_REPO])
    parser.add_argument("--revision", default=DATASET_REVISION)
    args = parser.parse_args()

    source = datasets.load_dataset(args.dataset_repo, split="train", revision=args.revision)
    converted = convert_dataset(source)
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    destination = local_dir / "train.parquet"
    converted.to_parquet(destination)
    metadata = {
        "dataset_repo": args.dataset_repo,
        "dataset_revision": args.revision,
        "source_rows": len(source),
        "training_rows": len(converted),
        "input_template": INPUT_TEMPLATE,
        "apply_chat_template": False,
        "ground_truth_field": "extracted (validated against solution)",
    }
    (local_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(converted)} training rows in source order: {destination}")


if __name__ == "__main__":
    main()
