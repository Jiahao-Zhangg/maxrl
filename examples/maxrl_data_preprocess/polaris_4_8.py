# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert the pinned POLARIS 4/8--7/8 subset to verl parquet format."""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

DATASET_REPO = "edbeeching/Polaris-Dataset-53K-4-8"
DATASET_REVISION = "ad1509a81e835614274acdb444c223164da0212c"
EXPECTED_ROWS = 20_371
EXPECTED_DIFFICULTIES = {"4/8", "5/8", "6/8", "7/8"}
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def has_required_text(example):
    return bool(str(example["problem"]).strip() and str(example["answer"]).strip())


def make_map_fn(example, idx):
    question = f"{example['problem']}{INSTRUCTION}"
    answer = str(example["answer"])

    if idx == 0:
        print("Question:", question)
        print("Answer:", type(answer), answer)

    return {
        "data_source": "polaris",
        "id": idx,
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": "train",
            "index": idx,
            "difficulty": example["difficulty"],
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/polaris_4_8")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    print(
        f"Loading {DATASET_REPO} at revision {DATASET_REVISION}...",
        flush=True,
    )
    source = datasets.load_dataset(
        DATASET_REPO,
        "default",
        split="train",
        revision=DATASET_REVISION,
    )
    missing_columns = {"problem", "answer", "difficulty"}.difference(
        source.column_names
    )
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    if len(source) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} pinned rows, found {len(source):,}"
        )

    difficulties = set(source.unique("difficulty"))
    if difficulties != EXPECTED_DIFFICULTIES:
        raise ValueError(
            "Unexpected difficulty values: "
            f"expected {sorted(EXPECTED_DIFFICULTIES)}, found {sorted(difficulties)}"
        )

    filtered = source.filter(has_required_text)
    skipped = len(source) - len(filtered)
    if skipped:
        print(f"Skipping {skipped} rows with an empty problem or answer")
    converted = filtered.map(
        function=make_map_fn,
        with_indices=True,
        remove_columns=filtered.column_names,
    )

    local_dir = os.path.abspath(os.path.expanduser(args.local_dir))
    os.makedirs(local_dir, exist_ok=True)
    converted.to_parquet(os.path.join(local_dir, "train.parquet"))
    with open(
        os.path.join(local_dir, "dataset_metadata.json"),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            {
                "dataset_repo": DATASET_REPO,
                "dataset_revision": DATASET_REVISION,
                "source_rows": len(source),
                "training_rows": len(converted),
                "difficulties": sorted(difficulties),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
