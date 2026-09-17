# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert hiyouga/math12k to verl's parquet format."""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

DATA_SOURCE = "hiyouga/math12k"
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def make_map_fn(split):
    def process_fn(example, idx):
        question = f"{example['problem']}{INSTRUCTION}"
        answer = str(example["answer"])

        if idx == 0:
            print("Question:", question)
            print("Answer:", type(answer), answer)

        return {
            "data_source": DATA_SOURCE,
            "id": idx,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": split, "index": idx},
        }

    return process_fn


def has_problem_and_answer(example):
    return bool(str(example["problem"]).strip() and str(example["answer"]).strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math12k")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    print(f"Loading {DATA_SOURCE} from Hugging Face...", flush=True)
    dataset = datasets.load_dataset(DATA_SOURCE, "default")

    local_dir = os.path.abspath(os.path.expanduser(args.local_dir))
    os.makedirs(local_dir, exist_ok=True)

    for split in ("train", "test"):
        raw_source = dataset[split]
        source = raw_source.filter(has_problem_and_answer)
        skipped = len(raw_source) - len(source)
        if skipped:
            print(f"Skipping {skipped} {split} rows with an empty problem or answer")
        converted = source.map(
            function=make_map_fn(split),
            with_indices=True,
            remove_columns=source.column_names,
        )
        converted.to_parquet(os.path.join(local_dir, f"{split}.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
