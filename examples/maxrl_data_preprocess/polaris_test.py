"""Reproduce a held-out Polaris 1/8 test set from the training manifest."""

import argparse
import hashlib
import json
import random
import re
import shutil
import unicodedata
from pathlib import Path

from datasets import Dataset, Features, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

TRAIN_REPO = "hi-todayis-jh/Polaris-1-8-3200"
TRAIN_REVISION = "b3bcd22296952f8100ad4b43037af24e79b6dbcb"
TRAIN_FILE = "data/train-00000-of-00001.parquet"
TARGET_REPO = "hi-todayis-jh/Polaris-Test"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_problem(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def build(output, seed=42):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = hf_hub_download(TRAIN_REPO, "sampling_manifest.json", repo_type="dataset", revision=TRAIN_REVISION)
    train_path = hf_hub_download(TRAIN_REPO, TRAIN_FILE, repo_type="dataset", revision=TRAIN_REVISION)
    training_manifest = json.loads(Path(manifest_path).read_text())
    source = training_manifest["source"]
    source_path = hf_hub_download(source["repo_id"], source["filename"], repo_type="dataset", revision=source["revision"])
    assert sha256(source_path) == source["sha256"]
    assert sha256(train_path) == training_manifest["output"]["data_sha256"]
    rows = [json.loads(line) for line in Path(source_path).read_text().splitlines() if line.strip()]
    train_rows = Dataset.from_parquet(train_path).to_list()
    train_indices = training_manifest["sampling"]["source_indices_in_dataset_order"]
    assert len(rows) == source["rows"] and len(train_rows) == len(train_indices) == 3200
    assert train_rows == [rows[index] for index in train_indices]
    filtering = training_manifest["filtering"]
    assert filtering["policy"] == "finite-real-no-explicit-proof-v1"
    eligible = filtering["eligible_source_indices_by_difficulty"]["1/8"]
    train_ids = set(train_indices)
    train_exact = {row["problem"] for row in train_rows}
    train_normalized = {normalized_problem(row["problem"]) for row in train_rows}
    pool, seen = [], set()
    exclusions = {"training_source_id": 0, "training_normalized_text": 0, "duplicate_normalized_text": 0}
    for index in eligible:
        row = rows[index]
        assert row["difficulty"] == "1/8"
        key = normalized_problem(row["problem"])
        if index in train_ids:
            exclusions["training_source_id"] += 1
        elif key in train_normalized:
            exclusions["training_normalized_text"] += 1
        elif key in seen:
            exclusions["duplicate_normalized_text"] += 1
        else:
            pool.append(index)
            seen.add(key)
    assert len(pool) >= 100
    indices = random.Random(seed).sample(pool, 100)
    selected = [{**rows[index], "source_index": index} for index in indices]
    assert len(set(indices)) == len({normalized_problem(row["problem"]) for row in selected}) == 100
    assert not set(indices) & train_ids
    assert not {row["problem"] for row in selected} & train_exact
    assert not {normalized_problem(row["problem"]) for row in selected} & train_normalized
    features = Features({"problem": Value("string"), "answer": Value("string"),
                         "difficulty": Value("string"), "source_index": Value("int64")})
    data = Dataset.from_list(selected, features=features)
    data_path = output / "data/test-00000-of-00001.parquet"
    data_path.parent.mkdir(exist_ok=True)
    if data_path.exists():
        assert Dataset.from_parquet(str(data_path)).to_list() == selected, "Existing test set differs"
    else:
        data.to_parquet(data_path)
    assert Dataset.from_parquet(str(data_path)).to_list() == selected
    manifest = {
        "schema_version": 1, "dataset_name": "Polaris-Test", "source": source,
        "excluded_training_dataset": {"repo_id": TRAIN_REPO, "revision": TRAIN_REVISION,
            "data_file": TRAIN_FILE, "data_sha256": sha256(train_path),
            "sampling_manifest_sha256": sha256(manifest_path), "rows": len(train_rows)},
        "filtering": {"policy": filtering["policy"], "origin": "Pinned training manifest eligibility list",
            "eligible_rows_before_training_exclusion": len(eligible), "remaining_pool_size": len(pool),
            "problem_normalization": "Unicode NFKC, followed by removal of all whitespace",
            "exclusions": exclusions, "eligible_source_indices_after_exclusion": pool},
        "sampling": {"seed": seed, "method": "random.Random(seed).sample(pool_in_source_order, 100)",
            "without_replacement": True, "source_indices_in_dataset_order": indices},
        "output": {"split": "test", "rows": len(data), "columns": data.column_names,
            "data_file": "data/test-00000-of-00001.parquet", "data_sha256": sha256(data_path),
            "source_fields_preserved_verbatim": True},
        "verification": {"training_rows_match_source_indices": True, "source_checksum": True,
            "training_checksum": True, "parquet_round_trip": True,
            "overlap_source_ids": 0, "overlap_exact_problems": 0, "overlap_normalized_problems": 0},
    }
    (output / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (output / "README.md").write_text(f"""---
license: apache-2.0
task_categories:
- text-generation
language:
- en
size_categories:
- n<1K
source_datasets:
- POLARIS-Project/Polaris-Dataset-53K
tags:
- math
- reasoning
- polaris
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*.parquet
---

# Polaris-Test

100 randomly sampled **1/8** Polaris questions, held out from
[`{TRAIN_REPO}`](https://huggingface.co/datasets/{TRAIN_REPO}/tree/{TRAIN_REVISION}).
The single split is `test`.

The source is [{source['repo_id']}](https://huggingface.co/datasets/{source['repo_id']}/tree/{source['revision']}).
This test set uses the same **finite-real-no-explicit-proof-v1** numerical-answer
and explicit-proof-wording filter as the pinned training set. The accepted source
indices are taken directly from that training set's sampling manifest.

After excluding training source IDs, matching question text (Unicode NFKC with
whitespace removed), and duplicate normalized questions, **{len(pool):,}** questions
remain eligible. Python `random.Random({seed}).sample(pool, 100)` samples uniformly
without replacement from this pool in source order. The returned order is retained.
Selection depends only on source/training data, never on model predictions.

The original `problem`, `answer`, and `difficulty` fields are unchanged.
`source_index` is the zero-based record position in the pinned source JSONL.
The original `1/8` difficulty is an upstream reference-model pass-rate label.

## Verified separation

- 100 unique source IDs and normalized question texts.
- Zero overlap with the pinned 3,200 training rows by source ID, exact text, or
  normalized text.
- Source and training file checksums verified; output Parquet round trip verified.
- Full source revisions, eligibility pool, selected indices and checksums are in
  [sampling_manifest.json](sampling_manifest.json).

This checks overlap with the named RL training dataset. It does not establish
absence from model pretraining or detect every paraphrase. Source answers are
preserved and have not been independently solved or relabeled.

## Usage

```python
from datasets import load_dataset
test = load_dataset("{TARGET_REPO}", split="test")
```

To reproduce, install `datasets` and `huggingface_hub`, then run
`python reproduce.py --output-dir ./reproduced --seed {seed}`.

## Attribution

Derived from the [POLARIS project](https://hkunlp.github.io/blog/2025/Polaris/)
dataset, which cites DeepScaleR-Preview-Dataset and AReal-boba-Data as sources.
Apache-2.0; see [LICENSE](LICENSE). Changes are selection, ordering, and the
addition of source indices and provenance documentation.
""", encoding="utf-8")
    script = Path(__file__).resolve()
    if script != (output / "reproduce.py").resolve():
        shutil.copy2(script, output / "reproduce.py")
    license_path = hf_hub_download(TRAIN_REPO, "LICENSE", repo_type="dataset", revision=TRAIN_REVISION)
    shutil.copy2(license_path, output / "LICENSE")
    return manifest, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    manifest, selected = build(args.output_dir, args.seed)
    print(json.dumps({"rows": 100, "eligible_pool": manifest["filtering"]["remaining_pool_size"],
                      "verification": manifest["verification"]}, indent=2), flush=True)
    if args.push:
        api = HfApi()
        assert api.whoami()["name"] == TARGET_REPO.split("/")[0]
        try:
            info = api.dataset_info(TARGET_REPO)
        except RepositoryNotFoundError:
            api.create_repo(TARGET_REPO, repo_type="dataset", private=False)
        else:
            old = hf_hub_download(TARGET_REPO, "sampling_manifest.json", repo_type="dataset", revision=info.sha)
            assert json.loads(Path(old).read_text()) == manifest, "Refusing to replace a different existing test set"
        commit = api.upload_folder(repo_id=TARGET_REPO, repo_type="dataset", folder_path=args.output_dir,
            allow_patterns=["README.md", "LICENSE", "reproduce.py", "sampling_manifest.json", "data/*.parquet"],
            commit_message="Add 100 held-out numerical Polaris 1/8 questions, seed 42")
        remote = load_dataset(TARGET_REPO, revision=commit.oid, split="test")
        assert remote.to_list() == selected, "Published data differs from verified local selection"
        receipt = {"repo_id": TARGET_REPO, "revision": commit.oid, "rows": len(remote),
                   "remote_round_trip": "passed", "url": f"https://huggingface.co/datasets/{TARGET_REPO}"}
        (args.output_dir.parent / "hub_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
