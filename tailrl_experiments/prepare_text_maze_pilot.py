"""Stream a small source prefix, excluding both released held-out splits."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATASET = "max-rl/maze_17x17_diverse_1.3m"
REVISION = "9b9ed56991cb045ba4227d9120dad337085db439"


def grid_key(sequence):
    tokens = sequence.split()
    return " ".join(tokens[tokens.index("GRID_START") + 1:tokens.index("GRID_END")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-mazes", type=int, default=8192)
    parser.add_argument("--eval-mazes", type=int, default=256)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    module_path = args.experiment / "src/to_rl_parquet.py"
    spec = importlib.util.spec_from_file_location("tailrl_dataset_builder", module_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    official_eval = pd.read_parquet(args.experiment / "data/eval1000/test_maze_17_continuous.parquet")
    excluded = {grid_key(row["ground_truth"]) for row in official_eval["reward_model"]}
    sft_test = json.loads((args.data_dir / "test.json").read_text())
    if isinstance(sft_test, dict):
        sft_test = sft_test["data"]
    excluded.update(grid_key(row["sequence"]) for row in sft_test)
    selected_eval = np.random.default_rng(0).choice(len(official_eval), args.eval_mazes, replace=False)
    eval_data = official_eval.iloc[np.sort(selected_eval)].copy()
    eval_data.to_parquet(args.data_dir / "pilot_eval.parquet", index=False)
    official_eval.to_parquet(args.data_dir / "official_eval1000.parquet", index=False)
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/main_1.3M.jsonl"
    rows = []
    seen = set()
    source_ids = []
    skipped = 0
    scanned = 0
    raw_path = args.data_dir / "pilot_source.jsonl"
    with requests.get(url, stream=True, timeout=(30, 120)) as response, raw_path.open("wb") as stream:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1024 * 1024):
            if not line:
                continue
            scanned += 1
            obj = json.loads(line)
            row = builder.build_rows([obj], "maze_17_continuous", idx_offset=len(rows))[0]
            key = grid_key(row["reward_model"]["ground_truth"])
            if key in excluded or key in seen:
                skipped += 1
                continue
            stream.write(line + b"\n")
            rows.append(row)
            seen.add(key)
            source_ids.append(obj["prompt_id"])
            if len(rows) % 2048 == 0:
                print(f"Selected {len(rows)} training mazes; scanned {scanned}", flush=True)
            if len(rows) >= args.train_mazes:
                break
    if len(rows) != args.train_mazes:
        raise ValueError(f"Expected {args.train_mazes} mazes, found {len(rows)}")
    assert not (seen & excluded)
    train_path = args.data_dir / "pilot_train.parquet"
    pd.DataFrame(rows).to_parquet(train_path, index=False)
    manifest = {
        "dataset": DATASET, "dataset_revision": REVISION, "source_file": "main_1.3M.jsonl",
        "selection": "first distinct source mazes after exclusion; no filtering on policy success",
        "train_count": len(rows), "eval_count": len(eval_data), "official_eval_count": len(official_eval),
        "excluded_unique_mazes": len(excluded), "source_rows_scanned": scanned, "skipped": skipped,
        "train_eval_overlap": 0, "eval_selection_seed": 0,
        "eval_source_indices": selected_eval.tolist(), "train_source_prompt_ids": source_ids,
        "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [train_path, args.data_dir / "pilot_eval.parquet", raw_path]},
    }
    (args.data_dir / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared train={len(rows)} / eval={len(eval_data)}; train/heldout overlap=0", flush=True)


if __name__ == "__main__":
    main()
