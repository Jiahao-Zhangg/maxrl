"""Convert the entire pinned maze corpus with bounded memory and held-out checks."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATASET_REVISION = "9b9ed56991cb045ba4227d9120dad337085db439"
RAW_SHA256 = "c1d055238c80a042f89b5f661e6779773f16cadb084c29d39f0c3eddb4fd59bc"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def grid_digest(sequence):
    tokens = sequence.split()
    grid = " ".join(tokens[tokens.index("GRID_START") + 1 : tokens.index("GRID_END")])
    return hashlib.sha256(grid.encode()).digest()


def prepare(experiment, data_dir):
    raw = data_dir / "main_1.3M.jsonl"
    if digest(raw) != RAW_SHA256:
        raise ValueError("Raw corpus does not match the pinned release")
    spec = importlib.util.spec_from_file_location("tailrl_dataset_builder", experiment / "src/to_rl_parquet.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    evaluation = pq.read_table(experiment / "data/eval1000/test_maze_17_continuous.parquet")
    if evaluation.num_rows != 1000:
        raise ValueError("Expected all 1000 released evaluation contexts")
    eval_keys = {grid_digest(row["ground_truth"]) for row in evaluation["reward_model"].to_pylist()}
    sft_test = json.loads((data_dir / "test.json").read_text())
    if isinstance(sft_test, dict):
        sft_test = sft_test["data"]
    sft_keys = {grid_digest(row["sequence"]) for row in sft_test}
    excluded = eval_keys | sft_keys
    seen = set()
    rows = []
    count = scanned = duplicates = excluded_eval = excluded_sft = 0
    train_path = data_dir / "full_train.parquet"
    temporary = train_path.with_suffix(".parquet.incomplete")
    writer = None
    try:
        with raw.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                scanned += 1
                obj = json.loads(line)
                row = builder.build_rows([obj], "maze_17_continuous", idx_offset=count)[0]
                key = grid_digest(row["reward_model"]["ground_truth"])
                if key in excluded:
                    excluded_eval += key in eval_keys
                    excluded_sft += key in sft_keys
                    continue
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                rows.append(row)
                count += 1
                if len(rows) == 2048:
                    table = pa.Table.from_pylist(rows)
                    if writer is None:
                        writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                    writer.write_table(table)
                    rows.clear()
                if scanned % 100000 == 0:
                    print(f"Converted {count} training contexts from {scanned} source mazes", flush=True)
        if scanned != 1299992 or count < 1298000 or writer is None:
            raise ValueError(f"Unexpected full-corpus size: scanned={scanned}, retained={count}")
        if rows:
            writer.write_table(pa.Table.from_pylist(rows))
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(train_path)
    evaluation_path = data_dir / "full_eval.parquet"
    pq.write_table(evaluation, evaluation_path)
    manifest = {
        "dataset": "max-rl/maze_17x17_diverse_1.3m",
        "revision": DATASET_REVISION,
        "raw_sha256": RAW_SHA256,
        "source_count": scanned,
        "train_count": count,
        "eval_count": 1000,
        "selection": "entire corpus, excluding released eval1000/SFT test grids and duplicate grids",
        "excluded_eval": excluded_eval,
        "excluded_sft": excluded_sft,
        "duplicate_grids": duplicates,
        "train_eval_overlap": len(seen & eval_keys),
        "train_sft_test_overlap": len(seen & sft_keys),
        "files_sha256": {p.name: digest(p) for p in (train_path, evaluation_path)},
    }
    (data_dir / "full_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.experiment.resolve(), args.data_dir.resolve())
