"""Convert an existing, audited mix without selecting or dropping any questions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from qwen3_experiments.lcb_coding_format import LCB_REVISION, convert_truth, dumps


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_dataset(source, leetcode_source, destination):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    originals = pq.read_table(source).to_pylist()
    needed = {row["extra_info"]["problem_id"] for row in originals
              if row["extra_info"]["platform"] == "leetcode"}
    leetcode = {}
    with Path(leetcode_source).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["task_id"] in needed:
                leetcode[row["task_id"]] = row
    if set(leetcode) != needed:
        raise ValueError("Missing original LeetCode rows")
    records, audit = [], []
    for row in originals:
        old_serialized = row["reward_model"]["ground_truth"]
        old = json.loads(old_serialized)
        original = leetcode.get(row["extra_info"]["problem_id"])
        truth = convert_truth(old, original)
        num_tests = len(truth["input_output"]["inputs"])
        if num_tests != row["extra_info"]["num_tests"]:
            raise ValueError(f"Test count changed for {row['id']}")
        records.append({**row, "data_source": "lcb_coding",
                        "reward_model": {"style": "rule", "ground_truth": dumps(truth)}})
        audit.append({"id": row["id"], "platform": row["extra_info"]["platform"],
                      "problem_id": row["extra_info"]["problem_id"], "tests": num_tests,
                      "mode": "functional" if truth["input_output"]["fn_name"] else "stdin",
                      "adapter": truth["adapter"],
                      "old_ground_truth_sha256": hashlib.sha256(old_serialized.encode()).hexdigest(),
                      "new_ground_truth_sha256": hashlib.sha256(dumps(truth).encode()).hexdigest()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), destination, compression="zstd")
    reloaded = pq.read_table(destination).to_pylist()
    if reloaded != records:
        raise ValueError("Parquet roundtrip changed records")
    for old, new in zip(originals, reloaded):
        for field in ("id", "prompt", "ability", "extra_info"):
            if old[field] != new[field]:
                raise ValueError(f"Original {field} changed")
    summary = {"rows": len(records), "source_counts": dict(Counter(a["platform"] for a in audit)),
               "mode_counts": dict(Counter(a["mode"] for a in audit)),
               "adapter_questions": sum(a["adapter"] is not None for a in audit),
               "test_cases": sum(a["tests"] for a in audit),
               "lcb_revision": LCB_REVISION, "source_parquet_sha256": digest(source),
               "parquet_sha256": digest(destination), "leetcode_source_sha256": digest(leetcode_source),
               "all_ids_prompts_and_provenance_unchanged": True, "all_test_counts_preserved": True,
               "records": audit}
    destination.with_suffix(".conversion.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--leetcode-source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    summary = convert_dataset(args.source, args.leetcode_source, args.destination)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
