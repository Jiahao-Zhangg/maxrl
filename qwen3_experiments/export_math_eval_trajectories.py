#!/usr/bin/env python3
"""Package only a published snapshot's verified raw trajectories for release assets."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

from math_eval_matrix_common import completed_point, point_directory, read_json, sha256, write_json


SECRET_PATTERN = re.compile(
    rb"(?:hf_[A-Za-z0-9]{20,}|gh[opusr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    rb"|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)
PUBLIC_FILES = ("run_manifest.json", "eval1_source_manifest.json", "point_receipts.csv", "all_points.csv")


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def check_public_bytes(data, private_prefixes):
    if SECRET_PATTERN.search(data) or any(prefix in data for prefix in private_prefixes):
        # Do not print the matched value: the diagnostic itself must not expose secrets.
        raise ValueError("Potential credential or private filesystem path; refusing publication")


def public_summary(summary):
    output = copy.deepcopy(summary)
    if "reused_from" in output:
        output["reused_from"].pop("output_root", None)
    return output


def inspect_rollouts(path, summary, private_prefixes):
    counts = Counter()
    with gzip.open(path, "rb") as stream:
        for line in stream:
            check_public_bytes(line, private_prefixes)
            record = json.loads(line)
            if record["score"] not in (0, 1) or record["output_tokens"] != len(record["output_token_ids"]):
                raise ValueError("Invalid saved rollout score/token count")
            counts["total_rollouts"] += 1
            counts["total_output_tokens"] += record["output_tokens"]
            counts["correct_rollouts"] += int(record["score"])
    if any(counts[key] != summary[key] for key in ("total_rollouts", "total_output_tokens", "correct_rollouts")):
        raise ValueError("Raw rollout counters do not match the saved summary")
    return counts


def add_bytes(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def add_file(archive, name, path):
    info = tarfile.TarInfo(name)
    info.size = path.stat().st_size
    info.mode = 0o644
    with path.open("rb") as stream:
        archive.addfile(info, stream)


def verify_archive(path, expected):
    seen = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("Unexpected, duplicate, or non-file archive member")
            size, digest = expected[member.name]
            hasher = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
            if member.size != size or hasher.hexdigest() != digest:
                raise ValueError("Archived file checksum mismatch")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Missing archive members")


def export_snapshot(root, snapshot, destination):
    root, snapshot, destination = Path(root).resolve(), Path(snapshot).resolve(), Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Export destination must be empty; existing artifacts are never overwritten")
    manifest = read_json(root / "manifest.json")
    published = read_json(snapshot / "snapshot.json")
    if manifest["fingerprint"] != published["manifest_fingerprint"]:
        raise ValueError("Source and published snapshot fingerprints differ")
    if read_json(snapshot / "run_manifest.json") != manifest:
        raise ValueError("Published run manifest differs from the source")
    with (snapshot / "point_receipts.csv").open(newline="") as stream:
        receipts = list(csv.DictReader(stream))
    if len(receipts) != published["completed_points"]:
        raise ValueError("Receipt count differs from the published snapshot")
    private_prefixes = [str(root).encode(), (str(Path.home()) + "/").encode()]
    public_files = {name: (snapshot / name).read_bytes() for name in PUBLIC_FILES}
    for data in public_files.values():
        check_public_bytes(data, private_prefixes)
    prefix = "math_eval_matrix_" + published["snapshot_date"].replace("-", "")
    if not re.fullmatch(r"math_eval_matrix_[0-9]{8}", prefix):
        raise ValueError("Invalid snapshot date")
    points, seen = [], set()
    for index, receipt in enumerate(receipts, 1):
        point = {key: receipt[key] for key in ("id", "model", "dataset", "protocol")}
        point.update(budget=int(receipt["budget"]), seed=int(receipt["seed"]))
        if any(not re.fullmatch(r"[a-z0-9_]+", point[key]) for key in ("model", "dataset", "protocol")):
            raise ValueError("Unsafe point identity")
        directory = point_directory(root, point).resolve()
        if not directory.is_relative_to(root) or directory in seen:
            raise ValueError("Unsafe or duplicate point directory")
        seen.add(directory)
        summary = completed_point(root, point, manifest, check_hashes=False)
        if summary is None or summary.get("raw_rollout_audit") != "passed":
            raise ValueError("Published point is missing or not audited")
        original_summary_hash = sha256(directory / "summary.json")
        if original_summary_hash != receipt["summary_sha256"]:
            raise ValueError("Source summary differs from the published receipt")
        if receipt["manifest_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Receipt fingerprint differs from the run")
        if set(summary["artifacts"]) != {"rollouts", "prompts"}:
            raise ValueError("Unexpected raw artifact types")
        files, sources = {}, {}
        for kind, artifact in summary["artifacts"].items():
            candidate = directory / artifact["file"]
            path = candidate.resolve()
            if not path.is_relative_to(directory) or candidate.is_symlink():
                raise ValueError("Unsafe raw artifact path")
            digest = sha256(path)
            if digest != artifact["sha256"] or digest != receipt[kind + "_sha256"]:
                raise ValueError("Source artifact differs from its summary/receipt")
            member = prefix + "/" + path.relative_to(root).as_posix()
            files[kind] = {"path": member, "size": path.stat().st_size, "sha256": digest}
            sources[kind] = path
        counts = inspect_rollouts(sources["rollouts"], summary, private_prefixes)
        check_public_bytes(sources["prompts"].read_bytes(), private_prefixes)
        exported_summary = json_bytes(public_summary(summary))
        check_public_bytes(exported_summary, private_prefixes)
        summary_name = prefix + "/" + (directory / "summary.json").relative_to(root).as_posix()
        files["summary"] = {"path": summary_name, "size": len(exported_summary),
                            "sha256": hashlib.sha256(exported_summary).hexdigest()}
        points.append({"point": point, "files": files, "sources": sources, "summary_bytes": exported_summary,
                       "original_summary_sha256": original_summary_hash, "counts": dict(counts)})
        if index % 25 == 0 or index == len(receipts):
            print(f"Verified/scanned {index}/{len(receipts)} points", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    export = {"schema_version": 1, "snapshot_date": published["snapshot_date"],
              "manifest_fingerprint": manifest["fingerprint"], "completed_points": len(points),
              "requested_points": published["requested_points"], "state": "partial",
              "raw_rollouts_byte_identical": True, "prompt_statistics_byte_identical": True,
              "summary_redactions": ["reused_from.output_root"],
              "full_input_dataset_snapshots_included": False, "archives": [], "points": []}
    for protocol in sorted({point["point"]["protocol"] for point in points}):
        selected = [point for point in points if point["point"]["protocol"] == protocol]
        filename = prefix.replace("_", "-") + "-" + protocol + ".tar"
        temporary = destination / (filename + ".tmp")
        expected = {}
        with tarfile.open(temporary, "w:") as archive:
            for name, data in public_files.items():
                member = prefix + "/" + name
                add_bytes(archive, member, data)
                expected[member] = (len(data), hashlib.sha256(data).hexdigest())
            for entry in selected:
                for kind, record in entry["files"].items():
                    if kind == "summary":
                        add_bytes(archive, record["path"], entry["summary_bytes"])
                    else:
                        add_file(archive, record["path"], entry["sources"][kind])
                    expected[record["path"]] = (record["size"], record["sha256"])
                export["points"].append({**entry["point"], "archive": filename, "files": entry["files"],
                                         "original_summary_sha256": entry["original_summary_sha256"], **entry["counts"]})
        if temporary.stat().st_size >= 2 * 2**30:
            raise ValueError("Archive exceeds the GitHub release asset limit; split before uploading")
        verify_archive(temporary, expected)
        path = destination / filename
        temporary.replace(path)
        totals = {key: sum(entry["counts"][key] for entry in selected)
                  for key in ("total_rollouts", "total_output_tokens", "correct_rollouts")}
        export["archives"].append({"name": filename, "protocol": protocol, "points": len(selected),
                                   "size": path.stat().st_size, "sha256": sha256(path), **totals})
        print(f"Packed and verified {filename}: {len(selected)} points", flush=True)
    index_path = destination / "raw_trajectories_manifest.json"
    write_json(index_path, export)
    sums = [f"{entry['sha256']}  {entry['name']}\n" for entry in export["archives"]]
    sums.append(f"{sha256(index_path)}  {index_path.name}\n")
    (destination / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")
    return export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    export = export_snapshot(args.output_root, args.snapshot_dir, args.export_dir)
    print(f"Ready to upload: {export['completed_points']}/{export['requested_points']} points", flush=True)


if __name__ == "__main__":
    main()
