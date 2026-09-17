"""Offline release export: only audited points, unchanged raw bytes, no private metadata."""

import csv
import gzip
import hashlib
import importlib
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen3_experiments"))
exporter = importlib.import_module("export_math_eval_trajectories")
common = importlib.import_module("math_eval_matrix_common")


def fixture_snapshot(tmp_path, response="2"):
    root, snapshot, destination = [tmp_path / name for name in ("source", "snapshot", "export")]
    manifest = {"fingerprint": "synthetic-test"}
    point = {"id": "m__d__eval2", "model": "m", "dataset": "d", "protocol": "eval2", "budget": 512, "seed": 0}
    directory = common.point_directory(root, point)
    directory.mkdir(parents=True)
    record = {"response": response, "ground_truth": "2", "output_tokens": 2, "output_token_ids": [1, 2], "score": 1}
    raw = gzip.compress((json.dumps(record) + "\n").encode(), mtime=0)
    (directory / "rollouts.jsonl.gz").write_bytes(raw)
    common.write_json(directory / "prompts.json", [{"unique_id": "d_0", "solved": True}])
    artifacts = {kind: {"file": name, "size": (directory / name).stat().st_size, "sha256": common.sha256(directory / name)}
                 for kind, name in (("rollouts", "rollouts.jsonl.gz"), ("prompts", "prompts.json"))}
    summary = {"status": "complete", "identity": common.point_identity(point, manifest), "artifacts": artifacts,
               "raw_rollout_audit": "passed", "total_rollouts": 1, "total_output_tokens": 2, "correct_rollouts": 1,
               "reused_from": {"output_root": str(root), "summary_sha256": "original-provenance"}}
    common.write_json(directory / "summary.json", summary)
    common.write_json(root / "manifest.json", manifest)
    common.write_json(snapshot / "run_manifest.json", manifest)
    common.write_json(snapshot / "eval1_source_manifest.json", {"fingerprint": "previous"})
    common.write_json(snapshot / "snapshot.json", {"snapshot_date": "2026-09-16", "completed_points": 1,
                                                 "requested_points": 2, "manifest_fingerprint": "synthetic-test"})
    receipt = {**point, "manifest_fingerprint": "synthetic-test", "summary_sha256": common.sha256(directory / "summary.json"),
               "rollouts_sha256": artifacts["rollouts"]["sha256"], "prompts_sha256": artifacts["prompts"]["sha256"]}
    with (snapshot / "point_receipts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(receipt))
        writer.writeheader()
        writer.writerow(receipt)
    (snapshot / "all_points.csv").write_text("model,dataset\nm,d\n")
    # A failed/incomplete attempt not named in the receipts must never be packaged.
    (directory / "unfinished.jsonl").write_text("not a completed artifact")
    return root, snapshot, destination, directory, raw


def test_export_preserves_raw_bytes_provenance_and_original_files(tmp_path):
    root, snapshot, destination, directory, raw = fixture_snapshot(tmp_path)
    before = (directory / "summary.json").read_bytes()
    manifest = exporter.export_snapshot(root, snapshot, destination)
    assert manifest["completed_points"] == 1 and manifest["requested_points"] == 2
    assert (directory / "summary.json").read_bytes() == before
    point = manifest["points"][0]
    assert point["original_summary_sha256"] == hashlib.sha256(before).hexdigest()
    assert point["total_rollouts"] == 1
    with tarfile.open(destination / point["archive"]) as archive:
        assert not any("unfinished" in name for name in archive.getnames())
        assert archive.extractfile(point["files"]["rollouts"]["path"]).read() == raw
        summary = json.load(archive.extractfile(point["files"]["summary"]["path"]))
        assert summary["reused_from"] == {"summary_sha256": "original-provenance"}
    assert str(root) not in json.dumps(manifest)
    assert (destination / "SHA256SUMS").is_file()
    with pytest.raises(ValueError, match="must be empty"):
        exporter.export_snapshot(root, snapshot, destination)


def test_export_rejects_changed_raw_bytes(tmp_path):
    root, snapshot, destination, directory, raw = fixture_snapshot(tmp_path)
    (directory / "rollouts.jsonl.gz").write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="summary/receipt"):
        exporter.export_snapshot(root, snapshot, destination)
    assert not destination.exists()


def test_export_rejects_secret_shaped_output_without_exposing_it(tmp_path):
    synthetic_secret = "hf_" + "A" * 32
    root, snapshot, destination, _, _ = fixture_snapshot(tmp_path, response=synthetic_secret)
    with pytest.raises(ValueError, match="Potential credential") as exc:
        exporter.export_snapshot(root, snapshot, destination)
    assert synthetic_secret not in str(exc.value)
    assert not destination.exists()


def test_export_rejects_inconsistent_snapshot_count(tmp_path):
    root, snapshot, destination, _, _ = fixture_snapshot(tmp_path)
    metadata = common.read_json(snapshot / "snapshot.json")
    metadata["completed_points"] = 2
    common.write_json(snapshot / "snapshot.json", metadata)
    with pytest.raises(ValueError, match="Receipt count"):
        exporter.export_snapshot(root, snapshot, destination)


def test_export_is_reproducible(tmp_path):
    root, snapshot, destination, _, _ = fixture_snapshot(tmp_path)
    first = exporter.export_snapshot(root, snapshot, destination)
    second = exporter.export_snapshot(root, snapshot, tmp_path / "second-export")
    assert first == second
