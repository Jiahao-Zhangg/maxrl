"""Lightweight shared bookkeeping for the resumable math evaluation matrix."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_config(path: Path) -> dict:
    config = read_json(path)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported matrix config version")
    for kind in ("models", "datasets"):
        keys = [item["key"] for item in config[kind]]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError(f"Missing or duplicate {kind}")
        for item in config[kind]:
            if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in item["key"]):
                raise ValueError("Keys must be lowercase path-safe identifiers")
            if len(item["revision"]) != 40:
                raise ValueError("All Hub inputs must be pinned to full commit hashes")
    for protocol, spec in config["evals"].items():
        if protocol not in ("eval1", "eval2", "eval3"):
            raise ValueError(f"Unknown protocol: {protocol}")
        if not spec["budgets"] or any(b < 1 for b in spec["budgets"]):
            raise ValueError("Budgets must be positive")
        if len(set(spec["budgets"])) != len(spec["budgets"]) or len(set(spec["seeds"])) != len(spec["seeds"]):
            raise ValueError("Duplicate budget or seed")
    if config["evals"].get("eval1", {}).get("samples_per_prompt", 4) != 4:
        raise ValueError("Eval1 is mean@4, not pass@4")
    return config


def make_tasks(config: dict) -> list[dict]:
    # Complete small benchmark/protocol groups first; all models see identical grids.
    datasets = sorted(config["datasets"], key=lambda item: item["expected_rows"])
    return [
        {"id": f"{model['key']}__{dataset['key']}__{protocol}", "model": model["key"],
         "dataset": dataset["key"], "protocol": protocol}
        for protocol in config["evals"]
        for dataset in datasets
        for model in config["models"]
    ]


def task_points(config: dict, task: dict):
    spec = config["evals"][task["protocol"]]
    # Finish all seeds at each budget together so an honest error bar is available early.
    for budget in spec["budgets"]:
        for seed in spec["seeds"]:
            yield {**task, "budget": budget, "seed": seed}


def point_directory(root: Path, point: dict) -> Path:
    return root / "results" / point["model"] / point["dataset"] / point["protocol"] / f"budget_{point['budget']}" / f"seed_{point['seed']}"


def rollout_seed(seed: int, position: int, attempt: int) -> int:
    # Mix the whole tuple: additive seed+attempt would make seed 1 attempt 0
    # identical to seed 0 attempt 1, invalidating independent seed repeats.
    value = f"math-eval-v1:{seed}:{position}:{attempt}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big") % (2**31 - 1)


def point_identity(point: dict, manifest: dict) -> dict:
    return {"manifest_fingerprint": manifest["fingerprint"], **point}


def completed_point(root: Path, point: dict, manifest: dict, *, check_hashes: bool = True) -> dict | None:
    path = point_directory(root, point) / "summary.json"
    if not path.exists():
        return None
    summary = read_json(path)
    if summary.get("identity") != point_identity(point, manifest) or summary.get("status") != "complete":
        raise ValueError(f"Incompatible saved point: {path}")
    for artifact in summary["artifacts"].values():
        file = path.parent / artifact["file"]
        if not file.is_file() or file.stat().st_size != artifact["size"]:
            raise ValueError(f"Incomplete saved artifact: {file}")
        if check_hashes and sha256(file) != artifact["sha256"]:
            raise ValueError(f"Saved artifact failed checksum: {file}")
    return summary
