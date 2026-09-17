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
"""Utilities for persisting training rollouts as a Hugging Face dataset."""

import gzip
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_MANIFEST_FILENAME = "rollout_manifest.json"
_RESERVED_COLUMNS = {"input", "output", "rollout_index", "score", "step"}


def _json_default(value: Any) -> Any:
    """Convert common tensor/NumPy scalar values into JSON-compatible values."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
            output.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_manifest(local_dir: Path) -> dict[str, Any]:
    manifest_path = local_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return {"format_version": 1, "steps": {}}
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("steps"), dict):
        raise ValueError(f"Unsupported rollout manifest: {manifest_path}")
    return manifest


def dump_rollout_step(
    local_dir: str | os.PathLike[str],
    *,
    step: int,
    inputs: Sequence[str],
    outputs: Sequence[str],
    scores: Sequence[float],
    extra_fields: Mapping[str, Sequence[Any]] | None = None,
) -> Path:
    """Atomically write every rollout from one completed training step.

    Each step is a separate gzip-compressed JSONL shard. Repeating a step after
    checkpoint recovery replaces that step's shard and manifest entry instead
    of producing duplicate records.
    """
    if step < 0:
        raise ValueError("rollout step must be nonnegative")
    num_rollouts = len(inputs)
    if len(outputs) != num_rollouts or len(scores) != num_rollouts:
        raise ValueError("inputs, outputs, and scores must have matching lengths")

    aligned_extra_fields: dict[str, Sequence[Any]] = {}
    for key, values in (extra_fields or {}).items():
        if key in _RESERVED_COLUMNS:
            raise ValueError(f"rollout extra field uses reserved column name: {key}")
        if isinstance(values, (str, bytes)):
            continue
        try:
            values_length = len(values)
        except TypeError:
            continue
        if values_length == num_rollouts:
            aligned_extra_fields[key] = values

    output_root = Path(local_dir).expanduser().resolve()
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    relative_path = Path("data") / f"step_{step:06d}.jsonl.gz"
    output_path = output_root / relative_path
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")

    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=1) as output_file:
            for rollout_index in range(num_rollouts):
                record = {
                    "step": step,
                    "rollout_index": rollout_index,
                    "input": inputs[rollout_index],
                    "output": outputs[rollout_index],
                    "score": scores[rollout_index],
                }
                record.update(
                    {key: values[rollout_index] for key, values in aligned_extra_fields.items()}
                )
                output_file.write(
                    json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
                )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    manifest = _load_manifest(output_root)
    manifest["steps"][str(step)] = {
        "file": relative_path.as_posix(),
        "num_rollouts": num_rollouts,
    }
    manifest["num_steps"] = len(manifest["steps"])
    manifest["num_rollouts"] = sum(
        int(step_metadata["num_rollouts"])
        for step_metadata in manifest["steps"].values()
    )
    _write_json_atomic(output_root / _MANIFEST_FILENAME, manifest)
    return output_path


def _write_dataset_card(local_dir: Path, metadata: Mapping[str, Any]) -> None:
    experiment_name = str(metadata.get("experiment_name", "training rollouts"))
    card = f"""---
pretty_name: {json.dumps(experiment_name + " rollouts")}
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/*.jsonl.gz"
---

# {experiment_name} rollouts

This dataset contains one compressed JSONL shard for every completed training
step. The `step` and `rollout_index` columns uniquely locate a rollout within
this training run. Run metadata and per-step row counts are recorded in
`{_MANIFEST_FILENAME}`.
"""
    card_path = local_dir / "README.md"
    temporary_path = card_path.with_name(f".{card_path.name}.tmp-{os.getpid()}")
    try:
        temporary_path.write_text(card, encoding="utf-8")
        os.replace(temporary_path, card_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _finalize_manifest(local_dir: Path, metadata: Mapping[str, Any]) -> None:
    manifest = _load_manifest(local_dir)
    if not manifest["steps"]:
        raise ValueError(f"No rollout step shards found under {local_dir}")
    manifest["run"] = dict(metadata)
    _write_json_atomic(local_dir / _MANIFEST_FILENAME, manifest)
    _write_dataset_card(local_dir, metadata)


def _local_upload_files(local_dir: Path) -> dict[str, int]:
    files = {}
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(local_dir)
        if ".cache" in relative_path.parts or ".tmp-" in path.name:
            continue
        files[relative_path.as_posix()] = path.stat().st_size
    return files


def upload_rollout_dataset_to_hf(
    local_dir: str | os.PathLike[str],
    *,
    repo_id: str,
    private: bool = False,
    num_workers: int = 4,
    metadata: Mapping[str, Any] | None = None,
    api: Any = None,
    verify_attempts: int = 6,
    verify_sleep_seconds: float = 10.0,
) -> str:
    """Upload and verify a completed rollout directory as an HF dataset repo."""
    if repo_id.count("/") != 1 or any(not component for component in repo_id.split("/")):
        raise ValueError("rollout dataset repo_id must have the form owner/name")
    if num_workers < 1:
        raise ValueError("rollout dataset upload workers must be positive")
    if verify_attempts < 1:
        raise ValueError("verify_attempts must be positive")

    output_root = Path(local_dir).expanduser().resolve()
    _finalize_manifest(output_root, metadata or {})
    local_files = _local_upload_files(output_root)
    if not any(path.startswith("data/") for path in local_files):
        raise ValueError(f"No rollout data files found under {output_root}")

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    if hasattr(api, "update_repo_settings"):
        api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=private)

    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=str(output_root),
            repo_type="dataset",
            private=private,
            allow_patterns=["README.md", _MANIFEST_FILENAME, "data/**"],
            ignore_patterns=["**/.cache/**", "**/.tmp-*"],
            num_workers=num_workers,
        )
    else:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(output_root),
            repo_type="dataset",
            allow_patterns=["README.md", _MANIFEST_FILENAME, "data/**"],
            ignore_patterns=["**/.cache/**", "**/.tmp-*"],
            commit_message="Upload completed training rollouts",
        )

    last_error = "repository metadata was unavailable"
    for attempt in range(verify_attempts):
        try:
            info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
            remote_files = {item.rfilename: item.size for item in info.siblings}
            missing = sorted(set(local_files) - set(remote_files))
            wrong_size = sorted(
                path for path, size in local_files.items() if remote_files.get(path) != size
            )
            if not missing and not wrong_size:
                return f"https://huggingface.co/datasets/{repo_id}"
            last_error = f"missing={missing[:3]}, wrong_size={wrong_size[:3]}"
        except Exception as error:
            last_error = str(error)
        if attempt + 1 < verify_attempts:
            time.sleep(verify_sleep_seconds)
    raise RuntimeError(f"rollout dataset upload verification failed: {last_error}")
