#!/usr/bin/env python3
"""Prepare a pinned FSDP actor without downloading optimizer or training state."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_math500_offset256_comparison import DATASET_SHA256, REPO_ROOT, sha256, write_json

TOKENIZER_FILES = ("added_tokens.json", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--step", type=int, default=150)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 30 * 2**30:
        raise RuntimeError("Need at least 30 GiB free to prepare and retain this checkpoint")
    reference = json.loads(args.reference_manifest.read_text())
    if sha256(REPO_ROOT / "data/math500/test.parquet") != DATASET_SHA256:
        raise ValueError("MATH-500 differs from the historical snapshot")
    prepared_path = root / "prepared.json"
    destination = root / "model"
    if prepared_path.is_file():
        previous = json.loads(prepared_path.read_text())
        if previous["checkpoint_repo"] != args.repo or previous["checkpoint_revision"] != args.revision:
            raise ValueError("Prepared model belongs to another checkpoint")
        if {p.name: sha256(p) for p in destination.iterdir() if p.is_file()} != previous["model_sha256"]:
            raise ValueError("Prepared model files have changed")
        print(f"Verified existing prepared model: {destination}", flush=True)
        return
    if destination.exists() or (root / "model.tmp").exists():
        raise FileExistsError("Unrecorded model output needs inspection before resuming")

    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise ValueError("The Hub returned a different checkpoint revision")
    prefix = f"global_step_{args.step}/actor/"
    names = [*TOKENIZER_FILES, "config.json", "generation_config.json", *(f"model_world_size_4_rank_{rank}.pt" for rank in range(4))]
    files = {f.rfilename: f for f in info.siblings}
    if any(prefix + name not in files for name in names):
        raise ValueError("Checkpoint does not contain the expected four-rank actor and tokenizer")
    staging = root / "staging"

    def download(name: str) -> tuple[str, dict]:
        remote = files[prefix + name]
        path = Path(hf_hub_download(args.repo, remote.rfilename, revision=args.revision, local_dir=staging))
        digest = sha256(path)
        if path.stat().st_size != remote.size or (remote.lfs and digest != remote.lfs.sha256):
            raise ValueError(f"Downloaded file failed integrity verification: {name}")
        print(f"Verified {name}: {remote.size} bytes", flush=True)
        return remote.rfilename, {"size": remote.size, "sha256": digest}

    with ThreadPoolExecutor(max_workers=4) as pool:
        inventory = dict(pool.map(download, names))
    actor = staging / f"global_step_{args.step}" / "actor"
    for name in TOKENIZER_FILES:
        if sha256(actor / name) != reference["model_sha256"][name]:
            raise ValueError(f"Tokenizer differs from the previous comparison: {name}")
    hub_manifest = {"checkpoint_repo": args.repo, "checkpoint_revision": args.revision, "step": args.step, "files": inventory}
    write_json(root / "hub_manifest.json", hub_manifest)

    temporary = root / "model.tmp"
    command = [sys.executable, str(REPO_ROOT / "scripts/model_merger.py"), "merge", "--backend", "fsdp", "--local_dir", str(actor), "--target_dir", str(temporary)]
    print("Merging verified FSDP actor on CPU", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, check=True)

    from safetensors import safe_open

    tensor_count = 0
    for path in temporary.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            tensor_count += len(tensors.keys())
    if tensor_count != 311:
        raise ValueError(f"Expected 311 saved Qwen3-1.7B tensors; found {tensor_count}")
    for name in TOKENIZER_FILES:
        if sha256(temporary / name) != reference["model_sha256"][name]:
            raise ValueError(f"Tokenizer changed during conversion: {name}")
    subprocess.run([sys.executable, str(REPO_ROOT / "qwen3_experiments/record_prepared_model_source.py"), "--model-dir", str(temporary), "--source-repo", args.repo + "@" + args.revision, "--source-step", str(args.step)], cwd=REPO_ROOT, check=True)
    temporary.rename(destination)
    write_json(
        prepared_path,
        {
            "checkpoint_repo": args.repo,
            "checkpoint_revision": args.revision,
            "step": args.step,
            "model_path": str(destination),
            "model_sha256": {p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file()},
            "source_files": inventory,
            "dataset_sha256": DATASET_SHA256,
            "tokenizer_matches_previous_comparison": True,
            "tensor_count": tensor_count,
            "merger_sha256": sha256(REPO_ROOT / "scripts/model_merger.py"),
        },
    )
    print(f"Prepared and pinned model: {destination}", flush=True)


if __name__ == "__main__":
    main()
