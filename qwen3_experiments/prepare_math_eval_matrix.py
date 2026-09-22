#!/usr/bin/env python3
"""Pin datasets and convert actor-only Hub checkpoints on a node-local disk."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from math_eval_matrix_common import fingerprint, read_config, read_json, sha256, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json")


def download_files(spec: dict, names: list[str], destination: Path, *, repo_type="model") -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.repo_info(spec["repo"], repo_type=repo_type, revision=spec["revision"], files_metadata=True)
    if info.sha != spec["revision"]:
        raise ValueError("Hub revision mismatch")
    remote = {entry.rfilename: entry for entry in info.siblings}
    if any(name not in remote for name in names):
        raise ValueError(f"Missing requested source files in {spec['repo']}")

    def download(name):
        entry = remote[name]
        path = Path(hf_hub_download(spec["repo"], name, repo_type=repo_type, revision=spec["revision"], local_dir=destination))
        digest = sha256(path)
        expected = getattr(entry.lfs, "sha256", None) if entry.lfs else None
        if path.stat().st_size != entry.size or (expected and digest != expected):
            raise ValueError(f"Download integrity failure: {name}")
        print(f"Verified {spec['key']}: {name} ({entry.size} bytes)", flush=True)
        return name, {"size": entry.size, "sha256": digest}

    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(download, names))


def normalize_dataset(rows: list[dict], spec: dict, suffix: str) -> list[dict]:
    if len(rows) != spec["expected_rows"]:
        raise ValueError(f"{spec['key']}: expected {spec['expected_rows']} rows, got {len(rows)}")
    normalized, seen = [], set()
    for index, row in enumerate(rows):
        question = row[spec["question_field"]]
        answer = row[spec["answer_field"]]
        if hasattr(answer, "tolist"):
            answer = answer.tolist()
        if isinstance(answer, list):
            if len(answer) != 1:
                raise ValueError("Expected one gold-answer expression; refusing to drop alternatives")
            answer = answer[0]
        if not isinstance(question, str) or not question.strip() or answer is None:
            raise ValueError(f"Missing question/answer in row {index}")
        if isinstance(answer, float) and answer.is_integer():
            answer = int(answer)
        answer = str(answer)
        if not answer.strip():
            raise ValueError(f"Empty answer in row {index}")
        if spec["key"] == "olympiadbench":
            if row.get("modality") != "Text-only" or any(row.get(f"image_{i}") is not None for i in range(1, 6)):
                raise ValueError("The text-only matrix cannot silently discard an image")
        question_hash = fingerprint(question)
        if question_hash in seen:
            raise ValueError(f"Duplicate question in {spec['key']}: row {index}")
        seen.add(question_hash)
        source_id = row.get("unique_id", row.get("url", row.get("id", index)))
        normalized.append({"id": index, "unique_id": f"{spec['key']}:{source_id}", "question": question,
                           "ground_truth": answer, "prompt": [{"role": "user", "content": question + suffix}],
                           "source_id": str(source_id)})
    return normalized


def prepare_datasets(config: dict, root: Path, scratch: Path) -> dict:
    import pandas as pd

    datasets = {}
    for spec in config["datasets"]:
        destination = root / "datasets" / f"{spec['key']}.json"
        receipt = root / "datasets" / f"{spec['key']}.manifest.json"
        if receipt.exists():
            previous = read_json(receipt)
            if previous["spec"] != spec or previous["prompt_suffix"] != config["prompt_suffix"] or sha256(destination) != previous["sha256"]:
                raise ValueError(f"Existing dataset snapshot differs: {destination}")
            datasets[spec["key"]] = previous
            continue
        source_dir = scratch / "source_datasets" / spec["key"]
        inventory = download_files(spec, [spec["file"]], source_dir, repo_type="dataset")
        source = source_dir / spec["file"]
        rows = pd.read_parquet(source).to_dict("records") if source.suffix == ".parquet" else [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        normalized = normalize_dataset(rows, spec, config["prompt_suffix"])
        write_json(destination, normalized)
        result = {"spec": spec, "path": str(destination), "num_prompts": len(normalized), "sha256": sha256(destination),
                  "prompt_suffix": config["prompt_suffix"], "source_files": inventory}
        write_json(receipt, result)
        datasets[spec["key"]] = result
    return datasets


def tensor_inventory(path: Path) -> dict:
    from safetensors import safe_open

    result = {}
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key in result:
                    raise ValueError(f"Duplicate tensor: {key}")
                tensor = stream.get_slice(key)
                result[key] = {"shape": tensor.get_shape(), "dtype": tensor.get_dtype()}
    if not result:
        raise ValueError(f"No model tensors at {path}")
    return result


def validate_tensor_inventory(actual: dict, base: Path) -> None:
    expected = tensor_inventory(base)
    # HF omits tied lm_head storage; FSDP and DeepSpeed may explicitly save it.
    if read_json(base / "config.json").get("tie_word_embeddings") and "lm_head.weight" in actual and "lm_head.weight" not in expected:
        expected["lm_head.weight"] = expected["model.embed_tokens.weight"]
    if actual != expected:
        raise ValueError(f"Converted tensors differ from base: missing={set(expected)-set(actual)}, extra={set(actual)-set(expected)}")


def prepare_model(spec: dict, scratch: Path, base: Path | None) -> dict:
    from huggingface_hub import HfApi

    model_root = scratch / "models" / spec["key"]
    destination = model_root / "model"
    receipt = model_root / "prepared.json"
    if receipt.exists():
        previous = read_json(receipt)
        if previous["spec"] != spec:
            raise ValueError("Existing model belongs to another source revision")
        for name, digest in previous["files_sha256"].items():
            if sha256(destination / name) != digest:
                raise ValueError(f"Prepared model checksum changed: {name}")
        return previous
    if destination.exists():
        raise FileExistsError(f"Unverified model directory needs inspection: {destination}")
    info = HfApi().model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
    available = {item.rfilename for item in info.siblings}
    source = model_root / "source"
    if spec["format"] == "huggingface":
        names = [name for name in available if name in ASSETS or name.endswith(".safetensors") or name == "model.safetensors.index.json"]
        inventory = download_files(spec, names, destination)
    elif spec["format"] == "fsdp":
        prefix = f"global_step_{spec['step']}/actor/"
        names = [prefix + f"model_world_size_4_rank_{rank}.pt" for rank in range(4)]
        names += [prefix + name for name in ASSETS if prefix + name in available]
        inventory = download_files(spec, names, source)
        actor = source / prefix
        if base is None:
            raise ValueError("Prepare the base model first")
        # L1's actor-only archive intentionally lacks tokenizer/config files.
        for name in ASSETS:
            if not (actor / name).exists() and (base / name).is_file():
                shutil.copy2(base / name, actor / name)
        command = [sys.executable, str(REPO_ROOT / "scripts/model_merger.py"), "merge", "--backend", "fsdp", "--local_dir", str(actor), "--target_dir", str(destination)]
        subprocess.run(command, check=True, cwd=REPO_ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    elif spec["format"] == "deepspeed":
        import torch
        from safetensors.torch import save_file

        name = f"global_step{spec['step']}/mp_rank_00_model_states.pt"
        inventory = download_files(spec, [name, "train_config.json"], source)
        train = read_json(source / "train_config.json")
        if train.get("pretrain") != "Qwen/Qwen3-1.7B-Base" or base is None:
            raise ValueError("Unexpected ER base model")
        checkpoint = torch.load(source / name, map_location="cpu", weights_only=False)
        state = checkpoint["module"]
        expected = tensor_inventory(base)
        if read_json(base / "config.json").get("tie_word_embeddings") and "lm_head.weight" in state:
            expected["lm_head.weight"] = expected["model.embed_tokens.weight"]
        if set(state) != set(expected):
            raise ValueError(f"ER tensor names differ: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}")
        destination.mkdir(parents=True)
        # Clone tied/shared storage so safetensors contains every expected parameter.
        state = {key: value.detach().to(torch.bfloat16).contiguous().clone() for key, value in state.items()}
        save_file(state, destination / "model.safetensors", metadata={"format": "pt"})
        for name in ASSETS:
            if (base / name).is_file():
                shutil.copy2(base / name, destination / name)
        del state, checkpoint
        gc.collect()
    else:
        raise ValueError(f"Unknown checkpoint format: {spec['format']}")
    tensors = tensor_inventory(destination)
    if base is not None:
        validate_tensor_inventory(tensors, base)
    if read_json(destination / "config.json").get("tie_word_embeddings") and "lm_head.weight" in tensors:
        from safetensors.torch import load_file
        import torch

        state = {}
        for file in destination.glob("*.safetensors"):
            state.update(load_file(file))
        if not torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
            raise ValueError("Checkpoint claims tied embeddings but saved lm_head differs")
        del state
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    if base is not None:
        reference = AutoTokenizer.from_pretrained(base, local_files_only=True)
        if tokenizer.get_vocab() != reference.get_vocab() or tokenizer.chat_template != reference.chat_template:
            raise ValueError("Model token IDs or chat template differ from the common base")
    result = {"spec": spec, "path": str(destination), "source_files": inventory, "tensor_count": len(tensors),
              "tensor_inventory_sha256": fingerprint(tensors),
              "files_sha256": {file.name: sha256(file) for file in sorted(destination.iterdir()) if file.is_file()},
              "vocab_sha256": fingerprint(tokenizer.get_vocab()), "chat_template_sha256": fingerprint(tokenizer.chat_template)}
    write_json(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--models-only", action="store_true")
    modes.add_argument("--datasets-only", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    root, scratch = args.output_root.resolve(), args.scratch.resolve()
    root.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch).free < 70 * 2**30:
        raise RuntimeError("Need 70 GiB node-local space for pinned source weights and conversions")
    previous = read_json(root / "prepared_inputs.json") if (root / "prepared_inputs.json").exists() else {}
    datasets = previous.get("datasets", {}) if args.models_only else prepare_datasets(config, root, scratch)
    models = previous.get("models", {}) if args.datasets_only else {}
    if args.datasets_only:
        write_json(root / "prepared_inputs.json", {"models": models, "datasets": datasets, "config": config})
        return
    ordered = sorted(config["models"], key=lambda item: item["format"] != "huggingface")
    base = None
    for model in ordered:
        print(f"Preparing model {model['key']}", flush=True)
        models[model["key"]] = prepare_model(model, scratch, base)
        if model["format"] == "huggingface":
            base = Path(models[model["key"]]["path"])
        write_json(root / "prepared_inputs.json", {"models": models, "datasets": datasets, "config": config})
    print("All pinned inputs prepared and verified", flush=True)


if __name__ == "__main__":
    main()
