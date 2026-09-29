"""Publish verified compression step-100 actors as directly loadable HF models."""

import argparse
import fcntl
import gc
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def checkpoint_identity(spec):
    root = Path(spec["training_root"])
    if digest(root / "plan.json") != spec["training_plan_sha256"]:
        raise ValueError("Training plan changed")
    state = read(root / "status.json")
    if state.get("state") != "complete" or state.get("exit_code", 0) != 0:
        raise ValueError("Training must finish successfully before export")
    if digest(spec["archive_receipt"]) != spec["archive_receipt_sha256"]:
        raise ValueError("Checkpoint archive receipt changed")
    archive = read(spec["archive_receipt"])
    if spec["format"] == "fsdp8":
        if (state.get("last_completed_step") != 100 or archive.get("state") != "archived_and_deleted"
                or archive.get("checkpoint") != "global_step_100"):
            raise ValueError("Expected a verified final step-100 checkpoint")
        source = {"repo": archive["repo_id"], "revision": archive["remote_commit"]}
    else:
        if state.get("completed_rollout_steps") != 100 or state.get("optimizer_updates") != 200:
            raise ValueError("ER final export must follow all 100 rollout steps")
        source = {"repo": archive["repo_id"], "revision": archive["revision"]}
    if source != spec["source"] or not spec["target"].endswith("-final"):
        raise ValueError("Wrong source checkpoint or final repository")
    return archive


def verify_merge_receipt(spec, archive, receipt):
    if {key: receipt[key] for key in ("repo", "revision")} != spec["source"]:
        raise ValueError("Merged model belongs to another checkpoint")
    expected = {f"global_step_100/actor/model_world_size_8_rank_{rank}.pt" for rank in range(8)}
    if not expected <= receipt["source_files"].keys():
        raise ValueError("Missing FSDP rank provenance")
    for name, metadata in receipt["source_files"].items():
        original = archive["files"][name.removeprefix("global_step_100/")]
        if metadata != {"size": original["size"], "sha256": original["sha256"]}:
            raise ValueError("Merged model does not match the archived checkpoint")
    names = set(receipt["merged_files"])
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= names:
        raise ValueError("Incomplete model/tokenizer export")
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("Missing exported weights")
    if any(Path(name).name != name for name in names):
        raise ValueError("Export files must be at the model root")


def verify_local_files(folder, hashes):
    for name, checksum in hashes.items():
        if digest(folder / name) != checksum:
            raise ValueError(f"Model file checksum changed: {name}")


def prepare_actor(spec, archive, work):
    from qwen3_experiments import eval_polaris_step80 as converter
    from qwen3_experiments.prepare_math_eval_matrix import download_files

    work.mkdir(parents=True, exist_ok=True)
    if spec["format"] == "huggingface":
        files = download_files({**spec["source"], "key": spec["key"]}, list(archive["files"]), work / "model")
        for name, metadata in files.items():
            if metadata != {"size": archive["files"][name], "sha256": archive["sha256"][name]}:
                raise ValueError("Existing ER final export differs from its verified archive")
        return archive["sha256"]
    receipt_path = work / "model_receipt.json"
    if not receipt_path.exists():
        cached = Path(spec.get("prepared_root", work / "no_cache"))
        if (cached / "model_receipt.json").is_file():
            receipt = read(cached / "model_receipt.json")
            verify_merge_receipt(spec, archive, receipt)
            if all((cached / "model" / name).is_file() for name in receipt["merged_files"]):
                destination = work / "model"
                destination.mkdir(exist_ok=True)
                for name in receipt["merged_files"]:
                    shutil.copy2(cached / "model" / name, destination / name)
                verify_local_files(destination, receipt["merged_files"])
                write(receipt_path, receipt)
        if not receipt_path.exists():
            # Any interrupted merge is confined to this exporter-owned directory.
            for name in ("model", "model_in_progress"):
                path = work / name
                if path.exists():
                    shutil.rmtree(path)
            converter.MODEL = spec["source"]["repo"]
            converter.REVISION = spec["source"]["revision"]
            converter.PREFIX = "global_step_100/actor/"
            converter.REPO = Path(spec["runtime"])
            converter.prepare_model(work)
    receipt = read(receipt_path)
    verify_merge_receipt(spec, archive, receipt)
    verify_local_files(work / "model", receipt["merged_files"])
    return receipt["merged_files"]


def validate_inventory(actual, reference, tied_embeddings):
    expected = dict(reference)
    # Transformers omits separate lm_head storage when it ties the embeddings.
    if tied_embeddings and "lm_head.weight" not in actual:
        expected.pop("lm_head.weight", None)
    if actual != expected:
        raise ValueError("Exported tensor shapes or dtypes differ from the base architecture")


def validate_loading(folder, base):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from qwen3_experiments.prepare_math_eval_matrix import tensor_inventory

    torch.set_num_threads(4)
    inventory = tensor_inventory(folder)
    config = AutoConfig.from_pretrained(folder, local_files_only=True)
    validate_inventory(inventory, tensor_inventory(base), config.tie_word_embeddings)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    original = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tokenizer.get_vocab() != original.get_vocab() or tokenizer.chat_template != original.chat_template:
        raise ValueError("Tokenizer vocabulary or thinking template changed")
    model, details = AutoModelForCausalLM.from_pretrained(
        folder, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager", output_loading_info=True,
    )
    if any(details.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Model cannot load cleanly: {details}")
    model.eval()
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 1 + 1?"}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True, return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(input_ids=tokens, use_cache=False).logits
    if not torch.isfinite(logits).all().item():
        raise ValueError("Loaded model produced nonfinite logits")
    result = {"transformers": transformers.__version__, "torch": torch.__version__, "device": "cpu",
              "tensor_count": len(inventory), "dtype": "bfloat16", "loading_info": details,
              "thinking_template_matches_base": True, "forward_logits_finite": True}
    del model, logits
    gc.collect()
    return result


def verify_public(api, repo, revision, folder, hashes):
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if info.private or info.sha != revision:
        raise ValueError("Final model is not public at the expected commit")
    remote = {item.rfilename: item for item in info.siblings}
    for name, checksum in hashes.items():
        entry = remote[name]
        path = folder / name
        if entry.size != path.stat().st_size:
            raise ValueError(f"Remote model size mismatch: {name}")
        if entry.lfs:
            if entry.lfs.sha256 != checksum:
                raise ValueError(f"Remote model checksum mismatch: {name}")
        else:
            body = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
            if blob != entry.blob_id:
                raise ValueError(f"Remote asset checksum mismatch: {name}")
    return info


def publish(spec, archive, work, hashes, validation):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError

    api, public = HfApi(), HfApi(token=False)
    folder = work / "model"
    if spec["format"] == "huggingface":
        if spec["source"]["repo"] != spec["target"]:
            raise ValueError("Existing ER final repository identity changed")
        revision = spec["source"]["revision"]
    else:
        manifest = {"source": spec["source"], "training_step": 100,
                    "training_dataset": "zjhhhh/compression_dataset", "method": spec["label"],
                    "base_model": "Qwen/Qwen3-1.7B", "files_sha256": hashes, "validation": validation}
        write(folder / "export_manifest.json", manifest)
        (folder / "README.md").write_text(
            "---\nbase_model: Qwen/Qwen3-1.7B\nlibrary_name: transformers\n"
            "pipeline_tag: text-generation\ntags:\n- qwen3\n- compression\n- reinforcement-learning\n---\n\n"
            f"# Qwen3-1.7B compression {spec['label']} final\n\n"
            "Standard BF16 Hugging Face model, exported from training step 100. "
            "Weights, configuration, tokenizer, and the thinking chat template are at the repository root.\n\n"
            f"Source checkpoint: [{spec['source']['repo']}](https://huggingface.co/{spec['source']['repo']}) "
            f"at commit `{spec['source']['revision']}`.\n\n"
            "Load this repository directly with `AutoModelForCausalLM.from_pretrained` or vLLM. "
            "Use `enable_thinking=True` in the tokenizer chat template for the existing thinking evaluations.\n\n"
            "The export manifest records source provenance, file checksums, tensor validation, "
            "and a successful Transformers CPU load and forward pass.\n"
        )
        hashes = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}
        try:
            existing = api.model_info(spec["target"], files_metadata=True)
        except RepositoryNotFoundError:
            existing = None
        if existing:
            names = {item.rfilename for item in existing.siblings}
            if names - {".gitattributes"}:
                if "export_manifest.json" not in names:
                    raise ValueError("Refusing to overwrite an unrelated final repository")
                path = hf_hub_download(spec["target"], "export_manifest.json", revision=existing.sha,
                                       local_dir=work / "remote_manifest")
                previous = read(path)
                if previous["source"] != spec["source"] or previous["files_sha256"] != manifest["files_sha256"]:
                    raise ValueError("Final repository already contains another model")
        api.create_repo(spec["target"], private=False, exist_ok=True)
        if existing and existing.private:
            api.update_repo_settings(spec["target"], private=False)
        commit = api.upload_folder(repo_id=spec["target"], folder_path=folder,
                                   allow_patterns=list(hashes), commit_message="Export verified step 100 for direct evaluation")
        revision = commit.oid
    verify_local_files(folder, hashes)
    verify_public(public, spec["target"], revision, folder, hashes)
    return {"state": "published_verified", "repo": spec["target"], "revision": revision,
            "public": True, "source": spec["source"], "training_step": 100,
            "files_sha256": hashes, "validation": validation, "verified_at": now()}


def cleanup(work, scratch, receipt):
    if (receipt.get("state") != "published_verified" or not receipt.get("public")
            or work.parent != scratch / "work" or work.is_symlink() or scratch.resolve() != scratch):
        raise ValueError("Cleanup requires a verified publication and an owned workspace")
    if work.exists():
        shutil.rmtree(work)


def run(plan_path):
    # The existing evaluation helpers also support standalone script imports.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qwen3_experiments.grpo_compute_control import require_compute

    plan = read(plan_path)
    require_compute(plan["job_id"])
    if socket.gethostname() != plan["node"] or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run the exporter on its compute node with GPUs disabled")
    for path, checksum in plan["code_sha256"].items():
        if digest(path) != checksum:
            raise ValueError(f"Export implementation changed: {path}")
    root, scratch = Path(plan_path).parent, Path(plan["scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    with (root / "export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = []
        for spec in plan["models"]:
            target_receipt = root / "receipts" / f"{spec['key']}.json"
            work = scratch / "work" / spec["key"]
            if target_receipt.exists():
                prior = read(target_receipt)
                if prior["repo"] != spec["target"] or prior["source"] != spec["source"]:
                    raise ValueError("Publication receipt belongs to another model")
                cleanup(work, scratch, prior)
                completed.append(spec["key"])
                continue
            archive = checkpoint_identity(spec)
            for attempt in range(1, 4):
                state = {"state": "preparing", "current": spec["key"], "completed": list(completed),
                         "pid": os.getpid(), "node": socket.gethostname(), "updated_at": now(), "attempt": attempt}
                try:
                    if shutil.disk_usage(scratch).free < 64 * 1024**3:
                        raise RuntimeError("Compute disk needs at least 64 GiB free for conversion")
                    write(root / "status.json", state)
                    print(json.dumps(state), flush=True)
                    hashes = prepare_actor(spec, archive, work)
                    write(root / "status.json", {**state, "state": "validating", "updated_at": now()})
                    validation = validate_loading(work / "model", Path(plan["base_model"]))
                    write(root / "status.json", {**state, "state": "publishing", "updated_at": now()})
                    receipt = publish(spec, archive, work, hashes, validation)
                    write(target_receipt, receipt)
                    cleanup(work, scratch, receipt)
                    receipt.update(temporary_files_deleted=True, cleaned_at=now())
                    write(target_receipt, receipt)
                    completed.append(spec["key"])
                    print(json.dumps({"state": "complete", "model": spec["key"], "repo": spec["target"],
                                      "revision": receipt["revision"], "temporary_files_deleted": True}), flush=True)
                    break
                except Exception as exc:
                    retry = attempt < 3 and not isinstance(exc, ValueError)
                    write(root / "status.json", {**state, "state": "retrying" if retry else "failed",
                                                 "error": str(exc), "updated_at": now()})
                    if not retry:
                        raise
                    time.sleep(15)
        write(root / "status.json", {"state": "complete", "completed": completed, "updated_at": now(),
                                      "compute_free_bytes": shutil.disk_usage(scratch).free})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan.resolve())


if __name__ == "__main__":
    main()
