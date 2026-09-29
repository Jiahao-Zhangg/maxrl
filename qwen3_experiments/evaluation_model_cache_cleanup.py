"""Remove a completed evaluation's model copies before its training handoff."""

import os
import re
import shutil
from pathlib import Path

from qwen3_experiments.grpo_compute_control import digest, now, read, write


def cleanup_model_caches(plan, previous):
    """Caller must verify all results and hold the allocation/evaluation locks."""
    root = Path(plan["predecessor_evaluation_root"])
    scratch = Path(previous["scratch"])
    if not scratch.is_absolute() or scratch == Path("/") or scratch.resolve() != scratch:
        raise ValueError("Evaluation scratch must be a direct absolute directory")
    cache = scratch / "models"
    if cache.is_symlink():
        raise ValueError("Evaluation model cache cannot be a symlink")
    protected = [Path(plan[key]).resolve() for key in ("model_path", "output_root")]
    protected += [root.resolve(), Path(previous["base_model"]).resolve()]
    protected += [Path(path).resolve() for path in plan["input_hashes"]]
    models = []
    keys = [spec["key"] for spec in previous["models"]]
    if len(keys) != 3 or len(set(keys)) != 3:
        raise ValueError("Expected three distinct completed Polaris model caches")
    for spec in previous["models"]:
        key = spec["key"]
        if not re.fullmatch(r"[a-z0-9_]+", key):
            raise ValueError("Invalid model cache key")
        folder = cache / key
        if folder.is_symlink() or folder.resolve() != folder:
            raise ValueError("Model cache path must not traverse symlinks")
        if any(p == folder or p.is_relative_to(folder) or folder.is_relative_to(p) for p in protected):
            raise ValueError("Refusing to remove training inputs or evaluation results")
        receipt = read(root / "prepared_models" / f"{key}.json")
        if receipt["repo"] != spec["repo"] or receipt["revision"] != spec["revision"]:
            raise ValueError("Model cache checkpoint identity changed")
        size = count = 0
        if folder.exists():
            if not folder.is_dir():
                raise ValueError("Model cache is not a directory")
            for path in folder.rglob("*"):
                if path.is_symlink():
                    raise ValueError("Refusing to follow a symlink inside the model cache")
                if path.stat().st_uid != os.getuid():
                    raise ValueError("Model cache belongs to another user")
                if path.is_file():
                    size += path.stat().st_size
                    count += 1
        models.append({"key": key, "repo": spec["repo"], "revision": spec["revision"],
                       "path": str(folder), "bytes": size, "files": count,
                       "state": "pending" if folder.exists() else "already_absent"})

    path = Path(plan["output_root"]) / "evaluation_model_cache_cleanup.json"
    identity = {"predecessor_plan_sha256": digest(root / "plan.json"),
                "training_plan_sha256": digest(Path(plan["output_root"]) / "plan.json")}
    if path.exists():
        old = read(path)
        if any(old.get(key) != value for key, value in identity.items()):
            raise ValueError("Model cache cleanup receipt belongs to another plan")
        if old.get("state") == "complete" and all(not Path(m["path"]).exists() for m in models):
            return old
    receipt = {**identity, "state": "cleaning", "models": models, "started_at": now(),
               "retained_base_model": plan["model_path"],
               "free_bytes_before": shutil.disk_usage(scratch).free}
    # Persist the exact deletion list before removing any model directory.
    write(path, receipt)
    for model in models:
        folder = Path(model["path"])
        if folder.exists():
            shutil.rmtree(folder)
            model["state"] = "deleted"
            write(path, receipt)
        if folder.exists() or folder.is_symlink():
            raise RuntimeError("Evaluation model cache cleanup is incomplete")
    receipt.update(state="complete", finished_at=now(),
                   deleted_bytes=sum(m["bytes"] for m in models if m["state"] == "deleted"),
                   free_bytes_after=shutil.disk_usage(scratch).free)
    write(path, receipt)
    return receipt
