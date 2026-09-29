"""Verify Hub checkpoint contents and preserve receipts before local cleanup."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def inventory(checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Checkpoint must be a real directory")
    files = {}
    for path in sorted(checkpoint.rglob("*")):
        relative = path.relative_to(checkpoint)
        if ".cache" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing checkpoint symlink: {relative}")
        if path.is_file():
            stat = path.stat()
            files[relative.as_posix()] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "inode": stat.st_ino,
            }
    if not files:
        raise ValueError("Checkpoint contains no files")
    return files


def write_receipt(path, receipt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_checkpoint(checkpoint, repo_id, receipt_path, api):
    checkpoint = Path(checkpoint).absolute()
    local = inventory(checkpoint)
    remote_info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    if not remote_info.sha:
        raise ValueError("Hub did not return an immutable commit")
    remote = {item.rfilename: item for item in remote_info.siblings}
    verified = {}
    for index, (relative, stat) in enumerate(local.items(), 1):
        name = f"{checkpoint.name}/{relative}"
        item = remote.get(name)
        if item is None or item.size != stat["size"]:
            raise ValueError(f"Missing file or size mismatch: {name}")
        sha256 = hashlib.sha256()
        git_sha1 = None
        if not item.lfs:
            git_sha1 = hashlib.sha1(f"blob {stat['size']}\0".encode())
        with (checkpoint / relative).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha256.update(chunk)
                if git_sha1 is not None:
                    git_sha1.update(chunk)
        if item.lfs:
            expected = item.lfs["sha256"] if isinstance(item.lfs, dict) else item.lfs.sha256
            if sha256.hexdigest() != expected:
                raise ValueError(f"SHA256 mismatch: {name}")
        elif git_sha1.hexdigest() != item.blob_id:
            raise ValueError(f"Git blob hash mismatch: {name}")
        verified[relative] = {**stat, "sha256": sha256.hexdigest()}
        print(f"Hash verified {index}/{len(local)}: {relative}", flush=True)
    if inventory(checkpoint) != local:
        raise ValueError("Checkpoint changed during verification; retaining local files")
    receipt = {
        "state": "verified",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint": checkpoint.name,
        "repo_id": repo_id,
        "remote_commit": remote_info.sha,
        "verified_at": timestamp(),
        "file_count": len(verified),
        "total_bytes": sum(stat["size"] for stat in local.values()),
        "files": verified,
    }
    write_receipt(receipt_path, receipt)
    return receipt


def check_local(checkpoint, receipt_path):
    checkpoint = Path(checkpoint)
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt["state"] != "verified" or str(checkpoint.resolve()) != receipt["checkpoint_path"]:
        raise ValueError("No matching verified receipt for this checkpoint")
    expected = {
        name: {key: record[key] for key in ("size", "mtime_ns", "inode")}
        for name, record in receipt["files"].items()
    }
    if inventory(checkpoint) != expected:
        raise ValueError("Checkpoint changed since verification; retaining local files")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "check-local", "mark-deleted"])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    if args.action == "verify":
        from huggingface_hub import HfApi

        if not args.repo_id:
            parser.error("verify requires --repo-id")
        verify_checkpoint(args.checkpoint, args.repo_id, args.receipt, HfApi())
    elif args.action == "check-local":
        check_local(args.checkpoint, args.receipt)
    else:
        receipt = json.loads(args.receipt.read_text())
        if args.checkpoint.exists() or args.checkpoint.is_symlink():
            raise ValueError("Checkpoint still exists; cannot mark it deleted")
        if str(args.checkpoint.absolute()) != receipt["checkpoint_path"] or receipt["state"] != "verified":
            raise ValueError("Checkpoint does not match its verified receipt")
        receipt.update(state="archived_and_deleted", deleted_at=timestamp())
        write_receipt(args.receipt, receipt)


if __name__ == "__main__":
    main()
