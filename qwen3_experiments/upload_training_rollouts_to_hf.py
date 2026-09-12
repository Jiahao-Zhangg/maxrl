"""Upload completed legacy trainer JSONL files while their training run continues."""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

from verl.utils.rollout_dataset import dump_rollout_step, upload_rollout_dataset_to_hf


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_matches(pid, start_time):
    """Do not mistake a recycled PID for the original training supervisor."""
    try:
        process = Path("/proc") / str(pid)
        fields = (process / "stat").read_text().rsplit(") ", 1)[1].split()
        return process.stat().st_uid == os.getuid() and fields[0] != "Z" and fields[19] == str(start_time)
    except (OSError, IndexError):
        return False


def complete_records(path, step, expected_rows):
    """A legacy JSONL write is not atomic: require a full, unchanged batch."""
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    signature = [after.st_ino, after.st_size, after.st_mtime_ns]
    if signature != [before.st_ino, before.st_size, before.st_mtime_ns] or not payload.endswith(b"\n"):
        return None
    try:
        records = [json.loads(line) for line in payload.splitlines()]
    except (ValueError, UnicodeError):
        return None
    if len(records) != expected_rows:
        return None
    for record in records:
        if (
            not isinstance(record, dict)
            or record.get("step") != step
            or not isinstance(record.get("input"), str)
            or not isinstance(record.get("output"), str)
            or not isinstance(record.get("score"), (int, float))
        ):
            raise ValueError(f"Invalid rollout record in {path}")
    return records, signature


class RolloutUploader:
    def __init__(self, source_dir, dataset_dir, state_file, repo_id, expected_rows, final_step, metadata, api, private=False):
        self.source_dir = Path(source_dir).resolve()
        self.dataset_dir = Path(dataset_dir).resolve()
        self.state_file = Path(state_file).resolve()
        self.repo_id = repo_id
        self.expected_rows = expected_rows
        self.final_step = final_step
        self.metadata = metadata
        self.api = api
        self.private = private
        if expected_rows < 1 or final_step < 1:
            raise ValueError("expected_rows and final_step must be positive")
        if self.state_file.is_relative_to(self.dataset_dir) or self.source_dir == self.dataset_dir:
            raise ValueError("Keep upload state and source JSONL files outside the staged dataset directory")
        identity = {
            "source_dir": str(self.source_dir),
            "dataset_dir": str(self.dataset_dir),
            "repo_id": repo_id,
            "expected_rows": expected_rows,
            "final_step": final_step,
        }
        self.state = {"identity": identity, "prepared": {}, "uploaded_steps": [], "initialized": False}
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text())
            if self.state.get("identity") != identity:
                raise ValueError("Existing uploader state belongs to a different rollout collection")

    def save_state(self, **updates):
        self.state.update(updates, updated_at=time.time())
        write_json_atomic(self.state_file, self.state)

    def initialize(self):
        if self.state["initialized"]:
            return
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", private=self.private, exist_ok=True)
        card = self.dataset_dir / "README.md"
        if not card.exists():
            card.write_text(
                f"# {self.metadata.get('experiment_name', 'Training')} rollouts\n\n"
                "Training is in progress. Each completed training batch will be uploaded "
                "as a compressed JSONL shard containing all prompts, responses and scores.\n\n"
                f"Training run: {self.metadata.get('wandb_url', '')}\n",
                encoding="utf-8",
            )
        self.api.upload_file(
            path_or_fileobj=str(card), path_in_repo="README.md", repo_id=self.repo_id,
            repo_type="dataset", commit_message="Initialize training rollout dataset",
        )
        self.save_state(initialized=True, status="waiting_for_rollouts")

    def tick(self, trainer_alive):
        self.initialize()
        for path in sorted(self.source_dir.glob("*.jsonl")):
            if not path.stem.isdigit() or not 1 <= int(path.stem) <= self.final_step:
                continue
            step = int(path.stem)
            stat = path.stat()
            signature = [stat.st_ino, stat.st_size, stat.st_mtime_ns]
            shard = self.dataset_dir / "data" / f"step_{step:06d}.jsonl.gz"
            if self.state["prepared"].get(str(step)) == signature and shard.is_file():
                continue
            completed = complete_records(path, step, self.expected_rows)
            if completed is None:
                continue
            records, signature = completed
            extra_keys = set().union(*(record.keys() for record in records)) - {
                "input", "output", "score", "step", "rollout_index",
            }
            dump_rollout_step(
                self.dataset_dir, step=step,
                inputs=[record["input"] for record in records],
                outputs=[record["output"] for record in records],
                scores=[record["score"] for record in records],
                extra_fields={key: [record.get(key) for record in records] for key in sorted(extra_keys)},
            )
            self.state["prepared"][str(step)] = signature
            self.state["uploaded_steps"] = [value for value in self.state["uploaded_steps"] if value != step]
            self.save_state(status="prepared")
            print(f"Prepared step {step}: {len(records)} rollouts", flush=True)

        prepared = {int(step) for step in self.state["prepared"]}
        if prepared != set(self.state["uploaded_steps"]):
            url = upload_rollout_dataset_to_hf(
                self.dataset_dir, repo_id=self.repo_id, private=self.private, api=self.api,
                metadata=self.metadata, verify_attempts=2, verify_sleep_seconds=5,
            )
            self.save_state(uploaded_steps=sorted(prepared), dataset_url=url, last_error=None)
            print(f"Verified {len(prepared)} steps ({len(prepared) * self.expected_rows} rollouts): {url}", flush=True)

        expected = set(range(1, self.final_step + 1))
        if set(self.state["uploaded_steps"]) == expected:
            status = "complete"
        elif not trainer_alive:
            status = "incomplete"
        else:
            status = "watching" if prepared else "waiting_for_rollouts"
        self.save_state(status=status)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--trainer-pid", required=True, type=int)
    parser.add_argument("--trainer-start-time", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--final-step", required=True, type=int)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-retries-after-exit", type=int, default=20)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.max_retries_after_exit < 1:
        parser.error("poll-seconds and max-retries-after-exit must be positive")

    from huggingface_hub import HfApi
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(args.repo_id)
    state_file = Path(args.state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        uploader = RolloutUploader(
            args.source_dir, args.dataset_dir, state_file, args.repo_id, args.expected_rows,
            args.final_step, json.loads(Path(args.metadata_json).read_text()), HfApi(), args.private,
        )
        failures_after_exit = 0
        while True:
            alive = process_matches(args.trainer_pid, args.trainer_start_time)
            try:
                status = uploader.tick(alive)
                if status == "complete":
                    return 0
                if status == "incomplete":
                    print("Training exited before all planned rollout steps were saved; available complete steps are uploaded.", flush=True)
                    return 1
            except Exception as error:
                uploader.save_state(status="retrying", last_error=f"{type(error).__name__}: {error}")
                print(f"Rollout upload will retry: {type(error).__name__}: {error}", flush=True)
                if not alive:
                    failures_after_exit += 1
                    if failures_after_exit >= args.max_retries_after_exit:
                        uploader.save_state(status="failed")
                        raise
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
