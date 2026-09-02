#!/usr/bin/env python3
"""Finalize a successful W&B run and recover any unsynced tail records."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
TASK_RUNNER_PREFIX = re.compile(r"\(TaskRunner pid=\d+\)\s")


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-pid", required=True, type=int)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--local-run-file", required=True, type=Path)
    parser.add_argument("--final-step", required=True, type=int)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--grace-seconds", type=int, default=90)
    parser.add_argument("--repair-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.training_pid <= 0 or args.final_step <= 0:
        parser.error("--training-pid and --final-step must be positive")
    if args.poll_seconds <= 0 or args.grace_seconds < 0:
        parser.error("poll seconds must be positive and grace seconds cannot be negative")
    return args


def process_start_time(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if len(fields) < 22 or fields[2] == "Z":
        return None
    return fields[21]


def training_reached_final_step(training_log: Path, final_step: int) -> bool:
    if not training_log.is_file():
        return False
    final_step_pattern = re.compile(
        rf"step:{final_step}(?:\s+-|\s|$)|training/global_step:{final_step}\.000|{final_step}/{final_step}"
    )
    with training_log.open(errors="replace") as stream:
        return any(final_step_pattern.search(line) for line in stream)


def read_local_history(local_run_file: Path) -> dict[int, dict[str, object]]:
    datastore = DataStore()
    datastore.open_for_scan(str(local_run_file))
    history: dict[int, dict[str, object]] = {}
    while True:
        try:
            raw = datastore.scan_data()
        except AssertionError as error:
            log(f"Local W&B journal has an incomplete tail, which is expected after a crash: {error}")
            break
        if raw is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(raw)
        if record.WhichOneof("record_type") != "history":
            continue
        values: dict[str, object] = {}
        for item in record.history.item:
            key = item.key or "/".join(item.nested_key)
            values[key] = json.loads(item.value_json)
        internal_step = values.get("_step")
        if isinstance(internal_step, int):
            history[internal_step] = {
                key: value for key, value in values.items() if not key.startswith("_")
            }
    return history


def read_final_console_metrics(training_log: Path, final_step: int) -> dict[str, object]:
    metrics: dict[str, object] | None = None
    marker = f"step:{final_step} - "
    with training_log.open(errors="replace") as stream:
        for raw in stream:
            line = ANSI_ESCAPE.sub("", raw)
            if marker not in line:
                continue
            candidate: dict[str, object] = {}
            for field in line.split(marker, 1)[1].strip().split(" - "):
                key, value = field.rsplit(":", 1)
                candidate[key] = float(value)
            metrics = candidate
    if metrics is None:
        raise RuntimeError(f"could not find the step-{final_step} console metric record")

    # The compact console record rounds values. Overlay the full-precision final
    # validation dictionary when it is available.
    fragments: list[str] = []
    capturing = False
    with training_log.open(errors="replace") as stream:
        for raw in stream:
            line = ANSI_ESCAPE.sub("", raw)
            match = TASK_RUNNER_PREFIX.search(line)
            if match is None:
                continue
            content = line[match.end() :].strip("\n\r")
            if content.startswith('(\"Final validation metrics:'):
                capturing = True
            if not capturing:
                continue
            stripped = content.lstrip()
            if not (content.startswith("(") or stripped.startswith(("'", '\"'))):
                continue
            fragments.append(content)
            if content.endswith("}')"):
                break
    if fragments:
        try:
            final_text = ast.literal_eval("\n".join(fragments))
            validation = ast.literal_eval(final_text.removeprefix("Final validation metrics: "))
            metrics.update(validation)
        except (SyntaxError, ValueError) as error:
            log(f"Could not recover full-precision validation metrics; using console values: {error}")
    return metrics


def remote_run(entity: str, project: str, run_id: str):
    return wandb.Api(timeout=30).run(f"{entity}/{project}/{run_id}")


def run_is_complete(run, final_step: int) -> bool:
    summary_step = run.summary.get("training/global_step")
    return (
        run.state == "finished"
        and run.lastHistoryStep is not None
        and run.lastHistoryStep >= final_step
        and summary_step == final_step
    )


def repair_run(args: argparse.Namespace) -> None:
    run = remote_run(args.entity, args.project, args.run_id)
    if run_is_complete(run, args.final_step):
        log(f"W&B run {args.run_id} already finished cleanly at step {args.final_step}")
        return

    remote_last_step = int(run.lastHistoryStep if run.lastHistoryStep is not None else -1)
    log(
        f"Repairing W&B run {args.run_id}: state={run.state}, "
        f"remote_last_step={remote_last_step}, final_step={args.final_step}"
    )
    local_history = read_local_history(args.local_run_file)
    final_metrics = read_final_console_metrics(args.training_log, args.final_step)

    missing: list[tuple[int, dict[str, object]]] = []
    for step in range(remote_last_step + 1, args.final_step + 1):
        if step in local_history:
            metrics = local_history[step]
            if metrics.get("training/global_step") != step:
                raise RuntimeError(
                    f"local W&B history step {step} has unexpected "
                    f"training/global_step={metrics.get('training/global_step')}"
                )
            missing.append((step, metrics))
        elif step == args.final_step:
            missing.append((step, final_metrics))
        else:
            raise RuntimeError(f"cannot recover missing W&B history step {step}")

    args.repair_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(args.repair_dir)
    with wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="must",
        dir=str(args.repair_dir),
    ) as resumed_run:
        for step, metrics in missing:
            resumed_run.log(metrics, step=step, commit=True)
            log(f"Restored W&B history step {step} with {len(metrics)} metrics")

    for _ in range(12):
        run = remote_run(args.entity, args.project, args.run_id)
        if run_is_complete(run, args.final_step):
            log(f"Verified W&B run {args.run_id} is finished at step {args.final_step}")
            return
        time.sleep(5)
    raise RuntimeError(
        f"W&B verification failed: state={run.state}, "
        f"last_history_step={run.lastHistoryStep}, "
        f"summary_step={run.summary.get('training/global_step')}"
    )


def main() -> None:
    args = parse_args()
    expected_start_time = process_start_time(args.training_pid)
    if expected_start_time is None:
        log(f"Training PID {args.training_pid} is already stopped")
    else:
        log(f"Waiting for training PID {args.training_pid} to finish")
        while process_start_time(args.training_pid) == expected_start_time:
            time.sleep(args.poll_seconds)

    if not training_reached_final_step(args.training_log, args.final_step):
        raise SystemExit(
            f"Training stopped before successful step {args.final_step}; "
            "leaving the W&B run state unchanged"
        )
    log(f"Training reached step {args.final_step}; allowing {args.grace_seconds}s for normal W&B shutdown")
    time.sleep(args.grace_seconds)
    repair_run(args)


if __name__ == "__main__":
    main()
