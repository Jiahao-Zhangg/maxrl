"""Run the authorized 1-estimator x 3-initialization x 1-seed pilot on GPUs 4,5."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    args.experiment = args.experiment.resolve()
    args.state_dir = args.state_dir.resolve()
    state_path = args.state_dir / "runs/pilot_sweep.json"
    queue = [2450, 3350, 3550]
    active = {}
    records = {}
    launcher = Path(__file__).with_name("run_text_maze_rb.py")

    def save():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(state_path)

    def stop(signum, frame):
        for process, _, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while queue or active:
        for gpu in (4, 5):
            if gpu not in active and queue:
                ckpt = queue.pop(0)
                name = f"pilot_rb_ck{ckpt}_seed0"
                log_path = args.state_dir / "runs" / f"{name}.log"
                stream = log_path.open("a")
                command = [sys.executable, "-u", str(launcher), "--experiment", str(args.experiment),
                           "--state-dir", str(args.state_dir), "--gpu", str(gpu), "--ckpt-step", str(ckpt)]
                process = subprocess.Popen(command, cwd=args.experiment, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (process, ckpt, stream)
                records[str(ckpt)] = {"status": "running", "gpu": gpu, "pid": process.pid,
                                      "started_at": time.time(), "log": str(log_path), "command": command}
                save()
                print(f"Started ckpt-{ckpt}, GPU {gpu}, PID {process.pid}", flush=True)
        for gpu, (process, ckpt, stream) in list(active.items()):
            code = process.poll()
            if code is not None:
                stream.close()
                records[str(ckpt)].update(status="complete" if code == 0 else "failed",
                                           returncode=code, finished_at=time.time())
                active.pop(gpu)
                save()
                print(f"Finished ckpt-{ckpt}, GPU {gpu}, exit={code}", flush=True)
        if queue or active:
            time.sleep(5)
    if any(r["status"] != "complete" for r in records.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
