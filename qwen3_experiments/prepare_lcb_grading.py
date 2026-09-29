"""Prepare a pinned CPU grading runtime on the training compute node."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen

from qwen3_experiments.lcb_coding_format import LCB_REVISION, LCB_TESTING_SHA256


def prepare(root, bubblewrap, python_bin):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not Path(bubblewrap).is_file() or not Path(python_bin).is_file():
        raise ValueError("An existing bubblewrap executable and Python environment are required")
    official = root / "official"
    for relative in ("lcb_runner/evaluation/testing_util.py", "LICENSE"):
        url = f"https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/{LCB_REVISION}/{relative}"
        data = urlopen(url, timeout=60).read()
        if relative.endswith("testing_util.py") and hashlib.sha256(data).hexdigest() != LCB_TESTING_SHA256:
            raise ValueError("Downloaded LCB source hash mismatch")
        destination = official / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    runner = root / "lcb_coding_sandbox_runner.py"
    shutil.copyfile(Path(__file__).with_name(runner.name), runner)
    plan = {"scratch": str(root), "official": str(official), "bubblewrap": str(Path(bubblewrap).resolve()),
            "python_bin": str(Path(python_bin).absolute()), "sandbox_runner": str(runner),
            "sandbox_runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
            "lcb_revision": LCB_REVISION}
    destination = root / "grading_plan.json"
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--bubblewrap", required=True, type=Path)
    parser.add_argument("--python-bin", default=sys.executable, type=Path)
    args = parser.parse_args()
    print(prepare(args.root, args.bubblewrap, args.python_bin))


if __name__ == "__main__":
    main()
