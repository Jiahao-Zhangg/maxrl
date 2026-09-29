"""Run the pinned LeetCodeDataset checker inside a bubblewrap namespace."""

import importlib.util
import json
import os
from pathlib import Path
import resource
import sys


def main():
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    payload = json.loads(Path("/input.json").read_text())
    saved = os.dup(1), os.dup(2)
    with open("/tmp/checker.log", "w") as output:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            path = "/official/leetcode_dataset/eval_lcd/execution.py"
            spec = importlib.util.spec_from_file_location("official_leetcode_execution", path)
            checker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(checker)
            result = checker.check_correctness(payload["problem"], payload["code"], payload["timeout"])
        except Exception as exc:
            result = {"infrastructure_error": f"{type(exc).__name__}: {exc}"}
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
