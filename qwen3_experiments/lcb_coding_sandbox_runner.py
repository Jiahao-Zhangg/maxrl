"""Execute only inside a credential-free bubblewrap namespace."""

import contextlib
import hashlib
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
    sys.set_int_max_str_digits(50000)
    payload = json.loads(Path("/input.json").read_text())
    output = os.dup(1), os.dup(2)
    with open("/tmp/grader.log", "w") as stream:
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                path = Path("/official/lcb_runner/evaluation/testing_util.py")
                if hashlib.sha256(path.read_bytes()).hexdigest() != "b7cb6a8a69807bb868150a61742e25d7bb5328bbe01d514471b6ec43c9fa9ed2":
                    raise ValueError("Official grader hash changed")
                spec = importlib.util.spec_from_file_location("pinned_lcb_testing", path)
                grader = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(grader)
                try:
                    results, metadata = grader.run_test(
                        {"input_output": json.dumps(payload["input_output"])},
                        test=payload["code"], timeout=payload["timeout"], debug=False,
                    )
                    result = {"results": [int(value) for value in results], "metadata": metadata}
                except BaseException as exc:
                    # LCB catches ordinary solution exceptions internally. A
                    # solution may also raise SystemExit or KeyboardInterrupt.
                    result = {"results": [-4], "metadata": {"error": f"{type(exc).__name__}: {exc}"}}
        except Exception as exc:
            result = {"infrastructure_error": f"{type(exc).__name__}: {exc}"}
        finally:
            stream.flush()
            for target in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
                try:
                    target.flush()
                except (OSError, ValueError):
                    pass
            os.dup2(output[0], 1)
            os.dup2(output[1], 2)
            os.close(output[0])
            os.close(output[1])
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
