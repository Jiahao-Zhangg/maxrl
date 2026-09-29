"""Run pinned NeMo Gym code_gen verification inside bubblewrap only."""

import ast
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys


def main():
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 << 20, 32 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    payload = json.loads(Path("/input.json").read_text())
    sys.path.insert(0, "/official/resources_servers/code_gen")
    original = os.dup(1), os.dup(2)
    with open("/tmp/grader.log", "w") as log:
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                import numpy as np
                from lcb_integration.testing_util import run_test

                # Keep the upstream multiprocessing/global-timeout implementation
                # verbatim, without initializing its unrelated Ray server wrapper.
                source = Path("/official/resources_servers/code_gen/lcb_integration/compute_code_generation_metrics.py")
                tree = ast.parse(source.read_text())
                functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                             and node.name in ("_temp_run", "check_correctness")]
                if len(functions) != 2:
                    raise ValueError("Unexpected official checker source")
                namespace = {"multiprocessing": multiprocessing, "json": json, "run_test": run_test}
                exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
                sys.set_int_max_str_digits(50000)
                results, metadata = namespace["check_correctness"](
                    {"input_output": json.dumps(payload["unit_tests"])}, payload["code"],
                    timeout=payload["timeout"], debug=False,
                )
                record = {"results": [int(v.item() if isinstance(v, np.generic) else v) for v in results],
                          "metadata": metadata}
        except Exception as exc:
            record = {"infrastructure_error": f"{type(exc).__name__}: {exc}"}
        finally:
            log.flush()
            os.dup2(original[0], 1)
            os.dup2(original[1], 2)
            for fd in original:
                os.close(fd)
    print(json.dumps(record, default=str), flush=True)


if __name__ == "__main__":
    main()
