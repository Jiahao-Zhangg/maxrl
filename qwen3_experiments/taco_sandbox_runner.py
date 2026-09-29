"""Run the unmodified TACO grader inside an external, isolated namespace."""

import contextlib
import json
import resource
import sys


def main():
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    with open("/input.json") as stream:
        payload = json.load(stream)
    resource.setrlimit(resource.RLIMIT_CPU, (payload["wall_timeout"], payload["wall_timeout"]))
    sys.path.insert(0, "/official")
    from metrics.testing_util import run_test

    # Keep program prints separate from the machine-readable grader result.
    with open("/tmp/grader.log", "w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                result = run_test({"input_output": payload["input_output"]}, test=payload["code"], debug=False)
                result = [int(value.item() if hasattr(value, "item") else value) for value in result]
                record = {"results": result, "grader_exception": None}
            except BaseException as exc:
                record = {"results": [-2], "grader_exception": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
