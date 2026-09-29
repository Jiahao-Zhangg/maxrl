"""Execute only inside the credential-free bubblewrap namespace."""

import contextlib
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import types


@contextlib.contextmanager
def captured_output():
    sys.stdout.flush()
    sys.stderr.flush()
    with open("/tmp/grader.log", "w") as stream:
        original = os.dup(1), os.dup(2)
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                yield
        finally:
            for output in (stream, sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
                try:
                    output.flush()
                except (OSError, ValueError):
                    pass
            os.dup2(original[0], 1)
            os.dup2(original[1], 2)
            for fd in original:
                os.close(fd)


def usaco(question, code):
    sys.path.insert(0, "/official/usaco")
    # The checker only needs its enum. Avoid importing the package's model and
    # retrieval integrations, which otherwise initialize unrelated libraries.
    for name, directory in (("USACOBench", "/official/usaco/USACOBench"),
                            ("USACOBench.evaluation", "/official/usaco/USACOBench/evaluation")):
        package = types.ModuleType(name)
        package.__path__ = [directory]
        sys.modules[name] = package
    path = "/official/usaco/USACOBench/evaluation/judges/usaco_utils.py"
    spec = importlib.util.spec_from_file_location("official_usaco_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outcomes, details = [], []
    for number, test in enumerate(question["tests"], 1):
        prediction = Path(f"/tmp/prediction_{number}.txt")
        # An early interpreter failure can otherwise make the upstream reader fail.
        prediction.touch()
        result = module.check_correctness(
            code, number, question["runtime_limit"], question["memory_limit_mb"],
            str(Path("/tests") / test["input_file"]), str(prediction),
            str(Path("/tests") / test["output_file"]),
        )
        result_type = int(result["result_type"])
        outcomes.append({1: 1, 2: 0, 3: -1, 4: -2, 5: -2, 6: -2}.get(result_type, -2))
        details.append({"test": number, "official_result_type": result_type,
                        "status": result["status"][:1500]})
        if outcomes[-1] != 1:
            break
    return {"results": outcomes, "metadata": details,
            "checker": "official USACOBench check_correctness, fail-fast"}


def code_contests(question, code):
    library = ctypes.CDLL("/official/code_contests/liboutputs_match.so")
    compare = library.official_outputs_match
    compare.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
    compare.restype = ctypes.c_int
    try:
        compile(code, "solution.py", "exec")
    except (SyntaxError, ValueError) as exc:
        return {"results": [-2], "metadata": [{"status": "compilation_error", "detail": str(exc)[:1000]}]}

    def limits():
        # Match TestOptions resource accounting in the pinned upstream sandbox.
        memory = question["memory_limit_bytes"] + (32 << 20)
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        cpu = max(1, int(question["runtime_limit"]))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    outcomes, details = [], []
    for number, test in enumerate(question["tests"]):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            work = Path(directory)
            (work / "solution.py").write_text(code)
            (work / "stdin.txt").write_text(test["input"])
            with (work / "stdin.txt").open("rb") as stdin, (work / "stdout.txt").open("wb") as stdout, (work / "stderr.txt").open("wb") as stderr:
                process = subprocess.Popen([sys.executable, "-I", str(work / "solution.py")],
                                           cwd=work, stdin=stdin, stdout=stdout, stderr=stderr,
                                           preexec_fn=limits, start_new_session=True)
                timed_out = False
                try:
                    process.wait(timeout=30 * question["runtime_limit"])
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            actual = (work / "stdout.txt").read_bytes()
            expected = test["output"].encode()
            if timed_out or process.returncode in (-signal.SIGKILL, -signal.SIGXCPU):
                value, status = -1, "time_limit_exceeded"
            elif process.returncode:
                value, status = -2, "runtime_error"
            else:
                value = int(compare(actual, len(actual), expected, len(expected)))
                status = "accepted" if value else "wrong_answer"
            outcomes.append(value)
            details.append({"test": number, "group": test["group"], "status": status,
                            "returncode": process.returncode,
                            "stderr": (work / "stderr.txt").read_text(errors="replace")[:1000]})
        if value != 1:
            break
    return {"results": outcomes, "metadata": details,
            "checker": "unmodified compiled CodeContests OutputsMatch; bubblewrap execution"}


def main():
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    payload = json.loads(Path("/input.json").read_text())
    with captured_output():
        try:
            function = {"usaco": usaco, "code_contests": code_contests}[payload["question"]["dataset"]]
            result = function(payload["question"], payload["code"])
        except Exception as exc:
            result = {"infrastructure_error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
