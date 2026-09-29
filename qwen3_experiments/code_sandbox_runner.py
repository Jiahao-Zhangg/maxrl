"""This entry point must run only inside the bubblewrap namespace."""

import contextlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys


@contextlib.contextmanager
def capture_grader_output(log_path):
    """Keep child-process output out of the JSON result pipe as well."""
    sys.stdout.flush()
    sys.stderr.flush()
    with open(log_path, 'w') as log:
        stdout_fd, stderr_fd = os.dup(1), os.dup(2)
        try:
            # TACO's file-I/O retry inherits fd 1/2 instead of capturing them.
            # Python's redirect_stdout alone cannot intercept that output.
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                yield
        finally:
            # Flush buffered writes to sys.__stdout__ before restoring its fd.
            for stream in (log, sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
                try:
                    stream.flush()
                except (AttributeError, OSError, ValueError):
                    pass
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (256,) * 2)
    payload = json.loads(Path('/input.json').read_text())
    resource.setrlimit(resource.RLIMIT_CPU, (payload['wall_timeout'],) * 2)
    sys.set_int_max_str_digits(50000)
    sys.path.insert(0, '/official')
    with capture_grader_output('/tmp/grader.log'):
        if payload['kind'] == 'taco':
            from metrics.testing_util import run_test
        else:
            spec = importlib.util.spec_from_file_location('lcb_grader', '/official/lcb_runner/evaluation/testing_util.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            run_test = module.run_test
        try:
            kwargs = {'timeout': 6} if payload['kind'] == 'lcb' else {}
            result = run_test({'input_output': payload['input_output']}, test=payload['code'], debug=False, **kwargs)
            metadata = {}
            if payload['kind'] == 'lcb':
                result, metadata = result
            record = {'results': [int(v.item() if hasattr(v, 'item') else v) for v in result],
                      'metadata': metadata}
        except BaseException as exc:
            record = {'results': [-2], 'error': f'{type(exc).__name__}: {exc}'}
    print(json.dumps(record, default=str), flush=True)


if __name__ == '__main__':
    main()
