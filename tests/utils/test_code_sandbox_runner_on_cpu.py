"""The grader's subprocess output must never contaminate its JSON result."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('raise_error', [False, True])
def test_json_channel_survives_native_and_child_output(tmp_path, raise_error):
    log = tmp_path / 'grader.log'
    script = r'''
import json, os, subprocess, sys
from qwen3_experiments.code_sandbox_runner import capture_grader_output

try:
    with capture_grader_output(sys.argv[1]):
        print('python stdout')
        print('python stderr', file=sys.stderr)
        print('buffered original stdout', file=sys.__stdout__)
        os.write(1, b'native stdout\n')
        os.write(2, b'native stderr\n')
        subprocess.run([sys.executable, '-c',
                        'import sys; print(0); print("child stderr", file=sys.stderr)'], check=True)
        if sys.argv[2] == 'True':
            raise RuntimeError('generated program failed')
except RuntimeError:
    pass
print(json.dumps({'results': [-3]}), flush=True)
'''
    result = subprocess.run([sys.executable, '-c', script, str(log), str(raise_error)],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True,
                            text=True, check=True, timeout=20)
    assert json.loads(result.stdout) == {'results': [-3]}
    assert result.stderr == ''
    captured = log.read_text()
    for text in ('python stdout', 'python stderr', 'buffered original stdout',
                 'native stdout', 'native stderr', '0\n', 'child stderr'):
        assert text in captured
