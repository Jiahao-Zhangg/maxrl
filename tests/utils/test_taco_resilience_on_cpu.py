import errno
from pathlib import Path

import pytest

from qwen3_experiments import taco_resilience as recovery


@pytest.mark.parametrize('text,expected', [
    ('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1 GiB', True),
    ('RuntimeError: CUDA error: out of memory', True),
    ('ValueError: No available memory for the cache blocks', True),
    ('OSError: [Errno 28] No space left on device', False),
    ('OSError: [Errno 122] Disk quota exceeded', False),
    ('ray.exceptions.OutOfMemoryError: Task killed due to the node running low on memory', False),
    ('configured automatic OOM fallback', False),
])
def test_gpu_oom_classification(text, expected):
    assert recovery.cuda_oom(text) is expected


def test_concurrency_fallback_survives_restarts(tmp_path):
    log = tmp_path / 'failed.log'
    log.write_text('RuntimeError: CUDA out of memory\n')
    failed = {'exit_code': 1, 'log': str(log)}
    assert recovery.fallback_concurrency(32, failed, {}) == 16
    assert recovery.fallback_concurrency(32, {}, {'max_num_seqs': 16}) == 16
    assert recovery.fallback_concurrency(32, {'exit_code': 0, 'log': str(log)}, {}) == 32
    log.write_text('Disk quota exceeded\n')
    assert recovery.fallback_concurrency(32, failed, {}) == 32


def test_closed_result_offload_preserves_data_and_receipt(tmp_path):
    source, destination, receipt = tmp_path / 'original', tmp_path / 'local/copy', tmp_path / 'local/receipt.json'
    source.write_bytes(b'audited evaluation responses')
    expected = {'size': source.stat().st_size, 'sha256': recovery.digest(source)}
    assert recovery.copy_closed(source, destination, expected, receipt) == expected['size']
    assert source.is_symlink() and source.resolve() == destination
    assert source.read_bytes() == b'audited evaluation responses'
    assert recovery.read(receipt)['sha256'] == expected['sha256']


def test_corrupt_result_is_not_replaced(tmp_path):
    source = tmp_path / 'original'
    source.write_bytes(b'changed data')
    with pytest.raises(ValueError, match='differs from its audit'):
        recovery.copy_closed(source, tmp_path / 'copy', {'size': 12, 'sha256': '0' * 64}, tmp_path / 'receipt')
    assert not source.is_symlink() and source.read_bytes() == b'changed data'


def test_quota_probe_detects_edquot_even_with_free_space(tmp_path, monkeypatch):
    original = Path.open
    def open_path(path, *args, **kwargs):
        if path.name.startswith('.recovery_probe_'):
            raise OSError(errno.EDQUOT, 'user quota exceeded')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_path)
    result = recovery.probe(tmp_path)
    assert result['free_bytes'] > 0
    assert result['writable'] is False and result['errno'] == errno.EDQUOT


def test_guard_survives_full_status_storage(tmp_path, monkeypatch):
    from qwen3_experiments import taco_grpo_pipeline as pipeline
    calls = []
    def tick(plan):
        calls.append(True)
        if len(calls) == 1:
            raise OSError(errno.ENOSPC, 'full')
    monkeypatch.setattr(recovery, 'tick', tick)
    monkeypatch.setattr(recovery.time, 'sleep', lambda *args: None)
    monkeypatch.setattr(pipeline, 'persist', lambda *args: False)
    monkeypatch.setattr(pipeline, 'state', lambda *args: {'state': 'complete'})
    plan = {'scratch': str(tmp_path), 'output_root': str(tmp_path), 'recovery_policy': {'interval_seconds': 30}}
    monkeypatch.setattr(pipeline, 'verify', lambda *args: plan)
    recovery.serve(plan)
    assert len(calls) == 2


def test_failed_step_cleanup_never_signals_other_job(monkeypatch):
    group = '0::/slurm/step_7/user/task_0\n'
    candidates = [{'pid': pid, 'cgroup': group, 'uid': 1000, 'start_ticks': '1', 'args': ['ray']}
                  for pid in (999991, 999992)]
    killed = []
    monkeypatch.setattr(recovery, 'processes', lambda: candidates)
    monkeypatch.setattr(recovery, 'process', lambda pid: None if any(p == pid for p, _ in killed)
                        else next(p for p in candidates if p['pid'] == pid))
    monkeypatch.setattr(recovery.subprocess, 'check_output', lambda *args, **kwargs:
                        'PID JOBID STEPID LOCALID GLOBALID\n999991 146102 7 0 0\n999992 146103 7 0 0\n')
    monkeypatch.setattr(recovery.os, 'kill', lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(recovery.time, 'sleep', lambda *args: None)
    recovery.terminate_failed_step({'job_id': '146102'}, {'slurm_step_id': '7', 'cgroup': group})
    assert [pid for pid, _ in killed] == [999991]
