import errno
from pathlib import Path

import pytest

from qwen3_experiments import taco_grpo_pipeline as pipeline


def test_full_home_falls_back_and_state_is_readable(tmp_path, monkeypatch):
    root, scratch = tmp_path / 'home', tmp_path / 'scratch'
    root.mkdir(); scratch.mkdir()
    plan = {'output_root': str(root), 'scratch': str(scratch)}
    original = pipeline.write
    def write(path, value):
        if Path(path).is_relative_to(root):
            raise OSError(errno.ENOSPC, 'full')
        return original(path, value)
    monkeypatch.setattr(pipeline, 'write', write)
    assert pipeline.persist(plan, 'status.json', {'state': 'running'})
    assert pipeline.state(plan, 'status.json')['state'] == 'running'


def test_both_disks_full_does_not_kill_controller(tmp_path, monkeypatch):
    plan = {'output_root': str(tmp_path / 'home'), 'scratch': str(tmp_path / 'scratch')}
    def fail(*args):
        raise OSError(errno.EDQUOT, 'full')
    monkeypatch.setattr(pipeline, 'write', fail)
    assert pipeline.persist(plan, 'status.json', {'state': 'running'}) is False


def test_incomplete_previous_queue_cannot_start_training(tmp_path):
    previous = tmp_path / 'previous'
    previous.mkdir()
    pipeline.write(previous / 'plan.json', {'test': True})
    pipeline.write(previous / 'queue_status.json', {'state': 'running_or_waiting_for_gpus'})
    plan = {'dependency': {'root': str(previous), 'plan_sha256': pipeline.digest(previous / 'plan.json')}}
    assert pipeline.dependency_ready(plan) is False


def test_resume_verified_deletion_of_empty_checkpoint(tmp_path):
    checkpoint = tmp_path / 'global_step_10'
    (checkpoint / 'actor').mkdir(parents=True)
    pipeline.finish_verified_deletion(checkpoint, {'files': {}})
    assert not checkpoint.exists()


def test_preserve_changed_checkpoint_after_interrupted_deletion(tmp_path):
    checkpoint = tmp_path / 'global_step_10'
    checkpoint.mkdir()
    shard = checkpoint / 'data.pt'
    shard.write_bytes(b'changed since remote verification')
    with pytest.raises(ValueError, match='changed'):
        pipeline.finish_verified_deletion(checkpoint, {'files': {}})
    assert shard.exists()


@pytest.fixture
def completed_predecessor(tmp_path, monkeypatch):
    root, scratch = tmp_path / 'training', tmp_path / 'training-scratch'
    previous, previous_scratch = tmp_path / 'evaluation', tmp_path / 'evaluation-scratch'
    for path in (root, scratch, previous, previous_scratch):
        path.mkdir()
    base = scratch / 'base'
    base.mkdir()
    (base / 'weights.bin').write_bytes(b'training base must survive cleanup')
    stages, frozen = [], {}
    for key in ('er_final', 'l0_step100', 'qwen3_1_7b'):
        folder = previous_scratch / 'models' / key
        model_path = folder / 'model'
        model_path.mkdir(parents=True)
        (model_path / 'weights.bin').write_bytes(key.encode())
        identity = {'repo': f'owner/{key}', 'revision': 'a' * 40}
        name = f'prepared_models/{key}.json'
        pipeline.write(previous / name, {**identity, 'path': str(model_path),
                       'merged_files': {'weights.bin': pipeline.digest(model_path / 'weights.bin')}})
        frozen[name] = pipeline.digest(previous / name)
        stages.append({'key': key, 'model': identity, 'owned_model_cache': str(folder)})
    pipeline.write(previous / 'plan.json', {'scratch': str(previous_scratch), 'stages': stages, 'frozen_files': frozen})
    pipeline.write(previous / 'queue_status.json', {'state': 'complete'})
    pipeline.write(previous / 'report/metrics.json', [{'keep': 'evaluation result'}])
    pipeline.write(previous / 'report/audit.json', {'complete': True, 'points': 105, 'new_points': 90,
                   'all_rollout_ledgers_verified': True, 'metrics_sha256': pipeline.digest(previous / 'report/metrics.json')})
    plan = {'output_root': str(root), 'runtime': str(root / 'runtime'), 'scratch': str(scratch),
            'checkpoint_dir': str(scratch / 'checkpoints'),
            'base_model': {'path': str(base), 'files_sha256': {'weights.bin': pipeline.digest(base / 'weights.bin')}},
            'dependency': {'root': str(previous), 'plan_sha256': pipeline.digest(previous / 'plan.json')}}
    pipeline.write(root / 'plan.json', plan)
    monkeypatch.setattr(pipeline, 'cache_users', lambda caches: [])
    return plan, previous, previous_scratch, base


def test_pretraining_cleanup_keeps_base_and_results(completed_predecessor):
    plan, previous, scratch, base = completed_predecessor
    result = pipeline.cleanup_predecessor_caches(plan)
    assert result['state'] == 'complete'
    assert not list((scratch / 'models').iterdir())
    assert (base / 'weights.bin').read_bytes() == b'training base must survive cleanup'
    assert pipeline.read(previous / 'report/metrics.json') == [{'keep': 'evaluation result'}]
    assert pipeline.cleanup_predecessor_caches(plan)['deleted_bytes'] == result['deleted_bytes']


def test_pretraining_cleanup_accepts_pinned_merge_metadata(completed_predecessor):
    plan, previous, scratch, _ = completed_predecessor
    model = pipeline.read(previous / 'prepared_models/l0_step100.json')
    model.pop('path')
    folder = scratch / 'models/l0_step100'
    pipeline.write(folder / 'model_receipt.json', model)
    (folder / 'model_prepare.lock').touch()
    assert pipeline.cleanup_predecessor_caches(plan)['state'] == 'complete'
    assert not folder.exists()


@pytest.mark.parametrize('blocker', ['incomplete', 'active', 'receipt_full', 'changed_base', 'overlap'])
def test_pretraining_cleanup_blocks_unsafe_handoff(completed_predecessor, monkeypatch, blocker):
    plan, previous, scratch, base = completed_predecessor
    if blocker == 'incomplete':
        pipeline.write(previous / 'queue_status.json', {'state': 'running'})
    elif blocker == 'active':
        monkeypatch.setattr(pipeline, 'cache_users', lambda caches: [123])
    elif blocker == 'receipt_full':
        monkeypatch.setattr(pipeline, 'persist', lambda *args: False)
    elif blocker == 'changed_base':
        (base / 'weights.bin').write_bytes(b'wrong base')
    else:
        plan['base_model']['path'] = str(scratch / 'models/er_final/model')
    with pytest.raises((RuntimeError, ValueError, OSError)):
        pipeline.cleanup_predecessor_caches(plan)
    assert len(list((scratch / 'models').glob('*/model/weights.bin'))) == 3


def test_training_never_launches_when_cleanup_fails(completed_predecessor, monkeypatch):
    plan, _, _, _ = completed_predecessor
    plan['holder_locks'] = []
    monkeypatch.setattr(pipeline, 'gpu_busy', lambda: False)
    def incomplete_cleanup(plan):
        raise RuntimeError('cleanup is incomplete')
    monkeypatch.setattr(pipeline, 'cleanup_predecessor_caches', incomplete_cleanup)
    started = []
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *args, **kwargs: started.append(True))
    with pytest.raises(RuntimeError, match='cleanup is incomplete'):
        pipeline.train(plan)
    assert not started


def test_pretraining_cleanup_resumes_partial_deletion(completed_predecessor, monkeypatch):
    plan, _, scratch, _ = completed_predecessor
    original = pipeline.finish_verified_deletion
    def interrupted(folder, receipt):
        if folder.name == 'l0_step100':
            raise OSError(errno.EIO, 'interrupted cleanup')
        return original(folder, receipt)
    monkeypatch.setattr(pipeline, 'finish_verified_deletion', interrupted)
    with pytest.raises(OSError):
        pipeline.cleanup_predecessor_caches(plan)
    assert not (scratch / 'models/er_final').exists()
    assert (scratch / 'models/l0_step100').exists()
    monkeypatch.setattr(pipeline, 'finish_verified_deletion', original)
    result = pipeline.cleanup_predecessor_caches(plan)
    assert result['state'] == 'complete'
    assert result['deleted_bytes'] == sum(len(key) for key in ('er_final', 'l0_step100', 'qwen3_1_7b'))
    assert not list((scratch / 'models').iterdir())
