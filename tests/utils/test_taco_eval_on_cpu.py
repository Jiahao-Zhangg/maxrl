"""Protect TACO sampling, after-thinking grading and priority-queue handoff."""

import errno
import json

import pytest

from qwen3_experiments import taco_eval as taco
from qwen3_experiments import taco_priority_queue as priority


def test_only_final_code_is_graded_without_eos_requirement():
    code, state = taco.extract_final_code('<think>```python\nprint("wrong")\n```</think>```python\nprint("right")\n```')
    assert (code, state) == ('print("right")', 'ok')
    assert taco.extract_final_code('<think>```python\nprint("right")\n```')[1] == 'missing_thinking_close'
    assert taco.extract_final_code('<think>x</think>print(3)') == ('print(3)', 'ok')
    assert taco.extract_final_code('<think>x</think>')[1] == 'empty_final'


def test_fixed_balanced_shards_reach_32_pending_requests():
    rows = [{"difficulty": level} for level in ("EASY", "MEDIUM", "HARD") for _ in range(200)]
    selected = taco.select_indices(rows)
    assert selected == taco.select_indices(rows)
    assert len(selected) == len(set(selected)) == 200
    for rank in range(4):
        shard = [rows[index]["difficulty"] for index in selected[rank * 50:(rank + 1) * 50]]
        assert shard.count('EASY') == shard.count('MEDIUM') == 25
        assert {taco.concurrency_for(round_number, rank) for round_number in range(2)} == {16, 32}
    assert [taco.concurrency_for(0, rank) for rank in range(4)] == [16, 16, 32, 32]


def test_raw_training_selection_does_not_depend_on_test_spj_metadata():
    rows = [{"difficulty": level} for level in ('EASY', 'MEDIUM') for _ in range(250)]
    original = taco.select_indices(rows)
    for index, row in enumerate(rows):
        row['special_judge'] = {'idx': index, 'special_judge': True}
    assert taco.select_indices(rows) == original


def test_selection_reports_insufficient_questions_without_replacement():
    rows = [{"difficulty": 'EASY'} for _ in range(99)] + [{"difficulty": 'MEDIUM'} for _ in range(150)]
    with pytest.raises(ValueError, match='only 99 available'):
        taco.select_indices(rows)


@pytest.mark.parametrize('payload,issue', [
    ('{}', 'invalid_test_lists'),
    ('{bad json', 'invalid_input_output_json'),
    ('{"inputs":[],"outputs":[]}', 'empty_test_cases'),
    ('{"inputs":["1","2"],"outputs":["3"]}', 'unpaired_test_cases'),
    ('{"inputs":["1"],"outputs":["3"]}', None),
])
def test_ungradeable_test_metadata_is_identified_without_grading(payload, issue):
    assert taco.test_case_issue({'input_output': payload}) == issue


def test_eligibility_filter_preserves_original_indices_and_question_counts():
    rows = [{"difficulty": level} for level in ('EASY', 'MEDIUM') for _ in range(250)]
    allowed = {index for index in range(len(rows)) if index % 3 != 0}
    selected = taco.select_indices(rows, eligible_indices=allowed)
    assert set(selected) <= allowed and len(selected) == len(set(selected)) == 200
    assert sum(rows[index]['difficulty'] == 'EASY' for index in selected) == 100
    assert sum(rows[index]['difficulty'] == 'MEDIUM' for index in selected) == 100


def test_prompt_excludes_solutions_and_private_tests():
    row = {"question": "Add two numbers", "solutions": "SECRET_SOLUTION", "starter_code": "",
           "input_output": json.dumps({"inputs": ["SECRET_TEST"], "outputs": ["SECRET_ANSWER"]})}
    prompt = taco.prompt_for(row)
    assert "Standard Input" in prompt
    assert "SECRET" not in prompt


def test_negative_official_codes_are_failures(tmp_path, monkeypatch):
    taco.write(tmp_path / 'plan.json', {})
    plan = {"scratch": str(tmp_path / 'scratch')}
    sample = {"id": "x", "difficulty": "EASY", "input_output": '{}'}
    monkeypatch.setattr(taco, 'grade_code', lambda *args: {"results": [1, -1], "grader_exception": None})
    result = taco.grade_response(tmp_path, plan, sample, {"text": '<think>x</think>print(3)'}, 16)
    assert result['correct'] is False


def test_priority_gate_requires_audited_results_and_preserves_other_stages(tmp_path):
    root, blocked = tmp_path / 'taco', tmp_path / 'l0'
    stage = {"output_root": str(blocked)}
    assert not priority.gated_dependency(stage, root, blocked, lambda _: True)
    assert priority.gated_dependency({"output_root": str(tmp_path / 'er')}, root, blocked, lambda _: True)
    taco.write(root / 'plan.json', {})
    taco.write(root / 'status.json', {"state": "complete"})
    taco.write(root / 'report/metrics.json', {"pass@1": 0.5})
    taco.write(root / 'report/audit.json', {"complete": True, "plan_sha256": taco.digest(root / 'plan.json'),
                                          "metrics_sha256": taco.digest(root / 'report/metrics.json')})
    assert priority.gated_dependency(stage, root, blocked, lambda _: True)
    assert not priority.gated_dependency(stage, root, blocked, lambda _: False)
    taco.write(root / 'report/metrics.json', {"changed": True})
    assert not priority.gated_dependency(stage, root, blocked, lambda _: True)


@pytest.mark.parametrize('error', [errno.ENOSPC, errno.EDQUOT])
def test_state_disk_full_preserves_scratch_mirror_and_retries(tmp_path, monkeypatch, error):
    real_write = taco.write
    calls = []
    root, scratch = tmp_path / 'root', tmp_path / 'scratch'

    def transient(path, value):
        if path == root / 'status.json':
            calls.append(value)
            if len(calls) == 1:
                raise OSError(error, 'full')
        real_write(path, value)

    monkeypatch.setattr(taco, 'write', transient)
    monkeypatch.setattr(taco.time, 'sleep', lambda _: None)
    taco.persist(root, {"scratch": str(scratch)}, 'status.json', {"state": "running"})
    assert len(calls) == 2
    assert taco.read(root / 'status.json') == taco.read(scratch / 'control_mirrors/status.json')


def test_sandbox_hides_host_home_network_and_credentials(tmp_path):
    plan = {"python_bin": '/independent/env/bin/python', "bubblewrap": '/tools/bwrap',
            "official": '/vendor/taco', "sandbox_runner": '/runner.py'}
    command = taco.sandbox_command(plan, tmp_path / 'payload.json')
    assert '--unshare-all' in command and '--clearenv' in command
    assert '/home' not in command and '/project' not in command
    assert 'HF_TOKEN' not in command and '/dev/nvidia0' not in command
