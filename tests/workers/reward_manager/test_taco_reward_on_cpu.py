import json
from types import SimpleNamespace

import pytest
import torch

from qwen3_experiments.code_grading import final_code
from verl.workers.reward_manager.taco import TacoRewardManager


@pytest.mark.parametrize('text,expected', [
    ('thinking only ```python\nprint(1)\n```', 'missing_thinking_close'),
    ('reason </think>   ', 'empty_final'),
    ('reason </think><think>again', 'unclosed_thinking'),
    ('reason ```python\nwrong\n``` </think>```python\nprint(1)\n```', 'ok'),
])
def test_extract_final(text, expected):
    code, reason = final_code(text)
    assert reason == expected
    if reason == 'ok':
        assert code == 'print(1)'


@pytest.mark.parametrize('check_eos,expected_accuracy,expected_rewards,expected_calls', [
    (True, [0, 1, 0], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], 1),
    (False, [1, 1, 1], [[0, 1, 0], [0, 1, 0], [0, 0, 1]], 3),
])
def test_eos_gate_uses_response_tokens_only(tmp_path, monkeypatch, check_eos, expected_accuracy,
                                          expected_rewards, expected_calls):
    import verl.workers.reward_manager.taco as module
    plan = tmp_path / 'grading.json'
    plan.write_text('{}')
    tokenizer = SimpleNamespace(eos_token_id=9, decode=lambda *args, **kwargs: '</think>```python\nprint(1)\n```')
    manager = TacoRewardManager(tokenizer, 0, grading_plan=str(plan), workers=2, check_eos=check_eos)
    calls = []
    def grade(*args):
        calls.append(args)
        return {'score': 1.0, 'seconds': 0.01}
    monkeypatch.setattr(module, 'grade', grade)
    class Data:
        batch = {'prompts': torch.tensor([[9, 2]] * 3),
                 'responses': torch.tensor([[1, 2, 9], [1, 9, 9], [1, 2, 3]]),
                 'attention_mask': torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])}
        def __len__(self):
            return 3
        def __getitem__(self, i):
            return SimpleNamespace(non_tensor_batch={'reward_model': {'ground_truth': json.dumps({'inputs': [''], 'outputs': ['1']})}})
    data = Data()
    result = manager(data, return_dict=True)
    assert data.batch['acc'].tolist() == expected_accuracy
    assert result['reward_tensor'].tolist() == expected_rewards
    assert len(calls) == expected_calls
