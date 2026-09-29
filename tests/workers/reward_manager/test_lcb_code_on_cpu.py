import json
from types import SimpleNamespace

import pytest
import torch

from verl.workers.reward_manager.lcb_code import LiveCodeBenchRewardManager


def test_binary_rewards_use_final_code_valid_tokens_and_no_eos(tmp_path, monkeypatch):
    import verl.workers.reward_manager.lcb_code as module

    calls = []
    def grade(truth, code):
        calls.append((truth, code))
        return {"score": float(code == "print(1)"), "reason": "livecodebench", "seconds": 0.1, "executed_tests": 1}
    monkeypatch.setattr(module, "LiveCodeBenchGrader", lambda plan: grade)
    plan = tmp_path / "grading.json"
    plan.write_text(json.dumps({}))
    texts = {1: "```python\nwrong\n```</think>```python\nprint(1)\n```",
             2: "```python\nprint(1)\n```", 3: "</think>```python\nprint(0)\n```"}
    tokenizer = SimpleNamespace(eos_token_id=9, decode=lambda tokens, **kw: texts[int(tokens[0])])
    manager = LiveCodeBenchRewardManager(tokenizer, 0, grading_plan=str(plan), workers=2)
    class Data:
        batch = {"prompts": torch.tensor([[9, 9]] * 4),
                 "responses": torch.tensor([[1, 8, 9], [2, 8, 9], [3, 8, 8], [0, 0, 0]]),
                 "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0],
                                                 [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])}
        def __len__(self):
            return 4
        def __getitem__(self, index):
            return SimpleNamespace(non_tensor_batch={"reward_model": {"ground_truth": "tests"}})
    data = Data()
    result = manager(data, return_dict=True)
    assert data.batch["acc"].tolist() == [1, 0, 0, 0]
    assert result["reward_tensor"].tolist() == [[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert sorted(code for _, code in calls) == ["print(0)", "print(1)"]
    assert result["reward_extra_info"]["grading_reason"][1] == "missing_thinking_close"


def test_reward_manager_does_not_swallow_infrastructure_errors(tmp_path, monkeypatch):
    import verl.workers.reward_manager.lcb_code as module

    def fail(truth, code):
        raise RuntimeError("grading unavailable")
    monkeypatch.setattr(module, "LiveCodeBenchGrader", lambda plan: fail)
    plan = tmp_path / "grading.json"
    plan.write_text("{}")
    tokenizer = SimpleNamespace(decode=lambda *a, **k: "</think>print(1)")
    manager = LiveCodeBenchRewardManager(tokenizer, 0, grading_plan=str(plan))
    class Data:
        batch = {"prompts": torch.tensor([[1]]), "responses": torch.tensor([[2]]),
                 "attention_mask": torch.tensor([[1, 1]])}
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return SimpleNamespace(non_tensor_batch={"reward_model": {"ground_truth": "tests"}})
    with pytest.raises(RuntimeError, match="grading unavailable"):
        manager(Data())
