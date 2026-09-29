import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from qwen3_experiments import lcb_verl_reward as reward
from qwen3_experiments.lcb_coding_grading import GradingInfrastructureError


def test_only_final_code_is_graded_without_eos(monkeypatch):
    calls = []

    def grade(truth, code):
        calls.append((truth, code))
        return {"score": 1.0, "reason": "livecodebench", "seconds": 0.1, "executed_tests": 3}

    with ThreadPoolExecutor(max_workers=2) as executor:
        monkeypatch.setattr(reward, "_service", lambda *args: (grade, executor))
        answer = "<think>```python\nprint('wrong')\n```</think>```python\nprint('right')\n```"
        result = asyncio.run(reward.compute_score("lcb", answer, "truth", grading_plan="plan"))
    assert calls == [("truth", "print('right')")]
    assert result["score"] == result["acc"] == 1.0
    assert result["grading_executed_tests"] == 3


def test_missing_thinking_close_never_executes_code(monkeypatch):
    def fail(*args):
        raise AssertionError("An unclosed thinking block must never reach the sandbox")

    monkeypatch.setattr(reward, "_service", fail)
    result = asyncio.run(
        reward.compute_score("lcb", "<think>```python\nprint(1)\n```", {}, grading_plan="plan")
    )
    assert result["score"] == 0
    assert result["grading_executed_tests"] == 0


def test_adapter_propagates_infrastructure_failure(monkeypatch):
    calls = []

    def fail(*args):
        calls.append(1)
        raise GradingInfrastructureError("sandbox unavailable")

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(reward, "_service", lambda *args: (fail, executor))
        with pytest.raises(GradingInfrastructureError, match="unavailable"):
            asyncio.run(
                reward.compute_score("lcb", "</think>```python\nprint(1)\n```", {}, grading_plan="plan")
            )
    assert len(calls) == 1


def test_grading_timeout_keeps_all_eight_rewards_and_binning(monkeypatch):
    from verl.trainer.ppo.prompt_binning import compute_prompt_binning

    def grade(truth, code):
        timeout = code == "print('timeout')"
        return {"score": 0.0 if timeout else 1.0,
                "reason": "grading_timeout" if timeout else "livecodebench",
                "seconds": 0.1, "executed_tests": 0 if timeout else 1}

    async def group():
        return await asyncio.gather(*(
            reward.compute_score("lcb_coding", f"</think>```python\nprint('{name}')\n```", {}, grading_plan="plan")
            for name in ["timeout"] + ["ok"] * 7
        ))

    with ThreadPoolExecutor(max_workers=8) as executor:
        monkeypatch.setattr(reward, "_service", lambda *args: (grade, executor))
        results = asyncio.run(group())
    assert len(results) == 8
    assert results[0]["score"] == results[0]["acc"] == results[0]["accuracy"] == 0
    assert results[0]["grading_reason"] == "grading_timeout"
    metrics = compute_prompt_binning([r["score"] for r in results], ["prompt"] * 8, ["lcb_coding"] * 8,
                                     expected_group_size=8)
    assert metrics["train_all_datasets_binning/fraction_of_prompts_in_(0.5, 1.0)"] == 1.0
