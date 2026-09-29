"""One binary LiveCodeBench reward manager for stdin and function tasks."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import torch

from qwen3_experiments.code_grading import final_code
from qwen3_experiments.lcb_coding_grading import LiveCodeBenchGrader
from verl.workers.reward_manager import register


@register("lcb_code")
class LiveCodeBenchRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source",
                 check_eos=False, score_after_thinking=True, grading_plan=None, workers=16):
        if check_eos or not score_after_thinking or not grading_plan:
            raise ValueError("LCB coding requires after-thinking, no EOS gate, and a pinned grading plan")
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be positive")
        self.tokenizer = tokenizer
        self.grader = LiveCodeBenchGrader(json.loads(Path(grading_plan).read_text()))
        self.workers = workers

    def __call__(self, data, return_dict=False):
        responses = data.batch["responses"]
        masks = data.batch["attention_mask"][:, data.batch["prompts"].shape[-1]:].bool()
        reward = torch.zeros_like(responses, dtype=torch.float32)

        def score_one(index):
            tokens = responses[index][masks[index]]
            if not len(tokens):
                return {"score": 0.0, "reason": "empty_response", "seconds": 0.0, "executed_tests": 0}
            code, reason = final_code(self.tokenizer.decode(tokens, skip_special_tokens=True))
            if reason != "ok":
                return {"score": 0.0, "reason": reason, "seconds": 0.0, "executed_tests": 0}
            truth = data[index].non_tensor_batch["reward_model"]["ground_truth"]
            return self.grader(truth, code)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            scores = list(pool.map(score_one, range(len(data))))
        for index, result in enumerate(scores):
            positions = masks[index].nonzero().flatten()
            if len(positions):
                reward[index, positions[-1]] = result["score"]
        data.batch["acc"] = torch.tensor([s["score"] for s in scores], device=responses.device)
        if return_dict:
            return {"reward_tensor": reward, "reward_extra_info": {
                "accuracy": [s["score"] for s in scores],
                "grading_seconds": [s["seconds"] for s in scores],
                "grading_reason": [s["reason"] for s in scores],
                "grading_executed_tests": [s["executed_tests"] for s in scores],
            }}
        return reward
