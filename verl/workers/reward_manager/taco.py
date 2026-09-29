"""Binary TACO reward with response-token EOS gating and isolated official grading."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import torch

from qwen3_experiments.code_grading import final_code, grade
from verl.workers.reward_manager import register


@register("taco_official")
class TacoRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source",
                 check_eos=True, score_after_thinking=True, grading_plan=None, workers=32):
        if not score_after_thinking or not grading_plan:
            raise ValueError("TACO requires after-thinking grading and a pinned sandbox plan")
        self.tokenizer = tokenizer
        self.check_eos = check_eos
        self.plan = json.loads(Path(grading_plan).read_text())
        self.workers = workers
        if check_eos and tokenizer.eos_token_id is None:
            raise ValueError("EOS gating requires a tokenizer EOS token")

    def __call__(self, data, return_dict=False):
        responses = data.batch["responses"]
        masks = data.batch["attention_mask"][:, data.batch["prompts"].shape[-1]:].bool()
        reward = torch.zeros_like(responses, dtype=torch.float32)

        def score_one(index):
            tokens = responses[index][masks[index]]
            if not len(tokens):
                return {"score": 0.0, "reason": "empty_response", "seconds": 0.0}
            if self.check_eos and not bool((tokens == self.tokenizer.eos_token_id).any()):
                return {"score": 0.0, "reason": "missing_eos", "seconds": 0.0}
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            code, reason = final_code(text)
            if reason != "ok":
                return {"score": 0.0, "reason": reason, "seconds": 0.0}
            tests = data[index].non_tensor_batch["reward_model"]["ground_truth"]
            return {**grade(self.plan, tests, code), "reason": "official_grader"}

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
            }}
        return reward
