"""Run synchronous rollout postprocessing with fake engine outputs and CPU tensors."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.utils.debug import performance

pytest.importorskip("vllm")
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

EOS = 99


@pytest.fixture
def generate(monkeypatch):
    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: (0, 0, 0, 0))

    def run(responses, *, cap, force_eos=True, pad=0, n=1):
        rollout = object.__new__(vLLMRollout)
        rollout.config = OmegaConf.create({
            "force_eos": force_eos,
            "calculate_log_probs": False,
            "response_length": cap,
            "free_cache_engine": False,
        })
        rollout.pad_token_id = pad
        rollout.sampling_params = SimpleNamespace(n=n)
        rollout.lora_kwargs = {}
        original_responses = deepcopy(responses)
        samples = [SimpleNamespace(token_ids=ids, finish_reason="length" if len(ids) == cap else "stop") for ids in responses]
        outputs = [SimpleNamespace(outputs=samples[i:i + n]) for i in range(0, len(samples), n)]
        rollout.inference_engine = SimpleNamespace(generate=lambda **kwargs: outputs)
        num_prompts = len(outputs)
        prompts = DataProto.from_dict(
            tensors={
                "input_ids": torch.tensor([[11, 12]] * num_prompts),
                "attention_mask": torch.ones((num_prompts, 2), dtype=torch.long),
                "position_ids": torch.tensor([[0, 1]] * num_prompts),
            },
            non_tensors={"raw_prompt_ids": np.array([[11, 12]] * num_prompts, dtype=object)},
            meta_info={"eos_token_id": EOS},
        )
        result = rollout.generate_sequences(prompts)
        assert responses == original_responses, "The engine's original token lists must stay intact"
        return result

    return run


@pytest.mark.parametrize("pad", [0, EOS])
def test_terminal_eos_precedes_padding_and_masks_in_repeated_rollouts(generate, pad):
    result = generate([[21, 22, 23], [31, EOS], [41], []], cap=5, pad=pad, n=2)
    expected = torch.tensor([
        [21, 22, EOS, pad, pad],
        [31, EOS, pad, pad, pad],
        [EOS, pad, pad, pad, pad],
        [EOS, pad, pad, pad, pad],
    ])
    torch.testing.assert_close(result.batch["responses"], expected)
    torch.testing.assert_close(result.batch["input_ids"][:, 2:], expected)
    assert result.batch["attention_mask"][:, 2:].sum(-1).tolist() == [3, 2, 1, 1]
    assert result.batch["prompts"].tolist() == [[11, 12]] * 4
    assert result.batch["position_ids"].shape == (4, 7)


def test_32000_token_output_gets_eos_without_exceeding_context(generate):
    result = generate([[21] * 32000], cap=32000)
    assert result.batch["responses"].shape == (1, 32000)
    assert result.batch["responses"][0, -1].item() == EOS
    assert result.batch["responses"][0, :-1].eq(21).all()
    assert result.batch["attention_mask"].sum().item() == 32002
    # Keep the engine's finish reason for diagnostics; the sample hit its cap.
    assert result.non_tensor_batch["finish_reasons"].tolist() == ["length"]


def test_disabled_force_eos_preserves_capped_output(generate):
    result = generate([[21, 22, 23]], cap=3, force_eos=False)
    assert result.batch["responses"].tolist() == [[21, 22, 23]]
    assert not result.batch["responses"].eq(EOS).any()


def test_capped_correct_answer_survives_terminal_eos_postprocessing(generate):
    from verl.workers.reward_manager.multi_thread_naive import MathVerifyScorer

    answer_ids = list(map(ord, r"\boxed{1}"))
    original = [ord(" ")] * (32000 - len(answer_ids) - 1) + answer_ids + [ord(" ")]
    result = generate([original], cap=32000)
    token_ids = result.batch["responses"][0].tolist()
    assert token_ids[-1] == EOS
    response_text = "".join(chr(token_id) for token_id in token_ids if token_id != EOS)
    assert MathVerifyScorer().compute_score(response_text, "1", timeout_score=0, per_item_timeout_s=0) == 1


def test_forced_eos_requires_recomputed_log_probabilities():
    config = OmegaConf.create({"force_eos": True, "calculate_log_probs": True})
    with pytest.raises(ValueError, match="recompute log probabilities"):
        vLLMRollout(model_path="unused", config=config, tokenizer=None, model_hf_config=None)
