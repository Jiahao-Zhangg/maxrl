import datasets
import pytest

from examples.maxrl_data_preprocess.compression import (
    EXPECTED_ROWS,
    audit_qwen3_prompts,
    convert_dataset,
    convert_example,
)


def make_example():
    return {
        "problem": "Find {x} when 2x = 1.",
        "solution": r"Ignore the intermediate \boxed{2}; the answer is \boxed{\frac{1}{2}}.",
        "extracted": r"\frac{1}{2}",
        "year": None,
    }


def test_prompt_and_bare_gold_preserve_er_format():
    converted = convert_example(make_example(), 7)
    assert converted["prompt"] == [
        {
            "role": "user",
            "content": (
                "<｜begin▁of▁sentence｜><｜User｜>"
                "Please reason step by step, and put your final answer within \\boxed{}. "
                "Question: Find {x} when 2x = 1.<｜Assistant｜>"
            ),
        }
    ]
    assert converted["reward_model"] == {"style": "rule", "ground_truth": r"\frac{1}{2}"}
    assert converted["extra_info"] == {"split": "train", "index": 7}


@pytest.mark.parametrize("field", ["problem", "solution", "extracted"])
def test_empty_fields_fail_instead_of_dropping_rows(field):
    example = make_example()
    example[field] = None
    with pytest.raises(ValueError, match=f"empty {field}"):
        convert_example(example, 0)


def test_inconsistent_gold_answer_is_rejected():
    example = make_example()
    example["extracted"] = "2"
    with pytest.raises(ValueError, match="does not match"):
        convert_example(example, 0)


@pytest.mark.parametrize("prompt_format", ["deepseek", "qwen3"])
def test_dataset_keeps_all_source_rows_fields_and_order(prompt_format):
    examples = [{**make_example(), "problem": f"Question {i}"} for i in range(EXPECTED_ROWS)]
    source = datasets.Dataset.from_list(examples)
    converted = convert_dataset(source, prompt_format=prompt_format)
    assert len(converted) == EXPECTED_ROWS
    assert converted["id"] == list(range(EXPECTED_ROWS))
    for name in source.column_names:
        assert converted[name] == source[name]


def test_wrong_subset_size_is_rejected():
    with pytest.raises(ValueError, match="3200-row ER training subset"):
        convert_dataset(datasets.Dataset.from_list([make_example()]))


def test_qwen3_uses_the_grpo_user_instruction_without_deepseek_tokens_or_solution():
    source = make_example()
    converted = convert_example(source, 7, prompt_format="qwen3")
    assert converted["prompt"] == [{
        "role": "user",
        "content": "Find {x} when 2x = 1.\nPlease reason step by step, and put your final answer within \\boxed{}.",
    }]
    assert source["solution"] not in converted["prompt"][0]["content"]
    assert converted["reward_model"] == convert_example(source, 7)["reward_model"]
    assert converted["extra_info"] == {"split": "train", "index": 7}


class ThinkingTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking=True):
        # Native Qwen3 thinking mode lets the model generate its opening marker.
        prompt = messages[0]["content"] + "<|im_start|>assistant\n"
        return prompt if enable_thinking else prompt + "<think>\n\n</think>\n\n"

    def encode(self, text, add_special_tokens=False):
        return list(text)


def test_prompt_audit_checks_all_rows_and_rejects_overflow_instead_of_dropping_it():
    rows = [convert_example(make_example(), 0, prompt_format="qwen3")]
    tokenizer = ThinkingTokenizer()
    result = audit_qwen3_prompts(rows, tokenizer, 1280)
    assert result["verified_prompt_count"] == 1
    assert result["enable_thinking"] is True
    assert result["overlong_prompts"] == 0
    with pytest.raises(ValueError, match="instead of filtering or truncating"):
        audit_qwen3_prompts(rows, tokenizer, result["max_observed_prompt_tokens"] - 1)


def test_prompt_audit_rejects_a_tokenizer_that_ignores_thinking_mode():
    class WrongTokenizer(ThinkingTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    rows = [convert_example(make_example(), 0, prompt_format="qwen3")]
    with pytest.raises(ValueError, match="enable Qwen3 thinking by default"):
        audit_qwen3_prompts(rows, WrongTokenizer(), 1280)
