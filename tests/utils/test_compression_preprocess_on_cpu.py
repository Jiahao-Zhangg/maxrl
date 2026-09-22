import datasets
import pytest

from examples.maxrl_data_preprocess.compression import EXPECTED_ROWS, convert_dataset, convert_example


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


def test_dataset_keeps_all_source_rows_fields_and_order():
    examples = [{**make_example(), "problem": f"Question {i}"} for i in range(EXPECTED_ROWS)]
    source = datasets.Dataset.from_list(examples)
    converted = convert_dataset(source)
    assert len(converted) == EXPECTED_ROWS
    assert converted["id"] == list(range(EXPECTED_ROWS))
    for name in source.column_names:
        assert converted[name] == source[name]


def test_wrong_subset_size_is_rejected():
    with pytest.raises(ValueError, match="3200-row ER training subset"):
        convert_dataset(datasets.Dataset.from_list([make_example()]))
