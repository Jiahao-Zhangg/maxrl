import pytest

from qwen3_experiments.codecontests_atcoder_mean8_eval import aggregate_atcoder


def fixture():
    questions, grades = [], []
    for index, correct in enumerate((6, 1)):
        family = ("arc", "agc")[index]
        for sample in range(8):
            identifier = f"{index}_{sample}"
            questions.append({"id": identifier, "question_hash_id": str(index), "sample_index": sample,
                              "subset_index": index, "original_source_index": index + 100,
                              "difficulty": family, "contest_family": family, "upstream_source": 5,
                              "atcoder_problem_id": f"{family}{index:03d}_a", "atcoder_contest_id": f"{family}{index:03d}",
                              "atcoder_problem_index": "A"})
            grades.append({"id": identifier, "score": int(sample < correct), "tokens": 100})
    return questions, grades


def test_atcoder_mean8_counts_every_sample():
    questions, grades = fixture()
    result = aggregate_atcoder(questions, grades, {"arc": 1, "agc": 1})
    assert result["arc"]["mean_at_8_percent"] == 75
    assert result["agc"]["mean_at_8_percent"] == 12.5


@pytest.mark.parametrize("field,value", [("contest_family", "abc"), ("upstream_source", 2)])
def test_wrong_task_scope_cannot_enter_results(field, value):
    questions, grades = fixture()
    questions[0][field] = value
    with pytest.raises(ValueError, match="outside"):
        aggregate_atcoder(questions, grades, {"arc": 1, "agc": 1})


def test_one_problem_cannot_count_twice_under_different_statements():
    questions, grades = fixture()
    for question in questions[8:]:
        question["atcoder_problem_id"] = questions[0]["atcoder_problem_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_atcoder(questions, grades, {"arc": 1, "agc": 1})


def test_incomplete_selection_cannot_produce_final_metric():
    questions, grades = fixture()
    with pytest.raises(ValueError, match="Incorrect number"):
        aggregate_atcoder(questions, grades, {"arc": 50, "agc": 50})
