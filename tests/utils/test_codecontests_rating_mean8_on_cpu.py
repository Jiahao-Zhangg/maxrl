import pytest

from qwen3_experiments.codecontests_rating_mean8_eval import aggregate_bands


def fixture():
    questions, grades = [], []
    for index, (band, rating, correct) in enumerate((("800-1000", 900, 6), ("1400-1600", 1500, 1))):
        for sample in range(8):
            identifier = f"{index}_{sample}"
            questions.append({"id": identifier, "question_hash_id": str(index), "sample_index": sample,
                              "subset_index": index, "original_source_index": index + 100,
                              "difficulty": band, "cf_rating": rating})
            grades.append({"id": identifier, "score": int(sample < correct), "tokens": 100})
    return questions, grades


def test_rating_groups_use_sample_accuracy_not_any_pass():
    questions, grades = fixture()
    result = aggregate_bands(questions, grades, {"800-1000": 1, "1400-1600": 1})
    assert result["800-1000"]["mean_at_8_percent"] == 75
    assert result["1400-1600"]["mean_at_8_percent"] == 12.5


def test_wrong_band_cannot_change_difficulty_comparison():
    questions, grades = fixture()
    questions[0]["cf_rating"] = 1100
    with pytest.raises(ValueError, match="outside"):
        aggregate_bands(questions, grades, {"800-1000": 1, "1400-1600": 1})


def test_incomplete_group_does_not_produce_final_metrics():
    questions, grades = fixture()
    with pytest.raises(ValueError, match="number of questions"):
        aggregate_bands(questions, grades, {"800-1000": 2, "1400-1600": 1})


def test_one_question_cannot_contribute_to_multiple_groups():
    questions, grades = fixture()
    questions[0].update(difficulty="1100-1300", cf_rating=1200)
    with pytest.raises(ValueError, match="multiple rating bands"):
        aggregate_bands(questions, grades, {"800-1000": 1, "1100-1300": 1, "1400-1600": 1})
