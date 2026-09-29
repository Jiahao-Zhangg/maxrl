import json

import pytest

from qwen3_experiments import nemotron_mean8_eval as evaluation


def records(correct_counts):
    questions, grades = [], []
    for index, correct in enumerate(correct_counts):
        for sample in range(8):
            identifier = f"{index}__sample_{sample}"
            questions.append({"id": identifier, "question_hash_id": str(index),
                              "sample_index": sample, "subset_index": index,
                              "original_source_index": index + 10, "difficulty": "Codeforces"})
            grades.append({"id": identifier, "score": int(sample < correct), "tokens": 100})
    return questions, grades


def test_mean_at_eight_counts_samples_instead_of_any_pass():
    questions, grades = records([1, 7])
    result = evaluation.aggregate(questions, grades)
    assert result["mean_at_8_percent"] == 50
    assert result["correct_samples"] == 8
    assert result["total_samples"] == 16
    assert result["correct_samples_histogram"] == {1: 1, 7: 1}


def test_incomplete_question_cannot_produce_a_final_mean():
    questions, grades = records([2])
    with pytest.raises(ValueError, match="eight distinct"):
        evaluation.aggregate(questions[:-1], grades[:-1])


def test_duplicated_sample_cannot_substitute_for_missing_sample():
    questions, grades = records([2])
    questions[-1] = dict(questions[0])
    grades[-1] = dict(grades[0])
    with pytest.raises(ValueError, match="eight distinct"):
        evaluation.aggregate(questions, grades)


def test_grades_must_belong_to_the_same_questions():
    questions, grades = records([2])
    grades[0]["id"] = "another-question"
    with pytest.raises(ValueError, match="identity"):
        evaluation.aggregate(questions, grades)


def test_empty_run_is_not_a_result():
    with pytest.raises(ValueError, match="Incomplete"):
        evaluation.aggregate([], [])


@pytest.mark.parametrize("outcomes,expected", [([1, 1], 1.0), ([1], 0.0), ([1, 0], 0.0), ([-1], 0.0)])
def test_all_test_cases_must_pass(tmp_path, monkeypatch, outcomes, expected):
    class Child:
        returncode = 0

        def communicate(self, timeout):
            return json.dumps({"results": outcomes}), ""

    monkeypatch.setattr(evaluation, "sandbox_command", lambda *_: ["bwrap"])
    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *_a, **_k: Child())
    tests = tmp_path / "tests.json"
    tests.write_text(json.dumps({"inputs": ["1", "2"], "outputs": ["1", "2"]}))
    plan = {"scratch": str(tmp_path), "unit_test_timeout_secs": 10}
    assert evaluation.grade(plan, {"unit_tests_file": str(tests)}, "print(1)")["score"] == expected


def test_infrastructure_failure_does_not_become_a_wrong_answer(tmp_path, monkeypatch):
    class Child:
        returncode = 0

        def communicate(self, timeout):
            return json.dumps({"infrastructure_error": "missing official checker"}), ""

    monkeypatch.setattr(evaluation, "sandbox_command", lambda *_: ["bwrap"])
    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *_a, **_k: Child())
    tests = tmp_path / "tests.json"
    tests.write_text(json.dumps({"inputs": ["1"], "outputs": ["1"]}))
    with pytest.raises(RuntimeError, match="missing official checker"):
        evaluation.grade({"scratch": str(tmp_path), "unit_test_timeout_secs": 10},
                         {"unit_tests_file": str(tests)}, "print(1)")
