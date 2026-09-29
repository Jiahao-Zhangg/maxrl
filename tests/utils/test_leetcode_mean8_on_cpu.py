import pytest

from qwen3_experiments import competition_eval as base
from qwen3_experiments.leetcode_mean8_eval import aggregate_difficulties, predecessor_complete


def samples():
    questions, grades = [], []
    for index, (level, correct) in enumerate((("Medium", 6), ("Hard", 1))):
        for sample in range(8):
            identifier = f"{index}_{sample}"
            questions.append({"id": identifier, "question_hash_id": str(index), "sample_index": sample,
                              "subset_index": index, "original_source_index": index + 100,
                              "difficulty": level, "dataset": "leetcode", "task_id": f"task-{index}"})
            grades.append({"id": identifier, "score": int(sample < correct), "tokens": 100})
    return questions, grades


def test_mean8_is_sample_accuracy_in_each_difficulty():
    questions, grades = samples()
    result = aggregate_difficulties(questions, grades, {"Medium": 1, "Hard": 1})
    assert result["Medium"]["mean_at_8_percent"] == 75
    assert result["Hard"]["mean_at_8_percent"] == 12.5


def test_duplicate_sample_cannot_replace_missing_sample():
    questions, grades = samples()
    questions[0]["sample_index"] = 1
    with pytest.raises(ValueError, match="eight distinct"):
        aggregate_difficulties(questions, grades, {"Medium": 1, "Hard": 1})


def test_difficulty_cannot_change_between_samples():
    questions, grades = samples()
    questions[0]["difficulty"] = "Hard"
    with pytest.raises(ValueError, match="identity changes"):
        aggregate_difficulties(questions, grades, {"Medium": 1, "Hard": 1})


def test_partial_sample_cannot_be_reported_as_fifty_questions():
    questions, grades = samples()
    with pytest.raises(ValueError, match="Incorrect number"):
        aggregate_difficulties(questions, grades, {"Medium": 50, "Hard": 50})


def predecessor_fixture(tmp_path, state="complete"):
    previous = {"output_root": str(tmp_path), "scratch": str(tmp_path),
                "order": ["prior"], "tasks": {"prior": {"questions": 1}}}
    base.write(tmp_path / "plan.json", previous)
    plan_hash = base.digest(tmp_path / "plan.json")
    base.write(tmp_path / "queue_status.json", {"state": state})
    base.write(tmp_path / "mean8_summary.json", {"plan_sha256": plan_hash})
    directory = tmp_path / "evaluation/prior"
    base.write(directory / "responses/0.json", {"response": "example"})
    base.write(directory / "grades/0.json", {"score": 1})
    base.write(directory / "metrics.json", {"mean_at_8_percent": 100})
    base.write(directory / "audit.json", {
        "complete": True, "questions": 1, "plan_sha256": plan_hash,
        "metrics_sha256": base.digest(directory / "metrics.json"),
        "response_hashes": {"0": base.digest(directory / "responses/0.json")},
        "grade_hashes": {"0": base.digest(directory / "grades/0.json")},
    })
    return {"predecessor": {"output_root": str(tmp_path), "plan_sha256": plan_hash}}


def test_followup_waits_until_predecessor_queue_finishes(tmp_path):
    plan = predecessor_fixture(tmp_path, state="evaluating")
    assert not predecessor_complete(plan)


def test_followup_accepts_verified_completed_results(tmp_path):
    assert predecessor_complete(predecessor_fixture(tmp_path))


def test_changed_prior_grades_prevent_followup_launch(tmp_path):
    plan = predecessor_fixture(tmp_path)
    base.write(tmp_path / "evaluation/prior/grades/0.json", {"score": 0})
    with pytest.raises(ValueError, match="changed after its audit"):
        predecessor_complete(plan)


def test_incomplete_prior_audit_prevents_followup_launch(tmp_path):
    plan = predecessor_fixture(tmp_path)
    path = tmp_path / "evaluation/prior/audit.json"
    audit = base.read(path)
    audit["grade_hashes"] = {}
    base.write(path, audit)
    with pytest.raises(ValueError, match="incomplete"):
        predecessor_complete(plan)
