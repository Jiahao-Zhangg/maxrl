import errno
import json

import pytest

from qwen3_experiments import competition_eval as evaluation
from qwen3_experiments import competition_grading as grading


def test_resume_rejects_a_response_from_another_model(tmp_path):
    evaluation.write(tmp_path / "plan.json", {"purpose": "test"})
    plan = {"output_root": str(tmp_path), "tasks": {"task": {"model": "base"}},
            "models": {"base": {"revision": "correct-revision"}}}
    question = {"id": "question", "source_index": 3}
    record = {"id": "question", "index": 3, "model_revision": "other-revision",
              "plan_sha256": evaluation.digest(tmp_path / "plan.json")}
    with pytest.raises(ValueError, match="another question, model or plan"):
        evaluation.verify_response(plan, "task", question, record)


def test_status_survives_home_enospc(tmp_path, monkeypatch):
    root, scratch = tmp_path / "home", tmp_path / "compute"
    root.mkdir()
    (root / ".disk_reserve").write_text("reserve")
    original = evaluation.write

    def full_home(path, value):
        if path.parent == root:
            raise OSError(errno.ENOSPC, "full")
        return original(path, value)

    monkeypatch.setattr(evaluation, "write", full_home)
    plan = {"output_root": str(root), "scratch": str(scratch)}
    assert evaluation.persist(plan, "status.json", {"state": "working"})
    assert evaluation.state(plan, "status.json")["state"] == "working"
    assert not (root / ".disk_reserve").exists()


@pytest.mark.parametrize("outcomes,expected", [([1, 1], 1.0), ([1], 0.0), ([1, 0], 0.0), ([-1], 0.0)])
def test_success_requires_every_test(tmp_path, monkeypatch, outcomes, expected):
    class Child:
        returncode = 0

        def communicate(self, timeout):
            return json.dumps({"results": outcomes}), ""

    monkeypatch.setattr(grading, "sandbox_command", lambda *_: ["bwrap"])
    monkeypatch.setattr(grading.subprocess, "Popen", lambda *_a, **_k: Child())
    plan = {"scratch": str(tmp_path), "usaco_tests": str(tmp_path)}
    question = {"dataset": "code_contests", "runtime_limit": 1,
                "tests": [{"input": "x", "output": "x", "group": "public_tests"}] * 2}
    assert grading.grade(plan, question, "print('x')")["score"] == expected


def test_infrastructure_failure_is_not_a_wrong_answer(tmp_path, monkeypatch):
    class Child:
        returncode = 0

        def communicate(self, timeout):
            return json.dumps({"infrastructure_error": "missing official checker"}), ""

    monkeypatch.setattr(grading, "sandbox_command", lambda *_: ["bwrap"])
    monkeypatch.setattr(grading.subprocess, "Popen", lambda *_a, **_k: Child())
    with pytest.raises(RuntimeError, match="missing official checker"):
        grading.grade({"scratch": str(tmp_path), "usaco_tests": str(tmp_path)},
                      {"dataset": "usaco", "runtime_limit": 1, "tests": [{}]}, "print(1)")
