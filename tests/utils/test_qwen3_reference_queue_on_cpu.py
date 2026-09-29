"""The reference run must wait for complete, audited predecessor results."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen3_experiments"))
import eval_qwen3_reference as reference


@pytest.mark.parametrize("state,count,audit,ready", [
    ("running", 3456, {"complete": True, "responses_verified": 3456, "complete_questions": 864}, False),
    ("failed", 3456, {"complete": True, "responses_verified": 3456, "complete_questions": 864}, False),
    ("complete", 3456, {"complete": False, "responses_verified": 3456, "complete_questions": 864}, False),
    ("complete", 3456, {"complete": True, "responses_verified": 3455, "complete_questions": 863}, False),
    ("complete", 3456, {"complete": True, "responses_verified": 3456, "complete_questions": 864}, True),
])
def test_wait_for_final_audit(tmp_path, state, count, audit, ready):
    reference.evaluator.write(tmp_path / "status.json", {"state": state, "completed_responses": count})
    reference.evaluator.write(tmp_path / "report/audit.json", audit)
    assert reference.predecessor_ready(tmp_path) is ready


def test_missing_audit_does_not_release_queue(tmp_path):
    reference.evaluator.write(tmp_path / "status.json", {"state": "complete", "completed_responses": 3456})
    assert not reference.predecessor_ready(tmp_path)


def test_question_snapshot_cannot_be_silently_replaced(tmp_path):
    source, destination = tmp_path / "original.json", tmp_path / "paired.json"
    source.write_text('[{"question": "one"}]')
    reference.immutable_copy(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    source.write_text('[{"question": "another"}]')
    with pytest.raises(AssertionError):
        reference.immutable_copy(source, destination)
