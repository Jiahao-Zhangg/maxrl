"""CPU-only regression tests for isolated shared-budget evaluation supervisors."""

import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("protocol", ["both", "eval3_skip_solved", "eval3_iid"])
def test_completed_eval3_protocols_are_isolated(tmp_path, monkeypatch, protocol):
    scripts = Path(__file__).resolve().parents[2] / "qwen3_experiments"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_math500_two_eval3")
    out = tmp_path / "outputs"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_text("fake weights; no model is loaded")
    selected = list(runner.SELECTIONS) if protocol == "both" else [protocol]
    for phase in selected:
        directory = out / phase / "results"
        directory.mkdir(parents=True)
        for budget in runner.BUDGETS:
            runner.result_paths(directory, phase, budget, "test_model")[-1].write_text("{}")
    monkeypatch.setattr(runner, "sha256", lambda path: runner.DATASET_SHA256)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: json.dumps(runner.PACKAGES))
    monkeypatch.setattr(runner, "audit_result", lambda args, phase, budget: {"fraction_solved": 0.5})
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)

    def refuse_gpu(*args, **kwargs):
        pytest.fail("A completed evaluation must not launch another GPU process")

    monkeypatch.setattr(runner.subprocess, "Popen", refuse_gpu)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_math500_two_eval3.py",
            "--protocol",
            protocol,
            "--gpu",
            "6",
            "--python",
            sys.executable,
            "--model-path",
            str(model),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--ipc-root",
            "/tmp/test_eval3_ipc",
            "--output-root",
            str(out),
            "--model-label",
            "test_model",
            "--checkpoint-repo",
            "test/repository",
            "--checkpoint-revision",
            "pinned",
        ],
    )
    # Keep the test's short IPC path virtual without creating anything outside tmp_path.
    original_mkdir = Path.mkdir

    def isolated_mkdir(path, *args, **kwargs):
        if str(path) != "/tmp/test_eval3_ipc":
            original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", isolated_mkdir)
    runner.main()
    control = out if protocol == "both" else out / protocol
    state = json.loads((control / "status.json").read_text())
    assert state["state"] == "complete"
    assert state["completed_points"] == state["total_points"] == 5 * len(selected)
    assert set(json.loads((control / "results.json").read_text())) == set(selected)
    if protocol != "both":
        assert not (out / "status.json").exists()
        other = (set(runner.SELECTIONS) - {protocol}).pop()
        assert not (out / other).exists()


def test_normalize_retains_zero_accuracy(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "qwen3_experiments"
    monkeypatch.syspath_prepend(str(scripts))
    runner = importlib.import_module("run_math500_cost_comparison")
    assert runner.normalize({"accuracy": 0.0})["accuracy"] == 0.0
    assert runner.normalize({"fraction_solved": 0.4})["accuracy"] == 0.4
