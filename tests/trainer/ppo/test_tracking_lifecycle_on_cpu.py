import gc
import sys
from types import SimpleNamespace

import pytest

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.tracking import Tracking


class _FakeTracking:
    def __init__(self):
        self.finish_calls = []

    def finish(self, exit_code=0):
        self.finish_calls.append(exit_code)


def test_tracking_finishes_wandb_exactly_once(monkeypatch):
    finish_calls = []
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: SimpleNamespace(id="test-run"),
        log=lambda **kwargs: None,
        finish=lambda **kwargs: finish_calls.append(kwargs["exit_code"]),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    tracking = Tracking("project", "experiment", default_backend="wandb")
    tracking.finish(exit_code=0)
    tracking.finish(exit_code=1)
    del tracking
    gc.collect()

    assert finish_calls == [0]


def test_ppo_trainer_finishes_tracking_after_success():
    trainer = object.__new__(RayPPOTrainer)
    tracking = _FakeTracking()

    def run_training_loop():
        trainer._tracking = tracking

    trainer._run_training_loop = run_training_loop
    trainer.fit()

    assert tracking.finish_calls == [0]
    assert trainer._tracking is None


def test_ppo_trainer_finishes_tracking_after_failure():
    trainer = object.__new__(RayPPOTrainer)
    tracking = _FakeTracking()

    def run_training_loop():
        trainer._tracking = tracking
        raise RuntimeError("training failed")

    trainer._run_training_loop = run_training_loop
    with pytest.raises(RuntimeError, match="training failed"):
        trainer.fit()

    assert tracking.finish_calls == [1]
    assert trainer._tracking is None
