"""Seed/sample-count isolation without changing the existing full RB defaults."""

import json

import pytest

from tailrl_experiments.run_text_maze_rb_full import (
    identity, is_complete, mark_complete, matches_identity, overrides, run_name,
)


@pytest.mark.parametrize("seed,n_rollouts", [(1, 16), (2, 16), (0, 32)])
def test_requested_variant_only_changes_seed_and_rollouts(tmp_path, seed, n_rollouts):
    baseline = dict(item.lstrip("+").split("=", 1) for item in overrides(tmp_path, tmp_path, tmp_path, checkpoint_step=2450))
    actual = dict(item.lstrip("+").split("=", 1) for item in overrides(
        tmp_path, tmp_path, tmp_path, checkpoint_step=2450, seed=seed, n_rollouts=n_rollouts,
    ))
    baseline["data.seed"] = str(seed)
    baseline["actor_rollout_ref.rollout.n"] = str(n_rollouts)
    assert actual == baseline
    assert actual["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == "1024"
    assert actual["actor_rollout_ref.actor.ppo_mini_batch_size"] == "256"
    assert actual["trainer.total_training_steps"] == "5001"
    assert actual["trainer.save_freq"] == "250"
    assert actual["trainer.test_freq"] == "1000"


def test_names_and_identities_are_distinct():
    arms = [(1, 16), (2, 16), (0, 32)]
    assert len({run_name(2450, seed, n) for seed, n in arms}) == 3
    assert run_name(2450) == "full_rb_ck2450_seed0_bs256_n16_4gpu"
    assert identity(2450, seed=2, n_rollouts=32)["n_rollouts"] == 32


def test_legacy_seed_zero_n16_records_still_work(tmp_path):
    old = {"checkpoint_step": 2450, "seed": 0, "world_size": 4, "smoke": False}
    assert matches_identity(old, identity(2450))
    assert not matches_identity(old, identity(2450, n_rollouts=32))
    (tmp_path / "COMPLETE.json").write_text(json.dumps({**old, "step": 5001}))
    assert is_complete(tmp_path, 2450)
    with pytest.raises(RuntimeError, match="does not match"):
        is_complete(tmp_path, 2450, n_rollouts=32)


def test_completion_cannot_cross_seed_or_sample_count(tmp_path):
    export = tmp_path / "checkpoints/global_step_5001/actor/huggingface/config.json"
    export.parent.mkdir(parents=True)
    export.write_text("{}")
    (tmp_path / "checkpoints/latest_checkpointed_iteration.txt").write_text("5001")
    mark_complete(tmp_path, 2450, seed=1, n_rollouts=16)
    assert is_complete(tmp_path, 2450, seed=1, n_rollouts=16)
    for seed, n_rollouts in ((2, 16), (1, 32), (0, 16)):
        with pytest.raises(RuntimeError, match="does not match"):
            is_complete(tmp_path, 2450, seed=seed, n_rollouts=n_rollouts)


@pytest.mark.parametrize("seed,n_rollouts", [(-1, 16), (2**32, 16), (0, 0), (0, 8)])
def test_reject_invalid_variants(seed, n_rollouts):
    with pytest.raises(ValueError):
        run_name(2450, seed, n_rollouts)
