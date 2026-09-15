"""Check additive response costs, the fixed-N formula, and the Math12K launcher."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage,
    compute_fixed_n_rb_offset_cost_aware_marginrl_outcome_advantage,
    compute_fixed_n_rb_offset_marginrl_costs,
    get_adv_estimator_fn,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def make_batch(lengths, rewards):
    response_mask = torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    token_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
    token_rewards[:, 0] = torch.tensor(rewards, dtype=torch.float32)
    return response_mask, token_rewards


def test_additive_costs_preserve_length_differences_below_256():
    mask, _ = make_batch([1, 128, 255, 256, 1024], [0] * 5)
    lengths, costs = compute_fixed_n_rb_offset_marginrl_costs(mask.float().requires_grad_())

    torch.testing.assert_close(lengths, torch.tensor([1.0, 128.0, 255.0, 256.0, 1024.0]))
    torch.testing.assert_close(costs, torch.tensor([257.0, 384.0, 511.0, 512.0, 1280.0]))
    assert not lengths.requires_grad
    assert not costs.requires_grad


@pytest.mark.parametrize("rewards", [[1, 0, 1, 0] * 4, [1] + [0] * 15, [0] * 16, [1] * 16])
def test_n16_advantages_match_closed_form_for_all_reward_branches(rewards):
    lengths = [16, 32, 64, 128, 192, 255, 256, 257, 384, 512, 768, 1024, 1280, 1536, 1792, 2048]
    mask, token_rewards = make_batch(lengths, rewards)
    estimator = get_adv_estimator_fn("fixed_n_rb_offset_cost_aware_marginrl")
    advantages, returns, diagnostics = estimator(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=np.array(["prompt"] * 16),
        expected_group_size=16,
        return_diagnostics=True,
    )

    length_tensor = torch.tensor(lengths, dtype=torch.float32)
    cost_ratio = (length_tensor + 256) / (length_tensor.mean() + 256)
    successes = sum(rewards)
    expected = torch.zeros(16)
    if successes:
        expected = torch.where(
            torch.tensor(rewards, dtype=torch.bool),
            16.0 / successes - cost_ratio,
            -successes / (successes + 1.0) * cost_ratio,
        )
    torch.testing.assert_close(diagnostics["optimizer_trajectory_advantages"], expected)
    torch.testing.assert_close(diagnostics["raw_trajectory_advantages"], expected / 16)
    torch.testing.assert_close(advantages, expected.unsqueeze(-1) * mask)
    torch.testing.assert_close(returns, advantages)
    torch.testing.assert_close(diagnostics["trajectory_costs"], length_tensor + 256)
    torch.testing.assert_close(
        diagnostics["group_q_hats"],
        torch.tensor([successes / (sum(lengths) + 16 * 256)]),
    )
    assert torch.isfinite(advantages).all()
    assert not diagnostics["inverse_cost_cap_mask"].any()
    assert torch.count_nonzero(advantages.masked_select(~mask)) == 0


def test_zero_offset_recovers_raw_length_estimator():
    mask, token_rewards = make_batch([2, 3, 5], [1, 0, 1])
    kwargs = dict(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=np.array(["prompt"] * 3),
        expected_group_size=3,
    )
    expected, _ = compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage(**kwargs)
    actual, _ = compute_fixed_n_rb_offset_cost_aware_marginrl_outcome_advantage(
        **kwargs, cost_offset_tokens=0,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("offset", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_rejects_invalid_offset(offset):
    with pytest.raises(ValueError, match="cost_offset_tokens must be finite and nonnegative"):
        compute_fixed_n_rb_offset_marginrl_costs(torch.ones(2, 4), cost_offset_tokens=offset)


@pytest.mark.parametrize("mask", [torch.zeros(2, 4), torch.full((2, 4), float("nan"))])
def test_offset_does_not_hide_invalid_response_lengths(mask):
    with pytest.raises(ValueError, match="lengths must be finite and strictly positive"):
        compute_fixed_n_rb_offset_marginrl_costs(mask)


def test_requires_matching_cost_and_loss_mask_shapes():
    with pytest.raises(ValueError, match="must have matching shapes"):
        compute_fixed_n_rb_offset_cost_aware_marginrl_outcome_advantage(
            token_level_rewards=torch.zeros(2, 4),
            response_mask=torch.ones(2, 4),
            trajectory_cost_mask=torch.ones(2, 5),
            index=np.array(["prompt"] * 2),
        )


@pytest.mark.parametrize(
    "estimator",
    [AdvantageEstimator.FIXED_N_RB_OFFSET_COST_AWARE_MARGINRL, "fixed_n_rb_offset_cost_aware_marginrl"],
)
def test_dispatch_uses_all_group_response_lengths_with_separate_loss_mask(estimator):
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    mask, token_rewards = make_batch([64, 512, 128, 1024], [1, 1, 0, 1])
    loss_mask = torch.zeros_like(mask)
    loss_mask[:, :2] = True
    data = DataProto.from_dict(
        tensors={"token_level_rewards": token_rewards, "response_mask": mask, "loss_mask": loss_mask},
        non_tensors={"uid": np.array(["a", "b", "a", "b"])},
    )
    result = compute_advantage(
        data=data,
        adv_estimator=estimator,
        num_repeat=2,
        multi_turn=True,
        # Unrelated capped/fixed-q settings must not change the additive formula.
        config={"cost_offset_tokens": 512, "cost_reference_tokens": 1, "max_inverse_cost": 1, "fixed_q_hat": 2},
    )
    expected = torch.tensor([2 - 576 / 608, 1 - 1024 / 1280, -0.5 * 640 / 608, 1 - 1536 / 1280])
    torch.testing.assert_close(result.batch["advantages"], expected.unsqueeze(-1) * loss_mask)
    metrics = result.meta_info["fixed_n_rb_marginrl_metrics"]
    prefix = "fixed_n_rb_offset_marginrl"
    assert metrics[f"{prefix}/trajectory_tokens_mean"] == pytest.approx(432.0)
    assert metrics[f"{prefix}/cost_mean"] == pytest.approx(944.0)
    assert metrics[f"{prefix}/total_cost_mean"] == pytest.approx(1888.0)
    assert metrics[f"{prefix}/q_hat_mean"] == pytest.approx((1 / 1216 + 2 / 2560) / 2)
    assert metrics[f"{prefix}/cap_ratio"] == 0.0
    assert all(key.startswith(f"{prefix}/") for key in metrics)


@pytest.mark.parametrize("config_name", ["ppo_trainer", "ppo_megatron_trainer"])
def test_offset_is_available_in_both_trainer_configs(config_name):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "verl/trainer/config"), version_base=None):
        config = compose(config_name=config_name)
        assert config.algorithm.cost_offset_tokens == 256.0
        config = compose(config_name=config_name, overrides=["algorithm.cost_offset_tokens=512"])
        assert config.algorithm.cost_offset_tokens == 512.0


@pytest.fixture
def launch_math12k(tmp_path):
    # Execute the real shell chain with local data sentinels and a Python stub.
    # No environment installs, dataset downloads, GPU checks, or training occur.
    data_root = tmp_path / "data"
    for relative in ("math12k/train.parquet", "aime25/test.parquet", "math500/test.parquet"):
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"prepared")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    captured_path = tmp_path / "command.json"
    python_stub = bin_dir / "python"
    python_stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if '-m' in args and 'verl.trainer.main_ppo' in args:\n"
        "    payload = {'overrides': args[args.index('verl.trainer.main_ppo') + 1:],\n"
        "               'seed': os.environ['SEED'],\n"
        "               'requirements': os.environ['MAXRL_REQUIREMENTS_FILE']}\n"
        "    Path(os.environ['OFFSET_TEST_COMMAND']).write_text(json.dumps(payload))\n"
        "elif args != ['-']:\n"
        "    raise SystemExit(f'Unexpected Python call: {args}')\n"
    )
    python_stub.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("MAXRL_")}
    env.update({
        "PATH": str(bin_dir) + os.pathsep + env["PATH"],
        "OFFSET_TEST_COMMAND": str(captured_path),
        "MAXRL_SKIP_ENV_SETUP": "1",
        # Upload lifecycle is covered separately with a local Hub stand-in.
        "MAXRL_UPLOAD_CHECKPOINTS": "0",
        "MAXRL_DATA_DIR": str(data_root),
        "MAXRL_OUTPUT_DIR": str(tmp_path / "outputs"),
        "MAXRL_RAY_DIR": str(tmp_path / "ray"),
        # The hard-clip-256 baseline uses reference=2048 and cap=8.
        "MAXRL_COST_REFERENCE_TOKENS": "2048", "MAXRL_MAX_INVERSE_COST": "8.0",
    })

    def run(variant="offset", *overrides, **environment):
        name = "f_cov_offset" if variant == "f_cov" else f"fixed_n_rb_{variant}"
        launcher = REPO_ROOT / f"qwen3_experiments/run_qwen3_1_7b_math12k_{name}_marginrl.sh"
        captured_path.unlink(missing_ok=True)
        subprocess.run(
            ["bash", str(launcher), *overrides], env=env | environment, cwd=tmp_path,
            check=True, capture_output=True, text=True, timeout=10,
        )
        payload = json.loads(captured_path.read_text())
        with initialize_config_dir(config_dir=str(REPO_ROOT / "verl/trainer/config"), version_base=None):
            config = compose(config_name="ppo_trainer", overrides=payload["overrides"])
        return config, payload

    return run


def test_math12k_launcher_matches_hard_clip_256_with_150_step_limit(launch_math12k):
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    config, command = launch_math12k()
    capped_config, capped_command = launch_math12k("capped")
    assert capped_config.algorithm.cost_reference_tokens / capped_config.algorithm.max_inverse_cost == 256
    offset_settings = OmegaConf.to_container(config, resolve=True)
    capped_settings = OmegaConf.to_container(capped_config, resolve=True)
    for settings in (offset_settings, capped_settings):
        for key in ("adv_estimator", "cost_offset_tokens", "cost_reference_tokens", "max_inverse_cost"):
            settings["algorithm"].pop(key)
        for key in ("experiment_name", "default_local_dir", "total_training_steps", "rollout_dataset"):
            settings["trainer"].pop(key)
    assert offset_settings == capped_settings
    assert config.trainer.default_local_dir != capped_config.trainer.default_local_dir
    assert command["seed"] == capped_command["seed"] == "79"
    assert command["requirements"] == capped_command["requirements"]
    assert not any(arg.startswith(("algorithm.max_inverse_cost=", "algorithm.cost_reference_tokens=")) for arg in command["overrides"])
    assert config.algorithm.adv_estimator == "fixed_n_rb_offset_cost_aware_marginrl"
    assert config.algorithm.cost_offset_tokens == 256
    assert config.actor_rollout_ref.model.path == "Qwen/Qwen3-1.7B-Base"
    assert config.data.train_files.endswith("math12k/train.parquet")
    assert config.data.train_batch_size == config.actor_rollout_ref.actor.ppo_mini_batch_size == 256
    assert config.actor_rollout_ref.rollout.n == 16
    assert config.data.max_response_length == 4096
    assert config.data.max_prompt_length == 1024
    assert config.actor_rollout_ref.actor.loss_agg_mode == "token-mean"
    assert config.actor_rollout_ref.actor.optim.lr == 1e-6
    assert config.actor_rollout_ref.actor.grad_clip == 0.3
    assert config.trainer.total_epochs == 5
    assert config.trainer.total_training_steps == 150
    assert config.trainer.save_freq == config.trainer.test_freq == 50
    assert config.trainer.rollout_dataset.enabled
    assert not capped_config.trainer.rollout_dataset.enabled
    assert config.trainer.rollout_dataset.hub_repo_id == (
        "zjhhhh/fixed-n-rb-offset-cost-aware-marginrl-qwen3-1.7b-base-math12k-offset256-token-mean-rollouts"
    )
    assert "math12k_offset256_token_mean" in config.trainer.experiment_name
    assert config.reward_model.reward_manager == "multi_thread"
    assert config.reward_model.get("reward_kwargs", {}) == {}
    assert config.custom_reward_function.path is None
    assert not config.actor_rollout_ref.rollout.force_eos
    trainer = SimpleNamespace(config=config, use_reference_policy=False, use_critic=False)
    RayPPOTrainer._validate_config(trainer)


def test_math12k_grader_retains_original_timeouts_and_eos_behavior(launch_math12k, monkeypatch):
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.workers.reward_manager import multi_thread_naive as reward_module

    # Construct the production grader, replacing only remote actor allocation.
    monkeypatch.setattr(reward_module.RewardScoreActor, "remote", lambda: object())
    config, _ = launch_math12k()
    manager = load_reward_manager(config, tokenizer=SimpleNamespace(eos_token_id=1), num_examine=0)
    assert isinstance(manager, reward_module.MultiThreadNaiveRewardManager)
    assert manager.num_reward_actors == 16
    assert manager._per_item_timeout_s == 1
    assert manager._per_batch_timeout_s == 10.0
    assert not manager._check_eos
    assert not manager._zero_reward_on_max_response_length


def test_math12k_offset_environment_and_caller_overrides_are_forwarded(launch_math12k):
    config, _ = launch_math12k("offset", "trainer.total_training_steps=7", MAXRL_COST_OFFSET_TOKENS="512")
    assert config.algorithm.cost_offset_tokens == 512
    assert "math12k_offset512_token_mean" in config.trainer.experiment_name
    assert config.trainer.total_training_steps == 7
    assert config.actor_rollout_ref.model.path == "Qwen/Qwen3-1.7B-Base"
    assert config.reward_model.reward_manager == "multi_thread"


def test_math12k_rollout_upload_destination_and_disable_override(launch_math12k):
    config, _ = launch_math12k(MAXRL_ROLLOUT_DATASET_HF_REPO="owner/custom-rollouts")
    assert config.trainer.rollout_dataset.enabled
    assert config.trainer.rollout_dataset.hub_repo_id == "owner/custom-rollouts"
    config, _ = launch_math12k(MAXRL_SAVE_ROLLOUT_DATASET="0")
    assert not config.trainer.rollout_dataset.enabled


def test_f_cov_launcher_preserves_offset_training_setup_and_saves_rollouts(launch_math12k, monkeypatch):
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role

    config, _ = launch_math12k("f_cov")
    previous, _ = launch_math12k("offset")
    values = [OmegaConf.to_container(item, resolve=True) for item in (config, previous)]
    for settings in values:
        settings["algorithm"].pop("adv_estimator")
        for key in ("experiment_name", "default_local_dir", "rollout_dataset"):
            settings["trainer"].pop(key)
    assert values[0] == values[1]
    assert config.algorithm.adv_estimator == "f_cov"
    assert config.algorithm.f_cov_num_prompts == config.data.train_batch_size == 256
    assert config.algorithm.cost_offset_tokens == 256
    assert config.trainer.rollout_dataset.enabled
    assert "f-cov" in config.trainer.rollout_dataset.hub_repo_id
    assert config.trainer.total_training_steps == 150
    assert config.trainer.save_freq == config.trainer.test_freq == 50
    monkeypatch.setattr(RayPPOTrainer, "_create_dataloader", lambda *args: None)
    trainer = RayPPOTrainer(config, SimpleNamespace(), {Role.ActorRollout: object()}, SimpleNamespace())
    assert not trainer.use_critic
    config.algorithm.f_cov_num_prompts = 128
    with pytest.raises(ValueError, match="full data.train_batch_size"):
        trainer._validate_config()
