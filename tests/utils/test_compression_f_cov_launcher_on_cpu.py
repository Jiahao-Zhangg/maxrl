"""Keep compression thinking settings fixed when switching to cross-context f_cov."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[2]
LAUNCHERS = {
    "f_cov": "run_qwen3_1_7b_compression_f_cov_l0_0.sh",
    "rb": "run_qwen3_1_7b_compression_per_context_rb_l0_0.sh",
    "maxrl": "run_qwen3_1_7b_compression_maxrl.sh",
}


@pytest.fixture
def launch(tmp_path):
    captured = tmp_path / "command.json"
    python = tmp_path / "capture_python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "assert 'verl.trainer.main_ppo' in args and '--cfg' in args\n"
        "payload = {'overrides': args[args.index('verl.trainer.main_ppo') + 1:args.index('--cfg')],\n"
        "           'seed': os.environ['SEED']}\n"
        "Path(os.environ['COMPRESSION_F_COV_CAPTURE']).write_text(json.dumps(payload))\n"
    )
    python.chmod(0o755)
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith(("L0_", "MAXRL_", "SLURM_")) and k not in ("DRY_RUN", "PREPARE_ONLY")}
    clean.update(PYTHON_BIN=str(python), COMPRESSION_F_COV_CAPTURE=str(captured), SLURM_JOB_ID="123")
    destinations = {
        "L0_RUN_DIR": str(tmp_path / "run with spaces"), "L0_DATA_DIR": str(tmp_path / "data"),
        "L0_CHECKPOINT_DIR": str(tmp_path / "checkpoints"), "L0_RAY_DIR": str(tmp_path / "ray"),
        "L0_ROLLOUT_DIR": str(tmp_path / "rollouts"), "L0_ROLLOUT_HF_REPO": "owner/test-rollouts",
    }

    def run(variant="f_cov", *overrides, default_destinations=False, **environment):
        captured.unlink(missing_ok=True)
        env = {**clean, **({} if default_destinations else destinations), "L0_CHECK_EOS": "false", **environment}
        subprocess.run(
            ["bash", str(REPO / "qwen3_experiments" / LAUNCHERS[variant]), *overrides, "--cfg", "job", "--resolve"],
            env=env, cwd=tmp_path, capture_output=True, text=True, check=True, timeout=10,
        )
        payload = json.loads(captured.read_text())
        with initialize_config_dir(config_dir=str(REPO / "verl/trainer/config"), version_base=None):
            config = compose(config_name="ppo_trainer", overrides=payload["overrides"])
        return config, payload

    return run


@pytest.mark.parametrize("baseline", ["rb", "maxrl"])
def test_only_advantage_and_experiment_label_change(launch, baseline):
    current, payload = launch()
    previous, old_payload = launch(baseline)
    configs = [OmegaConf.to_container(c, resolve=True) for c in (current, previous)]
    for config in configs:
        for key in ("adv_estimator", "f_cov_num_prompts"):
            config["algorithm"].pop(key)
        config["trainer"].pop("experiment_name")
    assert configs[0] == configs[1]
    assert payload["seed"] == old_payload["seed"] == "79"
    assert current.algorithm.adv_estimator == "f_cov" and current.algorithm.cost_offset_tokens == 0
    assert current.algorithm.f_cov_num_prompts == current.data.train_batch_size == 32
    assert current.actor_rollout_ref.rollout.n == 16
    assert current.actor_rollout_ref.model.path == "Qwen/Qwen3-1.7B"
    assert current.data.max_prompt_length == 1536 and current.data.max_response_length == 32768
    assert current.trainer.total_training_steps == 100 and current.trainer.save_freq == 10
    assert current.trainer.rollout_dataset.enabled and not current.trainer.rollout_dataset.private
    assert current.actor_rollout_ref.rollout.force_eos is False
    assert current.actor_rollout_ref.rollout.ignore_eos is False


def test_configured_grader_keeps_after_thinking_without_an_eos_gate(launch, monkeypatch):
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.workers.reward_manager import multi_thread_naive as reward

    config, _ = launch(L0_ADV_ESTIMATOR="maxrl", L0_COST_OFFSET_TOKENS="4096", L0_CHECK_EOS="true")
    assert config.algorithm.adv_estimator == "f_cov" and config.algorithm.cost_offset_tokens == 0
    monkeypatch.setattr(reward.RewardScoreActor, "remote", lambda: object())
    manager = load_reward_manager(
        config, tokenizer=SimpleNamespace(eos_token_id=1), num_examine=0,
        **config.reward_model.get("reward_kwargs", {}),
    )
    assert not manager._check_eos and manager._score_after_thinking
    assert not manager._zero_reward_on_max_response_length


def test_full_prompt_count_tracks_batch_override_and_rejects_stale_math12k_count(launch):
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    config, _ = launch("f_cov", "data.train_batch_size=64")
    assert config.algorithm.f_cov_num_prompts == 64
    trainer = SimpleNamespace(config=config, use_reference_policy=False, use_critic=False)
    RayPPOTrainer._validate_config(trainer)
    config.algorithm.f_cov_num_prompts = 256
    with pytest.raises(ValueError, match="full data.train_batch_size"):
        RayPPOTrainer._validate_config(trainer)


def test_default_outputs_and_public_rollout_repo_are_distinct(launch):
    current, _ = launch(default_destinations=True)
    previous, _ = launch("maxrl", default_destinations=True)
    assert "f_cov_l0_0_no_eos" in current.trainer.default_local_dir
    assert current.trainer.default_local_dir != previous.trainer.default_local_dir
    assert current.ray_init.ray_dir != previous.ray_init.ray_dir
    assert current.trainer.rollout_dataset.hub_repo_id == (
        "hi-todayis-jh/f-cov-l0-0-no-eos-qwen3-1.7b-compression-bs32-n16-32k-123-rollouts"
    )
    assert current.trainer.rollout_dataset.hub_repo_id != previous.trainer.rollout_dataset.hub_repo_id
    assert not current.trainer.rollout_dataset.private
