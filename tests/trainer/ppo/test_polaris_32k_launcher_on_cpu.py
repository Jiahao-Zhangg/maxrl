"""Resolve the complete Polaris launcher command without GPU or dataset access."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "qwen3_experiments/run_qwen3_0_6b_polaris_4_8_maxrl_bs16_32k.sh"


@pytest.fixture
def launch_config(tmp_path):
    # Stub only the external Python processes; execute the actual shell chain.
    # Existing-data sentinels also prevent dataset preparation/downloads.
    data_root = tmp_path / "data"
    for relative in ("polaris_4_8/train.parquet", "aime25/test.parquet", "math500/test.parquet"):
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"prepared")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"
    python_stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if '-m' in sys.argv and 'verl.trainer.main_ppo' in sys.argv:\n"
        "    Path(os.environ['POLARIS_TEST_COMMAND']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    python_stub.chmod(0o755)
    captured_path = tmp_path / "command.json"
    env = {key: value for key, value in os.environ.items() if not key.startswith("MAXRL_")}
    env.update({
        "PATH": str(bin_dir) + os.pathsep + env["PATH"],
        "MAXRL_SKIP_ENV_SETUP": "1",
        "MAXRL_DATA_DIR": str(data_root),
        "MAXRL_OUTPUT_DIR": str(tmp_path / "outputs"),
        "POLARIS_TEST_COMMAND": str(captured_path),
    })

    def run(*overrides):
        subprocess.run(
            ["bash", str(LAUNCHER), *overrides], env=env, cwd=tmp_path,
            check=True, capture_output=True, text=True, timeout=30,
        )
        command = json.loads(captured_path.read_text())
        overrides = command[command.index("verl.trainer.main_ppo") + 1:]
        with initialize_config_dir(config_dir=str(REPO_ROOT / "verl/trainer/config"), version_base=None):
            return compose(config_name="ppo_trainer", overrides=overrides)

    return run


def test_polaris_32k_batch_and_context_resolve_through_shared_launcher(launch_config):
    config = launch_config()
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    assert config.algorithm.adv_estimator == "maxrl"
    assert config.actor_rollout_ref.model.path == "Qwen/Qwen3-0.6B"
    assert config.data.train_files.endswith("polaris_4_8/train.parquet")
    assert config.data.train_batch_size == actor.ppo_mini_batch_size == 16
    assert rollout.n == 16
    assert config.data.train_batch_size * rollout.n == 256
    assert config.data.max_response_length == rollout.response_length == 32768
    assert rollout.prompt_length == 1024
    assert rollout.max_model_len == rollout.prompt_length + rollout.response_length
    assert rollout.enable_chunked_prefill
    assert rollout.max_num_batched_tokens >= rollout.max_model_len
    assert actor.ppo_micro_batch_size_per_gpu == 1
    assert rollout.log_prob_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu == 1
    assert actor.ppo_max_token_len_per_gpu >= rollout.max_model_len
    assert rollout.log_prob_max_token_len_per_gpu >= rollout.max_model_len
    assert config.trainer.n_gpus_per_node == 4
    assert config.trainer.total_epochs == 1
    assert config.trainer.total_training_steps is None
    assert config.trainer.save_freq == 250
    assert "bs16_n16_32k_1epoch" in config.trainer.experiment_name
    assert config.trainer.experiment_name in config.trainer.default_local_dir

    # Run the trainer's actual startup checks, including batch divisibility.
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = SimpleNamespace(config=config, use_reference_policy=False, use_critic=False)
    RayPPOTrainer._validate_config(trainer)


def test_caller_overrides_still_take_precedence(launch_config):
    config = launch_config(
        "trainer.total_training_steps=1", "actor_rollout_ref.rollout.n=8",
        "trainer.total_epochs=2", "trainer.save_freq=100",
    )
    assert config.trainer.total_training_steps == 1
    assert config.trainer.total_epochs == 2
    assert config.trainer.save_freq == 100
    assert config.actor_rollout_ref.rollout.n == 8
    assert config.data.train_batch_size == 16
