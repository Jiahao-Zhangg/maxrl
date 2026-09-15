"""Check the full shared launcher and grader parity without GPUs or downloads."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import datasets
import pyarrow.parquet as pq
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE = REPO_ROOT / "qwen3_experiments/run_qwen3_1_7b_math12k.sh"
LAUNCHER = REPO_ROOT / "qwen3_experiments/run_qwen3_4b_base_polaris53k_maxrl.sh"


@pytest.fixture
def launch_config(tmp_path):
    data_root = tmp_path / "data"
    for relative in (
        "math12k/train.parquet", "polaris53k/train.parquet", "aime25/test.parquet", "math500/test.parquet",
    ):
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
        "if 'verl.trainer.main_ppo' in sys.argv:\n"
        "    Path(os.environ['TEST_COMMAND']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    python_stub.chmod(0o755)
    captured = tmp_path / "command.json"
    env = {key: value for key, value in os.environ.items() if not key.startswith("MAXRL_")}
    env.update({
        "PATH": str(bin_dir) + os.pathsep + env["PATH"],
        "MAXRL_SKIP_ENV_SETUP": "1",
        "MAXRL_DATA_DIR": str(data_root),
        "MAXRL_OUTPUT_DIR": str(tmp_path / "outputs"),
        "MAXRL_UPLOAD_CHECKPOINTS": "0",
        "TEST_COMMAND": str(captured),
    })

    def run(script=LAUNCHER, *overrides, **environment):
        subprocess.run(
            ["bash", str(script), *overrides], env=env | environment, cwd=tmp_path,
            check=True, capture_output=True, text=True, timeout=30,
        )
        command = json.loads(captured.read_text())
        overrides = command[command.index("verl.trainer.main_ppo") + 1:]
        with initialize_config_dir(config_dir=str(REPO_ROOT / "verl/trainer/config"), version_base=None):
            return compose(config_name="ppo_trainer", overrides=overrides)

    return run


def test_only_requested_training_defaults_differ_from_math12k(launch_config):
    original = launch_config(BASE)
    config = launch_config()
    assert config.actor_rollout_ref.model.path == "Qwen/Qwen3-4B-Base"
    assert config.data.train_files.endswith("polaris53k/train.parquet")
    assert config.algorithm.adv_estimator == "maxrl"
    assert config.actor_rollout_ref.actor.loss_agg_mode == "token-mean"
    assert config.trainer.total_epochs == 1
    assert config.trainer.total_training_steps is None
    assert config.trainer.save_freq == 60
    assert config.trainer.test_freq == 50
    assert config.trainer.val_before_train is True
    assert config.trainer.val_on_last_step is True
    assert config.data.train_batch_size == config.actor_rollout_ref.actor.ppo_mini_batch_size == 256
    assert config.actor_rollout_ref.rollout.n == 16
    assert config.data.max_response_length == 4096
    assert config.trainer.rollout_dataset.enabled is False
    assert config.trainer.experiment_name == "maxrl_Qwen3-4B-Base_polaris53k_1epoch"
    # Keep interpolation live: inactive critic/reward tokenizer paths also
    # follow the policy model, without changing any grader options.
    expected = OmegaConf.create(OmegaConf.to_container(original, resolve=False))
    expected["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen3-4B-Base"
    expected["data"]["train_files"] = config.data.train_files
    expected["trainer"].update({
        "total_epochs": 1, "save_freq": 60,
        "experiment_name": config.trainer.experiment_name,
        "default_local_dir": config.trainer.default_local_dir,
    })
    assert OmegaConf.to_container(config, resolve=True) == OmegaConf.to_container(expected, resolve=True)

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    RayPPOTrainer._validate_config(SimpleNamespace(config=config, use_reference_policy=False, use_critic=False))


def test_same_reward_manager_defaults_as_original_launcher(launch_config, monkeypatch):
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.workers.reward_manager import multi_thread_naive

    monkeypatch.setattr(multi_thread_naive, "RewardScoreActor", SimpleNamespace(remote=lambda: object()))
    tokenizer = SimpleNamespace(eos_token_id=151645)
    configs = [launch_config(BASE), launch_config()]
    assert OmegaConf.to_container(configs[0].reward_model, resolve=False) == OmegaConf.to_container(
        configs[1].reward_model, resolve=False
    )
    assert configs[0].custom_reward_function == configs[1].custom_reward_function
    for config in configs:
        manager = load_reward_manager(config, tokenizer, 0, **config.reward_model.get("reward_kwargs", {}))
        assert type(manager) is multi_thread_naive.MultiThreadNaiveRewardManager
        assert manager.num_reward_actors == 16
        assert manager._batch_size == 8
        assert manager._max_inflight_batches == 64
        assert manager._per_item_timeout_s == 1
        assert manager._per_batch_timeout_s == 10
        assert manager._timeout_score == 0
        assert manager._check_eos is False
        assert manager._zero_reward_on_max_response_length is False
        assert manager._post_think_pre_box_token_limit is None


def test_full_polaris_converter_preserves_math12k_prompt_and_gold_behavior(tmp_path, monkeypatch):
    from examples.maxrl_data_preprocess.math12k import make_map_fn

    examples = [
        {"problem": "What is 2 + 2?", "answer": "4", "difficulty": "0/8"},
        {"problem": "What is 3 + 4?", "answer": "7", "difficulty": "8/8"},
    ]
    source = datasets.Dataset.from_list(examples)
    calls = []

    def load_dataset(repo_id, *args, **kwargs):
        calls.append(repo_id)
        assert repo_id == "POLARIS-Project/Polaris-Dataset-53K"
        return datasets.DatasetDict(train=source)

    output = tmp_path / "polaris53k"
    output.mkdir()
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    monkeypatch.setattr(sys, "argv", ["polaris.py", "--local_dir", str(output)])
    runpy.run_path(str(REPO_ROOT / "examples/maxrl_data_preprocess/polaris.py"), run_name="__main__")
    rows = pq.read_table(output / "train.parquet").to_pylist()
    assert calls == ["POLARIS-Project/Polaris-Dataset-53K"]
    assert len(rows) == len(examples)
    for index, (row, example) in enumerate(zip(rows, examples, strict=True)):
        reference = make_map_fn("train")(example.copy(), index)
        assert row["prompt"] == reference["prompt"]
        assert row["reward_model"] == reference["reward_model"]
        assert row["data_source"] == "polaris"


def test_explicit_caller_overrides_remain_supported(launch_config):
    config = launch_config(LAUNCHER, "trainer.save_freq=120", "data.train_batch_size=128")
    assert config.trainer.save_freq == 120
    assert config.data.train_batch_size == 128
