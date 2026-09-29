"""MaxRL formula and full-batch integration through the release V1 trainer."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorStack

from verl import DataProto
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import compute_maxrl_outcome_advantage
from verl.trainer.ppo.maxrl_algos import validate_maxrl_training_config
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories

VARIANTS = ["maxrl", "fixed_n_rb_offset_cost_aware_marginrl", "f_cov"]


@pytest.mark.parametrize("n", [2, 8, 16])
def test_maxrl_mean_normalization_preserves_original_formula(n):
    # Interleave all-fail, mixed, and all-success groups as V1 balancing does.
    rewards = torch.tensor([int(i < group % (n + 1)) for i in range(n) for group in range(n + 1)]).float()
    uids = np.tile(np.arange(n + 1), n)
    mask = torch.ones(len(rewards), 7)
    mask[::2, 3:] = 0
    token_rewards = torch.zeros_like(mask)
    token_rewards[:, 0] = rewards
    actual, returns = compute_maxrl_outcome_advantage(
        token_rewards.requires_grad_(), mask, uids, expected_group_size=n,
        norm_adv_by_std_in_grpo=True,
    )
    means = torch.tensor([group / n for group in uids], dtype=torch.float32)
    expected = (rewards - means) / (means + 1e-6)
    torch.testing.assert_close(actual, expected[:, None] * mask)
    torch.testing.assert_close(returns, actual)
    assert not actual.requires_grad


def make_data(padding=True):
    lengths = [3, 7, 4, 9] + ([1] if padding else [])
    rewards = [1, 0, 0, 0] + ([1] if padding else [])
    uids = ["a", "b", "a", "b"] + (["dummy"] if padding else [])
    mask = torch.arange(9)[None, :] < torch.tensor(lengths)[:, None]
    scores = torch.zeros_like(mask, dtype=torch.float32)
    scores[:, 0] = torch.tensor(rewards).float()
    return DataProto.from_dict(
        tensors={"response_mask": mask, "token_level_rewards": scores},
        non_tensors={"uid": np.array(uids)},
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_v1_ignores_padding_and_preserves_noncontiguous_prompt_groups(variant):
    config = AlgoConfig(adv_estimator=variant, cost_offset_tokens=0, f_cov_num_prompts=2)
    keys = ["a_0_0", "b_0_0", "a_1_0", "b_1_0"]
    expected = compute_advantage_for_multi_trajectories(make_data(False), keys, variant, num_repeat=2, config=config)
    padded = compute_advantage_for_multi_trajectories(
        make_data(), keys + ["dummy_0_0"], variant, num_repeat=2, config=config,
        padding_mask=[False] * 4 + [True],
    )
    torch.testing.assert_close(padded.batch["advantages"][:4], expected.batch["advantages"])
    assert torch.count_nonzero(padded.batch["advantages"][4]) == 0
    assert padded.meta_info == expected.meta_info
    if variant == "f_cov":
        # Full-batch H=(1/2 + 0/1)/2 and C=23/4. Computing separately
        # per prompt/rank would wrongly give the all-fail b group zero weight.
        assert padded.meta_info["f_cov_metrics"]["f_cov/H"] == 0.25
        assert padded.meta_info["f_cov_metrics"]["f_cov/global_cost_mean"] == 5.75
        assert padded.batch["advantages"][1, 0] < 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_v1_rejects_partial_groups_and_multiple_outputs_per_session(variant):
    data = make_data(False)
    with pytest.raises(ValueError, match="responses per prompt|rollout count mismatch"):
        compute_advantage_for_multi_trajectories(data, ["a_0_0", "b_0_0", "a_1_0", "b_1_0"], variant, num_repeat=16)
    with pytest.raises(ValueError, match="single output"):
        compute_advantage_for_multi_trajectories(data, ["a_0_0", "b_0_0", "a_0_1", "b_1_0"], variant, num_repeat=2)


@pytest.mark.parametrize("variant", VARIANTS)
def test_v1_transferqueue_writes_correct_nested_advantages_and_metrics(monkeypatch, variant):
    from verl.trainer.ppo.v1 import trainer_base

    padded = make_data(False)
    lengths = [3, 7, 4, 9]
    mask = torch.nested.nested_tensor([torch.ones(n, dtype=torch.int64) for n in lengths], layout=torch.jagged)
    scores = torch.nested.nested_tensor(
        [padded.batch["token_level_rewards"][i, :n] for i, n in enumerate(lengths)], layout=torch.jagged,
    )
    data = TensorDict({"uid": NonTensorStack.from_list(["a", "b", "a", "b"]),
                       "response_mask": mask, "rm_scores": scores}, batch_size=[4])
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", lambda **kw: data)
    saved = {}
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kw: saved.update(kw))
    config = OmegaConf.create({"algorithm": {
        "adv_estimator": variant, "gamma": 1, "lam": 1, "use_kl_in_reward": False,
        "cost_offset_tokens": 0, "f_cov_num_prompts": 2,
    }, "actor_rollout_ref": {"rollout": {"n": 2}}})
    batch = SimpleNamespace(keys=["a_0_0", "b_0_0", "a_1_0", "b_1_0"],
                            tags=[{} for _ in range(4)], partition_id="train")
    # KVBatchMeta's len is its key count; a small fake keeps this CPU-only.
    class Batch:
        keys, tags, partition_id = batch.keys, batch.tags, batch.partition_id

        def __len__(self):
            return len(self.keys)

    metrics = {}
    trainer_base.PPOTrainer._compute_advantage(SimpleNamespace(config=config), Batch(), metrics)
    expected = compute_advantage_for_multi_trajectories(
        padded, batch.keys, variant, num_repeat=2, config=config.algorithm,
    )
    for field in ("advantages", "returns"):
        for i, row in enumerate(saved["fields"][field].unbind()):
            torch.testing.assert_close(row, expected.batch[field][i, :lengths[i]])
    if variant == "f_cov":
        assert metrics["f_cov/H"] == 0.25
        assert metrics["f_cov/num_prompts"] == 2
    elif variant != "maxrl":
        assert metrics["fixed_n_rb_offset_marginrl/cost_mean"] == 5.75


def test_configuration_enforces_f_cov_full_batch_and_sync_mode():
    directory = Path(__file__).resolve().parents[3] / "verl/trainer/config"
    with initialize_config_dir(str(directory), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=[
            "algorithm.adv_estimator=f_cov", "data.train_batch_size=32", "actor_rollout_ref.rollout.n=16",
        ])
    validate_maxrl_training_config(config)
    assert config.algorithm.f_cov_num_prompts == 32
    config.algorithm.f_cov_num_prompts = 16
    with pytest.raises(ValueError, match="full data.train_batch_size"):
        validate_maxrl_training_config(config)
    config.algorithm.f_cov_num_prompts = 32
    config.trainer.v1.trainer_mode = "separate_async"
    with pytest.raises(ValueError, match="synchronous full-batch"):
        validate_maxrl_training_config(config)
