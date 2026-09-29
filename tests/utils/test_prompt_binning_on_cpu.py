import pytest

from verl.trainer.ppo.prompt_binning import accuracy_interval_fractions, compute_prompt_binning


def test_polaris_boundaries_and_zero_bins():
    bins = accuracy_interval_fractions([0, 1 / 1024, 1 / 8, 1 / 2, 7 / 8, 1])
    assert len(bins) == 13
    for label in ["[0.0, 0.0]", "(0.0, 0.0009765625]", "(0.0625, 0.125]",
                  "(0.25, 0.5]", "(0.5, 1.0)", "[1.0, 1.0]"]:
        assert bins[label] == 1 / 6
    assert sum(bins.values()) == pytest.approx(1)
    assert bins["(0.125, 0.25]"] == 0


def test_prompt_weighting_is_order_independent_and_per_source():
    # Unequal group sizes demonstrate that each prompt receives equal weight.
    metrics = compute_prompt_binning([1, 0, 1, 1], ["a", "b", "a", "a"], ["cf", "at", "cf", "cf"])
    assert metrics["train_all_datasets_binning/fraction_of_prompts_in_[1.0, 1.0]"] == 0.5
    assert metrics["train_binning_for_dataset_cf/fraction_of_prompts_in_[1.0, 1.0]"] == 1
    assert metrics["train_binning_for_dataset_at/fraction_of_prompts_in_[0.0, 0.0]"] == 1


def test_incomplete_or_non_binary_groups_are_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        compute_prompt_binning([1], ["a"], ["cf"], expected_group_size=8)
    with pytest.raises(ValueError, match="binary"):
        compute_prompt_binning([0.5], ["a"], ["cf"])
    with pytest.raises(ValueError, match="inconsistent"):
        compute_prompt_binning([1, 0], ["a", "a"], ["cf", "at"])


def test_trainer_binning_accepts_transfer_queue_linked_lists(monkeypatch):
    from types import SimpleNamespace

    import torch
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorStack

    from verl.trainer.ppo.v1 import trainer_base

    identities = TensorDict({
        "uid": NonTensorStack.from_list(["a", "a", "b", "b"]),
        "data_source": NonTensorStack.from_list(["cf", "cf", "at", "at"]),
    }, batch_size=[4])
    assert not hasattr(identities["uid"], "tolist")
    sequences = torch.nested.nested_tensor([torch.tensor([1, 2])] * 4, layout=torch.jagged)
    data = TensorDict({"prompts": sequences, "responses": sequences,
                       "rm_scores": torch.tensor([[0., 0.], [0., 0.], [0., 1.], [0., 1.]]),
                       "num_turns": torch.ones(4)}, batch_size=[4])
    monkeypatch.setattr(trainer_base, "get_metric_data_with_optional_routed_experts", lambda **kw: data)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", lambda **kw: identities)
    for name in ("compute_data_metrics", "compute_timing_metrics", "compute_throughout_metrics",
                 "compute_variance_proxy_metrics", "compute_moe_lb_metrics", "compute_spec_decode_metrics"):
        monkeypatch.setattr(trainer_base, name, lambda *a, **kw: {})
    trainer = SimpleNamespace(
        config=OmegaConf.create({"trainer": {"log_training_binning": True},
                                 "actor_rollout_ref": {"rollout": {"n": 2}, "model": {}}}),
        _rollout_moe_lb_metrics_accumulator=None, use_critic=False,
        _get_n_gpus_for_throughput=lambda: 1,
    )
    batch = SimpleNamespace(keys=["a_0", "a_1", "b_0", "b_1"], partition_id="train",
                            tags=[{"min_global_steps": 0, "max_global_steps": 0}] * 4)
    metrics = {}
    trainer_base.PPOTrainer._compute_metrics(trainer, batch, metrics, {}, global_steps=1, epoch=0)
    assert metrics["train_all_datasets_binning/fraction_of_prompts_in_[0.0, 0.0]"] == 0.5
    assert metrics["train_all_datasets_binning/fraction_of_prompts_in_[1.0, 1.0]"] == 0.5
    assert metrics["train_binning_for_dataset_at/fraction_of_prompts_in_[1.0, 1.0]"] == 1


def test_rollout_logging_accepts_transfer_queue_non_tensor_fields(monkeypatch):
    from types import SimpleNamespace

    import torch
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorStack

    from verl.trainer.ppo.v1 import trainer_base

    sequences = torch.nested.nested_tensor([torch.tensor([1, 2])] * 2, layout=torch.jagged)
    data = TensorDict({
        "prompts": sequences, "responses": sequences, "rm_scores": torch.tensor([[0., 0.], [0., 1.]]),
        "uid": NonTensorStack.from_list(["a", "a"]),
        "reward_model": NonTensorStack.from_list([{"ground_truth": "tests"}] * 2),
    }, batch_size=[2])
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", lambda **kw: data)
    dumped = {}
    trainer = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda ids, **kw: "text"),
        _dump_generations=lambda **kw: dumped.update(kw),
    )
    batch = SimpleNamespace(keys=["a_0_0", "a_1_0"], partition_id="train")
    trainer_base.PPOTrainer._log_rollout_data(trainer, batch, {}, "/unused")
    assert dumped["scores"] == [0, 1]
    assert dumped["gts"] == ["tests", "tests"]
