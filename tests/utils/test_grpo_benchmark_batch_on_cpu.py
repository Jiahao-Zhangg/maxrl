import pytest
from omegaconf import OmegaConf

from qwen3_experiments.grpo_release_benchmark import _first_loader_batch, replay_source_indices


@pytest.mark.parametrize("workers", [0, 4])
def test_replay_matches_actual_stateful_loader_with_shuffle(workers):
    config = OmegaConf.create(
        {"train_batch_size": 32, "dataloader_num_workers": workers, "shuffle": True, "seed": 42}
    )
    # These represent row IDs sampled from a larger dataset by another stack.
    expected = [(i * 97 + 11) % 3200 for i in range(32)]
    positions, _ = replay_source_indices(expected, config)
    actual = _first_loader_batch(positions, config).tolist()
    assert actual == expected
