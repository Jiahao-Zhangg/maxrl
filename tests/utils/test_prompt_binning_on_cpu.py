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
