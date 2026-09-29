"""Prompt-weighted binary accuracy bins compatible with the Polaris GRPO run."""

from collections import defaultdict


def accuracy_interval_fractions(accuracies):
    boundaries = [0.0, *(2.0 ** -power for power in range(10, -1, -1))]
    labels = ["[0.0, 0.0]"]
    labels += [f"({low}, {high}{')' if high == 1.0 else ']'}"
               for low, high in zip(boundaries, boundaries[1:], strict=False)]
    labels += ["[1.0, 1.0]"]
    counts = dict.fromkeys(labels, 0)
    for accuracy in accuracies:
        if not 0 <= accuracy <= 1:
            raise ValueError("Accuracy must be finite and between zero and one")
        if accuracy == 0:
            label = labels[0]
        elif accuracy == 1:
            label = labels[-1]
        else:
            index = next(i for i, high in enumerate(boundaries[1:]) if accuracy <= high)
            label = labels[index + 1]
        counts[label] += 1
    return {label: count / len(accuracies) if accuracies else 0.0 for label, count in counts.items()}


def compute_prompt_binning(scores, prompt_ids, data_sources, *, expected_group_size=None):
    """Aggregate by prompt identity, independently of batch reordering."""
    if not len(scores) == len(prompt_ids) == len(data_sources):
        raise ValueError("Binning fields have different lengths")
    groups = defaultdict(list)
    sources = {}
    for score, uid, source in zip(scores, prompt_ids, data_sources, strict=True):
        if score not in (0, 1):
            raise ValueError("Training accuracy binning requires binary rewards")
        if uid in sources and sources[uid] != source:
            raise ValueError("A prompt has inconsistent data sources")
        sources[uid] = source
        groups[uid].append(float(score))
    by_source = defaultdict(list)
    accuracies = []
    for uid, values in groups.items():
        if expected_group_size is not None and len(values) != expected_group_size:
            raise ValueError(f"Incomplete rollout group: {len(values)} != {expected_group_size}")
        accuracy = sum(values) / len(values)
        accuracies.append(accuracy)
        by_source[sources[uid]].append(accuracy)
    metrics = {
        f"train_all_datasets_binning/fraction_of_prompts_in_{label}": fraction
        for label, fraction in accuracy_interval_fractions(accuracies).items()
    }
    for source, values in by_source.items():
        metrics.update({
            f"train_binning_for_dataset_{source}/fraction_of_prompts_in_{label}": fraction
            for label, fraction in accuracy_interval_fractions(values).items()
        })
    return metrics
