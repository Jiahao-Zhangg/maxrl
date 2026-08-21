#!/usr/bin/env python3
"""Plot MATH-500 mean@4 accuracy against maximum output length."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


BUDGETS = (256, 512, 1024, 2048, 4096)
MODELS = (
    ("maxrl", "MaxRL", "#3366cc"),
    ("cost_aware", "MaxRL gradient divided by cost", "#d65f32"),
    ("rb_cost_aware", "MaxRL objective divided by cost", "#2a9d65"),
    ("fixed_n_rb_step50", "Fixed-N RB MarginRL — step 50", "#7f7f7f"),
    ("fixed_n_rb_step100", "Fixed-N RB MarginRL — step 100", "#9467bd"),
    (
        "fixed_n_rb_capped_step100",
        "Fixed-N RB capped cost — step 100",
        "#e377c2",
    ),
    (
        "fixed_n_rb_capped_step150",
        "Cost-Aware Marginal RL",
        "#8c564b",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[model_key for model_key, _, _ in MODELS],
        help="Model keys to plot; defaults to every configured model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_keys = set(args.models) if args.models else None
    selected_models = tuple(
        model for model in MODELS if selected_keys is None or model[0] in selected_keys
    )
    rows = []
    for model_key, model_title, _ in selected_models:
        for budget in BUDGETS:
            path = args.results_dir / f"{model_key}_max_tokens_{budget}_summary.json"
            with path.open(encoding="utf-8") as stream:
                summary = json.load(stream)
            rows.append(
                {
                    "model": model_key,
                    "model_title": model_title,
                    "max_output_len": budget,
                    "mean_at_4_accuracy": summary["mean_at_4_accuracy"],
                    "correct_responses": summary["correct_responses"],
                    "num_scored_responses": summary["num_scored_responses"],
                    "mean_output_tokens": summary["mean_output_tokens"],
                    "budget_exhaustion_rate": summary["budget_exhaustion_rate"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(11.5, 7.0))
    accuracies_by_model = {
        model_key: [
            row["mean_at_4_accuracy"]
            for row in rows
            if row["model"] == model_key
        ]
        for model_key, _, _ in MODELS
        if selected_keys is None or model_key in selected_keys
    }
    annotation_targets = []
    for index in range(len(BUDGETS)):
        ranked_models = sorted(
            accuracies_by_model,
            key=lambda key: accuracies_by_model[key][index],
        )
        targets = {}
        previous_target = None
        for model_key in ranked_models:
            target = accuracies_by_model[model_key][index]
            if previous_target is not None:
                target = max(target, previous_target + 0.022)
            targets[model_key] = target
            previous_target = target
        if previous_target is not None and previous_target > 0.735:
            shift = previous_target - 0.735
            targets = {key: value - shift for key, value in targets.items()}
        annotation_targets.append(targets)
    markers = {
        "maxrl": "o",
        "cost_aware": "s",
        "rb_cost_aware": "^",
        "fixed_n_rb_step50": "v",
        "fixed_n_rb_step100": "D",
        "fixed_n_rb_capped_step100": "P",
        "fixed_n_rb_capped_step150": "X",
    }
    for model_key, title, color in selected_models:
        model_rows = [row for row in rows if row["model"] == model_key]
        accuracies = [row["mean_at_4_accuracy"] for row in model_rows]
        axis.plot(
            BUDGETS,
            accuracies,
            marker=markers[model_key],
            markersize=7,
            linewidth=2.2,
            color=color,
            label=title,
        )
        for index, (budget, accuracy) in enumerate(
            zip(BUDGETS, accuracies, strict=True)
        ):
            direction = -1 if budget == BUDGETS[-1] else 1
            horizontal_alignment = "right" if direction < 0 else "left"
            if len(selected_models) > 3:
                annotation_offset = (
                    budget * (0.97 if direction < 0 else 1.03),
                    annotation_targets[index][model_key],
                )
                text_coordinates = "data"
            else:
                ranked_models = sorted(
                    accuracies_by_model,
                    key=lambda key: accuracies_by_model[key][index],
                )
                rank = ranked_models.index(model_key)
                spacing = 24 if len(selected_models) == 2 else 20
                vertical_offset = (
                    rank - (len(selected_models) - 1) / 2
                ) * spacing
                annotation_offset = (8 * direction, vertical_offset)
                text_coordinates = "offset points"
            axis.annotate(
                f"{accuracy:.1%}",
                (budget, accuracy),
                xytext=annotation_offset,
                textcoords=text_coordinates,
                ha=horizontal_alignment,
                va="center",
                fontsize=8.5,
                color=color,
            )

    axis.set_xscale("log", base=2)
    axis.set_xticks(BUDGETS, labels=[str(budget) for budget in BUDGETS])
    axis.set_ylim(0, 0.75)
    axis.grid(True, alpha=0.25)
    axis.set_xlabel("Maximum output tokens")
    axis.set_ylabel("MATH-500 mean@4 accuracy")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(loc="lower right", fontsize=8.5)
    axis.set_title("MATH-500 accuracy by generation budget")
    figure.tight_layout()
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {args.output}, {args.output.with_suffix('.pdf')}, and {csv_path}")


if __name__ == "__main__":
    main()
