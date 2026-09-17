#!/usr/bin/env python3
"""Produce mean@4 tables and question-count budget curves from audited points."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from math_eval_matrix_common import completed_point, make_tasks, read_json, task_points


def collect(root, manifest):
    rows = []
    for task in make_tasks(manifest["config"]):
        for point in task_points(manifest["config"], task):
            summary = completed_point(root, point, manifest, check_hashes=False)
            if summary is not None:
                rows.append({**point, **{key: summary[key] for key in (
                    "num_prompts", "num_questions_solved", "fraction_solved", "mean_at_4_accuracy",
                    "total_rollouts", "total_output_tokens", "mean_output_tokens", "elapsed_seconds")}})
    return rows


def aggregate(rows, config, protocol):
    output = []
    spec = config["evals"][protocol]
    for dataset in config["datasets"]:
        for model in config["models"]:
            for budget in spec["budgets"]:
                selected = [row for row in rows if row["protocol"] == protocol and row["dataset"] == dataset["key"]
                            and row["model"] == model["key"] and row["budget"] == budget]
                complete = {row["seed"] for row in selected} == set(spec["seeds"])
                values = [row["num_questions_solved"] for row in selected]
                output.append({"protocol": protocol, "dataset": dataset["key"], "model": model["key"], "budget": budget,
                               "completed_seeds": len(selected), "required_seeds": len(spec["seeds"]),
                               "complete": complete,
                               "mean_questions_solved": statistics.mean(values) if complete else None,
                               "std_questions_solved": (statistics.stdev(values) if len(values) > 1 else 0.0) if complete else None})
    return output


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def render_plot(path, rows, config, protocol):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    if not any(row["complete"] for row in rows):
        return
    colors = ["#d62728", "#ff7f0e", "#7a5195", "#2878b5", "#3a923a"]
    markers = ["o", "s", "D", "^", "*"]
    datasets = config["datasets"]
    columns = 2
    nrows = (len(datasets) + columns - 1) // columns
    with plt.rc_context({"font.size": 11, "axes.titlesize": 15, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180}):
        figure, axes = plt.subplots(nrows, columns, figsize=(12, 4.1 * nrows), squeeze=False)
        for index, dataset in enumerate(datasets):
            axis = axes.flat[index]
            for model_index, model in enumerate(config["models"]):
                selected = sorted([row for row in rows if row["complete"] and row["dataset"] == dataset["key"]
                                   and row["model"] == model["key"]], key=lambda row: row["budget"])
                if not selected:
                    continue
                x = [row["budget"] for row in selected]
                y = [row["mean_questions_solved"] for row in selected]
                spread = [row["std_questions_solved"] for row in selected]
                color, marker = colors[model_index % len(colors)], markers[model_index % len(markers)]
                axis.plot(x, y, color=color, marker=marker, linewidth=1.8, markersize=7, markeredgecolor="white", markeredgewidth=0.5)
                axis.fill_between(x, [v - s for v, s in zip(y, spread)], [v + s for v, s in zip(y, spread)], color=color, alpha=0.13)
            budgets = config["evals"][protocol]["budgets"]
            axis.set_xscale("log", base=2)
            axis.set_xticks(budgets, [f"{b // 1024}K" if b >= 1024 and b % 1024 == 0 else str(b) for b in budgets])
            axis.set_xlim(min(budgets) / 1.12, max(budgets) * 1.12)
            axis.set_ylim(0, dataset["expected_rows"])
            axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            axis.set_title(f"{dataset['label']} (n={dataset['expected_rows']})")
            axis.set_xlabel("Individual Budget" if protocol == "eval2" else "Averaged Shared Budget")
            axis.set_ylabel("Number of questions solved")
            axis.grid(True, linestyle="--", alpha=0.3)
        for axis in list(axes.flat)[len(datasets):]:
            axis.set_visible(False)
        handles = [Line2D([0], [0], color=colors[i % len(colors)], marker=markers[i % len(markers)],
                          linewidth=1.8, label=model["label"]) for i, model in enumerate(config["models"])]
        figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
        complete = sum(row["complete"] for row in rows)
        seed_count = len(config["evals"][protocol]["seeds"])
        figure.suptitle(f"Eval {protocol[-1]} · mean ± 1 SD across {seed_count} seeds", fontsize=16, fontweight="bold")
        figure.text(0.5, 0.005, f"{complete}/{len(rows)} model/dataset/budget points complete · output tokens, including failed attempts", ha="center", fontsize=9)
        figure.tight_layout(rect=(0, 0.105, 1, 0.95))
        for extension in ("png", "pdf", "svg"):
            destination = path.with_suffix("." + extension)
            temporary = destination.with_name(destination.stem + ".tmp." + extension)
            figure.savefig(temporary, bbox_inches="tight")
            temporary.replace(destination)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    manifest = read_json(root / "manifest.json")
    config = manifest["config"]
    report = root / "reports"
    report.mkdir(parents=True, exist_ok=True)
    rows = collect(root, manifest)
    write_csv(report / "all_points.csv", rows)
    eval1 = [row for row in rows if row["protocol"] == "eval1"]
    write_csv(report / "eval1_mean_at_4.csv", eval1)
    total = sum(len(list(task_points(config, task))) for task in make_tasks(config))
    lines = ["# Math evaluation matrix", "", f"Completed budget/seed points: {len(rows)}/{total}.", "",
             "All models use the same ordinary math prompt; L1 has no length instruction. "
             "Costs count generated output tokens, including failed attempts and EOS, not prompt tokens.", "",
             "## Eval 1: mean@4 accuracy", "", "The table is mean correctness over four independent responses per question, not pass@4.", ""]
    budgets = config["evals"]["eval1"]["budgets"]
    for dataset in config["datasets"]:
        lines += [f"### {dataset['label']} ({dataset['expected_rows']} questions)", "",
                  "| Model | " + " | ".join(map(str, budgets)) + " |", "|---|" + "---:|" * len(budgets)]
        for model in config["models"]:
            cells = []
            for budget in budgets:
                selected = [row for row in eval1 if row["dataset"] == dataset["key"] and row["model"] == model["key"] and row["budget"] == budget]
                cells.append(f"{statistics.mean(row['mean_at_4_accuracy'] for row in selected):.2%}" if selected else "pending")
            lines.append("| " + model["label"] + " | " + " | ".join(cells) + " |")
        lines.append("")
    for protocol in ("eval2", "eval3"):
        aggregated = aggregate(rows, config, protocol)
        write_csv(report / f"{protocol}_questions_solved.csv", aggregated)
        render_plot(report / f"{protocol}_questions_solved", aggregated, config, protocol)
        lines += [f"## Eval {protocol[-1]}", "", "Only points with every requested seed complete are plotted; bands show sample standard deviation, not confidence intervals.", ""]
        if protocol == "eval2" and config["evals"][protocol].get("stop_on_first_success", False):
            lines += ["Each question stops at its first correct response or budget exhaustion. "
                      "Unused tokens are not transferred; x is the allocated individual budget, not actual consumption.", ""]
        image = report / f"{protocol}_questions_solved.png"
        if image.exists():
            lines += [f"![{protocol}]({image.name})", ""]
    temporary = report / "results.md.tmp"
    temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    temporary.replace(report / "results.md")
    print(f"Updated reports: {len(rows)}/{total} points", flush=True)


if __name__ == "__main__":
    main()
