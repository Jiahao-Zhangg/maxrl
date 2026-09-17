#!/usr/bin/env python3
"""Plot Eval3 skip-solved results in the style of L1 Figure 2.

Edit the adjacent JSON for visual changes. Run with --audit once to check raw
rollouts; ordinary rerenders only read the small summary files. No GPU needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path(__file__).with_suffix(".json")
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FIELDS = (
    "dataset", "dataset_sha256", "num_prompts", "per_rollout_output_cap",
    "temperature", "top_p", "top_k", "seed", "grader", "grader_timeout_seconds",
    "max_prompt_len", "packages", "permutation_seed_scheme", "rollout_seed_scheme",
)
X_METRICS = {
    "actual_tokens_per_question": "Actual cumulative generated tokens / all 500 questions",
    "budget_per_prompt_reference": "Reference token budget per question b; total shared budget is 500 * b",
}
Y_METRICS = {
    "fraction_solved": "Questions solved at least once / all 500 questions, including unvisited questions",
    "num_prompts_solved": "Number of questions solved at least once, out of all 500 questions",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_run(summary_path: Path, summary: dict) -> tuple[dict, list]:
    """Check token accounting, seeded sweeps, skip-solved behavior and scores."""
    stem = summary_path.name.removesuffix("_summary.json")
    rollouts_path = summary_path.with_name(f"{stem}_rollouts.jsonl")
    prompts_path = summary_path.with_name(f"{stem}_prompts.jsonl")
    prompts = list(read_jsonl(prompts_path))
    num_prompts = summary["num_prompts"]
    prompt_map = {row["prompt_position"]: row for row in prompts}
    check(len(prompts) == num_prompts and set(prompt_map) == set(range(num_prompts)),
          f"Incomplete or duplicated prompt summaries: {prompts_path}")
    identities = [(prompt_map[i]["prompt_index"], prompt_map[i]["unique_id"]) for i in range(num_prompts)]
    check(len(set(identities)) == num_prompts, f"Duplicate question IDs: {prompts_path}")

    remaining = summary["total_global_output_budget"]
    solved = set()
    attempts = defaultdict(int)
    tokens = defaultdict(int)
    permutations = {}
    last_order = (-1, -1)
    count = 0
    for sequence, row in enumerate(read_jsonl(rollouts_path)):
        count += 1
        position = row["prompt_position"]
        check(position in prompt_map, f"Invalid question position: {rollouts_path}")
        check(identities[position] == (row["prompt_index"], row["unique_id"]), "Question ID mismatch")
        check(row["request_sequence_index"] == sequence, "Response sequence mismatch")
        check(position not in solved and not row["solved_before"], "An already-solved question was sampled")
        check(row["rollout_index"] == attempts[position], "Per-question attempt index mismatch")
        expected_seed = (summary["seed"] + 1_000_003 * position + attempts[position]) % (2**31 - 1)
        check(row["rollout_seed"] == expected_seed, "Response seed mismatch")

        round_index, rank = row["round_index"], row["permutation_rank"]
        check((round_index, rank) > last_order, "Sweep order is not strictly increasing")
        last_order = (round_index, rank)
        if round_index not in permutations:
            seed = (summary["seed"] + 2_000_033 * round_index) % (2**31 - 1)
            order = list(range(num_prompts))
            random.Random(seed).shuffle(order)
            digest = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode("ascii")).hexdigest()
            permutations[round_index] = (order, seed, digest)
        order, seed, digest = permutations[round_index]
        check(0 <= rank < num_prompts and order[rank] == position, "Shuffled sweep position mismatch")
        check(row["round_permutation_seed"] == seed, "Sweep seed mismatch")
        check(row["round_permutation_sha256"] == digest, "Sweep permutation hash mismatch")

        used, cap = row["output_tokens"], row["max_output_tokens"]
        check(0 < used <= cap <= summary["per_rollout_output_cap"], "Invalid response length or cap")
        check(row["global_budget_before"] == remaining, "Discontinuous token accounting")
        remaining -= used
        check(remaining >= 0 and row["global_budget_after"] == remaining, "Global token budget mismatch")
        check(row["score"] in (0, 1), "Non-binary response score")
        is_correct = row["score"] > 0
        check(row["newly_solved"] == is_correct and row["solved_after"] == is_correct, "Success flag mismatch")
        if is_correct:
            solved.add(position)
        attempts[position] += 1
        tokens[position] += used

    check(count == summary["total_rollouts"], "Response count differs from summary")
    check(len(solved) == summary["num_prompts_solved"], "Solved count differs from summary")
    check(sum(tokens.values()) == summary["global_output_budget_used"], "Used tokens differ from summary")
    check(remaining == summary["global_output_budget_remaining"], "Remaining budget differs from summary")
    check(remaining == 0 or len(solved) == num_prompts, "Evaluation ended before budget exhaustion or full success")
    for position, prompt in prompt_map.items():
        check(prompt["attempts"] == attempts[position], "Per-question attempt count differs")
        check(prompt["total_output_tokens"] == tokens[position], "Per-question token count differs")
        check(prompt["solved"] == (position in solved), "Per-question solved flag differs")
        check(prompt["sampled"] == (attempts[position] > 0), "Per-question sampled flag differs")
    return {
        "summary_file": str(summary_path),
        "rollouts_checked": count,
        "questions_checked": num_prompts,
        "used_tokens": sum(tokens.values()),
        "solved_questions": len(solved),
        "skip_solved_verified": True,
        "seeded_sweeps_verified": True,
    }, identities


def load_points(config: dict, repo_root: Path, audit: bool) -> tuple[list[dict], dict]:
    """Read actual costs, nominal budgets and solved counts as separate fields."""
    x_metric = config["axes"].get("x_metric", "actual_tokens_per_question")
    y_metric = config["axes"].get("y_metric", "fraction_solved")
    check(x_metric in X_METRICS and y_metric in Y_METRICS, "Unsupported axis metric")
    points, audits = [], []
    protocol = None
    reference_identities = None
    ids = [model["id"] for model in config["models"]]
    check(len(ids) == len(set(ids)), "Model IDs must be unique")
    for model in config["models"]:
        for budget in config["budgets"]:
            total_budget = config["num_prompts"] * budget
            stem = (
                f"{model['model_key']}_budget_per_prompt_{budget}"
                f"_global_budget_{total_budget}_rollout_cap_{config['rollout_cap']}"
            )
            summary_path = repo_root / model["result_dir"] / f"{stem}_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = {
                "evaluation": "eval4_cross_context_global_budget",
                "model_label": model["model_key"],
                "num_prompts": config["num_prompts"],
                "budget_per_prompt_reference": budget,
                "total_global_output_budget": total_budget,
                "per_rollout_output_cap": config["rollout_cap"],
                "permutation_schedule_is_model_independent": True,
            }
            for key, value in expected.items():
                check(summary.get(key) == value, f"Unexpected {key}: {summary_path}")
            # Older sweep summaries predate these two explicit fields. --audit
            # verifies the actual schedule for both old and new result formats.
            check(summary.get("question_selection", "sweep") == "sweep", "IID results are not Eval3 skip-solved")
            check(summary.get("success_aware_skipping", True) is True, "Skip-solved mode is disabled")
            current_protocol = {key: summary[key] for key in PROTOCOL_FIELDS}
            if protocol is None:
                protocol = current_protocol
            check(current_protocol == protocol, f"Evaluation protocol differs: {summary_path}")
            used = summary["global_output_budget_used"]
            solved = summary["num_prompts_solved"]
            check(0 < used <= total_budget, "Invalid used token count")
            check(total_budget - used == summary["global_output_budget_remaining"], "Summary budget mismatch")
            check(0 <= solved <= config["num_prompts"], "Invalid solved count")
            check(math.isclose(summary["fraction_solved"], solved / config["num_prompts"], abs_tol=1e-12),
                  "Solved fraction differs from solved count")
            if audit:
                result, identities = audit_run(summary_path, summary)
                result["summary_file"] = str(summary_path.relative_to(repo_root))
                if reference_identities is None:
                    reference_identities = identities
                check(identities == reference_identities, "Question IDs differ across models or budgets")
                audits.append(result)
            points.append({
                "model_id": model["id"],
                "model": model["label"],
                "model_key": model["model_key"],
                "training_step": model["training_step"],
                "budget_per_prompt_reference": budget,
                "actual_tokens_per_question": used / config["num_prompts"],
                "fraction_solved": solved / config["num_prompts"],
                "num_prompts_solved": solved,
                "num_prompts": config["num_prompts"],
                "global_output_budget_used": used,
                "total_rollouts": summary["total_rollouts"],
                "per_rollout_output_cap": summary["per_rollout_output_cap"],
                "summary_file": str(summary_path.relative_to(repo_root)),
                "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "checkpoint_repo": summary["checkpoint_repo"],
            })
    return points, {
        "protocol": protocol,
        "condition_count": len(points),
        "raw_audit_requested": audit,
        "raw_rollouts_checked": sum(row["rollouts_checked"] for row in audits),
        "conditions": audits,
        "x_metric": x_metric,
        "y_metric": y_metric,
        "x_definition": X_METRICS[x_metric],
        "y_definition": Y_METRICS[y_metric],
        "all_used_budgets_equal_reference": all(
            point["actual_tokens_per_question"] == point["budget_per_prompt_reference"] for point in points
        ),
    }


def draw_model(ax, x: np.ndarray, y: np.ndarray, model: dict, lines: dict, zorder: int) -> dict:
    """The paper style uses a log-linear fit, separate from measured markers."""
    slope, intercept = np.polyfit(np.log2(x), y, 1)
    predicted = slope * np.log2(x) + intercept
    residual = np.square(y - predicted).sum()
    total = np.square(y - y.mean()).sum()
    fit = {
        "model_id": model["id"], "model": model["label"],
        "slope_per_log2_token": float(slope), "intercept": float(intercept),
        "r_squared": float(1 - residual / total) if total else None,
        "line_mode": lines["mode"],
    }
    color = model["color"]
    mode = lines["mode"]
    if mode == "paper-fit":
        # Paper Figure 2: solid fit over the observed range, a dashed extension
        # to x_min / 1.5 and x_max * 1.5, and opaque measured points on top.
        factor = lines["extension_factor"]
        check(factor >= 1, "Fit extension_factor must be at least one")
        extended = np.geomspace(x.min() / factor, x.max() * factor, lines["fit_samples"])
        ax.plot(extended, slope * np.log2(extended) + intercept,
                color=color, linewidth=lines["width"], alpha=lines["extension_alpha"],
                linestyle=lines["extension_linestyle"], zorder=2)
        within = np.geomspace(x.min(), x.max(), lines["fit_samples"])
        ax.plot(within, slope * np.log2(within) + intercept,
                color=color, linewidth=lines["width"], alpha=lines["alpha"], zorder=2)
    elif mode == "connect":
        ax.plot(x, y, color=color, linewidth=lines["width"], alpha=lines["alpha"], zorder=2)
    else:
        check(mode == "none", f"Unknown line mode: {mode}")
    ax.scatter(x, y, s=model["marker_size"], marker=model["marker"], color=color,
               edgecolor=model["marker_edgecolor"], linewidth=model["marker_edgewidth"], zorder=zorder)
    return fit


def draw_detail(fig, points: list[dict], config: dict) -> None:
    """Show nearby results at one budget without displacing measured points."""
    from matplotlib.ticker import NullLocator, PercentFormatter, StrMethodFormatter

    detail = config.get("detail", {})
    if not detail.get("enabled", False):
        return
    axes, figure = config["axes"], config["figure"]
    x_metric = axes.get("x_metric", "actual_tokens_per_question")
    y_metric = axes.get("y_metric", "fraction_solved")
    budget_field = detail.get("budget_field", "budget_per_prompt_reference")
    ax = fig.add_axes(detail["axes_rect"], facecolor=figure["background"])
    if axes["xscale"] == "log":
        ax.set_xscale("log", base=axes["log_base"])
    ax.set_xlim(detail["xlim"])
    ax.set_ylim(detail["ylim"])
    ax.set_xticks(detail.get("xticks", [detail["budget"]]),
                  detail.get("xtick_labels", [f"{detail['budget'] / 1024:g}K"]))
    ax.set_yticks(detail["yticks"])
    ax.yaxis.set_major_formatter(
        PercentFormatter(1, decimals=0) if y_metric == "fraction_solved" else StrMethodFormatter("{x:.0f}")
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(length=0, labelsize=detail["tick_size"], pad=5)
    ax.set_title(detail["title"], fontsize=detail["title_size"], fontweight="bold", pad=8)
    if detail.get("xlabel"):
        ax.set_xlabel(detail["xlabel"], fontsize=detail["tick_size"], labelpad=6)
    ax.set_axisbelow(True)
    ax.grid(color=axes["grid_color"], alpha=axes["grid_alpha"],
            linewidth=axes["grid_width"], linestyle=axes["grid_linestyle"])
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(axes["spine_color"])
        spine.set_linewidth(axes["spine_width"])
    for model in config["models"]:
        rows = [point for point in points if point["model_id"] == model["id"]
                and point[budget_field] == detail["budget"]]
        check(len(rows) == 1, f"Detail budget missing or duplicated for {model['id']}")
        row = rows[0]
        x, y = row[x_metric], row[y_metric]
        check(detail["xlim"][0] <= x <= detail["xlim"][1], f"Detail x limits exclude {model['id']}")
        check(detail["ylim"][0] <= y <= detail["ylim"][1], f"Detail y limits exclude {model['id']}")
        ax.scatter(x, y, s=detail.get("marker_size", model["marker_size"]),
                   marker=model["marker"], color=model["color"], edgecolor=model["marker_edgecolor"],
                   linewidth=detail.get("marker_edgewidth", model["marker_edgewidth"]), zorder=5)
        left = model["id"] in detail["left_labels"]
        offset = detail["label_offset_points"] * (-1 if left else 1)
        label_offset = detail.get("label_offsets", {}).get(model["id"], [offset, 0])
        value = format(y, detail.get("value_format", ".1%" if y_metric == "fraction_solved" else ".0f"))
        ax.annotate(f"{model['label']} {value}", (x, y), xytext=label_offset, textcoords="offset points",
                    ha="right" if left else "left", va="center", fontsize=detail["label_size"],
                    color=figure["text_color"], zorder=6)


def render(points: list[dict], config: dict, output_dir: Path) -> list[dict]:
    """All appearance parameters live in JSON; data processing stays separate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator, PercentFormatter, StrMethodFormatter

    figure, axes, legend = config["figure"], config["axes"], config["legend"]
    x_metric = axes.get("x_metric", "actual_tokens_per_question")
    y_metric = axes.get("y_metric", "fraction_solved")
    rc = {
        "font.family": figure["font_family"], "font.size": figure["tick_size"],
        "text.color": figure["text_color"], "axes.labelcolor": figure["text_color"],
        "xtick.color": figure["text_color"], "ytick.color": figure["text_color"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=figure["size_inches"], facecolor=figure["background"])
        ax = fig.add_axes(figure["axes_rect"], facecolor=figure["background"])
        if axes["xscale"] == "log":
            ax.set_xscale("log", base=axes["log_base"])
        else:
            ax.set_xscale(axes["xscale"])
        ax.set_xlim(axes["xlim"])
        ax.set_ylim(axes["ylim"])
        ax.set_xticks(axes["xticks"], axes["xtick_labels"])
        ax.set_yticks(axes["yticks"])
        ax.yaxis.set_major_formatter(
            PercentFormatter(1, decimals=0) if y_metric == "fraction_solved" else StrMethodFormatter("{x:.0f}")
        )
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(length=0, labelsize=figure["tick_size"], pad=6)
        ax.set_axisbelow(True)
        ax.grid(color=axes["grid_color"], alpha=axes["grid_alpha"],
                linewidth=axes["grid_width"], linestyle=axes["grid_linestyle"])
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color(axes["spine_color"])
            spine.set_linewidth(axes["spine_width"])
        ax.set_title(figure["title"], fontsize=figure["title_size"], fontweight="bold", pad=figure["title_pad"])
        ax.set_xlabel(axes["xlabel"], fontsize=figure["label_size"], fontweight="bold", labelpad=axes["label_pad"])
        ax.set_ylabel(axes["ylabel"], fontsize=figure["label_size"], fontweight="bold", labelpad=axes["label_pad"])

        fits = []
        for index, model in enumerate(config["models"]):
            rows = sorted((row for row in points if row["model_id"] == model["id"]),
                          key=lambda row: row[x_metric])
            x = np.array([row[x_metric] for row in rows], dtype=float)
            y = np.array([row[y_metric] for row in rows], dtype=float)
            check(len(x) >= 2 and np.all(x > 0) and len(np.unique(x)) >= 2, "Need two distinct positive token counts")
            fit = draw_model(ax, x, y, model, config["lines"], zorder=5 + index)
            fit.update(x_metric=x_metric, y_metric=y_metric)
            fits.append(fit)

        handles, labels, bold_rows = [], [], []
        for group_index, group in enumerate(legend["groups"]):
            members = [model for model in config["models"] if model["group"] == group]
            if not members:
                continue
            if group_index:
                handles.append(Line2D([], [], linestyle="none"))
                labels.append("")
            bold_rows.append(len(labels))
            handles.append(Line2D([], [], linestyle="none"))
            labels.append(group)
            for model in members:
                handles.append(Line2D([], [], linestyle="none", marker=model["marker"],
                                      markersize=math.sqrt(model["marker_size"]),
                                      markerfacecolor=model["color"], markeredgecolor=model["marker_edgecolor"],
                                      markeredgewidth=model["marker_edgewidth"]))
                labels.append(model["label"])
        key = ax.legend(handles, labels, loc=legend["loc"], bbox_to_anchor=legend["bbox_to_anchor"],
                        fontsize=legend["font_size"], frameon=legend["frame"], framealpha=1,
                        facecolor=figure["background"], edgecolor=legend["frame_color"],
                        handlelength=legend["handle_length"], handletextpad=legend["handle_text_pad"],
                        labelspacing=legend["label_spacing"], borderpad=legend["border_pad"])
        key.get_frame().set_linewidth(legend["frame_width"])
        for index in bold_rows:
            key.get_texts()[index].set_fontweight("bold")

        draw_detail(fig, points, config)
        for extension in figure["formats"]:
            fig.savefig(output_dir / f"{config['output_stem']}.{extension}",
                        dpi=figure["dpi"], facecolor=figure["background"])
        plt.close(fig)
    return fits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Override the output directory in the JSON")
    parser.add_argument("--line-mode", choices=("paper-fit", "connect", "none"))
    parser.add_argument("--audit", action="store_true", help="Verify raw rollouts and solved-question skipping")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.line_mode:
        config["lines"]["mode"] = args.line_mode
    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or repo_root / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    points, validation = load_points(config, repo_root, args.audit)
    fits = render(points, config, output_dir)
    write_csv(output_dir / "points.csv", points)
    write_csv(output_dir / "fits.csv", fits)
    (output_dir / "config_used.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    # A later style-only rerender must not erase the completed raw-data audit.
    report_name = "audit.json" if args.audit else "summary_checks.json"
    (output_dir / report_name).write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir), "models": len(config["models"]), "points": len(points),
        "line_mode": config["lines"]["mode"], "raw_rollouts_checked": validation["raw_rollouts_checked"],
    }, indent=2))


if __name__ == "__main__":
    main()
