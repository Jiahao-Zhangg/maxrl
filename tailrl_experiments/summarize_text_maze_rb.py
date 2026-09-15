"""Summarize observed goal/shortest-path events and cost from saved rollouts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def pass_at_k(n, successes, k):
    if k > n:
        raise ValueError("Cannot estimate pass@k when k exceeds the sampled rollout count")
    if n - successes < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - successes + 1, n + 1)))


def summarize_validation(path):
    contexts = defaultdict(list)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["score"] not in (0.0, 1.0):
                raise ValueError("Found nonbinary training reward")
            if row["trajectory_cost"] != max(row["generated_action_length"], row["shortest_distance"]):
                raise ValueError("Found invalid cost")
            contexts[row["input"]].append(row)
    rows = [row for values in contexts.values() for row in values]
    samples_per_context = {len(values) for values in contexts.values()}
    if len(samples_per_context) != 1:
        raise ValueError(f"Unequal evaluation counts: {samples_per_context}")
    n = samples_per_context.pop()
    result = {"samples": len(rows), "contexts": len(contexts), "samples_per_context": n}
    for label, key in (("goal", "goal_reached"), ("shortest", "is_shortest")):
        counts = np.array([sum(row[key] for row in values) for values in contexts.values()], dtype=int)
        result[f"{label}_successes"] = int(counts.sum())
        result[f"{label}_rate"] = float(counts.mean() / n)
        result[f"{label}_contexts_with_success"] = int(np.count_nonzero(counts))
        for k in (1, 4, 16, 64, 128):
            if k <= n:
                result[f"{label}_pass_at_{k}"] = float(np.mean([pass_at_k(n, c, k) for c in counts]))
        # Context bootstrap: preserve each maze's bundle of correlated outcomes.
        indices = np.random.default_rng(0).integers(len(counts), size=(2000, len(counts)))
        means = counts[indices].mean(axis=1) / n
        result[f"{label}_context_bootstrap_95ci"] = np.quantile(means, [0.025, 0.975]).tolist()
    result["mean_action_length"] = float(np.mean([r["generated_action_length"] for r in rows]))
    result["mean_cost"] = float(np.mean([r["trajectory_cost"] for r in rows]))
    result["cost_floor_fraction"] = float(np.mean([r["cost_floor_active"] for r in rows]))
    result["mean_context_success_per_cost"] = float(np.mean([
        sum(r["goal_reached"] for r in values) / sum(r["trajectory_cost"] for r in values)
        for values in contexts.values()
    ]))
    successful = [r for r in rows if r["goal_reached"]]
    result["success_mean_path_ratio"] = (
        float(np.mean([r["path_length"] / r["shortest_distance"] for r in successful]))
        if successful else None
    )
    result["valid_format_rate"] = float(np.mean([r["valid_format"] for r in rows]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"design": "1 estimator x 3 initializations x 1 seed; 100 updates; binary goal reward",
               "cost": "max(generated action count, BFS shortest distance)", "arms": {}}
    for ckpt in (2450, 3350, 3550):
        run_dir = args.state_dir / "runs" / f"pilot_rb_ck{ckpt}_seed0"
        metrics_path = run_dir / "metrics.jsonl"
        records = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
        training = [r for r in records if "rb/group_count" in r["metrics"]]
        arm = {"complete": (run_dir / "COMPLETE.json").exists(),
               "updates": max((r["step"] for r in training), default=0),
               "training_rollouts": sum(r["metrics"]["rb/rollouts"] for r in training),
               "training_goal_successes": sum(r["metrics"]["rb/successes"] for r in training),
               "training_shortest_successes": sum(r["metrics"].get("rb/shortest_successes", 0) for r in training),
               "training_shortest_positive_advantages": sum(r["metrics"].get("rb/shortest_positive_advantages", 0) for r in training),
               "nonzero_gradient_steps": sum(r["metrics"].get("actor/grad_norm", 0) > 0 for r in training),
               "validation": {},
               "learning_curve": [{"step": r["step"],
                                   "goal_rate": r["metrics"]["rb/successes"] / r["metrics"]["rb/rollouts"],
                                   "shortest_rate": r["metrics"].get("rb/shortest_successes", 0) / r["metrics"]["rb/rollouts"],
                                   "cost_mean": r["metrics"]["rb/cost_mean"],
                                   "zero_success_group_fraction": r["metrics"]["rb/zero_success_group_fraction"]}
                                  for r in training]}
        for file in sorted((run_dir / "validation").glob("*.jsonl"), key=lambda p: int(p.stem)):
            arm["validation"][file.stem] = summarize_validation(file)
        if training:
            if not all(np.isfinite(r["metrics"].get("actor/grad_norm", 0)) for r in training):
                raise ValueError(f"Nonfinite gradient in ckpt-{ckpt}")
            arm["mean_step_seconds"] = float(np.mean([r["metrics"]["timing_s/step"] for r in training]))
        summary["arms"][str(ckpt)] = arm
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Text-maze per-context RB pilot", "",
             "Binary goal reward; cost = max(generated action length, shortest-path distance).", "",
             "Seed 0; 100 updates; batch 64 prompts x 16 rollouts; 8,192 training mazes; "
             "256 held-out mazes x 128 evaluation rollouts. Shortest-path success uses "
             "TailRL's first-goal-visit definition.", "",
             "| Init | Updates | Goal rate: before → after | Shortest rate: before → after | Shortest events: before → after | Mean cost: before → after |",
             "|---|---:|---:|---:|---:|---:|"]
    for ckpt, arm in summary["arms"].items():
        first = arm["validation"].get("0")
        last = arm["validation"].get("100")
        if first and last:
            def pair(key, factor=1, digits=3, before=first, after=last):
                return f"{factor * before[key]:.{digits}f} → {factor * after[key]:.{digits}f}"
            lines.append(f"| {ckpt} | {arm['updates']} | {pair('goal_rate', 100)}% | "
                         f"{pair('shortest_rate', 100)}% | {pair('shortest_successes', digits=0)} | {pair('mean_cost', digits=2)} |")
        else:
            lines.append(f"| {ckpt} | {arm['updates']} | pending | pending | pending | pending |")
    lines.extend(["", "Each evaluation point contains 32,768 trajectories. This is one seed and a short, reduced-data pilot; "
                  "it does not establish parity with the paper's 5,001-step, full-data TailRL results. "
                  "No TailRL comparison arm was trained in this 1-estimator experiment. "
                  "Confidence intervals in summary.json resample contexts, not training seeds; "
                  "an all-zero observed event has an uninformative zero-width bootstrap interval.", ""])
    lines.extend(["## Training signal", "",
                  "| Init | Goal successes sampled | Shortest successes sampled | Shortest successes with positive advantage | Steps with nonzero gradient |",
                  "|---|---:|---:|---:|---:|"])
    for ckpt, arm in summary["arms"].items():
        lines.append(f"| {ckpt} | {arm['training_goal_successes']:.0f} | {arm['training_shortest_successes']:.0f} | "
                     f"{arm['training_shortest_positive_advantages']:.0f} | {arm['nonzero_gradient_steps']} / {arm['updates']} |")
    lines.extend(["", "All-failure contexts receive zero advantage. The inherited AdamW optimizer is still stepped "
                  "on every batch; momentum and weight decay can change parameters on zero-gradient batches. "
                  "Thus a zero advantage is not a guarantee that a sparse arm's policy stays frozen. "
                  "This mechanism is present in the training code; this pilot does not isolate its causal contribution.", ""])
    if all("100" in arm["validation"] for arm in summary["arms"].values()):
        lines.extend(["## Held-out shortest-path pass@k", "",
                      "| Init | Pass@1 before → after | Pass@16 before → after | Pass@128 before → after |",
                      "|---|---:|---:|---:|"])
        for ckpt, arm in summary["arms"].items():
            before, after = arm["validation"]["0"], arm["validation"]["100"]
            values = [f"{before[f'shortest_pass_at_{k}']*100:.3f}% → {after[f'shortest_pass_at_{k}']*100:.3f}%"
                      for k in (1, 16, 128)]
            lines.append(f"| {ckpt} | " + " | ".join(values) + " |")
        lines.extend(["", "Pass@k is averaged across held-out contexts using the finite-sample estimator "
                      "1 - C(n-c,k)/C(n,k), with n=128. No extrapolation beyond the sampled rollout count.", ""])
    (args.output_dir / "RESULTS.md").write_text("\n".join(lines))
    if all("100" in arm["validation"] for arm in summary["arms"].values()):
        make_plot(summary, args.output_dir)
    print("\n".join(lines))


def make_plot(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    keys = [("goal_rate", "Goal success (%)", 100), ("shortest_rate", "Shortest-path success (%)", 100),
            ("mean_cost", "Mean cost (moves)", 1)]
    for axis, (key, label, scale) in zip(axes, keys):
        for i, (ckpt, arm) in enumerate(summary["arms"].items()):
            before, after = [arm["validation"][s][key] * scale for s in ("0", "100")]
            axis.plot([i - .13, i + .13], [before, after], color="#b8c6cf", linewidth=2, zorder=1)
            axis.scatter(i - .13, before, color="#8296a5", s=55, label="Before RL" if i == 0 else None, zorder=2)
            axis.scatter(i + .13, after, color="#167d8d", marker="D", s=55, label="After 100 updates" if i == 0 else None, zorder=2)
            axis.annotate(f"{before:.3g}", (i - .13, before), xytext=(-4, 8), textcoords="offset points", ha="right", fontsize=9)
            axis.annotate(f"{after:.3g}", (i + .13, after), xytext=(4, 8), textcoords="offset points", ha="left", fontsize=9)
        axis.set_xticks(range(3), [f"ckpt-{x}" for x in summary["arms"]])
        axis.set_title(label)
        axis.set_xlabel("Initialization")
        axis.grid(axis="y", alpha=.18)
        axis.margins(x=.22, y=.2)
        if key.endswith("rate"):
            axis.set_ylim(bottom=0)
    axes[0].legend(loc="best", frameon=False, fontsize=9)
    figure.suptitle("Per-context RB · binary goal reward · cost max(L, L*)", fontsize=15)
    figure.savefig(output_dir / "pilot_results.png")
    figure.savefig(output_dir / "pilot_results.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
