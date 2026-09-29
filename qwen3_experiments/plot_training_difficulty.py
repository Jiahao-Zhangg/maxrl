"""Plot the audited training-set evaluations without changing their results."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


MODELS = (("qwen3", "Qwen3-1.7B"), ("deepseek_r1", "DeepSeek-R1-Distill-Qwen-1.5B"))
DATASETS = (("polaris_train", "Polaris-1-8-3200", "#0072B2"),
            ("compression_train", "compression_dataset", "#D55E00"))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_verified_data(root):
    groups, sources, paired_questions = {}, {}, {}
    aggregate = read(root / "report/audit.json")
    assert aggregate["complete"] and aggregate["responses_verified"] == 3200
    assert digest(root / "report/metrics.json") == aggregate["metrics_sha256"]
    for model, _ in MODELS:
        folder = root / "runs" / model / "report"
        audit = read(folder / "audit.json")
        assert audit["complete"] and not audit["grader_errors"]
        assert audit["responses_verified"] == 1600 and audit["questions"] == 400
        for name in ("metrics", "per_sample"):
            path = folder / f"{name}.json"
            assert digest(path) == audit[f"{name}_sha256"]
            sources[str(path.relative_to(root))] = digest(path)
        rows = read(folder / "per_sample.json")
        assert len(rows) == len({row["sample_id"] for row in rows}) == 1600
        metrics = {row["dataset"]: row for row in read(folder / "metrics.json")}
        with (folder / "per_question.csv").open() as stream:
            questions = {(row["dataset"], row["question_id"]): row for row in csv.DictReader(stream)}
        assert len(questions) == 400
        for dataset, _, _ in DATASETS:
            selected = [row for row in rows if row["dataset"] == dataset]
            per_question = defaultdict(list)
            for row in selected:
                assert row["model"] == model and row["cap_tokens"] == 32768
                assert row["correct"] in (0, 1)
                assert 0 < row["original_output_tokens"] == row["used_output_tokens"] <= 32768
                per_question[row["question_id"]].append(row)
            assert len(selected) == 800 and len(per_question) == 200
            counts = Counter()
            for qid, responses in per_question.items():
                assert len(responses) == 4 and {row["sample_index"] for row in responses} == set(range(4))
                correct = int(sum(row["correct"] for row in responses))
                assert float(questions[(dataset, qid)]["mean_at_4"]) == correct / 4
                counts[correct] += 1
            metric = metrics[dataset]
            assert sum(counts.values()) == 200
            assert sum(k * n for k, n in counts.items()) == metric["correct_responses"]
            assert np.isclose(100 * sum(row["correct"] for row in selected) / 800,
                              metric["mean_at_4_percent"])
            lengths = np.array([row["original_output_tokens"] for row in selected], dtype=int)
            assert np.isclose(lengths.mean(), metric["mean_output_tokens"])
            groups[(model, dataset)] = {"counts": [counts[i] for i in range(5)], "lengths": lengths,
                                        "metrics": metric}
            if dataset in paired_questions:
                assert set(per_question) == paired_questions[dataset]
            else:
                paired_questions[dataset] = set(per_question)
    return groups, sources


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#BCC5CF")
    ax.tick_params(axis="both", length=0, pad=8, labelcolor="#344054")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7EBF0", linewidth=0.8)


def figure_frame(title, subtitle):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.0), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.715, wspace=0.14)
    fig.text(0.075, 0.955, title, fontsize=20, fontweight="semibold", color="#172B40")
    fig.text(0.075, 0.91, subtitle, fontsize=11.5, color="#526174")
    fig.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in DATASETS],
               loc="upper left", bbox_to_anchor=(0.066, 0.88), ncol=2, frameon=False,
               handlelength=1.45, columnspacing=2.5, fontsize=12)
    for ax in axes:
        style_axes(ax)
    return fig, axes


def correctness_figure(groups):
    fig, axes = figure_frame("Distribution of per-question accuracy",
                             "200 sampled questions per dataset; 4 responses per question; identical questions across models")
    locations = np.arange(5)
    for ax, (model, label) in zip(axes, MODELS):
        ax.set_title(label, fontsize=13, fontweight="semibold", pad=24, loc="left")
        for index, (dataset, _, color) in enumerate(DATASETS):
            heights = np.array(groups[(model, dataset)]["counts"]) / 2
            bars = ax.bar(locations + (index - 0.5) * 0.36, heights, width=0.34, color=color,
                          edgecolor="white", linewidth=0.6)
            ax.bar_label(bars, labels=[f"{value:g}%" for value in heights], padding=5,
                         fontsize=10, color="#25364A")
        ax.set_xticks(locations, ["0%\n0/4 correct", "25%\n1/4 correct", "50%\n2/4 correct",
                                 "75%\n3/4 correct", "100%\n4/4 correct"])
        ax.tick_params(axis="x", labelsize=10)
        ax.set_ylim(0, 75)
        ax.set_yticks(np.arange(0, 71, 10))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlabel("Per-question accuracy", labelpad=12)
        ax.set_xlim(-0.62, 4.62)
    axes[0].set_ylabel("Share of questions", labelpad=12)
    fig.text(0.075, 0.07, "Each bar counts questions, not individual responses. 0/4 means all four responses were incorrect.",
             fontsize=10, color="#526174")
    fig.text(0.075, 0.032, "Only text after completed </think> is graded. Thinking enabled; output limit: 32,768 tokens.",
             fontsize=10, color="#526174")
    return fig


def length_figure(groups):
    fig, axes = figure_frame("Distribution of response length",
                             "Empirical cumulative distributions; 800 responses per dataset and model; all responses included")
    for ax, (model, label) in zip(axes, MODELS):
        ax.set_title(label, fontsize=13, fontweight="semibold", pad=24, loc="left")
        for index, (dataset, _, color) in enumerate(DATASETS):
            values = np.sort(groups[(model, dataset)]["lengths"])
            x = np.r_[0, values, 32768]
            y = np.r_[0, np.arange(1, len(values) + 1) / len(values) * 100, 100]
            ax.step(x, y, where="post", color=color, linewidth=2.3,
                    linestyle="-" if index == 0 else "--")
            median = np.median(values)
            ax.scatter([median], [50], color=color, s=34, zorder=4, edgecolors="white", linewidth=0.7)
            ax.text(0.97, 0.19 - index * 0.073,
                    f"{('Polaris' if index == 0 else 'Compression')} median: {median:,.0f}",
                    transform=ax.transAxes, ha="right", color=color, fontsize=10.5)
        ax.set_xlim(0, 33500)
        ax.set_ylim(0, 104)
        ax.set_xticks([0, 8192, 16384, 24576, 32768], ["0", "8k", "16k", "24k", "32k"])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlabel("Generated response length (tokens)", labelpad=12)
        ax.axvline(32768, color="#AAB4C0", linewidth=1, linestyle=":", zorder=0)
    axes[0].set_ylabel("Share of responses at or below length", labelpad=12)
    fig.text(0.075, 0.07, "Curves farther right indicate longer responses. Tokens include thinking, final answer, and EOS; prompt tokens excluded.",
             fontsize=10, color="#526174")
    fig.text(0.075, 0.032, "1k = 1,024 tokens. The jump at 32k includes responses reaching the output limit; these are retained.",
             fontsize=10, color="#526174")
    return fig


def make_report(root, destination, groups, sources):
    distribution, lengths = [], []
    for model, label in MODELS:
        for dataset, dataset_label, _ in DATASETS:
            group = groups[(model, dataset)]
            for k, count in enumerate(group["counts"]):
                distribution.append({"model": label, "dataset": dataset_label, "correct_of_4": k,
                                     "question_count": count, "question_percent": count / 2,
                                     "sampled_questions": 200})
            values = group["lengths"]
            lengths.append({"model": label, "dataset": dataset_label, "responses": 800,
                            "mean_tokens": float(values.mean()), "median_tokens": float(np.median(values)),
                            "p10_tokens": float(np.quantile(values, 0.1)),
                            "p90_tokens": float(np.quantile(values, 0.9)),
                            "at_32768_tokens": int(np.sum(values == 32768)),
                            "at_32768_percent": float(np.mean(values == 32768) * 100)})
    write_csv(destination / "question_correctness_distribution.csv", distribution)
    write_csv(destination / "response_length_summary.csv", lengths)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "source_root": str(root),
               "source_sha256": sources, "source_audits_passed": True,
               "per_sample_and_per_question_scores_agree": True, "same_questions_for_both_models": True,
               "question_distributions": distribution, "response_lengths": lengths,
               "response_tokens_include": ["thinking", "final_answer", "EOS"],
               "response_tokens_exclude": ["prompt"], "length_cap_retained": True,
               "per_question_probability_is_estimated_from_four_samples": True}
    (destination / "distributions_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# 训练集难度与回答长度分布", "",
             "两套训练集各固定抽取 200 题，两个模型使用相同题目，每题生成 4 个回答。图表来自已完成并核验的评测，没有重新采样或生成。", "",
             "## 逐题正确率分布", "",
             "横轴为一题的四个回答中正确的比例，只有 0%、25%、50%、75%、100% 五档；纵轴为 200 题中落入该档的题目占比。"
             "这是四次采样的观察结果，不表示已精确测得每道题的真实成功概率。", "",
             "![逐题正确率分布](question_correctness_distribution.png)", "",
             "| 模型 | 数据集 | 0/4 | 1/4 | 2/4 | 3/4 | 4/4 |", "|---|---|---:|---:|---:|---:|---:|"]
    for model, label in MODELS:
        for dataset, dataset_label, _ in DATASETS:
            values = [f"{n} ({n / 2:g}%)" for n in groups[(model, dataset)]["counts"]]
            lines.append("| " + " | ".join([label, dataset_label, *values]) + " |")
    lines += ["", "Polaris 在两个模型上都更集中于 0/4；compression 有更多 4/4 的题目。此结论适用于本次随机抽样与推理设置。", "",
              "## 回答长度分布", "",
              "纵轴是长度小于等于横轴值的回答比例；曲线越靠右表示回答越长。每组纳入全部 800 个回答，包含答错、thinking 未结束和达到 32k 上限的输出。"
              "长度包括 thinking、最终回答与 EOS，不包括输入 prompt；1k = 1,024 tokens。", "",
              "![回答长度分布](response_length_ecdf.png)", "",
              "| 模型 | 数据集 | 平均 tokens | 中位数 tokens | 达到 32,768 tokens |", "|---|---|---:|---:|---:|"]
    for row in lengths:
        lines.append(f"| {row['model']} | {row['dataset']} | {row['mean_tokens']:,.1f} | {row['median_tokens']:,.1f} | "
                     f"{row['at_32768_tokens']} / 800 ({row['at_32768_percent']:.2f}%) |")
    lines += ["", "设置：thinking on；temperature 0.6、top-p 0.95、top-k 20、min-p 0、seed 42，最大输出 32,768 tokens。"
              "没有 system prompt；各模型使用自己的原生聊天模板。仅判最后一个完成的 `</think>` 后的回答。", "",
              "[双页矢量 PDF](training_difficulty_distributions.pdf) · [正确率 SVG](question_correctness_distribution.svg) · "
              "[长度 SVG](response_length_ecdf.svg) · [正确率数据 CSV](question_correctness_distribution.csv) · "
              "[长度汇总 CSV](response_length_summary.csv) · [校验与元数据](distributions_summary.json) · [原评测报告](../README.md)", ""]
    (destination / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root / "report/distributions"
    destination.mkdir(parents=True, exist_ok=True)
    groups, sources = load_verified_data(root)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelcolor": "#344054",
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    with PdfPages(destination / "training_difficulty_distributions.pdf") as pdf:
        for name, fig in (("question_correctness_distribution", correctness_figure(groups)),
                          ("response_length_ecdf", length_figure(groups))):
            fig.savefig(destination / f"{name}.png", dpi=220)
            fig.savefig(destination / f"{name}.svg")
            pdf.savefig(fig)
            plt.close(fig)
    make_report(root, destination, groups, sources)
    print(f"Saved both distributions, vector PDF/SVG, and verified source tables to {destination}")


if __name__ == "__main__":
    main()
