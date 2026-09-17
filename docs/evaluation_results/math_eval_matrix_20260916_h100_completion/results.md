# Math evaluation matrix

Completed budget/seed points: 700/700.

541 published GH200 points plus 159 audited H100 continuation points. The 541 original raw trajectories are available in the existing GitHub Release archives; the continuation adds the missing 159 points. See RAW_TRAJECTORIES.md for complete downloads and README.md for hardware and package provenance.

All models use the same ordinary math prompt. Failed attempts and EOS count toward output-token cost.

## Eval1: mean@4

Mean correctness across four responses per question, not pass@4.

### MATH-500 (500 questions)

| Model | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| F-Cov L+512 | 43.35% | 58.15% | 62.00% | 62.25% | 62.45% |
| Fixed-N RB L+512 | 45.65% | 60.35% | 63.40% | 63.60% | 63.65% |
| Qwen3-1.7B-Base | 18.25% | 44.40% | 55.70% | 57.65% | 57.55% |
| Efficient Reasoning | 43.70% | 57.50% | 63.45% | 64.80% | 65.05% |
| L1-Exact | 2.90% | 19.65% | 64.20% | 67.10% | 66.95% |

### Minerva Math (272 questions)

| Model | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| F-Cov L+512 | 16.73% | 24.72% | 25.92% | 26.01% | 25.74% |
| Fixed-N RB L+512 | 18.01% | 25.09% | 25.64% | 25.83% | 25.83% |
| Qwen3-1.7B-Base | 5.70% | 14.80% | 20.77% | 20.68% | 20.96% |
| Efficient Reasoning | 17.65% | 24.82% | 26.01% | 26.10% | 26.29% |
| L1-Exact | 2.11% | 5.97% | 26.29% | 27.85% | 29.14% |

### OlympiadBench (674 questions)

| Model | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| F-Cov L+512 | 9.46% | 22.89% | 27.34% | 28.08% | 28.04% |
| Fixed-N RB L+512 | 10.35% | 22.48% | 25.78% | 26.52% | 26.56% |
| Qwen3-1.7B-Base | 2.52% | 10.83% | 21.44% | 23.07% | 23.44% |
| Efficient Reasoning | 8.68% | 21.03% | 28.30% | 30.30% | 30.64% |
| L1-Exact | 1.52% | 4.82% | 27.04% | 29.75% | 29.93% |

### AMC22+23 (83 questions)

| Model | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| F-Cov L+512 | 10.84% | 26.51% | 31.93% | 30.42% | 31.93% |
| Fixed-N RB L+512 | 13.55% | 24.70% | 32.23% | 31.63% | 32.23% |
| Qwen3-1.7B-Base | 3.92% | 12.05% | 22.89% | 25.90% | 25.60% |
| Efficient Reasoning | 6.93% | 12.05% | 26.20% | 28.31% | 29.22% |
| L1-Exact | 1.20% | 6.33% | 31.93% | 34.04% | 34.04% |

## Eval2

Markers show measured means over seeds 0, 1, 2. Solid lines are log2-budget linear fits; dashed extensions are extrapolations. Dataset panels show solved-question counts; Macro Average equally weights the four dataset accuracies. Only complete seed groups are plotted; standard deviations are retained in CSV.

[Macro-average values and per-seed accuracies](eval2_macro_average.csv)

![eval2](eval2_questions_solved.png)

## Eval3

Markers show measured means over seeds 0, 1, 2. Solid lines are log2-budget linear fits; dashed extensions are extrapolations. Dataset panels show solved-question counts; Macro Average equally weights the four dataset accuracies. Only complete seed groups are plotted; standard deviations are retained in CSV.

[Macro-average values and per-seed accuracies](eval3_macro_average.csv)

![eval3](eval3_questions_solved.png)
