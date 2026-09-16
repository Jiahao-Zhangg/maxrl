# Eval1：Token Budget 折线图，Individual 使用 L+512

沿用 Eval2 / Eval3 的 Figure 2 绘图样式，展示 L1、ER、Base、MaxRL、CAM - Individual Budget（L+512）和 CAM - Shared Budget（f_cov）的 Eval1 结果。

## 运行

在仓库根目录执行，无需 GPU：

```bash
python qwen3_experiments/plot_math500_eval1_figure2.py
```

样式参数位于相邻的 `plot_math500_eval1_figure2.json`。默认输出到 `outputs/figures/math500_eval1_figure2_lplus512/`，包含 PNG、PDF、SVG、全部 30 个点的 `points.csv`、拟合参数 `fits.csv` 和 `config_used.json`。

核对原始回答：

```bash
python qwen3_experiments/plot_math500_eval1_figure2.py --audit
```

首次生成已核对 **60,000 条回答、30 组结果**。核验内容包括每题完整且唯一的四个样本、跨模型和上限档位的题目 ID、实际长度及生成上限、正确回答数、mean@4、平均长度、最大长度和达到上限的回答比例。

普通重绘只检查 summary，写入 `summary_checks.json`，并保留已有的 `audit.json`；更换数据后应重新运行 `--audit`。

## 坐标与指标

每个模型在五个单条回答生成上限下评估：256、512、1024、2048、4096。每档包含 500 道题，每题生成 4 条回答。

- **X：Token Budget**，为单条回答的生成上限，每个模型的五个点分别位于 256、512、1024、2048、4096。
- **Y：Mean Number of Questions Solved**，先对每题的四个正确性分数取平均，再对 500 题求和，即 `500 × mean@4 = 正确回答总数 / 4`。因此题数可以带有 0.25、0.50 或 0.75 的小数部分。
- 主图 X 轴为 log2 刻度，标签为 256、512、1K、2K、4K；显示范围为 230–4600，给两端散点留出空间。
- 每条折线按预算从小到大直接连接五个实测点。
- 主图 Y 轴为 0–400，每 50 题一个刻度，覆盖全部测量点。
- 当前仅展示加宽后的主图和右侧居中的分组图例，放大图关闭。

计算公式，记第 i 题第 j 次回答的输出长度为 `t[i,j]`、二元正确性为 `r[i,j]`：

```text
X = per-response generation limit
mean_output_tokens = sum(t[i,j]) / (500 × 4)
mean@4 = sum(r[i,j]) / (500 × 4)
Y = sum(r[i,j]) / 4 = 500 × mean@4
```

例如 L+512 的 4K 上限结果有 1,250 条正确回答，图中对应点为 **(4096, 312.50)**。其实际平均长度 371.373 tokens 保存在 `points.csv` 中。

### 4K 上限结果

| 模型 | 实际平均 tokens | mean@4 | 平均解出题数 |
|---|---:|---:|---:|
| L1 | 806.9085 | 67.85% | 339.25 |
| ER | 586.1840 | 65.50% | 327.50 |
| Base | 727.4300 | 58.35% | 291.75 |
| MaxRL | 1015.9830 | 69.70% | 348.50 |
| CAM - Individual Budget（L+512） | 371.3730 | 62.50% | 312.50 |
| CAM - Shared Budget（f_cov） | 265.2275 | 58.75% | 293.75 |

L1 使用 random 100–4000、step 300；ER、MaxRL、L+512、f_cov 使用 step 150；Base 为 Qwen3-1.7B-Base，f_cov 为 offset256 版本。所有结果使用同一 MATH-500 数据、temperature=0.6、top-p=0.95、top-k=-1、seed=0、MathVerify 0.9.0 和 1 秒评分超时。结果路径、checkpoint 与 summary 哈希保存在 `points.csv`。

## 微调位置

| 调整内容 | JSON 参数 |
|---|---|
| 方法名称、颜色、点形状与大小 | `models[].label / color / marker / marker_size` |
| 图例位置、字号 | `legend.bbox_to_anchor / font_size` |
| 画布、绘图区大小 | `figure.size_inches / axes_rect` |
| 横坐标指标 | `axes.x_metric`：`max_output_tokens` 为生成上限，`mean_output_tokens` 为实际平均用量 |
| 坐标标题、范围、刻度 | `axes.xlabel / ylabel / xlim / ylim / xticks / xtick_labels / yticks` |
| 放大图开关、标题、位置、范围 | `detail.enabled / title / axes_rect / xlim / ylim`，当前关闭 |
| 放大图 token 刻度 | `detail.xticks / xtick_labels / xlabel` |
| 放大图选择哪个生成上限 | `detail.budget`；`budget_field` 保持为 `max_output_tokens` |
| 放大图题数精度 | `detail.value_format`，当前为 `.2f` |
| 放大图文字位置 | `detail.left_labels / label_offset_points / label_offsets` |
| 连线方式、线宽与透明度 | `lines.mode / width / alpha`；当前为 `connect` |

更改 `x_metric` 时同步修改坐标标题、范围和刻度；例如切回实际用量可使用 `xlabel=Tokens Used`、`xlim=[170,1120]`，并在此范围内设置刻度。放大图配置保留；启用 `detail.enabled=true` 时应同步安排图例和放大图的位置。`label_offsets` 用模型 ID 映射到 `[水平偏移, 竖直偏移]`，单位为 points，只移动文字。

默认 `connect` 直接连接相邻实测点。`fits.csv` 另存参考拟合参数；若选择 `paper-fit`，图中使用普通最小二乘拟合 `Y = slope × log2(X) + intercept`，两端虚线延伸至最小横坐标除以 1.5、最大横坐标乘以 1.5。

需要显示拟合趋势时可另存一版：

```bash
python qwen3_experiments/plot_math500_eval1_figure2.py \
  --line-mode paper-fit \
  --output-dir outputs/figures/math500_eval1_fitted
```

代码中的 `load_points()` 定义坐标并检查评估协议，`audit_run()` 核对原始回答，绘图复用 `plot_math500_eval3_figure2.py`。放大图的刻度、数值格式及文字偏移均为可选配置，已有 Eval2 / Eval3 默认样式保持一致。
