# Eval3 skip-solved：L1 Figure 2 风格绘图

## 运行

在仓库根目录执行：

```bash
python qwen3_experiments/plot_math500_eval3_figure2.py
```

绘图只依赖 Python、NumPy、Matplotlib，不需要 GPU。默认读取已有的 30 个 summary，生成 PNG、PDF、SVG、绘图数据 CSV 和拟合参数 CSV。

首次核对原始回答，或更换结果文件后，可以运行：

```bash
python qwen3_experiments/plot_math500_eval3_figure2.py --audit
```

默认输出目录：`outputs/figures/math500_eval3_skip_solved_figure2/`。再次运行会更新同名图；仅调整样式时不需要重复 `--audit`。

## 微调位置

常用参数集中在相邻的 `plot_math500_eval3_figure2.json`，不需要改数据读取代码。

| 调整内容 | JSON 位置 |
|---|---|
| 每个模型的名称、颜色、点形状 | `models[].label / color / marker` |
| 点大小、描边 | `models[].marker_size / marker_edgecolor / marker_edgewidth` |
| 模型和曲线顺序 | `models` 数组顺序；越靠后，点的绘制层级越高 |
| 预算点的选择 | `budgets`，只能选存在结果的预算 |
| 图片尺寸、分辨率、导出格式 | `figure.size_inches / dpi / formats` |
| 绘图区位置和留白 | `figure.axes_rect`，依次为 left、bottom、width、height，均为全图比例 |
| 标题、字体和字号 | `figure.title / font_family / title_size / label_size / tick_size` |
| 横纵轴范围和刻度 | `axes.xlim / ylim / xticks / xtick_labels / yticks` |
| 横纵轴的数据指标 | `axes.x_metric / y_metric` |
| 坐标轴标题 | `axes.xlabel / ylabel` |
| 网格与坐标轴线 | `axes.grid_* / spine_*` |
| 拟合线或逐点连线 | `lines.mode`：`paper-fit`、`connect`、`none` |
| 线宽、透明度、外推范围 | `lines.width / alpha / extension_factor / extension_alpha` |
| 图例字号、位置与分组 | `legend.font_size / bbox_to_anchor / loc / groups` |
| 4K 局部放大开关与布局 | `detail.enabled / axes_rect / ylim / yticks` |
| 放大图的标注大小和左右位置 | `detail.label_size / label_offset_points / left_labels` |

`marker_size` 是 Matplotlib scatter 的面积，单位是 points²。当前 `y_metric=num_prompts_solved`，`ylim` 和 `yticks` 使用 0–500 的题目数量。若要恢复百分比，改为 `y_metric=fraction_solved`，并把纵轴范围、刻度改回 0–1。当前 `x_metric=budget_per_prompt_reference`，横轴为每题参考预算 b；也可改为 `actual_tokens_per_question` 显示实际总生成 tokens ÷ 500。

也可以复制配置并单独导出，保留当前版本：

```bash
python qwen3_experiments/plot_math500_eval3_figure2.py \
  --config path/to/my_style.json \
  --output-dir outputs/figures/my_eval3_version
```

想直接连接实测点：

```bash
python qwen3_experiments/plot_math500_eval3_figure2.py \
  --line-mode connect \
  --output-dir outputs/figures/math500_eval3_connected
```

## 与论文 Figure 2 的对应

参考：[L1 论文 Figure 2](https://arxiv.org/pdf/2503.04697)，当前 PDF 为 v2，第 6 页。针对当前 MATH-500 数据，输出一个面板，右侧放分组图例。

从原 PDF 的字体与矢量绘图元素核对了以下参数：

- DejaVu Sans；标题 16、粗体轴标题 15、刻度 14、图例 16.8。
- 白底，浅灰色左/下轴线，隐藏上/右轴线。
- 网格为 `#cccccc`、虚线、线宽 0.8、透明度 0.6。
- 红色 `#e41a1c`、橙色 `#ff7f00`、紫色 `#984ea3`、蓝色 `#377eb8`、绿色 `#4daf4a`。
- 两个 Our Methods 使用黑色描边圆点，其他模型使用星形、三角形、菱形、方形。
- 拟合线线宽 1.5、透明度 0.5；虚线外推透明度 0.4，两端 token 数分别除以、乘以 1.5。

**默认的细线是对 `log2(token_budget)` 做线性拟合，不是逐点连线。** `paper-fit` 使用普通最小二乘 `questions_solved = slope × log2(token_budget) + intercept`；实测结果始终由不透明散点表示，拟合不会修改散点。拟合只是显示趋势，虚线外推不代表实际测量结果；线超出纵轴范围时由绘图区裁剪。可以用 `connect` 切换为逐点连线。

图中 Base 的橙色菱形和 CAM - Shared Budget 的橙色圆点沿用了论文中两种方法共享橙色、靠形状区分的配色方式。所有方法都使用真实横坐标；接近的点可能重叠，可通过 `marker_size` 微调。

当前画布为 10.2 × 7.6 英寸，右侧为完整方法名留出空间；纵轴每 50 题的显示间距已增大，点面积改为 30 points²（L1 星形为 36），并减细描边。4K 下 Base=432、MaxRL=435，仅差 3 题，所以右下角增加一个局部放大图，按真实位置展示六个结果，并交错标注名称和题数。标题 “Zoomed View at 4K” 明确标示放大图，可通过 `detail.title` 修改。没有对散点作横向或纵向偏移。可用 `detail.enabled=false` 关闭放大图。

## 数据含义

- **Eval3 skip-solved**：共享生成 token 预算为 `500 × b`；按固定种子打乱顺序，反复遍历题目，已答对的题目在后续轮次跳过。
- **横轴 Shared Token Budget**：每题参考预算 b，为 256、512、1024、2048、4096；共享总预算为 `500 × b`。
- **纵轴 Number of Questions Solved**：500 题中至少有一次正确回答的题目数量，范围 0–500；未访问的题目不计为已解出。
- 每条回答上限为 4096 tokens，最终请求受剩余总预算约束。这里的横轴不是单条回答平均长度。
- 这 30 组结果均耗尽共享预算，因此横坐标恰好是 256、512、1024、2048、4096。
- 使用原始 Eval3 结果；此前的 mean@4、共同答对子集、log-ratio 和 epsilon 分析不参与本图计算。
- CAM - Individual Budget 对应原 L+256；CAM - Shared Budget 对应原 f_cov（L0=256）。图例、4K 标注和导出数据使用这两个新名称，原始模型标识与结果文件路径保留在配置中。
- L1 = random 100–4000，step 300；ER、MaxRL 和两个 CAM 模型都是 step 150。Base 为 Qwen3-1.7B-Base。
- 所有结果均为同一 MATH-500 数据、temperature=0.6、top-p=0.95、top-k=-1、seed=0、MathVerify 0.9.0，单题评分超时 1 秒。

| 模型 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| L1 | 108 | 197 | 367 | 417 | 444 |
| ER | 123 | 274 | 342 | 383 | 408 |
| Base | 101 | 187 | 306 | 394 | 432 |
| MaxRL | 73 | 147 | 343 | 402 | 435 |
| CAM - Individual Budget | 257 | 341 | 375 | 421 | 448 |
| CAM - Shared Budget | 292 | 356 | 398 | 431 | 458 |

部分旧结果的目录或 `evaluation` 字段仍称为 Eval4；这是当时共享预算评估的旧命名。该脚本拒绝 IID 不跳过题目的结果，并在 `--audit` 下逐条确认已经答对的题目没有再次采样。

## 代码与产物结构

- `load_points()`：结果路径、协议检查、横纵坐标定义。
- `audit_run()`：原始回答、题目 ID、随机种子、遍历顺序、跳过已解题和 token 账目的核验。
- `draw_model()`：拟合线、外推虚线、实测点。
- `draw_detail()`：单个预算的局部放大和错开排列的文字标注。
- `render()`：坐标轴、图例、外观和导出。
- `points.csv`：全部 30 个实测点及其原始 summary 路径、文件哈希、checkpoint。
- `fits.csv`：六条拟合线的斜率、截距、R²，以及拟合所使用的横纵轴指标。
- `config_used.json`：生成该版图时使用的完整参数。
- `audit.json`：首次 `--audit` 核验了 35,072 条回答、30 组结果。后续仅改样式不会覆盖该审计记录；数据更新后应重新运行 `--audit`。
- PDF 使用嵌入字体；SVG 保留可编辑文字。
