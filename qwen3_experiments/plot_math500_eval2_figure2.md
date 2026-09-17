# Eval2：Individual Token Budget

沿用 Eval3 的 Figure 2 绘图样式，展示六个模型在每题独立 token 预算下解出的题目数量。

## 最新：Individual 使用 cap8 step 150，预算为 512–12K

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --config qwen3_experiments/plot_math500_eval2_figure2_cap8_512_12k.json
```

CAM - Individual Budget 使用 **cap8（step 150）**，在 **512、1K、2K、4K、8K、12K** 下分别解出 **303、356、396、410、434、450** 题。六个点均为已有 Eval2 实测结果；整图 36 个 summary 已与先前逐回答核验的数据比对哈希，无需重新采样。沿用 512–12K 的坐标、拟合线和小散点，右下角为 **Zoomed View at 12K**。

图表输出到 `outputs/figures/math500_eval2_individual_budget_512_12288_cap8/`。cap8 的六组结果汇集在 `outputs/eval_math500_individual_budget_512_12288_cap8/results/`，同目录 `source_manifest.json` 记录源文件和哈希；图表目录的 `source_audit_reference.json` 记录历史核验来源。

12K 时 Individual 解出 450 题、Shared 解出 446 题，标签分别放在点的两侧，放大图上界扩展至 458；散点保留真实坐标。

## Individual 使用 L+256 step 100，重新测量 512–12K

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --config qwen3_experiments/plot_math500_eval2_figure2_lplus256_step100_512_12k.json --audit
```

CAM - Individual Budget 使用同一训练运行 `754f8ca2` 的 **L+256 step 100** checkpoint，固定版本为 `f9ab3bfd185c145ea0a00479668a9dae4438e7bf`。其 **512、1K、2K、4K、8K、12K** 六个点在 GPU 4、5 上重新采样；其他五个模型沿用此前的 30 个实测点。数据集、分词器、每题预算消耗方式、单条回答 4K 上限、采样设置与评分器保持原 Eval2 协议。

运行记录与六组新结果位于 `outputs/eval_math500_individual_budget_512_12288_lplus256_step100/`。`checkpoint_selection.json` 与 `model_stage_check.json` 记录 checkpoint 选择及 step 校验，GPU manifest 记录依赖、输入输出哈希和评测命令。

图表输出到 `outputs/figures/math500_eval2_individual_budget_512_12288_lplus256_step100/`，沿用 512–12K 坐标与 12K 放大图。`step100_vs_step150.csv` 提供两个 checkpoint 的逐预算题数和差值。六组新评测已完成，整图 36 组结果共核验 **317,505 条原始回答**；仅微调样式时可省略 `--audit`。

| Checkpoint | 512 | 1K | 2K | 4K | 8K | 12K |
|---|---:|---:|---:|---:|---:|---:|
| L+256 step 100 | 311 | 355 | 393 | 406 | 433 | 444 |
| L+256 step 150 | 319 | 353 | 397 | 409 | 422 | 438 |

12K 时 Individual 为 444 题，Shared 为 446 题，放大图将两者标签放在点的两侧以保持可读性；所有散点保留真实坐标。

## Individual 使用 L+256 step 150，预算为 512–12K

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --config qwen3_experiments/plot_math500_eval2_figure2_lplus256_512_12k.json
```

CAM - Individual Budget 使用 **L+256（step 150）**，在 **512、1K、2K、4K、8K、12K** 下分别解出 **319、353、397、409、422、438** 题。使用此前已核验的 Eval2 实测结果；纵轴为 50–500，右下角为 **Zoomed View at 12K**。

输出目录为 `outputs/figures/math500_eval2_individual_budget_512_12288_lplus256/`。L+256 的六组原始结果汇集在 `outputs/eval_math500_individual_budget_512_12288_lplus256/results/`，同目录 `source_manifest.json` 记录源文件与哈希。图表目录的 `source_audit_reference.json` 记录历史核验来源，并确认全部 36 个 summary 与已核验数据的哈希一致。

## Individual 使用 L+512，预算为 512–8K

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --config qwen3_experiments/plot_math500_eval2_figure2_lplus512_512_8k.json
```

保留 **512、1K、2K、4K、8K** 五个点，12K 已从散点和拟合输入中移除。右下角改为 **Zoomed View at 8K**；L+512 与 L1 在 8K 均解出 431 题，分别在点的两侧标注。f_cov 解出 429 题，标签位置单独微调以避免重叠，散点保留真实坐标。

输出目录为 `outputs/figures/math500_eval2_individual_budget_512_8192_lplus512/`。该图使用上一版已核验数据的 30 个点，数据来源与哈希核对结果记录在 `source_audit_reference.json`。

## Individual 使用 L+512，预算为 512–12K

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --config qwen3_experiments/plot_math500_eval2_figure2_lplus512_512_12k.json --audit
```

六个预算点为 **512、1K、2K、4K、8K、12K**。CAM - Individual Budget 使用 **L+512（step 150）**；其 512、1K 两个点在 GPU 4、5 上按原 Eval2 协议补测，其他 34 组结果沿用已有实测数据。全部六个模型仍使用每题独立累计预算，单次回答上限为 4096 tokens。

为显示低预算下的所有点，纵轴扩展至 **50–500，每 50 题一格**，并保留 **Zoomed View at 12K**。输出目录为 `outputs/figures/math500_eval2_individual_budget_512_12288_lplus512/`，包括 PNG、PDF、SVG、数据表、拟合参数、配置与完整原始回答核验报告。

原始结果汇集在 `outputs/eval_math500_individual_budget_512_12288_lplus512/results/`。同目录的 `materialized_results.json` 记录复用结果的源路径与文件哈希；`manifest_gpu4.json`、`manifest_gpu5.json` 记录补测时的 checkpoint、依赖、命令及结果哈希。微调外观时可省略 `--audit`。

## Individual 使用 L+512，预算为 2K、4K、8K、12K

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --config qwen3_experiments/plot_math500_eval2_figure2_lplus512_2k_12k.json
```

CAM - Individual Budget 使用 **L+512（step 150）**，四个预算点的解出题数为 **395、408、431、442**。沿用当前 2K–12K 样式：纵轴 300–500、每 25 题一格，右下角为 **Zoomed View at 12K**，使用小散点。8K 时 L+512 与 L1 均解出 431 题，两个点的真实坐标重合。

图表输出到 `outputs/figures/math500_eval2_individual_budget_2048_12288_lplus512/`。数据来自已有 Eval2 实测结果；`source_audit_reference.json` 记录历史核验报告，并确认全部 24 个 summary 的路径与哈希一致。

## Individual 使用 L+256，预算为 2K、4K、8K、12K

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --config qwen3_experiments/plot_math500_eval2_figure2_lplus256_2k_12k.json
```

CAM - Individual Budget 使用 **L+256（step 150）**，四个预算点的解出题数为 **397、409、422、438**。读取此前已核验的 Eval2 数据，无需重新采样。纵轴为 300–500、每 25 题一格；右下角标明 **Zoomed View at 12K**，使用小散点。Base 和 MaxRL 在 12K 均解出 427 题，分别在点的两侧标注。

图表输出到 `outputs/figures/math500_eval2_individual_budget_2048_12288_lplus256/`，包含 PNG、PDF、SVG、逐点数据和本次配置。`source_audit_reference.json` 记录历史核验报告，并确认全部 24 个 summary 的路径与哈希一致。

## 与 Shared Budget 对齐的 256–4K，Individual 使用 L+256

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --config qwen3_experiments/plot_math500_eval2_figure2_aligned_lplus256.json --audit
```

该版本将 CAM - Individual Budget 换成 **L+256（step 150）**。横轴仍为 **Individual Token Budget**，五个预算点为 256、512、1K、2K、4K；沿用与 Shared Budget 一致的坐标范围、图例名称和 4K 放大图。

L+256 的 256、512、1K 点在 GPU 4、5 上按相同 Eval2 协议补测，2K、4K 点复用此前已核验的结果。L+256 原始结果位于 `outputs/eval_math500_individual_budget_256_4096_lplus256/results/`；另外五个模型沿用上一版的已测结果，来源逐点记录在图表目录的 `points.csv` 中。每题独立耗尽预算，答对后仍继续生成；至少一次答对即计为解出该题。

图表单独输出到 `outputs/figures/math500_eval2_individual_budget_256_4096_lplus256/`。微调使用上面的 JSON 配置；仅改样式时可省略 `--audit`。

## 与 Shared Budget 对齐的 256–4K，Individual 使用 cap8

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --config qwen3_experiments/plot_math500_eval2_figure2_aligned.json --audit
```

该配置使用每题 **256、512、1024、2048、4096** tokens，横轴刻度、范围与 Shared Budget 图一致；纵轴为 0–500，每 50 题一格。右下角标明 **Zoomed View at 4K**。CAM - Individual Budget 沿用上次 Individual 图的 **cap8**，其他五个模型为 L1 random 100–4000、ER、Base、MaxRL、f_cov。

4K 时 L1 与 MaxRL 仅相差 1 题；放大图通过 `detail.marker_size / marker_edgewidth` 使用更小的点，避免遮挡，并保持真实坐标。

六个模型的 256、512、1K 点按原 Eval2 脚本重新评测，2K、4K 点经逐回答核验后从原结果复用。全部原始回答、逐题结果和 summary 汇集在 `outputs/eval_math500_individual_budget_256_4096/results/`；`manifest_gpu4.json` 和 `manifest_gpu5.json` 记录 checkpoint 版本、模型与结果文件哈希、复用来源以及评测命令。

评测任务由 `run_math500_eval2_aligned.py` 在 GPU 4、5 上分别运行，可利用同一结果目录继续未完成的预算点。`run_config.json` 记录运行环境和模型路径。使用与历史评测一致的依赖、数据、采样种子和评分协议，不改变训练或 Eval3 的结果。

新图单独输出到 `outputs/figures/math500_eval2_individual_budget_256_4096_cap8/`，包含 PNG、PDF、SVG、`points.csv`、`fits.csv`、配置和原始回答核验报告。微调时使用 `plot_math500_eval2_figure2_aligned.json`；下文保留先前 2K–12K 各版本的说明。

## 运行与微调

在仓库根目录执行，无需 GPU：

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py
```

脚本读取已有的 24 个 summary。外观参数集中在相邻的 `plot_math500_eval2_figure2.json`；Eval2 使用独立的配置和输出目录，绘图函数复用 `plot_math500_eval3_figure2.py`。

| 调整内容 | JSON 参数 |
|---|---|
| 模型名称、颜色、点形状 | `models[].label / color / marker` |
| 点面积与描边 | `models[].marker_size / marker_edgecolor / marker_edgewidth` |
| 画布尺寸、绘图区位置 | `figure.size_inches / axes_rect` |
| 坐标轴标题、范围和刻度 | `axes.xlabel / ylabel / xlim / ylim / xticks / yticks` |
| 字号 | `figure.title_size / label_size / tick_size` |
| 图例位置、字号 | `legend.bbox_to_anchor / font_size` |
| 拟合线与外推 | `lines.mode / width / alpha / extension_factor` |
| 12K 放大图 | `detail.enabled / axes_rect / ylim / yticks` |
| 放大图文字位置与大小 | `detail.left_labels / label_offset_points / label_size` |

主图纵轴范围为 300–500，每 25 题一个刻度；同样的题数差在图中的间距是原先 0–500 范围的 2.5 倍。当前点面积为 30 points²，L1 星形为 36。默认 `paper-fit` 对 `log2(token_budget)` 做普通最小二乘线性拟合，实线覆盖测量范围，两侧虚线延伸至最小预算除以 1.5、最大预算乘以 1.5。散点始终表示真实测量值，虚线只表示拟合外推。

若要逐点连线并另存一版：

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --line-mode connect \
  --output-dir outputs/figures/math500_eval2_connected
```

也可用 `--config path/to/my_style.json` 指定修改后的配置。改变放大图预算时，同步修改 `detail.budget / xlim / ylim / title`，并确保该预算在 `budgets` 中；`detail.budget_field` 保持为 `total_output_budget_per_prompt`。

## 指标与数据

- **Individual Token Budget**：每题各自的累计输出 token 预算，为 2048、4096、8192、12288。500 道题分别耗尽自己的预算，因此实际平均消耗也恰好等于横坐标。
- **Number of Questions Solved**：500 题中，在该题的回答列表里至少有一次答对的题目数量。
- 每条回答最多生成 4096 tokens，并受该题剩余预算限制。某题答对后仍继续生成，直至用完该题预算。
- CAM - Individual Budget 对应 L+256；CAM - Shared Budget 对应 f_cov（L0=256）。这些是模型名称；本图所有模型均使用 Eval2 的每题独立预算评估协议。
- L1 使用 random 100–4000、step 300；ER、MaxRL、两个 CAM 模型使用 step 150；Base 为 Qwen3-1.7B-Base。
- 同一 MATH-500 数据与题目顺序，temperature=0.6、top-p=0.95、top-k=-1、seed=0，MathVerify 0.9.0，评分超时 1 秒。

| 模型 | 2K | 4K | 8K | 12K |
|---|---:|---:|---:|---:|
| L1 | 364 | 394 | 431 | 434 |
| ER | 358 | 371 | 394 | 397 |
| Base | 360 | 380 | 413 | 427 |
| MaxRL | 376 | 393 | 420 | 427 |
| CAM - Individual Budget | 397 | 409 | 422 | 438 |
| CAM - Shared Budget | 390 | 413 | 429 | 446 |

12K 时 Base 与 MaxRL 均解出 427 题，坐标完全重合；放大图在两侧分别标注名称和数值。所有散点使用真实坐标。

## CAM - Individual Budget 使用 L+512 的版本

使用独立配置生成 L+512 对照版：

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --config qwen3_experiments/plot_math500_eval2_figure2_lplus512.json
```

该配置将 CAM - Individual Budget 对应的模型换为 `fixed_n_rb_offset512_step150`，结果读取自 `outputs/eval_math500_offset512_vs_cost_variants/eval2/results/`。图例仍使用 CAM - Individual Budget；主图纵轴为 300–500，刻度间隔为 25 题。

| Individual 模型 | 2K | 4K | 8K | 12K |
|---|---:|---:|---:|---:|
| L+256 | 397 | 409 | 422 | 438 |
| L+512 | 395 | 408 | 431 | 442 |

L+512 版本单独输出到 `outputs/figures/math500_eval2_figure2_lplus512/`，包含 PNG、PDF、SVG、CSV、配置及核验报告。已有 L+256 配置和图片可用于对照；之后微调 L+512 版时使用上面的 `--config` 参数。

L+512 版已使用 `--audit` 核对 24 组结果的全部 293,607 条回答。8K 时 L+512 与 L1 同为 431 题，散点坐标重合；12K 时 L+512 解出 442 题。

## CAM - Individual Budget 使用 cap8 的版本

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py \
  --config qwen3_experiments/plot_math500_eval2_figure2_cap8.json
```

该配置使用 `fixed_n_rb_capped_cap8_token_mean_step150`，结果读取自 `outputs/eval_math500_total_budget_12288/results/`。图例使用 CAM - Individual Budget，沿用纵轴 300–500、每 25 题一个刻度的样式。右下角以 “Zoomed View at 12K” 标题标明是 12K 预算结果的局部放大图，可通过 `detail.title` 修改。

| Individual 模型 | 2K | 4K | 8K | 12K |
|---|---:|---:|---:|---:|
| cap8 | 396 | 410 | 434 | 450 |

结果单独输出到 `outputs/figures/math500_eval2_figure2_cap8/`，包含 PNG、PDF、SVG、CSV、配置及核验报告。已使用 `--audit` 核对 24 组结果的全部 291,324 条回答；12K 解出 450 题，比 L+512 多 8 题，比 L+256 多 12 题。

## 输出与核验

默认输出目录为 `outputs/figures/math500_eval2_figure2/`：

- `math500_eval2_figure2.png / .pdf / .svg`：图片及矢量版本；SVG 保留可编辑文字。
- `points.csv`：24 个测量点、原始 summary 路径、文件哈希、checkpoint。
- `fits.csv`：六条拟合线的斜率、截距、R² 和使用的指标。
- `config_used.json`：此次绘图使用的完整配置。
- `summary_checks.json`：普通运行时的 summary 一致性检查。
- `audit.json`：使用 `--audit` 时核验原始回答的报告。

首次生成已用以下命令核对全部 **312,935 条回答、24 组结果**，检查了题目 ID、逐题采样序号与种子、剩余预算、累计 tokens、回答长度上限及答对题数：

```bash
python qwen3_experiments/plot_math500_eval2_figure2.py --audit
```

仅修改外观时无需重复原始回答核验；普通运行保留已有 `audit.json`。更换数据后应重新运行 `--audit`。
