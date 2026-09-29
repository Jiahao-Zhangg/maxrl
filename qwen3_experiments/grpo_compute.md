# GRPO on allocation 146102

从 login node 启动整套训练与上传流程：

```bash
bash qwen3_experiments/launch_grpo_compute.sh
```

只准备并核对配置，不启动训练：

```bash
bash qwen3_experiments/launch_grpo_compute.sh --prepare-only
```

需要更改目录或上传仓库前缀时，在首次准备前设置 `GRPO_RUN_DIR`、`GRPO_HF_PREFIX`。
`GRPO_JOB_ID` 默认 `146102`，`GRPO_PYTHON_BIN` 默认使用已配置的 maxrl 环境。

SSH 仅负责在 `orchard-flame-23` 启动持久 supervisor，随后返回。
Supervisor、Slurm training launcher、checkpoint monitor、Ray 和训练进程均在该 compute node。
Supervisor 使用独立会话，并将输入重定向至 `/dev/null`、输出写入文件，运行不依赖 login node 的终端。
训练开始前取得该 allocation 的既有锁，并确认 GPU 空闲；不会停止其他任务。
训练本身不运行训练前或训练中的评测。最终 checkpoint 的独立评测可按下文接入。

训练沿用 L+0 的初始 Qwen3-1.7B、3200 条 Polaris 数据、batch 32、每题 16 个回答、
1280-token prompt 上限、32768-token response 上限、学习率 1e-6、8 卡、100 步和 KL=0。
GRPO 使用组内标准差归一化。独立 runtime 固定 L+0 的训练代码，并加入本次 reward 修改。
原 L+0 工作目录不会被修改；模型文件和数据通过哈希核对。

训练评分启用 `check_eos=True` 和 `score_after_thinking=True`：

- 只检查实际 response 中未被 mask 的 EOS；prompt 或 padding 中的 EOS 不计。
- 只将最后一个 `</think>` 后的非空回答交给 MathVerify。
- 缺少 EOS、thinking 未闭合、回答为空或再次打开未闭合 thinking，reward 为 0。
- Qwen prompt 已提供 `<think>`，无需模型重复输出开始标记。
- `force_eos=False`，不会为截断输出补上 EOS。

每 10 步保存一个完整 checkpoint，包含模型、优化器、额外状态和 dataloader 状态。
Checkpoint 位于 compute node 的 `/tmp/grpo146102/checkpoints`。
监控器等待全部 8 个 rank 和保存完成标记，将每步上传到独立 Hugging Face 模型仓库
`hi-todayis-jh/grpo-qwen3-1.7b-polaris-1-8-3200-bs32-32k-146102-step_N`。
新仓库默认公开，上传前也会将已有目标仓库设为公开。
已启动的 146102 使用冻结的旧上传代码，其 step 10–100 的全部十个目标仓库已预先设为公开，
包括之前的私有仓库；实际远端可见性的核验记录为 `hf_checkpoint_archive/public_visibility_policy.json`。
只有远端大小和内容哈希通过校验、本地文件自校验以来未变化，才删除该本地 checkpoint。
上传失败保留本地文件并每 30 秒重试。第 100 步在训练进程退出后才允许删除。

共享运行目录默认是 `outputs/grpo_qwen3_1_7b_polaris_1_8_3200_bs32_32k_146102`，保留：

- `plan.json`、`resolved_config.yaml`、`config_comparison.json` 和代码快照。
- `launch.json`、`status.json`、`supervisor.log`、`train.log`。
- `hf_checkpoint_archive/status.json`、`upload.log`、`receipts/global_step_N.json`。

每份 receipt 记录远端仓库、不可变 commit、每个文件的哈希和删除时间，用于恢复 checkpoint。
`run_qwen3_1_7b_polaris_1_8_3200_grpo.sh` 是 supervisor 调用的训练子脚本；
需要自动上传和本地清理时，使用上述 `launch_grpo_compute.sh` 入口。

## 训练后的最终评测

`eval_l0_final.py` 同时支持 `training_kind="grpo"`。评测计划通过 `prepare-plan` 固定训练配置、
九个数据集的题目和 prompt tokens、采样参数、原始对照表与评测代码；`model_label="GRPO"`
使报告正确标注算法。可用 `reference_plan_root` 复用已经核验的 L+0 评测输入。

在该 allocation 的 compute node 上运行 `launch-queue --output-root <评测目录>`，
队列脱离 SSH 会话持续等待。`queue_node` 限定队列所在节点；login node 仅负责启动。
GRPO supervisor 的成功状态、独立训练退出记录、上传监控器成功退出，以及 step 100 的
八个分片上传校验 receipt 必须同时满足，才会固定 Hugging Face commit 并启动评测。
沿用训练使用的全部 `holder_locks`，取得锁并确认八张 GPU 空闲后接管；占用时继续等待。

评测使用九个数据集共 1,819 题、每题 4 个回答，最多输出 32,768 tokens。
只判最后一个 `</think>` 后的非空回答；thinking 未闭合、重新打开未闭合或后缀为空计 0。
**评测不额外要求 EOS**，沿用原 `eval_l0_final.py`；训练 reward 仍要求 EOS。
按同一批输出的精确 token 前缀计算 1k / 2k / 4k / 8k / 16k / 32k 预算结果。
`model_storage` 可将下载分片和合并模型放在 compute node 的本地磁盘；共享目录保存输入、
回答、校验记录与报告。`queue_status.json` 记录等待或运行状态，最终报告为 `report/README.md`。
