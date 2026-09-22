# 旧 CSV 隔离复现与 NN 概率拟合修正

2026-09-22。前一轮问题定位见 `异常诊断_KDE_NN.md`；本文记录实际修正与验证。
本轮没有覆盖既有 2000 条离线结果、旧投稿 CSV 或 Fig.6。

## 1. 旧 CSV 的复现边界

入口 `scripts/reproduce_legacy_kde.py` 从原 notebook 提取 **原封不动的
generate_eval_data 函数**，把 `dev` 的 `src/` 导出到新输出目录，由独立进程加载。
只执行 Push/SAC 的五个 seed，不执行 notebook 中其他场景单元。

```bash
conda run --no-capture-output -n gc_ope python scripts/reproduce_legacy_kde.py \
  --baseline /home/tacmon/workspace/ex_SSD/gc_ope \
  --output logs/legacy_kde_reproduction_20260922
```

输出目录必须不存在，重复执行需换新目录，防止覆盖。两个 CPU worker，BLAS 单线程。
checkpoint 和 replay buffer 从原路径只读加载，不复制大数据。

保留旧协议：XYZ；两侧分别标准化；高斯 KDE 带宽 1.0；历史每轮从 fixed 全部记录
有放回抽 100 条，包含当前轮；时间折扣 0.9；MC 10000 点、单次；过滤估计密度
不超过 1e-10 的样本。RB 每轮抽 1000 条，原代码的索引上界 `replay_buffer.size()`
也保留。执行顺序为 RB 抽样、历史抽样、RB KL、历史 KL。

旧全局随机状态没有保存，无法承诺逐数值相等。本轮每个 seed 独立进程从
`numpy.random.seed(0 + seed)` 开始，结果可重放。原 notebook、旧 CSV 的 SHA256
保存在 `manifest.json`，结束时再次核对。源码快照、原函数、driver 和每 seed
完整日志都在输出目录。`comparison_seed*.csv` 是相同 checkpoint 的逐行对照，
`comparison_summary.csv` 给出两列 KL 的均值、绝对误差及覆盖率。

实际完成五 seed × 100 checkpoint，退出码 0。旧 CSV 共 499 行，本次 498 行，
双方共同 checkpoint 497 行；共同点历史 KL 均值为 **旧 0.027579、本次 0.025228**。
各 seed 的旧/本次均值分别为 0.027090/0.027894、0.027207/0.024120、
0.025288/0.021594、0.024108/0.024118、0.034127/0.028419。
低 KL 的量级重现，但逐点绝对差最大达到 0.696513（seed 2），不能称为数值一致。

本次 seed 4 在 10000、20000 步的历史抽样没有成功记录，按旧函数跳过；旧 seed 1
缺少 10000 步，而本次抽到了成功记录。它们都是抽样造成的覆盖差异，不补造行。
原 notebook、旧 CSV 结束时摘要均未变化。独立目录内 README、audit.json 和
comparison.png 记录覆盖率及未平滑的对照图；旧 Fig.6 没有被重绘或改写。

## 2. NN 的层结构与训练规则

正式配置仍在 `configs/evaluate/push_same_family_all100.json`，现在明确写：

```json
{
  "hidden_layer_sizes": [16, 16],
  "n_epochs": 500,
  "early_stopping": true,
  "validation_fraction": 0.1,
  "min_epochs": 50,
  "n_iter_no_change": 30,
  "tol": 0.0001
}
```

网络仍为 2→16→16→1，没有扩大层宽。接口接受非空正整数列表/元组，也支持
不同层宽；旧 `hidden_width=16` 仍兼容，但不能同时传两个参数。

早停不再使用 sklearn 内置分类 accuracy，而是每 epoch 计算本侧留出集的
**加权二元 log-loss**。至少训练 50 epoch，连续 30 轮没有超过 1e-4 的改善则停止，
最大 500 epoch。最终恢复所有已训练轮次中验证损失最低的完整分类器。
500 是明确的优化上限，不是对 oracle KL 搜索得到的最优值，也不保证每次收敛。
两侧独立拟合，使用完全相同的训练规则，没有仅加强参考侧的特殊待遇。

划分沿用按记录分层的 10% 留出。历史有放回抽样可能产生跨分区重复记录，所以
验证损失用于优化和诊断，**不称为独立泛化成绩**。标准化仍使用本侧所有训练目标；
核平滑、抽样记录、时间权重、MC 方向及样本数没有变化。

`early_stopping=false` 仍保留原固定预算拟合路径；它不是本次正式配置。

## 3. 拟合质量不再只看有限 KL

每个 job JSON 保留最佳 epoch、实际训练轮数、停止原因、逐轮验证损失、训练损失、
验证集常数基线损失、概率范围及诊断 AUC。常数基线使用**训练分区**的加权成功率。
新 CSV 有 `fit_quality`、`fit_warnings`，控制台和 audit 也汇总提醒。

- 验证损失未优于常数基线：明确提醒，不能据此认定已经学会成功区域。
- 达到最大轮数仍未触发早停：明确提醒优化预算耗尽，不能声称已经收敛。
- 质量提醒保留 KL，不按指标大小删除样本；`status=ok` 只表示计算成功。
- `passed_checks` 只表示通过上述有限检查，不是概率校准或真实分布准确性的证明。

## 4. 已执行的真实数据验证

独立目录 `logs/nn_logloss_validation_20260922/`：五 seed × 100k、400k、410k、1M，
共 20 个 checkpoint、40 次独立拟合，MC 均为 **10000 × 5**。全部计算成功。
`kl_comparison.csv` 对照原正式 CSV；`fit_quality.csv` 汇总两侧训练诊断。

| checkpoint | 修改前 KL 五 seed 均值 | 新 log-loss 早停 |
| --- | ---: | ---: |
| 100000 | 0.000105 | 0.520387 |
| 400000 | 0.002364 | 0.076836 |
| 410000 | 0.168635 | 0.079824 |
| 1000000 | 0.911391 | 0.034170 |

百万步五 seed 分别由 1.227913 / 0.483083 / 0.104914 / 1.477233 / 1.263810，
变为 0.026611 / 0.022335 / 0.036557 / 0.060540 / 0.024808。
早期 KL 上升也保留：旧时两侧都欠拟合造成的相似平坦分布，不能解释为高准确度。

40 次拟合中有 4 次提醒：seed 1 / 100k 历史侧、seed 2 和 5 / 1M 参考侧达到预算；
seed 4 / 100k 参考侧未优于常数基线。其余 36 次通过这两项检查。
因此可以确认原 accuracy 早停问题已修正，但不能宣称所有 checkpoint 都充分拟合。
此次没有为降低 KL 而调整层宽、学习率、带宽或重复挑选随机种子。

这批验证先完成算法计算，再增加 CSV 的提醒列；原 job JSON 已包含全部诊断。
保留它的原始清单和结果，不回写文件伪装成最终字段版本。

另按单文件运行 7 组测试，共 48 项通过：概率早停与层结构 11 项、连续密度与采样
9 项、保存版本固定预算算法对照 4 项、离线协议与质量提醒 8 项、真实 Push/SAC
课程接口 9 项、五 seed 调度及中断 5 项、单入口续跑与清理 2 项。
接口测试故意只训练 2～5 epoch，因此这些测试中的未收敛 warning 不代表拟合验收通过。

## 5. 正式 NN 重跑命令

保留五 seed 并行、每 seed 四 worker，仅重跑 NN；由用户启动：

```bash
conda run --no-capture-output -n gc_ope python \
  /home/tacmon/workspace/ex_SSD/gc_ope_refactor/scripts/run_push_all100.py \
  --methods nn \
  --output /home/tacmon/workspace/ex_SSD/gc_ope_refactor/logs/nn_logloss_v2_all100_5x4
```

未指定 `--methods` 时仍按 NN→FM→NF→GMM 串行。默认输出已换成新目录，协议标识为
`push_same_family_nn_logloss_v2`，旧结果不能拿来断点续跑。
旧 KDE 仍是 legacy，独立标准化、带宽和参考分布的差异仍在；本次复现不把它
变成与四方法同口径的准确性排名。
