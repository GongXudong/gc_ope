# NF 与 FM 新增方法：实现、验证与全量结果

## 本次交付

新增两个独立方法，旧 `nf`、`fm` 的算法文件、默认参数、CSV 和图均保留：

| 方法名 | 图例 | 实现 | 正式配置 |
|---|---|---|---|
| `nf_reg` | NF (regularized) | `src/gc_ope/evaluate/evaluator_regularized_flow.py` | `configs/evaluate/push_nf_regularized_all100.json` |
| `fm_ensemble` | FM (ensemble) | `src/gc_ope/evaluate/evaluator_flow_ensemble.py` | `configs/evaluate/push_fm_ensemble_all100.json` |

候选 `fm_reg`（单模型加噪/早停）也作为独立实现保留用于追溯，但没有通过本轮的
跨 seed 验证，不列入最终完整 Fig.6，不把未通过的试验包装成成功改进。

## 改了什么

### NF：限制容量、目标噪声、验证选轮数

原 NF 为四个变换、隐藏宽度 32、固定 100 次更新。新 NF 使用两个变换、宽度 16，
在标准化目标上加入标准差 0.2 的高斯训练噪声，优化带时间权重的负对数似然。
预测密度和采样仍来自 NF 本身，不再给输出拟合 KDE，不改变 raw-space KL。

每次拟合将本模型已有成功目标中的 15% 独特坐标留出：相同目标的全部重复记录
进入同一侧，避免固定网格副本泄漏。最多 300 次更新，至少 30 次，每 10 次检查
加权验证 NLL，50 次未显著改善则结束。标准化只在训练侧拟合；选出最优轮数后，
重新初始化网络和 scaler，用本模型的全部成功记录重拟合这些轮数。

早期不足 10 个独特成功目标时，不伪造验证集，使用预先指定的 50 次预算并记录
`insufficient_unique_goals`。不足两个成功样本仍明确跳过。历史侧、当前全量参考侧
各自执行完全相同的算法，没有共享已拟合参数或交叉使用另一侧标签。

这是一种正则化方法变体，不是与旧 NF 行为等价的结构重构。

### FM：三模型等权密度集成

三个 FM 成员使用同一批带权成功目标，初始化种子为 0、1、2；每个成员 500 次
更新、宽度 32、每次抽 2000 个目标、目标噪声标准差 0.1，保留 RK4 的 32 步。
本次采用固定预算，不启用候选中效果不稳定的验证早停。

混合密度为 `p(x) = (p_0(x)+p_1(x)+p_2(x))/3`，对数密度通过 logsumexp 计算。
采样先等概率选择成员，再从该成员的流采样。不能平均 log density，那对应的不是
这个归一化混合分布。历史侧和参考侧都是独立拟合的三个成员集成。

这降低了单次初始化产生的波动，但计算更贵：一次密度查询需要三个成员的 ODE
密度评估。本次 20 worker 全量运行约 15 分 16 秒；不声称这是等算力比较。

## 如何选参数

沿用 `scripts/diagnose_history_estimators.py` 的历史留出协议：seed 1 的
100k/400k/1M，各两个外层划分，按 raw 空间的加权留出 NLL 选择；先保存候选
配置与 SHA256，再检查 seed 2～5，最后才计算当前全量参考 KL。
NF 内部的空间留出与外层按评估身份分组的留出是两层不同用途的验证。

所有阶段只使用每个 checkpoint 原已抽到的 100 条记录，不增加历史标签预算。
包含当前 checkpoint，时间权重仍为 0.9 的步数折扣；NN 的标签协议没有改。

本次探索过程完整保留：

| 输出目录（logs/ 下） | 结论 |
|---|---|
| `regularized_flows_search_v1` | NF noise=0.2 获选；FM 早停未超过原配置 |
| `fm_regularized_search_v2` | 单模型 1000 次加噪在 seed 1 获选，但跨 seed 不稳健，未采用 |
| `fm_ensemble_search_v1` | 三成员、noise=0.1 获选，通过跨 seed 检查 |

NF 在 seed 1 的外层留出 NLL 从 -2.0750 改善为 -2.4757。
seed 2～5 的 100k 留出 NLL 从 -0.2728 改善为 -1.4377；1M 则从 -1.9020
略变差为 -1.8322，表明收益并非所有阶段都一致。

FM 集成的 seed 1 留出 NLL 从 -2.4415 改善为 -2.4719；seed 2～5 的 100k 从
-1.6689 改善为 -2.0106，1M 从 -1.8021 改善为 -1.8035。

这些是迭代探索结果。原五个 seed 的曲线已看过，不把它们称为从未接触过的盲测；
多轮候选搜索的记录保留，不能只展示胜者。参数未按待评估全量参考的 KL 排名。

## 全量验证

正式输出：

- `logs/nf_regularized_all100_5x4/`
- `logs/fm_ensemble_all100_5x4/`

各自均覆盖 5 seed × 100 checkpoint，各 498 有效、2 条样本不足跳过、0 错误。
两条跳过均为 seed 2 的 10k/20k，与旧对应方法一致。
NF 有 12 个 checkpoint 提醒最佳验证轮数触及 300 次上限，未删除这些结果；FM
集成无拟合质量提醒。NF 的 996 个有效单侧拟合中，984 个使用验证选预算，12 个
使用稀疏样本预定短预算，全部最终使用各侧的全部成功记录重拟合。

每条有效记录均核验 MC 为 10000 × 5，所有 MC 样本保留；历史记录数为
`checkpoint/10000 × 100`，参考记录数为 676。每种新方法各 500 份任务 JSON
完整。与先前小规模检查重叠的有效 KL 逐点一致（最大绝对差 0）。

| 方法 | 有效配对点中 KL 降低 | 1M 旧 KL | 1M 新 KL | 1M 降幅 |
|---|---:|---:|---:|---:|
| NF → nf_reg | 480 / 498 | 0.098281 | 0.058735 | 40.2% |
| FM → fm_ensemble | 397 / 498 | 0.038921 | 0.034050 | 12.5% |

上述 1M 数值为五个 seed 的未平滑单点均值。每个 seed 的全程平均 KL 都下降。
按阶段汇总的配对均值如下，不能用早期巨大异常点的下降替代中后期比较：

| 步数范围 | NF 旧 → 新 | FM 旧 → 新 |
|---|---|---|
| 10k～100k | 3.7818 → 0.9394 | 8.2077 → 2.7794 |
| 110k～400k | 0.7720 → 0.2918 | 0.3331 → 0.2955 |
| 410k～1M | 0.2224 → 0.1158 | 0.0801 → 0.0678 |

这证明在当前离线协议和五个 seed 上，新方法的历史估计与各自全量参考更接近。
新方法的参考分布也随正则化/集成改变，不能把全部 KL 降幅解释为对共同真实分布
的精度提升。KDE legacy 仍有口径差异，也不能据此声称已经与 KDE 等精度。

## 图、测试与旧结果保护

新版九曲线图在 `plots/fig6_with_improved_flows/`：原七条曲线、NF 新版和 FM 集成
同时展示。旧曲线原始值逐点一致，旧结果与上一版图记录的 SHA256 相符，旧图未
覆盖。平滑版、未平滑版、完整范围图、配对 CSV、分阶段 CSV 与 `validation.json`
均已保存；保持逐 seed 平滑后汇总和 bootstrap 95% CI。

逐个文件执行的测试共 75 项通过：

| 文件（tests/evaluate/ 下） | 通过数 | 关键覆盖 |
|---|---:|---|
| `test_regularized_flow.py` | 12 | 禁用改动与旧版等价、分组隔离、最优轮数、全数据重拟合、非有限值 |
| `test_flow_ensemble.py` | 5 | 算术混合密度、匹配的采样、单成员等价、独立初始化 |
| `test_continuous_estimators.py` | 13 | raw/standardized 密度接口、Jacobian、采样、随机流、自身 KL |
| `test_offline_protocol.py` | 12 | 两侧同类且独立、历史/全量记录边界、MC、续跑、跳过 |
| `test_curriculum_estimators.py` | 16 | 真实 Push + 16 步 SAC 更新、XYZ 平面适配、reset 和样本不足回退 |
| `test_plot_fig6_sources.py` | 6 | 新旧曲线同时保留、来源与协议检查 |
| `test_history_validation.py` | 5 | 历史抽样、分组留出、权重、候选覆盖、不按 KL 选参 |
| `test_all100_launcher.py` | 6 | 调度、锁、配置与覆盖率约束 |

这完成了离线比较与接入验证，16 步 SAC 仅验证接口和优化链路，不代表已验证
长期课程学习收益。后续真实 Push/SAC 训练可通过原 estimator_config 接口选择
`nf_reg` 或 `fm_ensemble` 并传入上述正式参数，无需另写一套估计器。

## 重跑命令

以下全量结果已经完成，无需再跑；续跑遇到代码/配置哈希变化时请另选输出目录。

```bash
cd /home/tacmon/workspace/ex_SSD/gc_ope_refactor
conda run --no-capture-output -n gc_ope python scripts/run_push_all100.py \
  --methods nf_reg --config configs/evaluate/push_nf_regularized_all100.json \
  --output logs/nf_regularized_all100_5x4
conda run --no-capture-output -n gc_ope python scripts/run_push_all100.py \
  --methods fm_ensemble --config configs/evaluate/push_fm_ensemble_all100.json \
  --output logs/fm_ensemble_all100_5x4
```

绘图命令（已有输出目录禁止覆盖；复画时另选目录）：

```bash
conda run --no-capture-output -n gc_ope python scripts/plot_fig6.py \
  --nn-result-root logs/nn_logloss_v2_all100_5x4 \
  --gmm-em-result-root logs/gmm_em_all100_5x4 \
  --gmm-em-reg005-result-root logs/gmm_em_reg005_all100_5x4 \
  --nf-reg-result-root logs/nf_regularized_all100_5x4 \
  --fm-ensemble-result-root logs/fm_ensemble_all100_5x4 \
  --output plots/fig6_with_improved_flows
```

## 复核与等待记录

没有将全部候选的 KL 降低都视为成功；两个未通过跨 seed 检查的 FM 方案保留为
研究记录，没有替换旧 FM。保留所有失败候选、样本不足项、预算提醒和未改善点。
关键限制是有限 seed、历史分布内部留出、参考分布变化，以及集成额外算力。

长任务采用单独监控 subagent。主会话在等待阶段只续等同一监控，不查询任务或
做其他工作；收到结束报告后再核对原会话退出码和结果清单。三个监控分别经历
1、4、12 次等待窗口超时后收到 PROCESS_EXITED，均无监控错误；超时未被当作
实验结束。token/费用节省未测量。
