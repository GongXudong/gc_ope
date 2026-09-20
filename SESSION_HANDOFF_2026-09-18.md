# 会话交接文档（Handoff）
## OPE 密度估计实验：dev → exp/gmm-prediction-my-push-slide 全部变更记录 + 新环境工作计划

> 生成于 2026-09-18。本会话因工作环境转移而终止，本文档供新环境快速恢复。
> 终止状态：4000-job sweep（PID 67728）**已按用户要求废弃并终止**，用户随后给出了新的工作计划（第五节），
> 旧 sweep 的产出不作为新计划的交付物，但已产生的 2321 行 CSV 可留作参考或丢弃。

---

# 一、Git 状态

- 仓库：`/home/tacmon/workspace/ex_SSD/gc_ope`
- 当前分支：`exp/gmm-prediction-my-push-slide`
- 分叉点（与 dev 的 merge-base）：`e75391f`
- 分支上 4 个提交：

| 提交 | 说明 |
|------|------|
| `4c72719` | weighted-resampled GMM evaluator |
| `23e45ce` | P0/P0.5 replacement estimators（Histogram/Gaussian/NN/FlowMatching）+ 实验框架 |
| `f2898e4` | 范围限定 push/slide；oracle 截断修复（`ORACLE_MAX_REFERENCES=200000`）；plan 更新 |
| `d550ba9` | fig5/fig6 六方法对比图脚本 |

# 二、代码修改（相对 dev 分叉点）

## 已提交

- `4c72719`：新增 `src/gc_ope/evaluate/evaluator_gmm.py`（weighted-resampled GMM）、`scripts/evaluate_gmm_prediction.py`、对应测试。
- `23e45ce`：
  - 新增估计器 `src/gc_ope/evaluate/evaluator_replacements.py`（Histogram、Gaussian、NN=WeightedMLP 分类器、FlowMatching）；
  - 新增学习型估计器 `src/gc_ope/evaluate/evaluator_learned.py`（419 行，NN/FM 共用）；
  - 实验脚本：`scripts/replacement_run_job.py`（单 job runner，340 行）、`replacement_run_all.py`、`replacement_run_parallel.py`；
  - 绘图：`scripts/plot_kde_gmm_comparison.py`（含 `monte_carlo_kl`）、`scripts/plot_replacement_comparison.py`；
  - 三套测试（共 435 行）；`AGENTS.md`（41 行）；`claude-code-plan.md`（1797 行）。
- `f2898e4`：`replacement_run_job.py` 加入 oracle 成功点 >200000 时截断逻辑（前 N 个，非随机）；协议限定 push/slide。
- `d550ba9`：`scripts/make_density_grid_6methods.py`、`scripts/plot_replacement_fig5_fig6.py`（论文风格 fig5/fig6）。

## 工作区未提交（需一并带入新环境或决定去留）

| 文件 | 内容 |
|------|------|
| `AGENTS.md`（modified） | 追加"环境线程约束"节：i9-14900K 32 核/32GB、OpenBLAS DYNAMIC_ARCH MAX_THREADS=64、OMP/OPENBLAS/MKL_NUM_THREADS=1 约束、worker 数建议 |
| `scripts/plot_kde_gmm_comparison.py`（modified） | `monte_carlo_kl` 的 print 加 `BrokenPipeError` 防护（ProcessPool 子进程管道关闭时不崩） |
| `scripts/replacement_run_all100.py`（未跟踪，新） | 全量 100-checkpoint 补跑调度器：ProcessPoolExecutor + `fcntl.flock` 并发写保护 + `--skip-done` 断点续跑。注意：文件内注释写 ThreadPoolExecutor，实际实现是 ProcessPool（注释错误，未提交） |
| `scripts/plot_replacement_4methods_paper_style.py`（未跟踪，新） | 4 方法论文风格 fig5/fig6 绘图脚本，头部 docstring 记录了重要口径结论（见第五节 Q4），并生成了 `density_cache/`（见第四节） |
| `正式运行前Preflight.md`（未跟踪，新） | 15 项 Preflight 检查清单（本会话已全部完成，检查结论见附录） |

# 三、实验协议与参数（Preflight 已确认，未改动）

```
kappa = 0.9                      历史成功点时间折扣（kappa^Δt/10000）
KDE bandwidth = 0.2              oracle 与 estimator 两侧相同
GMM: 5 分量, reg_covar=1e-6, resample_size=1000
Histogram: 10 bins；Gaussian: reg_covar=1e-6
MC: 100000 samples × 5 repeats
random_state = 0
checkpoint 序列: [10000+10000*i for i in range(99)] + [1000000]（100 个）
```

核心口径（Preflight 15 项逐项核对通过）：

- **Oracle**：当前 checkpoint `t` 自己的评估 CSV 中 `termination=="reach target"` 的成功目标 (x,y)，拟合 KDE(bw=0.2)；**不是历史数据**。
- **历史数据无未来泄漏**：estimator 只见 `checkpoint < t` 的成功目标。
- **KL 方向**：`D_KL(oracle ‖ estimate)`，MC 采样来自 oracle，log 密度在 raw (x,y) 空间，两侧 Jacobian 一致。
- **Support filtering**：oracle 样本落在 estimate 支撑外（log q = −inf）时丢弃该样本并打印丢弃数（`MC-KL dropped N/100000`）；即实际计算的是**条件在 q>0 子集上的 KL**。
- **oracle 截断**：`ORACLE_MAX_REFERENCES=200000`，>200000 时取前 N 个（`head`，非随机）；当前 push/slide 单 checkpoint 成功点远小于此，未触发。
- **早期 checkpoint**：无历史文件/无历史成功点 → `skipped:*`（确定性结果，`--skip-done` 视为已完成）。
- **输出安全**：CSV append + `fcntl.flock`；`--skip-done` 只跳过 `ok`/`skipped:*`，`error` 行会重跑；`(task,seed,ckpt,method)` 唯一定位。
- **并行**：ProcessPool，`--workers 4`，每个 worker `OMP/OPENBLAS/MKL_NUM_THREADS=1`。i9-14900K 32 核/32GB，吞吐约 200–250 job/h。
- **数学 sanity**：KL(oracle‖oracle) 实测 ≈ 0（1e-5 量级，通过）。

# 四、项目物资 / 运行记录（全部在 .gitignore 下，需手动搬运）

## 必须随仓库一起搬（`.gitignore` 覆盖，不在 git 里）

| 路径 | 内容 | 备注 |
|------|------|------|
| `checkpoints/`（**1.9T**） | 全部 RL 模型 (.zip)、replay buffer (.pkl)、各 checkpoint 评估 CSV（`rl_model_*_steps_eval_res_on_fixed.csv`，项目里 77018+24894+14917 个） | 项目体积全部在此。只搬评估 CSV 可省掉大部分空间（.zip/.pkl 在新计划里不需要，可只搬 `*_eval_res_on_fixed.csv`） |
| `logs/replacement_experiment/kde_gmm_hist_gaussian_kl.csv` | 旧 P0 实验 584 行 | 覆盖 4 task（push/slide/reach/vvc）× 5 ckpt（100k/300k/500k/700k/1000k）× 6 method（kde/gmm/histogram/gaussian/nn/flow_matching）。**新计划只需要其中 push/slide 的 nn、flow_matching 行**；NN/FM 模型未保存（见下） |
| `logs/replacement_experiment/test_pipe.csv`、`plots/` 子目录 | 管道测试、上一轮 fig5/fig6 图（PDF/PNG）、`summary_table.csv`、`density_heatmap_6methods.csv`、`conclusion*.json` | 参考用 |
| `logs/replacement_experiment/all100/kde_gmm_hist_gaussian_kl.csv` | 已废弃 sweep 输出，2321 行（ok=2293, skipped=28, err=0） | push 侧 5 seed × 100 ckpt × 4 方法**已全部完成**；slide 停在 seed1 约 800k。用户已废弃，可留作参考 |
| `logs/replacement_experiment/all100/kde_gmm_hist_gaussian_kl copy.csv` | 用户手动快照（push 完成时刻，2120 ok） | |
| `logs/replacement_experiment/all100/plots_fig5_fig6_4methods/` | 4 方法 fig5/fig6 图 + `archive_4methods_with_histogram/` | |
| `logs/replacement_experiment/all100_run*.log` | 历次 sweep 启动日志（run3 为已废弃那次） | |
| `logs/replacement_experiment/all100/plots_fig5_fig6_4methods/density_cache/`（**25 个 npz**） | 40000 点网格上的密度值缓存，文件命名 `{task}_seed{s}_ckpt{c}_{method}_k0.9_bw0.2_gmm5x1000_hist10_rs0_grid40000_-0.2500_0.2500.npz`，内含 `density`（float64, 40000,） | **只覆盖 push/seed2 的 100k/150k/210k/250k/310k × 5 method（kde/gmm/gaussian/histogram/oracle_ref）= 25 个**。注意：这是**画图用的网格密度采样结果，不是 KDE/GMM 估计分布本身**（不含核位置、均值/协方差等模型参数），且仅这一个 seed 的一小段 ckpt。NN/FM/NF 也没有 |

## 无需搬运（本地痕迹）

- `f16trace.csv`（0 字节）、`plots_sweep.log`（0 字节）
- `outputs/`（6.4M，2025-11 旧实验输出）、`paper/`（38M，论文材料；若要在新环境出图可保留）

## 关键数据文件结构

- 评估 CSV：`checkpoints/{my_push|my_slide}/sac/seed_{1..5}/rl_model_{10000..990000,1000000}_steps_eval_res_on_fixed.csv`
  - 列：`x,y,z,length,termination,achieved x,achieved y,achieved z,cumulative_rewards,discounted_cumulative_rewards`
  - 目标空间用 **(x,y)**；`termination=="reach target"` 为成功
  - **NN 分布的离散化取法**：对该 CSV 中所有行（成功+失败）的 (x,y) 去重取离散点，逐点算 NN 概率
- 成功目标 = 评估 CSV 中 `termination=="reach target"` 的 (x,y)。

# 五、新工作计划（用户 2026-09-18 口述，本文档的"遗嘱"主体）

## 0. 背景

- KDE 在 my_push/my_slide sac seed1–5 上的拟合分析**论文投稿时已做过**（KDE 结果已在项目里）。
- 用户亲自对比过各方法后，最终确认保留 5 种对比方法：**KDE + GMM + NN + Flow Matching + Normalizing Flow**。**Histogram 与 Gaussian 已确认为要删除/废弃的实验对象，后续不再考虑**（旧 sweep 产出的 histogram/gaussian 行及对应代码路径在新计划中删除或废弃，新调度器、出图、KL 汇总一律不再包含这两种方法）。
- **最重要产出：fig6**（各估计器对 p_ag 的估计误差随训练进度的曲线，即 KL(oracle‖estimate) vs checkpoint）。fig5（估计分布上取点 vs 真实分布的散点对比）也顺带容易做。
- 核心思路（用户原话）：**"有了 csv，就能干什么都行了"** —— 每个估计器在每个 checkpoint 上的"估计分布"都落成 CSV，之后绘图/分析都是后处理。

## 1. 目标：为 5 种方法各产出"每 checkpoint 估计分布" CSV

覆盖范围：**至少 my_push/sac/seed 1–5 全部 100 个 checkpoint**（10k–990k 步 10k + 1000k）；my_slide 同规格（用户说"至少 push"，slide 按同样结构建议一起做，否则 fig6 只有 push 面板）。

| # | 方法 | 状态 |
|---|------|------|
| 1 | KDE 估计分布 | "师兄说他算过了"——**但在当前可查的磁盘上找不到任何保存了 KDE 原始估计分布（核位置、带宽、样本点等模型本身）的文件**。注意：`logs/replacement_experiment/`、`logs/gmm*` 等目录都是本 exp 分支工作之后创建的，里面只有 KL 数值记录和画图用的网格密度采样（`density_cache/`，且只有 push/seed2 一小段），**不能当作"师兄算过的 KDE 分布"的证据**。需要向师兄要，或全部重算落 CSV（KDE 单 job 实测 364s @500k，4 worker 下 100 ckpt/seed 约数小时） |
| 2 | GMM 估计分布 | 本分支的产物里**只有每 checkpoint 拟合后的 KL 记录（旧 P0 的 584 行 CSV 与 all100 CSV 的 `kl` 列），GMM 分布本身（5 个分量的均值/协方差/权重）从未单独保存** → 需重算落 CSV（重算廉价：GMM 单 job 实测 ~5–7s） |
| 3 | NN 估计分布 | **NN 正确实现尚未写好**（用户确认）；旧 `evaluator_learned.py` 的 WeightedMLP 实现未经验证、不可直接复用，旧 P0 的 100 行 NN 结果是"跑完即弃"、模型文件不存在（全盘无 .pt/.pth/.joblib）。需按第 4 节协议（goal→P(success) 分类器，保留 softmax、对离散化 goal 算概率、隐式均匀先验、归一化转分布）**新写正确的 NN 代码**并逐 checkpoint 落 CSV |
| 4 | Flow Matching 估计分布 | **Flow Matching 正确实现尚未写好**（用户确认）；旧代码 `FlowMatchingDensityEvaluator`（fm_epochs=80, fm_hidden=16, fm_samples=2000）未经验证、不可直接复用 → 需新写正确代码并逐 checkpoint 落 CSV |
| 5 | Normalizing Flow 估计分布 | **代码库中完全没有实现**（`evaluator_learned.py` 只有 WeightedMLP=NN 和 FlowMatching；"FlowMatching" 指 flow matching 生成式估计，不是 normalizing flow）→ 需新写估计器 |

> **总体状态**：NN / Flow Matching / Normalizing Flow 三者的正确实现**当前都没有写好**，需要从零验证或重写；KDE、GMM 的估计分布需重算落 CSV（KDE 可能要向师兄要结果）。

## 2. 统一 CSV schema（建议，新环境按此执行）

每个方法 × 每个 task/seed 一个 CSV，列：

```
task, seed, checkpoint, method, x_grid, y_grid, log_density
```

- 网格：固定 40×40 = 1600 点（或沿用 density_cache 的 40000 点 = 200×200，取 200×200 以保持与旧缓存可比）；
  范围沿用 cache 命名里的 `-0.25..0.25`（需在新环境对 push/slide 各自坐标域核实后定；push 的 x,y 域以评估 CSV 的 achieved 列为准）
- `log_density`：估计器在网格点上的 log 密度（raw (x,y) 空间）；
- 行 = 100 ckpt × 1600/40000 网格点；
- **文件组织：一算法一轨迹一 CSV**（`{method}_{task}_seed{seed}.csv`，见第 3 节脚本组织要求），不得多算法/多轨迹混写同一 CSV；
- 另出一张**汇总 CSV**（等价旧 P0 的 kl 列）：`task,seed,checkpoint,method,kl,kl_seed_std,kl_seed_mean, historical_successes, reference_successes, status`，fig6 直接画它。

## 3. 计算脚本组织要求（用户补充，新环境必须遵守）

- **禁止**把多个算法 / 多条训练轨迹的结果混进同一个 CSV（旧 sweep 的教训：4 方法 × 多轨迹 × 多 ckpt 全写在一张 CSV 里）。
- 每个估计器 × 每条训练轨迹（如 `checkpoints/my_push/sac/seed_1/` 这一条轨迹）一个独立 CSV，命名含 method + task + seed。
- 单轨迹脚本示例：`python scripts/fit_gmm_single.py --task push --seed 1`，该脚本对这条轨迹的 100 个 checkpoint 全部跑完并写出 `logs/.../gmm_push_seed1.csv`。
- 算法内部的多 checkpoint 拟合/采样需支持**多核并行加速**（单脚本内多线程/多进程；沿用 `OMP/OPENBLAS/MKL_NUM_THREADS=1` 单 worker 约束，worker 数自行调优，不占满 32 核）。
- 汇总级脚本（如 fig6 用）读各单轨迹 CSV 合并出图，不直接生成混合结果文件。

## 4. NN 协议（用户明确）

- 输入：历史评估成功/失败 (x,y)（checkpoint < t 的评估 CSV，κ=0.9 折扣同 KDE/GMM）；
- 模型：goal → P(success) 分类器（旧实现 `WeightedMLPDensityEvaluator`：MLP，hidden=16，epochs=100，lr=1e-3，加权采样，`fit_evaluator` 返回四元组）；
- **不做最后取整**，保留 softmax 倒数第二步 → P(goal 完成概率)；
- 离散点取法：对该 checkpoint 评估 CSV（`best_model_eval_res_on_fixed.csv` 同款 on_fixed 文件）中所有 (x,y) 去重，逐点算 P；
- 转分布：P(点) ∝ P(goal) × 均匀先验，归一化后落 CSV（每点权重 = P/ΣP，即"均匀网格先验 + 成功率似然"的乘积，用户已指出这隐含均匀先验）；
- 旧 P0 CSV 里 NN 行（100 行，4 task）可作对照参考，口径若不一致以用户本段描述为准。

## 5. 已知口径陷阱（旧脚本 docstring 已记录，新计划必须避开或声明）

`scripts/plot_replacement_4methods_paper_style.py` docstring 结论：

- **Histogram 的 MC-KL 是条件 KL**：box 密度在 oracle 支撑外为 0，`monte_carlo_kl` 丢弃 q=0 样本后在子集上求均值，可出现系统性负值（最负 −3.09），无下界。新方法（NN 概率图/FM/NF）若也出现 0 支撑外样本，要么同样丢弃并声明口径，要么用 `log(softmax+ε)` 平滑。**5 个新方法之间必须用同一口径的 KL**。
- 本实验是 **vanilla SAC 无课程**，旧 fig5 黄点（课程采样器 behavioral goals）画不出来，只能用"该 checkpoint 成功目标"做参考散点——只支持"拟合误差"结论，不支持课程采样结论。fig5 若画，必须标注单 seed 口径。
- 旧 sweep 吞吐参考：KDE @500k 单 job 364s；4 worker 全速 200–250 job/h。

## 6. 恢复执行顺序（新环境 checklist）

1. [ ] 搬代码：git 仓库 + 工作区 5 个未提交文件（第二节的"未提交"表）+ `checkpoints/` 下**只需 `*_eval_res_on_fixed.csv`**（可省 1.8T，只留评估 CSV 约 5–10GB）
2. [ ] 搬数据：旧 P0 CSV（584 行，留 push/slide 的 nn/fm 行做对照）、density_cache 25 个 npz（对照用）
3. [ ] 写统一出图脚本前，先写 **5 个估计器 × 单 checkpoint 的 smoke test**（push/seed1/100k，参考 Preflight 第 12 项流程）
4. [ ] 写 Normalizing Flow 估计器（`evaluator_learned.py` 增加 `NormalizingFlowDensityEvaluator`，或新文件；建议 2D 小 MLP-Radial/RealNVP，参数量与 GMM 同量级，random_state=0）
5. [ ] 写新调度器：**5 方法（kde/gmm/nn/flow_matching/normalizing_flow，不含 histogram/gaussian）** × 2 task × 5 seed × 100 ckpt = **5000 jobs**，输出统一 schema（第 2 节，按"一算法一轨迹一 CSV"，见第 3 节脚本组织要求），`--skip-done` + flock（照抄 `replacement_run_all100.py` 模式，但结果文件必须按算法 × 轨迹拆分，不混写）    **人工注：也就是说希望调度器大概是 python 调度器.py --算法 --目标训练轨迹如~/checkpoints/my_push/sac/seed_1这种 --num_worker表示并行加速因为在seed_1这种训练轨迹上面是没有后效性的不需要一定算完300000才能u算310000,   至于其他可选--参数你可以自由发挥大概是这样）**
6. [ ] 先跑 push（2500 jobs，估算 2–3 天 4-worker），出 fig6 push 面板；再决定 slide 是否全量
7. [ ] fig6：`sns.lineplot`，x=训练进度%，y=KL（oracle‖estimate），一条线一个方法，errorbar=seed 间 95% CI（旧图 `plots/p_ag_dist_between_truth_and_estimated_in_training/my_push/plot_omega.ipynb` 的画法，MD5 已与论文图核对）

## 7. 待确认问题（新环境开工前问用户）

- **Q1**：KDE 分布——师兄说"都计算过了"，但当前磁盘上找不到他保存的原始估计分布文件（本分支的 `density_cache` 只是画图网格采样，且只有 push/seed2 一段）。是向师兄要结果，还是直接重算落 CSV？
- **Q2**：slide 是否也要 5 方法全量 100 ckpt？（用户说"至少 push"，默认只做 push 的话 fig6 只有单任务面板）
- **Q3**：NN 的历史折扣 κ=0.9 是否与 KDE/GMM 完全一致？（旧实现 `WeightedMLPDensityEvaluator._build_targets` 用的是历史成功/失败点加权，需核对折扣项）
- **Q4**：NN 的"离散点取法"用 `on_fixed` 还是 `on_random` 评估 CSV？（旧 P0 全部用 on_fixed；用户举例写的是 best_model 的 on_fixed）
- **Q5**：Normalizing Flow 的结构/训练轮数没有历史协议，是否接受提议（2D RealNVP，10 blocks，epochs 同 FM=80）？
- **Q6**：FM 旧协议（fm_epochs=80, hidden=16, samples=2000）是否沿用？
- **Q7**：网格点数 200×200=40000（与 density_cache 一致）还是 40×40=1600（画图够用、快 60 倍）？

# 附录 A：Preflight 15 项结论摘要（本会话完成）

1. 协议 = 2 task × 5 seed × 100 ckpt × 4 method = 4000 jobs ✓（脚本注释曾误写 101/4040，实际 100/4000）
2. Oracle = 当前 ckpt 自身评估 CSV 成功目标 KDE(bw=0.2)，非历史 ✓
3. 无未来泄漏：历史文件严格 `< t` ✓
4. 4 方法同一历史输入（kappa=0.9、同 success 筛选、同 (x,y)）✓
5. 参数在 CLI 默认值与执行路径一致（kappa=0.9, bw=0.2, GMM 5/1e-6/1000, MC 100000×5, rs=0）✓
6. oracle bw=estimator bw=0.2 ✓（方法学上两边等带宽偏乐观，Preflight 只记录不改）
7. KL 方向 = D_KL(oracle‖estimate)，raw 空间 + Jacobian 一致；support filtering = 丢弃 q=0 样本（条件 KL，已声明）✓
8. ORACLE_MAX_REFERENCES=200000，取前 N 个（head，非随机），当前数据量未触发 ✓
9. 早期 ckpt 无历史 → skipped（确定性）✓
10. append+flock+唯一键+skip-done 语义正确（error 行重跑，ok/skipped 不重跑）✓
11. 4 worker + 单线程 BLAS，无 oversubscription，32GB 内存足够 ✓
12. smoke test（push/seed1/100k 四方法）通过：status=ok、kl/fixed_grid_kl finite、hist/ref>0、schema 正确 ✓
13. 一致性：4 方法 historical_successes 相同、oracle 样本数 = reference_successes ✓
14. KL(oracle‖oracle) ≈ 0 ✓
15. 最终统计未做（sweep 在 2321/4000 时按用户要求终止）

# 附录 B：终止时快照（2026-09-18）

```
all100 CSV: 2321 行  ok=2293  skipped=28（push 每 seed 4 条 + slide s1 8 条，早期 ckpt 无历史成功点）  err=0
push:  5 seed × 100 ckpt × 4 method 全部完成（2000 行）
slide: seed1 推进至约 800k（seed2–5 未开始）
主进程 67728: 已 kill -9；监控全部停止
```

# 附录 C：环境

- conda env `gc_ope`；i9-14900K 32 核 / 32GB；OpenBLAS DYNAMIC_ARCH MAX_THREADS=64
- 运行约束：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`（每 worker）
- git 身份：GongXudong；提交归属行按 harness 要求（Co-Authored-By: Claude Opus 5）
