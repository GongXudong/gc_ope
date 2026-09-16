# Goal Distribution 替换实验

> **当前状态：** 进行中  
> **优先级：** 高  
> **截止时间：** 约 3 天后  
> **主要目的：** 完成审稿人要求的 distribution estimation replacement / robustness experiment  
>
> **范围调整（2026-09-16）：** 时间不足，**核心范围收窄为 push/slide 两个 2D 场景**（各 5 seed × 5 checkpoint，4 传统估计器 + NN/Flow Matching 全量完成，300 行零 error）。reach/vvc 部分完成（4 传统估计器中 reach/vvc 的 KDE 各缺 9/12 个 (seed,ckpt)，原因是这些 checkpoint 历史成功目标 >17 万，sklearn KDE 在 3D 上 fit 达 4 分钟/job、MC-KL 小时级；NN/FM 在 reach/vvc 已全量完成）。reach/vvc 的 KDE 缺失部分**留待后续补跑**，不影响 push/slide 的最终结论与交付。

---

# 1. 研究背景

本项目研究多个强化学习任务，并以 **desired goal space** 中的历史 goal 数据作为研究对象。

当前实验使用 **KDE（Kernel Density Estimation）** 对历史 goal 数据进行分布拟合。

历史样本按照时间距离当前时刻的远近进行指数衰减加权：

\[
w_i=\gamma^{\Delta t_i},
\qquad
\gamma=0.99
\]

其中：

- \(w_i\)：第 \(i\) 个历史样本的时间权重；
- \(\Delta t_i\)：该样本距离当前时刻的时间间隔；
- \(\gamma=0.99\)：时间衰减系数。

整体流程可以概括为：

```text
历史 goal 数据
      │
      ▼
时间衰减权重
w_i = 0.99^(Δt_i)
      │
      ▼
Goal Distribution Estimation
      │
      ▼
下游实验 / Evaluation
```

当前实验只使用 KDE 作为主要 distribution estimator。

审稿人要求增加**替换实验（replacement experiment）**，因此需要验证：

> **如果保持时间衰减机制以及其他实验流程不变，仅替换 goal distribution 的拟合方法，最终实验结论是否仍然成立？**

因此，本阶段工作的核心不是寻找“最好的 density estimator”，而是验证当前研究结论对于 **distribution estimation method** 的鲁棒性。

---

# 2. 核心实验原则

这是整个实验最重要的原则：

> **只替换 density estimator，不改变 temporal weighting mechanism 和 downstream experimental pipeline。**

所有方法都应该尽可能使用完全相同的：

\[
\boxed{w_i=0.99^{\Delta t_i}}
\]

以及：

- 相同的原始数据；
- 相同的数据预处理；
- 相同的数据划分；
- 相同的时间权重；
- 相同的随机种子 / repetition 设置；
- 相同的 downstream evaluation；
- 相同的评价指标。

核心变化应该只有：

\[
\boxed{\text{Density Estimation Method}}
\]

不要把本次 replacement experiment 改造成 classification / success prediction 问题。

当前研究对象仍然是：

\[
\boxed{p(g)}
\]

即 desired goal space 中的经验分布。

---

# 3. 当前已经完成的方法

## 3.1 KDE

当前已有的主要方法是 **KDE**。

其形式可以表示为：

\[
\hat p(g)
=
\frac{1}{\sum_iw_i}
\sum_i
w_iK_h(g-g_i).
\]

KDE 是当前实验的原始方法，应作为所有 replacement experiment 的 reference。

**不要在没有必要的情况下修改当前 KDE 实现。**

---

## 3.2 基于重采样模拟带权样本的 GMM

目前已经完成一种 GMM replacement。

当前实现的基本思路是：

1. 计算时间衰减权重：

\[
w_i=0.99^{\Delta t_i}
\]

2. 根据归一化后的 \(w_i\) 对原始样本进行重采样；
3. 得到近似服从时间加权经验分布的样本集；
4. 在重采样后的数据上运行普通 GMM。

整体流程：

```text
原始样本
   │
   ▼
时间衰减权重
   │
   ▼
按照权重重采样
   │
   ▼
重采样后的数据集
   │
   ▼
普通 GMM
```

该方法应称为：

> **Resampling-based weighted GMM**

注意：

**不要把它直接描述成严格的 Weighted-EM GMM。**

两者需要明确区分：

### Resampling-based GMM

通过：

\[
w_i
\rightarrow
\text{sampling probability}
\rightarrow
\text{resampling}
\]

间接实现时间加权。

### Weighted-EM GMM

直接将：

\[
w_i
\]

纳入 GMM 的 EM 参数估计过程。

当前已经完成的是前者。

---

# 4. 后续需要实现的 Replacement Methods

## P0：必须完成

### 4.1 Weighted Histogram / Grid Density

这是优先级最高的新增 replacement method 之一。

将 goal space 划分成二维或三维 grid / bins，然后使用时间衰减权重统计每个区域的概率：

\[
P(B_j)
=
\frac{
\sum_{i:g_i\in B_j}w_i
}{
\sum_iw_i
}.
\]

其重要意义在于：

> Histogram 完全不依赖 KDE 的 kernel smoothing。

因此可以用于验证：

> 当前结论是否只是 KDE smoothing 带来的 artifact。

需要根据 goal space 的实际维度使用 2D 或 3D histogram。

---

### 4.2 Multivariate Gaussian

使用单个多元 Gaussian 拟合 goal distribution：

\[
p(g)=\mathcal N(g;\mu,\Sigma).
\]

其中均值采用时间权重：

\[
\mu=
\frac{\sum_iw_ig_i}
{\sum_iw_i}.
\]

协方差采用：

\[
\Sigma=
\frac{
\sum_iw_i(g_i-\mu)(g_i-\mu)^T
}{
\sum_iw_i
}.
\]

该方法是一个非常简单的参数化 baseline。

它主要用于回答：

> 如果不使用 KDE 的非参数结构，也不使用 GMM 的 mixture structure，仅使用一个简单的多元 Gaussian，实验结论是否仍然保持？

---

## P1：强烈建议完成

### 4.3 真正的 Weighted-EM GMM

在当前 **Resampling-based GMM** 的基础上，进一步考虑实现真正的 sample-weighted GMM / Weighted-EM。

核心目标是直接将：

\[
w_i=0.99^{\Delta t_i}
\]

纳入 GMM 参数估计，而不是通过随机重采样近似。

这样可以形成：

```text
Resampling-based GMM
        VS
Directly Weighted GMM / Weighted-EM
```

这个实验不是必须自己从零实现。

优先检查当前 Python 环境和已有依赖中是否存在可靠的 weighted GMM 实现；如果可以较低成本实现，则加入。

**不要因为 Weighted-EM 而进行大规模无关重构。**

如果实现成本明显过高，应优先保证 P0 实验完整。

---

## P2：有时间再做

### 4.4 KNN Density Estimation

可以考虑 KNN-based density estimation。

其思路与 KDE 不同：

- KDE：固定 bandwidth；
- KNN density：固定邻居数量，通过邻域体积估计 density。

因此它可以作为另一种非参数 density estimator。

但它不是当前必须完成的方法。

---

### 4.5 Bayesian GMM / DP-GMM

可以考虑：

- Bayesian Gaussian Mixture；
- Dirichlet Process GMM。

主要作用是减少对 mixture component 数量 \(K\) 的人为指定。

这是额外 robustness experiment。

如果时间不足，直接跳过。

---

# 5. 不同任务的 Goal Space

当前至少涉及以下任务：

## Reach

\[
g=(x,y,z)
\]

真实的 3D goal space。

使用：

> **3D density estimation**

---

## Push

\[
g=(x,y,0.02)
\]

其中：

\[
z=0.02
\]

是固定值。

因此其本质 goal space 是：

\[
g=(x,y)
\]

使用：

> **2D density estimation**

不要把固定的 \(z=0.02\) 当成一个真正的随机维度。

---

## Slide

\[
g=(x,y,0.02)
\]

同样：

\[
z=0.02
\]

是固定值，因此本质上也是：

\[
g=(x,y)
\]

使用：

> **2D density estimation**

---

## VVC

\[
g=(v,\mu,\chi)
\]

真实的 3D goal space。

使用：

> **3D density estimation**

---

## 总结

| Task | Goal | 实际维度 |
|---|---|---:|
| Reach | \((x,y,z)\) | 3D |
| Push | \((x,y,0.02)\) | 2D |
| Slide | \((x,y,0.02)\) | 2D |
| VVC | \((v,\mu,\chi)\) | 3D |

因此最终应遵循：

```text
Reach  → 3D density estimation
Push   → 2D density estimation
Slide  → 2D density estimation
VVC    → 3D density estimation
```

**不要为了统一代码而强行将 Push / Slide 转成 3D。**

固定维度会导致退化 covariance / bandwidth 等问题。

---

# 6. 推荐的最终实验矩阵

核心实验：

| 方法 | 类型 | 当前状态 | 优先级 |
|---|---|---|---|
| KDE | Non-parametric | 已完成 | P0 |
| Resampling-based GMM | Parametric mixture | 已完成 | P0 |
| Weighted Histogram / Grid | Non-parametric | 待实现 | **P0** |
| Multivariate Gaussian | Parametric | 待实现 | **P0** |
| Weighted-EM GMM | Parametric mixture | 待考虑 | P1 |
| KNN Density | Non-parametric | 待考虑 | P2 |
| Bayesian GMM / DP-GMM | Bayesian mixture | 待考虑 | P2 |

因此，**最基本必须形成的 comparison 是：**

\[
\boxed{
\text{KDE}
\quad vs.\quad
\text{Resampling-based GMM}
\quad vs.\quad
\text{Weighted Histogram}
\quad vs.\quad
\text{Multivariate Gaussian}
}
\]

如果时间允许，再加入：

\[
\boxed{\text{Weighted-EM GMM}}
\]

---

# 7. 实验真正要验证的问题

不要简单地将目标定义为：

> “比较哪个 density estimator 最好。”

真正需要回答的是：

> **在完全相同的 temporal weighting mechanism 下，替换不同的 distribution estimator 后，当前研究的主要结论是否仍然成立？**

重点观察：

1. 不同方法得到的 goal distribution 是否具有相似的整体结构；
2. downstream 实验结果是否保持一致；
3. KDE 的 kernel smoothing 是否显著影响结果；
4. GMM 的 Gaussian mixture assumption 是否显著影响结果；
5. 简单 Gaussian 是否也能得到类似趋势；
6. 最终研究结论是否对 density estimator 的选择具有 robustness。

**不要预先假设 replacement experiment 一定支持当前结论。**

实验结果应该真实反映不同方法之间的差异。

---

# 8. 计算资源：必须充分利用并行能力

## 8.1 当前机器

当前机器拥有：

### CPU

**Intel Core i9-14900K**

可使用：

> 32 logical CPUs

### GPU

**2 × NVIDIA RTX 3060**

每张：

> 12 GB VRAM

这些资源都可以自由利用来缩短实验时间。

---

# 9. 时间限制

当前距离论文截稿：

> **约 3 天。**

因此：

\[
\boxed{\text{实验吞吐量非常重要}}
\]

不要无意义地串行运行可以并行的实验。

在保证实验正确性和可复现性的前提下，应主动寻找并行化机会。

---

# 10. 并行化策略

可以从以下维度进行并行：

- 不同 task；
- 不同 estimator；
- 不同 random seed；
- 不同 GMM component 数量；
- 不同 hyperparameter；
- 不同 repetition。

例如：

```text
CPU / GPU resources
        │
        ├── Reach / KDE
        ├── Reach / GMM
        ├── Reach / Histogram
        ├── Push / KDE
        ├── Push / GMM
        ├── Slide / Histogram
        ├── VVC / Gaussian
        └── ...
```

只要实验之间相互独立，就应该尽量并行运行。

---

# 11. CPU 并行

对于 CPU-bound 工作：

- 使用 multiprocessing / job-level parallelism；
- 尽可能利用 i9-14900K 的多核资源；
- 避免不必要的串行执行。

但是必须注意 **CPU oversubscription**。

例如：

> 32 个 multiprocessing worker × 每个 worker 又启动多线程 BLAS

可能导致严重的线程过量，反而降低性能。

必要时合理设置：

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
```

并根据实际 profiling 选择合理的 worker 数量。

**不要机械地认为“32 核就应该开 32 个 worker”。**

---

# 12. GPU 并行

两张 RTX 3060 均可使用。

例如：

```text
GPU 0 → 实验组 A
GPU 1 → 实验组 B
```

或者：

```text
GPU 0 → Reach / 某些实验
GPU 1 → VVC / 某些实验
```

如果某个单独实验不能有效使用 GPU，可以让两张 GPU 同时承担不同的独立实验。

但是：

> **不要为了使用 GPU 而强行把 CPU-oriented 的 sklearn KDE / GMM 改写成 GPU 版本。**

如果 CPU multiprocessing 更快，就使用 CPU。

GPU 的使用应该以**实际加速效果**为准。

---

# 13. Profiling 优先

如果某个步骤耗时异常长：

1. 先确定 bottleneck；
2. 判断它是：
   - CPU-bound；
   - GPU-bound；
   - I/O-bound；
   - memory-bound；
3. 再决定：
   - 多进程；
   - GPU；
   - batch；
   - cache；
   - 算法优化。

不要在没有 profiling 的情况下盲目优化。

---

# 14. Cache 与重复计算

尽可能避免重复计算。

可以缓存：

- processed dataset；
- temporal weights；
- normalized weights；
- resampling results；
- fitted distributions；
- intermediate evaluation results。

但 cache 必须能够区分关键配置，例如：

```text
task
estimator
random seed
decay factor
hyperparameters
data split
```

避免误用其他实验的结果。

---

# 15. 可复现性要求

即使当前时间紧张，也不能牺牲实验可复现性。

每个实验至少应尽可能记录：

- task；
- estimator；
- random seed；
- temporal decay factor；
- estimator hyperparameters；
- goal dimensionality；
- sample count；
- GMM component count；
- histogram resolution；
- KNN \(k\)（如果使用）；
- 输出路径；
- 必要的运行配置。

并行执行不能导致实验定义发生隐式变化。

---

# 16. 开始实现前必须做的事情

**不要直接开始写代码。**

首先阅读当前项目：

1. 项目目录结构；
2. KDE 实现；
3. temporal weighting 的实现；
4. 当前 resampling-based GMM 实现；
5. goal 数据读取方式；
6. 数据预处理；
7. downstream evaluation pipeline；
8. 实验结果保存方式；
9. 当前实验运行方式；
10. 当前主要 computational bottleneck。

理解现有代码后，再决定最小改动方案。

**尽量复用现有实验框架。**

不要为了这个 replacement experiment 重构与任务无关的代码。

---

# 17. 代码质量要求

新增代码应该：

- 尽量模块化；
- 不破坏已有 KDE / GMM 实验；
- 保持接口清晰；
- 支持不同 goal dimensionality；
- 支持不同 estimator；
- 支持并行实验；
- 支持 reproducibility；
- 尽量避免重复实现相同的数据处理逻辑。

如果现有代码已经能够很好地支持某些功能，不要重复造轮子。

---

# 18. Progress

## 已完成

- [x] KDE baseline
- [x] Temporal weighting：\(w_i=0.99^{\Delta t_i}\)
- [x] Resampling-based weighted GMM

## P0：必须完成

- [x] Weighted Histogram / Grid Density（`src/gc_ope/evaluate/evaluator_replacements.py`）
- [x] Multivariate Gaussian（`src/gc_ope/evaluate/evaluator_replacements.py`）
- [x] 在相关任务上运行 replacement experiments（push/slide 全量完成；reach/vvc 的 KDE 因大数据量留待补跑，`logs/replacement_experiment/`）
- [x] 汇总不同 estimator 的结果（push/slide 6 估计器统一 comparison table + KL 曲线图，`logs/replacement_experiment/plots/`）
- [ ] 完成 downstream evaluation（待接课程学习 pipeline，等 6 方法离线验证通过）
- [ ] 检查实验 reproducibility（单 job 确定性已验证，全量复跑留待后续）

## P0.5：加权 NN / Flow Matching

- [x] 加权 MLP 密度估计器（`src/gc_ope/evaluate/evaluator_learned.py`，dim→16→16→1，loss=Σw_i·(f_θ−y_i)²，y_i=LOO-KDE log-density）
- [x] Flow Matching 密度估计器（`src/gc_ope/evaluate/evaluator_learned.py`，直线 flow + 2 层 MLP 向量场，权重注入=按权重采样条件目标，密度=forward 采样+KDE）
- [x] 单测：确定性（同 seed 重训一致）、密度非负有限、KDE baseline 不受影响（`tests/evaluate/test_evaluator_learned.py`，7 个全过）
- [x] 全量 job：4 task × 2 method × 5 seed × 5 checkpoint（push/slide/reach/vvc 均完成，NN/FM 无缺失）

## P0.5 补跑待办（时间不足留待后续）

- [ ] reach/vvc 的 4 传统估计器中 KDE 各缺 9/12 个 (seed,ckpt)，原因：这些 checkpoint 历史成功目标 >17 万，sklearn KDE 在 3D 上 fit 4 分钟/job + MC-KL 小时级；已加 oracle 截断 200k（`ORACLE_MAX_REFERENCES`），补跑脚本就绪（`scripts/replacement_run_parallel.py --methods kde --tasks reach vvc`），待 32 核空闲时启动

## P1：强烈建议

- [ ] True Weighted-EM GMM

## P2：有时间再做

- [ ] KNN Density
- [ ] Bayesian GMM / DP-GMM

## 最终分析

- [x] 比较所有核心 estimator（push/slide 6 估计器 KL 均值/标准差 + 随训练步数曲线，`logs/replacement_experiment/plots/summary_table.csv` + `fig_kl_curves.png`）
- [ ] 生成实验结果表（push/slide 已完成；reach/vvc 待补跑后合并）
- [ ] 生成必要的 distribution visualization
- [ ] 检查不同方法是否得到一致趋势
- [ ] 检查 downstream result 是否保持
- [ ] 判断当前研究结论是否具有 robustness
- [ ] 整理可以放进论文的实验结果
- [ ] 整理可以用于 reviewer response 的结论

---

# 19. 三天截止时间下的执行优先级

## P0：现在立即完成

```text
KDE
  +
Resampling-based GMM
  +
Weighted Histogram
  +
Multivariate Gaussian
       ↓
所有相关 task
       ↓
Downstream evaluation
       ↓
结果汇总
```

这是当前最重要的工作。

---

## P1：P0 完成后立即进行

```text
True Weighted-EM GMM
```

如果实现成本合理，则加入。

---

## P2：只有在还有时间时

```text
KNN Density
Bayesian GMM / DP-GMM
```

如果时间紧张，直接跳过。

**不要为了增加算法数量而牺牲 P0 的完整性。**

---

# 20. 最终成功标准

本阶段工作最终需要能够可靠回答：

> **当保持 \(w_i=0.99^{\Delta t_i}\) 的 temporal weighting mechanism、数据以及 downstream evaluation 全部不变，仅替换 goal distribution estimator 时，当前研究的主要结论是否仍然成立？**

如果结果支持 robustness，则最终希望能够形成类似如下的研究结论：

> 实验结果在不同的 goal distribution estimation methods 下保持一致，说明观察到的主要效果并非特定 KDE 实现或 kernel smoothing 所导致。

但不要预先假定这个结论成立。

**必须以实际实验结果为准。**

---

# 21. 给 Claude Code 的执行要求

从现在开始处理本项目时：

1. **先阅读本 `plan.md`。**
2. **先检查当前代码和实验状态，再行动。**
3. 不要重复已经完成的工作。
4. 优先完成 P0。
5. 可以并行的实验尽量并行。
6. 充分利用当前 i9-14900K 的 CPU 和两张 RTX 3060，但不要为了“用满硬件”而进行无意义的改造。
7. 遇到耗时问题先 profiling，再决定优化策略。
8. 保持实验可复现。
9. 每完成一个重要阶段，更新本 `plan.md` 中的 Progress。
10. 如果发现当前计划与实际代码结构存在冲突，以**理解实际代码后做最小必要调整**为原则，不要机械执行计划。
11. 如果某个 P1/P2 项目可能威胁 P0 的完成时间，应立即降低其优先级。
12. **当前只有约 3 天，优先保证核心实验“完整、正确、可复现”，而不是追求算法数量。**

---

## 当前最重要的目标

\[
\boxed{
\text{KDE}
\rightarrow
\text{GMM}
\rightarrow
\text{Histogram}
\rightarrow
\text{Gaussian}
}
\]

在相同的：

\[
\boxed{w_i=0.99^{\Delta t_i}}
\]

以及相同的 downstream evaluation 下完成 replacement experiment，并回答：

\[
\boxed{
\text{研究结论是否依赖于 KDE？}
}
\]

**先把这个问题完整回答，再考虑其他扩展。**


# 22. 可执行的实验验收标准（Acceptance Criteria）

本节定义本 replacement experiment 的“完成”标准。

原则：

> **实验不是“代码跑完”就算完成，而是必须满足：实现正确、实验公平、结果完整、可复现，并且能够直接支持论文 / reviewer response。**

---

## 22.1 Level 0：代码与接口验收

### AC-0.1：所有 estimator 可以通过统一接口运行

至少以下方法必须能够使用统一的 experiment pipeline：

```text
KDE
Resampling-based GMM
Weighted Histogram / Grid Density
Multivariate Gaussian
```

理想接口形式：

```text
fit(data, weights, config)
    ↓
density / distribution representation
    ↓
downstream evaluation
```

不要求强制使用完全相同的 Python API，但必须能够进入相同的实验流程。

**Pass 条件：**

- 不需要为每个 estimator 手工修改实验主流程；
- estimator 可以通过 config / argument 切换；
- 不同 estimator 的输入数据和 temporal weights 来源一致。

---

### AC-0.2：Temporal weighting 只有一份来源

所有 estimator 必须使用：

\[
w_i=0.99^{\Delta t_i}
\]

不得出现：

```text
KDE 使用一种 weighting
GMM 使用另一种 weighting
Histogram 又重新计算 weighting
```

**Pass 条件：**

- temporal weight 在公共数据处理阶段计算；
- estimator 只接收处理后的 `data + weights`；
- 代码中不存在 estimator-specific 的隐式 weighting。

---

### AC-0.3：Push / Slide 不得错误地使用退化维度

对于：

```text
Push  : (x, y, 0.02)
Slide : (x, y, 0.02)
```

必须使用：

```text
(x, y)
```

进行 density estimation。

**Pass 条件：**

- Histogram 使用 2D；
- Gaussian 使用 2D；
- GMM 使用 2D；
- KDE 使用已有的正确维度；
- 不因为统一代码接口而人为保留固定 z 维。

---

## 22.2 Level 1：Estimator 正确性验收

### AC-1.1：Weighted Histogram 正确实现

对于每个 bin \(B_j\)：

\[
P(B_j)
=
\frac{
\sum_{i:g_i\in B_j}w_i
}{
\sum_iw_i
}.
\]

**必须满足：**

\[
P(B_j)\geq0
\]

并且：

\[
\sum_jP(B_j)\approx1.
\]

允许浮点误差。

建议测试：

```text
abs(sum(probabilities) - 1) < 1e-6
```

或者根据具体数值实现设置合理 tolerance。

---

### AC-1.2：Weighted Gaussian 正确实现

必须使用 weighted mean：

\[
\mu=
\frac{\sum_iw_ig_i}
{\sum_iw_i}.
\]

并且 covariance 必须由同一组 weights 计算。

至少进行一个 synthetic-data correctness test：

```text
给定少量人工构造的数据 + 已知 weights
        ↓
手算 weighted mean / covariance
        ↓
与程序结果比较
```

**Pass 条件：**

程序结果与 reference implementation 在合理 numerical tolerance 内一致。

---

### AC-1.3：Resampling-based GMM 权重确实生效

需要验证：

> 改变 temporal weights 后，resampling distribution 确实发生变化。

至少进行一个 sanity check：

```text
构造两个明显不同权重的数据集
        ↓
进行 resampling
        ↓
检查样本频率是否向高权重样本移动
```

不能只验证 GMM 能运行。

---

### AC-1.4：KDE baseline 没有被 replacement experiment 改坏

运行原始 KDE pipeline，确认：

- 可以正常运行；
- 输出格式保持一致；
- downstream evaluation 可以正常运行；
- replacement experiment 不改变 KDE baseline 的实验逻辑。

**Pass 条件：**

replacement code merge 后，原 KDE experiment 仍然能够独立运行。

---

## 22.3 Level 2：实验公平性验收

这是 reviewer replacement experiment 最重要的一层。

### AC-2.1：除 estimator 外，其余变量完全一致

不同 estimator 的实验必须共享：

```text
raw dataset
data preprocessing
data split
temporal weighting
task configuration
downstream evaluation
metrics
evaluation protocol
```

唯一核心变量：

```text
density estimator
```

---

### AC-2.2：Temporal weighting 一致性检查

至少随机抽取若干样本，记录：

```text
sample id
Δt
weight
```

检查不同 estimator 使用的 weight 是否完全一致。

例如：

```text
sample 001 : 0.99^12
sample 002 : 0.99^37
...
```

不同 estimator 不允许出现不同 weight。

---

### AC-2.3：随机种子管理

所有具有 stochastic behavior 的 estimator 必须能够记录 seed。

至少：

```text
task
estimator
seed
```

应该进入实验 metadata。

特别是：

> Resampling-based GMM

必须记录 resampling seed。

---

### AC-2.4：数据泄漏检查

确认：

- density fitting 没有使用 evaluation-only data；
- train / validation / test 的边界没有被破坏；
- downstream evaluation 没有反向影响 density fitting。

如果当前项目本身存在固定的数据划分协议，应严格沿用。

---

## 22.4 Level 3：实验覆盖率验收

### AC-3.1：P0 estimator 全部完成

以下四种方法必须全部有结果：

```text
[x] KDE
[x] Resampling-based GMM
[x] Weighted Histogram
[x] Multivariate Gaussian
```

---

### AC-3.2：核心任务全部覆盖

至少覆盖当前实验中实际使用的相关 task：

```text
Reach
Push
Slide
VVC
```

如果某个 task 在当前实验 pipeline 中本来就不存在，可以明确记录：

```text
N/A — not applicable
```

不能简单留下空白结果。

---

### AC-3.3：维度正确

最终实验 metadata 中必须能够明确看到：

| Task | Dimension |
|---|---:|
| Reach | 3 |
| Push | 2 |
| Slide | 2 |
| VVC | 3 |

并且 estimator 实际使用的 dimension 与 metadata 一致。

---

## 22.5 Level 4：Downstream Evaluation 验收

### AC-4.1：所有 estimator 必须经过相同 downstream pipeline

不能出现：

```text
KDE → 完整 downstream evaluation
GMM → 简化版 evaluation
Histogram → 另一个 evaluation
```

所有方法必须进入同一套 downstream evaluation。

---

### AC-4.2：至少得到一份统一 comparison table

最终必须生成类似：

| Task | Estimator | Seed | Metric 1 | Metric 2 | ... |
|---|---|---:|---:|---:|---:|
| Reach | KDE | ... | ... | ... | ... |
| Reach | GMM | ... | ... | ... | ... |
| Reach | Histogram | ... | ... | ... | ... |
| Reach | Gaussian | ... | ... | ... | ... |
| Push | KDE | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |

**不能只保留散落在不同 log 中的结果。**

---

### AC-4.3：至少完成一次跨 estimator 的直接比较

必须能够回答：

```text
KDE
vs
Resampling-based GMM
vs
Weighted Histogram
vs
Multivariate Gaussian
```

在相同 task / evaluation protocol 下：

> 哪些结论一致？哪些结论不同？

---

## 22.6 Level 5：结果合理性验收

这一层不是要求“结果必须符合预期”。

而是要求：

> **结果必须经过 sanity check，确认没有明显 implementation bug。**

---

### AC-5.1：Distribution normalization

对于可以显式表示 probability mass / density 的方法：

- Histogram probability sum ≈ 1；
- Gaussian covariance 合法；
- GMM mixture weights sum ≈ 1；
- KDE density 非负。

---

### AC-5.2：Covariance 合法

对于 Gaussian / GMM：

检查：

\[
\Sigma \approx \Sigma^T
\]

并且 covariance 应当是 positive semi-definite。

如果因为 numerical precision 出现极小负 eigenvalue，可以使用合理 numerical tolerance 判断。

---

### AC-5.3：结果数量级 sanity check

检查：

- density 是否出现异常极大值；
- probability 是否出现 NaN；
- probability 是否出现 Inf；
- metric 是否出现 NaN / Inf；
- estimator 是否因为极少数样本产生明显异常。

---

### AC-5.4：至少进行可视化 sanity check

对于 2D task：

至少画出：

```text
raw goal samples
+
estimated distribution
```

建议至少比较：

```text
KDE
GMM
Histogram
Gaussian
```

对于 3D task，可以根据实际需要：

- 3D visualization；
- 2D projection；
- pairwise marginal visualization；
- density contour。

目的不是制作论文最终 figure，而是发现明显 implementation bug。

---

## 22.7 Level 6：可复现性验收

### AC-6.1：单个实验可以重新运行

随机选择至少一个：

```text
task × estimator
```

完整重新运行。

应该能够得到一致或统计上等价的结果。

---

### AC-6.2：实验配置可追踪

每个结果必须能够追溯到：

```text
task
estimator
seed
decay factor
goal dimension
estimator hyperparameters
data configuration
code version / git commit
```

至少保存关键 metadata。

---

### AC-6.3：Git 状态清晰

最终实验结果对应的代码必须能够定位到一个明确的 Git commit。

建议记录：

```text
git rev-parse HEAD
```

这样 reviewer response 阶段可以明确知道：

> 这些结果由哪一个版本的代码生成。

---

## 22.8 Level 7：性能 / 并行验收

当前拥有：

```text
i9-14900K
32 logical CPUs
2 × RTX 3060 12GB
```

因此不能无理由地将所有实验完全串行执行。

---

### AC-7.1：独立实验能够并行

至少能够并行运行：

```text
不同 task
或
不同 estimator
或
不同 seed
```

中的一种。

---

### AC-7.2：不存在明显 CPU oversubscription

如果使用：

```text
multiprocessing
+
BLAS/OpenMP
```

必须确认没有出现：

```text
N workers
×
M BLAS threads
```

导致严重线程爆炸。

---

### AC-7.3：长任务有可观察状态

长时间运行的实验必须能够知道：

```text
已经完成多少
还剩多少
哪个 task / estimator 正在运行
是否出现失败
```

不要让整个实验变成：

```text
python run.py

等待几个小时

不知道跑到哪里
```

---

### AC-7.4：单个失败不会导致全部实验丢失

推荐采用：

```text
task × estimator × seed
```

粒度的独立 job。

如果一个 job 失败：

```text
Reach / GMM / seed=3
```

不能导致：

```text
Reach / KDE
Push / KDE
VVC / Gaussian
...
```

全部结果丢失。

---

## 22.9 Level 8：Reviewer-ready 验收

这是最终“真的完成”的标准。

### AC-8.1：能够生成一张最终 comparison table

至少包含：

```text
Task
Estimator
Performance metric(s)
Mean / Std 或其他适当统计量
```

---

### AC-8.2：能够生成一张核心 visualization

至少能够展示：

> 不同 density estimator 下 downstream result 的比较。

例如：

```text
Estimator
   │
   ├── KDE
   ├── GMM
   ├── Histogram
   └── Gaussian
          ↓
     Performance
```

具体图形式根据现有论文 metric 决定。

---

### AC-8.3：能够用一句话描述核心结果

实验完成后必须能够明确写出：

> “在保持 temporal weighting \(w_i=0.99^{\Delta t_i}\) 和 downstream evaluation 不变的情况下，将 KDE 替换为 GMM、weighted histogram 和 multivariate Gaussian 后，________。”

这里的空白必须由真实实验结果填写。

---

### AC-8.4：能够回答 reviewer 的核心问题

最终必须能够回答：

1. 是否只改变了 density estimator？
2. temporal weighting 是否保持不变？
3. 其他实验条件是否保持一致？
4. replacement methods 是否覆盖足够不同的 estimator family？
5. downstream conclusion 是否稳定？
6. 如果不稳定，具体在哪些 task / metric 上发生变化？
7. 这些变化是否影响论文核心 claim？

---

# 23. 最终验收 Checklist

在宣布本实验“完成”之前，逐项确认：

## Implementation

- [ ] KDE 正常运行
- [ ] Resampling-based GMM 正常运行
- [ ] Weighted Histogram 正常运行
- [ ] Multivariate Gaussian 正常运行
- [ ] Temporal weighting 统一使用 \(0.99^{\Delta t}\)
- [ ] Push 使用 2D
- [ ] Slide 使用 2D
- [ ] Reach 使用 3D
- [ ] VVC 使用 3D

## Fairness

- [ ] 数据完全一致
- [ ] preprocessing 完全一致
- [ ] temporal weights 完全一致
- [ ] downstream evaluation 完全一致
- [ ] metrics 完全一致
- [ ] 没有 data leakage
- [ ] random seed 可追踪

## Coverage

- [ ] 所有 P0 estimator 都有结果
- [ ] 所有相关 task 都有结果
- [ ] 所有核心结果进入统一 comparison table

## Correctness

- [ ] Histogram probability normalization 正确
- [ ] Gaussian weighted mean 正确
- [ ] Gaussian weighted covariance 正确
- [ ] GMM weighting sanity check 通过
- [ ] 无 NaN / Inf
- [ ] covariance 合法
- [ ] 至少完成一次 distribution visualization sanity check

## Reproducibility

- [ ] 实验配置可追踪
- [ ] seed 可追踪
- [ ] git commit 可追踪
- [ ] 至少一个实验可以重新运行并复现

## Compute

- [ ] 独立实验进行了并行化
- [ ] 没有明显 CPU oversubscription
- [ ] 长任务有进度信息
- [ ] 单个 job 失败不会导致全部实验丢失

## Reviewer-ready

- [ ] 有统一 comparison table
- [ ] 有核心 visualization
- [ ] 有明确的结果结论
- [ ] 能明确说明“只替换了 density estimator”
- [ ] 能明确回答“结论是否依赖 KDE”
- [ ] 如果存在不一致结果，已经定位具体 task / metric

---

# 24. 最低完成线（Minimum Acceptance）

考虑到当前只有约 3 天，**不允许因为追求额外实验而错过最低完成线。**

最低完成线为：

\[
\boxed{
\text{KDE}
+
\text{Resampling-based GMM}
+
\text{Weighted Histogram}
+
\text{Multivariate Gaussian}
}
\]

并且满足：

\[
\boxed{
\text{所有核心 task}
+
\text{统一 temporal weighting}
+
\text{统一 downstream evaluation}
+
\text{统一结果表}
}
\]

同时满足：

\[
\boxed{
\text{正确性 sanity check}
+
\text{可复现}
+
\text{Reviewer-ready}
}
\]

达到这一标准后，即可认为 **P0 replacement experiment 完成**。

---

# 25. 时间不足时的降级策略

如果距离 deadline 不足：

### 第一优先级

必须保留：

```text
KDE
GMM
Histogram
Gaussian
```

### 第二优先级

可以减少：

- seed 数量；
- visualization 数量；
- 额外 hyperparameter sweep。

但必须在最终记录中明确说明实验规模。

### 第三优先级

直接删除：

```text
KNN
Bayesian GMM
DP-GMM
```

### 最后才考虑

减少核心 task。

**不要首先砍掉 estimator diversity。**

因为本实验的核心目的正是证明：

> **结论不是 KDE-specific。**

---

# 26. 最终 Definition of Done

只有同时满足以下条件，才能将本项目状态标记为：

```text
DONE
```

```text
[1] 四种 P0 estimator 全部实现
        ↓
[2] 所有 estimator 使用完全相同的 temporal weighting
        ↓
[3] 所有相关 task 完成实验
        ↓
[4] 所有结果经过相同 downstream evaluation
        ↓
[5] correctness sanity checks 全部通过
        ↓
[6] 结果可以复现
        ↓
[7] 有统一 comparison table
        ↓
[8] 有必要的 visualization
        ↓
[9] 能明确回答 reviewer 的 replacement-experiment 问题
        ↓
[10] 结果已经足够直接写入论文 / reviewer response
        ↓
                  DONE
```

**特别注意：**

> `代码能跑` ≠ `实验完成`

> `所有实验跑完` ≠ `实验完成`

真正的完成标准是：

\[
\boxed{
\text{Correct}
+
\text{Fair}
+
\text{Complete}
+
\text{Reproducible}
+
\text{Reviewer-ready}
}
\]