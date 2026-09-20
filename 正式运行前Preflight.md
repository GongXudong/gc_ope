# 正式运行前 Preflight：Push / Slide 传统密度估计实验

在正式启动大规模实验之前，请先不要直接运行全部 4000 个 jobs。

请你作为实验执行 Agent，对当前项目做一次**运行前完整检查（preflight）**。目标是确认：

> 本轮实验只研究 Push / Slide 两个任务，在 seed 1～5 上比较 KDE、GMM、Histogram、Gaussian 四种传统密度拟合方法；NN 和 Flow Matching 本轮完全不运行。

---

## 1. 首先确认本轮实验协议

请检查当前代码、`plan.md`、相关配置，并确认最终实验协议：

- Tasks:
  - `push`
  - `slide`

- Seeds:
  - `1, 2, 3, 4, 5`

- Methods:
  - `kde`
  - `gmm`
  - `histogram`
  - `gaussian`

- NN：本轮不运行
- Flow Matching：本轮不运行
- 其他 task：本轮不运行

Checkpoint 使用：

```text
10k, 20k, ..., 990k, 1000k
```

即：

```python
[10000 + 10000*i for i in range(99)] + [1000000]
```

这实际上是 **100 个 checkpoint**。

因此理论 job 数：

```text
2 tasks × 5 seeds × 100 checkpoints × 4 methods
= 4000 jobs
```

请检查代码注释中是否仍错误写成 101 checkpoints / 4040 jobs。

---

## 2. 检查 Oracle 定义

请重点检查 oracle，不要擅自修改实验定义。

当前协议应该是：

对于 checkpoint `t`：

1. 读取 checkpoint `t` 自己保存的 evaluation CSV；
2. 取其中：

```text
termination == "reach target"
```

的成功目标点；
3. Push / Slide 使用：

```text
(x, y)
```

作为目标空间；
4. 使用这些当前 checkpoint 的成功目标点拟合 oracle KDE；
5. 用该 oracle KDE 作为：

\[
p_t^{oracle}
\]

6. 使用 Monte Carlo：

\[
D_{KL}(p_t^{oracle}\|\hat p_t)
=
E_{x\sim p_t^{oracle}}
[
\log p_t^{oracle}(x)
-
\log \hat p_t(x)
]
\]

评价历史数据训练得到的估计器。

请确认当前代码确实按照这个定义执行。

特别确认：

> oracle 的数据来自“当前 checkpoint”的成功目标，而不是历史数据。

同时确认历史 estimator 不会看到当前 checkpoint 或未来 checkpoint 的数据。

---

## 3. 检查历史数据没有 future leakage

对于当前 checkpoint `t`：

```text
historical_files(...)
```

只能包含：

```text
checkpoint < t
```

的数据。

严禁：

```text
checkpoint >= t
```

的数据进入 estimator。

请实际检查代码，而不是只根据注释判断。

---

## 4. 检查四种 estimator 是否使用完全相同的输入

确认：

KDE / GMM / Histogram / Gaussian

四种方法：

- 使用相同的历史 CSV；
- 使用相同的 success 筛选；
- 使用相同的时间折扣参数；
- 使用相同的目标维度；
- 不允许某一种方法偷偷使用额外数据。

当前：

```text
kappa = 0.9
```

请确认四种传统方法都使用同一套时间权重。

---

## 5. 检查当前实验参数

默认参数应确认如下：

```text
kappa = 0.9
KDE bandwidth = 0.2
GMM components = 5
GMM reg_covar = 1e-6
GMM resample_size = 1000
random_state = 0

MC samples = 100000
MC repeats = 5
```

请确认这些参数在实际执行路径中确实生效，而不是仅存在于 CLI 默认值中。

---

## 6. 特别检查“oracle KDE bandwidth”

当前代码中 oracle KDE 使用：

```text
bandwidth = 0.2
```

请确认这一点，并明确报告：

```text
oracle bandwidth = 0.2
estimator KDE bandwidth = 0.2
```

本轮实验**不要自行修改这个定义**。

如果你认为这个定义存在方法学问题，请在 preflight report 中提出，但不要在未经确认的情况下修改代码。

---

## 7. 检查 KL 实现

请检查：

```python
monte_carlo_kl(...)
```

确认实际计算的是：

\[
D_{KL}(p_{oracle}\|p_{estimate})
\]

而不是反过来：

\[
D_{KL}(p_{estimate}\|p_{oracle})
\]

并确认：

- sample 来自 oracle；
- 使用 raw goal space 的 log density；
- 两边 density 的坐标尺度 / Jacobian 转换一致；
- 不会因为标准化空间与原始 `(x,y)` 空间混用而产生错误。

如果存在 support filtering，请明确报告它的行为。

---

## 8. 检查 reference_success 截断

当前代码可能存在：

```python
ORACLE_MAX_REFERENCES = 200000
```

或者等价逻辑。

请确认：

- 当当前 checkpoint 成功点数量 ≤ 200000 时，使用全部成功点；
- 当 > 200000 时，当前代码到底如何截断；
- 是否是前 200000 个；
- 是否存在随机采样；
- 该行为是否会影响 oracle。

本轮不要自行改变，先报告实际行为。

---

## 9. 检查 10k / 20k 等早期 checkpoint

请确认：

对于最早 checkpoint：

```text
10k
```

由于不存在更早历史 checkpoint，它应该没有历史数据，因此应被：

```text
skipped
```

而不是强行拟合。

同理，早期 checkpoint 如果没有历史成功样本，也应该按照当前代码定义跳过。

请确认这属于预期行为，而不是 bug。

---

## 10. 检查输出与断点续跑机制

确认正式实验不会因为中断导致已有结果被错误覆盖或重复计算。

重点检查：

- CSV append 是否安全；
- 是否有 lock；
- `(task, seed, checkpoint, method)` 是否能够唯一定位一个 job；
- `--skip-done` 是否真的只跳过已经成功完成的 job；
- `status != ok` 的 job 是否会被重新执行；
- 是否会产生重复 CSV rows。

如果发现 skip / resume 逻辑有问题，请先报告，不要直接大规模运行。

---

## 11. 检查并行策略

本实验规模为：

```text
4000 jobs
```

每个 job 默认：

```text
100000 MC samples × 5 repeats
```

因此不要直接无脑启动最大并发。

请检查：

- 当前 worker 数；
- NumPy / sklearn / BLAS 是否还会内部多线程；
- 是否存在 CPU oversubscription；
- 是否可能同时产生大量内存占用；
- 是否会导致机器 swap；
- 是否存在大量重复读取 CSV 的 I/O 瓶颈。

我的机器 CPU 是 i9，内存 32GB。

请根据实际代码给出一个合理的初始 worker 数，但**不要擅自把机器资源打满**。

---

## 12. 正式运行前必须做一个“小规模 smoke test”

不要直接运行 4000 jobs。

先运行至少：

```text
push
seed=1
checkpoint=100000
```

并让：

```text
KDE
GMM
Histogram
Gaussian
```

四种方法各运行一次。

然后检查：

1. 四个 job 都能正常完成；
2. `status == ok`；
3. `kl` 是 finite；
4. `fixed_grid_kl` 是 finite；
5. `historical_successes > 0`；
6. `reference_successes > 0`；
7. 输出 CSV schema 正确；
8. 四种方法使用相同 history；
9. oracle 使用当前 100k checkpoint 的成功点；
10. 没有读取 100k 之后的数据。

如果 smoke test 失败，不要继续 4000-job sweep。

---

## 13. 再做一个小型一致性检查

在 smoke test 后，请直接打印 / 检查：

```text
当前 checkpoint 成功点数量
历史成功点数量
oracle KDE 使用的样本数量
KDE estimator 使用的历史数据量
GMM estimator 使用的历史数据量
Histogram estimator 使用的历史数据量
Gaussian estimator 使用的历史数据量
```

确认四种 estimator 的历史数据源一致。

---

## 14. 检查一个数学 sanity check

至少验证：

对于同一个 checkpoint：

```text
oracle = current checkpoint successful goals → KDE
```

然后计算：

```text
KL(oracle || oracle)
```

理论上应该接近：

\[
0
\]

如果当前 `monte_carlo_kl()` 可以方便地支持这一测试，请做一次。

如果不能方便测试，不要为了测试而修改正式实验代码；只报告原因。

---

## 15. 检查结果数量

正式运行完成后理论上：

```text
2 × 5 × 100 × 4 = 4000
```

条 job 记录。

但注意：

其中一部分早期 checkpoint 可能：

```text
skipped:no_historical_files
skipped:no_historical_successes
```

所以：

> 4000 是理论 job 数，不代表 4000 个都会得到 KL。

请最终分别统计：

```text
total
ok
skipped
error
```

以及按：

```text
task
seed
checkpoint
method
```

统计。

---

# Preflight 输出格式

在正式运行之前，请给我一个简洁的报告：

## Experiment Protocol

- Tasks:
- Seeds:
- Checkpoints:
- Methods:
- Total jobs:
- Oracle definition:
- History rule:
- kappa:
- KDE bandwidth:
- GMM parameters:
- MC samples:
- MC repeats:

## Code Verification

逐项回答：

- [ ] Oracle 数据源正确
- [ ] 无 future leakage
- [ ] 四种 estimator 使用相同历史数据
- [ ] KL 方向正确
- [ ] raw-space Jacobian 正确
- [ ] checkpoint 数量正确
- [ ] skip/resume 正确
- [ ] 输出不会发生重复/覆盖
- [ ] 并行策略合理

## Smoke Test

报告：

```text
task:
seed:
checkpoint:

KDE:
GMM:
Histogram:
Gaussian:

historical_successes:
reference_successes:

KL values:
fixed_grid_KL values:
```

## Problems Found

如果存在问题，按：

```text
[BLOCKER]
[WARNING]
[INFO]
```

分类。

**只有当没有 BLOCKER，并且 smoke test 通过之后，才启动完整的 4000-job 实验。**

---

# 最重要的执行约束

本轮实验只做：

\[
\boxed{
Push/Slide
\times
Seed\ 1\text{--}5
\times
100\ checkpoints
\times
\{KDE,GMM,Histogram,Gaussian\}
}
\]

不要加入：

- NN
- Flow Matching
- Reach
- VVC
- 其他 task
- 其他 estimator

不要在正式运行前自行改变 oracle、KL 定义或实验参数。

如果发现实验协议与代码不一致，先停止并报告，不要自行“修正”后直接开跑。