# OPE算法实现分析报告

**创建日期**: 2026-01-12
**分析目标**: 详细分析当前OPE算法实现，诊断问题，与scope-rl对比，提出改进建议

---

## 1. 概述

### 1.1 模块架构

当前OPE模块位于 `/src/gc_ope/algorithm/ope/`，包含以下核心文件：

| 文件 | 职责 | 代码行数 |
|------|------|----------|
| `estimators.py` | OPE估计器 (DM, TIS, DR) | ~340行 |
| `fqe.py` | Fitted Q Evaluation训练器 | ~390行 |
| `algorithm_adapter.py` | SB3算法适配器 | ~495行 |
| `logged_dataset.py` | 数据集收集与缓存 | - |
| `ope_input.py` | OPE输入构建 | - |

### 1.2 数据流

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ LoggedDataset   │───▶│ build_ope_inputs │───▶│ OPEInputs       │
│ (行为策略数据)   │    │ (构建OPE输入)     │    │ (估计器输入)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌──────────────┐         ┌─────────────────┐
                       │ FQETrainer   │         │ dm/tis/dr       │
                       │ (训练Q函数)   │         │ _estimate()     │
                       └──────────────┘         └─────────────────┘
```

### 1.3 支持的OPE算法

| 算法 | 实现状态 | 文件位置 |
|------|----------|----------|
| DM (Direct Method) | ✅ 已实现 | `estimators.py:107-147` |
| TIS (Trajectory-wise IS) | ✅ 已实现 | `estimators.py:165-244` |
| PDIS (Per-Decision IS) | ❌ 未实现 | - |
| DR (Doubly Robust) | ✅ 已实现 | `estimators.py:247-337` |

---

## 2. 各算法实现详解

### 2.1 Direct Method (DM)

**位置**: `estimators.py:107-147`

**公式**:
```
V^DM = (1/N) * Σ_i Q(s_i, π_eval(s_i))
```

**实现特点**:
- 使用FQE训练的Q函数估计策略价值
- 支持两种模式：使用所有状态 或 仅使用初始状态 (`initial_only=True`)
- 有偏估计器，但方差较低

**代码片段** (`estimators.py:87-104`):
```python
def dm_compute_trajectory_values(
    inputs: OPEInputs, initial_only: bool = False
) -> np.ndarray:
    mask = inputs.step_index == 0 if initial_only else slice(None)
    values = inputs.q_sa_eval[mask]
    return np.asarray(values, dtype=np.float32)
```

**评估**: DM实现简洁正确，依赖FQE Q函数的准确性。

### 2.2 Trajectory-wise Importance Sampling (TIS)

**位置**: `estimators.py:165-244`

**公式**:
```
V^TIS = (1/M) * Σ_τ w_τ * G_τ
w_τ = Π_t [π_eval(a_t|s_t) / π_behavior(a_t|s_t)]
    = exp(Σ_t [log π_eval(a_t|s_t) - log π_behavior(a_t|s_t)])
```

**实现特点** (`estimators.py:180-202`):
```python
for idxs in traj_map.values():
    r = inputs.rewards[idxs]
    logp_b = inputs.behavior_log_prob[idxs]
    logp_e = inputs.eval_log_prob[idxs]

    # Clip log-importance weights to prevent overflow/underflow
    log_weights = np.clip(logp_e - logp_b, -10.0, 10.0)

    # Calculate trajectory weight
    weight = np.exp(log_weights.sum())

    # Additional safety check for infinity
    if not np.isfinite(weight):
        weight = 0.0

    # Hard clip the cumulative weight
    weight = np.clip(weight, 0.0, 1e4)

    discounts = np.power(gamma, np.arange(len(r)))
    returns.append(weight * np.sum(discounts * r))
```

**数值稳定性措施**:
1. Log-space裁剪: `[-10.0, 10.0]` → 最大权重 ~22000
2. 无穷大检查: 非有限值设为0
3. 硬裁剪: `[0.0, 1e4]`

**⚠️ 潜在问题**: 直接使用 `logp_e - logp_b` 计算重要性权重，未考虑连续动作空间的特殊性。

### 2.3 Per-Decision Importance Sampling (PDIS)

**实现状态**: ❌ 未实现

PDIS与TIS的区别在于使用逐步累积权重而非整条轨迹权重：
```
V^PDIS = (1/M) * Σ_τ Σ_t γ^t * w_{0:t} * r_t
w_{0:t} = Π_{k=0}^{t} [π_eval(a_k|s_k) / π_behavior(a_k|s_k)]
```

PDIS方差更低，因为权重不会随轨迹长度无限增长。

### 2.4 Doubly Robust (DR)

**位置**: `estimators.py:247-337`

**公式**:
```
V_τ^DR = Σ_t γ^t [w_t * (r_t - Q(s_t, a_t)) + w_{t-1} * Q(s_t, π_eval(s_t))]
w_t = Π_{k=0}^{t} [π_eval(a_k|s_k) / π_behavior(a_k|s_k)]
w_{-1} = 1
```

**实现特点** (`estimators.py:262-285`):
```python
for idxs in traj_map.values():
    r = inputs.rewards[idxs]
    logp_b = inputs.behavior_log_prob[idxs]
    logp_e = inputs.eval_log_prob[idxs]
    q_sa = inputs.q_sa_behavior[idxs]
    v_eval = inputs.q_sa_eval[idxs]

    # Clip importance ratios per step
    ratios = np.exp(np.clip(logp_e - logp_b, -10.0, 5.0))  # exp(5) ~ 148

    # Cumulative product with safety checks
    w_step = np.cumprod(ratios)
    w_step = np.clip(w_step, 0.0, 1e4)

    w_prev = np.concatenate([[1.0], w_step[:-1]])
    discounts = np.power(gamma, np.arange(len(r)))

    term = w_step * (r - q_sa) + w_prev * v_eval
    est.append(np.sum(discounts * term))
```

**数值稳定性措施**:
1. 更严格的log-space裁剪: `[-10.0, 5.0]` → 单步最大比率 ~148
2. 累积乘积后硬裁剪: `[0.0, 1e4]`

**⚠️ 潜在问题**: 同TIS，直接使用概率密度比计算重要性权重。

---

## 3. 核心问题诊断

### 3.1 问题A: 连续动作空间重要性权重计算错误 (Critical)

**症状**:
- TIS/DR估计结果与在线评估结果差异显著
- 权重出现无穷大或NaN
- 估计值绝对值异常大

**根因分析**:

当前实现直接计算概率密度比：
```python
weight = exp(log π_eval(a|s) - log π_behavior(a|s))
```

**问题在于**：
1. **评价策略通常是确定性的**: `π_eval(s) = a*` (单一动作)
2. **行为策略采样的动作 `a_t` 几乎不可能恰好等于 `a*`**
3. **直接计算概率比会导致**:
   - 当 `a_t ≠ a*` 时，`π_eval(a_t|s)` 的概率密度极低
   - 导致 `log π_eval - log π_behavior` 出现极端负值或正值
   - 最终权重趋向0或无穷大

**scope-rl的解决方案: 核函数相似度**

scope-rl使用核函数度量动作相似度，替代精确概率比：

```python
# scope-rl: 使用核函数相似度
δ(π_e, a_{0:t}) = Π_{t'=0}^{t} K(π_e(s_{t'}), a_{t'})

# 其中 K 是核函数（如高斯核）
K(x, y) = exp(-||x-y||^2 / (2h^2)) / sqrt(2πh^2)
```

**核函数的优势**:
1. 当 `a_t` 接近 `π_e(s)` 时，相似度高
2. 当 `a_t` 远离 `π_e(s)` 时，相似度低但不为0
3. 通过bandwidth参数 `h` 控制平滑度
4. 天然限制权重范围，避免数值问题

### 3.2 问题B: 数值稳定性处理不足 (Medium)

**当前clip策略分析**:

| 位置 | 裁剪范围 | 问题 |
|------|----------|------|
| TIS log-space | `[-10, 10]` | 长轨迹累加后仍可能溢出 |
| TIS 线性空间 | `[0, 1e4]` | 硬裁剪引入偏差 |
| DR log-space | `[-10, 5]` | 更严格但仍不够 |
| DR 线性空间 | `[0, 1e4]` | 硬裁剪引入偏差 |

**问题**:
1. 多层clip策略可能导致累积偏差
2. 将非有限权重设为0会导致低估
3. 硬裁剪阈值(1e4)缺乏理论依据

### 3.3 问题C: PDIS未实现 (Low)

PDIS (Per-Decision Importance Sampling) 未作为独立估计器实现。

**影响**:
- 缺少方差更低的IS估计器选项
- DR实际上使用了PDIS风格的逐步权重，但没有独立的PDIS估计器

---

## 4. 与scope-rl对比分析

### 4.1 架构差异

| 方面 | gc_ope | scope-rl |
|------|--------|----------|
| 数据格式 | 展平的transition | 展平的trajectory |
| 动作空间 | 连续/离散统一处理 | 离散/连续分离实现 |
| 重要性权重 | 概率密度比 | **核函数相似度** |
| Pscore格式 | 标量 `(N,)` | 向量 `(N, action_dim)` |
| 代码组织 | 函数式 | 类继承式 |

### 4.2 核函数实现 (scope-rl)

**位置**: `scope_rl/utils.py:314-453`

scope-rl支持5种核函数：

| 核函数 | 公式 | 特点 |
|--------|------|------|
| Gaussian | `exp(-d²/2h²) / √(2πh²)` | 平滑，无界支持，推荐 |
| Epanechnikov | `0.75(1-(d/h)²)/h` | MSE最优，有界 |
| Triangular | `(1-d/h)/h` | 线性衰减，有界 |
| Cosine | `(π/4)cos(πd/2h)/h` | 余弦衰减，有界 |
| Uniform | `1/(2h)` | 均匀权重，有界 |

**Gaussian核函数实现**:
```python
def gaussian_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0):
    """
    x, y: shape (n_samples, n_dim)
    bandwidth: 控制平滑度的超参数
    """
    distance = l2_distance(x, y)  # ||x - y||²
    return np.exp(-distance / (2 * bandwidth**2)) / np.sqrt(2 * np.pi * bandwidth**2)
```

### 4.3 相似度权重计算流程 (scope-rl)

**位置**: `scope_rl/ope/estimators_base.py:251-316`

```python
def _calc_similarity_weight(self, step_per_trajectory, action,
                            evaluation_policy_action, pscore_type,
                            kernel="gaussian", bandwidth=1.0):
    # 步骤1: 计算每步的核函数相似度
    similarity_weight = self._kernel_function[kernel](
        evaluation_policy_action,  # π_e(s_t)
        action,                    # a_t
        bandwidth=bandwidth,
    ).reshape((-1, step_per_trajectory))  # (n_traj, T)

    # 步骤2: 累积乘积
    similarity_weight = np.cumprod(similarity_weight, axis=1)

    # 步骤3: 根据类型返回
    if pscore_type == "trajectory_wise":
        # TIS: 整条轨迹使用同一权重
        similarity_weight = np.tile(
            similarity_weight[:, -1], (step_per_trajectory, 1)
        ).T
    # 否则返回 step_wise 权重 (PDIS/DR)

    return similarity_weight
```

### 4.4 关键差异总结

| 方面 | gc_ope (当前) | scope-rl | 影响 |
|------|---------------|----------|------|
| 权重计算 | `exp(logp_e - logp_b)` | `kernel(π_e(s), a) / π_b(a)` | 数值稳定性 |
| 确定性策略 | 假设随机策略 | 支持确定性策略 | 适用范围 |
| bandwidth | 无 | 可调超参数 | 偏差-方差权衡 |
| 核函数选择 | 无 | 5种可选 | 灵活性 |

---

## 5. 改进建议

### 5.1 短期改进 (立即可做)

#### 5.1.1 添加核函数相似度支持

**新增文件**: `kernel_utils.py`

```python
import numpy as np

def gaussian_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
    """Gaussian kernel similarity.

    Args:
        x: Evaluation policy actions (N, action_dim)
        y: Behavior policy actions (N, action_dim)
        bandwidth: Kernel bandwidth (controls smoothness)

    Returns:
        Similarity weights (N,)
    """
    distance = np.sum((x - y) ** 2, axis=-1)  # ||x - y||²
    return np.exp(-distance / (2 * bandwidth ** 2))
```

#### 5.1.2 修改TIS估计器支持核函数

**修改文件**: `estimators.py`

```python
def tis_compute_trajectory_values_kernel(
    inputs: OPEInputs,
    kernel: str = "gaussian",
    bandwidth: float = 1.0,
) -> np.ndarray:
    """TIS with kernel similarity weight for continuous actions."""
    from .kernel_utils import gaussian_kernel

    traj_map = _traj_slices(inputs.traj_id)
    gamma = inputs.gamma
    returns = []

    for idxs in traj_map.values():
        r = inputs.rewards[idxs]
        actions = inputs.actions[idxs]
        eval_actions = inputs.eval_action[idxs]
        logp_b = inputs.behavior_log_prob[idxs]

        # 核函数相似度
        similarity = gaussian_kernel(eval_actions, actions, bandwidth=bandwidth)

        # 行为策略权重
        behavior_weight = np.exp(np.clip(logp_b.sum(), -20.0, 20.0))

        # TIS权重 = 相似度乘积 / 行为策略概率
        weight = similarity.prod() / max(behavior_weight, 1e-10)
        weight = np.clip(weight, 0.0, 1e4)

        discounts = np.power(gamma, np.arange(len(r)))
        returns.append(weight * np.sum(discounts * r))

    return np.asarray(returns, dtype=np.float32)
```

### 5.2 中期改进 (需要重构)

#### 5.2.1 添加PDIS估计器

```python
def pdis_compute_trajectory_values(inputs: OPEInputs) -> np.ndarray:
    """Per-Decision Importance Sampling estimator."""
    traj_map = _traj_slices(inputs.traj_id)
    gamma = inputs.gamma
    returns = []

    for idxs in traj_map.values():
        r = inputs.rewards[idxs]
        logp_b = inputs.behavior_log_prob[idxs]
        logp_e = inputs.eval_log_prob[idxs]

        # 逐步累积权重
        ratios = np.exp(np.clip(logp_e - logp_b, -10.0, 5.0))
        w_step = np.cumprod(ratios)
        w_step = np.clip(w_step, 0.0, 1e4)

        discounts = np.power(gamma, np.arange(len(r)))
        returns.append(np.sum(discounts * w_step * r))

    return np.asarray(returns, dtype=np.float32)
```

#### 5.2.2 扩展OPEInputs支持eval_action

**修改文件**: `ope_input.py`

```python
@dataclass
class OPEInputs:
    # ... 现有字段 ...
    actions: np.ndarray          # 行为策略动作 (N, action_dim)
    eval_action: np.ndarray      # 评价策略动作 (N, action_dim) - 新增
```

---

## 6. 验证方案

### 6.1 单元测试

**测试核函数实现**:
```python
# tests/algorithm/ope/test_kernel.py
def test_gaussian_kernel():
    eval_actions = np.array([[1.0, 2.0], [3.0, 4.0]])
    behavior_actions = np.array([[1.1, 2.1], [3.1, 4.1]])

    similarity = gaussian_kernel(eval_actions, behavior_actions, bandwidth=1.0)

    assert similarity.shape == (2,)
    assert np.all(similarity > 0) and np.all(similarity <= 1)
```

### 6.2 集成测试

**对比测试**:
```python
# tests/algorithm/ope/test_ope_kernel.py
def test_tis_with_kernel_vs_without():
    dataset = collect_logged_dataset(env, behavior_algo, n_episodes=100)
    inputs = build_ope_inputs(dataset, eval_algo, gamma=0.99)

    result_without = tis_estimate(inputs)
    result_with = tis_estimate_kernel(inputs, kernel="gaussian", bandwidth=1.0)

    # 验证核函数版本结果更稳定
    assert np.isfinite(result_with.mean)
    assert result_with.ci_upper - result_with.ci_lower < result_without.ci_upper - result_without.ci_lower
```

### 6.3 端到端验证

按CLAUDE.md中的测试环境验证：
- **环境**: my_reach, my_push, my_slide, flycraft
- **算法**: HER + SAC
- **预期**: TIS/PDIS估计结果与在线评估差异趋近一致；DM/DR偏差不超过20%

---

## 7. 总结

### 7.1 问题优先级

| 优先级 | 问题 | 影响 | 建议 |
|--------|------|------|------|
| **P0** | 连续动作重要性权重计算错误 | TIS/DR结果不可靠 | 添加核函数相似度 |
| **P1** | 数值稳定性不足 | 偶发NaN/Inf | 改进clip策略 |
| **P2** | PDIS未实现 | 缺少低方差选项 | 添加PDIS估计器 |

### 7.2 改进路线图

```
短期
├── 添加 kernel_utils.py（已完成）
├── 修改 TIS 支持核函数（已完成）
└── 添加单元测试（已完成）

中期
├── 添加 PDIS 估计器（已完成）
├── 修改 DR 支持核函数（已完成）
├── 扩展 OPEInputs（已完成）
└── 添加集成测试（已完成）

长期
├── 重构为类继承架构
├── 支持多种核函数
└── 添加 bandwidth 自动选择
```

### 7.3 参考资源

- scope-rl OPE实现: `/home/maxine/ai4robot/ope-repos/scope-rl/scope_rl/ope/`
- scope-rl 设计文档: `/home/maxine/ai4robot/ope-repos/scope-rl/scope_rl/ope/BASIC_OPE_SUMMARY.md`
- 核函数实现: `/home/maxine/ai4robot/ope-repos/scope-rl/scope_rl/utils.py:314-453`

---

*文档生成时间: 2026-01-12*
