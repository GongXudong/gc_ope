# OPE Implementation Analysis

## 1. 当前实现概述

### 1.1 文件结构

```
src/gc_ope/algorithm/ope/
├── logged_dataset.py      (285行) - 数据收集与评价策略缓存
├── algorithm_adapter.py   (495行) - SB3算法统一接口
├── fqe.py                 (392行) - Fitted Q Evaluation实现
├── ope_input.py           (188行) - OPE输入准备流程编排
└── estimators.py          (338行) - DM/TIS/DR估计器实现
```

### 1.2 已实现方法状态表

| Method | Status | Key Features |
|--------|--------|--------------|
| **DM** (Direct Method) | ✅ 已实现 | 基于FQE的Q函数估计，支持bootstrap/normal/t-test置信区间 |
| **TIS** (Trajectory-wise IS) | ⚠️ 部分可用 | 实现了轨迹级重要性采样，但连续动作下存在数值不稳定 |
| **PDIS** (Per-Decision IS) | ❌ 未实现 | 缺失 |
| **DR** (Doubly Robust) | ⚠️ 不可用 | 实现了但数值极度不稳定，估计值偏差极大 |

### 1.3 数据流图

```
Logged Dataset Collection
    ↓
[logged_dataset.py] collect_logged_dataset()
    ↓
Evaluation Policy Cache (batch optimization)
    ↓
[algorithm_adapter.py] compute_action_log_prob()
    ↓
FQE Training (Q-function approximation)
    ↓
[fqe.py] FQETrainer.train()
    ↓
OPE Inputs Construction
    ↓
[ope_input.py] build_ope_inputs()
    ↓
Estimators (DM/TIS/DR)
    ↓
[estimators.py] estimate_policy_value()
```

---

## 2. 核心问题诊断 (重点)

### 问题 1: 连续动作重要性权重计算方式错误 ⚠️ **Critical**

#### 现象
- TIS/DR 估计值出现 `inf`/`nan` 或极端偏离真实值
- 测试中观察到重要性权重累积值超过 `1e4`（触发裁剪上界）
- 即使使用激进的权重裁剪 `[-10, 10]`，仍无法稳定估计

#### 根因分析
**当前实现** (`algorithm_adapter.py:compute_action_log_prob()`):
```python
# 直接使用概率密度比
log_prob_eval = eval_policy.actor.get_log_prob(obs, action)
log_prob_behavior = behavior_policy.actor.get_log_prob(obs, action)
importance_weight = exp(log_prob_eval - log_prob_behavior)
```

**问题所在**:
- 连续动作空间中，概率密度函数 (PDF) 的值可以 > 1
- 直接计算 `π_e(a|s) / π_b(a|s)` 会导致：
  - 当 `π_b(a|s)` 很小时，比值趋向无穷大
  - 当评价策略与行为策略差异较大时，密度比极不稳定
  - 累积重要性权重 `∏ᵗ ρₜ` 呈指数级增长/衰减

**正确做法** (参考 scope-rl):
使用**核函数相似度** (Kernel Similarity) 替代直接密度比：
```
w(a, a') = K((a - a') / h) / h^d
其中 K 是核函数（如高斯核），h 是带宽，d 是动作维度
```

#### 影响范围
- **TIS**: 完全不可靠，估计值随机波动
- **DR**: 由于依赖重要性权重，同样不可靠
- **DM**: 不受影响（不使用重要性权重）
- **PDIS**: 未实现，但若实现也会受影响

#### 相关代码位置
- `algorithm_adapter.py:155-180` - `compute_action_log_prob()`
- `algorithm_adapter.py:182-250` - `compute_action_log_prob_batch_dict()`
- `estimators.py:89-150` - `estimate_trajectory_value_tis()`
- `estimators.py:218-290` - `estimate_trajectory_value_dr()`

---

### 问题 2: DR 估计器数值不稳定 ⚠️ **High**

#### 现象
测试结果显示 DR 估计值极度偏离真实值：
```
真实值（在线评估）: -87.3
DR 估计值: -1.03e6
偏差: 11,800倍
```

#### 根因分析
DR 估计器的公式为：
```
V^DR = (1/n) Σᵢ [ Σₜ γᵗ ρₜ (rₜ - Q(sₜ,aₜ)) + Q(s₀,a₀) ]
```

**双重误差来源**:
1. **重要性权重问题** (继承自问题1)
   - 当前实现使用过于保守的裁剪: `log_weight ∈ [-10, 5]`
   - 即使裁剪后，累积权重仍可能极端偏离

2. **Q函数误差累积**
   - FQE 训练的 Q 函数本身存在近似误差
   - 当 `ρₜ` 很大时，`ρₜ (rₜ - Q(sₜ,aₜ))` 的误差被放大
   - 长轨迹中误差累积效应显著

**当前裁剪策略的问题**:
```python
# estimators.py:240-245
log_weight = np.clip(log_prob_eval - log_prob_behavior, -10, 5)
cumulative_weight = np.clip(cumulative_weight, 0, 1e4)
```
- `[-10, 5]` 意味着权重范围 `[4.5e-5, 148]`
- 对于长轨迹（T=100），累积权重仍可能达到 `148^100 ≈ ∞`
- 裁剪上界 `1e4` 过于宽松，无法有效控制方差

#### 影响范围
- DR 方法在当前实现下**完全不可用**
- 即使修复问题1，仍需重新设计裁剪策略

#### 相关代码位置
- `estimators.py:218-290` - `estimate_trajectory_value_dr()`
- `fqe.py:200-350` - FQE 训练逻辑

---

### 问题 3: PDIS 未实现 ⚠️ **Medium**

#### 影响
Per-Decision Importance Sampling (PDIS) 是 IS 类方法的重要变体：
- **理论优势**: 方差低于 TIS（不需要累积整条轨迹的权重）
- **适用场景**: 长轨迹、高方差环境
- **与 TIS 关系**: PDIS 通常比 TIS 更稳定

#### 缺失原因
从代码注释看，开发者意识到需要实现但尚未完成：
```python
# estimators.py 中只有 DM, TIS, DR 的实现
# PDIS 的函数签名和逻辑完全缺失
```

#### 相关代码位置
- `estimators.py` - 需要新增 `estimate_trajectory_value_pdis()` 函数

---

### 问题 4: 缺少 Self-Normalized 变体 ⚠️ **Low**

#### 影响
Self-Normalized Importance Sampling (SNIS) 的优势：
- 自动归一化重要性权重，减少方差
- 在权重分布不均匀时更稳定
- 是 IS 类方法的标准改进

#### 当前实现
所有 IS 类方法（TIS, DR）都使用普通 IS，未提供 SNIS 选项。

#### 相关代码位置
- `estimators.py:89-150` - TIS 实现
- `estimators.py:218-290` - DR 实现

---

## 3. 与 scope-rl 对比分析

### 3.1 重要性权重计算方式

| 项目 | gc-ope (当前) | scope-rl |
|------|---------------|----------|
| **连续动作处理** | 直接使用概率密度比 `π_e/π_b` | 使用核函数相似度 |
| **核函数类型** | 无 | 支持 Gaussian, Epanechnikov, Triangular, Cosine |
| **带宽选择** | 无 | 提供自动带宽选择（Scott's rule, Silverman's rule） |
| **数值稳定性** | 差（易溢出） | 好（核函数天然有界） |

### 3.2 连续动作处理方式

**scope-rl 的实现** (`scope_rl/utils.py`):
```python
def estimate_kernel_weights(
    action_behavior: np.ndarray,
    action_eval: np.ndarray,
    kernel: str = "gaussian",
    bandwidth: float = None
) -> np.ndarray:
    """
    使用核函数计算连续动作的相似度权重

    Parameters:
    - action_behavior: 行为策略采样的动作 (n_samples, action_dim)
    - action_eval: 评价策略的动作 (n_samples, action_dim)
    - kernel: 核函数类型
    - bandwidth: 带宽参数（若为None则自动选择）

    Returns:
    - weights: 相似度权重 (n_samples,)
    """
```

**关键优势**:
1. 核函数值域有界（如高斯核 ∈ [0, 1]），天然避免溢出
2. 带宽参数可调，平衡偏差-方差权衡
3. 支持多种核函数，适应不同数据分布

### 3.3 可复用代码指引

**推荐直接复用**:
- `scope_rl/utils.py:estimate_kernel_weights()` - 核函数权重计算
- `scope_rl/utils.py:select_bandwidth()` - 自动带宽选择
- `scope_rl/ope/continuous_ope.py` - 连续动作 OPE 完整实现

**需要适配的部分**:
- 输入格式：scope-rl 使用 numpy 数组，gc-ope 需要从 SB3 模型提取动作
- 观测空间：gc-ope 支持字典观测（GCRL），需要展平处理

---

## 4. 改进方案

### 方案 A: 核函数相似度权重 (针对问题1) ⚠️ **P0**

#### 算法思路
用核函数相似度替代概率密度比，计算重要性权重：

1. **评价策略动作采样**: 对每个状态 `sₜ`，从评价策略采样 `aₑ ~ πₑ(·|sₜ)`
2. **核函数相似度**: 计算 `w(aₜ, aₑ) = K((aₜ - aₑ) / h)`
3. **归一化**: `ρₜ = w(aₜ, aₑ) / Σⱼ w(aₜ, aⱼ)`（若使用多个样本）

#### 伪代码
```python
def compute_importance_weight_kernel(
    obs: np.ndarray,
    action_behavior: np.ndarray,
    eval_policy: BaseAlgorithm,
    kernel: str = "gaussian",
    bandwidth: float = None,
    n_samples: int = 10
) -> float:
    """
    使用核函数计算重要性权重

    Args:
        obs: 观测状态
        action_behavior: 行为策略采样的动作
        eval_policy: 评价策略
        kernel: 核函数类型
        bandwidth: 带宽（None则自动选择）
        n_samples: 评价策略采样数量

    Returns:
        importance_weight: 重要性权重
    """
    # 1. 从评价策略采样多个动作
    actions_eval = []
    for _ in range(n_samples):
        action, _ = eval_policy.predict(obs, deterministic=False)
        actions_eval.append(action)
    actions_eval = np.array(actions_eval)

    # 2. 计算核函数相似度
    if bandwidth is None:
        bandwidth = select_bandwidth(actions_eval, method="scott")

    # 3. 计算权重
    diff = actions_eval - action_behavior  # (n_samples, action_dim)
    if kernel == "gaussian":
        weights = np.exp(-0.5 * np.sum((diff / bandwidth)**2, axis=1))
    elif kernel == "epanechnikov":
        u = np.linalg.norm(diff / bandwidth, axis=1)
        weights = np.maximum(0, 0.75 * (1 - u**2))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # 4. 归一化
    importance_weight = np.mean(weights)
    return importance_weight
```

#### 可复用代码
- **核函数实现**: `scope_rl/utils.py:estimate_kernel_weights()`
- **带宽选择**: `scope_rl/utils.py:select_bandwidth()`
- **完整示例**: `scope_rl/ope/continuous_ope.py:ContinuousOPE`

#### 集成到 gc-ope
**修改位置**: `algorithm_adapter.py`

新增函数：
```python
def compute_importance_weight_kernel(
    obs: Union[np.ndarray, Dict],
    action_behavior: np.ndarray,
    eval_policy: BaseAlgorithm,
    kernel: str = "gaussian",
    bandwidth: float = None,
    n_samples: int = 10
) -> float:
    # 实现如上伪代码
    pass
```

修改调用：
```python
# estimators.py:estimate_trajectory_value_tis()
# 原代码:
# log_prob_eval = adapter.compute_action_log_prob(...)
# log_prob_behavior = adapter.compute_action_log_prob(...)
# weight = exp(log_prob_eval - log_prob_behavior)

# 新代码:
weight = adapter.compute_importance_weight_kernel(
    obs=obs_t,
    action_behavior=action_t,
    eval_policy=eval_policy,
    kernel="gaussian",
    n_samples=10
)
```

---

### 方案 B: PDIS 实现 (针对问题3) ⚠️ **P1**

#### 算法思路
Per-Decision Importance Sampling 对每个时间步单独计算重要性权重，避免累积：

```
V^PDIS = (1/n) Σᵢ Σₜ γᵗ ρₜ rₜ
其中 ρₜ = π_e(aₜ|sₜ) / π_b(aₜ|sₜ)  (单步权重，不累积)
```

**与 TIS 的区别**:
- TIS: `ρ₁:ₜ = ∏ₖ₌₁ᵗ ρₖ` (累积权重)
- PDIS: `ρₜ` (单步权重)

#### 伪代码
```python
def estimate_trajectory_value_pdis(
    trajectory: Dict[str, np.ndarray],
    eval_policy: BaseAlgorithm,
    behavior_policy: BaseAlgorithm,
    gamma: float,
    use_kernel: bool = True
) -> float:
    """
    Per-Decision Importance Sampling 估计

    Args:
        trajectory: 轨迹数据 {obs, actions, rewards, dones}
        eval_policy: 评价策略
        behavior_policy: 行为策略
        gamma: 折扣因子
        use_kernel: 是否使用核函数（连续动作）

    Returns:
        trajectory_value: 轨迹价值估计
    """
    obs = trajectory["obs"]
    actions = trajectory["actions"]
    rewards = trajectory["rewards"]
    T = len(rewards)

    trajectory_value = 0.0

    for t in range(T):
        # 1. 计算单步重要性权重
        if use_kernel:
            weight = compute_importance_weight_kernel(
                obs[t], actions[t], eval_policy
            )
        else:
            log_prob_eval = compute_action_log_prob(obs[t], actions[t], eval_policy)
            log_prob_behavior = compute_action_log_prob(obs[t], actions[t], behavior_policy)
            weight = np.exp(log_prob_eval - log_prob_behavior)

        # 2. 裁剪权重（可选）
        weight = np.clip(weight, 0.1, 10.0)

        # 3. 累加折扣奖励
        trajectory_value += (gamma ** t) * weight * rewards[t]

    return trajectory_value
```

#### 可复用代码
- **PDIS 实现**: `scope_rl/ope/importance_sampling.py:PDIS`
- **权重裁剪策略**: `scope_rl/ope/importance_sampling.py:clip_weights()`

#### 集成到 gc-ope
**修改位置**: `estimators.py`

新增函数：
```python
def estimate_trajectory_value_pdis(
    trajectory: Dict[str, np.ndarray],
    eval_policy: BaseAlgorithm,
    behavior_policy: BaseAlgorithm,
    adapter: AlgorithmAdapter,
    gamma: float = 0.99,
    use_kernel: bool = True,
    weight_clip: Tuple[float, float] = (0.1, 10.0)
) -> float:
    # 实现如上伪代码
    pass
```

在 `estimate_policy_value()` 中添加 PDIS 选项：
```python
if method == "pdis":
    trajectory_values = [
        estimate_trajectory_value_pdis(traj, eval_policy, behavior_policy, adapter, gamma)
        for traj in logged_dataset
    ]
```

---

### 方案 C: Self-Normalized 变体 (针对问题4) ⚠️ **P2**

#### 算法思路
Self-Normalized Importance Sampling 通过归一化权重减少方差：

```
V^SNIS = (Σᵢ ρᵢ Vᵢ) / (Σᵢ ρᵢ)
```

**优势**:
- 自动校正权重分布不均
- 在有限样本下方差更低
- 对权重估计误差更鲁棒

#### 伪代码
```python
def estimate_policy_value_snis(
    logged_dataset: List[Dict],
    eval_policy: BaseAlgorithm,
    behavior_policy: BaseAlgorithm,
    method: str = "tis"  # or "pdis"
) -> Tuple[float, float]:
    """
    Self-Normalized Importance Sampling

    Args:
        logged_dataset: 轨迹数据集
        eval_policy: 评价策略
        behavior_policy: 行为策略
        method: 基础方法（tis 或 pdis）

    Returns:
        (policy_value, std_error): 策略价值估计和标准误差
    """
    weighted_values = []
    weights = []

    for trajectory in logged_dataset:
        # 1. 计算轨迹价值和权重
        if method == "tis":
            value, weight = estimate_trajectory_value_tis_with_weight(trajectory, ...)
        elif method == "pdis":
            value, weight = estimate_trajectory_value_pdis_with_weight(trajectory, ...)

        weighted_values.append(weight * value)
        weights.append(weight)

    # 2. Self-Normalized 估计
    weighted_values = np.array(weighted_values)
    weights = np.array(weights)

    policy_value = np.sum(weighted_values) / np.sum(weights)

    # 3. 标准误差（需要特殊公式）
    n = len(logged_dataset)
    normalized_weights = weights / np.sum(weights)
    variance = np.sum(normalized_weights * (value - policy_value)**2)
    std_error = np.sqrt(variance / n)

    return policy_value, std_error
```

#### 可复用代码
- **SNIS 实现**: `scope_rl/ope/importance_sampling.py:SelfNormalizedIS`
- **方差估计**: `scope_rl/ope/importance_sampling.py:estimate_snis_variance()`

#### 集成到 gc-ope
**修改位置**: `estimators.py`

修改 `estimate_policy_value()` 函数签名：
```python
def estimate_policy_value(
    ...,
    method: str = "dm",
    self_normalized: bool = False  # 新增参数
) -> Tuple[float, Tuple[float, float]]:
    if self_normalized and method in ["tis", "pdis"]:
        return estimate_policy_value_snis(...)
    else:
        # 原有逻辑
        ...
```

---

## 5. 优先级排序与行动计划

### P0: 实现核函数相似度权重 (Critical)
**目标**: 修复连续动作重要性权重计算，使 TIS/DR 可用

**行动步骤**:
1. 从 scope-rl 复用 `estimate_kernel_weights()` 和 `select_bandwidth()`
2. 在 `algorithm_adapter.py` 中新增 `compute_importance_weight_kernel()`
3. 修改 `estimators.py` 中 TIS 和 DR 的权重计算逻辑
4. 运行测试 `tests/algorithm/ope/test_ope.py`，验证数值稳定性

**预期效果**:
- TIS 估计值不再出现 inf/nan
- TIS 估计值与在线评估结果偏差 < 20%

---

### P1: 实现 PDIS (High)
**目标**: 提供方差更低的 IS 变体

**行动步骤**:
1. 在 `estimators.py` 中新增 `estimate_trajectory_value_pdis()`
2. 集成核函数权重计算（复用 P0 的实现）
3. 在 `estimate_policy_value()` 中添加 "pdis" 选项
4. 编写单元测试 `test_pdis()`

**预期效果**:
- PDIS 方差低于 TIS（尤其在长轨迹场景）
- PDIS 估计值无偏（与 TIS 理论一致）

---

### P2: 实现 Self-Normalized 变体 (Medium)
**目标**: 在高方差场景下提供更稳定的估计

**行动步骤**:
1. 修改 `estimate_trajectory_value_tis()` 和 `estimate_trajectory_value_pdis()`，返回权重
2. 在 `estimators.py` 中新增 `estimate_policy_value_snis()`
3. 在 `estimate_policy_value()` 中添加 `self_normalized` 参数
4. 编写单元测试 `test_snis()`

**预期效果**:
- SNIS 方差低于普通 IS
- 在权重分布不均匀时更稳定

---

### P3: 优化 DR 估计器 (Low)
**目标**: 在修复问题1后，重新调优 DR 的裁剪策略

**行动步骤**:
1. 完成 P0（核函数权重）
2. 实验不同的权重裁剪范围
3. 调整 FQE 训练超参数（学习率、网络容量）
4. 验证 DR 估计值偏差 < 20%

**预期效果**:
- DR 估计值不再极端偏离
- DR 方差低于 TIS（理论优势）

---

## 6. 总结

### 核心发现
1. **Critical**: 连续动作重要性权重计算方式根本性错误，导致 TIS/DR 不可用
2. **High**: DR 估计器数值极度不稳定，需要修复权重计算 + 重新设计裁剪策略
3. **Medium**: 缺少 PDIS 实现，限制了方法选择
4. **Low**: 缺少 Self-Normalized 变体，高方差场景下缺少稳定选项

### 关键改进方向
- **立即行动**: 实现核函数相似度权重（P0）
- **短期目标**: 实现 PDIS（P1）
- **长期优化**: Self-Normalized 变体（P2）+ DR 调优（P3）

### 可复用资源
- `scope_rl/utils.py` - 核函数实现
- `scope_rl/ope/continuous_ope.py` - 连续动作 OPE 完整实现
- `scope_rl/ope/importance_sampling.py` - IS 类方法实现

---

**文档版本**: v1.0
**创建时间**: 2026-01-12
**作者**: Claude Code (基于代码分析)
