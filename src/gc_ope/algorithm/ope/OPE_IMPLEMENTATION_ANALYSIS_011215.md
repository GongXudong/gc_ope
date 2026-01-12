# OPE长期改进实现计划

## 目标
实现分析文档中的长期改进目标，同时解决CLAUDE.md中的已知潜在问题第2点（核函数相似度计算值过小）。

## 改进路线图状态

| 阶段 | 任务 | 状态 |
|------|------|------|
| 短期 | 添加 kernel_utils.py | ✅ 已完成 |
| 短期 | 修改 TIS 支持核函数 | ✅ 已完成 |
| 短期 | 添加单元测试 | ✅ 已完成 |
| 中期 | 添加 PDIS 估计器 | ✅ 已完成 |
| 中期 | 修改 DR 支持核函数 | ✅ 已完成 |
| 中期 | 扩展 OPEInputs | ✅ 已完成 |
| 中期 | 添加集成测试 | ✅ 已完成 |
| **长期** | **重构为类继承架构** | **本次重点** |
| **长期** | **支持多种核函数** | **本次重点** |
| **长期** | **添加 bandwidth 自动选择** | **本次重点** |

---

## 问题诊断：CLAUDE.md已知潜在问题第2点

### 问题描述
"重要性权重计算为极小值，用当前实现的Gaussian、epanechnikov核函数计算`similarity`，就已经很小了，计算`similarity_weight`就更小，导致到`weight`时也小。"

### 根因分析

1. **高斯核返回概率密度值**：
   ```python
   # 当前实现
   return np.exp(-distance / (2 * bandwidth**2)) / np.sqrt(2 * np.pi * bandwidth**2)
   ```
   - 归一化因子 `1/sqrt(2πh²)` 在 h=1.0 时约为 0.4
   - 高维空间中，即使动作接近，密度值也很小

2. **累积乘积指数衰减**：
   ```python
   similarity_weight = similarity.prod()  # 或 np.cumprod(similarity)
   ```
   - 20步轨迹：0.4^20 ≈ 1e-8
   - 权重趋近于零

3. **缺少自归一化机制**：
   - 当前实现直接使用原始权重
   - 没有跨轨迹归一化

### 解决方案

参考scope-rl的实现，采用以下方案：

| 方案 | 效果 | 优先级 |
|------|------|--------|
| 自归一化估计器 (SNTIS/SNPDIS/SNDR) | 权重除以平均权重，稳定数值 | P0 |
| 纯相似度核函数（无归一化因子） | 返回[0,1]范围的相似度 | P0 |
| 自动bandwidth选择 (Silverman's rule) | 根据数据自动选择最优h | P1 |
| 更多核函数选择 | 有界支持核函数更稳定 | P2 |

---

## 1. 架构重构设计

### 1.1 当前架构问题

- 函数式实现，代码重复多
- 核函数版本与普通版本分离
- 难以扩展新功能（如自归一化）

### 1.2 新架构设计

```
src/gc_ope/algorithm/ope/
├── kernel_utils.py          # 核函数工具（扩展）
├── bandwidth_selection.py   # 新增：bandwidth自动选择
├── estimators/              # 新增：估计器包
│   ├── __init__.py
│   ├── base.py              # 基类定义
│   ├── dm.py                # DM估计器
│   ├── tis.py               # TIS估计器（含SN版本）
│   ├── pdis.py              # PDIS估计器（含SN版本）
│   └── dr.py                # DR估计器（含SN版本）
├── estimators.py            # 保留：向后兼容的函数式API
└── ...
```

### 1.3 类继承架构

```python
# base.py
class BaseOPEEstimator(ABC):
    """OPE估计器基类"""
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma

    @abstractmethod
    def compute_trajectory_values(self, inputs: OPEInputs) -> np.ndarray:
        """计算轨迹级别的估计值"""
        pass

    def estimate(self, inputs: OPEInputs, ci_method: str = "bootstrap", **ci_kwargs) -> EstimateResult:
        """计算估计值和置信区间"""
        trajectory_values = self.compute_trajectory_values(inputs)
        return compute_estimate_with_ci(trajectory_values, ci_method=ci_method, **ci_kwargs)


class BaseISEstimator(BaseOPEEstimator):
    """重要性采样估计器基类"""
    def __init__(self, gamma: float = 0.99, use_kernel: bool = False,
                 kernel: str = "gaussian", bandwidth: float | str = "auto",
                 self_normalize: bool = False):
        super().__init__(gamma)
        self.use_kernel = use_kernel
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.self_normalize = self_normalize
```

---

## 2. 核函数扩展

### 2.1 修改kernel_utils.py

**关键改进**：添加纯相似度版本（无归一化因子）

```python
# 新增：纯相似度核函数（返回[0,1]范围）
def gaussian_similarity(x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
    """Gaussian similarity (without normalization factor).

    Returns values in (0, 1] range, where 1 means identical.
    """
    distance = l2_distance(x, y)
    return np.exp(-distance / (2 * bandwidth**2))


def epanechnikov_similarity(x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
    """Epanechnikov similarity (without normalization factor).

    Returns values in [0, 1] range.
    """
    distance = np.sqrt(l2_distance(x, y))
    u = distance / bandwidth
    return np.where(u < 1, 1 - u**2, 0.0)
```

### 2.2 新增核函数

```python
# 新增：三角核
def triangular_similarity(x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
    """Triangular similarity. Returns values in [0, 1] range."""
    distance = np.sqrt(l2_distance(x, y))
    u = distance / bandwidth
    return np.where(u < 1, 1 - u, 0.0)


# 新增：余弦核
def cosine_similarity(x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
    """Cosine similarity. Returns values in [0, 1] range."""
    distance = np.sqrt(l2_distance(x, y))
    u = distance / bandwidth
    return np.where(u < 1, np.cos(np.pi * u / 2), 0.0)


# 新增：均匀核
def uniform_similarity(x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
    """Uniform similarity. Returns 1 if within bandwidth, 0 otherwise."""
    distance = np.sqrt(l2_distance(x, y))
    return np.where(distance < bandwidth, 1.0, 0.0)
```

---

## 3. Bandwidth自动选择

### 3.1 新建bandwidth_selection.py

```python
"""Automatic bandwidth selection for kernel-based OPE estimators."""

import numpy as np
from typing import Literal


def silverman_bandwidth(data: np.ndarray) -> float:
    """Silverman's rule of thumb for bandwidth selection.

    h = 0.9 * min(std, IQR/1.34) * n^(-1/5)

    Args:
        data: Input data, shape (n_samples, n_dim).

    Returns:
        Optimal bandwidth.
    """
    n, d = data.shape
    std = data.std(axis=0).mean()
    iqr = (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)).mean()

    # Silverman's rule
    h = 0.9 * min(std, iqr / 1.34) * (n ** (-1 / 5))
    return max(h, 1e-6)  # 防止过小


def scott_bandwidth(data: np.ndarray) -> float:
    """Scott's rule for bandwidth selection.

    h = 1.06 * std * n^(-1/5)

    Args:
        data: Input data, shape (n_samples, n_dim).

    Returns:
        Optimal bandwidth.
    """
    n, d = data.shape
    std = data.std(axis=0).mean()
    h = 1.06 * std * (n ** (-1 / 5))
    return max(h, 1e-6)


def select_bandwidth(
    data: np.ndarray,
    method: Literal["silverman", "scott", "median"] = "silverman"
) -> float:
    """Select bandwidth using specified method.

    Args:
        data: Input data, shape (n_samples, n_dim).
        method: Selection method.

    Returns:
        Selected bandwidth.
    """
    if method == "silverman":
        return silverman_bandwidth(data)
    elif method == "scott":
        return scott_bandwidth(data)
    elif method == "median":
        # Median heuristic: h = median(||x_i - x_j||)
        from scipy.spatial.distance import pdist
        distances = pdist(data)
        return float(np.median(distances))
    else:
        raise ValueError(f"Unknown method: {method}")
```

---

## 4. 自归一化估计器

### 4.1 核心实现

自归一化是解决权重过小问题的关键：

```python
def self_normalize_weights(weights: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Self-normalize importance weights.

    w_normalized = w / (mean(w) + epsilon)

    Args:
        weights: Raw importance weights, shape (n_trajectories,) or (n_trajectories, n_steps).
        epsilon: Small constant for numerical stability.

    Returns:
        Self-normalized weights.
    """
    mean_weight = weights.mean(axis=0, keepdims=True)
    return weights / (mean_weight + epsilon)
```

### 4.2 SNTIS实现示例

```python
class SelfNormalizedTIS(TISEstimator):
    """Self-Normalized Trajectory-wise Importance Sampling.

    Normalizes weights by their mean to bound variance.
    Trades unbiasedness for stability.
    """

    def compute_trajectory_values(self, inputs: OPEInputs) -> np.ndarray:
        # 计算原始权重和回报
        raw_weights, returns = self._compute_raw_weights_and_returns(inputs)

        # 自归一化
        normalized_weights = self_normalize_weights(raw_weights)

        # 加权回报
        return normalized_weights * returns
```

---

## 5. 实现步骤

### Step 1: 扩展kernel_utils.py (~50行新增)
1. 添加纯相似度版本核函数（gaussian_similarity, epanechnikov_similarity）
2. 添加新核函数（triangular, cosine, uniform）
3. 更新KERNEL_FUNCTIONS注册表

### Step 2: 创建bandwidth_selection.py (~60行)
1. 实现silverman_bandwidth()
2. 实现scott_bandwidth()
3. 实现select_bandwidth()统一接口

### Step 3: 创建estimators/base.py (~100行)
1. 定义BaseOPEEstimator抽象基类
2. 定义BaseISEstimator基类（支持kernel和self_normalize）
3. 实现self_normalize_weights()工具函数

### Step 4: 创建estimators/tis.py (~120行)
1. 实现TISEstimator类
2. 实现SelfNormalizedTIS类
3. 支持kernel和auto bandwidth

### Step 5: 创建estimators/pdis.py (~120行)
1. 实现PDISEstimator类
2. 实现SelfNormalizedPDIS类

### Step 6: 创建estimators/dr.py (~120行)
1. 实现DREstimator类
2. 实现SelfNormalizedDR类

### Step 7: 创建estimators/dm.py (~60行)
1. 实现DMEstimator类

### Step 8: 更新estimators.py (~20行修改)
1. 保留函数式API向后兼容
2. 内部调用新的类实现

### Step 9: 添加测试 (~100行)
1. 测试自归一化功能
2. 测试bandwidth自动选择
3. 测试新核函数

---

## 6. 关键文件

| 文件 | 操作 | 行数 | 说明 |
|------|------|------|------|
| `kernel_utils.py` | 修改 | +50 | 添加纯相似度核函数 |
| `bandwidth_selection.py` | 新建 | ~60 | bandwidth自动选择 |
| `estimators/base.py` | 新建 | ~100 | 估计器基类 |
| `estimators/tis.py` | 新建 | ~120 | TIS + SNTIS |
| `estimators/pdis.py` | 新建 | ~120 | PDIS + SNPDIS |
| `estimators/dr.py` | 新建 | ~120 | DR + SNDR |
| `estimators/dm.py` | 新建 | ~60 | DM估计器 |
| `estimators/__init__.py` | 新建 | ~30 | 导出接口 |
| `estimators.py` | 修改 | +20 | 向后兼容 |
| `test_self_normalized.py` | 新建 | ~100 | 自归一化测试 |

---

## 7. 验证方案

### 7.1 单元测试
```bash
# 测试新核函数
pytest tests/algorithm/ope/test_kernel.py -v

# 测试bandwidth选择
pytest tests/algorithm/ope/test_bandwidth.py -v

# 测试自归一化估计器
pytest tests/algorithm/ope/test_self_normalized.py -v
```

### 7.2 集成测试
```bash
pytest tests/algorithm/ope/test_estimators_integration.py -v
```

### 7.3 端到端验证
```bash
cd tests/algorithm/ope && python test_ope.py
```

### 7.4 预期效果
1. **解决权重过小问题**：自归一化后权重在合理范围内
2. **数值稳定**：所有估计器输出有限值
3. **自动bandwidth**：根据数据自动选择合适的bandwidth
4. **向后兼容**：原有函数式API继续工作

---

## 8. 注意事项

1. **自归一化的代价**：引入偏差，但大幅降低方差
2. **bandwidth选择**：Silverman's rule适用于单峰分布，多峰分布可能需要调整
3. **代码规范**：每个文件不超过300行
4. **向后兼容**：保留原有estimators.py的函数式API

---

## 9. 总结

本次长期改进将：
1. **解决CLAUDE.md问题第2点**：通过自归一化和纯相似度核函数
2. **重构为类继承架构**：提高代码可维护性和扩展性
3. **支持多种核函数**：5种核函数可选
4. **添加bandwidth自动选择**：Silverman's rule和Scott's rule

预计新增代码约700行，分布在10个文件中，每个文件聚焦单一功能。
