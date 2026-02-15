# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Goal-Conditioned Off-Policy Evaluation (gc-ope) - A research project for evaluating goal-conditioned reinforcement learning policies using off-policy evaluation techniques. Assesses goal-achievement abilities of trained policies without requiring online interaction.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
pytest tests/
pytest tests/algorithm/ope/      # OPE tests only
pytest tests/env/                # Environment tests only

# Train policies (examples)
uv run scripts/train.py experiment_name=easy/ppo/seed_1 algo=ppo env=flycraft_for_ppo ...
uv run scripts/train.py experiment_name=easy/sac/seed_1 algo=sac env=flycraft_for_sac ...

# Evaluate checkpoints
uv run scripts/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/... env=flycraft_for_ppo ...
uv run scripts/evaluate_my_reach.py algo_type=sac ckpt_dir=checkpoints/... env=my_reach_for_sac ...
```

## Architecture

**Source code:** `src/gc_ope/`

- **algorithm/ope/** - Off-Policy Evaluation: FQE (Fitted Q Evaluation), importance sampling estimators, logged dataset handling
- **algorithm/curriculum/** - Curriculum learning wrappers (MEGA, OMEGA)
- **env/** - Environment factories and goal utilities for FlyCraft, Reach, Push, Slide, PointMaze, AntMaze
- **evaluate/** - KDE-based evaluators and result containers

**Configuration:** Hydra-based YAML configs in `configs/` with command-line overrides

**Scripts:**
- `scripts/train_policy/train.py` - Main training entry point
- `scripts/evaluate_ckpt/` - Per-environment evaluation scripts

## Supported Environments & Algorithms

**Environments:** FlyCraft, MyReach, MyPush, MySlide, PointMaze, AntMaze

**Algorithms:** PPO, SAC, HER (via Stable-Baselines3)

**OPE Methods:** FQE, importance sampling variants, KDE-based distribution evaluation

## Execution Rule

Execute files from their parent directory.

## 当前开发目标
**更新时间**：2026/1/12 9:37

### 核心开发任务
构建与 Stable-Baselines3（SB3）深度耦合的离线策略评估（OPE）算法库，充分复用 SB3 内部 API 实现功能开发，确保库与 SB3 生态的兼容性和调用效率。

### 测试验证任务
在指定 RL 环境与算法组合下，验证四类 OPE 算法的拟合效果：
- **测试环境**（按难度从高到低）：my_reach、my_push、my_slide、flycraft
- **测试 RL 算法**：Hindsight Experience Replay（HER）、Soft Actor-Critic（SAC）
- **待验证 OPE 算法**：Direct Method (DM)、Trajectory-wise Importance Sampling (TIS)、Per-Decision Importance Sampling (PDIS)、Doubly Robust (DR) 

#### 预期效果
当轨迹样本量充足时，满足以下指标：
- 无偏估计算法（TIS、PDIS）：OPE 估计结果与在线评估结果（近似真实值）差异趋近于一致；
- 有偏估计算法（DM、DR）：OPE 估计结果与在线评估结果的偏差不超过 20%。

### 已知潜在问题
1. **更新时间**：2026/1/12 9:37 **[已解决 2026/1/13]**
重要性权重计算可能出现无穷大等异常情况，导致 TIS、DR 估计器的输出结果绝对值显著偏离在线评估结果。推测根因：处理连续动作时，误将其按离散动作逻辑计算重要性权重（直接采用"评价策略动作概率密度 ÷ 行为策略动作概率密度"的计算方式），未使用核函数相似度进行适配。
**解决方案**：实现了核函数相似度权重（kernel similarity weight），支持 gaussian、epanechnikov、triangular、cosine、uniform 五种核函数。

2. **更新时间**：2026/1/12 15:39 **[已解决 2026/1/13]**
重要性权重计算为极小值，用当前实现的Gaussian、epanechnikov核函数计算`similarity`，就已经很小了，计算`similarity_weight`就更小，导致到`weight`时也小。
**解决方案**：
- 使用纯相似度核函数（无归一化因子），返回 [0,1] 范围的值
- 使用 log-space 计算累积权重，避免下溢
- 将默认 bandwidth 选择方法从 Silverman's rule 改为 median heuristic（更适合 OPE 场景）
- 实现自归一化估计器（SelfNormalizedTIS/PDIS/DR），通过 w_normalized = w / mean(w) 稳定数值

**验证结果**（my_reach 环境，真实值 -3.44 ± 1.98）：
- SN-TIS (Gaussian): -4.05（偏差 ~18%）
- PDIS (kernel, Uniform): -3.64（偏差 ~6%）

3. **更新时间**：2026/1/17 8:40 
- flycraft环境，her算法，测试中DM估计器里FQE的训练loss急剧升高，迅速大于1（不像其他环境下loss虽然在升，但稳定小于1），导致DM算法估计过高，DR类算法估计过小。详见日志[flycraft-her日志](scripts/ope/outputs/flycraft_her_10000_all_20260116_213857)
- 进一步挖原因，是因为flycraft环境下单episode轨迹step很长，大概都在100~300（最长400），不像其他三个环境reach push slide大概都在10左右（最长50）。

### 暂时搁置内容
针对目标条件强化学习（GCRL）的 OPE 算法专项开发暂不推进。当前处理方式为：将 GCRL 中字典类型的观测值拼接为向量，按普通 RL 场景统一处理。后续仅在“针对 GCRL 特殊性的优化可显著提升其 OPE 性能”的前提下，重新评估该方向的开发优先级。

## 编码规范要求
1. 单代码文件代码行数尽量不超过300行
2. 模块化代码文件，每个文件聚焦单一核心功能，避免功能冗余或混乱，确保模块间低耦合、高内聚
3. 对于重复出现、逻辑独立的代码片段，必须封装为可复用函数，明确函数输入输出、功能职责，提升代码复用性与可维护性；非重复但逻辑复杂、长度较长的代码片段，建议封装为函数，增强代码可读性。