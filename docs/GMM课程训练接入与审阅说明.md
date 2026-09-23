# GMM 接入 Push / SAC / OMEGA-PE 训练

这次将离线验证后定版的 GMM 用于真实课程学习。训练从随机初始化的 SAC 开始，能力估计器只接收本次训练过程中产生的评估结果。没有加载之前 vanilla SAC 的 checkpoint 或 CSV，也没有将离线拟合结果带入训练。

本次新增专用配置、五组种子的启动脚本和测试，复用重构时已经实现的估计器适配接口。训练主程序、SAC、评估回调、MEGA 和 OMEGA 的代码本次均未修改。正式百万步训练尚未启动，以下验证只证明接入可运行，不证明最终训练效果。

课程日志与后续制图已进一步对照师兄流程，见 [旧训练与数据制作流程核查](旧训练与数据制作流程核查.md)。课程沿用原 wrapper 的文本输出和 `train_log_process.py`，没有另加课程 CSV 格式。

## 1. 旧 KDE 在训练中负责什么

按实际调用顺序阅读：

1. `scripts/train_policy/train.py` 创建训练、最终评估、周期评估三组环境，并构造 SAC 和回调。
2. `src/gc_ope/env/get_vec_env.py` 只给训练环境加 `OMEGAWrapper`；另外两组环境保留原始目标采样，用于衡量原任务的成功率。
3. `MyEvalCallbackSTAT` 每 10000 个训练步执行 96 个评估 episode，将目标、成功标记、累计奖励交给训练环境的 `sync_evaluation_stat`。
4. `WeightedEvaluationResultContainer` 每接收一批记录，将已有权重乘以 0.9，新记录权重置为 1。成功和失败记录都保留；KDE/GMM 拟合时只取成功目标。
5. `MEGAWrapper.estimate_p_ag` 在数据更新后重新拟合能力分布，并以历史成功目标上的最小密度作为候选目标筛选阈值。没有成功目标时仍随机探索。
6. `OMEGAWrapper` 计算目标均匀分布与能力分布的 KL，据此更新随机探索比例 `alpha`。每个新 episode 以 `alpha` 概率随机选目标；否则先由环境采样 100 个候选目标，再用 MEGA 规则选目标。

MEGA 规则保留原实现：选取密度不低于阈值的候选中密度最低者；若所有候选都低于阈值，则取密度最高者。**课程目标来自环境采样出的候选集合，不是直接调用 GMM.sample 生成。** SAC 的损失、策略网络、经验回放和动作采样没有因此改写。

## 2. 替换与兼容边界

调用关系为：

```text
原 MyEvalCallbackSTAT → 原 sync_evaluation_stat → 原历史评估容器
                                            ↓
                          PlanarEvaluator → GMMEvaluator
                                            ↓
                          原 OMEGA / MEGA 目标选择 → 原 SAC
```

相关实现已随前面的重构提交在同一分支：

| 文件 | 职责 |
| --- | --- |
| `src/gc_ope/algorithm/curriculum/mega_wrapper.py` | 可选 `estimator_config` 注入；不传时保留 KDE 默认路径 |
| `src/gc_ope/algorithm/curriculum/omega_wrapper.py` | 透传估计器配置；保留 KL、alpha 和目标选择流程 |
| `src/gc_ope/evaluate/evaluator_planar.py` | 环境保留 XYZ 目标，模型只拟合 XY；复制标签与时间权重，不修改原容器 |
| `src/gc_ope/evaluate/evaluator_factory.py` | 构造最终版估计器 |
| `src/gc_ope/evaluate/evaluator_gmm.py` | 成功目标标准化、直接加权 GMM 拟合与密度查询 |
| `src/gc_ope/evaluate/utils/weighted_gmm.py` | 加权 EM，优化样本权重加权的对数似然 |

Push 的 z 恒定为物体高度的一半。旧 `env/utils/my_push/desired_goal_utils.py` 已只枚举 x、y，且积分单元为 `dx*dy`；因此平面适配不改变目标空间。GMM 的 StandardScaler 在成功目标上做不加权拟合，EM 中再使用时间权重，与定版离线 GMM 一致。候选打分和阈值使用同一密度尺度；网格归一化也会抵消坐标变换产生的常数因子。

GMM 为 5 个 full-covariance 分量；数据早期可用的不同成功点少于 5 时减少有效分量。`reg_covar=0.05` 加在标准化空间的协方差对角线上，保留定版参数。`n_init=1`、`max_iter=200`、`tol=0.001`、`random_state=0`。直接使用样本权重进行 EM，不恢复旧的重采样 GMM。

## 3. 在线 KL 与离线 KL 不同

训练中的量是 `KL(p_dg || p_ag)`：均匀目标分布对历史估计能力分布。继续使用旧网格积分：

```text
normalized_p_j = p(g_j) / sum_l p(g_l) / dV
KL_grid = u_density * sum_j [log(u_density) - log(normalized_p_j)] * dV
alpha = 1 / max(b + KL_grid, 1)
```

网格步长仍为 `[0.02, 0.02, 0.02]`，Push 的 `dV=0.02*0.02`；保留旧网格端点和求和规则。适配器调用 `uniform_grid_kl`，用 logsumexp 计算同一归一化，避免先指数化造成密度下溢。原 KDE 默认路径未改。

这里没有第二个“当前 checkpoint 全量拟合”的参考模型，也不调用离线实验的 MC 10000×5。离线协议用于验证估计质量，在线 KL 用于控制探索，两者职责不同。无成功记录时 `KL=inf, alpha=0` 是旧流程中的占位状态，目标选择实际退回环境随机采样。

## 4. 专用配置和旧正式实验的对齐

新增配置：`configs/train/env/my_push_omega_gmm.yaml`，继承 `my_push_omega.yaml`。比较基准是旧 `scripts/train_policy/shells/my_push/curriculum/sac_omega.sh` 的前五个正式实验，而不是该 YAML 未覆盖的默认值（默认 `sample_n=10,b=-3`）。本配置显式固定 `sample_n=100,b=0,kappa=0.9`，并列出全部 GMM 参数。

| 项目 | 正式 GMM 训练 |
| --- | --- |
| 场景 / 算法 | MyPushSparse-v0 / SAC，原 `sac_for_my_push` 配置 |
| 课程 / 数据来源 | OMEGA + MEGA / 周期性 evaluation，即 PE 路线 |
| 每次训练步数 | 1000000 |
| 网络 / batch / buffer | 原 `[256,256]` / 256 / 200000 |
| 训练环境数 | 1 |
| 周期评估 | 每 10000 步；16 个环境 × 每环境 6 个 episode = 96 条 |
| 训练后最终评估 | 16 个环境 × 每环境 30 个 episode = 480 条 |
| 历史折扣 / 候选数 / b | 0.9 / 100 / 0 |
| checkpoint 与 replay buffer | 每 10000 步保存一次，沿用原回调 |
| 设备 | SAC 沿用原 CUDA 配置；GMM 使用 CPU |
| 五次训练的调度 | 逐 seed 串行，每次训练内部并行评估 |

在线每批 96 条是原正式实验设置；不替换为离线每 checkpoint 抽 100 条。候选目标的 100、离线抽样的 100、周期评估的 96 是三件不同的事。

| 重复编号 | SAC seed | 训练环境 seed | 最终评估 seed | 周期评估 seed |
| --- | --- | --- | --- | --- |
| 1 | 5 | 2 | 8 | 9 |
| 2 | 12 | 15 | 14 | 17 |
| 3 | 26 | 27 | 23 | 29 |
| 4 | 37 | 31 | 36 | 33 |
| 5 | 43 | 46 | 49 | 44 |

这五组值逐项来自旧正式脚本。不同种子的训练环境和评估环境仍独立创建；GMM 的内部随机种子固定为 0，与离线定版配置一致。

## 5. 检查配置与启动

在本分支的仓库根目录执行；环境仍为 Conda `gc_ope`。不需要复制之前的 checkpoint 或实验 CSV。

```bash
# 只展示五条训练命令，不创建实验目录，不启动训练。
conda run -n gc_ope bash scripts/train_policy/shells/my_push/curriculum/sac_omega_gmm.sh --dry-run

# 检查第一组种子的 Hydra 展开配置，不开始训练。
conda run -n gc_ope env PYTHONPATH=src python scripts/train_policy/train.py \
  experiment_name=sac/omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_1 \
  algo=sac_for_my_push algo.seed=5 env=my_push_omega_gmm \
  env.train_env.seed=2 env.evaluation_env.seed=8 env.callback_env.seed=9 \
  callback=sac_omega evaluate=my_push train_steps=1e6 --cfg job --resolve

# 审阅后只运行第一组，或者去掉 --seed 1 顺序运行全部五组。
conda run --no-capture-output -n gc_ope bash scripts/train_policy/shells/my_push/curriculum/sac_omega_gmm.sh --seed 1
```

脚本固定 `PYTHONPATH` 指向当前仓库 `src`，避免 editable install 意外导入另一个 worktree。BLAS/OMP/MKL 单线程，防止多个评估环境过度争用线程。**这不是离线实验的 5×4 worker 调度。** 五个完整训练顺序执行，不同时创建五套训练和评估进程。

新实验名包含 `omega_gmm`，不会使用旧 `omega_...` 目录。第 N 次训练输出：

```text
logs_in_process/my_push/sac/my_push_sac_omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_N.txt
  # 完整训练输出，包含每次课程选点；旧 behavioral-goals notebook 从此解析。
checkpoints/my_push/sac/omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_N/
  best_model.zip
  rl_model_<步数>_steps.zip
  rl_model_replay_buffer_<步数>_steps.pkl
logs/my_push/sac/omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_N/
  console.log             # 完整控制台输出
  progress.csv           # 原 SB3 指标，包括 eval/success_rate、env/alpha 等
  evaluations.npz        # 每次回调的成功率相关原始结果
  hydra/.hydra/          # 本次配置和命令行覆盖项
```

已有同名日志/checkpoint 目录或课程文本日志时启动脚本会停止，避免覆盖；同 seed 同时启动也会在原子创建日志目录时被阻止。训练失败由 `pipefail` 返回失败状态。此脚本没有实现完整恢复 SAC + replay buffer + 历史评估容器的断点续训，不能将重新启动称为续跑。中断后的目录保留用于检查；后续从头重跑应使用另一个 `experiment_name`。

## 6. 本次验证结果与范围

在 `gc_ope_refactor` 中设置 `PYTHONPATH=src`，仅运行单个测试文件：

- `tests/evaluate/test_push_gmm_training.py`：**7 passed**。包括五组新旧 Hydra 配置对照、参数与定版离线 GMM 一致、启动脚本 dry-run/单 seed/非法 seed，以及实际 SAC + 真实 Push 评估回调。
- 回调集成测试中，为快速且确定地覆盖成功后的拟合路径，仅测试环境设为两步、近距离目标；成功标记由物理环境产生，没有手工伪造评估记录。两个回调仍各产生 96 条记录，检查权重从全 1 变为前 96 条 0.9、后 96 条 1，完成两次加权 GMM 拟合、SAC 梯度更新、模型和 replay buffer 保存；另强制进入 MEGA 分支验证选中的确是按原规则评分的候选目标。
- `tests/evaluate/test_curriculum_estimators.py`：**9 passed**。覆盖四种新估计器的真实环境接口、默认 KDE、数据不足回退、子进程统计同步。NN 两次更新的快速测试有预期的不收敛警告。
- 直接通过原 `scripts/train_policy/train.py` 运行 **64 步 CPU SAC**，使用新的 GMM 环境配置和真实 `SubprocVecEnv`，在 32、64 步各评估两次并保存模型/replay buffer，最后完成两个 episode 的评估，进程退出码 0。为控制验证成本，仅本次 smoke 减小网络、buffer、评估数量和进程数。原始目标分布下此短运行无成功样本，走随机回退；活跃 GMM 拟合路径由上述 96×2 回调测试覆盖，不能把 smoke 描述为已验证 GMM 提升成功率。

本机完整 smoke 日志在 `logs/gmm_curriculum_validation/hydra_smoke.log`，展开配置在同目录 `hydra_smoke/.hydra/`，回归测试日志在 `curriculum_tests.log`。这些运行产物按项目约定不提交 Git。

复验命令：

```bash
conda run -n gc_ope env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  pytest tests/evaluate/test_push_gmm_training.py -q
conda run -n gc_ope env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  pytest tests/evaluate/test_curriculum_estimators.py -q
```

建议审阅顺序：先看专用 YAML 和五条训练命令，再看新旧配置一致性测试，最后沿第 1、2 节的调用链检查历史权重、密度阈值和 alpha。完整训练性能、长期拟合耗时、CUDA 正式运行和百万步稳定性需要下一阶段实际训练验证，本次没有启动全量训练。
