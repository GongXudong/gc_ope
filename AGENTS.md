# AGENTS.md

## 环境

- 本项目用 conda 环境 `gc_ope` 运行，不用 uv/.venv（`.venv` 已弃用，可删除）。
- 运行任何命令前缀：`conda run -n gc_ope <cmd>`，或直接 `conda activate gc_ope`。
- 测试命令：`conda run -n gc_ope pytest tests/evaluate/test_evaluator_gmm.py -q`。跑单个测试文件，不要跑整个 `tests/`（部分测试会挂很久/超时）。

## 命令模式

- 训练/评估脚本都是 hydra 配置驱动：`conda run -n gc_ope scripts/train_policy/train.py <key=value...>`，参考 `scripts/*/shells/**/*.sh` 里的实际命令。
- 训练实验名（`experiment_name`）决定 checkpoint 落盘路径：`checkpoints/<env>/<experiment_name>/`。
- 评估脚本（`scripts/evaluate_new/`）读取一个 checkpoint 目录，对该目录下所有 `*.zip` 生成同名 CSV。

## 架构要点

- `src/gc_ope/` 分四块：`algorithm/`（SAC/PPO/HER 训练 + curriculum wrapper）、`env/`（6 个场景：my_reach/my_push/my_slide/my_point_maze/my_ant_maze/flycraft）、`evaluate/`（KDE/GMM 估计器 + 时间衰减评估结果容器）、`utils/`。
- 课程学习核心链路：`SyncEvaluationResultWrapper` → `MEGAWrapper`（KDE 拟合成功目标分布 p̂_ag，按 rig/discern/mega 三种方式采样 behavioral goal）→ `OMEGAWrapper`（在 MEGA 之上加 KL 计算和 α 混合全局随机探索）。
- wrapper 类型由 hydra 配置 `env` 决定：如 `env=my_push`（vanilla）vs `env=my_push_omega`（带 curriculum wrapper）。
- `callback` 决定训练时周期性评估的数据来源：`callback=sac`（vanilla）vs `callback=sac_omega`（OPE 路线）vs `callback=sac_omega_replay_buffer`（RB 路线）。`callback` 是最终判据，目录名里只写 `omega` 不足以区分 PE 和 RB。

## 实验目录命名

`checkpoints/<env>/<algo>/<设置>_seed_<n>/`，设置串按顺序编码：`{method}_dscnt_{κ}_b_{b}_n_{N}_eval_{Ne}_rb?_`，对应论文（PE-GCRL）中的消融轴：

- `seed_<n>`：同一设置的第 n 次独立重复，不是不同算法；
- `dscnt_0_9`：KDE 历史评估数据折扣因子 κ（默认 0.9）；
- `b_0`：OMEGA 平衡因子（默认 0）；
- `n_100`：每次 reset 采样候选目标数 N（默认 100）；
- `eval_96`：callback 每轮评估 episode 数（默认 96，对应 `callback.0.evaluate_nums_in_callback=6`，即 6×16 并行进程）；
- `orig_*`/`mega_*`/`omega_*`/`odiscern_*`：RIG/MEGA/OMEGA/DISCERN（O+MEGA）等基础课程方法；
- `*_rb_*`：从 replay buffer 统计（RB 路线），无后缀：从历史 evaluation 结果统计（PE 路线，即论文核心贡献 OMEGA-PE）；
- 裸 `seed_<n>`（无设置前缀）= vanilla SAC，无课程。

比较实验时先固定场景 + 算法，再比不同课程方法，最后汇总 5 个 seed。

## 约定

- 代码注释、文档、print 用中文，标识符用英文。
- 新增 evaluator 参考 `evaluator_gmm.py`（最近的提交加了 weighted-resampled GMM）；新增 shell 参考对应场景 `shells/` 下的已有格式，逐 seed 列出完整命令。
- `.gitignore` 已忽略 `checkpoints/`、`outputs/`、`logs/`、`plots/`、`nohup.out`、`paper/`、`初读理解.pdf` —— 这些是本地实验产物，不要提交。
