# AGENTS.md

## 先核对师兄旧流程，再动手

- 本仓库后续任何工作，动手前先读师兄对应的入口、配置、数据生产和后续消费代码，并用现存文件核对；能保持一致就保持一致，不能先按自己的习惯设计新流程。
- 旧代码边界：`main` 和 `dev` 中除 `初读理解.md` 外的内容。实验分支以后的新增实现不是旧惯例的证据。
- 先说明旧流程如何运行、文件在哪里、之后怎样用于分析；只做满足当前需求的必要兼容，不擅自替换文件格式、目录、指标或随机数规则。发现旧实现限制时明确说明，不顺手改协议。
- Push 后评估的 random/fixed 入口是 `scripts/evaluate_ckpt/`；`scripts/evaluate_new/evaluate_my_push_slide.py` 是 replay-buffer 目标评估，不能混用。
- 课程目标沿用 wrapper 的文本输出，完整保存到 `logs_in_process/<env>/<algo>/<env>_<algo>_<设置>_seed_<n>.txt`。旧 `train_log_process.py` 与 behavioral-goals notebook 读取该文件；普通 `logs/.../log.txt` 不包含子进程选点输出。完整说明见 `docs/旧训练与数据制作流程核查.md`。

## 环境

- 本项目用 conda 环境 `gc_ope` 运行，不用 uv/.venv（`.venv` 已弃用，可删除）。
- 运行任何命令前缀：`conda run -n gc_ope <cmd>`，或直接 `conda activate gc_ope`。
- 测试命令：`conda run -n gc_ope env PYTHONPATH=src pytest tests/evaluate/test_weighted_gmm.py -q`。跑单个测试文件，不要跑整个 `tests/`（部分测试会挂很久/超时）。

## 命令模式

- 训练/评估脚本都是 hydra 配置驱动：`conda run -n gc_ope scripts/train_policy/train.py <key=value...>`，参考 `scripts/*/shells/**/*.sh` 里的实际命令。
- 训练实验名（`experiment_name`）决定 checkpoint 落盘路径：`checkpoints/<env>/<experiment_name>/`。
- random/fixed 评估脚本（`scripts/evaluate_ckpt/`）读取一个 checkpoint 目录，对该目录下所有 `*.zip` 生成对应 CSV。

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
- 正式方法仅为 KDE/GMM/NN/NF/FM；`evaluator_gmm.py` 为直接加权 EM，NF 为验证正则化版本，FM 为三成员密度集成。历史变体从 Git 历史查阅，勿恢复为正式入口。新增 shell 参考对应场景 `shells/` 下的已有格式。
- `.gitignore` 已忽略 `checkpoints/`、`outputs/`、`logs/`、`plots/`、`nohup.out`、`paper/`、`初读理解.pdf` —— 这些是本地实验产物，不要提交。

## 环境线程约束

- 32 核 i9-14900K / 32 GB 内存。numpy/sklearn 用 OpenBLAS，默认 DYNAMIC_ARCH MAX_THREADS=64。
- `replacement_run_all100.py` 是 ThreadPoolExecutor（GIL 只限制 Python 层；numpy BLAS 内核在 C 层多线程，4 个 worker 可各用 ~8 线程，总占用 32 线程匹配 32 核）。
- 不建议用 ProcessPool 做大规模并行（每个进程独立加载 numpy/sklearn + BLAS，内存会 ×worker 数，且 BLAS 线程竞争会导致 128 线程争 32 核）。
- 运行大规模 sweep 前确认机器无其他 CPU 密集任务。
