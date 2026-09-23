# 师兄机器上的 GMM 课程训练

分支：`codex/refactor-offline-evaluation`。本实验沿用旧 Push/SAC/OMEGA-PE 的训练、评估、存储和课程日志流程，用定版直接加权 GMM 替换 KDE。无需本机的离线 CSV、旧 checkpoint 或其他 worktree。

## 1. 下载代码

建议使用独立目录，避免混入师兄已有实验的日志和 checkpoint：

```bash
git clone --branch codex/refactor-offline-evaluation --single-branch \
  https://github.com/GongXudong/gc_ope.git gc_ope_gmm
cd gc_ope_gmm
```

仓库目录名字和所在位置可以自定，启动脚本会定位自己的仓库根目录，不依赖 `/home/tacmon/...`。

## 2. Python 环境

统一使用 Conda `gc_ope`。如果已有该环境，跳过创建；否则创建 Python 3.12 环境：

```bash
conda create -n gc_ope python=3.12 pip -y
```

在新仓库根目录安装本分支，使用本次已验证的关键依赖版本：

```bash
conda run --no-capture-output -n gc_ope python -m pip install -e . \
  -c requirements/gmm_training_constraints.txt
conda run -n gc_ope python -m pip check
conda run -n gc_ope env PYTHONPATH=src python -c \
  'import torch; import gc_ope.evaluate.evaluator_gmm as m; print(m.__file__); print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available())'
```

打印的源码路径应位于新仓库中。正式 SAC 配置沿用旧 `device: cuda`，需可用的 CUDA PyTorch 和匹配的 NVIDIA 驱动；关键版本约束不代替驱动安装，也不保证所有机器的 CUDA 配置一致。已有环境会按上述约束调整依赖，原 Python 环境需自行保留时应先另行备份。

`pyproject.toml` 已声明 GMM/训练入口使用的依赖。平台需要编译 pybullet 时，沿用根 README 中的编译工具安装说明。无需 uv、复制 `.venv` 或移动本机 Conda 环境。

## 3. 先查看命令

```bash
conda run -n gc_ope bash scripts/train_policy/shells/my_push/curriculum/sac_omega_gmm.sh --dry-run
```

默认是 5 seed 顺序运行，每个 seed 从头训练 100 万步。参数、四类随机种子、课程逻辑保持原先核对后的设置。

## 4. 只运行剩余 seed 2～5

seed 1 已在原机器完成；其数据保留在原机器，不随 Git 上传。师兄只需在四个终端各进入本仓库根目录，分别执行：

```bash
# 终端 1
conda run --no-capture-output -n gc_ope bash scripts/train_policy/shells/my_push/curriculum/sac_omega_gmm.sh --seed 2
```

```bash
# 终端 2
conda run --no-capture-output -n gc_ope bash scripts/train_policy/shells/my_push/curriculum/sac_omega_gmm.sh --seed 3
```

```bash
# 终端 3
conda run --no-capture-output -n gc_ope bash scripts/train_policy/shells/my_push/curriculum/sac_omega_gmm.sh --seed 4
```

```bash
# 终端 4
conda run --no-capture-output -n gc_ope bash scripts/train_policy/shells/my_push/curriculum/sac_omega_gmm.sh --seed 5
```

每个 seed 内有 1 个训练环境、16 个最终评估环境、16 个周期评估环境，即 33 个环境子进程；四路同时运行共 132 个环境子进程，另有四个训练主进程等。训练模型默认也会共享当前可见 GPU。这些设置沿用旧实验，不能把“4 个 seed”理解成总共只有 4 个进程。原 32 GB 机器已发生并行内存不足；192 GB 机器的实际峰值和 GPU 余量本次没有实测。

如果希望顺序完成这四组，可在一个终端逐条执行同样的命令。去掉 `--seed` 会从 seed 1 开始顺序运行全部五组。

## 5. 文件保存和后续使用

第 N 个 seed 的三个输出位置均相对于新仓库：

```text
checkpoints/my_push/sac/omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_N/
logs/my_push/sac/omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_N/
logs_in_process/my_push/sac/my_push_sac_omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_N.txt
```

- checkpoint 每 10000 步保存一次；正常跑到 100 万步有 100 个周期模型，以及对应 replay buffer 和最佳模型。
- 普通日志保存 `progress.csv`、`evaluations.npz`、TensorBoard、控制台输出与 Hydra 配置。
- 过程文本日志保留实际课程选点、随机探索、KL/alpha 等旧 wrapper 输出，供原课程图解析器使用。
- random/fixed 评估在训练后另用旧 `scripts/evaluate_ckpt/` 执行，详见 [旧流程核查](旧训练与数据制作流程核查.md)。

已有同名结果时启动器拒绝覆盖。原机器中断的 seed 2～5 不要拷入师兄新目录再启动；本入口是从头训练，不恢复中断时的课程容器。将来汇总五个 seed 时，再把两台机器的相应 checkpoint、普通日志、过程日志合并到同一分析根目录，保持原目录结构。

## 6. 本次交付验证

此前本分支已通过真实 Push/SAC 短训练、GMM 回调拟合、旧课程文本解析和启动日志检查。此次只检查干净 Git 导出目录中的文件完整性、五 seed dry-run 和 Hydra 配置解析，不在内存不足的本机继续训练，也不声称已经在师兄机器上验证峰值内存。

进一步审阅见 [GMM 课程训练接入与审阅说明](GMM课程训练接入与审阅说明.md)。实验产物不在 Git 中；拉取分支获得的是代码、配置、测试和文档。
