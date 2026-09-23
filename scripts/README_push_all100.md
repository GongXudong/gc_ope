# Push/SAC 离线估计器全量实验

在重构 worktree 运行，读取原目录的 checkpoint，不改写旧结果。
此入口拟合各 checkpoint 的能力分布，并计算两模型 KL。

## 启动

```bash
bash /home/tacmon/workspace/ex_SSD/gc_ope_refactor/scripts/run_push_all100.sh
```

脚本自动使用 `conda run --no-capture-output -n gc_ope`，从任何工作目录启动均可。
需要提前检查输入和命令时：

```bash
bash /home/tacmon/workspace/ex_SSD/gc_ope_refactor/scripts/run_push_all100.sh --dry-run
```

`--dry-run` 不启动拟合、不创建输出目录。

## 固定的正式配置

- 方法依次为 **NN → FM → NF → GMM**，方法之间串行。
- 每个方法同时启动 **seed 1～5**，每个 seed 自己的进程池为 **4 workers**。
- 最多 **20 个计算 worker**；CPU-only，各 worker 的 BLAS 单线程。
- 每个 seed 计算 100 个 checkpoint：10000、20000、……、1000000。
- 共 4 × 5 × 100 = **2000 条记录**；样本不足会明确跳过，仍保留对应记录。
- 每份 fixed CSV 从全部记录有放回抽 100 条，包含当前 checkpoint；四方法共享抽样。
- 参考模型和历史估计模型使用同一种估计器；KL 为参考全量分布到历史估计分布。
- 每次 MC 10000 点，重复 5 次；FM 三个成员各 500 次更新、隐藏层 [32,32,32]。
- GMM 为直接加权 EM（reg_covar=0.05），NF 为 [16,16] 的正则化版本，NN 为概率 log-loss 早停。
- 正式方法名仅为 `kde/gmm/nn/nf/fm`，旧方法后缀不再支持。图中的 KDE 复用已保存旧投稿结果。

所有拟合参数显式保存在 `configs/evaluate/push_same_family_all100.json`，
启动时复制到输出目录的 `experiment.json`，续跑会核对参数、输入与代码。

## 日志与结果

默认输出根目录：

```text
gc_ope_refactor/logs/push_final_five_5x4/
├── master.log                       # 总进度；每 30 秒更新各 seed 完成条数
├── config.json                      # 调度参数、输入哈希及入口代码哈希
├── experiment.json                  # 本次拟合与 MC 的完整显式参数
├── audit.json                       # 总状态与各 seed 覆盖率检查
├── method_per_seed/                 # 完成检查后的 20 份汇总 CSV
└── runs/<method>/seed_<n>/
    ├── console.log                  # 子入口完整标准输出与报错
    ├── run.log                      # checkpoint 完成进度
    ├── config.json                  # 单 seed 输入、算法代码与环境版本
    ├── audit.json                   # 单 seed 覆盖率与有限值检查
    ├── <method>_push_seed<n>.csv     # 实时、原子写入的结果
    └── jobs/                        # 每 checkpoint 日志及 MC/拟合诊断
```

原始数据默认来自 `/home/tacmon/workspace/ex_SSD/gc_ope/checkpoints`。
迁移机器时可显式传入 `--checkpoint-root /路径/checkpoints --output /新结果目录`。

## 续跑与停止

**中断后重复相同启动命令即可续跑**：已经正常完成或明确样本不足的 checkpoint
不会重算，错误或缺失条目会补算。相同输出目录不允许重复启动。
参数或代码变化时会拒绝混用，需要新输出目录。

Ctrl-C 或 SIGTERM 会停止五个 seed 及其 worker，保留已落盘的结果。
任一 seed 失败会停止同阶段其余 seed，并且不进入下一个方法。
某个方法的五个 seed 都通过退出码、100 条覆盖率、重复行及有限值检查后，
才会开始下一个方法；不能仅根据进程退出码判断实验完整。

单个 seed 的跳过行不等于有效 KL。完成后应同时查看 `audit.json` 中的正常和
跳过计数，而不只是看到 2000 条就认定 2000 次拟合都成功。

## 开发验证

`tests/evaluate/test_all100_launcher.py` 在临时合成数据上验证五 seed 各四 worker、
四方法顺序、续跑、配置隔离、缺失输入、重复启动、Ctrl-C 和 seed 异常退出。
测试只有一个 checkpoint，模型仅训练两次、MC 只采 16 点，不是正式结果。
正式实验由用户手动启动。

## 画正式五方法图

```bash
conda run -n gc_ope python scripts/plot_fig6.py --output plots/fig6_final_methods_new
```

默认来源是 `configs/evaluate/fig6_sources.json`，指向已完成、已确认的四方法优化结果和旧 KDE。
它仅映射展示名称，不改写旧 CSV。迁移机器须准备这些 CSV，或修改来源路径并核对摘要。
若今后用正式方法名新跑了结果，可增加 `--result-root logs/push_final_five_5x4`。
图不平滑，五条实线，按 KDE/GMM/NN/FM/NF 顺序使用 Seaborn 默认 deep 配色。
全工作记录、旧实验版本及当前使用方法见 `docs/重构全过程与最终五方法报告.md`。
