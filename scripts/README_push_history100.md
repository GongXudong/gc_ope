# Push/SAC 四方法历史抽样实验

运行入口（在项目目录执行）：

```bash
conda run --no-capture-output -n gc_ope bash scripts/run_push_history100.sh
```

顺序固定为 NN → FM → NF → GMM。每个方法同时运行 seed 1～5，每个 seed 内
4 个 checkpoint worker；当前方法所有 seed 退出且结果检查通过后，再启动下一方法。
最多 20 个计算 worker，CPU-only，各进程 BLAS 单线程。Ctrl-C 会通知各 seed runner 清理 worker。

## 数据与参数

- 每个 checkpoint 从**全部** fixed-eval 记录中**有放回抽 100 条**，含成功/失败；不是抽100条成功记录。
- 使用 `checkpoint <= 当前 checkpoint` 的数据，包含当前 checkpoint；这是分布估计的同时间评估，不是严格历史预测。
- 抽样随机流由抽样种子0、任务、训练seed、来源checkpoint决定。每份来源文件在各方法、各后续checkpoint中取相同行；与并行顺序无关。
- NN使用这些记录的成功/失败标签；FM/NF/GMM使用其中成功目标。
- 时间权重为 `0.9 ** ((当前 checkpoint - 来源 checkpoint) / 10000)`，当前checkpoint权重为1。
- MC采样数10000，重复5次；参考分布仍使用当前checkpoint的完整成功记录。
- FM：500次更新、宽度32、每次抽样2000、学习率0.001、RK4步数32。
- NN：100轮、宽度16、10%内部验证及early stopping；NF：100轮、宽度32；GMM：5成分、加权重采样1000。
- 0个成功样本正常跳过；FM/NF少于2条成功记录、NN分层验证缺乏样本时注明跳过，不伪造结果。

## 输出、复现与续跑

默认目录：`logs/replacement_experiment/push_history100_inclusive_mc10000/`。

- `method_per_seed/`：20个结果CSV及各自参数配置JSON。
- `commands/`：展开后的20个单独seed脚本，可定位执行命令。
- `runs/<时间戳>/`：总日志、每方法每seed日志、退出码、覆盖率、完成记录、代码与输入文件哈希。
- `latest_run.txt`：最近一次运行日志目录。

所有seed命令带 `--skip-done`。`ok`和有明确原因的`skipped:*`不重复执行；错误记录可重跑。
输出参数配置不一致时拒绝续跑，必须使用新的输出目录。重复启动同一总输出目录由文件锁阻止。
总脚本同时检查进程退出码、检查点覆盖、状态和有效数值；发现错误会停止后续方法。

只看调度命令，不执行计算或写输出：

```bash
conda run -n gc_ope bash scripts/run_push_history100.sh --dry-run
```

小规模真实验证（仍然覆盖四方法×五seed，只缩小待评估checkpoint集合）：

```bash
conda run --no-capture-output -n gc_ope bash scripts/run_push_history100.sh \
  --checkpoints 10000 20000 500000 \
  --out-dir logs/replacement_experiment/push_history100_smoke_20260922
```

## KDE仅复用

KDE读取来源为 `plots/p_ag_dist_between_truth_and_estimated_in_training/my_push/sac/eval_data/`
下 `my_push_sac_seed_{1..5}_kde_0_9_eval_res_in_training.csv` 的 `$D_{KL}$ [his]` 列。
调度器不执行KDE任务，不改写旧文件；manifest记录五个旧文件的路径及哈希。

已对齐的是抽样数量、包含当前checkpoint的时间范围和每次MC样本数。
旧抽样未固定种子，无法重建旧KDE完全相同的100条记录。旧KL使用各模型各自标准化空间、
低密度截除、oracle KDE带宽1.0及一次MC；新runner保留raw-space KL、带宽0.2及5次MC。
因此旧KDE仍应明确标为legacy对照，不声称严格同口径。

## 本轮验证

历史抽样和调度边界单文件测试7项通过；四个evaluator单文件测试分别通过21/20/5/3项。
真实smoke覆盖4方法×5seed×3checkpoint，共60行；包含因抽样成功数不足而明确跳过的记录。
完整2000个评估任务已于2026-09-22执行完毕，耗时600.65秒；smoke结果未混入正式输出目录。
正式结果：NN/FM/NF各498条ok、2条正常跳过；GMM为499条ok、1条正常跳过。
合计1993条ok、7条正常跳过、0条error；跳过均来自seed2在10000/20000步的抽样成功数不足。
20个seed进程退出码均为0，四方法串行边界和每份CSV的完整100检查点覆盖均通过核验。
旧KDE五个文件的哈希保持不变。核验记录保存在正式输出的
`runs/20260922-113415-290253/verified_result.json`，监测由单个等待子代理完成。
