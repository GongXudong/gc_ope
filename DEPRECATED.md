# 本分支已废弃（DEPRECATED）

## 状态

分支 `exp/gmm-prediction-my-push-slide` 自 **2026-09-26** 起废弃，不要再在此分支上继续开发、实验或评估。

- 废弃标记 tag：`deprecated/exp-gmm-prediction-my-push-slide`（指向该分支 HEAD `94008f3`）
- 后续工作位置：仓库 `/home/tacmon/workspace/ex_SSD/gc_ope_refactor`，分支 `codex/refactor-offline-evaluation`
- 离线超参数搜索（含 KDE/GMM/NN/NF/FM）已在 refactor 分支接入，并复用 refactor 的 `evaluator_factory`，不要沿用本分支上的 evaluator 修改。

## 给 Codex / 后续协作者的说明

- 若你（Codex 或其他人）在本仓库看到这个分支：`gc_ope` 目录本身不是废弃的，废弃的只是这个分支。
- 任何新任务都应切换到 refactor 工作区进行；本分支只作为历史参考保留。
- 未提交的工作区改动（KDE 搜索等）已以兼容 evaluator_factory 的形式迁移到 refactor 分支，勿在本分支上继续补。
