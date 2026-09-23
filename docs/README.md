# 重构与正式五方法

迁移到师兄机器运行，先看 [GMM 远程运行说明](GMM远程运行说明.md)。

当前入口文档是 [重构全过程与最终五方法报告](重构全过程与最终五方法报告.md)。

在线课程学习接入见 [GMM 课程训练接入与审阅说明](GMM课程训练接入与审阅说明.md)，包含旧 KDE 角色、最小替换边界、五组训练命令及验证结果。

动手前先查 [旧训练与数据制作流程核查](旧训练与数据制作流程核查.md)：训练、课程过程日志、random/fixed 后评估及 notebook 的完整文件流。

- 正式方法：KDE、GMM、NN、FM、NF。
- 实验参数：`configs/evaluate/push_same_family_all100.json`。
- 已确认结果的绘图来源：`configs/evaluate/fig6_sources.json`。
- 全量启动和日志：`scripts/README_push_all100.md`。
- 其他文档是阶段记录，保留当时的事实；其中旧命令需要对应的历史提交。

实验 CSV、checkpoint 和 PNG/PDF 不包含在 Git 分支中；本机最终图位于
`plots/fig6_final_methods/`，全过程验证日志位于 `logs/finalization_validation/`。
