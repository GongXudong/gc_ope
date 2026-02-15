# 存档checkpoint
## 2026/1/11 23:15
### 今天刚完成的事
0. 和cc对话，学习了scope-rl里的OPE算法（应该早这么干！），整理在了`scope-rl/scope_rl/ope/BASIC_OPE_SUMMARY.md`。
1. 学习了pscore获取原理，离散和连续动作的区别，见[豆包chat](https://www.doubao.com/chat/35590048527003138)。
2. 通过看源码，跑通了PPO和SAC算法的pscore获取，见`tests/algorithm/ope/test_pscore.py`。
### 明天接着做的事
- [x] 问下豆包：`如果在获取连续动作-评价策略动作分布时，按照离散的来处理的，会不会出现重要性溢出的情况？`，初步check问题。
- [x] 手动在`CLAUDE.md`里写上发现的问题。
- [x] cc对话：`总结下当前OPE算法实现(/home/maxine/ai4robot/gc_ope/src/gc_ope/algorithm/ope)和存在的潜在问题，写在algorithm/ope下的一个markdown文件里。 向我确认要总结的内容和偏好。 我是希望先详细了解现在代码的实现，洞察算法性能差和运行中存在的问题，参考scoperl里OPE算法的实现(/home/maxine/ai4robot/ope-repos/scope-rl/scope_rl/ope/BASIC_OPE_SUMMARY.md)，改进我的算法，实现我当前的开发目标。`

## 2026/1/12 15:17
### 刚完成的事
1. 以上两个todo
2. 第三个todo，实现短期、中期开发路线 (见 `ope-improve` session)
3. 测了下当前实现在三个环境+两个算法上的表现，TIS、PDIS全0.0（极小值）下溢（这是高斯核在长轨迹上的乘积效应导致的，属于预期行为），需要解决
   1. 增大bandwidth，选择合适的 bandwidth 参数。
   2. 使用 Epanechnikov 核
### 等会儿接着做的事情
- [x] 实现长期开发路线，完善高斯核、bandwidth调整，改进重要性采样的计算

## 2026/1/16 20:42
### 刚完成的事
1. 以上一个todo，实现长期开发路线，解决了重要性采样方法极小的问题（见[CLAUDE.md](../../../../CLAUDE.md)### 已知潜在问题 第2点）。
### 等会儿接着做的事情
- [x] 对评估策略用来自行为策略数据的 goal 做评估，代码可参考[eval](../../../../scripts/evaluate_ckpt/evaluate_flycraft.py)。
  - 实现一半，[reach](evaluation/evaluate_my_reach.py)上待解决TODO即完成 2026/1/16 22:31
- [x] 补充运行flycraft环境、课程学习 omega 算法。
- [ ] 整理 OPE 算法核心公式，①包含进GCRL的goal，②结合代码库里的实现，交给师兄review。