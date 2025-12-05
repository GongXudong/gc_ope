# 待开发功能模块

## 1.编写训练脚本

- [x] 编写训练脚本，PPO，SAC
- [x] 编写配置文件，训练（FlyCraft，Reach）/评估
- [x] 编写评估脚本（FlyCraft），并行评估训练中记录的所有ckpt
- [ ] 分析评估数据
  - [x] 取训练中的某个checkpoint，根据在该checkpoint上的评估数据计算$p_{ag}$，使用不同方法计算$\tilde{p}_{ag}$，画图展示$p_{ag}$和$\tilde{p}_{ag}$的差异
    - [x] 根据replay buffer估计$\tilde{p}_{ag}$
    - [x] 根据历史评估数据估计$\tilde{p}_{ag}$
  - [x] 对一次训练，对每个checkpoint计算$D(p_{ag}, \tilde{p}_{ag})$，画图展示$D(p_{ag}, \tilde{p}_{ag})$随训练进行的变化趋势
    - [x] 计算两个KDE分布之间的距离
    - [x] 处理一次训练中的所有checkpoints，对比不同估计方法，画折线图，横轴是训练进度，纵轴是$D(p_{ag}, \tilde{p}_{ag})$
- [ ] 课程学习
  - [x] 编写训练脚本
  - [ ] 对比rollout/success_rate和rollout/ep_rew_mean，说明训练中采样到的目标的成功率更高了
  - [ ] 记录训练过程中采样到的目标，计算这些目标在$p_{ag}$上的概率，并画出概率图。说明采样到的目标在能力边界

## 2.训练与评估

### 2.1 训练

||FlyCraft (easy)|Reach|Push|Slide|PointMaze|AntMaze|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|PPO|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox">|
|SAC|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox">|
|HER|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox">|
|GCBC|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|

### 2.2 评估

||FlyCraft (easy)|Reach|Push|Slide|PointMaze|AntMaze|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|PPO|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox">|
|SAC|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox">|
|HER|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox">|
|GCBC|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|
