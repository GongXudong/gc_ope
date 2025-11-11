# 待开发功能模块

## 1.编写训练脚本

- [x] 编写训练脚本，PPO，SAC
- [x] 编写配置文件，训练（FlyCraft，Reach）/评估
- [x] 编写评估脚本（FlyCraft），并行评估训练中记录的所有ckpt
- [ ] 分析评估数据
- [ ] 根据replay buffer计算$p_{ag}$

## 2.训练与评估

### 2.1 训练

||FlyCraft (easy)|Reach|PointMaze|AntMaze|
|:-:|:-:|:-:|:-:|:-:|
|PPO|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox">|<input type="checkbox">|
|SAC|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox">|<input type="checkbox">|
|HER|<input type="checkbox" checked>|<input type="checkbox" checked>|<input type="checkbox">|<input type="checkbox">|
|GCBC|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|

### 2.2 评估

||FlyCraft (easy)|Reach|PointMaze|AntMaze|
|:-:|:-:|:-:|:-:|:-:|
|PPO|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|
|SAC|<input type="checkbox" checked>|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|
|HER|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|
|GCBC|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|<input type="checkbox">|
