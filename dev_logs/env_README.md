# 环境说明

## FlyCraft

## Reach

## Push

根据物体当前位置与目标位置计算奖励，在爪子碰到物体前，奖励不变，因此这一段无法衡量动作的有效性，所以使用稀疏奖励。

## Slide

根据物体当前位置与目标位置计算奖励，在爪子碰到物体前，奖励不变，因此这一段无法衡量动作的有效性，所以使用稀疏奖励。

注意：不考虑速度，滑块以高速穿过目标点，仍然算抵达目标，episode终止。

## PointMaze/AntMaze

* 格子行列序号$(i, j)$的坐标原点是地图的最左上角，$i$向下为正，$j$向右为正
* $(x,y)$的坐标原点是地图的中间点(x_map_center，y_map_center)，$x$向右为正，$y$向上为正
* Maze类中的cell_rowcol_to_xy()和cell_xy_to_rowcol()提供了他们之间的对应关系
* 采样desired_goal的逻辑：Maze类中，每个方格的边长是maze.maze_size_scaling，生成desired_goal时，是在以格子中心为中心，边长为2 * position_noise_range * maze.maze_size_scaling为边长的“小格子”内采样（注意，不是在Maze类型定义的格子中采样！！！）
