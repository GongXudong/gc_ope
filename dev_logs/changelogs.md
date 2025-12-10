# GC-OPE

TODO: sampling data compatible with scope-rl.
TODO: training and compare estimating $p_{ag}$ by historical evaluation and replay buffer.

## 2025-12-10

* Add training scripts of MEGA/OMEGA for FlyCraft.
* Add scripts of estimating $p_{ag}$ with data sampled from replay buffer, and add corresponding training scripts for MEGA/OMEGA.

## 2025-12-07

* Add test for KL(u|p) for all environments.
* Debug desired_goal_utils.generate_all_possible_dgs() for maze environments, and re-evaluate sac, her, ppo on pointmaze.

## 2025-12-02

* Add desired_goal_utils for all envs.

## 2025-11-27

* Add evaluation scripts on Push(sparse), Slide(sparse).

## 2025-11-26

* Add training scripts on Push(sparse), Slide(sparse).

## 2025-11-25

* Add evaluation scripts on AntMaze.

## 2025-11-21

* Add $D(p_{ag}, \tilde{p}_{ag})$ evaluation scripts.

## 2025-11-18

* Add training scripts on AntMaze.

## 2025-11-17

* Add evaluation scripts on PointMaze.

## 2025-11-14

* Add training scripts on PointMaze.

## 2025-11-11

* Add evaluation scripts on Reach.

## 2025-11-08

* Add training scripts on Reach.

## 2025-11-07

* Add KDE evaluator (class: EvaluationResultContainer, WeightedEvaluationResultContainer, KDEEvaluator).

## 2025-11-06

* Add OPE_roadmaps.

## Before 2025-11-05

* Add training scripts (PPO, SAC, HER) on FlyCraft.
* Add evaluation scirpts (PPO, SAC, HER) on FlyCraft.
* Set configuration patterns.
