#!/bin/bash

## generate desired goals randomly
uv run scripts/evaluate_my_point_maze.py algo_type=sac ckpt_dir=checkpoints/my_pointmaze/sac/seed_1 env=my_point_maze_for_sac process_num=64 eval_cfg=my_point_maze_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 eval_res_csv_file_suffix=eval_res_on_random seed=12
uv run scripts/evaluate_my_point_maze.py algo_type=sac ckpt_dir=checkpoints/my_pointmaze/sac/seed_2 env=my_point_maze_for_sac process_num=64 eval_cfg=my_point_maze_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 eval_res_csv_file_suffix=eval_res_on_random seed=25
uv run scripts/evaluate_my_point_maze.py algo_type=sac ckpt_dir=checkpoints/my_pointmaze/sac/seed_3 env=my_point_maze_for_sac process_num=64 eval_cfg=my_point_maze_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 eval_res_csv_file_suffix=eval_res_on_random seed=39
uv run scripts/evaluate_my_point_maze.py algo_type=sac ckpt_dir=checkpoints/my_pointmaze/sac/seed_4 env=my_point_maze_for_sac process_num=64 eval_cfg=my_point_maze_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 eval_res_csv_file_suffix=eval_res_on_random seed=47
uv run scripts/evaluate_my_point_maze.py algo_type=sac ckpt_dir=checkpoints/my_pointmaze/sac/seed_5 env=my_point_maze_for_sac process_num=64 eval_cfg=my_point_maze_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 eval_res_csv_file_suffix=eval_res_on_random seed=51

## generate desired goals by enumerating over fixed dg intervals
uv run scripts/evaluate_my_point_maze.py algo_type=sac ckpt_dir=checkpoints/my_pointmaze/sac/seed_1 env=my_point_maze_for_sac process_num=64 eval_cfg=my_point_maze_random_dg_set eval_cfg.dg_gen_method=fixed eval_cfg.n=5 eval_res_csv_file_suffix=eval_res_on_fixed seed=15
uv run scripts/evaluate_my_point_maze.py algo_type=sac ckpt_dir=checkpoints/my_pointmaze/sac/seed_2 env=my_point_maze_for_sac process_num=64 eval_cfg=my_point_maze_random_dg_set eval_cfg.dg_gen_method=fixed eval_cfg.n=5 eval_res_csv_file_suffix=eval_res_on_fixed seed=28
uv run scripts/evaluate_my_point_maze.py algo_type=sac ckpt_dir=checkpoints/my_pointmaze/sac/seed_3 env=my_point_maze_for_sac process_num=64 eval_cfg=my_point_maze_random_dg_set eval_cfg.dg_gen_method=fixed eval_cfg.n=5 eval_res_csv_file_suffix=eval_res_on_fixed seed=33
uv run scripts/evaluate_my_point_maze.py algo_type=sac ckpt_dir=checkpoints/my_pointmaze/sac/seed_4 env=my_point_maze_for_sac process_num=64 eval_cfg=my_point_maze_random_dg_set eval_cfg.dg_gen_method=fixed eval_cfg.n=5 eval_res_csv_file_suffix=eval_res_on_fixed seed=41
uv run scripts/evaluate_my_point_maze.py algo_type=sac ckpt_dir=checkpoints/my_pointmaze/sac/seed_5 env=my_point_maze_for_sac process_num=64 eval_cfg=my_point_maze_random_dg_set eval_cfg.dg_gen_method=fixed eval_cfg.n=5 eval_res_csv_file_suffix=eval_res_on_fixed seed=56
